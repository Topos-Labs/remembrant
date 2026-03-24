//! MCP (Model Context Protocol) server for Claude Code and Cursor integration.
//!
//! Implements the MCP spec over stdio using JSON-RPC 2.0.
//! Tools exposed:
//! - mem_search: Search across all knowledge (progressive Layer 1)
//! - mem_recall: Get details for a specific entry (Layer 2/3)
//! - mem_add: Add a fact or note
//! - mem_context: Get project context briefing
//! - mem_xpath: Semantic XPath structured queries
//! - mem_decide: Record a decision
//! - mem_update: Update memory content/confidence
//! - mem_delete: Delete a memory or fact
//! - mem_rethink: Revise a fact with full history

use std::io::{self, BufRead, Write};

use anyhow::Result;
use remembrant_engine::hybrid_search::HybridSearch;
use remembrant_engine::progressive::ProgressiveRetriever;
use remembrant_engine::store::duckdb::{DuckStore, Fact};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

// ---------------------------------------------------------------------------
// MCP protocol types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct McpTool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

#[derive(Debug, Serialize)]
struct McpToolResult {
    content: Vec<McpContent>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
}

#[derive(Debug, Serialize)]
struct McpContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

pub struct McpServer {
    store: DuckStore,
}

impl McpServer {
    pub fn new(store: DuckStore) -> Self {
        Self { store }
    }

    /// Run the MCP server over stdio (blocking).
    pub fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let error_resp = JsonRpcResponse {
                        jsonrpc: "2.0",
                        id: Value::Null,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {e}"),
                        }),
                    };
                    writeln!(stdout, "{}", serde_json::to_string(&error_resp)?)?;
                    stdout.flush()?;
                    continue;
                }
            };

            let response = self.handle_request(&request);

            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
        }

        Ok(())
    }

    fn handle_request(&self, req: &JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone().unwrap_or(Value::Null);

        match req.method.as_str() {
            "initialize" => self.handle_initialize(id),
            "initialized" => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(Value::Null),
                error: None,
            },
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, &req.params),
            "ping" => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(Value::Object(serde_json::Map::new())),
                error: None,
            },
            _ => JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", req.method),
                }),
            },
        }
    }

    // -----------------------------------------------------------------------
    // Protocol handlers
    // -----------------------------------------------------------------------

    fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        let result = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "remembrant",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "capabilities": {
                "tools": {
                    "listChanged": false,
                },
            },
        });

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn handle_tools_list(&self, id: Value) -> JsonRpcResponse {
        let tools = vec![
            McpTool {
                name: "mem_search".into(),
                description: "Search across all coding agent memory (sessions, memories, facts). \
                    Returns concise index entries. Use mem_recall to get details for specific results.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (natural language or keywords)"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }),
            },
            McpTool {
                name: "mem_recall".into(),
                description: "Get detailed information about a specific memory entry. \
                    Use IDs from mem_search results. Supports 'summary' or 'detail' depth.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Entry ID from mem_search (e.g., 'session:abc123')"
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["summary", "detail"],
                            "description": "Level of detail (default: summary)",
                            "default": "summary"
                        }
                    },
                    "required": ["id"]
                }),
            },
            McpTool {
                name: "mem_add".into(),
                description: "Add a fact or note to the knowledge base. \
                    Facts are subject-predicate-object triples. Notes are free-text.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["fact", "note"],
                            "description": "What to add"
                        },
                        "subject": {
                            "type": "string",
                            "description": "For facts: the entity (e.g., 'auth module')"
                        },
                        "predicate": {
                            "type": "string",
                            "description": "For facts: the relationship (e.g., 'uses', 'depends_on')"
                        },
                        "object": {
                            "type": "string",
                            "description": "For facts: the value (e.g., 'JWT tokens')"
                        },
                        "text": {
                            "type": "string",
                            "description": "For notes: free-text content"
                        },
                        "project": {
                            "type": "string",
                            "description": "Optional project ID"
                        }
                    },
                    "required": ["type"]
                }),
            },
            McpTool {
                name: "mem_context".into(),
                description: "Get a project context briefing with recent sessions, \
                    active facts, and key memories. Perfect for session start.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project": {
                            "type": "string",
                            "description": "Project ID or path fragment to filter by"
                        }
                    }
                }),
            },
            McpTool {
                name: "mem_xpath".into(),
                description: "Query memory using Semantic XPath for structured tree traversal. \
                    Much more precise than text search. Query the hierarchy: \
                    Project > Session > Decision/Memory/ToolCall/CodeEntity/Fact. \
                    Examples: '//Session[node~\"auth\"]/Decision', \
                    '//Fact[subject=\"auth module\"]', \
                    '//Session[agent=\"claude\"][-1]/ToolCall'.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Semantic XPath expression (e.g., '//Session[node~\"auth\"]/Decision')"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Max results (default: 20)",
                            "default": 20
                        }
                    },
                    "required": ["query"]
                }),
            },
            McpTool {
                name: "mem_decide".into(),
                description: "Record an architectural or design decision with reasoning.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "what": {
                            "type": "string",
                            "description": "What was decided"
                        },
                        "why": {
                            "type": "string",
                            "description": "Why this decision was made"
                        },
                        "alternatives": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Alternatives considered"
                        },
                        "project": {
                            "type": "string",
                            "description": "Optional project ID"
                        }
                    },
                    "required": ["what"]
                }),
            },
            McpTool {
                name: "mem_update".into(),
                description: "Update an existing memory's content or confidence. \
                    Use to correct, refine, or strengthen/weaken a memory.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Memory ID to update"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content (replaces existing)"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "New confidence score (0.0-1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["id"]
                }),
            },
            McpTool {
                name: "mem_delete".into(),
                description: "Delete a memory or fact that is no longer relevant. \
                    For facts, prefer mem_rethink (which invalidates and supersedes) \
                    over hard deletion. Use delete for clearly wrong entries.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "ID of the memory or fact to delete"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["memory", "fact"],
                            "description": "What kind of entry to delete"
                        }
                    },
                    "required": ["id", "type"]
                }),
            },
            McpTool {
                name: "mem_rethink".into(),
                description: "Revise a fact with new information. Invalidates the old fact \
                    and creates a new one that supersedes it. Preserves full revision history. \
                    Use this instead of mem_delete for facts that changed (not wrong, just outdated).".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "old_fact_id": {
                            "type": "string",
                            "description": "ID of the fact to supersede"
                        },
                        "subject": {
                            "type": "string",
                            "description": "New subject (or same as old)"
                        },
                        "predicate": {
                            "type": "string",
                            "description": "New predicate (or same as old)"
                        },
                        "object": {
                            "type": "string",
                            "description": "New value"
                        },
                        "why": {
                            "type": "string",
                            "description": "Reason for revision"
                        }
                    },
                    "required": ["old_fact_id", "subject", "predicate", "object"]
                }),
            },
        ];

        let result = serde_json::json!({ "tools": tools });
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn handle_tools_call(&self, id: Value, params: &Value) -> JsonRpcResponse {
        let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        let result = match tool_name {
            "mem_search" => self.tool_mem_search(&arguments),
            "mem_recall" => self.tool_mem_recall(&arguments),
            "mem_add" => self.tool_mem_add(&arguments),
            "mem_context" => self.tool_mem_context(&arguments),
            "mem_xpath" => self.tool_mem_xpath(&arguments),
            "mem_decide" => self.tool_mem_decide(&arguments),
            "mem_update" => self.tool_mem_update(&arguments),
            "mem_delete" => self.tool_mem_delete(&arguments),
            "mem_rethink" => self.tool_mem_rethink(&arguments),
            _ => Err(anyhow::anyhow!("Unknown tool: {tool_name}")),
        };

        match result {
            Ok(text) => {
                let tool_result = McpToolResult {
                    content: vec![McpContent {
                        content_type: "text".into(),
                        text,
                    }],
                    is_error: None,
                };
                JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(serde_json::to_value(tool_result).unwrap()),
                    error: None,
                }
            }
            Err(e) => {
                let tool_result = McpToolResult {
                    content: vec![McpContent {
                        content_type: "text".into(),
                        text: format!("Error: {e}"),
                    }],
                    is_error: Some(true),
                };
                JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(serde_json::to_value(tool_result).unwrap()),
                    error: None,
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tool implementations
    // -----------------------------------------------------------------------

    fn tool_mem_search(&self, args: &Value) -> Result<String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'query' is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let retriever = ProgressiveRetriever::new(&self.store);
        let results = retriever.search_index(query, limit)?;

        if results.is_empty() {
            return Ok(format!("No results found for '{query}'."));
        }

        let mut output = format!("Found {} results for '{query}':\n\n", results.len());
        for entry in &results {
            output.push_str(&format!(
                "- [{}] {} ({})\n",
                entry.id, entry.title, entry.entry_type
            ));
        }
        output.push_str("\nUse mem_recall with an ID for more details.");

        Ok(output)
    }

    fn tool_mem_recall(&self, args: &Value) -> Result<String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'id' is required"))?;
        let depth = args
            .get("depth")
            .and_then(|v| v.as_str())
            .unwrap_or("summary");

        let retriever = ProgressiveRetriever::new(&self.store);

        match depth {
            "detail" => match retriever.get_detail(id)? {
                Some(detail) => Ok(detail.content),
                None => Ok(format!("Entry not found: {id}")),
            },
            _ => {
                // Summary level
                let summaries = retriever.get_summaries(&[id])?;
                match summaries.into_iter().next() {
                    Some(s) => Ok(format!(
                        "{}\n\nType: {}\nAgent: {}\nProject: {}\nTime: {}",
                        s.summary,
                        s.entry_type,
                        s.agent.as_deref().unwrap_or("unknown"),
                        s.project.as_deref().unwrap_or("unknown"),
                        s.timestamp.as_deref().unwrap_or("unknown"),
                    )),
                    None => Ok(format!("Entry not found: {id}")),
                }
            }
        }
    }

    fn tool_mem_add(&self, args: &Value) -> Result<String> {
        let entry_type = args
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'type' is required"))?;
        let project = args.get("project").and_then(|v| v.as_str());

        match entry_type {
            "fact" => {
                let subject = args
                    .get("subject")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("'subject' required for facts"))?;
                let predicate = args
                    .get("predicate")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("'predicate' required for facts"))?;
                let object = args
                    .get("object")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("'object' required for facts"))?;

                let fact_id = uuid::Uuid::new_v4().to_string();
                let fact = Fact {
                    id: fact_id.clone(),
                    project_id: project.map(|s| s.to_string()),
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    object: object.to_string(),
                    confidence: 0.95,
                    source_session_id: None,
                    source_agent: Some("mcp_client".to_string()),
                    valid_at: None,
                    invalid_at: None,
                    superseded_by: None,
                    created_at: None,
                };

                self.store.upsert_fact(&fact)?;
                Ok(format!(
                    "Fact added (id: {fact_id}): {subject} {predicate} {object}"
                ))
            }
            "note" => {
                let text = args
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("'text' required for notes"))?;

                let id = self.store.insert_note(text, project)?;
                Ok(format!(
                    "Note added (id: {id}): {}",
                    &text[..text.len().min(80)]
                ))
            }
            _ => Err(anyhow::anyhow!(
                "Unknown type: {entry_type}. Use 'fact' or 'note'."
            )),
        }
    }

    fn tool_mem_context(&self, args: &Value) -> Result<String> {
        let project = args.get("project").and_then(|v| v.as_str());
        let topic = args.get("topic").and_then(|v| v.as_str());
        let max_tokens = args
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(1000) as usize;
        let format = args
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("text");

        let assembler =
            remembrant_engine::ContextAssembler::new(&self.store).with_max_tokens(max_tokens);

        let ctx = if let Some(topic) = topic {
            assembler.topic_context(topic, project)?
        } else {
            assembler.project_context(project)?
        };

        match format {
            "json" => ctx.to_json(),
            _ => Ok(ctx.to_prompt_block()),
        }
    }

    fn tool_mem_xpath(&self, args: &Value) -> Result<String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'query' is required"))?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

        let search = HybridSearch::new(&self.store);
        let results = search.search_xpath(query, limit)?;

        if results.is_empty() {
            return Ok(format!("No results for XPath query: {query}"));
        }

        let mut output = format!("XPath `{}` — {} results:\n\n", query, results.len());
        for r in &results {
            let path = r
                .metadata
                .get("xpath_path")
                .map(|s| s.as_str())
                .unwrap_or("");
            let node_type = r
                .metadata
                .get("node_type")
                .map(|s| s.as_str())
                .unwrap_or("");
            output.push_str(&format!(
                "- [{node_type}] {} (score: {:.2})\n  path: {path}\n",
                r.content, r.score,
            ));
        }

        Ok(output)
    }

    fn tool_mem_decide(&self, args: &Value) -> Result<String> {
        let what = args
            .get("what")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'what' is required"))?;
        let why = args.get("why").and_then(|v| v.as_str());
        let alternatives: Vec<String> = args
            .get("alternatives")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let project = args.get("project").and_then(|v| v.as_str());

        let decision_id = uuid::Uuid::new_v4().to_string();
        let decision = remembrant_engine::store::duckdb::Decision {
            id: decision_id.clone(),
            session_id: None,
            project_id: project.map(|s| s.to_string()),
            decision_type: Some("manual".to_string()),
            what: what.to_string(),
            why: why.map(|s| s.to_string()),
            alternatives,
            outcome: None,
            created_at: None,
            valid_until: None,
        };

        self.store.insert_decision(&decision)?;
        Ok(format!("Decision recorded (id: {decision_id}): {what}"))
    }

    fn tool_mem_update(&self, args: &Value) -> Result<String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'id' is required"))?;
        let content = args.get("content").and_then(|v| v.as_str());
        let confidence = args
            .get("confidence")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32);

        if content.is_none() && confidence.is_none() {
            return Err(anyhow::anyhow!(
                "At least one of 'content' or 'confidence' must be provided"
            ));
        }

        let updated = self.store.update_memory(id, content, confidence)?;
        if updated {
            let mut parts = Vec::new();
            if content.is_some() {
                parts.push("content");
            }
            if confidence.is_some() {
                parts.push("confidence");
            }
            Ok(format!("Memory {id} updated: {}", parts.join(" and ")))
        } else {
            Ok(format!("Memory not found: {id}"))
        }
    }

    fn tool_mem_delete(&self, args: &Value) -> Result<String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'id' is required"))?;
        let entry_type = args
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'type' is required"))?;

        let deleted = match entry_type {
            "memory" => self.store.delete_memory(id)?,
            "fact" => self.store.delete_fact(id)?,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown type: {entry_type}. Use 'memory' or 'fact'."
                ));
            }
        };

        if deleted {
            Ok(format!("{entry_type} {id} deleted"))
        } else {
            Ok(format!("{entry_type} not found: {id}"))
        }
    }

    fn tool_mem_rethink(&self, args: &Value) -> Result<String> {
        let old_fact_id = args
            .get("old_fact_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'old_fact_id' is required"))?;
        let subject = args
            .get("subject")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'subject' is required"))?;
        let predicate = args
            .get("predicate")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'predicate' is required"))?;
        let object = args
            .get("object")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'object' is required"))?;
        let why = args.get("why").and_then(|v| v.as_str());

        // Verify old fact exists
        let old_fact = self
            .store
            .get_fact(old_fact_id)?
            .ok_or_else(|| anyhow::anyhow!("Old fact not found: {old_fact_id}"))?;

        // Create new fact
        let new_id = uuid::Uuid::new_v4().to_string();
        let new_fact = remembrant_engine::store::duckdb::Fact {
            id: new_id.clone(),
            project_id: old_fact.project_id,
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence: old_fact.confidence,
            source_session_id: old_fact.source_session_id,
            source_agent: Some("mcp_rethink".to_string()),
            valid_at: None,
            invalid_at: None,
            superseded_by: None,
            created_at: None,
        };

        // Invalidate old, insert new
        self.store.invalidate_fact(old_fact_id, Some(&new_id))?;
        self.store.insert_fact(&new_fact)?;

        // Optionally record the revision as a decision
        if let Some(reason) = why {
            let decision = remembrant_engine::store::duckdb::Decision {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: None,
                project_id: new_fact.project_id.clone(),
                decision_type: Some("rethink".to_string()),
                what: format!(
                    "Revised: '{} {} {}' → '{subject} {predicate} {object}'",
                    old_fact.subject, old_fact.predicate, old_fact.object
                ),
                why: Some(reason.to_string()),
                alternatives: vec![],
                outcome: None,
                created_at: None,
                valid_until: None,
            };
            self.store.insert_decision(&decision)?;
        }

        Ok(format!(
            "Fact revised: '{} {} {}' → '{subject} {predicate} {object}' (old: {old_fact_id}, new: {new_id})",
            old_fact.subject, old_fact.predicate, old_fact.object
        ))
    }
}
