//! Gemini CLI artifact ingestion.
//!
//! Reads session JSON files, projects mapping, settings, and GEMINI.md from
//! `~/.gemini` and converts them into Remembrant domain structs ([`Session`],
//! [`ToolCall`], [`Memory`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{debug, warn};
use uuid::Uuid;

use super::{IngestResult, parse_iso_timestamp};
use crate::store::duckdb::{Memory, Session, ToolCall};

// ---------------------------------------------------------------------------
// Serde structs for on-disk JSON formats
// ---------------------------------------------------------------------------

/// Top-level session JSON (`~/.gemini/tmp/<hash>/chats/session-*.json`).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawSession {
    session_id: String,
    project_hash: Option<String>,
    start_time: Option<String>,
    last_updated: Option<String>,
    summary: Option<String>,
    #[serde(default)]
    messages: Vec<RawMessage>,
}

/// A single message inside a session.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMessage {
    #[allow(dead_code)]
    id: Option<String>,
    timestamp: Option<String>,
    #[serde(rename = "type")]
    msg_type: Option<String>,
    /// Content can be a string (model responses) or an array of content blocks
    /// (user messages). We use a custom deserializer to handle both.
    #[serde(default, deserialize_with = "deserialize_content")]
    content: MessageContent,
    /// Modern Gemini format: tool calls as a separate top-level array.
    #[serde(default)]
    tool_calls: Vec<RawToolCall>,
    /// Token usage breakdown (modern format).
    #[serde(default)]
    tokens: Option<RawTokens>,
}

/// Content can be either a plain string or an array of content blocks.
#[derive(Debug, Default)]
enum MessageContent {
    #[default]
    Empty,
    Text(String),
    Blocks(Vec<RawContent>),
}

impl MessageContent {
    /// Extract the first text content, regardless of variant.
    fn first_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s.as_str()),
            MessageContent::Blocks(blocks) => blocks.iter().find_map(|c| c.text.as_deref()),
            MessageContent::Empty => None,
        }
    }

    /// Count function_call blocks (old format only).
    fn function_call_count(&self) -> usize {
        match self {
            MessageContent::Blocks(blocks) => {
                blocks.iter().filter(|c| c.function_call.is_some()).count()
            }
            _ => 0,
        }
    }

    /// Iterate over function_call blocks (old format).
    fn function_calls(&self) -> Vec<&RawFunctionCall> {
        match self {
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|c| c.function_call.as_ref())
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn deserialize_content<'de, D>(deserializer: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct ContentVisitor;

    impl<'de> de::Visitor<'de> for ContentVisitor {
        type Value = MessageContent;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string, array of content blocks, or null")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<MessageContent, E> {
            Ok(MessageContent::Text(v.to_string()))
        }

        fn visit_string<E: de::Error>(self, v: String) -> Result<MessageContent, E> {
            Ok(MessageContent::Text(v))
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, seq: A) -> Result<MessageContent, A::Error> {
            let blocks: Vec<RawContent> =
                de::Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))?;
            Ok(MessageContent::Blocks(blocks))
        }

        fn visit_none<E: de::Error>(self) -> Result<MessageContent, E> {
            Ok(MessageContent::Empty)
        }

        fn visit_unit<E: de::Error>(self) -> Result<MessageContent, E> {
            Ok(MessageContent::Empty)
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}

/// Content block within a message (array format).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawContent {
    text: Option<String>,
    function_call: Option<RawFunctionCall>,
}

/// A function/tool call embedded in a model response (old format, in content array).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawFunctionCall {
    name: Option<String>,
    #[serde(default)]
    args: serde_json::Value,
}

/// A tool call in the modern format (top-level `toolCalls` array).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawToolCall {
    #[allow(dead_code)]
    id: Option<String>,
    name: Option<String>,
    #[serde(default)]
    args: serde_json::Value,
    status: Option<String>,
    timestamp: Option<String>,
}

/// Token usage breakdown.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(rename_all = "camelCase")]
struct RawTokens {
    #[serde(default)]
    input: i64,
    #[serde(default)]
    output: i64,
    #[serde(default)]
    cached: i64,
    #[serde(default)]
    thoughts: i64,
    #[serde(default)]
    tool: i64,
    #[serde(default)]
    total: i64,
}

/// `~/.gemini/projects.json`
#[derive(Debug, Deserialize)]
struct RawProjectsFile {
    /// Maps filesystem path -> project name.
    #[serde(default)]
    projects: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// GeminiIngester
// ---------------------------------------------------------------------------

/// Discovers and parses Gemini CLI artifacts from the local filesystem.
pub struct GeminiIngester {
    /// Root of the Gemini config directory (`~/.gemini`).
    base_path: PathBuf,
}

impl GeminiIngester {
    /// Create a new ingester pointing at the default `~/.gemini` directory.
    ///
    /// Returns `None` if the home directory cannot be determined or the
    /// `~/.gemini` directory does not exist.
    pub fn new() -> Option<Self> {
        let home = dirs::home_dir()?;
        let base = home.join(".gemini");
        if base.is_dir() {
            Some(Self { base_path: base })
        } else {
            warn!("~/.gemini directory not found");
            None
        }
    }

    /// Create an ingester with an explicit base path (useful for testing).
    pub fn with_base_path(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    /// Return the base path this ingester reads from.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    // -----------------------------------------------------------------------
    // Project map
    // -----------------------------------------------------------------------

    /// Read `projects.json` and return a map from **project hash** to
    /// **project name**.
    ///
    /// The on-disk format maps *filesystem path* to *name*. We store both
    /// path-keyed and hash-keyed entries so that session parsing can resolve
    /// a project name from the `projectHash` field found in session files.
    pub fn load_project_map(&self) -> Result<HashMap<String, String>> {
        let projects_path = self.base_path.join("projects.json");
        if !projects_path.is_file() {
            debug!("projects.json not found at {}", projects_path.display());
            return Ok(HashMap::new());
        }

        let data = std::fs::read_to_string(&projects_path)
            .with_context(|| format!("reading {}", projects_path.display()))?;
        let raw: RawProjectsFile = serde_json::from_str(&data).context("parsing projects.json")?;

        let mut map: HashMap<String, String> = HashMap::new();

        // Scan tmp/ directories so we know which hash dirs exist.
        // Store hash -> hash as a fallback name.
        let tmp_dir = self.base_path.join("tmp");
        if tmp_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&tmp_dir) {
                for entry in entries.flatten() {
                    let hash_name = entry.file_name().to_string_lossy().to_string();
                    map.insert(hash_name.clone(), hash_name.clone());
                }
            }
        }

        // Also store path -> name from projects.json.  During session parsing
        // we resolve via the projectHash field from the session file.
        for (path, name) in &raw.projects {
            map.insert(path.clone(), name.clone());
        }

        debug!("loaded {} project mappings", map.len());
        Ok(map)
    }

    // -----------------------------------------------------------------------
    // Session discovery
    // -----------------------------------------------------------------------

    /// Discover all session JSON file paths under `~/.gemini/tmp/*/chats/`.
    pub fn discover_sessions(&self) -> Vec<PathBuf> {
        let tmp_dir = self.base_path.join("tmp");
        let mut paths = Vec::new();

        if !tmp_dir.is_dir() {
            debug!("no tmp/ directory in {}", self.base_path.display());
            return paths;
        }

        let Ok(hash_dirs) = std::fs::read_dir(&tmp_dir) else {
            warn!("failed to read {}", tmp_dir.display());
            return paths;
        };

        for entry in hash_dirs.flatten() {
            let chats_dir = entry.path().join("chats");
            if !chats_dir.is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(&chats_dir) else {
                continue;
            };
            for file_entry in files.flatten() {
                let p = file_entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("json") {
                    paths.push(p);
                }
            }
        }

        paths.sort();
        debug!("discovered {} session files", paths.len());
        paths
    }

    // -----------------------------------------------------------------------
    // Session parsing
    // -----------------------------------------------------------------------

    /// Parse a single session JSON file into a domain [`Session`].
    pub fn parse_session(
        &self,
        json_path: &Path,
        project_map: &HashMap<String, String>,
    ) -> Result<Session> {
        let data = std::fs::read_to_string(json_path)
            .with_context(|| format!("reading session file {}", json_path.display()))?;
        let raw: RawSession = serde_json::from_str(&data).context("parsing session JSON")?;

        let started_at = raw.start_time.as_deref().and_then(parse_iso_timestamp);
        let ended_at = raw.last_updated.as_deref().and_then(parse_iso_timestamp);

        let duration_minutes = match (started_at, ended_at) {
            (Some(s), Some(e)) => {
                let dur = e.signed_duration_since(s);
                Some(dur.num_minutes() as i32)
            }
            _ => None,
        };

        let message_count = Some(raw.messages.len() as i32);

        // Count tool calls: modern `toolCalls` field + legacy `function_call` in content.
        let tool_call_count: i32 = raw
            .messages
            .iter()
            .map(|m| m.tool_calls.len() + m.content.function_call_count())
            .sum::<usize>() as i32;

        // Sum token counts from messages that have them.
        let total_tokens: Option<i32> = {
            let sum: i64 = raw
                .messages
                .iter()
                .filter_map(|m| m.tokens.as_ref())
                .map(|t| t.total)
                .sum();
            if sum > 0 { Some(sum as i32) } else { None }
        };

        // Resolve project id from the hash via the project map.
        let project_id = raw.project_hash.as_deref().and_then(|hash| {
            project_map
                .get(hash)
                .cloned()
                .or_else(|| Some(hash.to_string()))
        });

        // Build summary: prefer top-level summary, fall back to first user message.
        let summary = raw.summary.clone().or_else(|| {
            raw.messages
                .iter()
                .find(|m| m.msg_type.as_deref() == Some("user"))
                .and_then(|m| m.content.first_text())
                .map(|t| {
                    let trimmed = t.trim();
                    if trimmed.len() > 200 {
                        format!("{}...", &trimmed[..200])
                    } else {
                        trimmed.to_string()
                    }
                })
        });

        Ok(Session {
            id: raw.session_id,
            project_id,
            agent: "gemini".to_string(),
            started_at,
            ended_at,
            duration_minutes,
            message_count,
            tool_call_count: Some(tool_call_count),
            total_tokens,
            files_changed: Vec::new(),
            summary,
        })
    }

    // -----------------------------------------------------------------------
    // Tool-call extraction
    // -----------------------------------------------------------------------

    /// Extract tool calls from a session JSON file.
    pub fn parse_session_tool_calls(&self, json_path: &Path) -> Result<Vec<ToolCall>> {
        let data = std::fs::read_to_string(json_path)
            .with_context(|| format!("reading session file {}", json_path.display()))?;
        let raw: RawSession = serde_json::from_str(&data).context("parsing session JSON")?;

        let session_id = raw.session_id.clone();
        let mut calls = Vec::new();

        for msg in &raw.messages {
            let msg_timestamp = msg.timestamp.as_deref().and_then(parse_iso_timestamp);

            // Modern format: top-level toolCalls array.
            for tc in &msg.tool_calls {
                let command = if tc.args.is_null() {
                    None
                } else {
                    Some(tc.args.to_string())
                };
                let timestamp = tc
                    .timestamp
                    .as_deref()
                    .and_then(parse_iso_timestamp)
                    .or(msg_timestamp);
                let success = match tc.status.as_deref() {
                    Some("success") => Some(true),
                    Some("error") => Some(false),
                    _ => None,
                };

                calls.push(ToolCall {
                    id: Uuid::new_v4().to_string(),
                    session_id: Some(session_id.clone()),
                    tool_name: tc.name.clone(),
                    command,
                    success,
                    error_message: None,
                    duration_ms: None,
                    timestamp,
                });
            }

            // Legacy format: function_call in content blocks.
            for fc in msg.content.function_calls() {
                let command = if fc.args.is_null() {
                    None
                } else {
                    Some(fc.args.to_string())
                };

                calls.push(ToolCall {
                    id: Uuid::new_v4().to_string(),
                    session_id: Some(session_id.clone()),
                    tool_name: fc.name.clone(),
                    command,
                    success: None,
                    error_message: None,
                    duration_ms: None,
                    timestamp: msg_timestamp,
                });
            }
        }

        debug!(
            "extracted {} tool calls from {}",
            calls.len(),
            json_path.display()
        );
        Ok(calls)
    }

    // -----------------------------------------------------------------------
    // GEMINI.md as Memory
    // -----------------------------------------------------------------------

    /// Read `~/.gemini/GEMINI.md` and return it as a [`Memory`].
    pub fn parse_gemini_md(&self) -> Option<Memory> {
        let md_path = self.base_path.join("GEMINI.md");
        match std::fs::read_to_string(&md_path) {
            Ok(content) if !content.trim().is_empty() => {
                debug!("read GEMINI.md ({} bytes)", content.len());
                Some(Memory {
                    id: Uuid::new_v4().to_string(),
                    project_id: None, // global
                    content,
                    memory_type: Some("system_prompt".to_string()),
                    source_session_id: None,
                    confidence: 1.0,
                    access_count: 0,
                    created_at: None,
                    updated_at: None,
                    valid_until: None,
                })
            }
            Ok(_) => {
                debug!("GEMINI.md is empty");
                None
            }
            Err(e) => {
                debug!("GEMINI.md not found or unreadable: {e}");
                None
            }
        }
    }

    // -----------------------------------------------------------------------
    // Full ingestion
    // -----------------------------------------------------------------------

    /// Discover and parse all Gemini artifacts, returning an [`IngestResult`].
    ///
    /// The returned `IngestResult` carries summary counters. The parsed domain
    /// objects (sessions, tool calls, memories) are returned as the second
    /// element of the tuple so that callers can persist them.
    pub fn ingest_all(&self) -> (IngestResult, Vec<Session>, Vec<ToolCall>, Vec<Memory>) {
        let mut result = IngestResult::new("gemini");
        let mut sessions = Vec::new();
        let mut tool_calls = Vec::new();
        let mut memories = Vec::new();

        // Load project map (best-effort).
        let project_map = match self.load_project_map() {
            Ok(m) => m,
            Err(e) => {
                warn!("failed to load project map: {e}");
                result.errors.push(format!("project map: {e}"));
                HashMap::new()
            }
        };

        // Sessions and tool calls.
        let session_paths = self.discover_sessions();
        for path in &session_paths {
            match self.parse_session(path, &project_map) {
                Ok(session) => sessions.push(session),
                Err(e) => {
                    let msg = format!("session {}: {e}", path.display());
                    warn!("{msg}");
                    result.errors.push(msg);
                }
            }
            match self.parse_session_tool_calls(path) {
                Ok(calls) => tool_calls.extend(calls),
                Err(e) => {
                    let msg = format!("tool_calls {}: {e}", path.display());
                    warn!("{msg}");
                    result.errors.push(msg);
                }
            }
        }

        // GEMINI.md
        if let Some(mem) = self.parse_gemini_md() {
            memories.push(mem);
        }

        result.sessions_found = sessions.len();
        result.tool_calls_found = tool_calls.len();
        result.memories_found = memories.len();

        debug!(
            "ingestion complete: {} sessions, {} tool_calls, {} memories, {} errors",
            sessions.len(),
            tool_calls.len(),
            memories.len(),
            result.errors.len(),
        );

        (result, sessions, tool_calls, memories)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: create a temporary Gemini directory tree with sample data.
    /// Includes both legacy (content array with functionCall) and modern
    /// (content string with toolCalls array) formats.
    fn setup_temp_gemini(dir: &Path) {
        let hash = "abc123hash";
        let chats_dir = dir.join("tmp").join(hash).join("chats");
        fs::create_dir_all(&chats_dir).unwrap();

        // Legacy format: content as array, functionCall in content blocks
        let legacy_json = r#"{
            "sessionId": "s-001",
            "projectHash": "abc123hash",
            "startTime": "2026-02-19T06:18:42.407Z",
            "lastUpdated": "2026-02-19T06:28:42.407Z",
            "messages": [
                {
                    "id": "m-1",
                    "timestamp": "2026-02-19T06:18:42.408Z",
                    "type": "user",
                    "content": [{"text": "Hello, refactor the auth module"}]
                },
                {
                    "id": "m-2",
                    "timestamp": "2026-02-19T06:19:00.000Z",
                    "type": "model",
                    "content": [
                        {"text": "Sure, let me start by reading the file."},
                        {"functionCall": {"name": "read_file", "args": {"path": "src/auth.rs"}}}
                    ]
                },
                {
                    "id": "m-3",
                    "timestamp": "2026-02-19T06:20:00.000Z",
                    "type": "model",
                    "content": [{"text": "Here is the refactored code."}]
                }
            ]
        }"#;

        fs::write(
            chats_dir.join("session-2026-02-19T06-18-08s001.json"),
            legacy_json,
        )
        .unwrap();

        // Modern format: content as string, toolCalls as separate array, with tokens
        let modern_json = r#"{
            "sessionId": "s-002",
            "projectHash": "abc123hash",
            "startTime": "2026-03-10T10:00:00.000Z",
            "lastUpdated": "2026-03-10T10:15:00.000Z",
            "messages": [
                {
                    "id": "m-10",
                    "timestamp": "2026-03-10T10:00:00.000Z",
                    "type": "user",
                    "content": [{"text": "Fix the login bug"}]
                },
                {
                    "id": "m-11",
                    "timestamp": "2026-03-10T10:01:00.000Z",
                    "type": "gemini",
                    "content": "Let me look at the login module.",
                    "toolCalls": [
                        {
                            "id": "tc-1",
                            "name": "read_file",
                            "args": {"path": "src/login.rs"},
                            "status": "success",
                            "timestamp": "2026-03-10T10:01:05.000Z"
                        },
                        {
                            "id": "tc-2",
                            "name": "write_file",
                            "args": {"path": "src/login.rs", "content": "fixed"},
                            "status": "success",
                            "timestamp": "2026-03-10T10:01:10.000Z"
                        }
                    ],
                    "tokens": {"input": 1000, "output": 200, "cached": 0, "thoughts": 50, "tool": 0, "total": 1250},
                    "thoughts": [{"subject": "debugging", "description": "checking login flow"}]
                },
                {
                    "id": "m-12",
                    "timestamp": "2026-03-10T10:02:00.000Z",
                    "type": "gemini",
                    "content": "I've fixed the login bug.",
                    "tokens": {"input": 500, "output": 100, "cached": 0, "thoughts": 0, "tool": 0, "total": 600}
                }
            ]
        }"#;

        fs::write(
            chats_dir.join("session-2026-03-10T10-00-08s002.json"),
            modern_json,
        )
        .unwrap();

        // projects.json
        let projects = r#"{"projects": {"/tmp/myproject": "my-project"}}"#;
        fs::write(dir.join("projects.json"), projects).unwrap();

        // GEMINI.md
        fs::write(
            dir.join("GEMINI.md"),
            "# My global instructions\nBe helpful.",
        )
        .unwrap();
    }

    #[test]
    fn test_parse_legacy_session() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_gemini(&base);

        let ingester = GeminiIngester::with_base_path(&base);
        let project_map = ingester.load_project_map().unwrap();
        let sessions = ingester.discover_sessions();

        assert_eq!(sessions.len(), 2);

        // Legacy format session (s-001)
        let legacy_path = sessions
            .iter()
            .find(|p| p.to_string_lossy().contains("s001"))
            .unwrap();
        let session = ingester.parse_session(legacy_path, &project_map).unwrap();
        assert_eq!(session.id, "s-001");
        assert_eq!(session.agent, "gemini");
        assert_eq!(session.message_count, Some(3));
        assert_eq!(session.tool_call_count, Some(1));
        assert_eq!(session.duration_minutes, Some(10));
        assert!(session.started_at.is_some());
        assert!(session.ended_at.is_some());
        assert!(session.summary.unwrap().contains("auth module"));
        assert!(session.total_tokens.is_none()); // Legacy has no tokens
    }

    #[test]
    fn test_parse_modern_session() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_gemini(&base);

        let ingester = GeminiIngester::with_base_path(&base);
        let project_map = ingester.load_project_map().unwrap();
        let sessions = ingester.discover_sessions();

        // Modern format session (s-002)
        let modern_path = sessions
            .iter()
            .find(|p| p.to_string_lossy().contains("s002"))
            .unwrap();
        let session = ingester.parse_session(modern_path, &project_map).unwrap();
        assert_eq!(session.id, "s-002");
        assert_eq!(session.agent, "gemini");
        assert_eq!(session.message_count, Some(3));
        assert_eq!(session.tool_call_count, Some(2)); // 2 toolCalls in modern format
        assert_eq!(session.duration_minutes, Some(15));
        assert_eq!(session.total_tokens, Some(1850)); // 1250 + 600
        assert!(session.summary.unwrap().contains("login bug"));
    }

    #[test]
    fn test_parse_legacy_tool_calls() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_gemini(&base);

        let ingester = GeminiIngester::with_base_path(&base);
        let sessions = ingester.discover_sessions();

        let legacy_path = sessions
            .iter()
            .find(|p| p.to_string_lossy().contains("s001"))
            .unwrap();
        let calls = ingester.parse_session_tool_calls(legacy_path).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_name.as_deref(), Some("read_file"));
        assert_eq!(calls[0].session_id.as_deref(), Some("s-001"));
        assert!(calls[0].command.as_ref().unwrap().contains("src/auth.rs"));
        assert!(calls[0].timestamp.is_some());
        assert!(calls[0].success.is_none()); // Legacy has no status
    }

    #[test]
    fn test_parse_modern_tool_calls() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_gemini(&base);

        let ingester = GeminiIngester::with_base_path(&base);
        let sessions = ingester.discover_sessions();

        let modern_path = sessions
            .iter()
            .find(|p| p.to_string_lossy().contains("s002"))
            .unwrap();
        let calls = ingester.parse_session_tool_calls(modern_path).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_name.as_deref(), Some("read_file"));
        assert_eq!(calls[1].tool_name.as_deref(), Some("write_file"));
        assert_eq!(calls[0].success, Some(true));
        assert_eq!(calls[1].success, Some(true));
        assert!(calls[0].timestamp.is_some());
    }

    #[test]
    fn test_parse_gemini_md() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_gemini(&base);

        let ingester = GeminiIngester::with_base_path(&base);
        let memory = ingester.parse_gemini_md();

        assert!(memory.is_some());
        let mem = memory.unwrap();
        assert_eq!(mem.memory_type.as_deref(), Some("system_prompt"));
        assert!(mem.content.contains("global instructions"));
        assert_eq!(mem.confidence, 1.0);
    }

    #[test]
    fn test_ingest_all() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_gemini(&base);

        let ingester = GeminiIngester::with_base_path(&base);
        let (result, sessions, tool_calls, memories) = ingester.ingest_all();

        assert_eq!(result.sessions_found, 2); // 1 legacy + 1 modern
        assert_eq!(result.tool_calls_found, 3); // 1 legacy + 2 modern
        assert_eq!(result.memories_found, 1);
        assert_eq!(sessions.len(), 2);
        assert_eq!(tool_calls.len(), 3);
        assert_eq!(memories.len(), 1);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_missing_base_path() {
        let ingester = GeminiIngester::with_base_path("/nonexistent/path/.gemini");
        let sessions = ingester.discover_sessions();
        assert!(sessions.is_empty());

        let mem = ingester.parse_gemini_md();
        assert!(mem.is_none());
    }

    #[test]
    fn test_empty_session_messages() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        let chats_dir = base.join("tmp").join("hash1").join("chats");
        fs::create_dir_all(&chats_dir).unwrap();

        let json = r#"{
            "sessionId": "empty-session",
            "startTime": "2026-03-01T10:00:00Z",
            "messages": []
        }"#;
        let session_path = chats_dir.join("session-empty.json");
        fs::write(&session_path, json).unwrap();

        let ingester = GeminiIngester::with_base_path(&base);
        let map = HashMap::new();
        let session = ingester.parse_session(&session_path, &map).unwrap();

        assert_eq!(session.id, "empty-session");
        assert_eq!(session.message_count, Some(0));
        assert_eq!(session.tool_call_count, Some(0));
        assert!(session.summary.is_none());
    }
}
