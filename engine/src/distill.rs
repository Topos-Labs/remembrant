use std::collections::HashSet;

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::{DistillationConfig, DistillationLevel};
use crate::store::duckdb::{Decision, Memory, Session};

// ---------------------------------------------------------------------------
// LLM Client (OpenAI-compatible, works with LM Studio)
// ---------------------------------------------------------------------------

/// A lightweight client for OpenAI-compatible chat completion APIs.
///
/// Works out-of-the-box with LM Studio (localhost:1234) and can be pointed
/// at any compatible endpoint (OpenAI, Ollama, vLLM, etc.).
pub struct LlmClient {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: String,
}

impl LlmClient {
    /// Create a client pointing at the default LM Studio endpoint.
    pub fn new_local(model: &str) -> Self {
        Self::with_base_url("http://localhost:1234/v1", model)
    }

    /// Create a client with a custom base URL.
    pub fn with_base_url(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Send a chat completion request and return the response text.
    pub async fn chat(&self, system: &str, user: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        debug!(model = %self.model, "sending chat completion request");

        let body = ChatRequest {
            model: &self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: system,
                },
                ChatMessage {
                    role: "user",
                    content: user,
                },
            ],
            temperature: 0.3,
            max_tokens: 2000,
        };

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| {
                format!("failed to connect to LLM at {url} -- is the server running?")
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            anyhow::bail!("LLM returned HTTP {status}: {body_text}");
        }

        let chat_resp: ChatResponse = response
            .json()
            .await
            .context("failed to parse chat completion response")?;

        chat_resp
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("LLM returned empty choices array"))
    }
}

// ---------------------------------------------------------------------------
// Distillation output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DistilledSession {
    pub session_id: String,
    pub decisions: Vec<ExtractedDecision>,
    pub problems: Vec<ExtractedProblem>,
    pub patterns: Vec<ExtractedPattern>,
    pub entities: Vec<ExtractedEntity>,
    pub key_insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedDecision {
    pub what: String,
    pub why: Option<String>,
    pub alternatives: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedProblem {
    pub description: String,
    pub solution: Option<String>,
    /// One of: "solved", "open", "workaround"
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedPattern {
    pub name: String,
    pub description: String,
    pub frequency: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    /// One of: "function", "file", "class", "module", "crate", "package"
    pub entity_type: String,
    pub context: Option<String>,
}

// ---------------------------------------------------------------------------
// LLM JSON response shape (for serde parsing)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
struct LlmExtraction {
    #[serde(default)]
    decisions: Vec<LlmDecision>,
    #[serde(default)]
    problems: Vec<LlmProblem>,
    #[serde(default)]
    patterns: Vec<LlmPattern>,
    #[serde(default)]
    entities: Vec<LlmEntity>,
    #[serde(default)]
    key_insights: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LlmDecision {
    what: String,
    why: Option<String>,
    #[serde(default)]
    alternatives: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LlmProblem {
    description: String,
    solution: Option<String>,
    #[serde(default = "default_status")]
    status: String,
}

fn default_status() -> String {
    "open".to_string()
}

#[derive(Debug, Deserialize)]
struct LlmPattern {
    name: String,
    description: String,
    #[serde(default = "default_frequency")]
    frequency: Option<u32>,
}

fn default_frequency() -> Option<u32> {
    Some(1)
}

#[derive(Debug, Deserialize)]
struct LlmEntity {
    name: String,
    entity_type: String,
    context: Option<String>,
}

// ---------------------------------------------------------------------------
// Distiller
// ---------------------------------------------------------------------------

/// Extracts structured knowledge from session transcripts using an LLM
/// or keyword-based fallback.
pub struct Distiller {
    level: DistillationLevel,
    llm: Option<LlmClient>,
}

impl Distiller {
    /// Build a distiller from the application configuration.
    pub fn new(config: &DistillationConfig) -> Self {
        let llm = if config.llm_model.is_empty() {
            None
        } else if config.llm_provider.is_empty() {
            // Empty provider => local LM Studio
            Some(LlmClient::new_local(&config.llm_model))
        } else {
            // Treat provider as a base URL
            Some(LlmClient::with_base_url(
                &config.llm_provider,
                &config.llm_model,
            ))
        };

        Self {
            level: config.level,
            llm,
        }
    }

    /// Distill a session transcript into structured knowledge.
    ///
    /// Uses LLM if available and the distillation level warrants it,
    /// otherwise falls back to keyword extraction.
    pub async fn distill_session(
        &self,
        session: &Session,
        transcript_text: &str,
    ) -> Result<DistilledSession> {
        if self.level == DistillationLevel::None {
            debug!("distillation level is None, returning empty result");
            return Ok(DistilledSession {
                session_id: session.id.clone(),
                ..Default::default()
            });
        }

        let cleaned = Self::filter_noise(transcript_text);

        if self.level == DistillationLevel::Minimal {
            info!(session_id = %session.id, "keyword-only distillation (Minimal)");
            return Ok(self.extract_keywords_inner(&session.id, &cleaned));
        }

        // Balanced / Aggressive / Full: attempt LLM, fall back to keywords.
        if self.llm.is_some() {
            match self.extract_with_llm(session, &cleaned).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!(
                        error = %e,
                        "LLM extraction failed, falling back to keyword extraction"
                    );
                }
            }
        } else {
            debug!("no LLM configured, using keyword extraction");
        }

        Ok(self.extract_keywords_inner(&session.id, &cleaned))
    }

    /// Keyword-based fallback extraction (no LLM needed).
    pub fn extract_keywords(&self, text: &str) -> DistilledSession {
        self.extract_keywords_inner("unknown", text)
    }

    fn extract_keywords_inner(&self, session_id: &str, text: &str) -> DistilledSession {
        let entities = Self::extract_entities(text);
        let decisions = Self::extract_decisions(text);
        let problems = Self::extract_problems(text);
        let patterns = Self::extract_patterns(text);

        DistilledSession {
            session_id: session_id.to_string(),
            decisions,
            problems,
            patterns,
            entities,
            key_insights: Vec::new(),
        }
    }

    /// LLM-based extraction with structured prompts.
    async fn extract_with_llm(&self, session: &Session, text: &str) -> Result<DistilledSession> {
        let llm = self
            .llm
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no LLM client configured"))?;

        // Truncate for local models (max ~4000 chars).
        let truncated = if text.len() > 4000 {
            &text[..4000]
        } else {
            text
        };

        let system_prompt = "You are a knowledge extraction assistant. Analyze coding agent \
             transcripts and extract structured knowledge. Respond ONLY with valid JSON.";

        let user_prompt = format!(
            "Analyze this coding session transcript and extract:\n\
             1. Key decisions made (what was decided and why)\n\
             2. Problems encountered and their solutions\n\
             3. Notable patterns or conventions established\n\
             4. Important code entities (files, functions, classes)\n\n\
             Transcript:\n{truncated}\n\n\
             Respond with JSON:\n\
             {{\"decisions\": [{{\"what\": \"...\", \"why\": \"...\", \"alternatives\": [...]}}], \
             \"problems\": [{{\"description\": \"...\", \"solution\": \"...\", \
             \"status\": \"solved|open|workaround\"}}], \
             \"patterns\": [{{\"name\": \"...\", \"description\": \"...\"}}], \
             \"entities\": [{{\"name\": \"...\", \
             \"entity_type\": \"function|file|class|module\", \"context\": \"...\"}}], \
             \"key_insights\": [\"...\"]}}"
        );

        info!(session_id = %session.id, "extracting knowledge with LLM");
        let raw = llm.chat(system_prompt, &user_prompt).await?;

        // Try to parse the JSON response. The LLM might wrap it in markdown
        // code fences, so strip those first.
        let json_str = strip_code_fences(&raw);

        let extraction: LlmExtraction = serde_json::from_str(json_str).with_context(|| {
            format!(
                "failed to parse LLM JSON response: {}",
                &raw[..raw.len().min(200)]
            )
        })?;

        Ok(DistilledSession {
            session_id: session.id.clone(),
            decisions: extraction
                .decisions
                .into_iter()
                .map(|d| ExtractedDecision {
                    what: d.what,
                    why: d.why,
                    alternatives: d.alternatives,
                })
                .collect(),
            problems: extraction
                .problems
                .into_iter()
                .map(|p| ExtractedProblem {
                    description: p.description,
                    solution: p.solution,
                    status: p.status,
                })
                .collect(),
            patterns: extraction
                .patterns
                .into_iter()
                .map(|p| ExtractedPattern {
                    name: p.name,
                    description: p.description,
                    frequency: p.frequency.unwrap_or(1),
                })
                .collect(),
            entities: extraction
                .entities
                .into_iter()
                .map(|e| ExtractedEntity {
                    name: e.name,
                    entity_type: e.entity_type,
                    context: e.context,
                })
                .collect(),
            key_insights: extraction.key_insights,
        })
    }

    /// Convert distilled decisions into DuckDB `Decision` records.
    pub fn to_decisions(&self, distilled: &DistilledSession) -> Vec<Decision> {
        distilled
            .decisions
            .iter()
            .map(|d| Decision {
                id: Uuid::new_v4().to_string(),
                session_id: Some(distilled.session_id.clone()),
                project_id: None,
                decision_type: Some("extracted".to_string()),
                what: d.what.clone(),
                why: d.why.clone(),
                alternatives: d.alternatives.clone(),
                outcome: None,
                created_at: None,
                valid_until: None,
            })
            .collect()
    }

    /// Convert distilled results into DuckDB `Memory` records.
    ///
    /// Creates memories from problems, patterns, entities, and key insights.
    pub fn to_memories(&self, distilled: &DistilledSession) -> Vec<Memory> {
        let mut memories = Vec::new();

        // Problems -> memories
        for p in &distilled.problems {
            let content = match &p.solution {
                Some(sol) => format!(
                    "Problem: {} | Solution: {} [{}]",
                    p.description, sol, p.status
                ),
                None => format!("Problem: {} [{}]", p.description, p.status),
            };
            memories.push(Memory {
                id: Uuid::new_v4().to_string(),
                project_id: None,
                content,
                memory_type: Some("problem".to_string()),
                source_session_id: Some(distilled.session_id.clone()),
                confidence: 0.8,
                access_count: 0,
                created_at: None,
                updated_at: None,
                valid_until: None,
            });
        }

        // Patterns -> memories
        for p in &distilled.patterns {
            memories.push(Memory {
                id: Uuid::new_v4().to_string(),
                project_id: None,
                content: format!("Pattern [{}]: {}", p.name, p.description),
                memory_type: Some("pattern".to_string()),
                source_session_id: Some(distilled.session_id.clone()),
                confidence: 0.7,
                access_count: 0,
                created_at: None,
                updated_at: None,
                valid_until: None,
            });
        }

        // Entities -> memories
        for e in &distilled.entities {
            let content = match &e.context {
                Some(ctx) => format!("Entity [{}] {}: {}", e.entity_type, e.name, ctx),
                None => format!("Entity [{}] {}", e.entity_type, e.name),
            };
            memories.push(Memory {
                id: Uuid::new_v4().to_string(),
                project_id: None,
                content,
                memory_type: Some("entity".to_string()),
                source_session_id: Some(distilled.session_id.clone()),
                confidence: 0.9,
                access_count: 0,
                created_at: None,
                updated_at: None,
                valid_until: None,
            });
        }

        // Key insights -> memories
        for insight in &distilled.key_insights {
            memories.push(Memory {
                id: Uuid::new_v4().to_string(),
                project_id: None,
                content: insight.clone(),
                memory_type: Some("insight".to_string()),
                source_session_id: Some(distilled.session_id.clone()),
                confidence: 0.85,
                access_count: 0,
                created_at: None,
                updated_at: None,
                valid_until: None,
            });
        }

        memories
    }

    /// Filter noise: skip trivial messages, failed commands, duplicate content.
    pub fn filter_noise(text: &str) -> String {
        let mut seen = HashSet::new();
        let mut result_lines: Vec<&str> = Vec::new();
        let mut consecutive_long_block = 0u32;

        for line in text.lines() {
            let trimmed = line.trim();

            // Skip very short non-code lines.
            if trimmed.len() < 5
                && !trimmed.contains('/')
                && !trimmed.contains('.')
                && !trimmed.contains('{')
                && !trimmed.contains('}')
            {
                consecutive_long_block = 0;
                continue;
            }

            // Skip trivial filler lines.
            let lower = trimmed.to_lowercase();
            if lower == "please wait"
                || lower == "loading..."
                || lower == "thinking..."
                || lower == "please wait..."
                || lower.starts_with("waiting for")
            {
                continue;
            }

            // Detect long tool output blocks (file listings, build logs).
            // If we see > 50 consecutive lines that look like output, skip them.
            if looks_like_tool_output(trimmed) {
                consecutive_long_block += 1;
                if consecutive_long_block > 50 {
                    continue;
                }
            } else {
                consecutive_long_block = 0;
            }

            // Deduplicate identical lines.
            if !seen.insert(trimmed) {
                continue;
            }

            result_lines.push(line);
        }

        result_lines.join("\n")
    }

    // -----------------------------------------------------------------------
    // Keyword extraction helpers
    // -----------------------------------------------------------------------

    fn extract_entities(text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let mut seen = HashSet::new();

        // File paths: word boundaries around path-like strings.
        let file_re =
            Regex::new(r#"(?:^|[\s`"'(])([a-zA-Z0-9_./-]+/[a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)"#)
                .expect("file regex");
        for cap in file_re.captures_iter(text) {
            let path = cap[1].to_string();
            if seen.insert(path.clone()) {
                entities.push(ExtractedEntity {
                    name: path,
                    entity_type: "file".to_string(),
                    context: None,
                });
            }
        }

        // Rust functions: fn name
        let fn_re = Regex::new(r"\bfn\s+([a-zA-Z_][a-zA-Z0-9_]*)").expect("fn regex");
        for cap in fn_re.captures_iter(text) {
            let name = cap[1].to_string();
            if seen.insert(format!("fn:{name}")) {
                entities.push(ExtractedEntity {
                    name,
                    entity_type: "function".to_string(),
                    context: None,
                });
            }
        }

        // Python functions: def name
        let def_re = Regex::new(r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)").expect("def regex");
        for cap in def_re.captures_iter(text) {
            let name = cap[1].to_string();
            if seen.insert(format!("def:{name}")) {
                entities.push(ExtractedEntity {
                    name,
                    entity_type: "function".to_string(),
                    context: None,
                });
            }
        }

        // JS functions: function name
        let function_re =
            Regex::new(r"\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]*)").expect("function regex");
        for cap in function_re.captures_iter(text) {
            let name = cap[1].to_string();
            if seen.insert(format!("function:{name}")) {
                entities.push(ExtractedEntity {
                    name,
                    entity_type: "function".to_string(),
                    context: None,
                });
            }
        }

        // Classes: class Name
        let class_re = Regex::new(r"\bclass\s+([A-Z][a-zA-Z0-9_]*)").expect("class regex");
        for cap in class_re.captures_iter(text) {
            let name = cap[1].to_string();
            if seen.insert(format!("class:{name}")) {
                entities.push(ExtractedEntity {
                    name,
                    entity_type: "class".to_string(),
                    context: None,
                });
            }
        }

        entities
    }

    fn extract_decisions(text: &str) -> Vec<ExtractedDecision> {
        let decision_re = Regex::new(
            r"(?i)(decided|chose|went with|switched to|selected|picked|opted for)\s+(.+)",
        )
        .expect("decision regex");

        let mut decisions = Vec::new();
        for cap in decision_re.captures_iter(text) {
            let what = cap[2].trim().to_string();
            // Truncate at sentence boundary or 200 chars.
            let what = truncate_at_sentence(&what, 200);
            if !what.is_empty() {
                decisions.push(ExtractedDecision {
                    what,
                    why: None,
                    alternatives: Vec::new(),
                });
            }
        }

        decisions
    }

    fn extract_problems(text: &str) -> Vec<ExtractedProblem> {
        let problem_re =
            Regex::new(r"(?i)(?:error|bug|issue|failed|broken|crash|panic)[:.]?\s+(.+)")
                .expect("problem regex");

        let mut problems = Vec::new();
        let mut seen = HashSet::new();
        for cap in problem_re.captures_iter(text) {
            let desc = cap[1].trim().to_string();
            let desc = truncate_at_sentence(&desc, 200);
            if !desc.is_empty() && seen.insert(desc.clone()) {
                problems.push(ExtractedProblem {
                    description: desc,
                    solution: None,
                    status: "open".to_string(),
                });
            }
        }

        problems
    }

    fn extract_patterns(text: &str) -> Vec<ExtractedPattern> {
        let pattern_re = Regex::new(r"(?i)(pattern|always|never|convention|rule)[:.]?\s+(.+)")
            .expect("pattern regex");

        let mut patterns = Vec::new();
        let mut seen = HashSet::new();
        for cap in pattern_re.captures_iter(text) {
            let keyword = cap[1].to_lowercase();
            let desc = cap[2].trim().to_string();
            let desc = truncate_at_sentence(&desc, 200);
            if !desc.is_empty() && seen.insert(desc.clone()) {
                patterns.push(ExtractedPattern {
                    name: keyword,
                    description: desc,
                    frequency: 1,
                });
            }
        }

        patterns
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Strip markdown code fences (```json ... ```) that LLMs sometimes wrap
/// their JSON output in.
fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();

    // Try to find opening ```json or ``` and closing ```.
    let start = if let Some(pos) = trimmed.find("```json") {
        pos + 7
    } else if trimmed.starts_with("```") {
        3
    } else {
        return trimmed;
    };

    let inner = &trimmed[start..];
    // Skip a leading newline after the opening fence.
    let inner = inner.strip_prefix('\n').unwrap_or(inner);

    if let Some(end) = inner.rfind("```") {
        inner[..end].trim()
    } else {
        inner.trim()
    }
}

/// Heuristic: does this line look like tool output (build log, file listing)?
fn looks_like_tool_output(line: &str) -> bool {
    // Lines starting with common log prefixes or path-only lines.
    line.starts_with("   Compiling")
        || line.starts_with("   Downloading")
        || line.starts_with("    Finished")
        || line.starts_with("warning[")
        || line.starts_with("error[")
        || line.starts_with("  --> ")
        || (line.starts_with("drwx") || line.starts_with("-rw-"))
        || line.starts_with("total ")
}

/// Truncate a string at the first sentence boundary or at `max_len`.
fn truncate_at_sentence(text: &str, max_len: usize) -> String {
    let text = text.trim();
    if text.len() <= max_len {
        return text.to_string();
    }
    // Find the first sentence-ending punctuation within max_len.
    if let Some(pos) = text[..max_len].rfind(['.', '!', '?']) {
        text[..=pos].to_string()
    } else {
        format!("{}...", &text[..max_len.saturating_sub(3)])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(id: &str) -> Session {
        Session {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            agent: "claude".into(),
            started_at: None,
            ended_at: None,
            duration_minutes: None,
            message_count: None,
            tool_call_count: None,
            total_tokens: None,
            files_changed: vec![],
            summary: None,
        }
    }

    #[test]
    fn test_keyword_extraction_finds_file_paths() {
        let text = "I modified src/main.rs and engine/src/store.rs to fix the issue.";
        let distiller = Distiller {
            level: DistillationLevel::Minimal,
            llm: None,
        };
        let result = distiller.extract_keywords(text);

        let file_entities: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.entity_type == "file")
            .collect();
        assert!(
            file_entities.iter().any(|e| e.name == "src/main.rs"),
            "should find src/main.rs, got: {file_entities:?}"
        );
        assert!(
            file_entities
                .iter()
                .any(|e| e.name == "engine/src/store.rs"),
            "should find engine/src/store.rs, got: {file_entities:?}"
        );
    }

    #[test]
    fn test_keyword_extraction_finds_decisions() {
        let text = "After some discussion, we decided to use DuckDB for the structured store.\n\
                     We also chose serde for serialization.";
        let distiller = Distiller {
            level: DistillationLevel::Balanced,
            llm: None,
        };
        let result = distiller.extract_keywords(text);

        assert!(
            !result.decisions.is_empty(),
            "should extract at least one decision"
        );
        assert!(
            result
                .decisions
                .iter()
                .any(|d| d.what.to_lowercase().contains("duckdb")
                    || d.what.to_lowercase().contains("serde")),
            "decisions should mention DuckDB or serde: {:?}",
            result.decisions
        );
    }

    #[test]
    fn test_noise_filtering() {
        let text = "Starting work on the module.\n\
                     please wait\n\
                     loading...\n\
                     thinking...\n\
                     ok\n\
                     I will now create the file.\n\
                     I will now create the file.\n\
                     Done.";
        let filtered = Distiller::filter_noise(text);

        assert!(
            !filtered.contains("please wait"),
            "should remove 'please wait'"
        );
        assert!(
            !filtered.contains("loading..."),
            "should remove 'loading...'"
        );
        assert!(
            !filtered.contains("thinking..."),
            "should remove 'thinking...'"
        );
        // "ok" has < 5 chars and no special chars, should be removed.
        assert!(
            !filtered
                .to_lowercase()
                .split('\n')
                .any(|l| l.trim() == "ok"),
            "should remove short non-code lines"
        );
        // Duplicate line should appear only once.
        let count = filtered
            .lines()
            .filter(|l| l.contains("I will now create the file"))
            .count();
        assert_eq!(count, 1, "duplicate lines should be deduplicated");
        // Meaningful lines should remain.
        assert!(filtered.contains("Starting work on the module"));
        assert!(filtered.contains("Done."));
    }

    #[tokio::test]
    async fn test_distill_level_none_returns_empty() {
        let config = DistillationConfig {
            level: DistillationLevel::None,
            llm_provider: String::new(),
            llm_model: String::new(),
            api_key_env: String::new(),
        };
        let distiller = Distiller::new(&config);
        let session = make_session("s-none");
        let result = distiller
            .distill_session(&session, "lots of interesting content here")
            .await
            .unwrap();

        assert_eq!(result.session_id, "s-none");
        assert!(result.decisions.is_empty());
        assert!(result.problems.is_empty());
        assert!(result.patterns.is_empty());
        assert!(result.entities.is_empty());
        assert!(result.key_insights.is_empty());
    }

    #[test]
    fn test_to_decisions_converts_correctly() {
        let distilled = DistilledSession {
            session_id: "s-42".to_string(),
            decisions: vec![
                ExtractedDecision {
                    what: "use Rust for the engine".to_string(),
                    why: Some("performance and safety".to_string()),
                    alternatives: vec!["Go".to_string(), "Python".to_string()],
                },
                ExtractedDecision {
                    what: "use DuckDB for storage".to_string(),
                    why: None,
                    alternatives: vec![],
                },
            ],
            problems: vec![],
            patterns: vec![],
            entities: vec![],
            key_insights: vec![],
        };

        let distiller = Distiller {
            level: DistillationLevel::Balanced,
            llm: None,
        };
        let decisions = distiller.to_decisions(&distilled);

        assert_eq!(decisions.len(), 2);
        assert_eq!(decisions[0].what, "use Rust for the engine");
        assert_eq!(decisions[0].why, Some("performance and safety".to_string()));
        assert_eq!(decisions[0].alternatives, vec!["Go", "Python"]);
        assert_eq!(decisions[0].session_id, Some("s-42".to_string()));
        assert_eq!(decisions[0].decision_type, Some("extracted".to_string()));
        // Each decision should have a unique UUID.
        assert_ne!(decisions[0].id, decisions[1].id);
        // IDs should be valid UUIDs.
        assert!(
            Uuid::parse_str(&decisions[0].id).is_ok(),
            "id should be a valid UUID"
        );
    }

    #[test]
    fn test_to_memories_converts_all_types() {
        let distilled = DistilledSession {
            session_id: "s-mem".to_string(),
            decisions: vec![],
            problems: vec![ExtractedProblem {
                description: "build fails on CI".to_string(),
                solution: Some("pinned dependency version".to_string()),
                status: "solved".to_string(),
            }],
            patterns: vec![ExtractedPattern {
                name: "convention".to_string(),
                description: "always use anyhow::Result".to_string(),
                frequency: 3,
            }],
            entities: vec![ExtractedEntity {
                name: "DuckStore".to_string(),
                entity_type: "class".to_string(),
                context: Some("handles structured storage".to_string()),
            }],
            key_insights: vec!["DuckDB is fast for analytics".to_string()],
        };

        let distiller = Distiller {
            level: DistillationLevel::Balanced,
            llm: None,
        };
        let memories = distiller.to_memories(&distilled);

        // 1 problem + 1 pattern + 1 entity + 1 insight = 4
        assert_eq!(memories.len(), 4);

        let types: Vec<_> = memories
            .iter()
            .filter_map(|m| m.memory_type.as_deref())
            .collect();
        assert!(types.contains(&"problem"));
        assert!(types.contains(&"pattern"));
        assert!(types.contains(&"entity"));
        assert!(types.contains(&"insight"));

        // Check problem memory includes solution.
        let problem_mem = memories
            .iter()
            .find(|m| m.memory_type.as_deref() == Some("problem"))
            .unwrap();
        assert!(problem_mem.content.contains("pinned dependency version"));
    }

    #[test]
    fn test_strip_code_fences() {
        let with_fences = "```json\n{\"key\": \"value\"}\n```";
        assert_eq!(strip_code_fences(with_fences), "{\"key\": \"value\"}");

        let plain = "{\"key\": \"value\"}";
        assert_eq!(strip_code_fences(plain), plain);

        let with_plain_fences = "```\n{\"key\": \"value\"}\n```";
        assert_eq!(strip_code_fences(with_plain_fences), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_extract_functions_and_classes() {
        let text = "We defined fn process_batch and class DataLoader for the pipeline.\n\
                     Also added def compute_stats in the Python module.\n\
                     And function renderChart in the frontend.";
        let distiller = Distiller {
            level: DistillationLevel::Minimal,
            llm: None,
        };
        let result = distiller.extract_keywords(text);

        let names: Vec<_> = result.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"process_batch"), "should find Rust fn");
        assert!(names.contains(&"DataLoader"), "should find class");
        assert!(names.contains(&"compute_stats"), "should find Python def");
        assert!(names.contains(&"renderChart"), "should find JS function");
    }
}
