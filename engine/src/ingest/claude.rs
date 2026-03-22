//! Claude Code artifact ingestion.
//!
//! Reads Claude Code sessions, transcripts, and memory files from `~/.claude/`
//! and converts them into Remembrant domain structs ([`Session`], [`ToolCall`],
//! [`Memory`]) ready for storage in DuckDB.

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::store::duckdb::{Memory, Session, ToolCall};

// ---------------------------------------------------------------------------
// Serde structs — sessions-index.json
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SessionsIndex {
    #[allow(dead_code)]
    version: Option<u32>,
    entries: Vec<SessionEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionEntry {
    session_id: String,
    #[allow(dead_code)]
    full_path: Option<String>,
    #[allow(dead_code)]
    file_mtime: Option<u64>,
    #[allow(dead_code)]
    first_prompt: Option<String>,
    summary: Option<String>,
    message_count: Option<i32>,
    created: Option<String>,
    modified: Option<String>,
    #[allow(dead_code)]
    git_branch: Option<String>,
    #[allow(dead_code)]
    project_path: Option<String>,
    #[allow(dead_code)]
    is_sidechain: Option<bool>,
}

// ---------------------------------------------------------------------------
// Serde structs — JSONL transcript lines
// ---------------------------------------------------------------------------

/// A single line from a session `.jsonl` transcript.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptLine {
    #[serde(rename = "type")]
    line_type: Option<String>,
    message: Option<TranscriptMessage>,
    timestamp: Option<String>,
    #[allow(dead_code)]
    uuid: Option<String>,
    #[allow(dead_code)]
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TranscriptMessage {
    role: Option<String>,
    content: Option<serde_json::Value>,
    #[allow(dead_code)]
    stop_reason: Option<String>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
}

/// Represents a single content block inside an assistant message.
#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: Option<String>,
    /// Tool name (present when `block_type == "tool_use"`).
    name: Option<String>,
    /// Tool input (present when `block_type == "tool_use"`).
    input: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// IngestResult
// ---------------------------------------------------------------------------

/// Summary of what a full ingestion pass discovered.
#[derive(Debug, Default)]
pub struct IngestResult {
    pub sessions_count: usize,
    pub tool_calls_count: usize,
    pub memories_count: usize,
    pub sessions: Vec<Session>,
    pub tool_calls: Vec<ToolCall>,
    pub memories: Vec<Memory>,
}

// ---------------------------------------------------------------------------
// Token stats returned alongside tool calls from transcript parsing.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct TranscriptStats {
    pub tool_calls: Vec<ToolCall>,
    pub message_count: i32,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub files_changed: Vec<String>,
}

// ---------------------------------------------------------------------------
// ClaudeIngester
// ---------------------------------------------------------------------------

/// Discovers and parses Claude Code artifacts from the local filesystem.
pub struct ClaudeIngester {
    /// Root of the Claude Code config directory (typically `~/.claude`).
    base_path: PathBuf,
}

impl ClaudeIngester {
    /// Create a new ingester, auto-detecting `~/.claude`.
    ///
    /// Returns `Err` if the home directory cannot be determined.
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir().context("could not determine home directory")?;
        let base_path = home.join(".claude");
        debug!("ClaudeIngester base_path: {}", base_path.display());
        Ok(Self { base_path })
    }

    /// Create an ingester rooted at an explicit path (useful for testing).
    pub fn with_base_path(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    // -------------------------------------------------------------------
    // Discovery
    // -------------------------------------------------------------------

    /// Return every project directory found under `~/.claude/projects/`.
    pub fn discover_projects(&self) -> Vec<PathBuf> {
        let projects_dir = self.base_path.join("projects");
        if !projects_dir.is_dir() {
            debug!("no projects directory at {}", projects_dir.display());
            return Vec::new();
        }

        let mut dirs = Vec::new();
        match fs::read_dir(&projects_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        dirs.push(path);
                    }
                }
            }
            Err(e) => {
                warn!(
                    "failed to read projects dir {}: {e}",
                    projects_dir.display()
                );
            }
        }
        dirs.sort();
        debug!("discovered {} project dirs", dirs.len());
        dirs
    }

    // -------------------------------------------------------------------
    // Sessions index
    // -------------------------------------------------------------------

    /// Parse `sessions-index.json` inside a project directory and convert
    /// each entry into a domain [`Session`].
    pub fn parse_sessions_index(&self, project_dir: &Path) -> Result<Vec<Session>> {
        let index_path = project_dir.join("sessions-index.json");
        if !index_path.is_file() {
            debug!("no sessions-index.json in {}", project_dir.display());
            return Ok(Vec::new());
        }

        let data = fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;

        let index: SessionsIndex = serde_json::from_str(&data)
            .with_context(|| format!("failed to parse {}", index_path.display()))?;

        let project_id = project_id_from_dir(project_dir);

        let sessions: Vec<Session> = index
            .entries
            .into_iter()
            .map(|e| {
                let started_at = e.created.as_deref().and_then(parse_iso8601);
                let ended_at = e.modified.as_deref().and_then(parse_iso8601);
                let duration_minutes = match (started_at, ended_at) {
                    (Some(s), Some(e)) => {
                        let dur = e.signed_duration_since(s);
                        Some(dur.num_minutes() as i32)
                    }
                    _ => None,
                };

                Session {
                    id: e.session_id,
                    project_id: Some(project_id.clone()),
                    agent: "claude_code".to_string(),
                    started_at,
                    ended_at,
                    duration_minutes,
                    message_count: e.message_count,
                    tool_call_count: None, // filled later from transcript
                    total_tokens: None,    // filled later from transcript
                    files_changed: Vec::new(),
                    summary: e.summary,
                }
            })
            .collect();

        debug!(
            "parsed {} sessions from {}",
            sessions.len(),
            index_path.display()
        );
        Ok(sessions)
    }

    // -------------------------------------------------------------------
    // Session transcript
    // -------------------------------------------------------------------

    /// Parse a single `.jsonl` transcript file.
    ///
    /// Returns extracted tool calls and aggregate token/message statistics.
    /// Malformed lines are skipped with a warning.
    pub fn parse_session_transcript(
        &self,
        jsonl_path: &Path,
        session_id: &str,
    ) -> Result<TranscriptStats> {
        let data = fs::read_to_string(jsonl_path)
            .with_context(|| format!("failed to read {}", jsonl_path.display()))?;

        let mut stats = TranscriptStats::default();

        for (line_no, line) in data.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parsed: TranscriptLine = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        "{}:{}: skipping malformed line: {e}",
                        jsonl_path.display(),
                        line_no + 1
                    );
                    continue;
                }
            };

            // Count messages (user + assistant, skip file-history-snapshot etc.)
            let is_user = parsed.line_type.as_deref() == Some("user");
            let is_assistant = parsed.line_type.as_deref() == Some("assistant");

            if is_user || is_assistant {
                stats.message_count += 1;
            }

            // Accumulate token usage from assistant messages.
            if let Some(ref msg) = parsed.message {
                if let Some(ref usage) = msg.usage {
                    stats.total_input_tokens += usage.input_tokens.unwrap_or(0);
                    stats.total_output_tokens += usage.output_tokens.unwrap_or(0);
                }

                // Extract tool_use blocks from assistant content.
                if msg.role.as_deref() == Some("assistant") {
                    if let Some(content_val) = &msg.content {
                        let blocks = extract_content_blocks(content_val);
                        for block in blocks {
                            if block.block_type.as_deref() != Some("tool_use") {
                                continue;
                            }
                            let tool_name = block.name.clone();
                            let command = extract_tool_command(&block);

                            // Track files changed via Read/Write/Edit tools.
                            if let Some(ref name) = tool_name {
                                if matches!(name.as_str(), "Write" | "Edit" | "Read") {
                                    if let Some(ref input) = block.input {
                                        if let Some(fp) = input
                                            .get("file_path")
                                            .or_else(|| input.get("path"))
                                            .and_then(|v| v.as_str())
                                        {
                                            let fp = fp.to_string();
                                            if !stats.files_changed.contains(&fp) {
                                                stats.files_changed.push(fp);
                                            }
                                        }
                                    }
                                }
                            }

                            let tc = ToolCall {
                                id: Uuid::new_v4().to_string(),
                                session_id: Some(session_id.to_string()),
                                tool_name,
                                command,
                                success: None,
                                error_message: None,
                                duration_ms: None,
                                timestamp: parsed.timestamp.as_deref().and_then(parse_iso8601),
                            };
                            stats.tool_calls.push(tc);
                        }
                    }
                }
            }
        }

        debug!(
            "transcript {}: {} messages, {} tool calls, {} input tokens, {} output tokens",
            jsonl_path.display(),
            stats.message_count,
            stats.tool_calls.len(),
            stats.total_input_tokens,
            stats.total_output_tokens,
        );
        Ok(stats)
    }

    // -------------------------------------------------------------------
    // Memory
    // -------------------------------------------------------------------

    /// Read MEMORY.md files from a project directory's `memory/` subdirectory.
    pub fn parse_memory(&self, project_dir: &Path) -> Result<Vec<Memory>> {
        let memory_dir = project_dir.join("memory");
        let project_id = project_id_from_dir(project_dir);
        let mut memories = Vec::new();

        if !memory_dir.is_dir() {
            debug!("no memory dir at {}", memory_dir.display());
            return Ok(memories);
        }

        match fs::read_dir(&memory_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if ext != "md" {
                        continue;
                    }
                    match fs::read_to_string(&path) {
                        Ok(content) if !content.trim().is_empty() => {
                            let modified = fs::metadata(&path)
                                .ok()
                                .and_then(|m| m.modified().ok())
                                .map(|t| {
                                    let dt: chrono::DateTime<chrono::Utc> = t.into();
                                    dt.naive_utc()
                                });

                            memories.push(Memory {
                                id: Uuid::new_v4().to_string(),
                                project_id: Some(project_id.clone()),
                                content,
                                memory_type: Some("claude_code_memory".to_string()),
                                source_session_id: None,
                                confidence: 1.0,
                                access_count: 0,
                                created_at: modified,
                                updated_at: modified,
                                valid_until: None,
                            });
                        }
                        Ok(_) => {
                            debug!("skipping empty memory file {}", path.display());
                        }
                        Err(e) => {
                            warn!("failed to read memory file {}: {e}", path.display());
                        }
                    }
                }
            }
            Err(e) => {
                warn!("failed to read memory dir {}: {e}", memory_dir.display());
            }
        }

        debug!(
            "parsed {} memories from {}",
            memories.len(),
            memory_dir.display()
        );
        Ok(memories)
    }

    // -------------------------------------------------------------------
    // Top-level ingest
    // -------------------------------------------------------------------

    /// Discover all Claude Code projects and ingest sessions, tool calls,
    /// and memories.
    pub fn ingest_all(&self) -> Result<IngestResult> {
        let mut result = IngestResult::default();
        let project_dirs = self.discover_projects();

        for project_dir in &project_dirs {
            let project_id = project_id_from_dir(project_dir);

            // Sessions from index
            let mut sessions = self.parse_sessions_index(project_dir)?;
            let indexed_ids: std::collections::HashSet<String> =
                sessions.iter().map(|s| s.id.clone()).collect();

            // Discover orphaned .jsonl files not in the index (top-level)
            if let Ok(entries) = fs::read_dir(project_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                        continue;
                    }
                    let session_id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    if session_id.is_empty() || indexed_ids.contains(&session_id) {
                        continue;
                    }
                    debug!("found orphaned transcript: {}", path.display());
                    let modified = fs::metadata(&path)
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .map(|t| {
                            let dt: chrono::DateTime<chrono::Utc> = t.into();
                            dt.naive_utc()
                        });
                    sessions.push(Session {
                        id: session_id,
                        project_id: Some(project_id.clone()),
                        agent: "claude_code".to_string(),
                        started_at: modified,
                        ended_at: modified,
                        duration_minutes: None,
                        message_count: None,
                        tool_call_count: None,
                        total_tokens: None,
                        files_changed: Vec::new(),
                        summary: None,
                    });
                }
            }

            // Discover subagent transcripts inside <session-id>/subagents/
            if let Ok(entries) = fs::read_dir(project_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_dir() || path.file_name().map_or(true, |n| n == "memory") {
                        continue;
                    }
                    let subagents_dir = path.join("subagents");
                    if !subagents_dir.is_dir() {
                        continue;
                    }
                    let parent_session_id = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();

                    if let Ok(sub_entries) = fs::read_dir(&subagents_dir) {
                        for sub_entry in sub_entries.flatten() {
                            let sub_path = sub_entry.path();
                            if sub_path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                                continue;
                            }
                            let agent_id = sub_path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("")
                                .to_string();
                            if agent_id.is_empty() {
                                continue;
                            }

                            // Read optional .meta.json for description
                            let meta_path = subagents_dir.join(format!("{agent_id}.meta.json"));
                            let summary = if meta_path.is_file() {
                                fs::read_to_string(&meta_path)
                                    .ok()
                                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                                    .and_then(|v| {
                                        let desc = v.get("description")?.as_str()?;
                                        let agent_type = v.get("agentType")
                                            .and_then(|t| t.as_str())
                                            .unwrap_or("subagent");
                                        Some(format!("[{agent_type}] {desc}"))
                                    })
                            } else {
                                None
                            };

                            let modified = fs::metadata(&sub_path)
                                .ok()
                                .and_then(|m| m.modified().ok())
                                .map(|t| {
                                    let dt: chrono::DateTime<chrono::Utc> = t.into();
                                    dt.naive_utc()
                                });

                            let sub_session_id = format!("{parent_session_id}/{agent_id}");
                            sessions.push(Session {
                                id: sub_session_id,
                                project_id: Some(project_id.clone()),
                                agent: "claude_code".to_string(),
                                started_at: modified,
                                ended_at: modified,
                                duration_minutes: None,
                                message_count: None,
                                tool_call_count: None,
                                total_tokens: None,
                                files_changed: Vec::new(),
                                summary,
                            });
                        }
                    }
                }
            }

            // For each session, try to parse its transcript
            for session in &mut sessions {
                // Resolve transcript path: subagents use "parent/agent" IDs
                let jsonl_path = if session.id.contains('/') {
                    let parts: Vec<&str> = session.id.splitn(2, '/').collect();
                    project_dir.join(parts[0]).join("subagents").join(format!("{}.jsonl", parts[1]))
                } else {
                    project_dir.join(format!("{}.jsonl", session.id))
                };
                if jsonl_path.is_file() {
                    match self.parse_session_transcript(&jsonl_path, &session.id) {
                        Ok(stats) => {
                            session.tool_call_count = Some(stats.tool_calls.len() as i32);
                            let total = stats.total_input_tokens + stats.total_output_tokens;
                            session.total_tokens = Some(total as i32);
                            if session.message_count.is_none() || session.message_count == Some(0) {
                                session.message_count = Some(stats.message_count);
                            }
                            if !stats.files_changed.is_empty() {
                                session.files_changed = stats.files_changed;
                            }
                            // Derive summary from first user message if missing
                            if session.summary.is_none() {
                                session.summary = Self::extract_first_prompt(&jsonl_path);
                            }
                            result.tool_calls.extend(stats.tool_calls);
                        }
                        Err(e) => {
                            warn!("failed to parse transcript {}: {e}", jsonl_path.display());
                        }
                    }
                }
            }

            result.sessions.extend(sessions);

            // Memories
            match self.parse_memory(project_dir) {
                Ok(mems) => result.memories.extend(mems),
                Err(e) => {
                    warn!("failed to parse memory for {}: {e}", project_dir.display());
                }
            }
        }

        result.sessions_count = result.sessions.len();
        result.tool_calls_count = result.tool_calls.len();
        result.memories_count = result.memories.len();

        debug!(
            "ingest complete: {} sessions, {} tool calls, {} memories",
            result.sessions_count, result.tool_calls_count, result.memories_count
        );
        Ok(result)
    }

    /// Extract the first user prompt from a `.jsonl` transcript for use as summary.
    fn extract_first_prompt(jsonl_path: &Path) -> Option<String> {
        let data = fs::read_to_string(jsonl_path).ok()?;
        for line in data.lines() {
            let parsed: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
            if parsed.get("type").and_then(|t| t.as_str()) == Some("user") {
                let content = parsed
                    .pointer("/message/content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                if let Some(text) = content {
                    let trimmed = text.trim();
                    if trimmed.chars().count() > 200 {
                        let truncated: String = trimmed.chars().take(200).collect();
                        return Some(format!("{truncated}..."));
                    }
                    return Some(trimmed.to_string());
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse an ISO 8601 timestamp string (e.g. `"2026-03-12T23:26:44.124Z"`)
/// into a [`NaiveDateTime`]. Returns `None` on failure.
fn parse_iso8601(s: &str) -> Option<NaiveDateTime> {
    // Try RFC 3339 / ISO 8601 with timezone
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.naive_utc());
    }
    // Try without timezone (unlikely but defensive)
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(dt);
    }
    warn!("could not parse timestamp: {s}");
    None
}

/// Derive a project identifier from the directory name.
///
/// Claude Code uses path-derived names like `-Users-foo-Projects-bar`.
fn project_id_from_dir(dir: &Path) -> String {
    dir.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Extract [`ContentBlock`]s from a `serde_json::Value` that may be an array
/// of blocks or a single string.
fn extract_content_blocks(val: &serde_json::Value) -> Vec<ContentBlock> {
    match val {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| serde_json::from_value::<ContentBlock>(v.clone()).ok())
            .collect(),
        _ => Vec::new(),
    }
}

/// Best-effort extraction of a human-readable command string from a tool_use
/// content block.
fn extract_tool_command(block: &ContentBlock) -> Option<String> {
    let input = block.input.as_ref()?;

    // Bash tool: {"command": "..."}
    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
        return Some(cmd.to_string());
    }
    // Read/Write/Edit: use file_path
    if let Some(fp) = input.get("file_path").and_then(|v| v.as_str()) {
        let tool = block.name.as_deref().unwrap_or("unknown");
        return Some(format!("{tool} {fp}"));
    }
    // Grep: pattern + optional path
    if let Some(pat) = input.get("pattern").and_then(|v| v.as_str()) {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        return Some(format!("grep '{}' {}", pat, path));
    }
    // Glob
    if let Some(pat) = input.get("pattern").and_then(|v| v.as_str()) {
        return Some(format!("glob '{}'", pat));
    }

    // Fallback: compact JSON of the input
    Some(serde_json::to_string(input).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_project_dir() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let projects = tmp.path().join("projects").join("test-project");
        fs::create_dir_all(&projects).unwrap();
        // sessions-index.json
        fs::write(
            projects.join("sessions-index.json"),
            r#"{
                "version": 1,
                "entries": [
                    {
                        "sessionId": "abc-123",
                        "fullPath": "/tmp/abc-123.jsonl",
                        "fileMtime": 1770130128425,
                        "firstPrompt": "hello",
                        "summary": "Test session",
                        "messageCount": 4,
                        "created": "2026-01-05T01:04:45.879Z",
                        "modified": "2026-01-05T01:05:06.079Z",
                        "gitBranch": "main",
                        "projectPath": "/tmp/project",
                        "isSidechain": false
                    }
                ]
            }"#,
        )
        .unwrap();
        tmp
    }

    #[test]
    fn test_parse_sessions_index() {
        let tmp = setup_project_dir();
        let project_dir = tmp.path().join("projects").join("test-project");
        let ingester = ClaudeIngester::with_base_path(tmp.path());
        let sessions = ingester.parse_sessions_index(&project_dir).unwrap();

        assert_eq!(sessions.len(), 1);
        let s = &sessions[0];
        assert_eq!(s.id, "abc-123");
        assert_eq!(s.agent, "claude_code");
        assert_eq!(s.summary.as_deref(), Some("Test session"));
        assert_eq!(s.message_count, Some(4));
        assert!(s.started_at.is_some());
        assert!(s.ended_at.is_some());
        assert_eq!(s.project_id.as_deref(), Some("test-project"));
    }

    #[test]
    fn test_parse_session_transcript() {
        let tmp = TempDir::new().unwrap();
        let jsonl_path = tmp.path().join("session.jsonl");
        let lines = [
            // file-history-snapshot (should be skipped for message count)
            r#"{"type":"file-history-snapshot","messageId":"m1","snapshot":{},"isSnapshotUpdate":false}"#,
            // user message
            r#"{"parentUuid":null,"isSidechain":false,"type":"user","message":{"role":"user","content":"run ls"},"uuid":"u1","timestamp":"2026-03-12T23:26:44.124Z"}"#,
            // assistant with tool_use
            r#"{"parentUuid":"u1","isSidechain":false,"message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls -la","description":"list files"}}],"stop_reason":"tool_use","usage":{"input_tokens":100,"output_tokens":50}},"type":"assistant","uuid":"u2","timestamp":"2026-03-12T23:26:46.170Z","sessionId":"s1"}"#,
            // assistant text
            r#"{"parentUuid":"u2","message":{"role":"assistant","content":[{"type":"text","text":"Here are the files"}],"usage":{"input_tokens":200,"output_tokens":80}},"type":"assistant","uuid":"u3","timestamp":"2026-03-12T23:26:48.000Z"}"#,
            // malformed line (should be skipped)
            r#"NOT VALID JSON"#,
        ];
        fs::write(&jsonl_path, lines.join("\n")).unwrap();

        let ingester = ClaudeIngester::with_base_path(tmp.path());
        let stats = ingester
            .parse_session_transcript(&jsonl_path, "s1")
            .unwrap();

        // 1 user + 2 assistant = 3 messages (file-history-snapshot skipped)
        assert_eq!(stats.message_count, 3);
        assert_eq!(stats.tool_calls.len(), 1);
        assert_eq!(stats.tool_calls[0].tool_name.as_deref(), Some("Bash"));
        assert_eq!(stats.tool_calls[0].command.as_deref(), Some("ls -la"));
        assert_eq!(stats.tool_calls[0].session_id.as_deref(), Some("s1"));
        assert!(stats.tool_calls[0].timestamp.is_some());
        // Tokens: 100+200 input, 50+80 output
        assert_eq!(stats.total_input_tokens, 300);
        assert_eq!(stats.total_output_tokens, 130);
    }

    #[test]
    fn test_parse_memory() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("projects").join("my-project");
        let memory_dir = project_dir.join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(
            memory_dir.join("MEMORY.md"),
            "# Project Memory\n\n- Key fact 1\n- Key fact 2\n",
        )
        .unwrap();
        // Empty file should be skipped
        fs::write(memory_dir.join("empty.md"), "  ").unwrap();
        // Non-md file should be skipped
        fs::write(memory_dir.join("notes.txt"), "should be ignored").unwrap();

        let ingester = ClaudeIngester::with_base_path(tmp.path());
        let memories = ingester.parse_memory(&project_dir).unwrap();

        assert_eq!(memories.len(), 1);
        assert!(memories[0].content.contains("Key fact 1"));
        assert_eq!(
            memories[0].memory_type.as_deref(),
            Some("claude_code_memory")
        );
        assert_eq!(memories[0].project_id.as_deref(), Some("my-project"));
        assert_eq!(memories[0].confidence, 1.0);
    }

    #[test]
    fn test_discover_projects() {
        let tmp = TempDir::new().unwrap();
        let projects = tmp.path().join("projects");
        fs::create_dir_all(projects.join("project-a")).unwrap();
        fs::create_dir_all(projects.join("project-b")).unwrap();
        // A file (not a dir) should be ignored
        fs::write(projects.join("stray-file"), "ignored").unwrap();

        let ingester = ClaudeIngester::with_base_path(tmp.path());
        let dirs = ingester.discover_projects();

        assert_eq!(dirs.len(), 2);
        let names: Vec<&str> = dirs
            .iter()
            .map(|d| d.file_name().unwrap().to_str().unwrap())
            .collect();
        assert!(names.contains(&"project-a"));
        assert!(names.contains(&"project-b"));
    }

    #[test]
    fn test_ingest_all_end_to_end() {
        let tmp = setup_project_dir();
        let project_dir = tmp.path().join("projects").join("test-project");

        // Write a transcript for the session in the index
        let transcript = r#"{"type":"user","message":{"role":"user","content":"hi"},"uuid":"u1","timestamp":"2026-01-05T01:04:46.000Z"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Write","input":{"file_path":"/tmp/foo.rs","content":"fn main() {}"}}],"stop_reason":"tool_use","usage":{"input_tokens":50,"output_tokens":25}},"uuid":"u2","timestamp":"2026-01-05T01:04:47.000Z"}"#;
        fs::write(project_dir.join("abc-123.jsonl"), transcript).unwrap();

        // Write a memory file
        let mem_dir = project_dir.join("memory");
        fs::create_dir_all(&mem_dir).unwrap();
        fs::write(mem_dir.join("MEMORY.md"), "# Test\n- fact").unwrap();

        let ingester = ClaudeIngester::with_base_path(tmp.path());
        let result = ingester.ingest_all().unwrap();

        assert_eq!(result.sessions_count, 1);
        assert_eq!(result.tool_calls_count, 1);
        assert_eq!(result.memories_count, 1);

        // Session should have been enriched from transcript
        let session = &result.sessions[0];
        assert_eq!(session.tool_call_count, Some(1));
        assert_eq!(session.total_tokens, Some(75));
        assert!(session.files_changed.contains(&"/tmp/foo.rs".to_string()));
    }

    #[test]
    fn test_parse_iso8601_variants() {
        // Standard RFC 3339 with fractional seconds
        let dt = parse_iso8601("2026-03-12T23:26:44.124Z");
        assert!(dt.is_some());
        assert_eq!(dt.unwrap().year(), 2026);

        // Without fractional seconds
        let dt2 = parse_iso8601("2026-01-05T01:04:45Z");
        assert!(dt2.is_some());

        // Invalid
        let dt3 = parse_iso8601("not-a-date");
        assert!(dt3.is_none());
    }

    // Bring NaiveDate into scope for the year assertion
    use chrono::Datelike;
}
