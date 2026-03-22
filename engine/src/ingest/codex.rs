//! Codex CLI ingestion parser.
//!
//! Reads Codex CLI artifacts from `~/.codex/` and converts them into
//! Remembrant domain structs ([`Session`], [`ToolCall`], [`Memory`]) for
//! storage in DuckDB.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
#[cfg(test)]
use chrono::Timelike;
use serde::Deserialize;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::store::duckdb::{Memory, Session, ToolCall};

// ---------------------------------------------------------------------------
// Serde structs for Codex JSONL format
// ---------------------------------------------------------------------------

/// A single line in a rollout JSONL file.
#[derive(Debug, Deserialize)]
struct RolloutLine {
    timestamp: Option<String>,
    #[serde(rename = "type")]
    line_type: String,
    payload: serde_json::Value,
}

/// The `payload` of a `session_meta` line.
#[derive(Debug, Deserialize)]
struct SessionMetaPayload {
    id: String,
    timestamp: Option<String>,
    cwd: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    originator: Option<String>,
    cli_version: Option<String>,
    #[serde(default)]
    source: Option<String>,
    model_provider: Option<String>,
    git: Option<GitInfo>,
}

#[derive(Debug, Deserialize)]
struct GitInfo {
    #[serde(default)]
    #[allow(dead_code)]
    commit_hash: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    branch: Option<String>,
    #[serde(default)]
    repository_url: Option<String>,
}

/// The `payload` of a `response_item` line (only the fields we need).
#[derive(Debug, Deserialize)]
struct ResponseItemPayload {
    #[serde(rename = "type")]
    item_type: Option<String>,
    #[allow(dead_code)]
    role: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    content: Vec<ContentBlock>,
    /// For function_call items.
    name: Option<String>,
    /// For function_call items.
    arguments: Option<String>,
    /// For function_call_output items.
    output: Option<String>,
    /// Call ID linking function_call to function_call_output.
    call_id: Option<String>,
    /// Status for function_call_output (e.g. "completed", "failed").
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    block_type: Option<String>,
    #[allow(dead_code)]
    text: Option<String>,
}

/// A single line in `history.jsonl`.
#[derive(Debug, Clone, Deserialize)]
pub struct HistoryEntry {
    pub session_id: String,
    pub ts: i64,
    pub text: String,
}

// ---------------------------------------------------------------------------
// IngestResult
// ---------------------------------------------------------------------------

/// Summary of what the ingestion discovered and parsed.
#[derive(Debug, Default)]
pub struct IngestResult {
    pub sessions: Vec<Session>,
    pub tool_calls: Vec<ToolCall>,
    pub memories: Vec<Memory>,
    pub history_entries: Vec<HistoryEntry>,
    pub session_files_found: usize,
    pub parse_errors: usize,
}

// ---------------------------------------------------------------------------
// CodexIngester
// ---------------------------------------------------------------------------

/// Discovers and parses Codex CLI artifacts from the filesystem.
pub struct CodexIngester {
    base_path: PathBuf,
}

impl CodexIngester {
    /// Create a new ingester pointing at `~/.codex`.
    ///
    /// Returns `None` if the home directory cannot be resolved or `~/.codex`
    /// does not exist.
    pub fn new() -> Option<Self> {
        let home = std::env::var_os("HOME").map(PathBuf::from).or_else(|| {
            #[allow(deprecated)]
            std::env::home_dir()
        })?;
        let base = home.join(".codex");
        if base.is_dir() {
            Some(Self { base_path: base })
        } else {
            debug!("~/.codex directory not found");
            None
        }
    }

    /// Create an ingester with an explicit base path (useful for tests).
    pub fn with_base_path(base_path: PathBuf) -> Self {
        Self { base_path }
    }

    /// Return the base path (`~/.codex`).
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    // -----------------------------------------------------------------------
    // Discovery
    // -----------------------------------------------------------------------

    /// Discover all rollout JSONL session files under `sessions/`.
    ///
    /// Layout: `sessions/YYYY/MM/DD/rollout-*.jsonl`
    pub fn discover_sessions(&self) -> Vec<PathBuf> {
        let sessions_dir = self.base_path.join("sessions");
        if !sessions_dir.is_dir() {
            return Vec::new();
        }
        let mut paths = Vec::new();
        Self::walk_jsonl_files(&sessions_dir, &mut paths);
        paths.sort();
        paths
    }

    /// Recursively collect `*.jsonl` files under `dir`.
    fn walk_jsonl_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::walk_jsonl_files(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "jsonl") {
                out.push(path);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Session parsing
    // -----------------------------------------------------------------------

    /// Parse a single rollout JSONL file into a [`Session`].
    pub fn parse_session(&self, jsonl_path: &Path) -> Result<Session> {
        let content = std::fs::read_to_string(jsonl_path)
            .with_context(|| format!("failed to read session file: {}", jsonl_path.display()))?;

        let mut session_id: Option<String> = None;
        let mut started_at: Option<NaiveDateTime> = None;
        let mut ended_at: Option<NaiveDateTime> = None;
        let mut project_id: Option<String> = None;
        let mut message_count: i32 = 0;
        let mut tool_call_count: i32 = 0;
        let mut summary_parts: Vec<String> = Vec::new();
        let mut last_timestamp: Option<NaiveDateTime> = None;

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parsed: RolloutLine = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        "skipping malformed line {} in {}: {e}",
                        line_num + 1,
                        jsonl_path.display()
                    );
                    continue;
                }
            };

            // Track last timestamp for ended_at estimation.
            if let Some(ref ts_str) = parsed.timestamp
                && let Some(ts) = parse_iso_timestamp(ts_str)
            {
                last_timestamp = Some(ts);
            }

            match parsed.line_type.as_str() {
                "session_meta" => {
                    if let Ok(meta) = serde_json::from_value::<SessionMetaPayload>(parsed.payload) {
                        session_id = Some(meta.id.clone());
                        started_at = meta.timestamp.as_deref().and_then(parse_iso_timestamp);
                        project_id = derive_project_id(&meta);

                        // Build a summary snippet from metadata.
                        let mut parts = Vec::new();
                        if let Some(ref src) = meta.source {
                            parts.push(format!("source={src}"));
                        }
                        if let Some(ref ver) = meta.cli_version {
                            parts.push(format!("v{ver}"));
                        }
                        if let Some(ref provider) = meta.model_provider {
                            parts.push(format!("provider={provider}"));
                        }
                        if !parts.is_empty() {
                            summary_parts.push(parts.join(", "));
                        }
                    }
                }
                "response_item" => {
                    if let Ok(item) = serde_json::from_value::<ResponseItemPayload>(parsed.payload)
                    {
                        match item.item_type.as_deref() {
                            Some("message") => {
                                message_count += 1;
                            }
                            Some("function_call") => {
                                tool_call_count += 1;
                            }
                            Some("function_call_output") => {
                                // Counted under the function_call that triggered it.
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    // Other event types (e.g. "error", "usage") — skip for now.
                }
            }
        }

        // Estimate ended_at from the last line's timestamp.
        if ended_at.is_none() {
            ended_at = last_timestamp;
        }

        let duration_minutes = match (started_at, ended_at) {
            (Some(s), Some(e)) => {
                let dur = e.signed_duration_since(s);
                Some(dur.num_minutes() as i32)
            }
            _ => None,
        };

        let id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let summary = if summary_parts.is_empty() {
            None
        } else {
            Some(summary_parts.join("; "))
        };

        Ok(Session {
            id,
            project_id,
            agent: "codex".to_string(),
            started_at,
            ended_at,
            duration_minutes,
            message_count: Some(message_count),
            tool_call_count: Some(tool_call_count),
            total_tokens: None,        // Not available in JSONL rollout files.
            files_changed: Vec::new(), // Would require deeper analysis of tool outputs.
            summary,
        })
    }

    // -----------------------------------------------------------------------
    // Tool call extraction
    // -----------------------------------------------------------------------

    /// Extract [`ToolCall`] records from a rollout JSONL file.
    pub fn parse_session_tool_calls(&self, jsonl_path: &Path) -> Result<Vec<ToolCall>> {
        let content = std::fs::read_to_string(jsonl_path).with_context(|| {
            format!(
                "failed to read session file for tool calls: {}",
                jsonl_path.display()
            )
        })?;

        // First pass: collect function_call entries keyed by call_id.
        // Second pass: match function_call_output to them.
        // We do a single pass and build up state.

        let mut session_id: Option<String> = None;
        let mut calls: Vec<ToolCall> = Vec::new();
        // Map call_id -> index in `calls` for matching outputs.
        let mut call_id_index: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parsed: RolloutLine = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        "skipping malformed line {} in {}: {e}",
                        line_num + 1,
                        jsonl_path.display()
                    );
                    continue;
                }
            };

            match parsed.line_type.as_str() {
                "session_meta" => {
                    if let Ok(meta) = serde_json::from_value::<SessionMetaPayload>(parsed.payload) {
                        session_id = Some(meta.id);
                    }
                }
                "response_item" => {
                    if let Ok(item) = serde_json::from_value::<ResponseItemPayload>(parsed.payload)
                    {
                        let timestamp = parsed.timestamp.as_deref().and_then(parse_iso_timestamp);

                        match item.item_type.as_deref() {
                            Some("function_call") => {
                                let tool_name = item.name.clone().unwrap_or_default();
                                let command = item.arguments.clone().or({
                                    // If arguments is not a string field,
                                    // it might be embedded elsewhere.
                                    None
                                });

                                let tc = ToolCall {
                                    id: Uuid::new_v4().to_string(),
                                    session_id: session_id.clone(),
                                    tool_name: Some(tool_name),
                                    command,
                                    success: None,
                                    error_message: None,
                                    duration_ms: None,
                                    timestamp,
                                };
                                let idx = calls.len();
                                if let Some(ref cid) = item.call_id {
                                    call_id_index.insert(cid.clone(), idx);
                                }
                                calls.push(tc);
                            }
                            Some("function_call_output") => {
                                // Try to match back to the originating call.
                                if let Some(ref cid) = item.call_id
                                    && let Some(&idx) = call_id_index.get(cid)
                                {
                                    let tc = &mut calls[idx];
                                    let success = item
                                        .status
                                        .as_deref()
                                        .map(|s| s == "completed" || s == "success");
                                    tc.success = success;
                                    if let Some(ref status) = item.status
                                        && (status == "failed" || status == "error")
                                    {
                                        tc.error_message = item.output.clone();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(calls)
    }

    // -----------------------------------------------------------------------
    // History parsing
    // -----------------------------------------------------------------------

    /// Parse `history.jsonl` into a list of [`HistoryEntry`] records.
    pub fn parse_history(&self) -> Result<Vec<HistoryEntry>> {
        let history_path = self.base_path.join("history.jsonl");
        if !history_path.is_file() {
            debug!("no history.jsonl found at {}", history_path.display());
            return Ok(Vec::new());
        }
        let content = std::fs::read_to_string(&history_path)
            .with_context(|| format!("failed to read history file: {}", history_path.display()))?;

        let mut entries = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<HistoryEntry>(line) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    warn!("skipping malformed history line {}: {e}", line_num + 1);
                }
            }
        }
        Ok(entries)
    }

    // -----------------------------------------------------------------------
    // Rules -> Memories
    // -----------------------------------------------------------------------

    /// Read rule files from `rules/` and convert them into [`Memory`] records.
    pub fn parse_rules(&self) -> Result<Vec<Memory>> {
        let rules_dir = self.base_path.join("rules");
        if !rules_dir.is_dir() {
            debug!("no rules directory found");
            return Ok(Vec::new());
        }

        let mut memories = Vec::new();
        let entries = std::fs::read_dir(&rules_dir)
            .with_context(|| format!("failed to read rules directory: {}", rules_dir.display()))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    warn!("failed to read rule file {}: {e}", path.display());
                    continue;
                }
            };
            if content.trim().is_empty() {
                continue;
            }

            let file_name = path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default();

            let now = chrono::Utc::now().naive_utc();
            memories.push(Memory {
                id: Uuid::new_v4().to_string(),
                project_id: None,
                content,
                memory_type: Some("rule".to_string()),
                source_session_id: None,
                confidence: 1.0,
                access_count: 0,
                created_at: Some(now),
                updated_at: Some(now),
                valid_until: None,
            });

            debug!("parsed rule file: {file_name}");
        }

        Ok(memories)
    }

    // -----------------------------------------------------------------------
    // Full ingestion
    // -----------------------------------------------------------------------

    /// Discover and parse all Codex CLI artifacts, returning an [`IngestResult`].
    pub fn ingest_all(&self) -> Result<IngestResult> {
        let mut result = IngestResult::default();

        // 1. Discover and parse sessions.
        let session_paths = self.discover_sessions();
        result.session_files_found = session_paths.len();
        debug!("discovered {} session files", session_paths.len());

        for path in &session_paths {
            match self.parse_session(path) {
                Ok(session) => {
                    debug!("parsed session: {}", session.id);
                    result.sessions.push(session);
                }
                Err(e) => {
                    warn!("failed to parse session {}: {e}", path.display());
                    result.parse_errors += 1;
                }
            }

            match self.parse_session_tool_calls(path) {
                Ok(calls) => {
                    debug!(
                        "extracted {} tool calls from {}",
                        calls.len(),
                        path.display()
                    );
                    result.tool_calls.extend(calls);
                }
                Err(e) => {
                    warn!("failed to parse tool calls from {}: {e}", path.display());
                    result.parse_errors += 1;
                }
            }
        }

        // 2. Parse history.
        match self.parse_history() {
            Ok(entries) => {
                debug!("parsed {} history entries", entries.len());
                result.history_entries = entries;
            }
            Err(e) => {
                warn!("failed to parse history: {e}");
                result.parse_errors += 1;
            }
        }

        // 3. Parse rules into memories.
        match self.parse_rules() {
            Ok(mems) => {
                debug!("parsed {} rule memories", mems.len());
                result.memories = mems;
            }
            Err(e) => {
                warn!("failed to parse rules: {e}");
                result.parse_errors += 1;
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse an ISO 8601 timestamp string (with or without trailing `Z`) into
/// a [`NaiveDateTime`].
///
/// Handles formats like:
/// - `2026-03-05T05:00:04.837Z`
/// - `2026-03-05T05:00:04.618Z`
/// - `2026-03-05T05:00:04Z`
fn parse_iso_timestamp(s: &str) -> Option<NaiveDateTime> {
    // Strip trailing 'Z' and try parsing.
    let s = s.trim_end_matches('Z');

    // Try with fractional seconds.
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(dt);
    }
    // Try without fractional seconds.
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt);
    }
    None
}

/// Parse a unix timestamp (seconds since epoch) into a [`NaiveDateTime`].
#[allow(dead_code)]
fn parse_unix_timestamp(ts: i64) -> Option<NaiveDateTime> {
    chrono::DateTime::from_timestamp(ts, 0).map(|dt| dt.naive_utc())
}

/// Derive a project identifier from session metadata.
///
/// Prefers the git repository URL, then falls back to the working directory.
fn derive_project_id(meta: &SessionMetaPayload) -> Option<String> {
    // Try git repo URL first — extract "owner/repo" from GitHub URLs.
    if let Some(ref git) = meta.git
        && let Some(ref url) = git.repository_url
    {
        if let Some(slug) = extract_repo_slug(url) {
            return Some(slug);
        }
        // Fall back to the full URL if we can't extract a slug.
        if !url.is_empty() {
            return Some(url.clone());
        }
    }
    // Fall back to cwd — use last path component as project name.
    if let Some(ref cwd) = meta.cwd {
        let p = Path::new(cwd);
        if let Some(name) = p.file_name() {
            return Some(name.to_string_lossy().to_string());
        }
    }
    None
}

/// Extract `"owner/repo"` from a GitHub URL.
///
/// Handles `https://github.com/owner/repo.git` and similar forms.
fn extract_repo_slug(url: &str) -> Option<String> {
    let url = url.trim_end_matches(".git").trim_end_matches('/');
    // Look for github.com/owner/repo pattern.
    if let Some(idx) = url.find("github.com/") {
        let after = &url[idx + "github.com/".len()..];
        let parts: Vec<&str> = after.splitn(3, '/').collect();
        if parts.len() >= 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            return Some(format!("{}/{}", parts[0], parts[1]));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: create a temp directory with Codex-like structure.
    fn setup_temp_codex(
        dir: &Path,
        session_jsonl: &str,
        history_jsonl: Option<&str>,
        rules: Option<Vec<(&str, &str)>>,
    ) {
        // Create sessions dir.
        let session_dir = dir.join("sessions").join("2026").join("03").join("05");
        fs::create_dir_all(&session_dir).unwrap();
        fs::write(
            session_dir.join("rollout-2026-03-05T00-00-04-test-id.jsonl"),
            session_jsonl,
        )
        .unwrap();

        // History.
        if let Some(h) = history_jsonl {
            fs::write(dir.join("history.jsonl"), h).unwrap();
        }

        // Rules.
        if let Some(rule_files) = rules {
            let rules_dir = dir.join("rules");
            fs::create_dir_all(&rules_dir).unwrap();
            for (name, content) in rule_files {
                fs::write(rules_dir.join(name), content).unwrap();
            }
        }
    }

    fn sample_session_jsonl() -> String {
        let lines = vec![
            r#"{"timestamp":"2026-03-05T05:00:04.837Z","type":"session_meta","payload":{"id":"019cbc5e-test-id","timestamp":"2026-03-05T05:00:04.618Z","cwd":"/home/user/myproject","originator":"codex_exec","cli_version":"0.107.0","source":"exec","model_provider":"openai","git":{"commit_hash":"abc123","branch":"main","repository_url":"https://github.com/alice/myrepo.git"}}}"#,
            r#"{"timestamp":"2026-03-05T05:00:05.000Z","type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"Fix the bug"}]}}"#,
            r#"{"timestamp":"2026-03-05T05:00:06.000Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will fix it."}]}}"#,
            r#"{"timestamp":"2026-03-05T05:00:07.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"call_1","arguments":"{\"cmd\":\"ls -la\"}"}}"#,
            r#"{"timestamp":"2026-03-05T05:00:08.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_1","output":"total 42\ndrwxr-xr-x ...","status":"completed"}}"#,
            r#"{"timestamp":"2026-03-05T05:00:10.000Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done!"}]}}"#,
        ];
        lines.join("\n")
    }

    #[test]
    fn test_parse_session_extracts_metadata_and_counts() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_codex(&base, &sample_session_jsonl(), None, None);

        let ingester = CodexIngester::with_base_path(base);
        let paths = ingester.discover_sessions();
        assert_eq!(paths.len(), 1);

        let session = ingester.parse_session(&paths[0]).unwrap();
        assert_eq!(session.id, "019cbc5e-test-id");
        assert_eq!(session.agent, "codex");
        assert_eq!(session.project_id.as_deref(), Some("alice/myrepo"));
        assert_eq!(session.message_count, Some(3));
        assert_eq!(session.tool_call_count, Some(1));
        assert!(session.started_at.is_some());
        assert!(session.ended_at.is_some());
        // Duration should be ~0 minutes (6 seconds of data).
        assert_eq!(session.duration_minutes, Some(0));
    }

    #[test]
    fn test_parse_tool_calls_with_output_matching() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_codex(&base, &sample_session_jsonl(), None, None);

        let ingester = CodexIngester::with_base_path(base);
        let paths = ingester.discover_sessions();
        let calls = ingester.parse_session_tool_calls(&paths[0]).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_name.as_deref(), Some("shell"));
        assert_eq!(calls[0].command.as_deref(), Some("{\"cmd\":\"ls -la\"}"));
        assert_eq!(calls[0].success, Some(true));
        assert!(calls[0].session_id.is_some());
        assert!(calls[0].timestamp.is_some());
    }

    #[test]
    fn test_parse_history_and_rules() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        let history = r#"{"session_id":"sess-1","ts":1760308619,"text":"add hyperparameter tuning"}
{"session_id":"sess-2","ts":1760308700,"text":"fix compile error"}"#;
        let rules = vec![("default.rules", "Always write tests for new code.")];

        setup_temp_codex(&base, &sample_session_jsonl(), Some(history), Some(rules));

        let ingester = CodexIngester::with_base_path(base);

        let entries = ingester.parse_history().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].session_id, "sess-1");
        assert!(entries[0].text.contains("hyperparameter"));

        let memories = ingester.parse_rules().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].memory_type.as_deref(), Some("rule"));
        assert!(memories[0].content.contains("Always write tests"));
        assert_eq!(memories[0].confidence, 1.0);
    }

    #[test]
    fn test_ingest_all_integrates_everything() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        let history = r#"{"session_id":"sess-1","ts":1760308619,"text":"do stuff"}"#;
        let rules = vec![("style.rules", "Use 4-space indentation.")];

        setup_temp_codex(&base, &sample_session_jsonl(), Some(history), Some(rules));

        let ingester = CodexIngester::with_base_path(base);
        let result = ingester.ingest_all().unwrap();

        assert_eq!(result.session_files_found, 1);
        assert_eq!(result.sessions.len(), 1);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.history_entries.len(), 1);
        assert_eq!(result.memories.len(), 1);
        assert_eq!(result.parse_errors, 0);
    }

    #[test]
    fn test_malformed_lines_skipped_gracefully() {
        let jsonl = format!(
            "{}\n{}\n{}",
            r#"{"timestamp":"2026-03-05T05:00:04.837Z","type":"session_meta","payload":{"id":"test-graceful","timestamp":"2026-03-05T05:00:04.618Z","cwd":"/tmp"}}"#,
            "this is not valid json",
            r#"{"timestamp":"2026-03-05T05:00:05.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}"#,
        );

        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_codex(&base, &jsonl, None, None);

        let ingester = CodexIngester::with_base_path(base);
        let paths = ingester.discover_sessions();
        let session = ingester.parse_session(&paths[0]).unwrap();

        // Should still parse successfully, just skip the bad line.
        assert_eq!(session.id, "test-graceful");
        assert_eq!(session.message_count, Some(1));
    }

    #[test]
    fn test_parse_iso_timestamp_variants() {
        let ts1 = parse_iso_timestamp("2026-03-05T05:00:04.837Z");
        assert!(ts1.is_some());
        let dt = ts1.unwrap();
        // Verify the parsed date components rather than a hardcoded epoch value.
        assert_eq!(
            dt.date(),
            chrono::NaiveDate::from_ymd_opt(2026, 3, 5).unwrap()
        );
        assert_eq!(dt.time().hour(), 5);
        assert_eq!(dt.time().minute(), 0);
        assert_eq!(dt.time().second(), 4);

        let ts2 = parse_iso_timestamp("2026-03-05T05:00:04Z");
        assert!(ts2.is_some());

        let ts3 = parse_iso_timestamp("not a timestamp");
        assert!(ts3.is_none());
    }

    #[test]
    fn test_parse_unix_timestamp() {
        let dt = parse_unix_timestamp(1760308619);
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.and_utc().timestamp(), 1760308619);
    }

    #[test]
    fn test_extract_repo_slug() {
        assert_eq!(
            extract_repo_slug("https://github.com/alice/myrepo.git"),
            Some("alice/myrepo".to_string())
        );
        assert_eq!(
            extract_repo_slug("https://github.com/org/repo"),
            Some("org/repo".to_string())
        );
        assert_eq!(extract_repo_slug("https://gitlab.com/x/y"), None);
        assert_eq!(extract_repo_slug(""), None);
    }

    #[test]
    fn test_derive_project_id_from_cwd_fallback() {
        let meta = SessionMetaPayload {
            id: "test".into(),
            timestamp: None,
            cwd: Some("/home/user/cool-project".into()),
            originator: None,
            cli_version: None,
            source: None,
            model_provider: None,
            git: None,
        };
        assert_eq!(derive_project_id(&meta), Some("cool-project".to_string()));
    }
}
