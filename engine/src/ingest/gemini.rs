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
    #[serde(default)]
    content: Vec<RawContent>,
}

/// Content block within a message.
///
/// Most content blocks carry a `text` field. Tool-use blocks carry a
/// `function_call` object instead.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawContent {
    text: Option<String>,
    function_call: Option<RawFunctionCall>,
}

/// A function/tool call embedded in a model response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawFunctionCall {
    name: Option<String>,
    #[serde(default)]
    args: serde_json::Value,
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

        // Count tool calls (function_call content blocks).
        let tool_call_count: i32 = raw
            .messages
            .iter()
            .flat_map(|m| &m.content)
            .filter(|c| c.function_call.is_some())
            .count() as i32;

        // Resolve project id from the hash via the project map.
        let project_id = raw.project_hash.as_deref().and_then(|hash| {
            project_map
                .get(hash)
                .cloned()
                .or_else(|| Some(hash.to_string()))
        });

        // Build a short summary from the first user message (truncated).
        let summary = raw
            .messages
            .iter()
            .find(|m| m.msg_type.as_deref() == Some("user"))
            .and_then(|m| m.content.first())
            .and_then(|c| c.text.as_deref())
            .map(|t| {
                let trimmed = t.trim();
                if trimmed.len() > 200 {
                    format!("{}...", &trimmed[..200])
                } else {
                    trimmed.to_string()
                }
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
            total_tokens: None, // Gemini CLI doesn't expose token counts in session files
            files_changed: Vec::new(), // Not tracked in session JSON
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
            let timestamp = msg.timestamp.as_deref().and_then(parse_iso_timestamp);
            for content in &msg.content {
                if let Some(ref fc) = content.function_call {
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
                        success: None, // Not determinable from session JSON
                        error_message: None,
                        duration_ms: None,
                        timestamp,
                    });
                }
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
    fn setup_temp_gemini(dir: &Path) {
        let hash = "abc123hash";
        let chats_dir = dir.join("tmp").join(hash).join("chats");
        fs::create_dir_all(&chats_dir).unwrap();

        let session_json = r#"{
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
            session_json,
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
    fn test_parse_session() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_gemini(&base);

        let ingester = GeminiIngester::with_base_path(&base);
        let project_map = ingester.load_project_map().unwrap();
        let sessions = ingester.discover_sessions();

        assert_eq!(sessions.len(), 1);

        let session = ingester.parse_session(&sessions[0], &project_map).unwrap();
        assert_eq!(session.id, "s-001");
        assert_eq!(session.agent, "gemini");
        assert_eq!(session.message_count, Some(3));
        assert_eq!(session.tool_call_count, Some(1));
        assert_eq!(session.duration_minutes, Some(10));
        assert!(session.started_at.is_some());
        assert!(session.ended_at.is_some());
        assert!(session.summary.unwrap().contains("auth module"));
    }

    #[test]
    fn test_parse_tool_calls() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();
        setup_temp_gemini(&base);

        let ingester = GeminiIngester::with_base_path(&base);
        let sessions = ingester.discover_sessions();

        let calls = ingester.parse_session_tool_calls(&sessions[0]).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_name.as_deref(), Some("read_file"));
        assert_eq!(calls[0].session_id.as_deref(), Some("s-001"));
        assert!(calls[0].command.as_ref().unwrap().contains("src/auth.rs"));
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

        assert_eq!(result.sessions_found, 1);
        assert_eq!(result.tool_calls_found, 1);
        assert_eq!(result.memories_found, 1);
        assert_eq!(sessions.len(), 1);
        assert_eq!(tool_calls.len(), 1);
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
