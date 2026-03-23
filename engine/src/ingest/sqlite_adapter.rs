//! Generic SQLite adapter for agents that store data in SQLite databases.
//!
//! Supports Goose, OpenCode, Cursor, OpenClaw, and any future agent that
//! uses SQLite as its storage backend. Column mappings are provided via
//! [`DynamicAgentConfig`] so new agents can be added in config alone.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use tracing::{debug, warn};

use super::adapter::{
    expand_tilde, AgentAdapter, AgentMeta, DynamicAgentConfig, IngestOutput, SqliteMapping,
};
use crate::store::duckdb::{Session, ToolCall};

// ---------------------------------------------------------------------------
// GenericSqliteAdapter
// ---------------------------------------------------------------------------

/// A generic adapter that reads from any SQLite database using configurable
/// table/column mappings.
pub struct GenericSqliteAdapter {
    meta: AgentMeta,
    base_path: PathBuf,
    mapping: SqliteMapping,
}

impl GenericSqliteAdapter {
    /// Create from a [`DynamicAgentConfig`].
    ///
    /// Returns `None` if the config doesn't have SQLite mapping.
    pub fn from_config(config: &DynamicAgentConfig) -> Option<Self> {
        let mapping = config.sqlite.clone()?;
        Some(Self {
            meta: AgentMeta {
                id: config.id.clone(),
                display_name: config.display_name.clone(),
                storage_format: "sqlite".into(),
                default_path: config.path.clone(),
            },
            base_path: expand_tilde(&config.path),
            mapping,
        })
    }

    /// Create directly with explicit parameters (useful for testing).
    pub fn new(meta: AgentMeta, base_path: PathBuf, mapping: SqliteMapping) -> Self {
        Self {
            meta,
            base_path,
            mapping,
        }
    }

    /// Resolve the full path to the SQLite database file.
    fn db_path(&self) -> PathBuf {
        self.base_path.join(&self.mapping.db_file)
    }

    /// Parse a timestamp string according to the configured format.
    fn parse_timestamp(&self, value: &rusqlite::types::Value) -> Option<NaiveDateTime> {
        match value {
            rusqlite::types::Value::Text(s) => parse_timestamp_str(
                s,
                &self.mapping.session_columns.timestamp_format,
            ),
            rusqlite::types::Value::Integer(ts) => {
                match self.mapping.session_columns.timestamp_format.as_str() {
                    "unix_millis" => {
                        chrono::DateTime::from_timestamp_millis(*ts).map(|dt| dt.naive_utc())
                    }
                    _ => {
                        // Default: unix seconds
                        chrono::DateTime::from_timestamp(*ts, 0).map(|dt| dt.naive_utc())
                    }
                }
            }
            rusqlite::types::Value::Real(f) => {
                chrono::DateTime::from_timestamp(*f as i64, 0).map(|dt| dt.naive_utc())
            }
            _ => None,
        }
    }

    /// Read sessions from the SQLite database.
    fn read_sessions(&self, conn: &rusqlite::Connection) -> Result<Vec<Session>> {
        let cols = &self.mapping.session_columns;
        let agent_id = &self.meta.id;

        // Build SELECT columns dynamically based on what's mapped.
        let mut select_cols = vec![cols.id.clone()];
        if let Some(ref c) = cols.project_id {
            select_cols.push(c.clone());
        }
        if let Some(ref c) = cols.started_at {
            select_cols.push(c.clone());
        }
        if let Some(ref c) = cols.ended_at {
            select_cols.push(c.clone());
        }
        if let Some(ref c) = cols.message_count {
            select_cols.push(c.clone());
        }
        if let Some(ref c) = cols.summary {
            select_cols.push(c.clone());
        }

        let sql = format!(
            "SELECT {} FROM {}",
            select_cols.join(", "),
            self.mapping.sessions_table
        );

        debug!("[{}] executing: {sql}", agent_id);

        let mut stmt = conn.prepare(&sql).context("preparing session query")?;

        let sessions = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;

                let mut col_idx = 1usize;
                let project_id: Option<String> = if cols.project_id.is_some() {
                    let v = row.get(col_idx).ok();
                    col_idx += 1;
                    v
                } else {
                    None
                };

                let started_at_val: Option<rusqlite::types::Value> = if cols.started_at.is_some() {
                    let v = row.get(col_idx).ok();
                    col_idx += 1;
                    v
                } else {
                    None
                };

                let ended_at_val: Option<rusqlite::types::Value> = if cols.ended_at.is_some() {
                    let v = row.get(col_idx).ok();
                    col_idx += 1;
                    v
                } else {
                    None
                };

                let message_count: Option<i32> = if cols.message_count.is_some() {
                    let v = row.get(col_idx).ok();
                    col_idx += 1;
                    v
                } else {
                    None
                };

                let summary: Option<String> = if cols.summary.is_some() {
                    let _v = row.get(col_idx).ok();
                    _v
                } else {
                    None
                };

                Ok((id, project_id, started_at_val, ended_at_val, message_count, summary))
            })
            .context("querying sessions")?;

        let mut result = Vec::new();
        for row_result in sessions {
            let (id, project_id, started_at_val, ended_at_val, message_count, summary) =
                row_result.context("reading session row")?;

            let started_at = started_at_val.as_ref().and_then(|v| self.parse_timestamp(v));
            let ended_at = ended_at_val.as_ref().and_then(|v| self.parse_timestamp(v));

            let duration_minutes = match (started_at, ended_at) {
                (Some(s), Some(e)) => Some(e.signed_duration_since(s).num_minutes() as i32),
                _ => None,
            };

            result.push(Session {
                id,
                project_id,
                agent: agent_id.clone(),
                started_at,
                ended_at,
                duration_minutes,
                message_count,
                tool_call_count: None,
                total_tokens: None,
                files_changed: vec![],
                summary,
            });
        }

        Ok(result)
    }

    /// Read tool calls from the SQLite database (if a tool calls table is mapped).
    fn read_tool_calls(&self, conn: &rusqlite::Connection) -> Result<Vec<ToolCall>> {
        let table = match &self.mapping.tool_calls_table {
            Some(t) => t,
            None => return Ok(vec![]),
        };
        let tc_cols = match &self.mapping.tool_call_columns {
            Some(c) => c,
            None => return Ok(vec![]),
        };

        let mut select_cols = vec![tc_cols.id.clone()];
        if let Some(ref c) = tc_cols.session_id {
            select_cols.push(c.clone());
        }
        if let Some(ref c) = tc_cols.tool_name {
            select_cols.push(c.clone());
        }
        if let Some(ref c) = tc_cols.command {
            select_cols.push(c.clone());
        }
        if let Some(ref c) = tc_cols.success {
            select_cols.push(c.clone());
        }
        if let Some(ref c) = tc_cols.timestamp {
            select_cols.push(c.clone());
        }

        let sql = format!("SELECT {} FROM {}", select_cols.join(", "), table);
        debug!("[{}] executing: {sql}", self.meta.id);

        let mut stmt = conn.prepare(&sql).context("preparing tool call query")?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let mut idx = 1usize;

                let session_id: Option<String> = if tc_cols.session_id.is_some() {
                    let v = row.get(idx).ok();
                    idx += 1;
                    v
                } else {
                    None
                };
                let tool_name: Option<String> = if tc_cols.tool_name.is_some() {
                    let v = row.get(idx).ok();
                    idx += 1;
                    v
                } else {
                    None
                };
                let command: Option<String> = if tc_cols.command.is_some() {
                    let v = row.get(idx).ok();
                    idx += 1;
                    v
                } else {
                    None
                };
                let success: Option<bool> = if tc_cols.success.is_some() {
                    let v = row.get(idx).ok();
                    idx += 1;
                    v
                } else {
                    None
                };
                let timestamp_str: Option<String> = if tc_cols.timestamp.is_some() {
                    row.get(idx).ok()
                } else {
                    None
                };

                Ok((id, session_id, tool_name, command, success, timestamp_str))
            })
            .context("querying tool calls")?;

        let mut result = Vec::new();
        for row_result in rows {
            let (id, session_id, tool_name, command, success, timestamp_str) =
                row_result.context("reading tool call row")?;

            let timestamp = timestamp_str.as_deref().and_then(|s| {
                parse_timestamp_str(s, &self.mapping.session_columns.timestamp_format)
            });

            result.push(ToolCall {
                id,
                session_id,
                tool_name,
                command,
                success,
                error_message: None,
                duration_ms: None,
                timestamp,
            });
        }

        Ok(result)
    }
}

impl AgentAdapter for GenericSqliteAdapter {
    fn meta(&self) -> &AgentMeta {
        &self.meta
    }

    fn detect(&self) -> bool {
        self.db_path().is_file()
    }

    fn ingest(&self) -> Result<IngestOutput> {
        let db_path = self.db_path();
        if !db_path.is_file() {
            return Ok(IngestOutput::default());
        }

        let conn = rusqlite::Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening SQLite database: {}", db_path.display()))?;

        let mut output = IngestOutput::default();

        match self.read_sessions(&conn) {
            Ok(sessions) => {
                debug!(
                    "[{}] read {} sessions from {}",
                    self.meta.id,
                    sessions.len(),
                    db_path.display()
                );
                output.sessions = sessions;
            }
            Err(e) => {
                let msg = format!("{}: sessions: {e}", self.meta.id);
                warn!("{msg}");
                output.errors.push(msg);
            }
        }

        match self.read_tool_calls(&conn) {
            Ok(calls) => {
                debug!(
                    "[{}] read {} tool calls",
                    self.meta.id,
                    calls.len()
                );
                output.tool_calls = calls;
            }
            Err(e) => {
                let msg = format!("{}: tool_calls: {e}", self.meta.id);
                warn!("{msg}");
                output.errors.push(msg);
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a timestamp string in a specified format.
fn parse_timestamp_str(s: &str, format: &str) -> Option<NaiveDateTime> {
    match format {
        "unix_seconds" => {
            s.parse::<i64>()
                .ok()
                .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
                .map(|dt| dt.naive_utc())
        }
        "unix_millis" => {
            s.parse::<i64>()
                .ok()
                .and_then(chrono::DateTime::from_timestamp_millis)
                .map(|dt| dt.naive_utc())
        }
        _ => {
            // Default: ISO 8601
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.naive_utc())
                .ok()
                .or_else(|| {
                    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
                        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
                        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
                        .ok()
                })
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-built configs for known SQLite-based agents
// ---------------------------------------------------------------------------

/// Return default configs for all known SQLite-based agents.
pub fn known_sqlite_agents() -> Vec<DynamicAgentConfig> {
    vec![
        goose_config(),
        opencode_config(),
        cursor_config(),
        openclaw_config(),
    ]
}

/// Goose (Block Inc.) - stores conversations in SQLite.
pub fn goose_config() -> DynamicAgentConfig {
    use super::adapter::*;
    DynamicAgentConfig {
        id: "goose".into(),
        display_name: "Goose".into(),
        enabled: true,
        path: "~/.config/goose".into(),
        adapter_type: "sqlite".into(),
        sqlite: Some(SqliteMapping {
            db_file: "goose.db".into(),
            sessions_table: "conversations".into(),
            session_columns: SessionColumnMap {
                id: "id".into(),
                project_id: Some("workspace".into()),
                started_at: Some("created_at".into()),
                ended_at: Some("updated_at".into()),
                message_count: Some("message_count".into()),
                summary: Some("title".into()),
                content: Some("messages".into()),
                timestamp_format: "iso8601".into(),
            },
            tool_calls_table: None,
            tool_call_columns: None,
        }),
        jsonl: None,
    }
}

/// OpenCode - stores sessions in SQLite.
pub fn opencode_config() -> DynamicAgentConfig {
    use super::adapter::*;
    DynamicAgentConfig {
        id: "opencode".into(),
        display_name: "OpenCode".into(),
        enabled: true,
        path: "~/.local/share/opencode".into(),
        adapter_type: "sqlite".into(),
        sqlite: Some(SqliteMapping {
            db_file: "opencode.db".into(),
            sessions_table: "sessions".into(),
            session_columns: SessionColumnMap {
                id: "id".into(),
                project_id: Some("project_path".into()),
                started_at: Some("created_at".into()),
                ended_at: Some("updated_at".into()),
                message_count: None,
                summary: Some("title".into()),
                content: None,
                timestamp_format: "unix_seconds".into(),
            },
            tool_calls_table: Some("tool_calls".into()),
            tool_call_columns: Some(ToolCallColumnMap {
                id: "id".into(),
                session_id: Some("session_id".into()),
                tool_name: Some("tool_name".into()),
                command: Some("input".into()),
                success: Some("success".into()),
                timestamp: Some("created_at".into()),
            }),
        }),
        jsonl: None,
    }
}

/// Cursor - stores workspace state in .vscdb (SQLite).
pub fn cursor_config() -> DynamicAgentConfig {
    use super::adapter::*;
    DynamicAgentConfig {
        id: "cursor".into(),
        display_name: "Cursor".into(),
        enabled: true,
        path: "~/.cursor".into(),
        adapter_type: "sqlite".into(),
        sqlite: Some(SqliteMapping {
            db_file: "User/globalStorage/state.vscdb".into(),
            sessions_table: "ItemTable".into(),
            session_columns: SessionColumnMap {
                id: "key".into(),
                project_id: None,
                started_at: None,
                ended_at: None,
                message_count: None,
                summary: None,
                content: Some("value".into()),
                timestamp_format: "iso8601".into(),
            },
            tool_calls_table: None,
            tool_call_columns: None,
        }),
        jsonl: None,
    }
}

/// OpenClaw (formerly Claude Dev / Roo Code) - stores tasks in SQLite.
pub fn openclaw_config() -> DynamicAgentConfig {
    use super::adapter::*;
    DynamicAgentConfig {
        id: "openclaw".into(),
        display_name: "OpenClaw".into(),
        enabled: true,
        path: "~/.openclaw".into(),
        adapter_type: "sqlite".into(),
        sqlite: Some(SqliteMapping {
            db_file: "openclaw.db".into(),
            sessions_table: "tasks".into(),
            session_columns: SessionColumnMap {
                id: "id".into(),
                project_id: Some("workspace".into()),
                started_at: Some("created_at".into()),
                ended_at: Some("completed_at".into()),
                message_count: Some("message_count".into()),
                summary: Some("description".into()),
                content: Some("conversation".into()),
                timestamp_format: "iso8601".into(),
            },
            tool_calls_table: Some("tool_executions".into()),
            tool_call_columns: Some(ToolCallColumnMap {
                id: "id".into(),
                session_id: Some("task_id".into()),
                tool_name: Some("tool_name".into()),
                command: Some("input".into()),
                success: Some("success".into()),
                timestamp: Some("executed_at".into()),
            }),
        }),
        jsonl: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db(dir: &std::path::Path) -> PathBuf {
        let db_path = dir.join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        conn.execute_batch(
            "CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                project TEXT,
                created_at TEXT,
                updated_at TEXT,
                msg_count INTEGER,
                title TEXT
            );
            INSERT INTO sessions VALUES (
                'sess-1', 'my-project',
                '2026-03-20T10:00:00Z', '2026-03-20T10:30:00Z',
                5, 'Fix auth bug'
            );
            INSERT INTO sessions VALUES (
                'sess-2', 'my-project',
                '2026-03-21T14:00:00Z', '2026-03-21T14:45:00Z',
                12, 'Add user dashboard'
            );

            CREATE TABLE tool_calls (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                tool TEXT,
                cmd TEXT,
                ok INTEGER,
                ts TEXT
            );
            INSERT INTO tool_calls VALUES (
                'tc-1', 'sess-1', 'read_file', 'src/auth.rs', 1, '2026-03-20T10:05:00Z'
            );
            INSERT INTO tool_calls VALUES (
                'tc-2', 'sess-1', 'write_file', 'src/auth.rs', 1, '2026-03-20T10:10:00Z'
            );
            INSERT INTO tool_calls VALUES (
                'tc-3', 'sess-2', 'shell', 'cargo test', 0, '2026-03-21T14:20:00Z'
            );"
        ).unwrap();

        db_path
    }

    fn test_mapping() -> SqliteMapping {
        use super::super::adapter::*;
        SqliteMapping {
            db_file: "test.db".into(),
            sessions_table: "sessions".into(),
            session_columns: SessionColumnMap {
                id: "id".into(),
                project_id: Some("project".into()),
                started_at: Some("created_at".into()),
                ended_at: Some("updated_at".into()),
                message_count: Some("msg_count".into()),
                summary: Some("title".into()),
                content: None,
                timestamp_format: "iso8601".into(),
            },
            tool_calls_table: Some("tool_calls".into()),
            tool_call_columns: Some(ToolCallColumnMap {
                id: "id".into(),
                session_id: Some("session_id".into()),
                tool_name: Some("tool".into()),
                command: Some("cmd".into()),
                success: Some("ok".into()),
                timestamp: Some("ts".into()),
            }),
        }
    }

    #[test]
    fn test_detect_missing_db() {
        let tmp = TempDir::new().unwrap();
        let adapter = GenericSqliteAdapter::new(
            AgentMeta {
                id: "test".into(),
                display_name: "Test".into(),
                storage_format: "sqlite".into(),
                default_path: tmp.path().to_str().unwrap().into(),
            },
            tmp.path().to_path_buf(),
            test_mapping(),
        );
        assert!(!adapter.detect());
    }

    #[test]
    fn test_detect_present_db() {
        let tmp = TempDir::new().unwrap();
        create_test_db(tmp.path());

        let adapter = GenericSqliteAdapter::new(
            AgentMeta {
                id: "test".into(),
                display_name: "Test".into(),
                storage_format: "sqlite".into(),
                default_path: tmp.path().to_str().unwrap().into(),
            },
            tmp.path().to_path_buf(),
            test_mapping(),
        );
        assert!(adapter.detect());
    }

    #[test]
    fn test_ingest_sessions() {
        let tmp = TempDir::new().unwrap();
        create_test_db(tmp.path());

        let adapter = GenericSqliteAdapter::new(
            AgentMeta {
                id: "test_agent".into(),
                display_name: "Test Agent".into(),
                storage_format: "sqlite".into(),
                default_path: tmp.path().to_str().unwrap().into(),
            },
            tmp.path().to_path_buf(),
            test_mapping(),
        );

        let output = adapter.ingest().unwrap();
        assert_eq!(output.session_count(), 2);
        assert_eq!(output.tool_call_count(), 3);
        assert!(output.errors.is_empty());

        // Verify session data
        let s1 = output.sessions.iter().find(|s| s.id == "sess-1").unwrap();
        assert_eq!(s1.agent, "test_agent");
        assert_eq!(s1.project_id.as_deref(), Some("my-project"));
        assert_eq!(s1.message_count, Some(5));
        assert_eq!(s1.summary.as_deref(), Some("Fix auth bug"));
        assert!(s1.started_at.is_some());
        assert!(s1.ended_at.is_some());
        assert_eq!(s1.duration_minutes, Some(30));

        // Verify tool call data
        let tc1 = output.tool_calls.iter().find(|t| t.id == "tc-1").unwrap();
        assert_eq!(tc1.session_id.as_deref(), Some("sess-1"));
        assert_eq!(tc1.tool_name.as_deref(), Some("read_file"));
        assert_eq!(tc1.command.as_deref(), Some("src/auth.rs"));
        assert_eq!(tc1.success, Some(true));

        let tc3 = output.tool_calls.iter().find(|t| t.id == "tc-3").unwrap();
        assert_eq!(tc3.success, Some(false));
    }

    #[test]
    fn test_ingest_no_tool_calls_table() {
        let tmp = TempDir::new().unwrap();
        create_test_db(tmp.path());

        let mut mapping = test_mapping();
        mapping.tool_calls_table = None;
        mapping.tool_call_columns = None;

        let adapter = GenericSqliteAdapter::new(
            AgentMeta {
                id: "test".into(),
                display_name: "Test".into(),
                storage_format: "sqlite".into(),
                default_path: tmp.path().to_str().unwrap().into(),
            },
            tmp.path().to_path_buf(),
            mapping,
        );

        let output = adapter.ingest().unwrap();
        assert_eq!(output.session_count(), 2);
        assert_eq!(output.tool_call_count(), 0); // No tool calls table mapped
    }

    #[test]
    fn test_from_config() {
        let config = goose_config();
        let adapter = GenericSqliteAdapter::from_config(&config);
        assert!(adapter.is_some());
        let adapter = adapter.unwrap();
        assert_eq!(adapter.meta().id, "goose");
        assert_eq!(adapter.meta().display_name, "Goose");
    }

    #[test]
    fn test_from_config_no_sqlite() {
        let config = DynamicAgentConfig {
            id: "test".into(),
            display_name: "Test".into(),
            enabled: true,
            path: "~/.test".into(),
            adapter_type: "jsonl".into(),
            sqlite: None,
            jsonl: None,
        };
        assert!(GenericSqliteAdapter::from_config(&config).is_none());
    }

    #[test]
    fn test_known_sqlite_agents() {
        let agents = known_sqlite_agents();
        assert_eq!(agents.len(), 4);
        let ids: Vec<&str> = agents.iter().map(|a| a.id.as_str()).collect();
        assert!(ids.contains(&"goose"));
        assert!(ids.contains(&"opencode"));
        assert!(ids.contains(&"cursor"));
        assert!(ids.contains(&"openclaw"));
    }

    #[test]
    fn test_parse_timestamp_str_iso() {
        let dt = parse_timestamp_str("2026-03-20T10:00:00Z", "iso8601");
        assert!(dt.is_some());
        let dt = dt.unwrap();
        assert_eq!(dt.date(), chrono::NaiveDate::from_ymd_opt(2026, 3, 20).unwrap());
        assert_eq!(dt.time(), chrono::NaiveTime::from_hms_opt(10, 0, 0).unwrap());
    }

    #[test]
    fn test_parse_timestamp_str_unix() {
        let dt = parse_timestamp_str("1774011600", "unix_seconds");
        assert!(dt.is_some());

        let dt_ms = parse_timestamp_str("1774011600000", "unix_millis");
        assert!(dt_ms.is_some());
    }

    #[test]
    fn test_parse_timestamp_str_sqlite_format() {
        let dt = parse_timestamp_str("2026-03-20 10:00:00", "iso8601");
        assert!(dt.is_some());
    }

    #[test]
    fn test_ingest_empty_db() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, project TEXT, created_at TEXT, updated_at TEXT, msg_count INTEGER, title TEXT);"
        ).unwrap();

        let adapter = GenericSqliteAdapter::new(
            AgentMeta {
                id: "test".into(),
                display_name: "Test".into(),
                storage_format: "sqlite".into(),
                default_path: tmp.path().to_str().unwrap().into(),
            },
            tmp.path().to_path_buf(),
            test_mapping(),
        );

        let output = adapter.ingest().unwrap();
        assert_eq!(output.session_count(), 0);
    }
}
