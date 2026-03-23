//! Pluggable agent adapter system.
//!
//! Defines the [`AgentAdapter`] trait that all agent ingesters must implement,
//! a normalized [`IngestOutput`] struct, and an [`AdapterRegistry`] for
//! discovering and running adapters.

use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::store::duckdb::{Memory, Session, ToolCall};

// ---------------------------------------------------------------------------
// Normalized output types
// ---------------------------------------------------------------------------

/// Normalized output from any agent adapter.
///
/// Every adapter produces the same struct regardless of the underlying
/// storage format (JSONL, SQLite, JSON, Markdown, etc.).
#[derive(Debug, Default)]
pub struct IngestOutput {
    pub sessions: Vec<Session>,
    pub tool_calls: Vec<ToolCall>,
    pub memories: Vec<Memory>,
    pub errors: Vec<String>,
}

impl IngestOutput {
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn tool_call_count(&self) -> usize {
        self.tool_calls.len()
    }

    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Merge another output into this one.
    pub fn merge(&mut self, other: IngestOutput) {
        self.sessions.extend(other.sessions);
        self.tool_calls.extend(other.tool_calls);
        self.memories.extend(other.memories);
        self.errors.extend(other.errors);
    }
}

impl std::fmt::Display for IngestOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sessions={}, tool_calls={}, memories={}, errors={}",
            self.sessions.len(),
            self.tool_calls.len(),
            self.memories.len(),
            self.errors.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Agent metadata
// ---------------------------------------------------------------------------

/// Metadata describing an agent adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMeta {
    /// Unique agent identifier (e.g. "claude_code", "goose", "cursor").
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Storage format: "jsonl", "sqlite", "json", "markdown".
    pub storage_format: String,
    /// Default base path (with `~` for home directory).
    pub default_path: String,
}

// ---------------------------------------------------------------------------
// AgentAdapter trait
// ---------------------------------------------------------------------------

/// Trait that all agent ingesters must implement.
///
/// Native adapters (Claude, Codex, Gemini) implement this with custom parsing
/// logic. Generic adapters (SQLite, JSONL) implement it with config-driven
/// column mappings.
pub trait AgentAdapter: Send + Sync {
    /// Return metadata about this adapter.
    fn meta(&self) -> &AgentMeta;

    /// Check whether the agent's data directory exists and has data.
    fn detect(&self) -> bool;

    /// Run full ingestion and return normalized output.
    fn ingest(&self) -> Result<IngestOutput>;
}

// ---------------------------------------------------------------------------
// AdapterRegistry
// ---------------------------------------------------------------------------

/// Registry of all available agent adapters.
///
/// Adapters are registered at startup and can be discovered, filtered,
/// and run from a single place.
pub struct AdapterRegistry {
    adapters: Vec<Box<dyn AgentAdapter>>,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    /// Register an adapter.
    pub fn register(&mut self, adapter: Box<dyn AgentAdapter>) {
        self.adapters.push(adapter);
    }

    /// Return all registered adapters.
    pub fn adapters(&self) -> &[Box<dyn AgentAdapter>] {
        &self.adapters
    }

    /// Return only adapters whose agent data is detected on this system.
    pub fn detected(&self) -> Vec<&dyn AgentAdapter> {
        self.adapters
            .iter()
            .filter(|a| a.detect())
            .map(|a| a.as_ref())
            .collect()
    }

    /// Return adapter by agent ID.
    pub fn get(&self, agent_id: &str) -> Option<&dyn AgentAdapter> {
        self.adapters
            .iter()
            .find(|a| a.meta().id == agent_id)
            .map(|a| a.as_ref())
    }

    /// Ingest from all detected adapters, merging results.
    pub fn ingest_all_detected(&self) -> IngestOutput {
        let mut output = IngestOutput::default();
        for adapter in self.detected() {
            let meta = adapter.meta();
            match adapter.ingest() {
                Ok(result) => {
                    tracing::info!(
                        "[{}] ingested: {}",
                        meta.display_name,
                        result,
                    );
                    output.merge(result);
                }
                Err(e) => {
                    tracing::warn!("[{}] ingestion failed: {e}", meta.display_name);
                    output.errors.push(format!("{}: {e}", meta.id));
                }
            }
        }
        output
    }

    /// Ingest from a specific agent by ID.
    pub fn ingest_agent(&self, agent_id: &str) -> Result<IngestOutput> {
        let adapter = self
            .get(agent_id)
            .ok_or_else(|| anyhow::anyhow!("unknown agent: {agent_id}"))?;
        adapter.ingest()
    }

    /// List all registered agent IDs.
    pub fn agent_ids(&self) -> Vec<&str> {
        self.adapters.iter().map(|a| a.meta().id.as_str()).collect()
    }

    /// List detected agent IDs.
    pub fn detected_ids(&self) -> Vec<&str> {
        self.detected().iter().map(|a| a.meta().id.as_str()).collect()
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Config-driven dynamic agent definition
// ---------------------------------------------------------------------------

/// Configuration for a dynamically-defined agent adapter.
///
/// This allows adding new agents via YAML config alone, without writing
/// Rust code. The adapter type determines which generic parser to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAgentConfig {
    /// Unique agent identifier.
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Whether this agent is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Base path to the agent's data directory (supports `~`).
    pub path: String,
    /// Adapter type: "sqlite", "jsonl", "json", "markdown".
    pub adapter_type: String,
    /// For SQLite adapters: table/column mappings.
    #[serde(default)]
    pub sqlite: Option<SqliteMapping>,
    /// For JSONL adapters: field path mappings.
    #[serde(default)]
    pub jsonl: Option<JsonlMapping>,
}

fn default_true() -> bool {
    true
}

/// SQLite table and column mappings for generic SQLite adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqliteMapping {
    /// Path to the SQLite database file, relative to the agent's base path.
    pub db_file: String,
    /// Table containing conversation/session data.
    pub sessions_table: String,
    /// Column mappings for the sessions table.
    pub session_columns: SessionColumnMap,
    /// Optional: table containing tool call data.
    #[serde(default)]
    pub tool_calls_table: Option<String>,
    /// Optional: column mappings for tool calls.
    #[serde(default)]
    pub tool_call_columns: Option<ToolCallColumnMap>,
}

/// Maps generic session fields to actual SQLite column names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionColumnMap {
    pub id: String,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub ended_at: Option<String>,
    #[serde(default)]
    pub message_count: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    /// Column containing the full conversation text/JSON (for extraction).
    #[serde(default)]
    pub content: Option<String>,
    /// Timestamp format: "iso8601", "unix_seconds", "unix_millis".
    #[serde(default = "default_ts_format")]
    pub timestamp_format: String,
}

fn default_ts_format() -> String {
    "iso8601".to_string()
}

/// Maps generic tool call fields to actual SQLite column names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallColumnMap {
    pub id: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub success: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
}

/// JSONL field path mappings for generic JSONL adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonlMapping {
    /// Glob pattern for files to ingest, relative to base path.
    pub file_pattern: String,
    /// JSON path to session ID field.
    pub session_id_path: String,
    /// JSON path to timestamp field.
    #[serde(default)]
    pub timestamp_path: Option<String>,
    /// JSON path to message content.
    #[serde(default)]
    pub content_path: Option<String>,
    /// JSON path to tool call name.
    #[serde(default)]
    pub tool_name_path: Option<String>,
    /// Line type field and value that indicates a tool call.
    #[serde(default)]
    pub tool_call_type: Option<String>,
}

// ---------------------------------------------------------------------------
// Helper: expand tilde in paths
// ---------------------------------------------------------------------------

/// Expand a leading `~` in a path string to the user's home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_output_merge() {
        let mut a = IngestOutput::default();
        a.sessions.push(Session {
            id: "s1".into(),
            project_id: None,
            agent: "test".into(),
            started_at: None,
            ended_at: None,
            duration_minutes: None,
            message_count: None,
            tool_call_count: None,
            total_tokens: None,
            files_changed: vec![],
            summary: None,
        });

        let mut b = IngestOutput::default();
        b.sessions.push(Session {
            id: "s2".into(),
            project_id: None,
            agent: "test2".into(),
            started_at: None,
            ended_at: None,
            duration_minutes: None,
            message_count: None,
            tool_call_count: None,
            total_tokens: None,
            files_changed: vec![],
            summary: None,
        });
        b.errors.push("some error".into());

        a.merge(b);
        assert_eq!(a.session_count(), 2);
        assert_eq!(a.errors.len(), 1);
    }

    #[test]
    fn test_ingest_output_display() {
        let output = IngestOutput::default();
        let s = format!("{output}");
        assert!(s.contains("sessions=0"));
        assert!(s.contains("tool_calls=0"));
    }

    #[test]
    fn test_expand_tilde() {
        let expanded = expand_tilde("~/.claude");
        assert!(!expanded.starts_with("~"));
        assert!(expanded.to_string_lossy().contains(".claude"));

        let abs = expand_tilde("/absolute/path");
        assert_eq!(abs, PathBuf::from("/absolute/path"));

        let rel = expand_tilde("relative/path");
        assert_eq!(rel, PathBuf::from("relative/path"));
    }

    struct MockAdapter {
        meta: AgentMeta,
        detected: bool,
    }

    impl AgentAdapter for MockAdapter {
        fn meta(&self) -> &AgentMeta {
            &self.meta
        }
        fn detect(&self) -> bool {
            self.detected
        }
        fn ingest(&self) -> Result<IngestOutput> {
            Ok(IngestOutput::default())
        }
    }

    #[test]
    fn test_registry_basic() {
        let mut reg = AdapterRegistry::new();
        reg.register(Box::new(MockAdapter {
            meta: AgentMeta {
                id: "test_agent".into(),
                display_name: "Test Agent".into(),
                storage_format: "jsonl".into(),
                default_path: "~/.test".into(),
            },
            detected: true,
        }));
        reg.register(Box::new(MockAdapter {
            meta: AgentMeta {
                id: "missing_agent".into(),
                display_name: "Missing".into(),
                storage_format: "sqlite".into(),
                default_path: "~/.missing".into(),
            },
            detected: false,
        }));

        assert_eq!(reg.agent_ids().len(), 2);
        assert_eq!(reg.detected_ids().len(), 1);
        assert_eq!(reg.detected_ids()[0], "test_agent");
        assert!(reg.get("test_agent").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_ingest_all_detected() {
        let mut reg = AdapterRegistry::new();
        reg.register(Box::new(MockAdapter {
            meta: AgentMeta {
                id: "a".into(),
                display_name: "A".into(),
                storage_format: "jsonl".into(),
                default_path: "~/.a".into(),
            },
            detected: true,
        }));
        reg.register(Box::new(MockAdapter {
            meta: AgentMeta {
                id: "b".into(),
                display_name: "B".into(),
                storage_format: "sqlite".into(),
                default_path: "~/.b".into(),
            },
            detected: false,
        }));

        let output = reg.ingest_all_detected();
        assert!(output.errors.is_empty());
    }

    #[test]
    fn test_registry_ingest_unknown_agent() {
        let reg = AdapterRegistry::new();
        let result = reg.ingest_agent("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_agent_config_serde() {
        let yaml = r#"
id: goose
display_name: Goose
enabled: true
path: "~/.config/goose"
adapter_type: sqlite
sqlite:
  db_file: goose.db
  sessions_table: conversations
  session_columns:
    id: id
    started_at: created_at
    summary: title
    content: messages
    timestamp_format: iso8601
"#;
        let config: DynamicAgentConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.id, "goose");
        assert!(config.enabled);
        assert!(config.sqlite.is_some());
        let sqlite = config.sqlite.unwrap();
        assert_eq!(sqlite.db_file, "goose.db");
        assert_eq!(sqlite.sessions_table, "conversations");
        assert_eq!(sqlite.session_columns.id, "id");
    }
}
