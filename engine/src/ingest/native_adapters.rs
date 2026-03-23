//! Native adapter wrappers for existing Claude, Codex, and Gemini ingesters.
//!
//! These wrap the existing parser implementations behind the [`AgentAdapter`]
//! trait so they can be used through the unified [`AdapterRegistry`].

use anyhow::Result;
use std::path::PathBuf;

use super::adapter::{expand_tilde, AgentAdapter, AgentMeta, IngestOutput};
use super::claude::ClaudeIngester;
use super::codex::CodexIngester;
use super::gemini::GeminiIngester;

// ---------------------------------------------------------------------------
// Claude Code adapter
// ---------------------------------------------------------------------------

/// Adapter wrapping the native [`ClaudeIngester`].
pub struct ClaudeAdapter {
    meta: AgentMeta,
    base_path: PathBuf,
}

impl ClaudeAdapter {
    pub fn new(path: &str) -> Self {
        Self {
            meta: AgentMeta {
                id: "claude_code".into(),
                display_name: "Claude Code".into(),
                storage_format: "jsonl".into(),
                default_path: path.into(),
            },
            base_path: expand_tilde(path),
        }
    }

    pub fn default() -> Self {
        Self::new("~/.claude")
    }
}

impl AgentAdapter for ClaudeAdapter {
    fn meta(&self) -> &AgentMeta {
        &self.meta
    }

    fn detect(&self) -> bool {
        self.base_path.join("projects").is_dir()
    }

    fn ingest(&self) -> Result<IngestOutput> {
        let ingester = ClaudeIngester::with_base_path(&self.base_path);
        let result = ingester.ingest_all()?;
        Ok(IngestOutput {
            sessions: result.sessions,
            tool_calls: result.tool_calls,
            memories: result.memories,
            errors: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Codex adapter
// ---------------------------------------------------------------------------

/// Adapter wrapping the native [`CodexIngester`].
pub struct CodexAdapter {
    meta: AgentMeta,
    base_path: PathBuf,
}

impl CodexAdapter {
    pub fn new(path: &str) -> Self {
        Self {
            meta: AgentMeta {
                id: "codex".into(),
                display_name: "Codex CLI".into(),
                storage_format: "jsonl".into(),
                default_path: path.into(),
            },
            base_path: expand_tilde(path),
        }
    }

    pub fn default() -> Self {
        Self::new("~/.codex")
    }
}

impl AgentAdapter for CodexAdapter {
    fn meta(&self) -> &AgentMeta {
        &self.meta
    }

    fn detect(&self) -> bool {
        self.base_path.join("sessions").is_dir()
    }

    fn ingest(&self) -> Result<IngestOutput> {
        let ingester = CodexIngester::with_base_path(self.base_path.clone());
        let result = ingester.ingest_all()?;
        Ok(IngestOutput {
            sessions: result.sessions,
            tool_calls: result.tool_calls,
            memories: result.memories,
            errors: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Gemini adapter
// ---------------------------------------------------------------------------

/// Adapter wrapping the native [`GeminiIngester`].
pub struct GeminiAdapter {
    meta: AgentMeta,
    base_path: PathBuf,
}

impl GeminiAdapter {
    pub fn new(path: &str) -> Self {
        Self {
            meta: AgentMeta {
                id: "gemini".into(),
                display_name: "Gemini CLI".into(),
                storage_format: "json".into(),
                default_path: path.into(),
            },
            base_path: expand_tilde(path),
        }
    }

    pub fn default() -> Self {
        Self::new("~/.gemini")
    }
}

impl AgentAdapter for GeminiAdapter {
    fn meta(&self) -> &AgentMeta {
        &self.meta
    }

    fn detect(&self) -> bool {
        self.base_path.join("tmp").is_dir()
    }

    fn ingest(&self) -> Result<IngestOutput> {
        let ingester = GeminiIngester::with_base_path(&self.base_path);
        let (result, sessions, tool_calls, memories) = ingester.ingest_all();
        Ok(IngestOutput {
            sessions,
            tool_calls,
            memories,
            errors: result.errors,
        })
    }
}

// ---------------------------------------------------------------------------
// Factory: build default registry with all native adapters
// ---------------------------------------------------------------------------

/// Build an [`AdapterRegistry`] pre-loaded with all native adapters
/// using their default paths.
pub fn build_default_registry() -> super::adapter::AdapterRegistry {
    let mut registry = super::adapter::AdapterRegistry::new();
    registry.register(Box::new(ClaudeAdapter::default()));
    registry.register(Box::new(CodexAdapter::default()));
    registry.register(Box::new(GeminiAdapter::default()));
    registry
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_claude_adapter_detect() {
        let tmp = TempDir::new().unwrap();
        let adapter = ClaudeAdapter::new(tmp.path().to_str().unwrap());
        assert!(!adapter.detect());

        fs::create_dir_all(tmp.path().join("projects")).unwrap();
        assert!(adapter.detect());
    }

    #[test]
    fn test_codex_adapter_detect() {
        let tmp = TempDir::new().unwrap();
        let adapter = CodexAdapter::new(tmp.path().to_str().unwrap());
        assert!(!adapter.detect());

        fs::create_dir_all(tmp.path().join("sessions")).unwrap();
        assert!(adapter.detect());
    }

    #[test]
    fn test_gemini_adapter_detect() {
        let tmp = TempDir::new().unwrap();
        let adapter = GeminiAdapter::new(tmp.path().to_str().unwrap());
        assert!(!adapter.detect());

        fs::create_dir_all(tmp.path().join("tmp")).unwrap();
        assert!(adapter.detect());
    }

    #[test]
    fn test_claude_adapter_ingest_empty() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("projects")).unwrap();

        let adapter = ClaudeAdapter::new(tmp.path().to_str().unwrap());
        let output = adapter.ingest().unwrap();
        assert_eq!(output.session_count(), 0);
        assert_eq!(output.tool_call_count(), 0);
    }

    #[test]
    fn test_codex_adapter_ingest_empty() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("sessions")).unwrap();

        let adapter = CodexAdapter::new(tmp.path().to_str().unwrap());
        let output = adapter.ingest().unwrap();
        assert_eq!(output.session_count(), 0);
    }

    #[test]
    fn test_gemini_adapter_ingest_empty() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("tmp")).unwrap();

        let adapter = GeminiAdapter::new(tmp.path().to_str().unwrap());
        let output = adapter.ingest().unwrap();
        assert_eq!(output.session_count(), 0);
    }

    #[test]
    fn test_claude_adapter_ingest_with_data() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("projects").join("test-project");
        fs::create_dir_all(&project_dir).unwrap();

        // Write a sessions index
        fs::write(
            project_dir.join("sessions-index.json"),
            r#"{"version":1,"entries":[{"sessionId":"s1","summary":"test","messageCount":2,"created":"2026-01-01T00:00:00Z","modified":"2026-01-01T00:01:00Z"}]}"#,
        ).unwrap();

        let adapter = ClaudeAdapter::new(tmp.path().to_str().unwrap());
        let output = adapter.ingest().unwrap();
        assert_eq!(output.session_count(), 1);
        assert_eq!(output.sessions[0].agent, "claude_code");
    }

    #[test]
    fn test_build_default_registry() {
        let reg = build_default_registry();
        assert_eq!(reg.agent_ids().len(), 3);
        assert!(reg.get("claude_code").is_some());
        assert!(reg.get("codex").is_some());
        assert!(reg.get("gemini").is_some());
    }

    #[test]
    fn test_registry_ingest_agent() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("projects")).unwrap();

        let mut reg = super::super::adapter::AdapterRegistry::new();
        reg.register(Box::new(ClaudeAdapter::new(tmp.path().to_str().unwrap())));

        let output = reg.ingest_agent("claude_code").unwrap();
        assert_eq!(output.session_count(), 0);

        let err = reg.ingest_agent("nonexistent");
        assert!(err.is_err());
    }

    #[test]
    fn test_adapter_meta() {
        let claude = ClaudeAdapter::default();
        assert_eq!(claude.meta().id, "claude_code");
        assert_eq!(claude.meta().storage_format, "jsonl");

        let codex = CodexAdapter::default();
        assert_eq!(codex.meta().id, "codex");

        let gemini = GeminiAdapter::default();
        assert_eq!(gemini.meta().id, "gemini");
        assert_eq!(gemini.meta().storage_format, "json");
    }
}
