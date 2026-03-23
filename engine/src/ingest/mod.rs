pub mod adapter;
pub mod claude;
pub mod codex;
pub mod gemini;
pub mod native_adapters;
#[cfg(feature = "sqlite-adapters")]
pub mod sqlite_adapter;

use serde::{Deserialize, Serialize};

pub use adapter::{
    AdapterRegistry, AgentAdapter, AgentMeta, DynamicAgentConfig, IngestOutput,
};
pub use claude::ClaudeIngester;
pub use codex::CodexIngester;
pub use gemini::GeminiIngester;
pub use native_adapters::{
    build_default_registry, ClaudeAdapter, CodexAdapter, GeminiAdapter,
};
#[cfg(feature = "sqlite-adapters")]
pub use sqlite_adapter::GenericSqliteAdapter;

/// Summary of what an ingestion pass discovered and parsed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestResult {
    pub agent: String,
    pub sessions_found: usize,
    pub tool_calls_found: usize,
    pub memories_found: usize,
    pub errors: Vec<String>,
}

impl IngestResult {
    pub fn new(agent: &str) -> Self {
        Self {
            agent: agent.to_string(),
            ..Default::default()
        }
    }
}

impl std::fmt::Display for IngestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] sessions={}, tool_calls={}, memories={}, errors={}",
            self.agent,
            self.sessions_found,
            self.tool_calls_found,
            self.memories_found,
            self.errors.len()
        )
    }
}

/// Helper to parse ISO 8601 timestamps (e.g. "2026-03-12T23:26:44.124Z") into NaiveDateTime.
pub fn parse_iso_timestamp(s: &str) -> Option<chrono::NaiveDateTime> {
    // Try with fractional seconds first, then without
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f%:z"))
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_iso_timestamp() {
        let ts = parse_iso_timestamp("2026-03-12T23:26:44.124Z");
        assert!(ts.is_some());
        let dt = ts.unwrap();
        assert_eq!(chrono::Datelike::year(&dt.and_utc()), 2026);

        let ts2 = parse_iso_timestamp("2026-01-05T01:04:45Z");
        assert!(ts2.is_some());

        let bad = parse_iso_timestamp("not a timestamp");
        assert!(bad.is_none());
    }

    #[test]
    fn test_ingest_result_display() {
        let mut r = IngestResult::new("claude_code");
        r.sessions_found = 5;
        r.tool_calls_found = 42;
        r.memories_found = 3;
        let s = format!("{r}");
        assert!(s.contains("claude_code"));
        assert!(s.contains("sessions=5"));
    }
}
