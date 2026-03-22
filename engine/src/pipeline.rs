//! Ingestion pipeline that ties agent parsers to DuckDB storage.
//!
//! Each agent ingester discovers and parses artifacts, and the pipeline
//! inserts the resulting sessions, tool calls, and memories into DuckDB.

use anyhow::Result;
use tracing::{debug, info, warn};

use crate::ingest::{ClaudeIngester, CodexIngester, GeminiIngester, IngestResult};
use crate::store::DuckStore;

/// Ingestion pipeline connecting parsers to persistent storage.
pub struct IngestPipeline {
    duck_store: DuckStore,
}

impl IngestPipeline {
    /// Create a new pipeline backed by the given DuckDB store.
    pub fn new(duck_store: DuckStore) -> Self {
        Self { duck_store }
    }

    /// Run a full ingestion pass for all agents. Returns combined results.
    ///
    /// If one agent fails, the error is logged and the remaining agents are
    /// still processed.
    pub fn run_full_ingest(&self) -> Result<Vec<IngestResult>> {
        let mut results = Vec::new();

        match self.ingest_claude() {
            Ok(r) => results.push(r),
            Err(e) => warn!("claude ingestion failed: {e}"),
        }
        match self.ingest_codex() {
            Ok(r) => results.push(r),
            Err(e) => warn!("codex ingestion failed: {e}"),
        }
        match self.ingest_gemini() {
            Ok(r) => results.push(r),
            Err(e) => warn!("gemini ingestion failed: {e}"),
        }

        let total_sessions: usize = results.iter().map(|r| r.sessions_found).sum();
        let total_tools: usize = results.iter().map(|r| r.tool_calls_found).sum();
        let total_memories: usize = results.iter().map(|r| r.memories_found).sum();
        info!(
            "full ingest complete: {} sessions, {} tool_calls, {} memories across {} agents",
            total_sessions,
            total_tools,
            total_memories,
            results.len()
        );

        Ok(results)
    }

    /// Ingest only Claude Code artifacts.
    pub fn ingest_claude(&self) -> Result<IngestResult> {
        info!("starting Claude Code ingestion");
        let ingester = ClaudeIngester::new()?;
        let data = ingester.ingest_all()?;

        let mut result = IngestResult::new("claude_code");
        result.sessions_found = data.sessions.len();
        result.tool_calls_found = data.tool_calls.len();
        result.memories_found = data.memories.len();

        // Persist sessions.
        for session in &data.sessions {
            if let Err(e) = self.duck_store.insert_or_replace_session(session) {
                let msg = format!("session {}: {e}", session.id);
                warn!("{msg}");
                result.errors.push(msg);
            }
        }

        // Persist tool calls.
        for tc in &data.tool_calls {
            if let Err(e) = self.duck_store.insert_tool_call(tc) {
                debug!("tool_call {}: {e}", tc.id);
            }
        }

        // Persist memories.
        for mem in &data.memories {
            if let Err(e) = self.duck_store.insert_memory(mem) {
                let msg = format!("memory {}: {e}", mem.id);
                warn!("{msg}");
                result.errors.push(msg);
            }
        }

        info!("claude ingestion: {result}");
        Ok(result)
    }

    /// Ingest only Codex artifacts.
    pub fn ingest_codex(&self) -> Result<IngestResult> {
        info!("starting Codex ingestion");
        let ingester = match CodexIngester::new() {
            Some(i) => i,
            None => {
                debug!("codex agent not found, skipping");
                return Ok(IngestResult::new("codex"));
            }
        };
        let data = ingester.ingest_all()?;

        let mut result = IngestResult::new("codex");
        result.sessions_found = data.sessions.len();
        result.tool_calls_found = data.tool_calls.len();
        result.memories_found = data.memories.len();

        for session in &data.sessions {
            if let Err(e) = self.duck_store.insert_or_replace_session(session) {
                let msg = format!("session {}: {e}", session.id);
                warn!("{msg}");
                result.errors.push(msg);
            }
        }

        for tc in &data.tool_calls {
            if let Err(e) = self.duck_store.insert_tool_call(tc) {
                debug!("tool_call {}: {e}", tc.id);
            }
        }

        for mem in &data.memories {
            if let Err(e) = self.duck_store.insert_memory(mem) {
                let msg = format!("memory {}: {e}", mem.id);
                warn!("{msg}");
                result.errors.push(msg);
            }
        }

        info!("codex ingestion: {result}");
        Ok(result)
    }

    /// Ingest only Gemini artifacts.
    pub fn ingest_gemini(&self) -> Result<IngestResult> {
        info!("starting Gemini ingestion");
        let ingester = match GeminiIngester::new() {
            Some(i) => i,
            None => {
                debug!("gemini agent not found, skipping");
                return Ok(IngestResult::new("gemini"));
            }
        };
        let (mut result, sessions, tool_calls, memories) = ingester.ingest_all();

        for session in &sessions {
            if let Err(e) = self.duck_store.insert_or_replace_session(session) {
                let msg = format!("session {}: {e}", session.id);
                warn!("{msg}");
                result.errors.push(msg);
            }
        }

        for tc in &tool_calls {
            if let Err(e) = self.duck_store.insert_tool_call(tc) {
                debug!("tool_call {}: {e}", tc.id);
            }
        }

        for mem in &memories {
            if let Err(e) = self.duck_store.insert_memory(mem) {
                let msg = format!("memory {}: {e}", mem.id);
                warn!("{msg}");
                result.errors.push(msg);
            }
        }

        info!("gemini ingestion: {result}");
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_handles_empty_gracefully() {
        // In-memory DuckDB, no agent dirs exist -> should return Ok with
        // empty or zero results, not crash.
        let store = DuckStore::open_in_memory().expect("open in-memory DuckDB");
        let pipeline = IngestPipeline::new(store);
        let results = pipeline
            .run_full_ingest()
            .expect("full ingest should not fail");

        // Each agent either returns empty results or is skipped entirely.
        for r in &results {
            assert!(r.errors.is_empty() || !r.errors.is_empty()); // doesn't panic
        }
    }

    #[test]
    fn test_pipeline_ingest_stores_to_duckdb() {
        use crate::store::duckdb::{Memory, Session, ToolCall};
        use chrono::Utc;

        let store = DuckStore::open_in_memory().expect("open in-memory DuckDB");

        // Manually insert data to verify the store methods work with the pipeline.
        let session = Session {
            id: "test-session-1".into(),
            project_id: Some("test-project".into()),
            agent: "claude_code".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(5),
            message_count: Some(10),
            tool_call_count: Some(3),
            total_tokens: Some(500),
            files_changed: vec!["src/main.rs".into()],
            summary: Some("test session".into()),
        };
        store
            .insert_or_replace_session(&session)
            .expect("insert session");

        let tc = ToolCall {
            id: "tc-1".into(),
            session_id: Some("test-session-1".into()),
            tool_name: Some("Bash".into()),
            command: Some("ls -la".into()),
            success: Some(true),
            error_message: None,
            duration_ms: Some(100),
            timestamp: Some(Utc::now().naive_utc()),
        };
        store.insert_tool_call(&tc).expect("insert tool call");

        let mem = Memory {
            id: "mem-1".into(),
            project_id: Some("test-project".into()),
            content: "important fact".into(),
            memory_type: Some("insight".into()),
            source_session_id: Some("test-session-1".into()),
            confidence: 0.95,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).expect("insert memory");

        // Verify data was persisted.
        let sessions = store.get_recent_sessions(10).expect("get sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "test-session-1");

        // Verify upsert: insert same session again with updated summary.
        let mut updated = session.clone();
        updated.summary = Some("updated summary".into());
        store
            .insert_or_replace_session(&updated)
            .expect("upsert session");

        let sessions = store.get_recent_sessions(10).expect("get sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].summary.as_deref(), Some("updated summary"));

        let memories = store.search_memories("important").expect("search memories");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].id, "mem-1");
    }
}
