use std::fmt;

use anyhow::{Context, Result};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::embedding::EmbedProvider;
use crate::store::duckdb::{Memory, Session, ToolCall};
use crate::store::lance::LanceStore;

// ---------------------------------------------------------------------------
// Granularity
// ---------------------------------------------------------------------------

/// Granularity levels for embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// Entire session summary.
    Session,
    /// Single user-assistant exchange.
    Exchange,
    /// A specific decision made.
    Decision,
    /// A code file change.
    CodeChange,
    /// A tool invocation.
    ToolCall,
    /// A memory / insight.
    Memory,
}

impl fmt::Display for Granularity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Session => "session",
            Self::Exchange => "exchange",
            Self::Decision => "decision",
            Self::CodeChange => "code_change",
            Self::ToolCall => "tool_call",
            Self::Memory => "memory",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// EmbedChunk
// ---------------------------------------------------------------------------

/// A text chunk ready to be embedded.
#[derive(Debug, Clone)]
pub struct EmbedChunk {
    pub id: String,
    pub content: String,
    pub granularity: Granularity,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub file_path: Option<String>,
    pub language: Option<String>,
}

// ---------------------------------------------------------------------------
// EmbedStats
// ---------------------------------------------------------------------------

/// Stats from an embedding run.
#[derive(Debug, Default)]
pub struct EmbedStats {
    pub chunks_processed: usize,
    pub chunks_embedded: usize,
    pub chunks_stored: usize,
    pub errors: usize,
}

impl fmt::Display for EmbedStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "processed={}, embedded={}, stored={}, errors={}",
            self.chunks_processed, self.chunks_embedded, self.chunks_stored, self.errors,
        )
    }
}

// ---------------------------------------------------------------------------
// EmbedPipeline
// ---------------------------------------------------------------------------

/// The embedding pipeline: extracts chunks from ingested data, embeds them
/// via a pluggable [`EmbedProvider`], and stores the results in LanceDB.
pub struct EmbedPipeline {
    lance_store: LanceStore,
    batch_size: usize,
}

impl EmbedPipeline {
    /// Create a new pipeline.
    pub fn new(lance_store: LanceStore, batch_size: usize) -> Self {
        Self {
            lance_store,
            batch_size: batch_size.max(1),
        }
    }

    // -------------------------------------------------------------------
    // Chunk extraction
    // -------------------------------------------------------------------

    /// Extract embeddable chunks from sessions at multiple granularities.
    ///
    /// For each session we produce:
    /// - One `Session`-level chunk (summary + metadata).
    /// - One `CodeChange` chunk per changed file.
    pub fn extract_session_chunks(sessions: &[Session]) -> Vec<EmbedChunk> {
        let mut chunks = Vec::new();

        for session in sessions {
            // Session-level summary chunk
            let summary = session.summary.as_deref().unwrap_or("(no summary)");
            let files_str = if session.files_changed.is_empty() {
                "(none)".to_string()
            } else {
                session.files_changed.join(", ")
            };

            let content = format!(
                "Session: {summary}\n\
                 Agent: {agent}, Project: {project}\n\
                 Messages: {msgs}, Tools: {tools}\n\
                 Files: {files}",
                agent = session.agent,
                project = session.project_id.as_deref().unwrap_or("unknown"),
                msgs = session.message_count.unwrap_or(0),
                tools = session.tool_call_count.unwrap_or(0),
                files = files_str,
            );

            if !content.trim().is_empty() {
                chunks.push(EmbedChunk {
                    id: Uuid::new_v4().to_string(),
                    content,
                    granularity: Granularity::Session,
                    project_id: session.project_id.clone(),
                    session_id: Some(session.id.clone()),
                    file_path: None,
                    language: None,
                });
            }

            // One CodeChange chunk per changed file
            for file in &session.files_changed {
                let content = format!(
                    "File changed: {file}\nSession: {summary}\nAgent: {agent}",
                    agent = session.agent,
                );
                chunks.push(EmbedChunk {
                    id: Uuid::new_v4().to_string(),
                    content,
                    granularity: Granularity::CodeChange,
                    project_id: session.project_id.clone(),
                    session_id: Some(session.id.clone()),
                    file_path: Some(file.clone()),
                    language: None,
                });
            }
        }

        chunks
    }

    /// Extract embeddable chunks from memories.
    pub fn extract_memory_chunks(memories: &[Memory]) -> Vec<EmbedChunk> {
        memories
            .iter()
            .filter(|m| !m.content.trim().is_empty())
            .map(|m| {
                let mem_type = m.memory_type.as_deref().unwrap_or("general");
                let content = format!("[{mem_type}] {}", m.content);
                EmbedChunk {
                    id: Uuid::new_v4().to_string(),
                    content,
                    granularity: Granularity::Memory,
                    project_id: m.project_id.clone(),
                    session_id: m.source_session_id.clone(),
                    file_path: None,
                    language: None,
                }
            })
            .collect()
    }

    /// Extract embeddable chunks from tool calls.
    pub fn extract_tool_call_chunks(tool_calls: &[ToolCall]) -> Vec<EmbedChunk> {
        tool_calls
            .iter()
            .filter(|tc| tc.tool_name.is_some() || tc.command.is_some())
            .map(|tc| {
                let tool_name = tc.tool_name.as_deref().unwrap_or("unknown");
                let command = tc.command.as_deref().unwrap_or("");
                let content = format!("Tool: {tool_name}\nCommand: {command}");
                EmbedChunk {
                    id: Uuid::new_v4().to_string(),
                    content,
                    granularity: Granularity::ToolCall,
                    project_id: None,
                    session_id: tc.session_id.clone(),
                    file_path: None,
                    language: None,
                }
            })
            .collect()
    }

    // -------------------------------------------------------------------
    // Embed + store
    // -------------------------------------------------------------------

    /// Embed and store chunks. Processes in batches of `self.batch_size`.
    pub async fn embed_and_store(
        &self,
        chunks: &[EmbedChunk],
        provider: &impl EmbedProvider,
    ) -> Result<EmbedStats> {
        let mut stats = EmbedStats::default();

        // Filter out empty content up front
        let non_empty: Vec<&EmbedChunk> = chunks
            .iter()
            .filter(|c| !c.content.trim().is_empty())
            .collect();

        stats.chunks_processed = non_empty.len();
        info!(
            chunks = non_empty.len(),
            batch_size = self.batch_size,
            "starting embed_and_store"
        );

        for batch in non_empty.chunks(self.batch_size) {
            // Build the &[&str] slice that EmbedProvider expects.
            let texts: Vec<&str> = batch.iter().map(|c| c.content.as_str()).collect();

            let embeddings = match provider.embed_texts(&texts).await {
                Ok(embs) => embs,
                Err(e) => {
                    warn!(error = %e, batch_len = batch.len(), "embedding batch failed");
                    stats.errors += batch.len();
                    continue;
                }
            };

            if embeddings.len() != batch.len() {
                warn!(
                    expected = batch.len(),
                    got = embeddings.len(),
                    "embedding count mismatch, skipping batch"
                );
                stats.errors += batch.len();
                continue;
            }

            stats.chunks_embedded += embeddings.len();

            for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
                let store_result = match chunk.granularity {
                    Granularity::Memory => {
                        self.lance_store
                            .insert_memory_embedding(
                                &chunk.id,
                                embedding,
                                &chunk.content,
                                &chunk.granularity.to_string(),
                                chunk.project_id.as_deref().unwrap_or("unknown"),
                            )
                            .await
                    }
                    _ => {
                        self.lance_store
                            .insert_code_embedding(
                                &chunk.id,
                                embedding,
                                &chunk.content,
                                &chunk.granularity.to_string(),
                                chunk.project_id.as_deref().unwrap_or("unknown"),
                                chunk.file_path.as_deref(),
                                chunk.language.as_deref(),
                            )
                            .await
                    }
                };

                match store_result {
                    Ok(()) => {
                        stats.chunks_stored += 1;
                        debug!(id = %chunk.id, granularity = %chunk.granularity, "stored chunk");
                    }
                    Err(e) => {
                        warn!(id = %chunk.id, error = %e, "failed to store chunk");
                        stats.errors += 1;
                    }
                }
            }
        }

        info!(%stats, "embed_and_store complete");
        Ok(stats)
    }

    /// Full pipeline: extract chunks from all data, embed, and store.
    pub async fn run(
        &self,
        sessions: &[Session],
        memories: &[Memory],
        tool_calls: &[ToolCall],
        provider: &impl EmbedProvider,
    ) -> Result<EmbedStats> {
        info!(
            sessions = sessions.len(),
            memories = memories.len(),
            tool_calls = tool_calls.len(),
            "running embed pipeline"
        );

        let mut all_chunks = Vec::new();
        all_chunks.extend(Self::extract_session_chunks(sessions));
        all_chunks.extend(Self::extract_memory_chunks(memories));
        all_chunks.extend(Self::extract_tool_call_chunks(tool_calls));

        info!(total_chunks = all_chunks.len(), "extracted chunks");

        self.embed_and_store(&all_chunks, provider)
            .await
            .context("embed_and_store failed")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbedder;
    use chrono::Utc;

    fn make_session(id: &str, summary: &str, files: Vec<&str>) -> Session {
        Session {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            agent: "claude".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(10),
            message_count: Some(5),
            tool_call_count: Some(3),
            total_tokens: Some(1200),
            files_changed: files.into_iter().map(String::from).collect(),
            summary: Some(summary.into()),
        }
    }

    fn make_memory(id: &str, content: &str, mem_type: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            content: content.into(),
            memory_type: Some(mem_type.into()),
            source_session_id: None,
            confidence: 0.9,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        }
    }

    fn make_tool_call(id: &str, tool: &str, cmd: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            session_id: Some("s-1".into()),
            tool_name: Some(tool.into()),
            command: Some(cmd.into()),
            success: Some(true),
            error_message: None,
            duration_ms: Some(100),
            timestamp: Some(Utc::now().naive_utc()),
        }
    }

    // -----------------------------------------------------------------------
    // Extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_session_chunks() {
        let sessions = vec![
            make_session(
                "s-1",
                "Refactored auth module",
                vec!["src/auth.rs", "src/main.rs"],
            ),
            make_session("s-2", "Fixed bug in parser", vec!["src/parser.rs"]),
        ];

        let chunks = EmbedPipeline::extract_session_chunks(&sessions);

        // s-1: 1 session chunk + 2 code-change chunks = 3
        // s-2: 1 session chunk + 1 code-change chunk  = 2
        assert_eq!(chunks.len(), 5);

        // Verify session-level chunk content
        let session_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.granularity == Granularity::Session)
            .collect();
        assert_eq!(session_chunks.len(), 2);
        assert!(session_chunks[0].content.contains("Refactored auth module"));
        assert!(session_chunks[0].content.contains("Agent: claude"));
        assert!(session_chunks[0].content.contains("Messages: 5"));

        // Verify code-change chunks
        let code_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.granularity == Granularity::CodeChange)
            .collect();
        assert_eq!(code_chunks.len(), 3);
        assert!(
            code_chunks
                .iter()
                .any(|c| c.file_path.as_deref() == Some("src/auth.rs"))
        );
        assert!(
            code_chunks
                .iter()
                .any(|c| c.file_path.as_deref() == Some("src/parser.rs"))
        );
    }

    #[test]
    fn test_extract_memory_chunks() {
        let memories = vec![
            make_memory("m-1", "DuckDB is used for structured storage", "insight"),
            make_memory("m-2", "LanceDB handles embeddings", "pattern"),
            make_memory("m-3", "", "empty"), // empty content should be skipped
        ];

        let chunks = EmbedPipeline::extract_memory_chunks(&memories);

        assert_eq!(chunks.len(), 2); // empty one skipped
        assert!(chunks[0].content.starts_with("[insight]"));
        assert!(chunks[0].content.contains("DuckDB"));
        assert!(chunks[1].content.starts_with("[pattern]"));
        assert_eq!(chunks[0].granularity, Granularity::Memory);
    }

    #[test]
    fn test_extract_tool_call_chunks() {
        let tool_calls = vec![
            make_tool_call("tc-1", "bash", "cargo build"),
            make_tool_call("tc-2", "read", "/src/main.rs"),
        ];

        let chunks = EmbedPipeline::extract_tool_call_chunks(&tool_calls);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("Tool: bash"));
        assert!(chunks[0].content.contains("Command: cargo build"));
        assert_eq!(chunks[0].granularity, Granularity::ToolCall);
    }

    // -----------------------------------------------------------------------
    // Integration test with mock embedder + real LanceDB (temp dir)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_embed_and_store_with_mock() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let dim = 8;
        let lance = LanceStore::open_with_dim(tmp.path().join("lance"), dim)
            .await
            .expect("open LanceStore");

        let pipeline = EmbedPipeline::new(lance, 2);
        let embedder = MockEmbedder::new(dim as usize);

        let sessions = vec![make_session(
            "s-1",
            "Added embed pipeline",
            vec!["src/embed_pipeline.rs"],
        )];
        let memories = vec![make_memory("m-1", "Pipeline design is modular", "insight")];
        let tool_calls = vec![make_tool_call("tc-1", "write", "embed_pipeline.rs")];

        let stats = pipeline
            .run(&sessions, &memories, &tool_calls, &embedder)
            .await
            .expect("pipeline run");

        // session chunk + code-change chunk + memory chunk + tool-call chunk = 4
        assert_eq!(stats.chunks_processed, 4);
        assert_eq!(stats.chunks_embedded, 4);
        assert_eq!(stats.chunks_stored, 4);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_empty_inputs_produce_no_chunks() {
        assert!(EmbedPipeline::extract_session_chunks(&[]).is_empty());
        assert!(EmbedPipeline::extract_memory_chunks(&[]).is_empty());
        assert!(EmbedPipeline::extract_tool_call_chunks(&[]).is_empty());
    }

    #[test]
    fn test_tool_call_without_name_or_command_skipped() {
        let tc = ToolCall {
            id: "tc-empty".into(),
            session_id: None,
            tool_name: None,
            command: None,
            success: None,
            error_message: None,
            duration_ms: None,
            timestamp: None,
        };
        let chunks = EmbedPipeline::extract_tool_call_chunks(&[tc]);
        assert!(chunks.is_empty());
    }
}
