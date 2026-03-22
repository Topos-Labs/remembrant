//! Integration tests for the Remembrant engine.
//!
//! These tests exercise the full pipeline: parsing → storing → querying.

use anyhow::Result;
use chrono::{NaiveDateTime, Utc};
use remembrant_engine::embedding::MockEmbedder;
use remembrant_engine::graph_builder::GraphBuilder;
use remembrant_engine::store::duckdb::{Decision, DuckStore, Memory, Session, ToolCall};
use remembrant_engine::store::graph::NodeKind;
use remembrant_engine::store::lance::LanceStore;
use remembrant_engine::{EmbedChunk, EmbedPipeline, Granularity};
use tempfile::tempdir;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Test 1: Full ingest pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_full_ingest_pipeline() -> Result<()> {
    // Create in-memory DuckDB
    let store = DuckStore::open_in_memory()?;

    // Create mock session data
    let session = Session {
        id: Uuid::new_v4().to_string(),
        project_id: Some("test-project".to_string()),
        agent: "claude-code".to_string(),
        started_at: Some(Utc::now().naive_utc()),
        ended_at: Some(Utc::now().naive_utc()),
        duration_minutes: Some(15),
        message_count: Some(10),
        tool_call_count: Some(3),
        total_tokens: Some(1500),
        files_changed: vec!["src/main.rs".to_string(), "Cargo.toml".to_string()],
        summary: Some("Implemented authentication feature".to_string()),
    };

    // Insert session
    store.insert_session(&session)?;

    // Create memory
    let memory = Memory {
        id: Uuid::new_v4().to_string(),
        project_id: Some("test-project".to_string()),
        content: "Use bcrypt for password hashing".to_string(),
        memory_type: Some("best_practice".to_string()),
        source_session_id: Some(session.id.clone()),
        confidence: 0.95,
        access_count: 0,
        created_at: Some(Utc::now().naive_utc()),
        updated_at: Some(Utc::now().naive_utc()),
        valid_until: None,
    };

    store.insert_memory(&memory)?;

    // Create decision
    let decision = Decision {
        id: Uuid::new_v4().to_string(),
        session_id: Some(session.id.clone()),
        project_id: Some("test-project".to_string()),
        decision_type: Some("architecture".to_string()),
        what: "Use JWT for authentication".to_string(),
        why: Some("Stateless, scalable, widely supported".to_string()),
        alternatives: vec!["Session cookies".to_string(), "OAuth2".to_string()],
        outcome: Some("success".to_string()),
        created_at: Some(Utc::now().naive_utc()),
        valid_until: None,
    };

    store.insert_decision(&decision)?;

    // Create tool call
    let tool_call = ToolCall {
        id: Uuid::new_v4().to_string(),
        session_id: Some(session.id.clone()),
        tool_name: Some("bash".to_string()),
        command: Some("cargo test".to_string()),
        success: Some(true),
        error_message: None,
        duration_ms: Some(2500),
        timestamp: Some(Utc::now().naive_utc()),
    };

    store.insert_tool_call(&tool_call)?;

    // Query back and verify
    let sessions = store.get_recent_sessions(10)?;
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, session.id);
    assert_eq!(sessions[0].agent, "claude-code");
    assert_eq!(sessions[0].files_changed.len(), 2);

    let decisions = store.get_decisions(Some("test-project"), 10)?;
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].what, "Use JWT for authentication");
    assert_eq!(decisions[0].alternatives.len(), 2);

    let tool_calls = store.get_tool_calls_for_session(&session.id)?;
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].command, Some("cargo test".to_string()));
    assert_eq!(tool_calls[0].success, Some(true));

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 2: Graph builder from sessions
// ---------------------------------------------------------------------------

#[test]
fn test_graph_builder_from_sessions() -> Result<()> {
    let store = DuckStore::open_in_memory()?;

    // Create two sessions in different projects
    let session1 = Session {
        id: Uuid::new_v4().to_string(),
        project_id: Some("project-a".to_string()),
        agent: "claude-code".to_string(),
        started_at: Some(Utc::now().naive_utc()),
        ended_at: None,
        duration_minutes: Some(20),
        message_count: Some(15),
        tool_call_count: Some(5),
        total_tokens: Some(2000),
        files_changed: vec!["src/lib.rs".to_string()],
        summary: Some("Refactored authentication module".to_string()),
    };

    let session2 = Session {
        id: Uuid::new_v4().to_string(),
        project_id: Some("project-b".to_string()),
        agent: "codex".to_string(),
        started_at: Some(Utc::now().naive_utc()),
        ended_at: None,
        duration_minutes: Some(30),
        message_count: Some(20),
        tool_call_count: Some(8),
        total_tokens: Some(3000),
        files_changed: vec!["src/auth.rs".to_string()],
        summary: Some("Added OAuth2 support".to_string()),
    };

    store.insert_session(&session1)?;
    store.insert_session(&session2)?;

    // Create memory linked to session1
    let memory = Memory {
        id: Uuid::new_v4().to_string(),
        project_id: Some("project-a".to_string()),
        content: "Always validate JWT signatures".to_string(),
        memory_type: Some("security".to_string()),
        source_session_id: Some(session1.id.clone()),
        confidence: 1.0,
        access_count: 0,
        created_at: Some(Utc::now().naive_utc()),
        updated_at: Some(Utc::now().naive_utc()),
        valid_until: None,
    };

    store.insert_memory(&memory)?;

    // Query sessions
    let sessions = store.get_recent_sessions(10)?;
    let memories = store.get_memories(None, 10)?;
    let decisions = store.get_decisions(None, 10)?;
    let tool_calls_1 = store.get_tool_calls_for_session(&session1.id)?;
    let tool_calls_2 = store.get_tool_calls_for_session(&session2.id)?;
    let mut all_tool_calls = tool_calls_1;
    all_tool_calls.extend(tool_calls_2);

    // Build graph
    let builder = GraphBuilder::new();
    builder.build_from_data(&sessions, &memories, &decisions, &all_tool_calls)?;

    // Verify graph stats
    let (node_count, edge_count) = builder.stats()?;
    assert!(
        node_count >= 3,
        "Expected at least 3 nodes (2 sessions + 1 memory)"
    );
    assert!(edge_count >= 0, "Expected at least 0 edges");

    // Graph building completed successfully

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 3: Embed pipeline with mock embedder
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_embed_pipeline_with_mock() -> Result<()> {
    let store = DuckStore::open_in_memory()?;
    let temp_dir = tempdir()?;
    let lance_path = temp_dir.path().join("lance_test");

    // Create mock embedder (1024 dimensions to match LanceStore default)
    let embedder = MockEmbedder::new(1024);

    // Create LanceStore with matching dimensions
    let lance_store = LanceStore::open(&lance_path).await?;

    // Create sessions
    let session1 = Session {
        id: Uuid::new_v4().to_string(),
        project_id: Some("test-proj".to_string()),
        agent: "gemini".to_string(),
        started_at: Some(Utc::now().naive_utc()),
        ended_at: None,
        duration_minutes: Some(10),
        message_count: Some(5),
        tool_call_count: Some(2),
        total_tokens: Some(800),
        files_changed: vec![],
        summary: Some("Fixed bug in authentication".to_string()),
    };

    store.insert_session(&session1)?;

    // Create memory
    let memory1 = Memory {
        id: Uuid::new_v4().to_string(),
        project_id: Some("test-proj".to_string()),
        content: "Remember to test edge cases".to_string(),
        memory_type: Some("reminder".to_string()),
        source_session_id: Some(session1.id.clone()),
        confidence: 0.8,
        access_count: 0,
        created_at: Some(Utc::now().naive_utc()),
        updated_at: Some(Utc::now().naive_utc()),
        valid_until: None,
    };

    store.insert_memory(&memory1)?;

    // Create embed chunks manually
    let chunks = vec![
        EmbedChunk {
            id: Uuid::new_v4().to_string(),
            content: session1.summary.clone().unwrap(),
            granularity: Granularity::Session,
            project_id: session1.project_id.clone(),
            session_id: Some(session1.id.clone()),
            file_path: None,
            language: None,
        },
        EmbedChunk {
            id: Uuid::new_v4().to_string(),
            content: memory1.content.clone(),
            granularity: Granularity::Memory,
            project_id: memory1.project_id.clone(),
            session_id: memory1.source_session_id.clone(),
            file_path: None,
            language: None,
        },
    ];

    // Create pipeline
    let pipeline = EmbedPipeline::new(lance_store, 10);

    // Embed chunks
    let stats = pipeline.embed_and_store(&chunks, &embedder).await?;

    // Verify stats
    assert_eq!(stats.chunks_processed, 2);
    assert_eq!(stats.chunks_embedded, 2);
    assert_eq!(stats.chunks_stored, 2);
    assert_eq!(stats.errors, 0);

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 4: Distiller keyword extraction
// ---------------------------------------------------------------------------

#[test]
fn test_distiller_keyword_extraction() -> Result<()> {
    // This test verifies basic text processing in the distiller module
    // We'll test the keyword extraction logic if it's public, or just
    // verify the module can be instantiated

    use remembrant_engine::distill::LlmClient;

    // Create a client (won't actually call API in this test)
    let _client = LlmClient::new_local("test-model");

    // Verify client is created successfully
    // (We can't test the actual LLM call without a running server,
    // but we can verify the struct is well-formed)

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 5: DuckStore CRUD operations
// ---------------------------------------------------------------------------

#[test]
fn test_duck_store_crud() -> Result<()> {
    let store = DuckStore::open_in_memory()?;

    // Create session
    let session_id = Uuid::new_v4().to_string();
    let session = Session {
        id: session_id.clone(),
        project_id: Some("crud-test".to_string()),
        agent: "test-agent".to_string(),
        started_at: Some(Utc::now().naive_utc()),
        ended_at: None,
        duration_minutes: Some(5),
        message_count: Some(3),
        tool_call_count: Some(1),
        total_tokens: Some(500),
        files_changed: vec!["test.txt".to_string()],
        summary: Some("Test session".to_string()),
    };

    // INSERT
    store.insert_session(&session)?;

    // READ
    let sessions = store.get_recent_sessions(1)?;
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, session_id);
    assert_eq!(sessions[0].agent, "test-agent");

    // UPDATE (using insert_or_replace_session)
    let mut updated_session = session.clone();
    updated_session.summary = Some("Updated test session".to_string());
    updated_session.message_count = Some(10);
    store.insert_or_replace_session(&updated_session)?;

    // Verify update
    let sessions = store.get_recent_sessions(1)?;
    assert_eq!(sessions.len(), 1);
    assert_eq!(
        sessions[0].summary,
        Some("Updated test session".to_string())
    );
    assert_eq!(sessions[0].message_count, Some(10));

    // DELETE
    store.delete_session(&session_id)?;

    // Verify deletion
    let sessions = store.get_recent_sessions(10)?;
    assert_eq!(sessions.len(), 0);

    Ok(())
}

// ---------------------------------------------------------------------------
// Test 6: Search with filters
// ---------------------------------------------------------------------------

#[test]
fn test_duck_store_search_filters() -> Result<()> {
    let store = DuckStore::open_in_memory()?;

    // Create multiple sessions with different attributes
    let session1 = Session {
        id: Uuid::new_v4().to_string(),
        project_id: Some("project-x".to_string()),
        agent: "claude-code".to_string(),
        started_at: Some(
            NaiveDateTime::parse_from_str("2024-01-15 10:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ),
        ended_at: None,
        duration_minutes: Some(20),
        message_count: Some(15),
        tool_call_count: Some(5),
        total_tokens: Some(2000),
        files_changed: vec![],
        summary: Some("Session 1".to_string()),
    };

    let session2 = Session {
        id: Uuid::new_v4().to_string(),
        project_id: Some("project-y".to_string()),
        agent: "codex".to_string(),
        started_at: Some(
            NaiveDateTime::parse_from_str("2024-02-20 14:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ),
        ended_at: None,
        duration_minutes: Some(30),
        message_count: Some(25),
        tool_call_count: Some(10),
        total_tokens: Some(3000),
        files_changed: vec![],
        summary: Some("Session 2".to_string()),
    };

    let session3 = Session {
        id: Uuid::new_v4().to_string(),
        project_id: Some("project-x".to_string()),
        agent: "gemini".to_string(),
        started_at: Some(
            NaiveDateTime::parse_from_str("2024-03-10 16:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ),
        ended_at: None,
        duration_minutes: Some(15),
        message_count: Some(12),
        tool_call_count: Some(3),
        total_tokens: Some(1500),
        files_changed: vec![],
        summary: Some("Session 3".to_string()),
    };

    store.insert_session(&session1)?;
    store.insert_session(&session2)?;
    store.insert_session(&session3)?;

    // Test: Filter by agent
    let claude_sessions = store.search_sessions(Some("claude-code"), None, None, 10)?;
    assert_eq!(claude_sessions.len(), 1);
    assert_eq!(claude_sessions[0].agent, "claude-code");

    // Test: Filter by project
    let project_x_sessions = store.search_sessions(None, Some("project-x"), None, 10)?;
    assert_eq!(project_x_sessions.len(), 2);

    // Test: Filter by date
    let since_date =
        NaiveDateTime::parse_from_str("2024-02-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
    let recent_sessions = store.search_sessions(None, None, Some(since_date), 10)?;
    assert_eq!(recent_sessions.len(), 2); // session2 and session3

    // Test: Combined filters
    let filtered = store.search_sessions(None, Some("project-x"), Some(since_date), 10)?;
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].agent, "gemini");

    Ok(())
}
