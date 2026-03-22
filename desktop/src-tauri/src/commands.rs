use std::path::PathBuf;

use remembrant_engine::store::DuckStore;
use remembrant_engine::AppConfig;
use serde_json::Value;

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn open_store() -> Result<DuckStore, String> {
    let config = AppConfig::load().map_err(|e| format!("Config error: {e}"))?;
    let db_path = expand_tilde(&config.storage.duckdb_path);
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory: {e}"))?;
    }
    DuckStore::open(&db_path).map_err(|e| format!("DuckDB error: {e}"))
}

#[tauri::command]
pub fn get_stats() -> Result<Value, String> {
    let store = open_store()?;
    let sessions = store.count_sessions().map_err(|e| e.to_string())?;
    let memories = store.count_memories().map_err(|e| e.to_string())?;
    let decisions = store.count_decisions().map_err(|e| e.to_string())?;
    let tool_calls = store.count_tool_calls().map_err(|e| e.to_string())?;
    let projects = store.get_project_ids().map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "sessions": sessions,
        "memories": memories,
        "decisions": decisions,
        "tool_calls": tool_calls,
        "projects": projects.len(),
    }))
}

#[tauri::command]
pub fn get_sessions(limit: Option<usize>, project: Option<String>, agent: Option<String>) -> Result<Value, String> {
    let store = open_store()?;
    let limit = limit.unwrap_or(100);
    let sessions = store
        .search_sessions(agent.as_deref(), project.as_deref(), None, limit)
        .map_err(|e| e.to_string())?;
    serde_json::to_value(&sessions).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_session_detail(id: String) -> Result<Value, String> {
    let store = open_store()?;
    let sessions = store
        .search_sessions(None, None, None, 10_000)
        .map_err(|e| e.to_string())?;

    let session = sessions
        .into_iter()
        .find(|s| s.id == id)
        .ok_or_else(|| "Session not found".to_string())?;

    let tool_calls = store
        .get_tool_calls_for_session(&id)
        .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "session": session,
        "tool_calls": tool_calls,
    }))
}

#[tauri::command]
pub fn get_memories(project: Option<String>, limit: Option<usize>) -> Result<Value, String> {
    let store = open_store()?;
    let memories = store
        .get_memories(project.as_deref(), limit.unwrap_or(100))
        .map_err(|e| e.to_string())?;
    serde_json::to_value(&memories).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_decisions(project: Option<String>) -> Result<Value, String> {
    let store = open_store()?;
    let decisions = store
        .get_decisions(project.as_deref(), 100)
        .map_err(|e| e.to_string())?;
    serde_json::to_value(&decisions).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_projects() -> Result<Value, String> {
    let store = open_store()?;
    let projects = store.get_project_ids().map_err(|e| e.to_string())?;
    serde_json::to_value(&projects).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn search_sessions(query: String) -> Result<Value, String> {
    let store = open_store()?;
    let sessions = store
        .search_sessions_by_summary(&query)
        .map_err(|e| e.to_string())?;
    serde_json::to_value(&sessions).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn search_memories(query: String) -> Result<Value, String> {
    let store = open_store()?;
    let memories = store
        .search_memories(&query)
        .map_err(|e| e.to_string())?;
    serde_json::to_value(&memories).map_err(|e| e.to_string())
}
