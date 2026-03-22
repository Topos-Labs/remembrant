// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::get_stats,
            commands::get_sessions,
            commands::get_session_detail,
            commands::get_memories,
            commands::get_decisions,
            commands::get_projects,
            commands::search_sessions,
            commands::search_memories,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Remembrant");
}
