use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use clap::{Parser, Subcommand};
use remembrant_engine::distill::Distiller;
use remembrant_engine::embed_pipeline::EmbedPipeline;
use remembrant_engine::embedding::{EmbedProvider, LmStudioEmbedder};
use remembrant_engine::graph_builder::{self, GraphBackend, GraphBuilder};
use remembrant_engine::repo_embed::RepoEmbedder;
use remembrant_engine::store::{DuckStore, LanceStore};
#[cfg(feature = "code-analysis")]
use remembrant_engine::store::GraphStoreBackend;
use remembrant_engine::{AppConfig, ClaudeIngester, CodexIngester, GeminiIngester, detect_agents};

#[derive(Parser)]
#[command(
    name = "rem",
    about = "Remembrant: shared persistent memory for coding agents"
)]
#[command(version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize remembrant, create config, scan agents
    Init,

    /// Start file watcher daemon
    Watch,

    /// Stop daemon
    Stop,

    /// Semantic search across sessions
    Search {
        /// Search query
        query: String,

        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Filter by agent
        #[arg(long)]
        agent: Option<String>,

        /// Filter by date (e.g. "2d", "1w", ISO date)
        #[arg(long)]
        since: Option<String>,

        /// Filter by content type
        #[arg(long, name = "type")]
        content_type: Option<String>,

        /// Use exact matching instead of semantic
        #[arg(long)]
        exact: bool,
    },

    /// Exact text match
    Find {
        /// Text to find
        query: String,
    },

    /// Recent sessions
    Recent {
        /// Maximum number of results
        #[arg(long, default_value_t = 20)]
        limit: usize,

        /// Filter by agent
        #[arg(long)]
        agent: Option<String>,

        /// Filter by project
        #[arg(long)]
        project: Option<String>,
    },

    /// Daily context briefing
    Brief {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Only show today's activity
        #[arg(long)]
        today: bool,
    },

    /// Cross-project patterns
    Patterns {
        /// Optional topic to focus on
        topic: Option<String>,
    },

    /// Decision journal
    Decisions {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Show all decisions (not just recent)
        #[arg(long)]
        all: bool,
    },

    /// Find related content for a file
    Related {
        /// File path to find related content for
        path: String,
    },

    /// Dependency graph for a file
    Graph {
        /// File path to graph
        path: String,
    },

    /// Chronological view of a topic
    Timeline {
        /// Topic to view
        topic: String,

        /// Filter by date
        #[arg(long)]
        since: Option<String>,
    },

    /// Quick manual note
    Note {
        /// Note text
        text: String,

        /// Associate with project
        #[arg(long)]
        project: Option<String>,

        /// Add a tag
        #[arg(long)]
        tag: Option<Vec<String>>,
    },

    /// Remove a session
    Forget {
        /// Session ID to remove
        #[arg(long)]
        session: String,
    },

    /// Generate agent memory files
    Export {
        /// Filter by project
        #[arg(long)]
        project: Option<String>,

        /// Output format (e.g. markdown, json)
        #[arg(long, default_value = "markdown")]
        format: String,

        /// Output path
        #[arg(long)]
        output: Option<String>,
    },

    /// Embed a repository
    Embed {
        /// Repository path
        path: String,

        /// Update existing embeddings
        #[arg(long)]
        update: bool,
    },

    /// Full ingest pipeline: parse agents → DuckDB → embed → distill → graph
    Ingest {
        /// Skip embedding step (no LM Studio needed)
        #[arg(long)]
        skip_embed: bool,

        /// Skip LLM distillation step
        #[arg(long)]
        skip_distill: bool,
    },

    /// Daemon and DB status
    Status,

    /// Analytics
    Stats,

    /// Garbage collect old/orphaned data
    Gc,

    /// Deep code analysis using Infiniloom AST parsing
    Analyze {
        /// Repository path to analyze
        path: String,

        /// Project ID (default: directory name)
        #[arg(long)]
        project: Option<String>,
    },

    /// Search using Semantic XPath query
    #[command(name = "xpath")]
    XPath {
        /// The XPath query (e.g., '//Session[node~"auth"]/Decision')
        query: String,

        /// Maximum tree depth to load (default: 4)
        #[arg(long, default_value = "4")]
        depth: usize,

        /// Maximum results to show
        #[arg(long, short, default_value = "20")]
        limit: usize,

        /// Show tree structure of results
        #[arg(long)]
        tree: bool,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Expand a leading `~/` to the actual home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

/// Open DuckDB at the configured (tilde-expanded) path.
fn open_store(config: &AppConfig) -> Result<DuckStore> {
    let db_path = expand_tilde(&config.storage.duckdb_path);
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    DuckStore::open(&db_path)
}

/// Return the path to the PID file.
fn pid_file_path() -> Result<PathBuf> {
    let dir = AppConfig::config_dir()?;
    Ok(dir.join("daemon.pid"))
}

/// Format a file size in human-readable form.
fn human_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Parse relative date strings like "2d", "1w", or ISO dates.
fn parse_since(s: &str) -> Option<NaiveDateTime> {
    let now = chrono::Utc::now().naive_utc();
    if let Some(days) = s.strip_suffix('d') {
        let n: i64 = days.parse().ok()?;
        return Some(now - chrono::Duration::days(n));
    }
    if let Some(weeks) = s.strip_suffix('w') {
        let n: i64 = weeks.parse().ok()?;
        return Some(now - chrono::Duration::weeks(n));
    }
    if let Some(hours) = s.strip_suffix('h') {
        let n: i64 = hours.parse().ok()?;
        return Some(now - chrono::Duration::hours(n));
    }
    // Try ISO datetime first, then just date
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
        .ok()
        .or_else(|| {
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .ok()
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        })
}

/// Open LanceDB at the configured (tilde-expanded) path.
async fn open_lance_store(config: &AppConfig) -> Result<LanceStore> {
    let lance_path = expand_tilde(&config.storage.lancedb_path);
    std::fs::create_dir_all(&lance_path)
        .with_context(|| format!("failed to create directory {}", lance_path.display()))?;
    LanceStore::open_with_dim(&lance_path, config.embedding.dimensions as i32).await
}

/// Truncate a string to a maximum character width, appending "..." if truncated.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

fn cmd_init() -> Result<()> {
    println!("Remembrant: initializing...\n");

    // 1. Load or create config
    let config = AppConfig::load()?;
    let config_dir = AppConfig::config_dir()?;
    std::fs::create_dir_all(&config_dir)?;
    println!("[config] Configuration at {}", config_dir.display());

    // 2. Detect agents
    println!("\n--- Agent Detection ---");
    let detection = detect_agents();

    let mut agents_found = 0u32;
    if let Some(ref info) = detection.claude_code {
        println!(
            "  [+] {} -- {} ({} sessions)",
            info.name,
            info.base_path.display(),
            info.session_count
        );
        agents_found += 1;
    }
    if let Some(ref info) = detection.codex {
        println!(
            "  [+] {} -- {} ({} sessions)",
            info.name,
            info.base_path.display(),
            info.session_count
        );
        agents_found += 1;
    }
    if let Some(ref info) = detection.gemini {
        println!(
            "  [+] {} -- {} ({} sessions)",
            info.name,
            info.base_path.display(),
            info.session_count
        );
        agents_found += 1;
    }
    if agents_found == 0 {
        println!("  (no agents detected)");
    }

    // 3. Open DuckDB and init schema
    let db_path = expand_tilde(&config.storage.duckdb_path);
    println!("\n[storage] DuckDB at {}", db_path.display());
    let store = open_store(&config)?;
    println!("[storage] Schema initialized");

    // 4. Initial ingestion
    println!("\n--- Initial Ingestion ---");
    let mut total_sessions = 0usize;
    let mut total_memories = 0usize;
    let mut total_tool_calls = 0usize;

    // Claude Code
    if detection.claude_code.is_some() {
        match ClaudeIngester::new() {
            Ok(ingester) => match ingester.ingest_all() {
                Ok(result) => {
                    let s_count = result.sessions.len();
                    let m_count = result.memories.len();
                    let t_count = result.tool_calls.len();

                    for session in &result.sessions {
                        if let Err(e) = store.insert_session(session) {
                            eprintln!("  [!] Failed to insert Claude session {}: {e}", session.id);
                        }
                    }
                    for memory in &result.memories {
                        if let Err(e) = store.insert_memory(memory) {
                            eprintln!("  [!] Failed to insert Claude memory: {e}");
                        }
                    }

                    total_sessions += s_count;
                    total_memories += m_count;
                    total_tool_calls += t_count;
                    println!(
                        "  Claude Code: {s_count} sessions, {t_count} tool calls, {m_count} memories"
                    );
                }
                Err(e) => eprintln!("  [!] Claude Code ingestion error: {e}"),
            },
            Err(e) => eprintln!("  [!] Claude Code ingester init error: {e}"),
        }
    }

    // Codex
    if detection.codex.is_some() {
        if let Some(ingester) = CodexIngester::new() {
            match ingester.ingest_all() {
                Ok(result) => {
                    let s_count = result.sessions.len();
                    let m_count = result.memories.len();
                    let t_count = result.tool_calls.len();

                    for session in &result.sessions {
                        if let Err(e) = store.insert_session(session) {
                            eprintln!("  [!] Failed to insert Codex session {}: {e}", session.id);
                        }
                    }
                    for memory in &result.memories {
                        if let Err(e) = store.insert_memory(memory) {
                            eprintln!("  [!] Failed to insert Codex memory: {e}");
                        }
                    }

                    total_sessions += s_count;
                    total_memories += m_count;
                    total_tool_calls += t_count;
                    println!(
                        "  Codex CLI:   {s_count} sessions, {t_count} tool calls, {m_count} memories"
                    );
                }
                Err(e) => eprintln!("  [!] Codex ingestion error: {e}"),
            }
        }
    }

    // Gemini
    if detection.gemini.is_some() {
        if let Some(ingester) = GeminiIngester::new() {
            let (result, sessions, _tool_calls, memories) = ingester.ingest_all();

            for session in &sessions {
                if let Err(e) = store.insert_session(session) {
                    eprintln!("  [!] Failed to insert Gemini session {}: {e}", session.id);
                }
            }
            for memory in &memories {
                if let Err(e) = store.insert_memory(memory) {
                    eprintln!("  [!] Failed to insert Gemini memory: {e}");
                }
            }

            total_sessions += result.sessions_found;
            total_memories += result.memories_found;
            total_tool_calls += result.tool_calls_found;
            println!(
                "  Gemini CLI:  {} sessions, {} tool calls, {} memories",
                result.sessions_found, result.tool_calls_found, result.memories_found
            );
            if !result.errors.is_empty() {
                for err in &result.errors {
                    eprintln!("  [!] Gemini: {err}");
                }
            }
        }
    }

    println!("\n--- Summary ---");
    println!("  Agents detected:   {agents_found}");
    println!("  Sessions ingested: {total_sessions}");
    println!("  Tool calls found:  {total_tool_calls}");
    println!("  Memories ingested: {total_memories}");
    println!("\nInitialization complete. Run `rem status` to verify.");

    Ok(())
}

fn cmd_status() -> Result<()> {
    let config = AppConfig::load()?;
    let config_dir = AppConfig::config_dir()?;

    println!("Remembrant Status\n");

    // 1. Daemon status
    let pid_path = pid_file_path()?;
    if pid_path.exists() {
        let pid_str = std::fs::read_to_string(&pid_path).unwrap_or_default();
        let pid_str = pid_str.trim();
        // Check if process is running
        let running = std::process::Command::new("kill")
            .args(["-0", pid_str])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if running {
            println!("[daemon] Running (PID {pid_str})");
        } else {
            println!("[daemon] Stale PID file (process {pid_str} not running)");
        }
    } else {
        println!("[daemon] Not running");
    }

    // 2. Detected agents
    println!("\n--- Agents ---");
    let detection = detect_agents();
    let mut any_agent = false;
    if let Some(ref info) = detection.claude_code {
        println!(
            "  Claude Code: {} ({} sessions)",
            info.base_path.display(),
            info.session_count
        );
        any_agent = true;
    }
    if let Some(ref info) = detection.codex {
        println!(
            "  Codex CLI:   {} ({} sessions)",
            info.base_path.display(),
            info.session_count
        );
        any_agent = true;
    }
    if let Some(ref info) = detection.gemini {
        println!(
            "  Gemini CLI:  {} ({} sessions)",
            info.base_path.display(),
            info.session_count
        );
        any_agent = true;
    }
    if !any_agent {
        println!("  (no agents detected)");
    }

    // 3. Storage
    println!("\n--- Storage ---");
    println!("  Config dir:  {}", config_dir.display());

    let db_path = expand_tilde(&config.storage.duckdb_path);
    if db_path.exists() {
        let size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        println!(
            "  DuckDB:      {} ({})",
            db_path.display(),
            human_size(size)
        );

        // Row counts
        match open_store(&config) {
            Ok(store) => {
                let sessions = store.get_recent_sessions(usize::MAX)?;
                println!("  Sessions:    {}", sessions.len());

                let memories = store.search_memories("")?;
                println!("  Memories:    {}", memories.len());
            }
            Err(e) => {
                eprintln!("  [!] Could not open DuckDB: {e}");
            }
        }
    } else {
        println!("  DuckDB:      {} (not created yet)", db_path.display());
        println!("  Run `rem init` first.");
    }

    let lance_path = expand_tilde(&config.storage.lancedb_path);
    if lance_path.exists() {
        println!("  LanceDB:     {}", lance_path.display());
    }

    let graph_path = expand_tilde(&config.storage.graph_db_path);
    if graph_path.exists() {
        println!("  Graph DB:    {}", graph_path.display());
    }

    Ok(())
}

async fn cmd_watch() -> Result<()> {
    let config = AppConfig::load()?;

    // Write PID file
    let pid = std::process::id();
    let pid_path = pid_file_path()?;
    if let Some(parent) = pid_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&pid_path, pid.to_string())?;
    println!("[watch] PID {pid} written to {}", pid_path.display());

    let store = open_store(&config)?;
    let debounce_secs = std::cmp::max(config.watch.debounce_ms / 1000, 5);

    println!(
        "[watch] Watching for changes (polling every {debounce_secs}s). Press Ctrl+C to stop."
    );

    let pid_path_clone = pid_path.clone();
    let shutdown = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to listen for Ctrl+C");
    };

    let poll_loop = async {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(debounce_secs)).await;

            // Re-ingest Claude Code
            if let Ok(ingester) = ClaudeIngester::new() {
                match ingester.ingest_all() {
                    Ok(result) => {
                        if result.sessions_count > 0 {
                            let mut inserted = 0usize;
                            for session in &result.sessions {
                                // Insert, ignoring duplicates
                                if store.insert_session(session).is_ok() {
                                    inserted += 1;
                                }
                            }
                            for memory in &result.memories {
                                let _ = store.insert_memory(memory);
                            }
                            if inserted > 0 {
                                println!(
                                    "[watch] Claude Code: ingested {inserted} sessions, {} memories",
                                    result.memories_count
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[watch] Claude Code ingestion error: {e}");
                    }
                }
            }

            // Re-ingest Codex
            if let Some(ingester) = CodexIngester::new() {
                match ingester.ingest_all() {
                    Ok(result) => {
                        let mut inserted = 0usize;
                        for session in &result.sessions {
                            if store.insert_session(session).is_ok() {
                                inserted += 1;
                            }
                        }
                        for memory in &result.memories {
                            let _ = store.insert_memory(memory);
                        }
                        if inserted > 0 {
                            println!(
                                "[watch] Codex: ingested {inserted} sessions, {} memories",
                                result.memories.len()
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("[watch] Codex ingestion error: {e}");
                    }
                }
            }

            // Re-ingest Gemini
            if let Some(ingester) = GeminiIngester::new() {
                let (_result, sessions, _tool_calls, memories) = ingester.ingest_all();
                let mut inserted = 0usize;
                for session in &sessions {
                    if store.insert_session(session).is_ok() {
                        inserted += 1;
                    }
                }
                for memory in &memories {
                    let _ = store.insert_memory(memory);
                }
                if inserted > 0 {
                    println!(
                        "[watch] Gemini: ingested {inserted} sessions, {} memories",
                        memories.len()
                    );
                }
            }
        }
    };

    tokio::select! {
        _ = shutdown => {
            println!("\n[watch] Shutting down...");
        }
        _ = poll_loop => {}
    }

    // Cleanup PID file
    if pid_path_clone.exists() {
        let _ = std::fs::remove_file(&pid_path_clone);
        println!("[watch] PID file removed");
    }

    Ok(())
}

fn cmd_stop() -> Result<()> {
    let pid_path = pid_file_path()?;

    if !pid_path.exists() {
        println!("[stop] No daemon PID file found. Is the watcher running?");
        return Ok(());
    }

    let pid_str = std::fs::read_to_string(&pid_path).context("failed to read PID file")?;
    let pid_str = pid_str.trim();

    println!("[stop] Sending SIGTERM to PID {pid_str}...");

    let status = std::process::Command::new("kill")
        .args(["-TERM", pid_str])
        .status()
        .context("failed to send SIGTERM")?;

    if status.success() {
        println!("[stop] Signal sent successfully");
    } else {
        eprintln!("[stop] kill returned non-zero status (process may already be stopped)");
    }

    // Remove PID file
    if let Err(e) = std::fs::remove_file(&pid_path) {
        eprintln!("[stop] Warning: could not remove PID file: {e}");
    } else {
        println!("[stop] PID file removed");
    }

    println!("[stop] Daemon stopped.");
    Ok(())
}

fn cmd_recent(limit: usize) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;
    let sessions = store.get_recent_sessions(limit)?;

    if sessions.is_empty() {
        println!("No sessions found. Run `rem init` to ingest agent data.");
        return Ok(());
    }

    println!(
        "{:<36}  {:<12}  {:<20}  {:>5}  {:>5}  {}",
        "SESSION ID", "AGENT", "STARTED", "MSGS", "TOOLS", "SUMMARY"
    );
    println!("{}", "-".repeat(110));

    for s in &sessions {
        let started = s
            .started_at
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "-".to_string());
        let msgs = s
            .message_count
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let tools = s
            .tool_call_count
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let summary = s
            .summary
            .as_deref()
            .unwrap_or("-")
            .chars()
            .take(40)
            .collect::<String>();

        let id_display = if s.id.len() > 36 {
            format!("{}...", &s.id[..33])
        } else {
            s.id.clone()
        };

        println!(
            "{:<36}  {:<12}  {:<20}  {:>5}  {:>5}  {}",
            id_display, s.agent, started, msgs, tools, summary
        );
    }

    println!("\n{} session(s) shown", sessions.len());
    Ok(())
}

async fn cmd_search(
    query: &str,
    project: Option<&str>,
    agent: Option<&str>,
    since: Option<&str>,
    content_type: Option<&str>,
    exact: bool,
) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let since_dt = since.and_then(parse_since);

    if exact {
        // Exact search via DuckDB ILIKE
        let memories = store.search_memories(query)?;
        if memories.is_empty() {
            println!("No results found for exact query: {query}");
            return Ok(());
        }
        println!("Exact search results for \"{query}\":\n");
        let mut shown = 0usize;
        for m in &memories {
            // Apply optional filters
            if let Some(proj) = project {
                if let Some(ref pid) = m.project_id {
                    if !pid.to_lowercase().contains(&proj.to_lowercase()) {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            if let Some(ref ctype) = content_type {
                if let Some(ref mt) = m.memory_type {
                    if !mt.to_lowercase().contains(&ctype.to_lowercase()) {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            if let Some(since_dt) = since_dt {
                if let Some(ref created) = m.created_at {
                    if *created < since_dt {
                        continue;
                    }
                }
            }

            let mtype = m.memory_type.as_deref().unwrap_or("unknown");
            let proj = m.project_id.as_deref().unwrap_or("-");
            println!("  Memory ({mtype}) -- {proj}");
            println!("    {}", truncate(&m.content, 80));
            println!();
            shown += 1;
        }
        println!("{shown} result(s).");
        return Ok(());
    }

    // Semantic search: try LM Studio, fall back to exact
    let embedder = LmStudioEmbedder::from_config(&config.embedding);
    let embed_result = embedder.embed_texts(&[query]).await;

    match embed_result {
        Ok(vectors) if !vectors.is_empty() => {
            let query_vec = &vectors[0];
            let lance = open_lance_store(&config).await?;

            // Search memories
            let mem_results = lance.search_memories(query_vec, 10).await?;
            // Search code
            let code_results = lance.search_code(query_vec, 10).await?;

            // Combine into a unified list sorted by distance
            struct SearchHit {
                distance: f32,
                kind: String,
                label: String,
                project: String,
                content: String,
            }

            let mut hits: Vec<SearchHit> = Vec::new();

            for m in &mem_results {
                // Apply filters
                if let Some(proj) = project {
                    if !m.project_id.to_lowercase().contains(&proj.to_lowercase()) {
                        continue;
                    }
                }
                if let Some(ref ctype) = content_type {
                    if !m.memory_type.to_lowercase().contains(&ctype.to_lowercase()) {
                        continue;
                    }
                }
                hits.push(SearchHit {
                    distance: m.distance,
                    kind: "Memory".to_string(),
                    label: m.memory_type.clone(),
                    project: m.project_id.clone(),
                    content: m.content.clone(),
                });
            }

            for c in &code_results {
                if let Some(proj) = project {
                    if !c.project_id.to_lowercase().contains(&proj.to_lowercase()) {
                        continue;
                    }
                }
                let file_label = c.file_path.as_deref().unwrap_or(&c.project_id);
                hits.push(SearchHit {
                    distance: c.distance,
                    kind: "Code".to_string(),
                    label: c.granularity.clone(),
                    project: file_label.to_string(),
                    content: c.content.clone(),
                });
            }

            hits.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if hits.is_empty() {
                println!("No semantic results found for: {query}");
                return Ok(());
            }

            println!("Semantic search results for \"{query}\":\n");
            for hit in &hits {
                println!(
                    "[{:.3}] {} ({}) -- {}",
                    hit.distance, hit.kind, hit.label, hit.project
                );
                println!("  {}", truncate(&hit.content, 80));
                println!();
            }
            println!("{} result(s).", hits.len());
        }
        Ok(_) => {
            eprintln!("Warning: embedding returned empty result. Falling back to exact search.");
            return Box::pin(cmd_search(query, project, agent, since, content_type, true)).await;
        }
        Err(e) => {
            eprintln!(
                "Warning: could not connect to LM Studio ({e:#}). Falling back to exact search."
            );
            return Box::pin(cmd_search(query, project, agent, since, content_type, true)).await;
        }
    }

    Ok(())
}

fn cmd_find(query: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let memories = store.search_memories(query)?;
    let sessions = store.search_sessions_by_summary(query)?;

    let total = memories.len() + sessions.len();
    if total == 0 {
        println!("No results found for: {query}");
        return Ok(());
    }

    println!("Find results for \"{query}\":\n");

    if !memories.is_empty() {
        println!("--- Memories ({}) ---", memories.len());
        for m in &memories {
            let mtype = m.memory_type.as_deref().unwrap_or("unknown");
            let proj = m.project_id.as_deref().unwrap_or("-");
            println!("  [{mtype}] {proj}");
            println!("    {}", truncate(&m.content, 80));
            println!();
        }
    }

    if !sessions.is_empty() {
        println!("--- Sessions ({}) ---", sessions.len());
        for s in &sessions {
            let started = s
                .started_at
                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "-".to_string());
            let summary = s.summary.as_deref().unwrap_or("-");
            let id_short = if s.id.len() > 12 {
                format!("{}...", &s.id[..12])
            } else {
                s.id.clone()
            };
            println!(
                "  [{id_short}] {}: {} -- {}",
                s.agent,
                started,
                truncate(summary, 60)
            );
        }
        println!();
    }

    println!("{total} result(s) total.");
    Ok(())
}

fn cmd_brief(project: Option<&str>, today: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let now = chrono::Utc::now().naive_utc();
    let today_date = now.date();

    // Determine time window
    let since = if today {
        now - chrono::Duration::hours(24)
    } else {
        now - chrono::Duration::days(3)
    };

    let sessions = store.search_sessions(None, project, Some(since), 1000)?;
    let memories = store.get_memories(project, 10)?;

    let window_label = if today { "24h" } else { "3 days" };

    println!("=== Daily Brief ({}) ===\n", today_date.format("%Y-%m-%d"));

    // Recent sessions
    println!("Recent Sessions (last {window_label}):");
    if sessions.is_empty() {
        println!("  (no sessions)");
    } else {
        for s in &sessions {
            let summary = s.summary.as_deref().unwrap_or("(no summary)");
            let duration = s
                .duration_minutes
                .map(|d| format!("{d} min"))
                .unwrap_or_else(|| "-".to_string());
            let msgs = s
                .message_count
                .map(|n| format!("{n} msgs"))
                .unwrap_or_else(|| "-".to_string());
            let proj = s.project_id.as_deref().unwrap_or("-");
            println!(
                "  - [{}] {}: {} ({}, {})",
                s.agent,
                proj,
                truncate(summary, 50),
                duration,
                msgs
            );
        }
    }
    println!();

    // Recent memories
    println!("Recent Memories:");
    if memories.is_empty() {
        println!("  (no memories)");
    } else {
        for m in &memories {
            let mtype = m.memory_type.as_deref().unwrap_or("unknown");
            println!("  - [{mtype}] {}", truncate(&m.content, 60));
        }
    }
    println!();

    // Active projects
    let mut project_counts: HashMap<String, (usize, i32, i32)> = HashMap::new();
    let mut total_msgs = 0i32;
    let mut total_tools = 0i32;

    for s in &sessions {
        let proj = s.project_id.as_deref().unwrap_or("(unknown)").to_string();
        let entry = project_counts.entry(proj).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += s.message_count.unwrap_or(0);
        entry.2 += s.tool_call_count.unwrap_or(0);
        total_msgs += s.message_count.unwrap_or(0);
        total_tools += s.tool_call_count.unwrap_or(0);
    }

    println!("Active Projects:");
    if project_counts.is_empty() {
        println!("  (none)");
    } else {
        let mut sorted: Vec<_> = project_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        for (proj, (count, _, _)) in &sorted {
            let label = if *count == 1 { "session" } else { "sessions" };
            println!("  - {proj} ({count} {label})");
        }
    }
    println!();

    println!(
        "Summary: {} sessions, {} messages, {} tool calls across {} projects.",
        sessions.len(),
        total_msgs,
        total_tools,
        project_counts.len()
    );

    Ok(())
}

fn cmd_related(path: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let builder = build_graph(&store)?;

    let node_id = match builder.find_node_id(path)? {
        Some(id) => id,
        None => {
            println!("No node found matching: {path}");
            println!("Tip: use a file path (e.g. src/main.rs), session ID, or project ID.");
            return Ok(());
        }
    };

    let neighbors = builder.find_related(&node_id, 2)?;
    let output = graph_builder::format_related(path, &neighbors);
    print!("{output}");
    Ok(())
}

fn cmd_graph(path: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let builder = build_graph(&store)?;

    let node_id = match builder.find_node_id(path)? {
        Some(id) => id,
        None => {
            println!("No node found matching: {path}");
            println!("Tip: use a file path (e.g. src/main.rs), session ID, or project ID.");
            return Ok(());
        }
    };

    let node_name = GraphBackend::get_node(builder.backend(), &node_id)?
        .map(|(_, _, name, _)| name)
        .unwrap_or_else(|| path.to_string());

    let neighbors = builder.find_related(&node_id, 2)?;
    let output = graph_builder::format_graph_tree(path, &node_name, &neighbors);
    print!("{output}");
    Ok(())
}

fn cmd_timeline(topic: &str, since: Option<&str>) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let sessions = store.search_sessions_by_summary(topic)?;
    let memories = store.search_memories(topic)?;

    // Apply optional since filter
    let (sessions, memories) = if let Some(since_str) = since {
        if let Some(since_dt) = parse_since(since_str) {
            let filtered_sessions: Vec<_> = sessions
                .into_iter()
                .filter(|s| s.started_at.map_or(false, |dt| dt >= since_dt))
                .collect();
            let filtered_memories: Vec<_> = memories
                .into_iter()
                .filter(|m| m.created_at.map_or(false, |dt| dt >= since_dt))
                .collect();
            (filtered_sessions, filtered_memories)
        } else {
            eprintln!("Warning: could not parse --since value: {since_str}");
            (sessions, memories)
        }
    } else {
        (sessions, memories)
    };

    let output = graph_builder::format_timeline(topic, &sessions, &memories);
    print!("{output}");
    Ok(())
}

fn cmd_patterns(topic: Option<&str>) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    // Load pattern memories
    let all_memories = store.get_memories(None, 10_000)?;
    let pattern_memories: Vec<_> = all_memories
        .iter()
        .filter(|m| {
            m.memory_type
                .as_deref()
                .map(|t| t.contains("pattern"))
                .unwrap_or(false)
        })
        .filter(|m| {
            if let Some(t) = topic {
                m.content.to_lowercase().contains(&t.to_lowercase())
            } else {
                true
            }
        })
        .collect();

    if pattern_memories.is_empty() {
        println!("No patterns found.");
        if topic.is_some() {
            println!("Try without a topic filter, or run distillation to extract patterns.");
        }
        return Ok(());
    }

    // Group patterns by content and collect projects
    let mut pattern_groups: HashMap<String, (Vec<String>, Option<NaiveDateTime>)> = HashMap::new();
    for m in &pattern_memories {
        let content = m.content.clone();
        let entry = pattern_groups
            .entry(content)
            .or_insert_with(|| (Vec::new(), None));
        if let Some(ref pid) = m.project_id {
            if !entry.0.contains(pid) {
                entry.0.push(pid.clone());
            }
        }
        if let Some(created) = m.created_at {
            match entry.1 {
                None => entry.1 = Some(created),
                Some(existing) if created < existing => entry.1 = Some(created),
                _ => {}
            }
        }
    }

    // Sort by number of projects (descending)
    let mut sorted: Vec<_> = pattern_groups.into_iter().collect();
    sorted.sort_by(|a, b| b.1.0.len().cmp(&a.1.0.len()));

    println!("Cross-Project Patterns:\n");
    for (content, (projects, first_seen)) in &sorted {
        let count = projects.len();
        let projects_str = if projects.is_empty() {
            "(no project)".to_string()
        } else {
            projects.join(", ")
        };
        let first_str = first_seen
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!("[{count}x] {}", truncate(content, 80));
        println!("  Projects: {projects_str}");
        println!("  First seen: {first_str}");
        println!();
    }

    let project_set: std::collections::HashSet<&str> = sorted
        .iter()
        .flat_map(|(_, (projs, _))| projs.iter().map(|s| s.as_str()))
        .collect();
    println!(
        "Found {} patterns across {} projects.",
        sorted.len(),
        project_set.len()
    );
    Ok(())
}

fn cmd_decisions(project: Option<&str>, all: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let limit = if all { 10_000 } else { 20 };
    let decisions = store.get_decisions(project, limit)?;

    if decisions.is_empty() {
        println!("No decisions found.");
        return Ok(());
    }

    let total = store.count_decisions()?;

    println!("Decision Journal:\n");
    for d in &decisions {
        let date = d
            .created_at
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "????-??-??".to_string());
        let proj = d.project_id.as_deref().unwrap_or("?");
        println!("{}  [{}] {}", date, proj, truncate(&d.what, 70));
        if let Some(ref why) = d.why {
            println!("  Why: {}", truncate(why, 70));
        }
        if !d.alternatives.is_empty() {
            println!("  Alternatives: {}", d.alternatives.join(", "));
        }
        println!();
    }

    if !all && total > decisions.len() {
        println!(
            "Showing {} of {} decisions. Use --all to see all.",
            decisions.len(),
            total
        );
    }

    Ok(())
}

fn cmd_note(text: &str, project: Option<&str>, tags: Option<&[String]>) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let id = store.insert_note(text, project)?;

    let tag_str = tags.map(|t| t.join(", ")).unwrap_or_default();

    println!("Note saved: {id}");
    if let Some(p) = project {
        println!("  Project: {p}");
    }
    if !tag_str.is_empty() {
        println!("  Tags: {tag_str}");
    }
    Ok(())
}

fn cmd_forget(session_id: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    if store.delete_session(session_id)? {
        println!("Session {session_id} deleted.");
    } else {
        println!("Session {session_id} not found.");
    }
    Ok(())
}

fn cmd_export(project: Option<&str>, format: &str, output: Option<&str>) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let memories = store.get_memories(project, 10_000)?;
    let decisions = store.get_decisions(project, 10_000)?;
    let sessions = if let Some(proj) = project {
        store.get_project_sessions(proj)?
    } else {
        store.get_recent_sessions(100)?
    };

    let content = match format {
        "json" => {
            let data = serde_json::json!({
                "project": project.unwrap_or("all"),
                "generated_at": chrono::Utc::now().naive_utc().format("%Y-%m-%d").to_string(),
                "memories": memories,
                "decisions": decisions,
                "sessions": sessions,
            });
            serde_json::to_string_pretty(&data).context("failed to serialize JSON")?
        }
        _ => {
            // Markdown: CLAUDE.md-style document
            let mut md = String::new();
            let proj_label = project.unwrap_or("All Projects");
            md.push_str(&format!("# Project Memory: {proj_label}\n\n"));

            // Key Decisions
            md.push_str("## Key Decisions\n");
            if decisions.is_empty() {
                md.push_str("- (none recorded)\n");
            } else {
                for d in &decisions {
                    let date = d
                        .created_at
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_default();
                    md.push_str(&format!("- {} ({})\n", d.what, date));
                }
            }
            md.push('\n');

            // Patterns
            let patterns: Vec<_> = memories
                .iter()
                .filter(|m| {
                    m.memory_type
                        .as_deref()
                        .map(|t| t.contains("pattern"))
                        .unwrap_or(false)
                })
                .collect();
            md.push_str("## Patterns\n");
            if patterns.is_empty() {
                md.push_str("- (none recorded)\n");
            } else {
                for m in &patterns {
                    md.push_str(&format!("- {}\n", m.content));
                }
            }
            md.push('\n');

            // Insights
            let insights: Vec<_> = memories
                .iter()
                .filter(|m| {
                    m.memory_type
                        .as_deref()
                        .map(|t| t == "insight" || t == "note")
                        .unwrap_or(false)
                })
                .collect();
            md.push_str("## Insights\n");
            if insights.is_empty() {
                md.push_str("- (none recorded)\n");
            } else {
                for m in &insights {
                    md.push_str(&format!("- {}\n", m.content));
                }
            }
            md.push('\n');

            // Recent Sessions
            md.push_str("## Recent Sessions\n");
            let session_limit = sessions.len().min(20);
            if sessions.is_empty() {
                md.push_str("- (none)\n");
            } else {
                for s in &sessions[..session_limit] {
                    let summary = s.summary.as_deref().unwrap_or("(no summary)");
                    let date = s
                        .started_at
                        .map(|dt| dt.format("%Y-%m-%d").to_string())
                        .unwrap_or_default();
                    let id_short = if s.id.len() > 8 { &s.id[..8] } else { &s.id };
                    md.push_str(&format!("- {id_short}: {summary} ({date}, {})\n", s.agent));
                }
            }
            md.push('\n');

            let today = chrono::Utc::now()
                .naive_utc()
                .format("%Y-%m-%d")
                .to_string();
            md.push_str(&format!("Generated by Remembrant on {today}\n"));

            md
        }
    };

    match output {
        Some(path) => {
            let out_path = expand_tilde(path);
            std::fs::write(&out_path, &content)
                .with_context(|| format!("failed to write to {}", out_path.display()))?;
            println!("Exported to {}", out_path.display());
        }
        None => {
            print!("{content}");
        }
    }

    Ok(())
}

fn cmd_stats() -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let session_count = store.count_sessions()?;
    let memory_count = store.count_memories()?;
    let decision_count = store.count_decisions()?;
    let tool_call_count = store.count_tool_calls()?;

    // Per-agent breakdown
    let agent_counts = store.get_agent_session_counts()?;
    let agent_str = if agent_counts.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = agent_counts
            .iter()
            .map(|(agent, count)| format!("{agent}: {count}"))
            .collect();
        format!(" ({})", parts.join(", "))
    };

    println!("Remembrant Statistics:\n");
    println!("Sessions:    {session_count}{agent_str}");
    println!("Memories:    {memory_count}");
    println!("Decisions:   {decision_count}");
    println!("Tool calls:  {tool_call_count}");

    // Per-project
    let project_counts = store.get_project_session_counts()?;
    if !project_counts.is_empty() {
        println!("\nTop Projects:");
        for (project, count) in &project_counts {
            let label = if *count == 1 { "session" } else { "sessions" };
            println!("  {:<20} {count} {label}", project);
        }
    }

    // Storage sizes
    let db_path = expand_tilde(&config.storage.duckdb_path);
    let lance_path = expand_tilde(&config.storage.lancedb_path);

    let duck_size = if db_path.exists() {
        std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let lance_size = if lance_path.exists() {
        dir_size(&lance_path)
    } else {
        0
    };

    println!(
        "\nStorage: DuckDB {}, LanceDB {}",
        human_size(duck_size),
        human_size(lance_size)
    );

    Ok(())
}

/// Compute total size of a directory recursively.
fn dir_size(path: &PathBuf) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let meta = entry.metadata();
            if let Ok(meta) = meta {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += dir_size(&entry.path());
                }
            }
        }
    }
    total
}

fn cmd_gc() -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    let retention_days = config.retention.raw_transcripts_days;
    let cutoff = chrono::Utc::now().naive_utc() - chrono::Duration::days(retention_days as i64);

    // Get DB size before
    let db_path = expand_tilde(&config.storage.duckdb_path);
    let size_before = if db_path.exists() {
        std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let deleted = store.gc_sessions_before(cutoff)?;

    let size_after = if db_path.exists() {
        std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let freed = size_before.saturating_sub(size_after);

    println!("Garbage collection:");
    println!("  Deleted {deleted} sessions older than {retention_days} days");
    println!("  Freed ~{}", human_size(freed));

    Ok(())
}

async fn cmd_ingest(skip_embed: bool, skip_distill: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    println!("╔══════════════════════════════════════════╗");
    println!("║  REMEMBRANT — Full Ingest Pipeline       ║");
    println!("╚══════════════════════════════════════════╝\n");

    // ── Step 1: Detect & parse agent artifacts ──────────────────────
    println!("▸ Step 1/4: Parsing agent artifacts...");
    let detection = detect_agents();

    let mut all_sessions = Vec::new();
    let mut all_memories = Vec::new();
    let mut all_tool_calls = Vec::new();

    // Claude Code
    if detection.claude_code.is_some() {
        match ClaudeIngester::new() {
            Ok(ingester) => match ingester.ingest_all() {
                Ok(result) => {
                    println!(
                        "  ✓ Claude Code: {} sessions, {} tool calls, {} memories",
                        result.sessions.len(),
                        result.tool_calls.len(),
                        result.memories.len()
                    );
                    for s in &result.sessions {
                        let _ = store.insert_or_replace_session(s);
                    }
                    for m in &result.memories {
                        let _ = store.insert_memory(m);
                    }
                    for tc in &result.tool_calls {
                        let _ = store.insert_tool_call(tc);
                    }
                    all_sessions.extend(result.sessions);
                    all_memories.extend(result.memories);
                    all_tool_calls.extend(result.tool_calls);
                }
                Err(e) => eprintln!("  ✗ Claude Code: {e}"),
            },
            Err(e) => eprintln!("  ✗ Claude Code init: {e}"),
        }
    }

    // Codex
    if detection.codex.is_some() {
        if let Some(ingester) = CodexIngester::new() {
            match ingester.ingest_all() {
                Ok(result) => {
                    println!(
                        "  ✓ Codex CLI:   {} sessions, {} tool calls, {} memories",
                        result.sessions.len(),
                        result.tool_calls.len(),
                        result.memories.len()
                    );
                    for s in &result.sessions {
                        let _ = store.insert_or_replace_session(s);
                    }
                    for m in &result.memories {
                        let _ = store.insert_memory(m);
                    }
                    for tc in &result.tool_calls {
                        let _ = store.insert_tool_call(tc);
                    }
                    all_sessions.extend(result.sessions);
                    all_memories.extend(result.memories);
                    all_tool_calls.extend(result.tool_calls);
                }
                Err(e) => eprintln!("  ✗ Codex CLI: {e}"),
            }
        }
    }

    // Gemini
    if detection.gemini.is_some() {
        if let Some(ingester) = GeminiIngester::new() {
            let (result, sessions, tool_calls, memories) = ingester.ingest_all();
            println!(
                "  ✓ Gemini CLI:  {} sessions, {} tool calls, {} memories",
                result.sessions_found, result.tool_calls_found, result.memories_found
            );
            for s in &sessions {
                let _ = store.insert_or_replace_session(s);
            }
            for m in &memories {
                let _ = store.insert_memory(m);
            }
            for tc in &tool_calls {
                let _ = store.insert_tool_call(tc);
            }
            all_sessions.extend(sessions);
            all_memories.extend(memories);
            all_tool_calls.extend(tool_calls);
        }
    }

    let total_s = all_sessions.len();
    let total_m = all_memories.len();
    let total_tc = all_tool_calls.len();
    println!(
        "\n  Total: {total_s} sessions, {total_tc} tool calls, {total_m} memories → DuckDB ✓"
    );

    // ── Step 2: LLM Distillation ────────────────────────────────────
    if skip_distill {
        println!("\n▸ Step 2/4: Distillation — skipped (--skip-distill)");
    } else {
        println!("\n▸ Step 2/4: Distilling with LLM...");

        let distiller = Distiller::new(&config.distillation);
        let has_llm = !config.distillation.llm_model.is_empty();

        if has_llm {
            println!("  Using LLM: {}", config.distillation.llm_model);
        } else {
            println!("  No LLM configured — using keyword extraction fallback");
            println!("  Tip: Set distillation.llm_model in ~/.remembrant/config.yaml");
        }

        let mut decisions_count = 0usize;
        let mut patterns_count = 0usize;
        let mut problems_count = 0usize;

        for session in &all_sessions {
            let summary = session.summary.as_deref().unwrap_or("");
            let files = session.files_changed.join(", ");
            let text = format!(
                "Session: {}\nAgent: {}\nSummary: {}\nFiles: {}",
                session.id, session.agent, summary, files
            );

            match distiller.distill_session(session, &text).await {
                Ok(distilled) => {
                    for d in distiller.to_decisions(&distilled) {
                        let _ = store.insert_decision(&d);
                        decisions_count += 1;
                    }
                    for m in distiller.to_memories(&distilled) {
                        let _ = store.insert_memory(&m);
                        patterns_count += 1;
                    }
                    problems_count += distilled.problems.len();
                }
                Err(e) => {
                    eprintln!("  ✗ Distill session {}: {e}", session.id);
                }
            }
        }

        println!(
            "  Extracted: {decisions_count} decisions, {patterns_count} patterns, {problems_count} problems"
        );
    }

    // ── Step 3: Embeddings ──────────────────────────────────────────
    if skip_embed {
        println!("\n▸ Step 3/4: Embedding — skipped (--skip-embed)");
    } else {
        println!("\n▸ Step 3/4: Embedding with LM Studio...");
        println!("  Model: {}", config.embedding.model);
        println!("  Dimensions: {}", config.embedding.dimensions);

        let embedder = LmStudioEmbedder::from_config(&config.embedding);

        // Test connection first
        match embedder.embed_texts(&["test"]).await {
            Ok(_) => println!("  ✓ LM Studio connection OK"),
            Err(e) => {
                eprintln!("  ✗ LM Studio not reachable: {e}");
                eprintln!("  Start LM Studio and load an embedding model, then retry.");
                eprintln!("  Skipping embedding step.\n");
                println!("▸ Step 4/4: Building graph...");
                let graph = build_graph(&store)?;
                let node_count = graph.node_count();
                let edge_count = graph.edge_count();
                println!("  Graph: {node_count} nodes, {edge_count} edges");
                print_summary(total_s, total_m, total_tc);
                return Ok(());
            }
        }

        let lance = open_lance_store(&config).await?;
        let pipeline = EmbedPipeline::new(lance, config.embedding.batch_size);

        let stats = pipeline
            .run(&all_sessions, &all_memories, &all_tool_calls, &embedder)
            .await?;

        println!(
            "  Embedded: {} chunks ({} stored, {} errors)",
            stats.chunks_embedded, stats.chunks_stored, stats.errors
        );
    }

    // ── Step 4: Graph ───────────────────────────────────────────────
    println!("\n▸ Step 4/4: Building relationship graph...");
    let graph = build_graph(&store)?;
    let node_count = graph.node_count();
    let edge_count = graph.edge_count();
    println!("  Graph: {node_count} nodes, {edge_count} edges");

    print_summary(total_s, total_m, total_tc);
    Ok(())
}

fn print_summary(sessions: usize, memories: usize, tool_calls: usize) {
    println!("\n╔══════════════════════════════════════════╗");
    println!("║  Pipeline complete                       ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Sessions:   {:>6}                      ║", sessions);
    println!("║  Memories:   {:>6}                      ║", memories);
    println!("║  Tool calls: {:>6}                      ║", tool_calls);
    println!("╚══════════════════════════════════════════╝");
    println!("\nRun `rem stats` for full analytics.");
    println!("Run `rem recent` to browse sessions.");
    println!("Run `rem search <query>` to search.");
}

/// Build an in-memory graph from all DuckDB data.
fn build_graph(store: &DuckStore) -> Result<GraphBuilder<remembrant_engine::store::GraphStore>> {
    let sessions = store.get_recent_sessions(10_000)?;
    let memories = store.get_memories(None, 10_000)?;
    let decisions = store.get_decisions(None, 10_000)?;
    let tool_calls = Vec::new();

    let builder = GraphBuilder::new();
    builder.build_from_data(&sessions, &memories, &decisions, &tool_calls)?;
    Ok(builder)
}

#[allow(unused_variables)]
fn cmd_analyze(path: &str, project: Option<&str>) -> Result<()> {
    #[cfg(not(feature = "code-analysis"))]
    {
        anyhow::bail!(
            "Code analysis requires the 'code-analysis' feature.\n\
             Rebuild with: cargo build --features code-analysis"
        );
    }

    #[cfg(feature = "code-analysis")]
    {
        use remembrant_engine::code_analysis::CodeAnalyzer;

        let config = AppConfig::load()?;
        let store = open_store(&config)?;
        let repo_path = expand_tilde(path);

        let project_id = project
            .map(String::from)
            .unwrap_or_else(|| {
                repo_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            });

        println!("Analyzing {}...", repo_path.display());
        println!("  Project: {project_id}");

        let analyzer = CodeAnalyzer::new(&project_id, &repo_path);

        // Build a graph store for population
        let graph = build_graph(&store)?;

        let result = analyzer.analyze(&store, graph.store())?;

        println!("\nAnalysis complete:");
        println!("  Files analyzed:    {}", result.files_analyzed);
        println!("  Symbols extracted: {}", result.symbols_extracted);
        println!("  Dependencies:      {}", result.dependencies_found);
        println!("  Duration:          {}ms", result.duration_ms);

        Ok(())
    }
}

async fn cmd_embed(path: &str, _update: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let abs_path =
        std::fs::canonicalize(path).with_context(|| format!("failed to resolve path: {path}"))?;

    println!("Embedding repository: {}", abs_path.display());

    let embedder = RepoEmbedder::new(&abs_path);

    // Discover and chunk first so we can report counts before embedding.
    let (chunks, file_count) = embedder.chunk_all()?;
    println!("Found {} chunks from {} files", chunks.len(), file_count);

    if chunks.is_empty() {
        println!("No embeddable files found. Nothing to do.");
        return Ok(());
    }

    // Create LM Studio embedder.
    let embed_provider = LmStudioEmbedder::from_config(&config.embedding);

    // Open LanceDB.
    let lance_store = open_lance_store(&config).await?;

    let result = embedder
        .embed_and_store(&embed_provider, &lance_store, config.embedding.batch_size)
        .await;

    match result {
        Ok(result) => {
            println!("\nEmbed complete:");
            println!("  Files:  {}", result.files_found);
            println!(
                "  Chunks: {} created, {} embedded",
                result.chunks_created, result.chunks_embedded
            );
            if result.errors > 0 {
                println!("  Errors: {}", result.errors);
            }
        }
        Err(e) => {
            let err_msg = format!("{e:#}");
            if err_msg.contains("connect") || err_msg.contains("LM Studio") {
                eprintln!(
                    "Error: Could not connect to LM Studio.\n\n\
                     To use `rem embed`, you need LM Studio running locally:\n\
                     1. Download LM Studio from https://lmstudio.ai\n\
                     2. Start it and load an embedding model (e.g. nomic-embed-text)\n\
                     3. Start the local server (it listens on http://localhost:1234)\n\
                     4. Run `rem embed` again\n\n\
                     Underlying error: {e:#}"
                );
            } else {
                return Err(e).context("embed failed");
            }
        }
    }

    Ok(())
}

fn cmd_xpath(query: &str, depth: usize, limit: usize, show_tree: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let store = open_store(&config)?;

    // Parse the query
    let parsed = remembrant_engine::xpath_query::parse(query)
        .map_err(|e| anyhow::anyhow!("Parse error at position {}: {}", e.position, e.message))?;

    // Build the tree
    let builder = remembrant_engine::TreeBuilder::new(&store);
    let root = builder.build_tree(depth)?;

    // Evaluate with keyword scorer (no embeddings needed)
    let scorer = remembrant_engine::semantic_scorer::keyword_scorer;
    let results = remembrant_engine::xpath_query::evaluate(&parsed, &root, &scorer);

    // Display results
    if results.is_empty() {
        println!("No results found for: {query}");
        return Ok(());
    }

    let display_count = results.len().min(limit);
    println!("Found {} results for: {query}\n", results.len());

    for (i, result) in results.iter().take(display_count).enumerate() {
        println!(
            "{}. [{}] {} (score: {:.3})",
            i + 1,
            result.node_type,
            result.name,
            result.weight,
        );
        if show_tree {
            let path_str = result.path.join(" -> ");
            println!("   Path: {path_str}");
        }
    }

    if results.len() > display_count {
        println!(
            "\n... and {} more results (use --limit to show more)",
            results.len() - display_count
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init => {
            cmd_init()?;
        }
        Commands::Watch => {
            cmd_watch().await?;
        }
        Commands::Stop => {
            cmd_stop()?;
        }
        Commands::Search {
            query,
            project,
            agent,
            since,
            content_type,
            exact,
        } => {
            cmd_search(
                &query,
                project.as_deref(),
                agent.as_deref(),
                since.as_deref(),
                content_type.as_deref(),
                exact,
            )
            .await?;
        }
        Commands::Find { query } => {
            cmd_find(&query)?;
        }
        Commands::Recent { limit, .. } => {
            cmd_recent(limit)?;
        }
        Commands::Brief { project, today } => {
            cmd_brief(project.as_deref(), today)?;
        }
        Commands::Patterns { topic } => {
            cmd_patterns(topic.as_deref())?;
        }
        Commands::Decisions { project, all } => {
            cmd_decisions(project.as_deref(), all)?;
        }
        Commands::Related { path } => {
            cmd_related(&path)?;
        }
        Commands::Graph { path } => {
            cmd_graph(&path)?;
        }
        Commands::Timeline { topic, since } => {
            cmd_timeline(&topic, since.as_deref())?;
        }
        Commands::Note { text, project, tag } => {
            cmd_note(
                &text,
                project.as_deref(),
                tag.as_ref().map(|v| v.as_slice()),
            )?;
        }
        Commands::Forget { session } => {
            cmd_forget(&session)?;
        }
        Commands::Export {
            project,
            format,
            output,
        } => {
            cmd_export(project.as_deref(), &format, output.as_deref())?;
        }
        Commands::Embed { path, update } => {
            cmd_embed(&path, update).await?;
        }
        Commands::Ingest {
            skip_embed,
            skip_distill,
        } => {
            cmd_ingest(skip_embed, skip_distill).await?;
        }
        Commands::Status => {
            cmd_status()?;
        }
        Commands::Stats => {
            cmd_stats()?;
        }
        Commands::Gc => {
            cmd_gc()?;
        }
        Commands::Analyze { path, project } => {
            cmd_analyze(&path, project.as_deref())?;
        }
        Commands::XPath {
            query,
            depth,
            limit,
            tree,
        } => {
            cmd_xpath(&query, depth, limit, tree)?;
        }
    }

    Ok(())
}
