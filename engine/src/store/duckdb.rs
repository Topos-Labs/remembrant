use anyhow::{Context, Result};
use chrono::{NaiveDateTime, Utc};
use duckdb::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Domain structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub project_id: Option<String>,
    pub agent: String,
    pub started_at: Option<NaiveDateTime>,
    pub ended_at: Option<NaiveDateTime>,
    pub duration_minutes: Option<i32>,
    pub message_count: Option<i32>,
    pub tool_call_count: Option<i32>,
    pub total_tokens: Option<i32>,
    pub files_changed: Vec<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub id: String,
    pub session_id: Option<String>,
    pub project_id: Option<String>,
    pub decision_type: Option<String>,
    pub what: String,
    pub why: Option<String>,
    pub alternatives: Vec<String>,
    pub outcome: Option<String>,
    pub created_at: Option<NaiveDateTime>,
    pub valid_until: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub project_id: Option<String>,
    pub content: String,
    pub memory_type: Option<String>,
    pub source_session_id: Option<String>,
    pub confidence: f32,
    pub access_count: i32,
    pub created_at: Option<NaiveDateTime>,
    pub updated_at: Option<NaiveDateTime>,
    pub valid_until: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub session_id: Option<String>,
    pub tool_name: Option<String>,
    pub command: Option<String>,
    pub success: Option<bool>,
    pub error_message: Option<String>,
    pub duration_ms: Option<i32>,
    pub timestamp: Option<NaiveDateTime>,
}

/// A temporal fact extracted from a coding session.
/// Facts have validity windows: they are true from `valid_at` until `invalid_at`.
/// When a contradicting fact is found, the old fact's `invalid_at` is set and a new
/// fact is created, preserving the full history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub project_id: Option<String>,
    pub subject: String,   // entity the fact is about (e.g., "auth module")
    pub predicate: String, // relationship (e.g., "uses", "depends_on", "is_located_at")
    pub object: String,    // value (e.g., "JWT tokens", "src/auth.rs")
    pub confidence: f32,
    pub source_session_id: Option<String>,
    pub source_agent: Option<String>,
    pub valid_at: Option<NaiveDateTime>,
    pub invalid_at: Option<NaiveDateTime>, // None = still valid
    pub superseded_by: Option<String>,     // ID of the fact that replaced this one
    pub created_at: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStat {
    pub file_path: String,
    pub project_id: String,
    pub language: Option<String>,
    pub lines_of_code: Option<i32>,
    pub token_count: Option<i32>,
    pub complexity: Option<f64>,
    pub change_frequency: i32,
    pub last_modified: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSymbol {
    pub id: String, // "project:file:symbol:line"
    pub project_id: String,
    pub file_path: String,
    pub symbol_name: String,
    pub symbol_kind: String, // function, class, struct, method, etc.
    pub signature: Option<String>,
    pub docstring: Option<String>,
    pub start_line: i32,
    pub end_line: i32,
    pub visibility: Option<String>, // public, private, protected
    pub parent_symbol: Option<String>,
    pub pagerank_score: f64,
    pub reference_count: i32,
    pub language: Option<String>,
    pub content_hash: Option<String>, // BLAKE3
    pub indexed_at: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDependency {
    pub id: String,
    pub project_id: String,
    pub from_symbol: String,
    pub to_symbol: String,
    pub relationship: String, // calls, imports, inherits, implements, references
    pub from_file: String,
    pub to_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisRun {
    pub project_id: String,
    pub commit_hash: Option<String>,
    pub files_analyzed: i32,
    pub symbols_extracted: i32,
    pub dependencies_found: i32,
    pub chunks_generated: i32,
    pub duration_ms: i32,
    pub analyzed_at: Option<NaiveDateTime>,
}

// ---------------------------------------------------------------------------
// Helpers — store Vec<String> as JSON text in DuckDB
// ---------------------------------------------------------------------------

fn vec_to_json(v: &[String]) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "[]".to_string())
}

fn json_to_vec(s: &str) -> Vec<String> {
    serde_json::from_str(s).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Graph row structs (for DuckPGQ-backed graph storage)
// ---------------------------------------------------------------------------

/// A row from the `graph_nodes` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNodeRow {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub properties: String, // JSON
}

/// A neighbor row returned from a graph adjacency query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNeighborRow {
    pub node: GraphNodeRow,
    pub edge_kind: String,
    pub direction: String, // "outgoing" or "incoming"
}

// ---------------------------------------------------------------------------
// DuckStore
// ---------------------------------------------------------------------------

/// Persistent store backed by an embedded DuckDB database.
pub struct DuckStore {
    conn: Mutex<Connection>,
}

impl DuckStore {
    /// Access the underlying connection mutex (for modules that need custom queries).
    pub fn connection(&self) -> &Mutex<Connection> {
        &self.conn
    }

    /// Open (or create) a DuckDB database at `path` and initialise the schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("failed to open DuckDB at {}", path.as_ref().display()))?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory DuckDB instance (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("failed to open in-memory DuckDB")?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    // -----------------------------------------------------------------------
    // Schema
    // -----------------------------------------------------------------------

    /// Create all tables if they do not already exist.
    pub fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                visibility TEXT DEFAULT 'public',
                tags TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP DEFAULT current_timestamp
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                agent TEXT NOT NULL,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                duration_minutes INTEGER,
                message_count INTEGER,
                tool_call_count INTEGER,
                total_tokens INTEGER,
                files_changed TEXT,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                project_id TEXT,
                decision_type TEXT,
                what TEXT NOT NULL,
                why TEXT,
                alternatives TEXT,
                outcome TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp,
                valid_until TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                content TEXT NOT NULL,
                memory_type TEXT,
                source_session_id TEXT,
                confidence REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT current_timestamp,
                updated_at TIMESTAMP DEFAULT current_timestamp,
                valid_until TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                tool_name TEXT,
                command TEXT,
                success BOOLEAN,
                error_message TEXT,
                duration_ms INTEGER,
                timestamp TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_session_id TEXT,
                source_agent TEXT,
                valid_at TIMESTAMP DEFAULT current_timestamp,
                invalid_at TIMESTAMP,
                superseded_by TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp
            );

            CREATE TABLE IF NOT EXISTS file_stats (
                file_path TEXT,
                project_id TEXT,
                language TEXT,
                lines_of_code INTEGER,
                token_count INTEGER,
                complexity REAL,
                change_frequency INTEGER DEFAULT 0,
                last_modified TIMESTAMP,
                PRIMARY KEY (file_path, project_id)
            );

            CREATE TABLE IF NOT EXISTS code_symbols (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_kind TEXT NOT NULL,
                signature TEXT,
                docstring TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                visibility TEXT,
                parent_symbol TEXT,
                pagerank_score REAL DEFAULT 0.0,
                reference_count INTEGER DEFAULT 0,
                language TEXT,
                content_hash TEXT,
                indexed_at TIMESTAMP DEFAULT current_timestamp
            );

            CREATE TABLE IF NOT EXISTS code_dependencies (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                from_symbol TEXT NOT NULL,
                to_symbol TEXT NOT NULL,
                relationship TEXT NOT NULL,
                from_file TEXT NOT NULL,
                to_file TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS code_analysis_runs (
                project_id TEXT PRIMARY KEY,
                commit_hash TEXT,
                files_analyzed INTEGER DEFAULT 0,
                symbols_extracted INTEGER DEFAULT 0,
                dependencies_found INTEGER DEFAULT 0,
                chunks_generated INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                analyzed_at TIMESTAMP DEFAULT current_timestamp
            );

            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                properties TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                FOREIGN KEY (from_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (to_id) REFERENCES graph_nodes(id)
            );
            ",
        )
        .context("failed to initialise DuckDB schema")?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Full-Text Search (BM25 via DuckDB FTS extension)
    // -----------------------------------------------------------------------

    /// Install and load the DuckDB FTS extension, then create full-text
    /// indexes on key tables. Call this once after `init_schema` (or on
    /// demand before the first FTS query).
    ///
    /// FTS indexes use BM25 scoring which is far superior to ILIKE for
    /// identifier and keyword search. The GrepRAG paper shows simple lexical
    /// retrieval matches complex methods for code search.
    ///
    /// This is idempotent — safe to call multiple times. If the FTS extension
    /// is unavailable the method returns an error but doesn't crash.
    pub fn init_fts(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");

        conn.execute_batch("INSTALL fts; LOAD fts;")
            .context("failed to load DuckDB FTS extension")?;

        // Drop existing FTS indexes before recreating (idempotent).
        // DuckDB FTS uses PRAGMA which doesn't support IF NOT EXISTS,
        // so we drop first to avoid "already exists" errors.
        let _ = conn.execute_batch("PRAGMA drop_fts_index('memories');");
        let _ = conn.execute_batch("PRAGMA drop_fts_index('facts');");
        let _ = conn.execute_batch("PRAGMA drop_fts_index('sessions');");
        let _ = conn.execute_batch("PRAGMA drop_fts_index('code_symbols');");
        let _ = conn.execute_batch("PRAGMA drop_fts_index('decisions');");

        // Create FTS indexes on searchable columns.
        // stemmer='none' preserves identifiers exactly (important for code).
        conn.execute_batch(
            "PRAGMA create_fts_index('memories', 'id', 'content', stemmer='porter', overwrite=1);
             PRAGMA create_fts_index('facts', 'id', 'subject', 'predicate', 'object', stemmer='porter', overwrite=1);
             PRAGMA create_fts_index('sessions', 'id', 'summary', stemmer='porter', overwrite=1);
             PRAGMA create_fts_index('code_symbols', 'id', 'symbol_name', 'file_path', 'signature', 'docstring', stemmer='none', overwrite=1);
             PRAGMA create_fts_index('decisions', 'id', 'what', 'why', 'alternatives', stemmer='porter', overwrite=1);",
        )
        .context("failed to create FTS indexes")?;

        tracing::info!("FTS indexes created on memories, facts, sessions, code_symbols, decisions");
        Ok(())
    }

    /// BM25 full-text search over memories. Returns results ranked by relevance.
    /// Falls back to ILIKE if FTS indexes haven't been created.
    pub fn search_memories_fts(&self, query: &str) -> Result<Vec<(Memory, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.project_id, m.content, m.memory_type, m.source_session_id,
                    m.confidence, m.access_count, m.created_at, m.updated_at, m.valid_until,
                    fts.score
             FROM memories m
             JOIN (SELECT id, fts_main_memories.match_bm25(id, ?) AS score
                   FROM memories) fts ON m.id = fts.id
             WHERE fts.score IS NOT NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS memory search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let memory = Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    memory_type: row.get(3)?,
                    source_session_id: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    access_count: row.get::<_, i32>(6).unwrap_or(0),
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    valid_until: row.get(9)?,
                };
                let score: f64 = row.get::<_, f64>(10).unwrap_or(0.0);
                Ok((memory, score))
            })
            .context("failed to query FTS memories")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS memory row")?);
        }
        Ok(results)
    }

    /// BM25 full-text search over facts.
    pub fn search_facts_fts(&self, query: &str) -> Result<Vec<(Fact, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT f.id, f.project_id, f.subject, f.predicate, f.object,
                    f.confidence, f.source_session_id, f.source_agent,
                    f.valid_at, f.invalid_at, f.superseded_by, f.created_at,
                    fts.score
             FROM facts f
             JOIN (SELECT id, fts_main_facts.match_bm25(id, ?) AS score
                   FROM facts) fts ON f.id = fts.id
             WHERE fts.score IS NOT NULL AND f.invalid_at IS NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS fact search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let fact = Fact {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    source_session_id: row.get(6)?,
                    source_agent: row.get(7)?,
                    valid_at: row.get(8)?,
                    invalid_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    created_at: row.get(11)?,
                };
                let score: f64 = row.get::<_, f64>(12).unwrap_or(0.0);
                Ok((fact, score))
            })
            .context("failed to query FTS facts")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS fact row")?);
        }
        Ok(results)
    }

    /// BM25 full-text search over sessions (by summary).
    pub fn search_sessions_fts(&self, query: &str) -> Result<Vec<(Session, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.project_id, s.agent, s.started_at, s.ended_at,
                    s.duration_minutes, s.message_count, s.tool_call_count,
                    s.total_tokens, s.files_changed, s.summary,
                    fts.score
             FROM sessions s
             JOIN (SELECT id, fts_main_sessions.match_bm25(id, ?) AS score
                   FROM sessions) fts ON s.id = fts.id
             WHERE fts.score IS NOT NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS session search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                let session = Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                };
                let score: f64 = row.get::<_, f64>(11).unwrap_or(0.0);
                Ok((session, score))
            })
            .context("failed to query FTS sessions")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS session row")?);
        }
        Ok(results)
    }

    /// BM25 full-text search over code symbols.
    /// Uses stemmer='none' for exact identifier matching.
    pub fn search_code_symbols_fts(&self, query: &str) -> Result<Vec<(CodeSymbol, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT cs.id, cs.project_id, cs.file_path, cs.symbol_name, cs.symbol_kind,
                    cs.signature, cs.docstring, cs.start_line, cs.end_line,
                    cs.visibility, cs.parent_symbol, cs.pagerank_score,
                    cs.reference_count, cs.language, cs.content_hash, cs.indexed_at,
                    fts.score
             FROM code_symbols cs
             JOIN (SELECT id, fts_main_code_symbols.match_bm25(id, ?) AS score
                   FROM code_symbols) fts ON cs.id = fts.id
             WHERE fts.score IS NOT NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS code_symbols search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let symbol = CodeSymbol {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    file_path: row.get(2)?,
                    symbol_name: row.get(3)?,
                    symbol_kind: row.get(4)?,
                    signature: row.get(5)?,
                    docstring: row.get(6)?,
                    start_line: row.get(7)?,
                    end_line: row.get(8)?,
                    visibility: row.get(9)?,
                    parent_symbol: row.get(10)?,
                    pagerank_score: row.get::<_, f64>(11).unwrap_or(0.0),
                    reference_count: row.get::<_, i32>(12).unwrap_or(0),
                    language: row.get(13)?,
                    content_hash: row.get(14)?,
                    indexed_at: row.get(15)?,
                };
                let score: f64 = row.get::<_, f64>(16).unwrap_or(0.0);
                Ok((symbol, score))
            })
            .context("failed to query FTS code_symbols")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS code_symbol row")?);
        }
        Ok(results)
    }

    /// BM25 full-text search over decisions.
    pub fn search_decisions_fts(&self, query: &str) -> Result<Vec<(Decision, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT d.id, d.session_id, d.project_id, d.decision_type, d.what,
                    d.why, d.alternatives, d.outcome, d.created_at, d.valid_until,
                    fts.score
             FROM decisions d
             JOIN (SELECT id, fts_main_decisions.match_bm25(id, ?) AS score
                   FROM decisions) fts ON d.id = fts.id
             WHERE fts.score IS NOT NULL
             ORDER BY fts.score DESC",
            )
            .context("failed to prepare FTS decision search")?;

        let rows = stmt
            .query_map(params![query], |row| {
                let alts_str: String = row.get::<_, String>(6).unwrap_or_default();
                let decision = Decision {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    project_id: row.get(2)?,
                    decision_type: row.get(3)?,
                    what: row.get(4)?,
                    why: row.get(5)?,
                    alternatives: json_to_vec(&alts_str),
                    outcome: row.get(7)?,
                    created_at: row.get(8)?,
                    valid_until: row.get(9)?,
                };
                let score: f64 = row.get::<_, f64>(10).unwrap_or(0.0);
                Ok((decision, score))
            })
            .context("failed to query FTS decisions")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read FTS decision row")?);
        }
        Ok(results)
    }

    /// Check whether FTS indexes have been created.
    pub fn has_fts(&self) -> bool {
        let conn = self.conn.lock().expect("lock poisoned");
        // If the FTS macro table exists, indexes are active.
        conn.prepare("SELECT * FROM fts_main_memories.docs LIMIT 0")
            .is_ok()
    }

    // -----------------------------------------------------------------------
    // Inserts
    // -----------------------------------------------------------------------

    /// Insert a session record.
    pub fn insert_session(&self, session: &Session) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        let files_json = vec_to_json(&session.files_changed);
        conn.execute(
            "INSERT INTO sessions (
                id, project_id, agent, started_at, ended_at,
                duration_minutes, message_count, tool_call_count,
                total_tokens, files_changed, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                session.id,
                session.project_id,
                session.agent,
                session.started_at,
                session.ended_at,
                session.duration_minutes,
                session.message_count,
                session.tool_call_count,
                session.total_tokens,
                files_json,
                session.summary,
            ],
        )
        .context("failed to insert session")?;
        Ok(())
    }

    /// Insert a decision record.
    pub fn insert_decision(&self, decision: &Decision) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        let alts_json = vec_to_json(&decision.alternatives);
        conn.execute(
            "INSERT INTO decisions (
                id, session_id, project_id, decision_type, what,
                why, alternatives, outcome, created_at, valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                decision.id,
                decision.session_id,
                decision.project_id,
                decision.decision_type,
                decision.what,
                decision.why,
                alts_json,
                decision.outcome,
                decision
                    .created_at
                    .unwrap_or_else(|| Utc::now().naive_utc()),
                decision.valid_until,
            ],
        )
        .context("failed to insert decision")?;
        Ok(())
    }

    /// Insert a tool call record.
    pub fn insert_tool_call(&self, tool_call: &ToolCall) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO tool_calls (
                id, session_id, tool_name, command, success,
                error_message, duration_ms, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                tool_call.id,
                tool_call.session_id,
                tool_call.tool_name,
                tool_call.command,
                tool_call.success,
                tool_call.error_message,
                tool_call.duration_ms,
                tool_call.timestamp,
            ],
        )
        .context("failed to insert tool_call")?;
        Ok(())
    }

    /// Insert or replace a session record (upsert).
    ///
    /// If a session with the same `id` already exists, it is replaced with
    /// the new data. This makes re-ingestion idempotent.
    pub fn insert_or_replace_session(&self, session: &Session) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        let files_json = vec_to_json(&session.files_changed);
        conn.execute(
            "INSERT OR REPLACE INTO sessions (
                id, project_id, agent, started_at, ended_at,
                duration_minutes, message_count, tool_call_count,
                total_tokens, files_changed, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                session.id,
                session.project_id,
                session.agent,
                session.started_at,
                session.ended_at,
                session.duration_minutes,
                session.message_count,
                session.tool_call_count,
                session.total_tokens,
                files_json,
                session.summary,
            ],
        )
        .context("failed to insert_or_replace session")?;
        Ok(())
    }

    /// Insert a memory record.
    pub fn insert_memory(&self, memory: &Memory) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO memories (
                id, project_id, content, memory_type, source_session_id,
                confidence, access_count, created_at, updated_at, valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                memory.id,
                memory.project_id,
                memory.content,
                memory.memory_type,
                memory.source_session_id,
                memory.confidence,
                memory.access_count,
                memory.created_at.unwrap_or_else(|| Utc::now().naive_utc()),
                memory.updated_at.unwrap_or_else(|| Utc::now().naive_utc()),
                memory.valid_until,
            ],
        )
        .context("failed to insert memory")?;
        Ok(())
    }

    /// Increment access_count for a memory (called on retrieval).
    pub fn touch_memory(&self, memory_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, updated_at = ? WHERE id = ?",
            params![Utc::now().naive_utc(), memory_id],
        )
        .context("failed to touch memory")?;
        Ok(())
    }

    /// Update a memory's content and/or confidence.
    pub fn update_memory(
        &self,
        memory_id: &str,
        new_content: Option<&str>,
        new_confidence: Option<f32>,
    ) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");
        let now = Utc::now().naive_utc();

        let (set_clause, mut values): (String, Vec<Box<dyn duckdb::ToSql>>) =
            match (new_content, new_confidence) {
                (Some(c), Some(conf)) => (
                    "content = ?, confidence = ?, updated_at = ?".into(),
                    vec![
                        Box::new(c.to_string()) as Box<dyn duckdb::ToSql>,
                        Box::new(conf),
                        Box::new(now),
                    ],
                ),
                (Some(c), None) => (
                    "content = ?, updated_at = ?".into(),
                    vec![
                        Box::new(c.to_string()) as Box<dyn duckdb::ToSql>,
                        Box::new(now),
                    ],
                ),
                (None, Some(conf)) => (
                    "confidence = ?, updated_at = ?".into(),
                    vec![Box::new(conf) as Box<dyn duckdb::ToSql>, Box::new(now)],
                ),
                (None, None) => return Ok(false),
            };

        values.push(Box::new(memory_id.to_string()));
        let sql = format!("UPDATE memories SET {set_clause} WHERE id = ?");
        let params_ref: Vec<&dyn duckdb::ToSql> = values.iter().map(|b| b.as_ref()).collect();
        let affected = conn
            .execute(&sql, params_ref.as_slice())
            .context("failed to update memory")?;
        Ok(affected > 0)
    }

    /// Delete a memory by ID.
    pub fn delete_memory(&self, memory_id: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");
        let affected = conn
            .execute("DELETE FROM memories WHERE id = ?", params![memory_id])
            .context("failed to delete memory")?;
        Ok(affected > 0)
    }

    /// Delete a fact by ID (hard delete, not invalidation).
    pub fn delete_fact(&self, fact_id: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");
        let affected = conn
            .execute("DELETE FROM facts WHERE id = ?", params![fact_id])
            .context("failed to delete fact")?;
        Ok(affected > 0)
    }

    /// Get a single memory by ID.
    pub fn get_memory(&self, memory_id: &str) -> Result<Option<Memory>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, project_id, content, memory_type, source_session_id,
                    confidence, access_count, created_at, updated_at, valid_until
             FROM memories WHERE id = ?",
        )?;
        let mut rows = stmt.query(params![memory_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                content: row.get(2)?,
                memory_type: row.get(3)?,
                source_session_id: row.get(4)?,
                confidence: row.get(5)?,
                access_count: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
                valid_until: row.get(9)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get a single fact by ID.
    pub fn get_fact(&self, fact_id: &str) -> Result<Option<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, project_id, subject, predicate, object, confidence,
                    source_session_id, source_agent, valid_at, invalid_at,
                    superseded_by, created_at
             FROM facts WHERE id = ?",
        )?;
        let mut rows = stmt.query(params![fact_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Fact {
                id: row.get(0)?,
                project_id: row.get(1)?,
                subject: row.get(2)?,
                predicate: row.get(3)?,
                object: row.get(4)?,
                confidence: row.get(5)?,
                source_session_id: row.get(6)?,
                source_agent: row.get(7)?,
                valid_at: row.get(8)?,
                invalid_at: row.get(9)?,
                superseded_by: row.get(10)?,
                created_at: row.get(11)?,
            }))
        } else {
            Ok(None)
        }
    }

    /// Upsert a project record.
    pub fn upsert_project(&self, id: &str, name: &str, path: &str) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO projects (id, name, path, updated_at)
             VALUES (?, ?, ?, ?)
             ON CONFLICT (id) DO UPDATE SET
                name = excluded.name,
                path = excluded.path,
                updated_at = excluded.updated_at",
            params![id, name, path, Utc::now().naive_utc()],
        )
        .context("failed to upsert project")?;
        Ok(())
    }

    /// Upsert file stats (change_frequency increments on each call).
    pub fn upsert_file_stat(&self, file_path: &str, project_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO file_stats (file_path, project_id, change_frequency, last_modified)
             VALUES (?, ?, 1, ?)
             ON CONFLICT (file_path, project_id) DO UPDATE SET
                change_frequency = file_stats.change_frequency + 1,
                last_modified = excluded.last_modified",
            params![file_path, project_id, Utc::now().naive_utc()],
        )
        .context("failed to upsert file_stat")?;
        Ok(())
    }

    /// Get hot files (most frequently changed) for a project.
    pub fn get_hot_files(&self, project: Option<&str>, limit: usize) -> Result<Vec<(String, i32)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut result = Vec::new();

        if let Some(proj) = project {
            let mut stmt = conn.prepare(
                "SELECT file_path, change_frequency FROM file_stats WHERE project_id = ? ORDER BY change_frequency DESC LIMIT ?"
            )?;
            let rows = stmt.query_map(params![proj, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?))
            })?;
            for row in rows {
                result.push(row?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT file_path, change_frequency FROM file_stats ORDER BY change_frequency DESC LIMIT ?"
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?))
            })?;
            for row in rows {
                result.push(row?);
            }
        }

        Ok(result)
    }

    /// Insert a fact record.
    pub fn insert_fact(&self, fact: &Fact) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO facts (
                id, project_id, subject, predicate, object, confidence,
                source_session_id, source_agent, valid_at, invalid_at,
                superseded_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                fact.id,
                fact.project_id,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.confidence,
                fact.source_session_id,
                fact.source_agent,
                fact.valid_at.unwrap_or_else(|| Utc::now().naive_utc()),
                fact.invalid_at,
                fact.superseded_by,
                fact.created_at.unwrap_or_else(|| Utc::now().naive_utc()),
            ],
        )
        .context("failed to insert fact")?;
        Ok(())
    }

    /// Invalidate a fact by setting `invalid_at` and optionally linking to successor.
    pub fn invalidate_fact(&self, fact_id: &str, superseded_by: Option<&str>) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");
        let now = Utc::now().naive_utc();
        let affected = conn
            .execute(
                "UPDATE facts SET invalid_at = ?, superseded_by = ?
                 WHERE id = ? AND invalid_at IS NULL",
                params![now, superseded_by, fact_id],
            )
            .context("failed to invalidate fact")?;
        Ok(affected > 0)
    }

    /// Insert a new fact, automatically invalidating any contradicting facts
    /// (same subject + predicate + project, still valid).
    pub fn upsert_fact(&self, fact: &Fact) -> Result<()> {
        // Find existing valid facts with the same subject+predicate
        let existing = self.get_active_facts_for_subject(
            &fact.subject,
            &fact.predicate,
            fact.project_id.as_deref(),
        )?;

        // Invalidate contradicting facts (different object value)
        for old in &existing {
            if old.object != fact.object {
                self.invalidate_fact(&old.id, Some(&fact.id))?;
            }
        }

        // If an identical fact already exists and is valid, skip insertion
        if existing.iter().any(|f| f.object == fact.object) {
            return Ok(());
        }

        self.insert_fact(fact)
    }

    /// Get all currently valid facts (invalid_at IS NULL).
    pub fn get_active_facts(&self, project: Option<&str>, limit: usize) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let (sql, use_project) = if project.is_some() {
            (
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE invalid_at IS NULL AND project_id ILIKE ?
                 ORDER BY valid_at DESC NULLS LAST
                 LIMIT ?",
                true,
            )
        } else {
            (
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE invalid_at IS NULL
                 ORDER BY valid_at DESC NULLS LAST
                 LIMIT ?",
                false,
            )
        };

        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare get_active_facts")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<Fact> {
            Ok(Fact {
                id: row.get(0)?,
                project_id: row.get(1)?,
                subject: row.get(2)?,
                predicate: row.get(3)?,
                object: row.get(4)?,
                confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                source_session_id: row.get(6)?,
                source_agent: row.get(7)?,
                valid_at: row.get(8)?,
                invalid_at: row.get(9)?,
                superseded_by: row.get(10)?,
                created_at: row.get(11)?,
            })
        };

        let rows = if use_project {
            let pattern = format!("%{}%", project.unwrap());
            stmt.query_map(params![pattern, limit as i64], map_row)
                .context("failed to query active facts")?
        } else {
            stmt.query_map(params![limit as i64], map_row)
                .context("failed to query active facts")?
        };

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Get active facts for a specific subject and predicate.
    fn get_active_facts_for_subject(
        &self,
        subject: &str,
        predicate: &str,
        project: Option<&str>,
    ) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let (sql, params_vec): (&str, Vec<Box<dyn duckdb::ToSql>>) = if let Some(proj) = project {
            (
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE invalid_at IS NULL AND subject = ? AND predicate = ? AND project_id = ?",
                vec![
                    Box::new(subject.to_string()),
                    Box::new(predicate.to_string()),
                    Box::new(proj.to_string()),
                ],
            )
        } else {
            (
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE invalid_at IS NULL AND subject = ? AND predicate = ?",
                vec![
                    Box::new(subject.to_string()),
                    Box::new(predicate.to_string()),
                ],
            )
        };

        let params_ref: Vec<&dyn duckdb::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare subject facts query")?;

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok(Fact {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    source_session_id: row.get(6)?,
                    source_agent: row.get(7)?,
                    valid_at: row.get(8)?,
                    invalid_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    created_at: row.get(11)?,
                })
            })
            .context("failed to query subject facts")?;

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Search facts by subject or object text (ILIKE).
    pub fn search_facts(&self, query: &str) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let pattern = format!("%{query}%");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE (subject ILIKE ? OR object ILIKE ?)
                 ORDER BY invalid_at IS NULL DESC, valid_at DESC NULLS LAST",
            )
            .context("failed to prepare search_facts")?;

        let rows = stmt
            .query_map(params![pattern, pattern], |row| {
                Ok(Fact {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    source_session_id: row.get(6)?,
                    source_agent: row.get(7)?,
                    valid_at: row.get(8)?,
                    invalid_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    created_at: row.get(11)?,
                })
            })
            .context("failed to query facts")?;

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Get the full temporal history for a subject (all facts, including invalidated).
    pub fn get_fact_history(&self, subject: &str) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE subject = ?
                 ORDER BY valid_at ASC NULLS LAST",
            )
            .context("failed to prepare get_fact_history")?;

        let rows = stmt
            .query_map(params![subject], |row| {
                Ok(Fact {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    source_session_id: row.get(6)?,
                    source_agent: row.get(7)?,
                    valid_at: row.get(8)?,
                    invalid_at: row.get(9)?,
                    superseded_by: row.get(10)?,
                    created_at: row.get(11)?,
                })
            })
            .context("failed to query fact history")?;

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Count facts in the database.
    pub fn count_facts(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))
            .context("failed to count facts")?;
        Ok(count as usize)
    }

    /// Count currently valid facts.
    pub fn count_active_facts(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM facts WHERE invalid_at IS NULL",
                [],
                |row| row.get(0),
            )
            .context("failed to count active facts")?;
        Ok(count as usize)
    }

    /// Insert a code symbol record.
    pub fn insert_code_symbol(&self, symbol: &CodeSymbol) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO code_symbols (
                id, project_id, file_path, symbol_name, symbol_kind,
                signature, docstring, start_line, end_line, visibility,
                parent_symbol, pagerank_score, reference_count, language,
                content_hash, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                symbol.id,
                symbol.project_id,
                symbol.file_path,
                symbol.symbol_name,
                symbol.symbol_kind,
                symbol.signature,
                symbol.docstring,
                symbol.start_line,
                symbol.end_line,
                symbol.visibility,
                symbol.parent_symbol,
                symbol.pagerank_score,
                symbol.reference_count,
                symbol.language,
                symbol.content_hash,
                symbol.indexed_at.unwrap_or_else(|| Utc::now().naive_utc()),
            ],
        )
        .context("failed to insert code_symbol")?;
        Ok(())
    }

    /// Insert a code dependency record.
    pub fn insert_code_dependency(&self, dep: &CodeDependency) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT INTO code_dependencies (
                id, project_id, from_symbol, to_symbol, relationship,
                from_file, to_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                dep.id,
                dep.project_id,
                dep.from_symbol,
                dep.to_symbol,
                dep.relationship,
                dep.from_file,
                dep.to_file,
            ],
        )
        .context("failed to insert code_dependency")?;
        Ok(())
    }

    /// Insert an analysis run record.
    pub fn insert_analysis_run(&self, run: &AnalysisRun) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO code_analysis_runs (
                project_id, commit_hash, files_analyzed, symbols_extracted,
                dependencies_found, chunks_generated, duration_ms, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                run.project_id,
                run.commit_hash,
                run.files_analyzed,
                run.symbols_extracted,
                run.dependencies_found,
                run.chunks_generated,
                run.duration_ms,
                run.analyzed_at.unwrap_or_else(|| Utc::now().naive_utc()),
            ],
        )
        .context("failed to insert analysis_run")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return the most recent sessions ordered by `started_at` descending.
    pub fn get_recent_sessions(&self, limit: usize) -> Result<Vec<Session>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, agent, started_at, ended_at,
                        duration_minutes, message_count, tool_call_count,
                        total_tokens, files_changed, summary
                 FROM sessions
                 ORDER BY started_at DESC NULLS LAST
                 LIMIT ?",
            )
            .context("failed to prepare get_recent_sessions")?;

        let rows = stmt
            .query_map(params![limit as i64], |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                Ok(Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                })
            })
            .context("failed to query recent sessions")?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.context("failed to read session row")?);
        }
        Ok(sessions)
    }

    /// Search sessions by agent, project, or since date.
    pub fn search_sessions(
        &self,
        agent: Option<&str>,
        project: Option<&str>,
        since: Option<NaiveDateTime>,
        limit: usize,
    ) -> Result<Vec<Session>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut conditions = Vec::new();
        let mut param_values: Vec<Box<dyn duckdb::ToSql>> = Vec::new();

        if let Some(agent) = agent {
            conditions.push("agent ILIKE ?".to_string());
            param_values.push(Box::new(format!("%{agent}%")));
        }
        if let Some(project) = project {
            conditions.push("project_id ILIKE ?".to_string());
            param_values.push(Box::new(format!("%{project}%")));
        }
        if let Some(since) = since {
            conditions.push("started_at >= ?".to_string());
            param_values.push(Box::new(since));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT id, project_id, agent, started_at, ended_at,
                    duration_minutes, message_count, tool_call_count,
                    total_tokens, files_changed, summary
             FROM sessions
             {where_clause}
             ORDER BY started_at DESC NULLS LAST
             LIMIT ?"
        );

        param_values.push(Box::new(limit as i64));

        let params_ref: Vec<&dyn duckdb::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .context("failed to prepare search_sessions")?;

        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                Ok(Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                })
            })
            .context("failed to query sessions")?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.context("failed to read session row")?);
        }
        Ok(sessions)
    }

    /// Get sessions from today (last 24 hours).
    pub fn get_todays_sessions(&self) -> Result<Vec<Session>> {
        let since = Utc::now().naive_utc() - chrono::Duration::hours(24);
        self.search_sessions(None, None, Some(since), 1000)
    }

    /// Get tool calls for a session.
    pub fn get_tool_calls_for_session(&self, session_id: &str) -> Result<Vec<ToolCall>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, tool_name, command, success,
                        error_message, duration_ms, timestamp
                 FROM tool_calls
                 WHERE session_id = ?
                 ORDER BY timestamp ASC NULLS LAST",
            )
            .context("failed to prepare get_tool_calls_for_session")?;

        let rows = stmt
            .query_map(params![session_id], |row| {
                Ok(ToolCall {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    tool_name: row.get(2)?,
                    command: row.get(3)?,
                    success: row.get(4)?,
                    error_message: row.get(5)?,
                    duration_ms: row.get(6)?,
                    timestamp: row.get(7)?,
                })
            })
            .context("failed to query tool calls")?;

        let mut tool_calls = Vec::new();
        for row in rows {
            tool_calls.push(row.context("failed to read tool_call row")?);
        }
        Ok(tool_calls)
    }

    /// Get all memories, optionally filtered by project.
    pub fn get_memories(&self, project: Option<&str>, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let (sql, use_project) = if project.is_some() {
            (
                "SELECT id, project_id, content, memory_type, source_session_id,
                        confidence, access_count, created_at, updated_at, valid_until
                 FROM memories
                 WHERE project_id ILIKE ?
                 ORDER BY updated_at DESC NULLS LAST
                 LIMIT ?",
                true,
            )
        } else {
            (
                "SELECT id, project_id, content, memory_type, source_session_id,
                        confidence, access_count, created_at, updated_at, valid_until
                 FROM memories
                 ORDER BY updated_at DESC NULLS LAST
                 LIMIT ?",
                false,
            )
        };

        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare get_memories")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<Memory> {
            Ok(Memory {
                id: row.get(0)?,
                project_id: row.get(1)?,
                content: row.get(2)?,
                memory_type: row.get(3)?,
                source_session_id: row.get(4)?,
                confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                access_count: row.get::<_, i32>(6).unwrap_or(0),
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
                valid_until: row.get(9)?,
            })
        };

        let rows = if use_project {
            let pattern = format!("%{}%", project.unwrap());
            stmt.query_map(params![pattern, limit as i64], map_row)
                .context("failed to query memories")?
        } else {
            stmt.query_map(params![limit as i64], map_row)
                .context("failed to query memories")?
        };

        let mut memories = Vec::new();
        for row in rows {
            memories.push(row.context("failed to read memory row")?);
        }
        Ok(memories)
    }

    /// Count sessions in the database.
    pub fn count_sessions(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .context("failed to count sessions")?;
        Ok(count as usize)
    }

    /// Count memories in the database.
    pub fn count_memories(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .context("failed to count memories")?;
        Ok(count as usize)
    }

    /// Count tool calls in the database.
    pub fn count_tool_calls(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM tool_calls", [], |row| row.get(0))
            .context("failed to count tool_calls")?;
        Ok(count as usize)
    }

    /// Search sessions by summary ILIKE.
    pub fn search_sessions_by_summary(&self, query: &str) -> Result<Vec<Session>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let pattern = format!("%{query}%");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, agent, started_at, ended_at,
                        duration_minutes, message_count, tool_call_count,
                        total_tokens, files_changed, summary
                 FROM sessions
                 WHERE summary ILIKE ?
                 ORDER BY started_at DESC NULLS LAST",
            )
            .context("failed to prepare search_sessions_by_summary")?;

        let rows = stmt
            .query_map(params![pattern], |row| {
                let files_str: String = row.get::<_, String>(9).unwrap_or_default();
                Ok(Session {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    agent: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    duration_minutes: row.get(5)?,
                    message_count: row.get(6)?,
                    tool_call_count: row.get(7)?,
                    total_tokens: row.get(8)?,
                    files_changed: json_to_vec(&files_str),
                    summary: row.get(10)?,
                })
            })
            .context("failed to query sessions by summary")?;

        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.context("failed to read session row")?);
        }
        Ok(sessions)
    }

    /// Count decisions in the database.
    pub fn count_decisions(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM decisions", [], |row| row.get(0))
            .context("failed to count decisions")?;
        Ok(count as usize)
    }

    /// Get all decisions, optionally filtered by project.
    pub fn get_decisions(&self, project: Option<&str>, limit: usize) -> Result<Vec<Decision>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let (sql, use_project) = if project.is_some() {
            (
                "SELECT id, session_id, project_id, decision_type, what,
                        why, alternatives, outcome, created_at, valid_until
                 FROM decisions
                 WHERE project_id ILIKE ?
                 ORDER BY created_at DESC NULLS LAST
                 LIMIT ?",
                true,
            )
        } else {
            (
                "SELECT id, session_id, project_id, decision_type, what,
                        why, alternatives, outcome, created_at, valid_until
                 FROM decisions
                 ORDER BY created_at DESC NULLS LAST
                 LIMIT ?",
                false,
            )
        };

        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare get_decisions")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<Decision> {
            let alts_str: String = row.get::<_, String>(6).unwrap_or_default();
            Ok(Decision {
                id: row.get(0)?,
                session_id: row.get(1)?,
                project_id: row.get(2)?,
                decision_type: row.get(3)?,
                what: row.get(4)?,
                why: row.get(5)?,
                alternatives: json_to_vec(&alts_str),
                outcome: row.get(7)?,
                created_at: row.get(8)?,
                valid_until: row.get(9)?,
            })
        };

        let rows = if use_project {
            let pattern = format!("%{}%", project.unwrap());
            stmt.query_map(params![pattern, limit as i64], map_row)
                .context("failed to query decisions")?
        } else {
            stmt.query_map(params![limit as i64], map_row)
                .context("failed to query decisions")?
        };

        let mut decisions = Vec::new();
        for row in rows {
            decisions.push(row.context("failed to read decision row")?);
        }
        Ok(decisions)
    }

    /// Get all sessions for a specific project.
    pub fn get_project_sessions(&self, project_id: &str) -> Result<Vec<Session>> {
        self.search_sessions(None, Some(project_id), None, 10_000)
    }

    /// Get distinct project IDs from sessions.
    pub fn get_project_ids(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT project_id FROM sessions
                 WHERE project_id IS NOT NULL
                 ORDER BY project_id",
            )
            .context("failed to prepare get_project_ids")?;

        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .context("failed to query project IDs")?;

        let mut ids = Vec::new();
        for row in rows {
            ids.push(row.context("failed to read project_id row")?);
        }
        Ok(ids)
    }

    /// Delete a session and its related tool calls.
    pub fn delete_session(&self, session_id: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "DELETE FROM tool_calls WHERE session_id = ?",
            params![session_id],
        )
        .context("failed to delete tool_calls for session")?;

        let affected = conn
            .execute("DELETE FROM sessions WHERE id = ?", params![session_id])
            .context("failed to delete session")?;

        Ok(affected > 0)
    }

    /// Delete old sessions before a given date. Returns number deleted.
    pub fn gc_sessions_before(&self, before: NaiveDateTime) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");

        // First delete tool_calls for those sessions
        conn.execute(
            "DELETE FROM tool_calls WHERE session_id IN (
                SELECT id FROM sessions WHERE started_at < ?
            )",
            params![before],
        )
        .context("failed to delete old tool_calls")?;

        let affected = conn
            .execute("DELETE FROM sessions WHERE started_at < ?", params![before])
            .context("failed to delete old sessions")?;

        Ok(affected)
    }

    /// Insert a note as a memory.
    pub fn insert_note(&self, content: &str, project: Option<&str>) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().naive_utc();
        let memory = Memory {
            id: id.clone(),
            project_id: project.map(|s| s.to_string()),
            content: content.to_string(),
            memory_type: Some("note".to_string()),
            source_session_id: None,
            confidence: 1.0,
            access_count: 0,
            created_at: Some(now),
            updated_at: Some(now),
            valid_until: None,
        };
        self.insert_memory(&memory)?;
        Ok(id)
    }

    /// Get per-agent session counts.
    pub fn get_agent_session_counts(&self) -> Result<Vec<(String, usize)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare("SELECT agent, COUNT(*) FROM sessions GROUP BY agent ORDER BY COUNT(*) DESC")
            .context("failed to prepare agent session counts")?;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query agent session counts")?;

        let mut counts = Vec::new();
        for row in rows {
            let (agent, count) = row.context("failed to read agent count row")?;
            counts.push((agent, count as usize));
        }
        Ok(counts)
    }

    /// Get per-project session counts.
    pub fn get_project_session_counts(&self) -> Result<Vec<(String, usize)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT COALESCE(project_id, '(unknown)'), COUNT(*)
                 FROM sessions
                 GROUP BY project_id
                 ORDER BY COUNT(*) DESC",
            )
            .context("failed to prepare project session counts")?;

        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query project session counts")?;

        let mut counts = Vec::new();
        for row in rows {
            let (project, count) = row.context("failed to read project count row")?;
            counts.push((project, count as usize));
        }
        Ok(counts)
    }

    /// Full-text search over memory content using DuckDB `ILIKE`.
    pub fn search_memories(&self, query: &str) -> Result<Vec<Memory>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let pattern = format!("%{query}%");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, content, memory_type, source_session_id,
                        confidence, access_count, created_at, updated_at, valid_until
                 FROM memories
                 WHERE content ILIKE ?
                 ORDER BY updated_at DESC NULLS LAST",
            )
            .context("failed to prepare search_memories")?;

        let rows = stmt
            .query_map(params![pattern], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    content: row.get(2)?,
                    memory_type: row.get(3)?,
                    source_session_id: row.get(4)?,
                    confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                    access_count: row.get::<_, i32>(6).unwrap_or(0),
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    valid_until: row.get(9)?,
                })
            })
            .context("failed to query memories")?;

        let mut memories = Vec::new();
        for row in rows {
            memories.push(row.context("failed to read memory row")?);
        }
        Ok(memories)
    }

    // -----------------------------------------------------------------------
    // Code analysis queries
    // -----------------------------------------------------------------------

    /// Get all symbols for a project.
    pub fn get_symbols_for_project(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<CodeSymbol>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, file_path, symbol_name, symbol_kind,
                        signature, docstring, start_line, end_line, visibility,
                        parent_symbol, pagerank_score, reference_count, language,
                        content_hash, indexed_at
                 FROM code_symbols
                 WHERE project_id = ?
                 ORDER BY file_path, start_line
                 LIMIT ?",
            )
            .context("failed to prepare get_symbols_for_project")?;

        let rows = stmt
            .query_map(params![project_id, limit as i64], |row| {
                Ok(CodeSymbol {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    file_path: row.get(2)?,
                    symbol_name: row.get(3)?,
                    symbol_kind: row.get(4)?,
                    signature: row.get(5)?,
                    docstring: row.get(6)?,
                    start_line: row.get(7)?,
                    end_line: row.get(8)?,
                    visibility: row.get(9)?,
                    parent_symbol: row.get(10)?,
                    pagerank_score: row.get::<_, f64>(11).unwrap_or(0.0),
                    reference_count: row.get::<_, i32>(12).unwrap_or(0),
                    language: row.get(13)?,
                    content_hash: row.get(14)?,
                    indexed_at: row.get(15)?,
                })
            })
            .context("failed to query symbols for project")?;

        let mut symbols = Vec::new();
        for row in rows {
            symbols.push(row.context("failed to read code_symbol row")?);
        }
        Ok(symbols)
    }

    /// Get top symbols by pagerank score.
    pub fn get_top_symbols(&self, project_id: &str, limit: usize) -> Result<Vec<CodeSymbol>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, file_path, symbol_name, symbol_kind,
                        signature, docstring, start_line, end_line, visibility,
                        parent_symbol, pagerank_score, reference_count, language,
                        content_hash, indexed_at
                 FROM code_symbols
                 WHERE project_id = ?
                 ORDER BY pagerank_score DESC
                 LIMIT ?",
            )
            .context("failed to prepare get_top_symbols")?;

        let rows = stmt
            .query_map(params![project_id, limit as i64], |row| {
                Ok(CodeSymbol {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    file_path: row.get(2)?,
                    symbol_name: row.get(3)?,
                    symbol_kind: row.get(4)?,
                    signature: row.get(5)?,
                    docstring: row.get(6)?,
                    start_line: row.get(7)?,
                    end_line: row.get(8)?,
                    visibility: row.get(9)?,
                    parent_symbol: row.get(10)?,
                    pagerank_score: row.get::<_, f64>(11).unwrap_or(0.0),
                    reference_count: row.get::<_, i32>(12).unwrap_or(0),
                    language: row.get(13)?,
                    content_hash: row.get(14)?,
                    indexed_at: row.get(15)?,
                })
            })
            .context("failed to query top symbols")?;

        let mut symbols = Vec::new();
        for row in rows {
            symbols.push(row.context("failed to read code_symbol row")?);
        }
        Ok(symbols)
    }

    /// Get callers of a symbol (dependencies where this symbol is the target).
    pub fn get_callers_of(
        &self,
        symbol_name: &str,
        project_id: &str,
    ) -> Result<Vec<CodeDependency>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, from_symbol, to_symbol, relationship,
                        from_file, to_file
                 FROM code_dependencies
                 WHERE project_id = ? AND to_symbol = ?",
            )
            .context("failed to prepare get_callers_of")?;

        let rows = stmt
            .query_map(params![project_id, symbol_name], |row| {
                Ok(CodeDependency {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    from_symbol: row.get(2)?,
                    to_symbol: row.get(3)?,
                    relationship: row.get(4)?,
                    from_file: row.get(5)?,
                    to_file: row.get(6)?,
                })
            })
            .context("failed to query callers")?;

        let mut deps = Vec::new();
        for row in rows {
            deps.push(row.context("failed to read code_dependency row")?);
        }
        Ok(deps)
    }

    /// Get callees of a symbol (dependencies where this symbol is the source).
    pub fn get_callees_of(
        &self,
        symbol_name: &str,
        project_id: &str,
    ) -> Result<Vec<CodeDependency>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, from_symbol, to_symbol, relationship,
                        from_file, to_file
                 FROM code_dependencies
                 WHERE project_id = ? AND from_symbol = ?",
            )
            .context("failed to prepare get_callees_of")?;

        let rows = stmt
            .query_map(params![project_id, symbol_name], |row| {
                Ok(CodeDependency {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    from_symbol: row.get(2)?,
                    to_symbol: row.get(3)?,
                    relationship: row.get(4)?,
                    from_file: row.get(5)?,
                    to_file: row.get(6)?,
                })
            })
            .context("failed to query callees")?;

        let mut deps = Vec::new();
        for row in rows {
            deps.push(row.context("failed to read code_dependency row")?);
        }
        Ok(deps)
    }

    /// Get all symbols in a specific file.
    pub fn get_symbols_in_file(
        &self,
        file_path: &str,
        project_id: &str,
    ) -> Result<Vec<CodeSymbol>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, file_path, symbol_name, symbol_kind,
                        signature, docstring, start_line, end_line, visibility,
                        parent_symbol, pagerank_score, reference_count, language,
                        content_hash, indexed_at
                 FROM code_symbols
                 WHERE project_id = ? AND file_path = ?
                 ORDER BY start_line",
            )
            .context("failed to prepare get_symbols_in_file")?;

        let rows = stmt
            .query_map(params![project_id, file_path], |row| {
                Ok(CodeSymbol {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    file_path: row.get(2)?,
                    symbol_name: row.get(3)?,
                    symbol_kind: row.get(4)?,
                    signature: row.get(5)?,
                    docstring: row.get(6)?,
                    start_line: row.get(7)?,
                    end_line: row.get(8)?,
                    visibility: row.get(9)?,
                    parent_symbol: row.get(10)?,
                    pagerank_score: row.get::<_, f64>(11).unwrap_or(0.0),
                    reference_count: row.get::<_, i32>(12).unwrap_or(0),
                    language: row.get(13)?,
                    content_hash: row.get(14)?,
                    indexed_at: row.get(15)?,
                })
            })
            .context("failed to query symbols in file")?;

        let mut symbols = Vec::new();
        for row in rows {
            symbols.push(row.context("failed to read code_symbol row")?);
        }
        Ok(symbols)
    }

    /// Count symbols in the database.
    pub fn count_symbols(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM code_symbols", [], |row| row.get(0))
            .context("failed to count symbols")?;
        Ok(count as usize)
    }

    /// Count dependencies in the database.
    pub fn count_dependencies(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM code_dependencies", [], |row| {
                row.get(0)
            })
            .context("failed to count dependencies")?;
        Ok(count as usize)
    }

    /// Get the last analysis run for a project.
    pub fn get_last_analysis(&self, project_id: &str) -> Result<Option<AnalysisRun>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT project_id, commit_hash, files_analyzed, symbols_extracted,
                        dependencies_found, chunks_generated, duration_ms, analyzed_at
                 FROM code_analysis_runs
                 WHERE project_id = ?",
            )
            .context("failed to prepare get_last_analysis")?;

        let result = stmt.query_row(params![project_id], |row| {
            Ok(AnalysisRun {
                project_id: row.get(0)?,
                commit_hash: row.get(1)?,
                files_analyzed: row.get::<_, i32>(2).unwrap_or(0),
                symbols_extracted: row.get::<_, i32>(3).unwrap_or(0),
                dependencies_found: row.get::<_, i32>(4).unwrap_or(0),
                chunks_generated: row.get::<_, i32>(5).unwrap_or(0),
                duration_ms: row.get::<_, i32>(6).unwrap_or(0),
                analyzed_at: row.get(7)?,
            })
        });

        match result {
            Ok(run) => Ok(Some(run)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(anyhow::Error::new(e).context("failed to get last analysis")),
        }
    }

    // -----------------------------------------------------------------------
    // Graph: DuckPGQ extension and property graph definition
    // -----------------------------------------------------------------------

    /// Load the DuckPGQ community extension.
    /// Returns a descriptive error if the extension is not available.
    pub fn load_duckpgq(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute_batch("INSTALL duckpgq FROM community; LOAD duckpgq;")
            .context(
                "failed to load DuckPGQ extension — \
                 make sure the extension is available (INSTALL duckpgq FROM community)",
            )?;
        Ok(())
    }

    /// Initialise the DuckPGQ property graph over `graph_nodes` / `graph_edges`.
    ///
    /// This must be called **after** `init_schema()` (which creates the tables)
    /// and is intentionally separate because the DuckPGQ extension may not be
    /// available in all environments (tests, CI).
    pub fn init_graph(&self) -> Result<()> {
        self.load_duckpgq()?;

        let conn = self.conn.lock().expect("lock poisoned");
        // DuckPGQ syntax: define vertex / edge tables for the property graph.
        // NOTE: DuckPGQ does not support `IF NOT EXISTS` on CREATE PROPERTY GRAPH,
        // so we use DROP IF EXISTS first for idempotency.
        conn.execute_batch(
            "DROP PROPERTY GRAPH IF EXISTS remembrant_graph;
             CREATE PROPERTY GRAPH remembrant_graph
             VERTEX TABLES (graph_nodes)
             EDGE TABLES (graph_edges SOURCE KEY (from_id) REFERENCES graph_nodes (id)
                                      DESTINATION KEY (to_id) REFERENCES graph_nodes (id));",
        )
        .context("failed to create property graph definition")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Graph CRUD (standard SQL — no extension required)
    // -----------------------------------------------------------------------

    /// Insert or replace a graph node (upsert).
    pub fn insert_graph_node(
        &self,
        id: &str,
        kind: &str,
        name: &str,
        properties: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO graph_nodes (id, kind, name, properties)
             VALUES (?, ?, ?, ?)",
            params![id, kind, name, properties],
        )
        .context("failed to insert graph node")?;
        Ok(())
    }

    /// Insert a graph edge. Duplicates (same from_id, to_id, kind) are ignored.
    pub fn insert_graph_edge(
        &self,
        from_id: &str,
        to_id: &str,
        kind: &str,
        properties: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");

        // Check for duplicate edge
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM graph_edges
                 WHERE from_id = ? AND to_id = ? AND kind = ?",
                params![from_id, to_id, kind],
                |row| row.get(0),
            )
            .unwrap_or(false);

        if exists {
            return Ok(());
        }

        let edge_id = uuid::Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO graph_edges (id, from_id, to_id, kind, properties)
             VALUES (?, ?, ?, ?, ?)",
            params![edge_id, from_id, to_id, kind, properties],
        )
        .context("failed to insert graph edge")?;
        Ok(())
    }

    /// Get a graph node by ID.
    pub fn get_graph_node(&self, id: &str) -> Result<Option<GraphNodeRow>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let result = conn.query_row(
            "SELECT id, kind, name, properties FROM graph_nodes WHERE id = ?",
            params![id],
            |row| {
                Ok(GraphNodeRow {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    name: row.get(2)?,
                    properties: row.get::<_, String>(3).unwrap_or_else(|_| "{}".to_string()),
                })
            },
        );
        match result {
            Ok(node) => Ok(Some(node)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(anyhow::Error::new(e).context("failed to get graph node")),
        }
    }

    /// Delete a graph node and all its incident edges. Returns `true` if the
    /// node existed.
    pub fn delete_graph_node(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().expect("lock poisoned");

        // Delete incident edges first (to honour foreign key semantics).
        conn.execute(
            "DELETE FROM graph_edges WHERE from_id = ? OR to_id = ?",
            params![id, id],
        )
        .context("failed to delete incident graph edges")?;

        let affected = conn
            .execute("DELETE FROM graph_nodes WHERE id = ?", params![id])
            .context("failed to delete graph node")?;

        Ok(affected > 0)
    }

    /// Query all neighbors of `id`, optionally filtered by edge kind.
    /// Uses standard SQL (no PGQ extension needed).
    pub fn query_graph_neighbors(
        &self,
        id: &str,
        edge_kind: Option<&str>,
    ) -> Result<Vec<GraphNeighborRow>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let (sql, use_kind) = if edge_kind.is_some() {
            (
                "SELECT n.id, n.kind, n.name, n.properties, e.kind AS edge_kind, 'outgoing' AS direction
                 FROM graph_edges e JOIN graph_nodes n ON e.to_id = n.id
                 WHERE e.from_id = ? AND e.kind = ?
                 UNION ALL
                 SELECT n.id, n.kind, n.name, n.properties, e.kind AS edge_kind, 'incoming' AS direction
                 FROM graph_edges e JOIN graph_nodes n ON e.from_id = n.id
                 WHERE e.to_id = ? AND e.kind = ?",
                true,
            )
        } else {
            (
                "SELECT n.id, n.kind, n.name, n.properties, e.kind AS edge_kind, 'outgoing' AS direction
                 FROM graph_edges e JOIN graph_nodes n ON e.to_id = n.id
                 WHERE e.from_id = ?
                 UNION ALL
                 SELECT n.id, n.kind, n.name, n.properties, e.kind AS edge_kind, 'incoming' AS direction
                 FROM graph_edges e JOIN graph_nodes n ON e.from_id = n.id
                 WHERE e.to_id = ?",
                false,
            )
        };

        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare neighbor query")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<GraphNeighborRow> {
            Ok(GraphNeighborRow {
                node: GraphNodeRow {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    name: row.get(2)?,
                    properties: row.get::<_, String>(3).unwrap_or_else(|_| "{}".to_string()),
                },
                edge_kind: row.get(4)?,
                direction: row.get(5)?,
            })
        };

        let rows = if use_kind {
            let ek = edge_kind.unwrap();
            stmt.query_map(params![id, ek, id, ek], map_row)
                .context("failed to query graph neighbors")?
        } else {
            stmt.query_map(params![id, id], map_row)
                .context("failed to query graph neighbors")?
        };

        let mut neighbors = Vec::new();
        for row in rows {
            neighbors.push(row.context("failed to read graph neighbor row")?);
        }
        Ok(neighbors)
    }

    /// Count graph nodes.
    pub fn count_graph_nodes(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0))
            .context("failed to count graph nodes")?;
        Ok(count as usize)
    }

    /// Count graph edges.
    pub fn count_graph_edges(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_edges", [], |row| row.get(0))
            .context("failed to count graph edges")?;
        Ok(count as usize)
    }

    /// Delete all graph nodes and edges.
    pub fn clear_graph(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lock poisoned");
        conn.execute_batch("DELETE FROM graph_edges; DELETE FROM graph_nodes;")
            .context("failed to clear graph")?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Graph PGQ queries (require DuckPGQ extension)
    // -----------------------------------------------------------------------

    /// Find the shortest path between two nodes (up to `max_depth` hops).
    /// Returns the sequence of node IDs along the path, or an empty vec if no
    /// path exists. Requires the DuckPGQ extension to be loaded via
    /// `init_graph()`.
    ///
    /// DuckPGQ SQL/PGQ syntax (intended):
    /// ```sql
    /// SELECT path_nodes
    /// FROM GRAPH_TABLE(remembrant_graph
    ///     MATCH p = ANY SHORTEST (a:node WHERE a.id = ?)-[e:edge]->{1,N}(b:node WHERE b.id = ?)
    ///     COLUMNS (path_nodes(p))
    /// )
    /// ```
    pub fn pgq_shortest_path(
        &self,
        from_id: &str,
        to_id: &str,
        max_depth: usize,
    ) -> Result<Vec<String>> {
        if from_id == to_id {
            return Ok(vec![from_id.to_string()]);
        }

        let conn = self.conn.lock().expect("lock poisoned");

        // DuckPGQ shortest path query.
        // The exact syntax may vary across DuckPGQ versions; if this fails the
        // caller gets a descriptive error.
        let sql = format!(
            "FROM GRAPH_TABLE(remembrant_graph
                 MATCH p = ANY SHORTEST (a:graph_nodes WHERE a.id = ?)-[e:graph_edges]->{{1,{max_depth}}}(b:graph_nodes WHERE b.id = ?)
                 COLUMNS (vertices(p) AS path_vertices)
             )"
        );

        let result = conn.query_row(&sql, params![from_id, to_id], |row| {
            // DuckPGQ returns path vertices as a LIST; extract as a string
            // representation and parse, or read directly if the driver supports
            // list types.
            let raw: String = row.get(0)?;
            Ok(raw)
        });

        match result {
            Ok(raw) => {
                // Parse the returned list representation.  DuckPGQ returns
                // something like `[{id: a, ...}, {id: b, ...}]` — extract IDs.
                let ids: Vec<String> = raw
                    .split("id: ")
                    .skip(1)
                    .filter_map(|s| s.split([',', '}'].as_ref()).next())
                    .map(|s| s.trim().to_string())
                    .collect();
                Ok(ids)
            }
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(Vec::new()),
            Err(e) => Err(anyhow::Error::new(e)
                .context("DuckPGQ shortest_path query failed — ensure init_graph() was called")),
        }
    }

    /// Run PageRank on the graph and return the top `limit` nodes with scores.
    /// Requires the DuckPGQ extension to be loaded via `init_graph()`.
    ///
    /// DuckPGQ syntax (intended):
    /// ```sql
    /// SELECT node_id, pagerank_score
    /// FROM pagerank(remembrant_graph, graph_nodes, graph_edges)
    /// ORDER BY pagerank_score DESC LIMIT ?
    /// ```
    pub fn pgq_pagerank(&self, limit: usize) -> Result<Vec<(String, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let mut stmt = conn
            .prepare(
                "SELECT node_id, pagerank_score
                 FROM pagerank(remembrant_graph, graph_nodes, graph_edges)
                 ORDER BY pagerank_score DESC
                 LIMIT ?",
            )
            .context("DuckPGQ pagerank query failed — ensure init_graph() was called")?;

        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })
            .context("failed to execute pagerank query")?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read pagerank row")?);
        }
        Ok(results)
    }

    /// Pattern match: find all nodes connected to `node_id` via a specific
    /// edge kind in the given direction ("outgoing" or "incoming").
    /// Requires the DuckPGQ extension to be loaded via `init_graph()`.
    pub fn pgq_pattern_match(
        &self,
        node_id: &str,
        edge_kind: &str,
        direction: &str,
    ) -> Result<Vec<GraphNodeRow>> {
        let conn = self.conn.lock().expect("lock poisoned");

        // Use SQL/PGQ MATCH with a WHERE filter on edge kind.
        let sql = if direction == "outgoing" {
            "FROM GRAPH_TABLE(remembrant_graph
                 MATCH (a:graph_nodes WHERE a.id = ?)-[e:graph_edges WHERE e.kind = ?]->(b:graph_nodes)
                 COLUMNS (b.id, b.kind, b.name, b.properties)
             )"
        } else {
            "FROM GRAPH_TABLE(remembrant_graph
                 MATCH (a:graph_nodes)<-[e:graph_edges WHERE e.kind = ?]-(b:graph_nodes WHERE b.id = ?)
                 COLUMNS (a.id, a.kind, a.name, a.properties)
             )"
        };

        let mut stmt = conn
            .prepare(sql)
            .context("DuckPGQ pattern_match query failed — ensure init_graph() was called")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<GraphNodeRow> {
            Ok(GraphNodeRow {
                id: row.get(0)?,
                kind: row.get(1)?,
                name: row.get(2)?,
                properties: row.get::<_, String>(3).unwrap_or_else(|_| "{}".to_string()),
            })
        };

        let rows = if direction == "outgoing" {
            stmt.query_map(params![node_id, edge_kind], map_row)
                .context("failed to execute pattern match")?
        } else {
            stmt.query_map(params![edge_kind, node_id], map_row)
                .context("failed to execute pattern match")?
        };

        let mut results = Vec::new();
        for row in rows {
            results.push(row.context("failed to read pattern match row")?);
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Code analysis (existing)
    // -----------------------------------------------------------------------

    /// Clear all symbols for a project. Returns number of symbols deleted.
    pub fn clear_symbols_for_project(&self, project_id: &str) -> Result<usize> {
        let conn = self.conn.lock().expect("lock poisoned");

        // Delete dependencies first
        conn.execute(
            "DELETE FROM code_dependencies WHERE project_id = ?",
            params![project_id],
        )
        .context("failed to delete code_dependencies for project")?;

        // Delete symbols
        let affected = conn
            .execute(
                "DELETE FROM code_symbols WHERE project_id = ?",
                params![project_id],
            )
            .context("failed to delete code_symbols for project")?;

        Ok(affected)
    }

    /// Aggregate tool call statistics across all sessions.
    pub fn get_tool_call_stats(&self) -> Result<Vec<(String, i64, i64, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT COALESCE(tool_name, '(unknown)') AS tn,
                    COUNT(*) AS cnt,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) AS ok,
                    AVG(duration_ms) AS avg_dur
             FROM tool_calls
             GROUP BY tn
             ORDER BY cnt DESC",
            )
            .context("failed to prepare tool call stats")?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f64>(3).unwrap_or(0.0),
                ))
            })
            .context("failed to query tool call stats")?;

        let mut stats = Vec::new();
        for row in rows {
            stats.push(row.context("failed to read tool call stats row")?);
        }
        Ok(stats)
    }

    /// Get daily session counts grouped by agent, for the last N days.
    pub fn get_session_timeline(
        &self,
        days: i64,
        agent: Option<&str>,
    ) -> Result<Vec<(String, String, i64)>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let cutoff = Utc::now().naive_utc() - chrono::Duration::days(days);

        if let Some(ag) = agent {
            let mut stmt = conn
                .prepare(
                    "SELECT CAST(started_at AS DATE) AS day, agent, COUNT(*) AS cnt
                 FROM sessions
                 WHERE started_at >= ? AND agent = ?
                 GROUP BY day, agent
                 ORDER BY day",
                )
                .context("failed to prepare session timeline")?;
            let rows = stmt
                .query_map(params![cutoff, ag], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })
                .context("failed to query session timeline")?;
            let mut result = Vec::new();
            for row in rows {
                result.push(row.context("failed to read timeline row")?);
            }
            Ok(result)
        } else {
            let mut stmt = conn
                .prepare(
                    "SELECT CAST(started_at AS DATE) AS day, agent, COUNT(*) AS cnt
                 FROM sessions
                 WHERE started_at >= ?
                 GROUP BY day, agent
                 ORDER BY day",
                )
                .context("failed to prepare session timeline")?;
            let rows = stmt
                .query_map(params![cutoff], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })
                .context("failed to query session timeline")?;
            let mut result = Vec::new();
            for row in rows {
                result.push(row.context("failed to read timeline row")?);
            }
            Ok(result)
        }
    }

    /// Get all facts (active + invalidated), optionally filtered by project.
    pub fn get_all_facts(&self, project: Option<&str>, limit: usize) -> Result<Vec<Fact>> {
        let conn = self.conn.lock().expect("lock poisoned");

        let (sql, use_project) = if project.is_some() {
            (
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 WHERE project_id ILIKE ?
                 ORDER BY created_at DESC NULLS LAST
                 LIMIT ?",
                true,
            )
        } else {
            (
                "SELECT id, project_id, subject, predicate, object, confidence,
                        source_session_id, source_agent, valid_at, invalid_at,
                        superseded_by, created_at
                 FROM facts
                 ORDER BY created_at DESC NULLS LAST
                 LIMIT ?",
                false,
            )
        };

        let mut stmt = conn
            .prepare(sql)
            .context("failed to prepare get_all_facts")?;

        let map_row = |row: &duckdb::Row| -> duckdb::Result<Fact> {
            Ok(Fact {
                id: row.get(0)?,
                project_id: row.get(1)?,
                subject: row.get(2)?,
                predicate: row.get(3)?,
                object: row.get(4)?,
                confidence: row.get::<_, f32>(5).unwrap_or(1.0),
                source_session_id: row.get(6)?,
                source_agent: row.get(7)?,
                valid_at: row.get(8)?,
                invalid_at: row.get(9)?,
                superseded_by: row.get(10)?,
                created_at: row.get(11)?,
            })
        };

        let rows = if use_project {
            let pattern = format!("%{}%", project.unwrap());
            stmt.query_map(params![pattern, limit as i64], map_row)
                .context("failed to query all facts")?
        } else {
            stmt.query_map(params![limit as i64], map_row)
                .context("failed to query all facts")?
        };

        let mut facts = Vec::new();
        for row in rows {
            facts.push(row.context("failed to read fact row")?);
        }
        Ok(facts)
    }

    /// Get per-agent aggregated stats: sessions, total tokens, average duration.
    pub fn get_agent_stats(&self) -> Result<Vec<(String, i64, i64, f64)>> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT agent,
                    COUNT(*) AS sessions,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(AVG(duration_minutes), 0) AS avg_duration
             FROM sessions
             GROUP BY agent
             ORDER BY sessions DESC",
            )
            .context("failed to prepare agent stats")?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f64>(3).unwrap_or(0.0),
                ))
            })
            .context("failed to query agent stats")?;

        let mut stats = Vec::new();
        for row in rows {
            stats.push(row.context("failed to read agent stats row")?);
        }
        Ok(stats)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(id: &str) -> Session {
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
            files_changed: vec!["src/main.rs".into()],
            summary: Some("refactored module".into()),
        }
    }

    fn make_memory(id: &str, content: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            content: content.into(),
            memory_type: Some("insight".into()),
            source_session_id: None,
            confidence: 0.9,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        }
    }

    #[test]
    fn test_open_in_memory_and_schema() {
        let store = DuckStore::open_in_memory().expect("open in-memory");
        store.init_schema().expect("re-init schema");
    }

    #[test]
    fn test_insert_and_get_sessions() {
        let store = DuckStore::open_in_memory().unwrap();
        store.insert_session(&make_session("s-1")).unwrap();
        store.insert_session(&make_session("s-2")).unwrap();

        let recent = store.get_recent_sessions(10).unwrap();
        assert_eq!(recent.len(), 2);
        assert!(recent.iter().any(|s| s.id == "s-1"));
        assert!(recent.iter().any(|s| s.id == "s-2"));
    }

    #[test]
    fn test_insert_decision() {
        let store = DuckStore::open_in_memory().unwrap();
        let d = Decision {
            id: "d-1".into(),
            session_id: Some("s-1".into()),
            project_id: Some("proj-1".into()),
            decision_type: Some("architecture".into()),
            what: "use DuckDB for structured store".into(),
            why: Some("embedded, fast analytics".into()),
            alternatives: vec!["SQLite".into(), "Postgres".into()],
            outcome: None,
            created_at: None,
            valid_until: None,
        };
        store.insert_decision(&d).unwrap();
    }

    #[test]
    fn test_insert_and_search_memories() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_memory(&make_memory(
                "m-1",
                "The DuckDB store handles structured data",
            ))
            .unwrap();
        store
            .insert_memory(&make_memory("m-2", "LanceDB handles vector embeddings"))
            .unwrap();

        let results = store.search_memories("duckdb").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m-1");

        let all = store.search_memories("handles").unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_get_recent_sessions_limit() {
        let store = DuckStore::open_in_memory().unwrap();
        for i in 0..5 {
            store
                .insert_session(&make_session(&format!("s-{i}")))
                .unwrap();
        }
        let limited = store.get_recent_sessions(3).unwrap();
        assert_eq!(limited.len(), 3);
    }

    // -------------------------------------------------------------------
    // Graph CRUD tests (standard SQL, no PGQ extension required)
    // -------------------------------------------------------------------

    #[test]
    fn test_graph_insert_and_get_node() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("n1", "CodeEntity", "authenticate", r#"{"lang":"rust"}"#)
            .unwrap();

        let node = store
            .get_graph_node("n1")
            .unwrap()
            .expect("node should exist");
        assert_eq!(node.id, "n1");
        assert_eq!(node.kind, "CodeEntity");
        assert_eq!(node.name, "authenticate");
        assert_eq!(node.properties, r#"{"lang":"rust"}"#);
    }

    #[test]
    fn test_graph_get_missing_node() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(store.get_graph_node("nonexistent").unwrap().is_none());
    }

    #[test]
    fn test_graph_upsert_node() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("n1", "Concept", "old_name", "{}")
            .unwrap();
        store
            .insert_graph_node("n1", "Concept", "new_name", "{}")
            .unwrap();

        let node = store.get_graph_node("n1").unwrap().unwrap();
        assert_eq!(node.name, "new_name");
        assert_eq!(store.count_graph_nodes().unwrap(), 1);
    }

    #[test]
    fn test_graph_insert_edge_and_neighbors() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "CodeEntity", "foo", "{}")
            .unwrap();
        store
            .insert_graph_node("b", "CodeEntity", "bar", "{}")
            .unwrap();
        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap();

        // Outgoing from a
        let neighbors = store.query_graph_neighbors("a", None).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].node.id, "b");
        assert_eq!(neighbors[0].edge_kind, "CALLS");
        assert_eq!(neighbors[0].direction, "outgoing");

        // Incoming to b (query from b's perspective)
        let neighbors_b = store.query_graph_neighbors("b", None).unwrap();
        assert_eq!(neighbors_b.len(), 1);
        assert_eq!(neighbors_b[0].node.id, "a");
        assert_eq!(neighbors_b[0].direction, "incoming");
    }

    #[test]
    fn test_graph_neighbor_edge_kind_filter() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "CodeEntity", "foo", "{}")
            .unwrap();
        store
            .insert_graph_node("b", "CodeEntity", "bar", "{}")
            .unwrap();
        store.insert_graph_node("c", "Module", "baz", "{}").unwrap();

        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap();
        store.insert_graph_edge("a", "c", "IMPORTS", "{}").unwrap();

        let calls_only = store.query_graph_neighbors("a", Some("CALLS")).unwrap();
        assert_eq!(calls_only.len(), 1);
        assert_eq!(calls_only[0].node.id, "b");

        let imports_only = store.query_graph_neighbors("a", Some("IMPORTS")).unwrap();
        assert_eq!(imports_only.len(), 1);
        assert_eq!(imports_only[0].node.id, "c");
    }

    #[test]
    fn test_graph_duplicate_edge_ignored() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "CodeEntity", "foo", "{}")
            .unwrap();
        store
            .insert_graph_node("b", "CodeEntity", "bar", "{}")
            .unwrap();

        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap();
        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap(); // duplicate

        assert_eq!(store.count_graph_edges().unwrap(), 1);
    }

    #[test]
    fn test_graph_delete_node_cascades_edges() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "CodeEntity", "foo", "{}")
            .unwrap();
        store
            .insert_graph_node("b", "CodeEntity", "bar", "{}")
            .unwrap();
        store
            .insert_graph_node("c", "CodeEntity", "baz", "{}")
            .unwrap();

        store.insert_graph_edge("a", "b", "CALLS", "{}").unwrap();
        store.insert_graph_edge("c", "a", "IMPORTS", "{}").unwrap();

        assert_eq!(store.count_graph_edges().unwrap(), 2);

        // Delete node a — both incident edges should be removed
        assert!(store.delete_graph_node("a").unwrap());
        assert!(store.get_graph_node("a").unwrap().is_none());
        assert_eq!(store.count_graph_edges().unwrap(), 0);

        // Deleting again returns false
        assert!(!store.delete_graph_node("a").unwrap());
    }

    #[test]
    fn test_graph_counts() {
        let store = DuckStore::open_in_memory().unwrap();
        assert_eq!(store.count_graph_nodes().unwrap(), 0);
        assert_eq!(store.count_graph_edges().unwrap(), 0);

        store
            .insert_graph_node("a", "Concept", "auth", "{}")
            .unwrap();
        store.insert_graph_node("b", "Concept", "db", "{}").unwrap();
        store
            .insert_graph_edge("a", "b", "RELATES_TO", "{}")
            .unwrap();

        assert_eq!(store.count_graph_nodes().unwrap(), 2);
        assert_eq!(store.count_graph_edges().unwrap(), 1);
    }

    #[test]
    fn test_graph_clear() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_graph_node("a", "Concept", "auth", "{}")
            .unwrap();
        store.insert_graph_node("b", "Concept", "db", "{}").unwrap();
        store
            .insert_graph_edge("a", "b", "RELATES_TO", "{}")
            .unwrap();

        store.clear_graph().unwrap();

        assert_eq!(store.count_graph_nodes().unwrap(), 0);
        assert_eq!(store.count_graph_edges().unwrap(), 0);
    }

    // -------------------------------------------------------------------
    // Facts (temporal knowledge graph) tests
    // -------------------------------------------------------------------

    fn make_fact(id: &str, subject: &str, predicate: &str, object: &str) -> Fact {
        Fact {
            id: id.to_string(),
            project_id: Some("proj-1".into()),
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence: 0.9,
            source_session_id: Some("s-1".into()),
            source_agent: Some("claude".into()),
            valid_at: Some(Utc::now().naive_utc()),
            invalid_at: None,
            superseded_by: None,
            created_at: Some(Utc::now().naive_utc()),
        }
    }

    #[test]
    fn test_insert_and_get_facts() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();
        store
            .insert_fact(&make_fact("f-2", "db", "uses", "DuckDB"))
            .unwrap();

        let facts = store.get_active_facts(None, 100).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(store.count_facts().unwrap(), 2);
        assert_eq!(store.count_active_facts().unwrap(), 2);
    }

    #[test]
    fn test_invalidate_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();

        assert!(store.invalidate_fact("f-1", Some("f-2")).unwrap());
        assert_eq!(store.count_active_facts().unwrap(), 0);
        assert_eq!(store.count_facts().unwrap(), 1); // still exists, just invalidated

        // Can't invalidate twice
        assert!(!store.invalidate_fact("f-1", None).unwrap());
    }

    #[test]
    fn test_upsert_fact_supersedes_contradicting() {
        let store = DuckStore::open_in_memory().unwrap();
        // Insert original fact: auth uses JWT
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();

        // Upsert contradicting fact: auth uses OAuth2
        store
            .upsert_fact(&make_fact("f-2", "auth", "uses", "OAuth2"))
            .unwrap();

        // Old fact should be invalidated, new fact should be active
        let active = store.get_active_facts(None, 100).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].object, "OAuth2");

        // Total facts should be 2 (history preserved)
        assert_eq!(store.count_facts().unwrap(), 2);
    }

    #[test]
    fn test_upsert_fact_skips_duplicate() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();

        // Upsert same fact (same subject+predicate+object)
        store
            .upsert_fact(&make_fact("f-2", "auth", "uses", "JWT"))
            .unwrap();

        // Should still be 1 fact (duplicate skipped)
        assert_eq!(store.count_facts().unwrap(), 1);
    }

    #[test]
    fn test_search_facts() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth module", "uses", "JWT tokens"))
            .unwrap();
        store
            .insert_fact(&make_fact("f-2", "db layer", "uses", "DuckDB"))
            .unwrap();

        let results = store.search_facts("auth").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "auth module");

        let results = store.search_facts("DuckDB").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_update_and_delete_memory() {
        let store = DuckStore::open_in_memory().unwrap();
        let mem = Memory {
            id: "m-upd".into(),
            project_id: None,
            content: "original content".into(),
            memory_type: Some("note".into()),
            source_session_id: None,
            confidence: 0.8,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).unwrap();

        // Update content
        assert!(
            store
                .update_memory("m-upd", Some("revised content"), None)
                .unwrap()
        );
        let fetched = store.get_memory("m-upd").unwrap().unwrap();
        assert_eq!(fetched.content, "revised content");
        assert_eq!(fetched.confidence, 0.8); // unchanged

        // Update confidence
        assert!(store.update_memory("m-upd", None, Some(0.95)).unwrap());
        let fetched = store.get_memory("m-upd").unwrap().unwrap();
        assert_eq!(fetched.confidence, 0.95);

        // Update nonexistent
        assert!(!store.update_memory("nope", Some("x"), None).unwrap());

        // Delete
        assert!(store.delete_memory("m-upd").unwrap());
        assert!(store.get_memory("m-upd").unwrap().is_none());
        assert!(!store.delete_memory("m-upd").unwrap()); // already gone
    }

    #[test]
    fn test_get_and_delete_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-del", "auth", "uses", "JWT"))
            .unwrap();

        let fact = store.get_fact("f-del").unwrap().unwrap();
        assert_eq!(fact.subject, "auth");

        assert!(store.delete_fact("f-del").unwrap());
        assert!(store.get_fact("f-del").unwrap().is_none());
        assert!(!store.delete_fact("f-del").unwrap());
    }

    #[test]
    fn test_fact_history() {
        let store = DuckStore::open_in_memory().unwrap();
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();
        store.invalidate_fact("f-1", Some("f-2")).unwrap();
        store
            .insert_fact(&make_fact("f-2", "auth", "uses", "OAuth2"))
            .unwrap();

        let history = store.get_fact_history("auth").unwrap();
        assert_eq!(history.len(), 2);
        // First should be the older one (sorted by valid_at ASC)
        assert_eq!(history[0].object, "JWT");
        assert!(history[0].invalid_at.is_some());
        assert_eq!(history[1].object, "OAuth2");
        assert!(history[1].invalid_at.is_none());
    }

    // -----------------------------------------------------------------------
    // Edge case tests for self-editing tools
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_memory_both_fields() {
        let store = DuckStore::open_in_memory().unwrap();
        let mem = Memory {
            id: "m-both".into(),
            project_id: None,
            content: "original".into(),
            memory_type: Some("note".into()),
            source_session_id: None,
            confidence: 0.5,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).unwrap();

        // Update both content and confidence simultaneously
        assert!(
            store
                .update_memory("m-both", Some("new content"), Some(0.99))
                .unwrap()
        );
        let fetched = store.get_memory("m-both").unwrap().unwrap();
        assert_eq!(fetched.content, "new content");
        assert_eq!(fetched.confidence, 0.99);
    }

    #[test]
    fn test_update_memory_no_changes() {
        let store = DuckStore::open_in_memory().unwrap();
        let mem = Memory {
            id: "m-noop".into(),
            project_id: None,
            content: "unchanged".into(),
            memory_type: None,
            source_session_id: None,
            confidence: 0.7,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).unwrap();

        // Pass None for both — no fields to update
        let result = store.update_memory("m-noop", None, None);
        // Should either return false or succeed gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn test_delete_nonexistent_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(!store.delete_fact("nonexistent-id").unwrap());
    }

    #[test]
    fn test_delete_nonexistent_memory() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(!store.delete_memory("nonexistent-id").unwrap());
    }

    #[test]
    fn test_get_nonexistent_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(store.get_fact("nonexistent-id").unwrap().is_none());
    }

    #[test]
    fn test_get_nonexistent_memory() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(store.get_memory("nonexistent-id").unwrap().is_none());
    }

    #[test]
    fn test_invalidate_nonexistent_fact() {
        let store = DuckStore::open_in_memory().unwrap();
        assert!(!store.invalidate_fact("nonexistent-id", None).unwrap());
    }

    #[test]
    fn test_fact_history_empty_subject() {
        let store = DuckStore::open_in_memory().unwrap();
        let history = store.get_fact_history("no-such-subject").unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn test_search_facts_no_results() {
        let store = DuckStore::open_in_memory().unwrap();
        let results = store.search_facts("zzzznonexistent").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_upsert_fact_different_predicate_not_superseded() {
        let store = DuckStore::open_in_memory().unwrap();
        // Insert: auth "uses" JWT
        store
            .insert_fact(&make_fact("f-1", "auth", "uses", "JWT"))
            .unwrap();

        // Upsert with different predicate: auth "prefers" OAuth2
        store
            .upsert_fact(&make_fact("f-2", "auth", "prefers", "OAuth2"))
            .unwrap();

        // Both should be active (different predicates don't conflict)
        let active = store.get_active_facts(None, 100).unwrap();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_update_memory_confidence_bounds() {
        let store = DuckStore::open_in_memory().unwrap();
        let mem = Memory {
            id: "m-bounds".into(),
            project_id: None,
            content: "test".into(),
            memory_type: None,
            source_session_id: None,
            confidence: 0.5,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&mem).unwrap();

        // Set confidence to 0.0 and 1.0 (boundary values)
        assert!(store.update_memory("m-bounds", None, Some(0.0)).unwrap());
        let fetched = store.get_memory("m-bounds").unwrap().unwrap();
        assert_eq!(fetched.confidence, 0.0);

        assert!(store.update_memory("m-bounds", None, Some(1.0)).unwrap());
        let fetched = store.get_memory("m-bounds").unwrap().unwrap();
        assert_eq!(fetched.confidence, 1.0);
    }

    #[test]
    fn test_multiple_supersessions_chain() {
        let store = DuckStore::open_in_memory().unwrap();

        // Create a chain: f1 -> f2 -> f3
        store
            .insert_fact(&make_fact("f-1", "db", "uses", "SQLite"))
            .unwrap();
        store
            .upsert_fact(&make_fact("f-2", "db", "uses", "Postgres"))
            .unwrap();
        store
            .upsert_fact(&make_fact("f-3", "db", "uses", "DuckDB"))
            .unwrap();

        // Only the latest should be active
        let active = store.get_active_facts(None, 100).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].object, "DuckDB");

        // History should have all 3
        let history = store.get_fact_history("db").unwrap();
        assert_eq!(history.len(), 3);
    }
}
