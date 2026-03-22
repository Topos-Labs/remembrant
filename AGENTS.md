# AGENTS.md

Guidelines for AI coding agents working on the Remembrant codebase.

## Project Overview

Remembrant is a Rust-based CLI tool (`rem`) that ingests coding agent artifacts from Claude Code, Codex CLI, and Gemini CLI into a triple-database (DuckDB + LanceDB + DuckPGQ property graph) for shared persistent memory across AI coding agents. It includes Semantic XPath queries, optional 26-language code analysis via Infiniloom, and a Tauri v2 desktop app.

## Architecture

### Workspace Structure

This is a Cargo workspace (edition 2024) with three crates:

- **engine** (`remembrant-engine`): Core library — ingestion, storage, search, graph, Semantic XPath
- **cli** (`remembrant`): Binary crate providing the `rem` CLI tool (22 commands)
- **desktop** (`src-tauri`): Tauri v2 desktop application

### Key Modules

**engine/src/**:
- `store/` — Database implementations
  - `duckdb.rs` — DuckStore: structured data + DuckPGQ graph (uses `Mutex<Connection>`)
  - `lance.rs` — LanceStore: vector embeddings + symbol embeddings (async)
  - `graph.rs` — GraphStoreBackend trait + in-memory GraphStore (fallback)
- `ingest/` — Agent-specific parsers
  - `claude.rs` — ClaudeIngester (JSONL transcripts)
  - `codex.rs` — CodexIngester (SQLite database)
  - `gemini.rs` — GeminiIngester (JSON sessions)
- `semantic_tree.rs` — TreeBuilder, TreeNode, TreeNodeType, TreeSchema (hierarchical memory model)
- `xpath_query.rs` — XPathQuery parser + evaluator (weighted-set algorithm from arXiv:2603.01160)
- `semantic_scorer.rs` — SemanticScorer (embedding cache) + keyword_scorer (fallback)
- `graph_builder.rs` — `GraphBuilder<B: GraphBackend>` generic over backend (in-memory or DuckDB)
- `code_analysis.rs` — Infiniloom bridge (feature-gated: `code-analysis`)
- `repo_embed.rs` — RepoEmbedder (AST chunking, BLAKE3 hashing, secret scanning)
- `embed_pipeline.rs` — EmbedPipeline for batching embeddings
- `embedding.rs` — EmbedProvider trait (LmStudioEmbedder, MockEmbedder)
- `distill.rs` — Distiller for LLM-based insight extraction
- `watcher.rs` — FileWatcher for monitoring agent directories
- `pipeline.rs` — IngestPipeline for orchestrating ingestion
- `config.rs` — AppConfig
- `detect.rs` — Agent detection utilities

**cli/src/**:
- `main.rs` — 22 subcommands: init, watch, stop, search, find, recent, brief, patterns, decisions, related, graph, timeline, note, forget, export, embed, ingest, xpath, analyze, status, stats, gc

### Feature Flags

- **default** — No optional features
- **code-analysis** — Enables Infiniloom integration (AST parsing, secret scanning, BLAKE3 hashing). Adds `infiniloom-engine` and `blake3` dependencies.

## Development Guidelines

### Building and Testing

```bash
# Build the entire workspace
cargo build

# Build with code-analysis feature
cargo build --features code-analysis

# Run all tests (167 tests)
cargo test

# Run specific crate tests
cargo test -p remembrant-engine
cargo test -p remembrant

# Run the CLI
cargo run --bin rem -- --help

# Format code
cargo fmt --all

# Check for issues
cargo clippy --workspace
```

### Code Style

- **Edition**: 2024 Rust
- **Error Handling**: Use `anyhow::Result` for application code, `thiserror` for library errors
- **Async**: Use Tokio runtime (`#[tokio::main]` or `#[tokio::test]`)
- **Logging**: Use `tracing` macros (`tracing::info!`, `tracing::error!`, etc.)
- **Serialization**: Use `serde` with `#[derive(Serialize, Deserialize)]`

### Important Patterns

#### GraphBuilder (Generic over Backend)

GraphBuilder is generic over the `GraphBackend` trait, supporting both in-memory and DuckDB backends:

```rust
pub trait GraphBackend {
    fn add_node(&self, id: &str, kind: &str, name: &str, properties: &str) -> Result<()>;
    fn add_edge(&self, from_id: &str, to_id: &str, kind: &str, properties: &str) -> Result<()>;
    fn get_node(&self, id: &str) -> Result<Option<(String, String, String, String)>>;
    fn delete_node(&self, id: &str) -> Result<bool>;
    fn query_neighbors(&self, id: &str, edge_kind: Option<&str>) -> Result<Vec<NeighborInfo>>;
    fn node_count(&self) -> Result<usize>;
    fn edge_count(&self) -> Result<usize>;
}

// In-memory (for tests and fallback)
pub type InMemoryGraphBuilder = GraphBuilder<GraphStore>;

// Persistent (DuckDB + DuckPGQ)
pub type DuckGraphBuilder = GraphBuilder<DuckStore>;
```

#### DuckStore (Synchronous + DuckPGQ)

DuckStore uses `Mutex<Connection>` for thread-safe access. DuckPGQ adds graph queries:

```rust
impl DuckStore {
    // Structured data
    pub fn insert_session(&self, session: &Session) -> Result<()>;

    // Graph CRUD (stored in graph_nodes/graph_edges tables)
    pub fn insert_graph_node(&self, id: &str, kind: &str, name: &str, props: &str) -> Result<()>;
    pub fn insert_graph_edge(&self, from: &str, to: &str, kind: &str, props: &str) -> Result<()>;

    // DuckPGQ queries
    pub fn load_duckpgq(&self) -> Result<()>;
    pub fn init_graph(&self) -> Result<()>;
    pub fn pgq_shortest_path(&self, from: &str, to: &str, max_depth: usize) -> Result<Vec<String>>;
    pub fn pgq_pagerank(&self) -> Result<Vec<(String, f64)>>;
    pub fn pgq_pattern_match(&self, pattern_sql: &str) -> Result<Vec<Vec<String>>>;

    // Connection accessor (for TreeBuilder)
    pub fn connection(&self) -> &Mutex<Connection>;
}
```

#### LanceStore (Async)

LanceStore is async throughout, with two table types:

```rust
impl LanceStore {
    pub async fn open(path: &Path) -> Result<Self>;

    // General embeddings
    pub async fn insert_embeddings(&self, chunks: Vec<EmbedChunk>) -> Result<()>;
    pub async fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>>;

    // Symbol embeddings (code analysis)
    pub async fn insert_symbol_embedding(&self, symbol: SymbolEmbedding) -> Result<()>;
    pub async fn search_symbols(&self, query: &[f32], limit: usize) -> Result<Vec<SymbolSearchResult>>;
}
```

#### EmbedProvider (Not dyn-compatible)

The `EmbedProvider` trait uses `impl Future`. Always use generics:

```rust
// CORRECT
pub async fn process<P: EmbedProvider>(provider: &P) -> Result<()> { }

// INCORRECT - will not compile
pub async fn process(provider: &dyn EmbedProvider) -> Result<()> { }
```

#### Semantic XPath

The XPath system has three layers:

1. **TreeBuilder** (`semantic_tree.rs`) — Builds hierarchical memory tree from DuckDB, lazy-loads children
2. **XPathQuery** (`xpath_query.rs`) — Recursive descent parser for XPath-like syntax
3. **SemanticScorer** (`semantic_scorer.rs`) — Embedding-based similarity for `~` operator

```rust
// Parse and evaluate a Semantic XPath query
let query = parse_xpath("//Session[node~\"auth\"]/Decision")?;
let results = evaluate_xpath(&query, &root, &scorer);
for weighted_node in results {
    println!("{}: {:.2}", weighted_node.node.name, weighted_node.weight);
}
```

### Testing

- Unit tests go in the same file (bottom of file)
- Integration tests go in `engine/tests/`
- Use `DuckStore::open_in_memory()` for tests (no cleanup needed)
- Use `MockEmbedder` for embedding tests
- Use `tempfile::tempdir()` for LanceDB paths in tests
- Feature-gated tests: `#[cfg(feature = "code-analysis")]`

### Database Schema

**DuckDB Tables**:
- `sessions` — Session metadata (id, project_id, agent, timestamps, summary)
- `decisions` — Decisions with rationale (id, session_id, what, why, alternatives)
- `memories` — Memory notes (id, project_id, content, confidence, access_count)
- `tool_calls` — Tool call history (id, session_id, tool_name, command, success)
- `file_stats` — File statistics (file_path, project_id, language, LOC, complexity)
- `code_symbols` — AST-parsed symbols (file_path, name, kind, signature, lines)
- `code_dependencies` — Import/call dependencies between files
- `code_analysis_runs` — Analysis run metadata
- `graph_nodes` — Property graph nodes (id, kind, name, properties JSON)
- `graph_edges` — Property graph edges (from_id, to_id, kind, properties JSON)

**LanceDB Tables**:
- `embeddings` — General vector embeddings (id, project_id, content, vector)
- `symbol_embeddings` — Code symbol embeddings (13 columns including PageRank)

### Common Pitfalls

1. **Don't modify test artifact directories**: Never modify `~/.claude`, `~/.codex`, or `~/.gemini` in tests
2. **Use in-memory databases**: Always use `DuckStore::open_in_memory()` for tests
3. **Handle async correctly**: LanceStore is async, DuckStore is sync — don't mix patterns
4. **EmbedProvider generics**: Never try to use `&dyn EmbedProvider`
5. **Tilde expansion**: Config paths use `~/` — expand with `dirs::home_dir()`
6. **Feature gates**: Code analysis imports must be behind `#[cfg(feature = "code-analysis")]`
7. **GraphBackend is all-string**: Node kind, properties are strings (serialized JSON for properties)

### Adding New Features

1. Add core logic to `engine/src/`
2. Add tests (unit in same file, integration in `engine/tests/`)
3. Add CLI subcommand in `cli/src/main.rs` if needed
4. Update AGENTS.md and README.md
5. Run `cargo test && cargo fmt --all && cargo clippy --workspace`

### Dependencies

Key dependencies (see workspace `Cargo.toml`):
- `duckdb` 1.x — Embedded SQL database + DuckPGQ extension
- `lancedb` 0.27 — Vector database
- `arrow-array`, `arrow-schema` 57 — Arrow format (must match lancedb)
- `tokio` — Async runtime
- `clap` 4 — CLI argument parsing
- `serde`, `serde_json`, `serde_yaml` — Serialization
- `tracing`, `tracing-subscriber` — Logging
- `notify`, `notify-debouncer-mini` — File watching
- `reqwest` 0.12 — HTTP client (LM Studio)
- `chrono`, `uuid` — Date/time and IDs

Optional (code-analysis feature):
- `infiniloom-engine` — AST parsing, secret scanning (26 languages)
- `blake3` — Content-addressable hashing
