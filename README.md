# Remembrant

**Shared Persistent Memory for AI Coding Agents**

Remembrant captures, indexes, and connects everything AI coding agents produce — sessions, decisions, tool calls, code entities — across Claude Code, Codex CLI, and Gemini CLI. It stores this in a triple-database architecture (DuckDB + LanceDB + property graph) and exposes it through a powerful CLI (`rem`), Semantic XPath queries, and a web dashboard.

No agent works in isolation anymore. Every session builds on everything that came before.

## Key Features

- **Multi-Agent Ingestion** — Native parsers for Claude Code (JSONL), Codex CLI (SQLite), and Gemini CLI (JSON)
- **Triple-Database Architecture** — DuckDB (structured + graph via DuckPGQ), LanceDB (vector search), in-memory graph (fallback)
- **Semantic XPath** — Tree-structured memory queries based on [arXiv:2603.01160](https://arxiv.org/abs/2603.01160), 176.7% better recall than flat RAG
- **DuckPGQ Graph Queries** — SQL/PGQ property graph: PageRank, shortest path, pattern matching — all inside DuckDB
- **Code Analysis** — AST parsing for 26 languages via Infiniloom integration (feature-gated)
- **Repository Embedding** — Embed entire codebases with BLAKE3 content-addressable chunks
- **Security Scanning** — Secret detection and redaction before embedding (via Infiniloom)
- **LLM Distillation** — Extract insights, patterns, and decisions from raw sessions
- **File Watching** — Real-time monitoring of agent artifact directories
- **Web Dashboard** — Built-in web UI for visual exploration
- **Local-First** — All data stays local; uses LM Studio for embeddings

## Quick Start

```bash
# Build and install the CLI
cargo install --path cli

# Initialize (creates config, scans for agents)
rem init

# Ingest sessions from all detected agents
rem ingest

# Search across all sessions
rem search "authentication refactor"

# Semantic XPath query
rem xpath '//Session[node~"auth"]/Decision'

# Embed a repository for code search
rem embed /path/to/project

# View stats
rem stats
```

### With Code Analysis (26 languages)

```bash
# Build with Infiniloom integration
cargo install --path cli --features code-analysis

# Analyze a repository (AST parsing, symbol extraction, dependency graph)
rem analyze /path/to/project
```

## Architecture

```
                    +-----------+
                    |  rem CLI  |  22 commands
                    +-----+-----+
                          |
                    +-----+-----+
                    |  Engine   |  remembrant-engine
                    +-----+-----+
                          |
          +---------------+---------------+
          |               |               |
    +-----+-----+  +-----+-----+  +------+------+
    |  DuckDB   |  |  LanceDB  |  | Graph Store |
    |  + DuckPGQ|  |  (Vector)  |  | (In-Memory) |
    +-----------+  +-----------+  +-------------+
```

### DuckDB (Structured Data + Property Graph)

Stores all structured data and provides SQL/PGQ graph queries:

| Table | Purpose |
|-------|---------|
| `sessions` | Session metadata (agent, project, timestamps, summary) |
| `decisions` | Decisions with rationale (what, why, alternatives) |
| `memories` | Memory notes (content, confidence, access_count) |
| `tool_calls` | Tool call history (command, success/failure) |
| `file_stats` | File statistics (LOC, complexity, change frequency) |
| `code_symbols` | AST-parsed symbols (functions, classes, structs) |
| `code_dependencies` | Import/call dependencies between files |
| `graph_nodes` | Property graph nodes (via DuckPGQ) |
| `graph_edges` | Property graph edges (via DuckPGQ) |

### LanceDB (Vector Embeddings)

| Table | Purpose |
|-------|---------|
| `embeddings` | Session/memory embeddings for semantic search |
| `symbol_embeddings` | Code symbol embeddings with PageRank scores |

### DuckPGQ (Property Graph)

DuckPGQ extends DuckDB with SQL/PGQ for graph operations — no separate graph database needed:

```sql
-- Shortest path between two code entities
SELECT * FROM GRAPH_TABLE(memory_graph
  MATCH (a:Node)-[p:Edge]->{1,5}(b:Node)
  WHERE a.id = 'fn-auth' AND b.id = 'fn-jwt'
  COLUMNS (path_length(p))
);

-- PageRank across the knowledge graph
SELECT node_id, pagerank FROM pgq_pagerank('memory_graph', 'Node', 'Edge');

-- Pattern matching
SELECT * FROM GRAPH_TABLE(memory_graph
  MATCH (a:Node)-[:CALLS]->(b:Node)-[:IMPORTS]->(c:Node)
  COLUMNS (a.name, b.name, c.name)
);
```

## Semantic XPath

Based on the paper ["Semantic XPath: Tree-Structured Memory Access for LLM Agents"](https://arxiv.org/abs/2603.01160), Remembrant implements a weighted-set evaluation algorithm over a hierarchical memory tree:

```
Root
├── Project "remembrant"
│   ├── Session "2026-03-20T14:30:00"
│   │   ├── Decision "use DuckPGQ for graph"
│   │   ├── Memory "DuckPGQ supports PageRank"
│   │   └── ToolCall "cargo test"
│   └── Session "2026-03-21T09:00:00"
│       └── CodeEntity "graph_builder.rs"
│           ├── Symbol "GraphBackend (trait)"
│           └── Symbol "GraphBuilder (struct)"
└── Project "infiniloom"
    └── ...
```

### Query Examples

```bash
# All decisions about authentication
rem xpath '//Decision[node~"auth"]'

# Recent sessions with their tool calls
rem xpath '/Root/Project/Session[position()>last()-5]/ToolCall'

# Code entities that relate to "graph"
rem xpath '//CodeEntity[node~"graph"]/Symbol'

# Decisions in a specific project
rem xpath '/Root/Project[@name="remembrant"]/Session/Decision'

# Combined semantic + structural query
rem xpath '//Session[node~"refactor"]/Decision[node~"performance"]'
```

The `~` operator triggers semantic similarity scoring (cosine similarity via LM Studio embeddings), while `@attr="value"` does exact attribute matching.

## Supported Agents

| Agent | Artifact Location | Format |
|-------|------------------|--------|
| **Claude Code** | `~/.claude/projects/*/` | JSONL transcripts + MEMORY.md |
| **Codex CLI** | `~/.codex/sessions/` | SQLite database |
| **Gemini CLI** | `~/.gemini/tmp/*/chats/` | JSON session files |

## CLI Reference

| Command | Description |
|---------|-------------|
| `rem init` | Initialize config, scan for agents |
| `rem watch` | Start file watcher daemon |
| `rem stop` | Stop the watcher daemon |
| `rem ingest` | Ingest sessions from all agents |
| `rem search <query>` | Semantic search (with `--project`, `--agent`, `--since` filters) |
| `rem find <text>` | Exact text search |
| `rem recent` | Show recent sessions |
| `rem brief` | Daily context briefing |
| `rem patterns [topic]` | Find cross-project patterns |
| `rem decisions` | View decision journal |
| `rem related <path>` | Find related content for a file |
| `rem graph <path>` | Show dependency graph |
| `rem timeline <topic>` | Chronological topic view |
| `rem note <text>` | Add a manual note |
| `rem forget --session <id>` | Remove a session |
| `rem export` | Generate agent memory files |
| `rem embed <path>` | Embed a repository for code search |
| `rem xpath <query>` | Semantic XPath query |
| `rem analyze <path>` | AST code analysis (requires `code-analysis` feature) |
| `rem status` | Show daemon and database status |
| `rem stats` | Show analytics and statistics |
| `rem gc` | Garbage collect old/orphaned data |

## Code Analysis (Feature-Gated)

When built with `--features code-analysis`, Remembrant integrates with [Infiniloom](https://github.com/Topos-Labs/infiniloom) for deep code understanding:

- **26-language AST parsing** via tree-sitter (Python, JS, TS, Rust, Go, Java, C, C++, and 18 more)
- **Symbol extraction** — functions, classes, structs, traits, interfaces
- **Dependency graph** — imports, calls, inheritance relationships
- **PageRank ranking** — identify the most important symbols in a codebase
- **BLAKE3 content hashing** — content-addressable chunk deduplication
- **Secret scanning** — detect and redact secrets before embedding

```bash
# Analyze a Rust project
rem analyze /path/to/rust-project --project my-project

# The symbols are stored in DuckDB and LanceDB for querying
rem search "GraphBuilder" --type symbol
rem xpath '//CodeEntity/Symbol[@kind="function"]'
```

## Configuration

Config lives at `~/.config/remembrant/config.toml`:

```toml
[storage]
duckdb_path = "~/.remembrant/data.db"
lancedb_path = "~/.remembrant/lance"

[agents.claude_code]
enabled = true
watch_path = "~/.claude/projects"

[agents.codex]
enabled = true
db_path = "~/.codex/sessions"

[agents.gemini]
enabled = true
watch_path = "~/.gemini/tmp"

[embedding]
provider = "lmstudio"
model = "nomic-embed-text"
endpoint = "http://localhost:1234/v1"
dimensions = 768
```

## Project Structure

```
remembrant/
├── engine/                    # Core library (remembrant-engine)
│   ├── src/
│   │   ├── store/
│   │   │   ├── duckdb.rs      # DuckStore + DuckPGQ graph queries
│   │   │   ├── lance.rs       # LanceStore (vector + symbol embeddings)
│   │   │   ├── graph.rs       # In-memory GraphStore + GraphStoreBackend trait
│   │   │   └── mod.rs
│   │   ├── ingest/
│   │   │   ├── claude.rs      # Claude Code parser (JSONL)
│   │   │   ├── codex.rs       # Codex CLI parser (SQLite)
│   │   │   └── gemini.rs      # Gemini CLI parser (JSON)
│   │   ├── semantic_tree.rs   # Tree-structured memory model (TreeBuilder, TreeNode)
│   │   ├── xpath_query.rs     # Semantic XPath parser + evaluator
│   │   ├── semantic_scorer.rs # Embedding-based semantic similarity scoring
│   │   ├── graph_builder.rs   # Generic GraphBuilder<B: GraphBackend>
│   │   ├── code_analysis.rs   # Infiniloom bridge (feature-gated)
│   │   ├── repo_embed.rs      # Repository embedder (AST chunking, secret scan)
│   │   ├── embed_pipeline.rs  # Embedding batch pipeline
│   │   ├── embedding.rs       # EmbedProvider trait (LmStudio, Mock)
│   │   ├── distill.rs         # LLM distillation
│   │   ├── pipeline.rs        # Ingestion pipeline orchestrator
│   │   ├── watcher.rs         # File system watcher
│   │   ├── detect.rs          # Agent detection utilities
│   │   └── config.rs          # AppConfig
│   └── tests/                 # Integration tests
├── cli/                       # CLI binary (rem)
│   └── src/main.rs            # 22 subcommands
├── .github/workflows/         # CI/CD
├── AGENTS.md                  # Guidelines for AI coding agents
└── Cargo.toml                 # Workspace config (edition 2024)
```

## Development

```bash
# Build the workspace
cargo build

# Run all tests (167 tests)
cargo test

# Build with code-analysis feature
cargo build --features code-analysis

# Run specific crate tests
cargo test -p remembrant-engine
cargo test -p remembrant

# Format and lint
cargo fmt --all
cargo clippy --workspace
```

### Key Design Decisions

- **Edition 2024 Rust** — latest language features
- **Generic GraphBuilder** — `GraphBuilder<B: GraphBackend>` works with both in-memory and DuckDB backends
- **DuckPGQ over separate graph DB** — zero data sync, same tables, built-in PageRank
- **Feature-gated Infiniloom** — optional `code-analysis` feature avoids heavy tree-sitter deps when not needed
- **Content-addressable chunks** — BLAKE3 hashing for deduplication
- **Secrets never embedded** — security scanning runs before chunking/embedding

## How It Works

1. **Detect** — Scan for installed agents (Claude Code, Codex, Gemini)
2. **Ingest** — Agent-specific parsers extract sessions, tool calls, decisions, memories
3. **Store** — Structured data goes to DuckDB, graph relationships built via DuckPGQ
4. **Embed** — LM Studio generates embeddings, stored in LanceDB
5. **Index** — Build hierarchical memory tree (Project > Session > Decision/Memory/ToolCall/CodeEntity > Symbol)
6. **Query** — CLI provides semantic search, XPath queries, graph traversal, and analytics
7. **Distill** — LLM extracts high-level insights and cross-project patterns

## License

MIT
