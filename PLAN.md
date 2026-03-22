# Remembrant — Project Plan

> Shared persistent memory for coding agents. Layers of knowledge, all recoverable.

## Vision

Remembrant is a local-first supersystem that captures, indexes, and connects everything coding agents produce — conversations, code, decisions, patterns — across Claude Code, Codex CLI, and Gemini CLI. It stores this in a triple-database architecture (DuckDB + LanceDB + LadybugDB) and makes it queryable via CLI and web UI.

No agent works in isolation anymore. Every session builds on everything that came before.

---

## Why This Matters

### The Problem

Coding agents have amnesia. Each session starts from scratch. Knowledge is scattered across:
- `~/.claude/projects/*/` — Claude Code transcripts
- `~/.codex/sessions/` — Codex session data
- `~/.gemini/tmp/*/chats/` — Gemini conversation logs
- `CLAUDE.md`, `AGENTS.md`, `GEMINI.md` — manual memory files
- Git history — decisions buried in commit messages

No tool today combines structured queries + semantic search + graph relationships across all three agents.

### The Gap

| Capability | Mem0 | Letta | Zep/Graphiti | Cortex | **Remembrant** |
|------------|------|-------|-------------|--------|---------------|
| DuckDB (structured) | - | - | - | - | **Yes** |
| LanceDB (vector) | - | - | - | - | **Yes** |
| Graph DB | - | - | Neo4j | Basic | **LadybugDB** |
| Claude Code support | - | - | - | MCP | **Native** |
| Codex CLI support | - | - | - | - | **Native** |
| Gemini CLI support | - | - | - | - | **Native** |
| Auto-ingestion | - | - | Partial | Watch | **Full** |
| Cross-project memory | - | - | - | - | **Yes** |
| Temporal tracking | - | - | Yes | - | **Yes** |
| Local-first | - | - | - | Yes | **Yes** |

---

## Architecture

### Triple-Database Design

```
                    ┌─────────────────────┐
                    │   INGESTION LAYER   │
                    │                     │
                    │  File Watchers      │
                    │  Git Hooks          │
                    │  Agent Hooks        │
                    │  Cron Scans         │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   PROCESSING LAYER  │
                    │                     │
                    │  Parsers (per agent)│
                    │  Infiniloom (code)  │
                    │  Distiller (LLM)    │
                    │  Entity Extractor   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼───────┐ ┌─────▼──────┐ ┌───────▼────────┐
     │    DuckDB      │ │  LanceDB   │ │   LadybugDB    │
     │  (Structured)  │ │  (Vector)  │ │    (Graph)     │
     │                │ │            │ │                │
     │ Sessions       │ │ Embeddings │ │ Code entities  │
     │ Decisions      │ │ Semantic   │ │ Concept links  │
     │ File stats     │ │ search     │ │ Memory links   │
     │ Tool calls     │ │ Similarity │ │ Dependencies   │
     │ Metrics        │ │ Clustering │ │ Temporal edges │
     └────────────────┘ └────────────┘ └────────────────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    QUERY LAYER      │
                    │                     │
                    │  CLI (rem)          │
                    │  Web UI             │
                    └─────────────────────┘
```

### What Goes Where

**DuckDB** — The structured truth
- Session metadata (agent, project, timestamp, duration, token count)
- Decisions with rationale (what, why, alternatives considered)
- Tool call history (command, success/failure, error messages)
- File statistics (LOC, complexity, change frequency)
- Cross-project metrics and analytics

**LanceDB** — The semantic layer
- Code embeddings at multiple granularities (session, function, decision)
- Natural language embeddings of conversations and decisions
- Hybrid search: vector similarity + metadata filters
- Embedding model: `voyage-code-3` for code, general model for text

**LadybugDB** — The relationship web
- Code entity graphs: function→calls→function, file→imports→file
- Concept graphs: pattern→used_in→project, problem→solved_by→solution
- Memory graphs: memory_note→relates_to→memory_note, insight→derived_from→session
- Temporal edges: valid_at / invalid_at for tracking knowledge evolution
- Cross-project links: shared_pattern→appears_in→[project_a, project_b]

### Combined Query Examples

**"How did I implement auth before?"**
1. LanceDB: semantic search "authentication implementation" → top candidates
2. DuckDB: filter by successful sessions, sort by recency
3. LadybugDB: traverse pattern→used_in→project to find all instances

**"What depends on this module?"**
1. LadybugDB: Cypher query for transitive IMPORTS/CALLS
2. DuckDB: get file stats (complexity, test coverage) for each dependent
3. LanceDB: find semantically similar modules not directly linked

**"What decisions have I made about databases?"**
1. DuckDB: query decisions table WHERE topic LIKE '%database%'
2. LanceDB: semantic search for database-related conversations
3. LadybugDB: traverse decision→led_to→implementation chains

---

## Agent Artifact Sources

### Claude Code (`~/.claude/`)
| Path | Format | Content |
|------|--------|---------|
| `~/.claude/projects/*/` | JSONL | Full conversation transcripts |
| `~/.claude/projects/*/memory/MEMORY.md` | Markdown | Per-project auto-memory |
| `~/.claude/settings.json` | JSON | Global settings |
| `CLAUDE.md` (project root) | Markdown | Project instructions |

### Codex CLI (`~/.codex/`)
| Path | Format | Content |
|------|--------|---------|
| `~/.codex/sessions/<year>/` | SQLite | Session data with threads |
| `~/.codex/history.jsonl` | JSONL | Command/conversation history |
| `~/.codex/skills/` | Markdown | User-defined skills (SKILL.md) |
| `~/.codex/config.toml` | TOML | Configuration |
| `AGENTS.md` (project root) | Markdown | Project instructions |

### Gemini CLI (`~/.gemini/`)
| Path | Format | Content |
|------|--------|---------|
| `~/.gemini/tmp/<hash>/chats/session-*.json` | JSON | Full sessions with messages, tokens, tool calls |
| `~/.gemini/tmp/<hash>/logs.json` | JSON | Command logs |
| `~/.gemini/tmp/<hash>/shell_history` | Text | Shell history |
| `~/.gemini/projects.json` | JSON | Project→hash mapping |
| `GEMINI.md` (project root) | Markdown | Project instructions |

---

## Data Model

### DuckDB Schema

```sql
-- Core tables
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    visibility TEXT DEFAULT 'public',  -- public | private | sandbox
    tags TEXT[],
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    agent TEXT NOT NULL,               -- claude-code | codex | gemini
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    duration_minutes INTEGER,
    message_count INTEGER,
    tool_call_count INTEGER,
    total_tokens INTEGER,
    files_changed TEXT[],
    summary TEXT                        -- distilled summary
);

CREATE TABLE decisions (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    project_id TEXT REFERENCES projects(id),
    decision_type TEXT,                -- architecture | tooling | pattern | refactor
    what TEXT NOT NULL,
    why TEXT,
    alternatives TEXT[],
    outcome TEXT,                       -- success | failed | unknown
    created_at TIMESTAMP,
    valid_until TIMESTAMP              -- temporal: when superseded
);

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    project_id TEXT,                    -- NULL = global
    content TEXT NOT NULL,
    memory_type TEXT,                  -- pattern | solution | convention | note
    source_session_id TEXT,
    confidence FLOAT DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    valid_until TIMESTAMP              -- temporal tracking
);

CREATE TABLE tool_calls (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    tool_name TEXT,
    command TEXT,
    success BOOLEAN,
    error_message TEXT,
    duration_ms INTEGER,
    timestamp TIMESTAMP
);

CREATE TABLE file_stats (
    file_path TEXT,
    project_id TEXT REFERENCES projects(id),
    language TEXT,
    lines_of_code INTEGER,
    token_count INTEGER,
    complexity FLOAT,
    change_frequency INTEGER,          -- times modified across sessions
    last_modified TIMESTAMP,
    PRIMARY KEY (file_path, project_id)
);
```

### LanceDB Tables

```
code_embeddings:
  id: TEXT
  embedding: VECTOR[1024]          -- voyage-code-3
  content: TEXT                     -- actual code/text
  granularity: TEXT                 -- session | function | class | decision | exchange
  project_id: TEXT
  file_path: TEXT
  language: TEXT
  created_at: TIMESTAMP

memory_embeddings:
  id: TEXT
  embedding: VECTOR[1024]
  content: TEXT
  memory_type: TEXT
  project_id: TEXT
  created_at: TIMESTAMP
```

### LadybugDB Graph Schema

```cypher
-- Node types
(:CodeEntity {id, name, kind, file_path, project_id})
  -- kind: function | class | module | file | type

(:Concept {id, name, description})
  -- e.g., "authentication", "caching", "error handling"

(:Memory {id, content, type, project_id, created_at, valid_until})

(:Pattern {id, name, description, occurrence_count})

(:Problem {id, description, severity})

(:Solution {id, description, success_rate})

-- Edge types
(:CodeEntity)-[:CALLS]->(:CodeEntity)
(:CodeEntity)-[:IMPORTS]->(:CodeEntity)
(:CodeEntity)-[:INHERITS]->(:CodeEntity)
(:CodeEntity)-[:IMPLEMENTS]->(:Concept)

(:Pattern)-[:USED_IN]->(:Project)
(:Pattern)-[:SOLVES]->(:Problem)
(:Problem)-[:SOLVED_BY]->(:Solution)
(:Solution)-[:IMPLEMENTED_IN]->(:CodeEntity)

(:Memory)-[:RELATES_TO]->(:Memory)
(:Memory)-[:DERIVED_FROM]->(:Session)
(:Memory)-[:ABOUT]->(:CodeEntity)
(:Memory)-[:SUPERSEDES {valid_from, valid_until}]->(:Memory)

(:Concept)-[:RELATED_TO]->(:Concept)
(:Concept)-[:APPEARS_IN]->(:Project)
```

---

## Memory Tiers

| Tier | Content | Retention | Storage |
|------|---------|-----------|---------|
| **Raw** | Full conversation transcripts | 30 days | Filesystem (original files) |
| **Distilled** | LLM-summarized decisions, patterns, solutions | Forever | DuckDB + LanceDB |
| **Indexed** | Embeddings at multiple granularities | Forever | LanceDB |
| **Graph** | Entity relationships, concept links, temporal edges | Forever | LadybugDB |
| **Metrics** | Token usage, tool stats, session analytics | Forever | DuckDB |

### Distillation Strategy

**Automatic extraction from each session:**
1. Decisions (what was decided and why)
2. Problems solved (issue → solution pairs)
3. Patterns used (recurring approaches)
4. Code entities discussed (functions, files, classes)
5. Open questions (unresolved from session)

**Distillation levels** (user-configurable):
- `none` — store raw only, no processing
- `minimal` — keyword + entity extraction only (no LLM)
- `balanced` — LLM summarization for sessions >10 messages (default)
- `aggressive` — heavy summarization, discard raw after 7 days
- `full` — LLM + entity linking + pattern detection + graph building

---

## CLI Design

### Core Commands

```bash
# Setup
rem init                              # Auto-detect agents, create config, initial scan
rem watch                             # Start daemon (file watchers + hooks)
rem stop                              # Stop daemon

# Search & Recall
rem search "authentication flow"      # Semantic search across everything
rem search "redis" --project myapp    # Scoped to project
rem search "error handling" --since 7d # Time-scoped
rem find "class UserService" --exact  # Exact text match

# Daily Workflow
rem brief                             # Today's context briefing
rem brief --project infiniloom        # Project-specific briefing
rem recent                            # Recent sessions across all agents
rem recent --agent claude-code        # Agent-specific

# Knowledge
rem patterns                          # Cross-project pattern library
rem patterns "caching"                # Patterns matching topic
rem decisions                         # Decision journal
rem decisions --project infiniloom    # Project decisions

# Relationships
rem related src/auth.rs               # Everything related to this file
rem graph src/auth.rs                 # Dependency graph for file
rem timeline "JWT"                    # Chronological view of topic

# Memory Management
rem note "try approach X next"        # Quick manual note
rem forget --session <id>             # Remove a session
rem export --project X                # Generate CLAUDE.md / AGENTS.md / GEMINI.md

# Repository Embedding
rem embed /path/to/repo               # Full repo embedding via Infiniloom
rem embed --update                     # Re-embed changed files only

# Admin
rem status                            # Daemon status, DB sizes, session counts
rem stats                             # Analytics dashboard
rem gc                                # Garbage collect expired raw data
```

### Example Workflows

**Morning startup:**
```bash
$ rem brief
Remembrant Brief — 2026-03-21

Recent Work (last 24h):
  infiniloom: Added Puppet/YAML/Dockerfile tree-sitter support (Claude Code, 47min)
  cortex: Researched LanceDB integration (Gemini CLI, 23min)

Decisions:
  - infiniloom: Used extern C + tree-sitter-language crate to bypass version mismatch
  - cortex: Chose LanceDB over Qdrant (embedded, Rust SDK, multimodal)

Patterns Detected:
  Tree-sitter integration pattern used 3x across projects
  → infiniloom (2026-03-16), agent-typed (2026-02-14), symphony (2026-01-22)

Open Questions:
  - infiniloom: Should LSP integration be added to embed command?
```

**Mid-task search:**
```bash
$ rem search "how to handle tree-sitter version mismatch"
  1. [infiniloom] 2026-03-16 — Claude Code session
     Used extern C + tree-sitter-language crate. tree-sitter-dockerfile 0.2.0
     uses tree-sitter 0.20, bypassed via compatibility shim.
     Files: engine/src/parser/init.rs:27, engine/Cargo.toml

  2. [agent-typed] 2026-02-14 — Codex session
     Similar version conflict with tree-sitter-hcl. Pinned to specific commit.
     Files: parser/src/hcl.rs:15
```

**Export to agent memory files:**
```bash
$ rem export --project infiniloom --format claude
Updated: /Users/alex/Projects/ai/infiniloom/CLAUDE.md
  Added: 3 new patterns, 2 decisions, 1 convention
```

---

## Cross-Project Intelligence

### Project Visibility

```yaml
# ~/.remembrant/config.yaml
projects:
  infiniloom:
    visibility: public
    share_patterns: true
    tags: [rust, open-source, cli-tool]

  client-acme:
    visibility: private
    share_patterns: false

  experiments:
    visibility: sandbox
    share_patterns: false
```

- `public` — patterns searchable from any project
- `private` — isolated, never appears in cross-project queries
- `sandbox` — experimental, excluded from pattern detection

### Pattern Library

Automatically built from recurring solutions:

```
Pattern: JWT Authentication with Refresh Tokens
Used in: infiniloom (2026-03-01), cortex (2026-02-15), agent-typed (2026-01-10)
Success rate: 100% (3/3)
Key files: auth.rs, jwt.rs, middleware.rs
Decision: Chose JWT over session cookies for stateless API auth
```

### Decision Journal

Cross-project decision tracking:

```
2026-03-16 | infiniloom | Tree-sitter over LSP for parsing
  Why: Faster, no external deps, 26 languages

2026-03-10 | cortex | LanceDB over Pinecone for vectors
  Why: Open-source, embedded, Rust SDK

2026-02-14 | agent-typed | Rust over Python for CLI
  Why: Performance, single binary, error handling
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Rust | Performance, matches Infiniloom, single binary |
| CLI framework | `clap` | Standard Rust CLI, derive macros |
| File watching | `notify` crate | Cross-platform fsnotify |
| Structured DB | DuckDB (`duckdb` crate) | Embedded OLAP, mature Rust SDK |
| Vector DB | LanceDB (`lancedb` crate) | Embedded, Arrow-native, hybrid search |
| Graph DB | LadybugDB | Embedded columnar graph, Cypher queries |
| Code parsing | Infiniloom (library crate) | 26-language tree-sitter, chunking, embedding |
| Graph DB | LadybugDB (fallback: Kuzu) | Embedded columnar graph, Cypher queries |
| Embedding model | nomic-embed (local default) | Free, local-first. Optional: voyage-code-3 |
| LLM distillation | LM Studio (local default) | Free, private. Optional: Haiku, GPT-4o-mini |
| Daemon | launchd (macOS first) | Native macOS service management |
| Web UI | Axum + HTMX (or Leptos) | Lightweight, Rust-native |
| Config | YAML | Human-readable, familiar |
| Serialization | serde + Arrow | Fast, zero-copy where possible |

---

## Implementation Phases

### Phase 0: Skeleton (Week 1)
- [ ] Repo setup: Cargo workspace, CI, linting
- [ ] Config system: `~/.remembrant/config.yaml`
- [ ] CLI skeleton with `clap` (all commands stubbed)
- [ ] DuckDB schema creation on `rem init`
- [ ] LanceDB table creation on `rem init`

### Phase 1: Ingestion (Weeks 2-3)
- [ ] Claude Code parser: read `~/.claude/projects/*/` JSONL transcripts
- [ ] Codex parser: read `~/.codex/sessions/` SQLite + `history.jsonl`
- [ ] Gemini parser: read `~/.gemini/tmp/*/chats/session-*.json`
- [ ] File watcher daemon (`rem watch`)
- [ ] Session metadata → DuckDB
- [ ] Initial scan on `rem init` (index existing sessions)

### Phase 2: Search & Recall (Weeks 3-4)
- [ ] Embed sessions into LanceDB (via Infiniloom or direct embedding API)
- [ ] `rem search` — semantic search with metadata filters
- [ ] `rem find` — exact text match via DuckDB
- [ ] `rem recent` — list recent sessions
- [ ] `rem brief` — daily context briefing

### Phase 3: Distillation (Weeks 4-5)
- [ ] LLM-based session summarization (decisions, problems, patterns)
- [ ] Entity extraction (code entities, file paths, function names)
- [ ] Keyword extraction (TF-IDF fallback when no LLM configured)
- [ ] Multi-granularity embedding (session, exchange, decision, code)
- [ ] Distillation levels (none → full)

### Phase 4: Graph Layer (Weeks 5-6)
- [ ] LadybugDB integration
- [ ] Code entity extraction via Infiniloom tree-sitter
- [ ] Concept/pattern graph building
- [ ] Memory→memory relationship links
- [ ] Temporal edges (valid_at / valid_until)
- [ ] `rem related`, `rem graph`, `rem timeline` commands

### Phase 5: Cross-Project Intelligence (Weeks 6-7)
- [ ] Project visibility system (public/private/sandbox)
- [ ] Pattern detection across projects
- [ ] Decision journal aggregation
- [ ] `rem patterns`, `rem decisions` commands
- [ ] `rem export` — auto-generate CLAUDE.md / AGENTS.md / GEMINI.md

### Phase 6: Repository Embedding (Week 7-8)
- [ ] `rem embed /path/to/repo` via Infiniloom library
- [ ] Incremental re-embedding (changed files only)
- [ ] Code entity→graph integration
- [ ] Cross-reference: agent discussions ↔ actual code

### Phase 7: Web UI (Weeks 8-10)
- [ ] Axum server with HTMX (or Leptos SPA)
- [ ] Search interface with filters
- [ ] Knowledge graph visualization
- [ ] Session timeline browser
- [ ] Decision journal view
- [ ] Analytics dashboard (token usage, agent stats)

### Phase 8: Polish & Launch (Weeks 10-12)
- [ ] Homebrew formula
- [ ] cargo install support
- [ ] Documentation site
- [ ] Performance optimization (batch embedding, query caching)
- [ ] Garbage collection for expired raw data
- [ ] Integration tests with real agent artifacts

---

## Configuration

```yaml
# ~/.remembrant/config.yaml

# Agent directories (auto-detected, override here)
agents:
  claude-code:
    enabled: true
    path: ~/.claude
  codex:
    enabled: true
    path: ~/.codex
  gemini:
    enabled: true
    path: ~/.gemini

# Database paths
storage:
  base_dir: ~/.remembrant
  duckdb: ~/.remembrant/remembrant.duckdb
  lancedb: ~/.remembrant/lancedb/
  ladybugdb: ~/.remembrant/graph/

# Embedding
embedding:
  model: voyage-code-3
  api_key_env: VOYAGE_API_KEY       # read from env var
  batch_size: 100
  dimensions: 1024

# Distillation
distillation:
  level: balanced                    # none | minimal | balanced | aggressive | full
  llm_provider: anthropic            # anthropic | openai
  llm_model: claude-haiku-4-5-20251001
  api_key_env: ANTHROPIC_API_KEY

# Memory retention
retention:
  raw_transcripts_days: 30           # delete raw after N days
  distilled: forever
  embeddings: forever

# Cross-project
cross_project:
  enabled: true
  default_visibility: public         # public | private | sandbox

# Watch daemon
watch:
  debounce_ms: 5000                  # batch events within window
  auto_start: false                  # start daemon on login

# Infiniloom integration
infiniloom:
  binary: infiniloom                 # or path to binary
  # Or use as library crate (default when built together)
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| First useful result after install | < 5 minutes |
| Semantic search latency | < 200ms |
| Full project indexing | < 1 hour |
| Daily brief generation | < 2 seconds |
| Cross-project pattern detection | < 5 seconds |
| Storage overhead per session | < 50KB (distilled) |
| Zero-config agent detection | 3/3 agents auto-detected |

---

## Competitive Positioning

**Remembrant is NOT:**
- A chatbot memory framework (that's Mem0, Letta)
- A RAG pipeline (that's LangChain, LlamaIndex)
- An MCP server (that's Cortex)
- A cloud service (that's Zep)

**Remembrant IS:**
- A local-first persistent memory system for coding agents
- A triple-database architecture (structured + vector + graph)
- The first tool to unify Claude Code + Codex + Gemini artifacts
- Automatic ingestion — no manual memory.add() calls
- Cross-project pattern recognition
- Temporal knowledge tracking (what was true when)

---

## Design Decisions

1. **Graph DB** — LadybugDB first, Kuzu as fallback if SDK isn't ready. Both are embedded columnar graph DBs with Cypher support.
2. **Local-first embeddings** — Local models as default (nomic-embed, all-MiniLM). Paid APIs (voyage-code-3, OpenAI) as optional upgrade for higher quality.
3. **Local-first distillation** — LM Studio as default for LLM summarization. Cloud LLMs (Haiku, GPT-4o-mini) as optional for users who prefer speed/quality.
4. **Web UI** — Deferred. CLI is the priority. Web UI framework TBD later.
5. **Platform** — macOS first. Daemon via launchd. Linux (systemd) support follows.
