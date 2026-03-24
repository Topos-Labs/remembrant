# Remembrant: Vision & Roadmap

## The Goal

Make Remembrant the universal memory layer for AI coding agents — 100x better than anything existing. Not just memory, but a **complete cognitive substrate**: decision tracking, code understanding, project evolution, cross-agent coordination, and analytics. Every agent session builds on everything that came before.

No competitor combines all four pillars:
1. **Cross-agent persistent memory** (Claude Code, Codex, Gemini, OpenHands)
2. **Deep per-project code RAG** (AST-aware, dependency-graph-driven)
3. **Decision/fact extraction with deduplication** (temporal knowledge graph)
4. **Analytics & observability** (prompt patterns, project evolution, agent effectiveness)

---

## Current State Assessment

### What Works Well
- Triple-store architecture (DuckDB + LanceDB + graph) is sound and unique
- Pluggable agent adapters for Claude Code/Codex/Gemini
- Semantic XPath with tree-structured memory is elegant
- BLAKE3 content-addressable chunks enable dedup
- MCP server with 10+ tools for Claude Code/Cursor integration
- Hybrid search with RRF merging across backends
- Fact temporal validity (valid_at/invalid_at/superseded_by)

### Critical Gaps (Ranked by Impact)

| Gap | Impact | Competitors That Solve It |
|-----|--------|--------------------------|
| No semantic deduplication of memories/facts | Repeated knowledge bloats context | Mem0 (ADD/UPDATE/DELETE), Graphiti |
| Naive code chunking (50-line splits) | Poor code retrieval quality | Greptile, Sweep, Aider |
| No NL summaries of code chunks | 12% retrieval quality loss | Greptile |
| Consolidation doesn't execute merges | Dead code — finds candidates but does nothing | Letta, LangMem |
| No BM25/FTS for identifier search | Misses exact matches critical for code | Sourcegraph, GrepRAG paper |
| Distillation truncates at 4000 chars | Loses content from long sessions | — |
| No cross-file dependency graph for code | Misses structural relationships | Aider (PageRank), CodeCompass, RANGER |
| No prompt repetition detection | Can't identify "missing memory" signals | — (novel) |
| No decision outcome tracking | Decisions made but never validated | — (novel) |
| Tool call outputs not captured | Loses 50%+ of session knowledge | — |
| No agent-driven memory curation | Agent can't decide what's important | Letta (8 tools), Mem0, LangMem |
| No memory pressure warnings | Context overflows silently | Letta (75% budget warning) |
| OpenHands integration missing | Misses a major agent platform | — |

---

## Architecture: Target State

```
                     ┌──────────────────────────────────┐
                     │        Agent Interfaces           │
                     │  MCP Server | CLI | Web Dashboard │
                     └──────────────┬───────────────────┘
                                    │
                     ┌──────────────▼───────────────────┐
                     │         Query Router              │
                     │  classifies → routes → merges     │
                     └──────────────┬───────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
   ┌────────▼────────┐    ┌────────▼────────┐    ┌─────────▼────────┐
   │   BM25 / FTS    │    │  Vector Search  │    │  Graph Traversal │
   │   (DuckDB FTS)  │    │   (LanceDB)     │    │  (DuckDB+DuckPGQ)│
   │  identifiers,   │    │  semantic sim,  │    │  dependencies,   │
   │  exact matches   │    │  NL queries     │    │  call chains,    │
   │                  │    │                 │    │  PageRank         │
   └────────┬────────┘    └────────┬────────┘    └─────────┬────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                     ┌──────────────▼───────────────────┐
                     │    Reciprocal Rank Fusion (RRF)   │
                     │    + MMR diversity + recency boost │
                     └──────────────┬───────────────────┘
                                    │
                     ┌──────────────▼───────────────────┐
                     │    Reranker (CodeRankLLM or       │
                     │    cross-encoder, top-N only)     │
                     └──────────────┬───────────────────┘
                                    │
                     ┌──────────────▼───────────────────┐
                     │    Context Assembly & Budgeting   │
                     │  dependency expansion, dedup,     │
                     │  token budget fitting, ordering   │
                     └──────────────────────────────────┘

Ingestion Pipeline:
  agent artifact → parser → extract sessions/decisions/facts/tool calls
    → LLM distillation (chunked, not truncated)
    → dedup check (vector similarity + LLM ADD/UPDATE/DELETE/NONE)
    → store in DuckDB + embed in LanceDB + build graph edges

Code Indexing Pipeline:
  file change → tree-sitter AST parse → function-level chunking
    → NL summary generation → hash comparison with stored chunks
    → embed changed chunks only (CodeRankEmbed or Voyage Code-3)
    → upsert LanceDB + update DuckDB metadata + rebuild FTS
    → extract/update dependency graph edges
```

---

## Roadmap: 4 Tiers

### TIER 0: Fix Foundations (Week 1-2)
*Make what exists actually work correctly.*

#### 0.1 Execute Consolidation Merges
The consolidation module finds merge candidates but never executes. Implement the actual merge: combine content, update confidence, redirect references, tombstone the merged-from record.

#### 0.2 Capture Tool Call Outputs
Currently only tool name + input are stored. Parse stdout/stderr from tool calls (especially bash, read, write). This is ~50% of session knowledge being thrown away.

#### 0.3 Chunked Distillation
Distillation truncates at 4000 chars. Implement sliding-window distillation: chunk transcripts into 4000-char windows with 500-char overlap, extract from each, then deduplicate extracted facts across windows.

#### 0.4 Cross-Session Dedup
Hash session transcripts on ingest. Skip already-ingested sessions. Currently re-ingesting produces duplicates.

#### 0.5 Fix Fact Deduplication
Before inserting any new fact/memory, search existing records by vector similarity (top-5). Use the Mem0 pattern: classify each match as ADD (new), UPDATE (merge with existing), DELETE (supersedes existing), or NONE (already known). This single change eliminates the biggest source of bloat.

---

### TIER 1: Deep Code RAG via Infiniloom (Week 2-4)
*Make code understanding world-class. Infiniloom already has most of this — wire it in.*

#### 1.1 Complete Infiniloom Bridge (code_analysis.rs)
The bridge defines its own CodeSymbol/CodeDependency structs but never calls DuckDB store methods.
DuckDB already has complete schema + CRUD (`insert_code_symbol`, `insert_code_dependency`,
`insert_analysis_run`, `clear_symbols_for_project`). Wire the bridge to use them:
- Map Infiniloom `IndexSymbol` to DuckDB `CodeSymbol` (add docstring, content_hash)
- Store ALL dependency edge types (calls, imports, inherits, implements — not just calls + file_imports)
- Store `Symbol.extends`/`Symbol.implements` relationships
- Store `Symbol.docstring` — currently parsed by Infiniloom but discarded
- Call `clear_symbols_for_project()` before re-analysis (idempotent)
- Call `insert_analysis_run()` to record run metadata

#### 1.2 Upgrade Chunking Strategy
Switch from `ChunkStrategy::Symbol` to `ChunkStrategy::Semantic` (Infiniloom has 6 strategies):
- `semantic`: splits at semantic boundaries (function/class declarations) — best for mixed content
- `dependency`: groups by dependency relationships — SOTA per Hydra paper (arXiv:2602.11671)
- Use `BudgetEnforcer` for exact token budgets per model (supports 27 models via tiktoken)
- Use `OutputFormatter` for agent-specific context: XML (Claude), Markdown (GPT), YAML (Gemini)

#### 1.3 Incremental Indexing via IncrementalScanner
Infiniloom has `IncrementalScanner` + `RepoCache` with BLAKE3 dirty-flag tracking:
- On file change: only re-parse + re-embed changed files
- `FileWatcher` (feature-gated) for real-time filesystem monitoring
- `--diff` / `--since` git-driven incremental updates
- DuckDB `content_hash` column already exists — use it for skip logic

#### 1.4 NL Summary Generation for Code Chunks
Greptile proved 12% retrieval improvement by embedding NL descriptions of code:
- For each code chunk, generate 1-2 sentence NL description via LLM
- Store `Symbol.docstring` as free NL description (already parsed, just store it)
- Embed the NL summary for semantic search, keep raw code for BM25
- Cache summaries, re-generate only when BLAKE3 hash changes

#### 1.5 BM25/Full-Text Search via DuckDB FTS
Add DuckDB FTS extension for identifier-exact search:
- Create FTS index on code_symbols (symbol_name + file_path + signature + docstring)
- Create FTS index on memories/facts/decisions (content)
- GrepRAG paper: simple lexical retrieval matches complex methods for code
- Identifier-weighted re-ranking: boost symbol_name matches over comment matches

#### 1.6 Dependency Graph with PageRank
Infiniloom provides full 4-phase graph: add files → extract imports → detect cycles (Tarjan) → PageRank:
- Store edges in DuckDB `code_dependencies` table (already exists)
- Use `SymbolRanker` weighted PageRank (ref 0.4, type 0.25, file 0.2, size 0.15)
- Store PageRank scores in `code_symbols.pagerank_score` (column exists)
- On retrieval: auto-expand retrieved chunks with direct dependencies
- Expose via MCP: `find_symbol`, `get_callers`, `get_callees`, `impact`

#### 1.7 Code-Specific Embedding Model
Switch from nomic-embed-text to a code-specific model:
- **Local**: CodeRankEmbed (137M params, MIT, outperforms 10x larger models) via ONNX
- **API**: Voyage Code-3 (32K context, Matryoshka dims, 300+ languages)
- Use query prefix: "Represent this query for searching relevant code:"

---

### TIER 2: Universal Agent Memory (Week 3-6)
*Make memory actually intelligent.*

#### 2.1 Semantic Memory Deduplication (Mem0 Pattern)
For every new memory/fact/decision extracted:
1. Vector search existing records (top-5 by cosine similarity)
2. Send candidates + new item to LLM with structured prompt
3. LLM classifies: ADD (genuinely new), UPDATE (merge with match), DELETE (old is wrong), NONE (already known)
4. Execute the classified action
This is how Mem0 achieves their claimed +26% accuracy.

#### 2.2 Temporal Knowledge Graph (Graphiti Pattern)
Upgrade facts to bi-temporal tracking:
- Every fact has `valid_at` (when it became true) and `invalid_at` (when it stopped being true)
- New facts that contradict existing facts trigger automatic invalidation of the old fact
- Point-in-time queries: "What was true about auth at time T?"
- Never delete — mark invalid. Full revision history preserved.
- Already have `valid_at`/`invalid_at`/`superseded_by` fields in DuckDB — implement the logic

#### 2.3 Agent-Driven Memory Curation (MCP Tools)
Expand MCP server with Letta-inspired self-editing memory tools:
- `mem_add` — already exists, enhance with dedup check
- `mem_update` — update existing memory by ID with new content
- `mem_rethink` — already exists, make it trigger re-evaluation + potential invalidation chain
- `mem_forget` — soft-delete (mark invalid) rather than hard delete
- `mem_consolidate` — trigger merge of related memories on demand
- `mem_priority` — let agent mark memories as critical/routine
- `mem_link` — create explicit relationships between memories

#### 2.4 Memory Pressure System
Track context budget and signal agents:
- Compute estimated tokens for assembled context
- Three pressure levels: Normal (<60%), Elevated (60-85%), Critical (>85%)
- At Elevated: suggest consolidation, drop low-confidence memories
- At Critical: aggressive pruning, keep only high-confidence + recent
- Expose via MCP: `mem_context` returns pressure level alongside content

#### 2.5 Decision Outcome Tracking
Close the feedback loop on decisions:
- When a decision is recorded, create a pending "outcome" record
- In subsequent sessions, look for signals that reference the decision (same topic, same files)
- Track: was the decision kept, reversed, or modified?
- Build decision effectiveness scores over time
- Surface: "You decided X last week. That decision was later reversed — consider Y instead."

#### 2.6 Prompt Repetition Detection
Detect "missing memory" signals:
- On each user prompt, compute semantic similarity to recent prompts (last 30 days)
- If similarity > 0.85 to a previous prompt in a different session: flag as repeated
- Track repetition counts per topic
- Auto-suggest: "This has been asked 4 times. Consider adding to project memory."
- Analytics: show repetition heatmap by topic

#### 2.7 OpenHands Integration
OpenHands has NO persistent memory — only session-scoped event streams:
- Write an OpenHands adapter (event stream parser)
- OpenHands events → Remembrant sessions/tool calls/decisions
- Expose Remembrant as an MCP server that OpenHands can query
- This makes Remembrant the memory layer OpenHands fundamentally lacks

#### 2.8 Background Consolidation Daemon
Run memory maintenance automatically:
- Periodic (hourly/daily) consolidation pass
- Merge similar memories, update confidence scores
- Expire memories past TTL
- Rebuild FTS indexes
- Compute analytics (prompt repetition, decision outcomes)
- Trigger via `rem watch` or standalone `rem consolidate --daemon`

---

### TIER 3: Analytics & Intelligence (Week 5-8)
*Make the invisible visible.*

#### 3.1 Project Evolution Timeline
Track how projects evolve through agent interactions:
- **What changed**: files modified, functions added/removed/changed
- **Why**: linked decisions and rationale
- **Who decided**: which agent, which session
- **Impact**: did the change stick or get reverted?
- Queryable: `rem timeline auth --since 2026-01` shows evolution of auth-related code
- Visualization in web dashboard

#### 3.2 Agent Effectiveness Analytics
Per-agent, per-project metrics:
- **Session metrics**: duration, token usage, tool call count, files touched
- **Success signals**: did the session produce a commit? Did tests pass?
- **Error patterns**: common tool failures, repeated corrections
- **Knowledge gaps**: topics where the agent repeatedly needed re-explanation
- **Comparative**: Claude vs Codex vs Gemini effectiveness by task type

#### 3.3 Cross-Project Pattern Mining
Identify patterns that repeat across projects:
- Same architectural decisions made in multiple projects
- Common error patterns and their solutions
- Reusable code patterns (auth, API clients, error handling)
- Surface via: `rem patterns --cross-project`
- Auto-suggest: "You implemented JWT auth in 3 projects. Here's your preferred pattern."

#### 3.4 Knowledge Graph Visualization (Web Dashboard)
Enhance the existing axum web dashboard:
- Interactive graph view of the knowledge graph (decisions, facts, code entities, relationships)
- Timeline view showing project evolution
- Analytics dashboards (prompt repetition, agent effectiveness, decision outcomes)
- Search interface with hybrid results + provenance links
- Use D3.js or similar for graph visualization

#### 3.5 Context Quality Scoring
Measure and improve context assembly quality:
- After each MCP `mem_context` call, track whether the agent used the provided context
- If the agent immediately searches for something else: context was insufficient
- If the agent references provided facts: context was useful
- Build a relevance feedback loop that improves retrieval over time

#### 3.6 Multi-Agent Conflict Detection
When multiple agents work on the same project:
- Detect contradictory decisions (Agent A chose approach X, Agent B chose approach Y)
- Surface conflicts: "Claude decided to use JWT. Codex later chose session cookies. Unresolved."
- Track consensus: facts confirmed by multiple agents get higher confidence
- Track handoffs: when one agent's output becomes another's input

---

## Key Technical Decisions

### Embedding Model Strategy
- **Default (local)**: CodeRankEmbed via ONNX (137M, MIT, code-optimized)
- **Alternative (local)**: nomic-embed-text via LM Studio (current, general-purpose)
- **Alternative (API)**: Voyage Code-3 (best quality, 32K context)
- Config: `[embedding] model = "coderank-embed"` with fallback chain

### Deduplication Strategy (Mem0 Pattern)
```
new_item → embed → vector_search(top_5) → LLM_classify(ADD|UPDATE|DELETE|NONE)
  → ADD:    insert new record
  → UPDATE: merge content, update confidence, keep older timestamp
  → DELETE: invalidate matched record, insert new
  → NONE:   skip (bump access_count on matched record)
```

### Code Chunking Strategy (Infiniloom-Powered)
```
file → Infiniloom IncrementalScanner (BLAKE3 dirty check)
  → tree-sitter AST parse (26 languages)
  → ChunkStrategy::Semantic (semantic boundary splitting)
  → BudgetEnforcer (exact token budget per model)
  → SecurityScanner (redact secrets before embedding)
  → NL summary generation (docstring or LLM)
  → embed NL summary (CodeRankEmbed) → LanceDB
  → store raw code + metadata → DuckDB + FTS index
```

### Hybrid Search Pipeline
```
query → classify(code|memory|decision|mixed)
  → parallel: BM25(DuckDB FTS) + Vector(LanceDB) + Graph(DuckPGQ)
  → RRF merge (k=60, weights by query type)
  → MMR diversity filter (λ=0.7)
  → rerank top-20 (CodeRankLLM or cross-encoder)
  → dependency expansion (add imports/callers for code results)
  → token budget fitting
  → ordered context assembly
```

### Memory Taxonomy (LangChain-Inspired)
- **Semantic memory**: Facts, decisions, conventions — deduplicated, temporal, mutable
- **Episodic memory**: Session transcripts, tool call history — append-only, immutable
- **Procedural memory**: Project rules, coding patterns — high-confidence, rarely changes
- **Code memory**: Symbols, dependencies, chunks — tied to source, incrementally updated

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Code retrieval relevance (MRR@10) | ~0.5 (estimated, naive chunks) | >0.8 (AST chunks + NL summaries + hybrid) |
| Memory dedup rate | 0% (no dedup) | >90% of duplicates caught |
| Prompt repetition detection | None | Detect >80% of repeated prompts |
| Decision outcome tracking | None | >50% of decisions have outcome recorded |
| Incremental index time (per file change) | Full re-embed | <2s per changed file |
| Context assembly latency | ~500ms | <200ms (Zep's target) |
| Agent platforms supported | 3 (Claude, Codex, Gemini) | 5+ (add OpenHands, Aider) |
| Cross-project pattern detection | Basic | Auto-surface in context briefings |

---

## What Makes This 100x Better Than Alternatives

1. **Mem0/Zep/Letta** are general-purpose conversation memory — they don't understand code. Remembrant has AST parsing, dependency graphs, PageRank, code-specific embeddings.

2. **Greptile/Sourcegraph Cody** are code search — they don't remember decisions, track project evolution, or work across agents. Remembrant unifies code understanding with agent memory.

3. **Claude Code's CLAUDE.md** is 200 lines of flat text with no search, no structure, no cross-project, no analytics. Remembrant is a queryable knowledge graph with semantic search.

4. **OpenHands** has zero persistent memory. Remembrant can be its entire memory layer via MCP.

5. **No existing solution** combines: multi-agent ingestion + temporal knowledge graph + AST-aware code RAG + hybrid search + decision tracking + prompt analytics + project evolution. That's the moat.
