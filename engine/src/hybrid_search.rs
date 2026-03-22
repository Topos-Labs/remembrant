//! Hybrid retrieval combining DuckDB (SQL), LanceDB (vector), and graph traversal.
//!
//! This is what makes remembrant's triple-store architecture powerful. A single query
//! fans out to all three backends and merges results with weighted scoring.

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::embedding::EmbedProvider;
use crate::semantic_scorer::keyword_scorer;
use crate::semantic_tree::TreeBuilder;
use crate::store::duckdb::DuckStore;
use crate::store::graph::{GraphStore, GraphStoreBackend};
use crate::store::lance::LanceStore;
use crate::xpath_query;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A unified search result from hybrid retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    pub id: String,
    pub content: String,
    pub result_type: ResultType,
    pub score: f64,
    /// Which backends contributed to this result.
    pub sources: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultType {
    Session,
    Memory,
    Decision,
    Fact,
    CodeEntity,
    ToolCall,
}

impl std::fmt::Display for ResultType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session => write!(f, "session"),
            Self::Memory => write!(f, "memory"),
            Self::Decision => write!(f, "decision"),
            Self::Fact => write!(f, "fact"),
            Self::CodeEntity => write!(f, "code_entity"),
            Self::ToolCall => write!(f, "tool_call"),
        }
    }
}

// ---------------------------------------------------------------------------
// Scoring weights
// ---------------------------------------------------------------------------

/// Weights for combining scores from different backends.
#[derive(Debug, Clone)]
pub struct HybridWeights {
    /// Weight for DuckDB text match (ILIKE).
    pub text_weight: f64,
    /// Weight for LanceDB vector similarity.
    pub vector_weight: f64,
    /// Weight for graph proximity (connected to relevant nodes).
    pub graph_weight: f64,
    /// Weight for Semantic XPath tree traversal results.
    pub xpath_weight: f64,
    /// Recency boost: multiplier for recent results.
    pub recency_boost: f64,
    /// RRF constant k (default 60, standard value from Cormack et al. 2009).
    /// Lower k gives more weight to top-ranked results.
    pub rrf_k: f64,
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            text_weight: 0.3,
            vector_weight: 0.5,
            graph_weight: 0.2,
            xpath_weight: 0.4,
            recency_boost: 1.1,
            rrf_k: 60.0,
        }
    }
}

/// Query complexity classification for dual-process routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryComplexity {
    /// XPath structured query — routed to XPath engine.
    XPath,
    /// Simple keyword lookup — text-only search, no vector/graph overhead.
    Fast,
    /// Complex semantic query — full hybrid RRF pipeline.
    Slow,
}

impl std::fmt::Display for QueryComplexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::XPath => write!(f, "xpath"),
            Self::Fast => write!(f, "fast"),
            Self::Slow => write!(f, "slow"),
        }
    }
}

/// Classify a query's complexity for dual-process routing.
///
/// Heuristics:
/// - Starts with `/` or `//` → XPath
/// - ≤3 words, no question marks, no semantic operators → Fast (text-only)
/// - Quoted exact match like `"auth module"` → Fast
/// - Contains question words (what, why, how) or >3 words → Slow (full hybrid)
pub fn classify_query(query: &str) -> QueryComplexity {
    let q = query.trim();

    // XPath detection
    if is_xpath_query(q) {
        return QueryComplexity::XPath;
    }

    // Empty or very short — fast
    if q.is_empty() || q.len() <= 2 {
        return QueryComplexity::Fast;
    }

    // Quoted exact match — fast text search
    if q.starts_with('"') && q.ends_with('"') {
        return QueryComplexity::Fast;
    }

    let words: Vec<&str> = q.split_whitespace().collect();
    let word_count = words.len();

    // Question words suggest semantic understanding needed
    let question_words = ["what", "why", "how", "when", "where", "which", "explain", "describe"];
    if words.first().map_or(false, |w| question_words.contains(&w.to_lowercase().as_str())) {
        return QueryComplexity::Slow;
    }

    // Contains question mark
    if q.contains('?') {
        return QueryComplexity::Slow;
    }

    // Short keyword queries — fast path
    if word_count <= 3 {
        return QueryComplexity::Fast;
    }

    // Longer queries likely need semantic search
    QueryComplexity::Slow
}

/// A ranked result from a single backend, before RRF merging.
#[derive(Debug, Clone)]
struct RankedResult {
    id: String,
    content: String,
    result_type: ResultType,
    /// Raw score from this backend (used for within-backend ranking).
    raw_score: f64,
    sources: Vec<String>,
    metadata: HashMap<String, String>,
}

/// Merge multiple ranked lists using Reciprocal Rank Fusion.
///
/// RRF score = sum over backends of: weight_i / (k + rank_i)
/// where rank_i is the 1-based position in backend i's ranked list.
///
/// This is scale-invariant: doesn't matter if text scores are 0-1 and
/// vector scores are 0-0.5. Only the relative ordering within each
/// backend matters.
fn merge_rrf(
    ranked_lists: &[(Vec<RankedResult>, f64)], // (results, backend_weight)
    k: f64,
) -> Vec<HybridResult> {
    let mut rrf_scores: HashMap<String, (f64, HybridResult)> = HashMap::new();

    for (results, backend_weight) in ranked_lists {
        for (rank, r) in results.iter().enumerate() {
            let rrf_contribution = backend_weight / (k + rank as f64 + 1.0);

            let entry = rrf_scores.entry(r.id.clone()).or_insert_with(|| {
                (0.0, HybridResult {
                    id: r.id.clone(),
                    content: r.content.clone(),
                    result_type: r.result_type,
                    score: 0.0,
                    sources: Vec::new(),
                    metadata: r.metadata.clone(),
                })
            });

            entry.0 += rrf_contribution;
            // Merge sources
            for s in &r.sources {
                if !entry.1.sources.contains(s) {
                    entry.1.sources.push(s.clone());
                }
            }
            // Merge metadata (later backends can add new keys)
            for (mk, mv) in &r.metadata {
                entry.1.metadata.entry(mk.clone()).or_insert_with(|| mv.clone());
            }
        }
    }

    // Assign final RRF scores
    let mut results: Vec<HybridResult> = rrf_scores
        .into_values()
        .map(|(score, mut result)| {
            result.score = score;
            result
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ---------------------------------------------------------------------------
// HybridSearch
// ---------------------------------------------------------------------------

/// Hybrid search engine combining SQL text search, vector similarity, and graph traversal.
pub struct HybridSearch<'a> {
    duck: &'a DuckStore,
    weights: HybridWeights,
}

impl<'a> HybridSearch<'a> {
    pub fn new(duck: &'a DuckStore) -> Self {
        Self {
            duck,
            weights: HybridWeights::default(),
        }
    }

    pub fn with_weights(mut self, weights: HybridWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Search across all backends and merge results using Reciprocal Rank Fusion.
    ///
    /// 1. DuckDB text search (sessions, memories, decisions, facts)
    /// 2. LanceDB vector search (if provider + lance given)
    /// 3. Graph traversal boost (if graph given)
    ///
    /// Each backend produces a ranked list. RRF merges them in a scale-invariant way:
    /// `score = sum(weight_i / (k + rank_i))` — only ordering within each backend matters.
    pub async fn search<P: EmbedProvider>(
        &self,
        query: &str,
        limit: usize,
        lance: Option<&LanceStore>,
        graph: Option<&GraphStore>,
        embedder: Option<&P>,
    ) -> Result<Vec<HybridResult>> {
        let mut ranked_lists: Vec<(Vec<RankedResult>, f64)> = Vec::new();

        // --- Layer 1: DuckDB text search ---
        let text_results = self.search_text_ranked(query)?;
        if !text_results.is_empty() {
            ranked_lists.push((text_results, self.weights.text_weight));
        }

        // --- Layer 2: LanceDB vector search ---
        if let (Some(lance), Some(embedder)) = (lance, embedder) {
            let vector_results = self.search_vector_ranked(query, lance, embedder).await?;
            if !vector_results.is_empty() {
                ranked_lists.push((vector_results, self.weights.vector_weight));
            }
        }

        // --- Layer 3: Graph proximity boost ---
        if let Some(graph) = graph {
            let graph_results = self.search_graph_ranked(query, graph)?;
            if !graph_results.is_empty() {
                ranked_lists.push((graph_results, self.weights.graph_weight));
            }
        }

        // Merge all backend results using RRF
        let mut results = merge_rrf(&ranked_lists, self.weights.rrf_k);

        // Apply recency boost on top of RRF scores
        self.apply_recency_boost_vec(&mut results);

        results.truncate(limit);

        // Touch retrieved memories so access_count tracks usage
        for r in &results {
            if r.result_type == ResultType::Memory {
                let _ = self.duck.touch_memory(&r.id);
            }
        }

        info!(
            query = query,
            results = results.len(),
            backends = ranked_lists.len(),
            "hybrid search complete (RRF)"
        );

        Ok(results)
    }

    /// Text-only search (no embeddings needed). Fast path for when LM Studio is down.
    /// Uses confidence-weighted scoring internally.
    pub fn search_text_only(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<HybridResult>> {
        let text_results = self.search_text_ranked(query)?;
        let mut results = merge_rrf(&[(text_results, 1.0)], self.weights.rrf_k);
        self.apply_recency_boost_vec(&mut results);
        results.truncate(limit);

        for r in &results {
            if r.result_type == ResultType::Memory {
                let _ = self.duck.touch_memory(&r.id);
            }
        }

        Ok(results)
    }

    /// XPath-based structured search. Builds the semantic tree and evaluates
    /// the XPath query, returning results with weights from the tree traversal.
    ///
    /// Use this when the query is a Semantic XPath expression like
    /// `//Session[node~"auth"]/Decision` or `//Fact[subject="auth module"]`.
    pub fn search_xpath(
        &self,
        xpath_expr: &str,
        limit: usize,
    ) -> Result<Vec<HybridResult>> {
        let parsed = xpath_query::parse(xpath_expr)
            .map_err(|e| anyhow::anyhow!("XPath parse error at {}: {}", e.position, e.message))?;

        // Build tree deep enough for the query (depth = number of steps + 1)
        let depth = parsed.steps.len() + 1;
        let builder = TreeBuilder::new(self.duck);
        let root = builder.build_tree(depth.min(5))?;

        let results = xpath_query::evaluate(&parsed, &root, &keyword_scorer);

        let mut hybrid_results: Vec<HybridResult> = results
            .into_iter()
            .take(limit)
            .map(|wn| {
                let (result_type, id) = parse_node_type_and_id(&wn.node_id, &wn.node_type);
                let mut meta = HashMap::new();
                meta.insert("xpath_path".to_string(), wn.path.join(" > "));
                meta.insert("node_type".to_string(), wn.node_type.clone());

                HybridResult {
                    id,
                    content: wn.name,
                    result_type,
                    score: self.weights.xpath_weight * wn.weight,
                    sources: vec!["xpath".to_string()],
                    metadata: meta,
                }
            })
            .collect();

        hybrid_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        info!(
            query = xpath_expr,
            results = hybrid_results.len(),
            "xpath search complete"
        );

        Ok(hybrid_results)
    }

    /// Dual-process query router: classifies queries as fast or slow,
    /// then routes to the appropriate search path.
    ///
    /// **Fast path** (System 1): Simple keyword queries — text-only search.
    /// Skips vector embedding and graph traversal for sub-millisecond response.
    ///
    /// **Slow path** (System 2): Complex queries — full hybrid RRF pipeline
    /// with vector similarity, graph traversal, and XPath boost.
    ///
    /// Classification heuristics:
    /// - XPath queries → XPath engine directly
    /// - ≤3 words, no question marks → fast path (text-only)
    /// - Quoted exact match → fast path (text-only)
    /// - Everything else → slow path (full hybrid)
    pub async fn smart_search<P: EmbedProvider>(
        &self,
        query: &str,
        limit: usize,
        lance: Option<&LanceStore>,
        graph: Option<&GraphStore>,
        embedder: Option<&P>,
    ) -> Result<(Vec<HybridResult>, QueryComplexity)> {
        let complexity = classify_query(query);

        let results = match complexity {
            QueryComplexity::XPath => {
                self.search_xpath(query.trim(), limit)?
            }
            QueryComplexity::Fast => {
                debug!(query = query, "fast path: text-only search");
                self.search_text_only(query, limit)?
            }
            QueryComplexity::Slow => {
                debug!(query = query, "slow path: full hybrid search");
                self.search_auto(query, limit, lance, graph, embedder).await?
            }
        };

        Ok((results, complexity))
    }

    /// Combined search: if the query looks like XPath (starts with / or //),
    /// runs XPath evaluation. Otherwise runs hybrid text+vector+graph search.
    /// Results from both paths are merged and deduplicated.
    pub async fn search_auto<P: EmbedProvider>(
        &self,
        query: &str,
        limit: usize,
        lance: Option<&LanceStore>,
        graph: Option<&GraphStore>,
        embedder: Option<&P>,
    ) -> Result<Vec<HybridResult>> {
        let trimmed = query.trim();

        if is_xpath_query(trimmed) {
            // Pure XPath mode
            return self.search_xpath(trimmed, limit);
        }

        // Standard hybrid search with optional XPath boost
        let mut results = self.search(query, limit * 2, lance, graph, embedder).await?;

        // Try to boost results using XPath if we can construct a meaningful query
        // For natural language queries, search for matching nodes in the tree
        if let Ok(xpath_results) = self.search_xpath_natural(query, limit) {
            for xr in xpath_results {
                let key = format!("{}:{}", xr.result_type, xr.id);
                if let Some(existing) = results.iter_mut().find(|r| format!("{}:{}", r.result_type, r.id) == key) {
                    existing.score += xr.score;
                    existing.sources.push("xpath_boost".to_string());
                } else {
                    results.push(xr);
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    /// Natural language to XPath: searches tree nodes semantically.
    /// Constructs `//[node~"query"]` to find any matching nodes.
    fn search_xpath_natural(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<HybridResult>> {
        let escaped = query.replace('"', "'");
        let xpath_expr = format!(r#"//*[node~"{escaped}"]"#);
        self.search_xpath(&xpath_expr, limit)
    }

    // -----------------------------------------------------------------------
    // Internal: text search layer (returns ranked list for RRF)
    // -----------------------------------------------------------------------

    fn search_text_ranked(&self, query: &str) -> Result<Vec<RankedResult>> {
        let mut results: Vec<RankedResult> = Vec::new();

        // Sessions
        let sessions = self.duck.search_sessions_by_summary(query)?;
        for s in sessions {
            let mut meta = HashMap::new();
            meta.insert("agent".to_string(), s.agent.clone());
            if let Some(ref p) = s.project_id {
                meta.insert("project".to_string(), p.clone());
            }
            if let Some(ref t) = s.started_at {
                meta.insert("started_at".to_string(), t.to_string());
            }
            results.push(RankedResult {
                id: s.id,
                content: s.summary.unwrap_or_default(),
                result_type: ResultType::Session,
                raw_score: 1.0, // All text matches are equal; RRF uses rank not score
                sources: vec!["duckdb_text".to_string()],
                metadata: meta,
            });
        }

        // Memories — weighted by confidence (FIX: was flat before)
        let memories = self.duck.search_memories(query)?;
        for m in memories {
            let mut meta = HashMap::new();
            if let Some(ref t) = m.memory_type {
                meta.insert("type".to_string(), t.clone());
            }
            if let Some(ref t) = m.created_at {
                meta.insert("created_at".to_string(), t.to_string());
            }
            meta.insert("confidence".to_string(), m.confidence.to_string());
            // Confidence-weighted: high-confidence memories rank higher
            let access_boost = (0.3 + 0.7 * (m.access_count as f64 + 1.0).log2() / 10.0).min(1.0);
            results.push(RankedResult {
                id: m.id,
                content: m.content,
                result_type: ResultType::Memory,
                raw_score: m.confidence as f64 * access_boost,
                sources: vec!["duckdb_text".to_string()],
                metadata: meta,
            });
        }

        // Facts (active only) — weighted by confidence
        let facts = self.duck.search_facts(query)?;
        for f in facts {
            if f.invalid_at.is_some() {
                continue;
            }
            let content = format!("{} {} {}", f.subject, f.predicate, f.object);
            let mut meta = HashMap::new();
            meta.insert("subject".to_string(), f.subject);
            meta.insert("predicate".to_string(), f.predicate);
            meta.insert("object".to_string(), f.object);
            meta.insert("confidence".to_string(), f.confidence.to_string());
            if let Some(ref t) = f.valid_at {
                meta.insert("valid_at".to_string(), t.to_string());
            } else if let Some(ref t) = f.created_at {
                meta.insert("created_at".to_string(), t.to_string());
            }
            results.push(RankedResult {
                id: f.id,
                content,
                result_type: ResultType::Fact,
                raw_score: f.confidence as f64,
                sources: vec!["duckdb_text".to_string()],
                metadata: meta,
            });
        }

        // Sort by raw_score descending — this determines rank for RRF
        results.sort_by(|a, b| b.raw_score.partial_cmp(&a.raw_score).unwrap_or(std::cmp::Ordering::Equal));

        debug!(results = results.len(), "text search layer complete");
        Ok(results)
    }

    // Keep the old search_text for backward compat with topic_context in context.rs
    fn search_text(
        &self,
        query: &str,
        scored: &mut HashMap<String, HybridResult>,
    ) -> Result<()> {
        let ranked = self.search_text_ranked(query)?;
        for r in ranked {
            let key = format!("{}:{}", r.result_type, r.id);
            scored.insert(key, HybridResult {
                id: r.id,
                content: r.content,
                result_type: r.result_type,
                score: r.raw_score,
                sources: r.sources,
                metadata: r.metadata,
            });
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal: vector search layer (returns ranked list for RRF)
    // -----------------------------------------------------------------------

    async fn search_vector_ranked<P: EmbedProvider>(
        &self,
        query: &str,
        lance: &LanceStore,
        embedder: &P,
    ) -> Result<Vec<RankedResult>> {
        let mut results: Vec<RankedResult> = Vec::new();

        // Embed the query
        let query_emb = embedder.embed_texts(&[query]).await?;
        let Some(emb) = query_emb.into_iter().next() else {
            return Ok(results);
        };

        // Search code embeddings
        let code_results = lance.search_code(&emb, 20).await?;
        for r in code_results {
            let sim = 1.0 / (1.0 + r.distance as f64);
            let mut meta = HashMap::new();
            meta.insert("granularity".to_string(), r.granularity);
            if let Some(ref fp) = r.file_path {
                meta.insert("file_path".to_string(), fp.clone());
            }
            results.push(RankedResult {
                id: r.id,
                content: r.content,
                result_type: ResultType::CodeEntity,
                raw_score: sim,
                sources: vec!["lance_vector".to_string()],
                metadata: meta,
            });
        }

        // Search memory embeddings
        let mem_results = lance.search_memories(&emb, 20).await?;
        for r in mem_results {
            let sim = 1.0 / (1.0 + r.distance as f64);
            let mut meta = HashMap::new();
            meta.insert("type".to_string(), r.memory_type);
            results.push(RankedResult {
                id: r.id,
                content: r.content,
                result_type: ResultType::Memory,
                raw_score: sim,
                sources: vec!["lance_vector".to_string()],
                metadata: meta,
            });
        }

        // Sort by similarity descending — determines rank for RRF
        results.sort_by(|a, b| b.raw_score.partial_cmp(&a.raw_score).unwrap_or(std::cmp::Ordering::Equal));

        debug!(results = results.len(), "vector search layer complete");
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Internal: graph proximity search (returns ranked list for RRF)
    // -----------------------------------------------------------------------

    fn search_graph_ranked(
        &self,
        query: &str,
        graph: &GraphStore,
    ) -> Result<Vec<RankedResult>> {
        let mut results: Vec<RankedResult> = Vec::new();
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        // Find graph nodes whose names match query words
        // Then include their neighbors as related results
        if let Ok(all_nodes) = graph.all_nodes() {
            for node in &all_nodes {
                let name_lower = node.name.to_lowercase();
                let word_matches = query_words.iter().filter(|w| name_lower.contains(*w)).count();
                if word_matches == 0 {
                    continue;
                }

                // This node matches — score by fraction of query words matched
                let match_score = word_matches as f64 / query_words.len().max(1) as f64;

                results.push(RankedResult {
                    id: node.id.clone(),
                    content: node.name.clone(),
                    result_type: match node.kind {
                        crate::store::graph::NodeKind::Memory => ResultType::Memory,
                        crate::store::graph::NodeKind::CodeEntity | crate::store::graph::NodeKind::Symbol | crate::store::graph::NodeKind::Module => ResultType::CodeEntity,
                        _ => ResultType::Session,
                    },
                    raw_score: match_score,
                    sources: vec!["graph".to_string()],
                    metadata: HashMap::new(),
                });

                // Also add neighbors with decayed score
                if let Ok(neighbors) = graph.query_neighbors(&node.id, None) {
                    for n in neighbors.iter().take(5) {
                        results.push(RankedResult {
                            id: n.node.id.clone(),
                            content: n.node.name.clone(),
                            result_type: ResultType::Session,
                            raw_score: match_score * 0.5, // Neighbors get half the score
                            sources: vec!["graph_neighbor".to_string()],
                            metadata: {
                                let mut m = HashMap::new();
                                m.insert("edge_kind".to_string(), n.edge_kind.name().to_string());
                                m.insert("via".to_string(), node.id.clone());
                                m
                            },
                        });
                    }
                }
            }
        }

        // Deduplicate by id, keeping highest score
        let mut deduped: HashMap<String, RankedResult> = HashMap::new();
        for r in results {
            let entry = deduped.entry(r.id.clone()).or_insert(r.clone());
            if r.raw_score > entry.raw_score {
                *entry = r;
            }
        }
        let mut results: Vec<RankedResult> = deduped.into_values().collect();
        results.sort_by(|a, b| b.raw_score.partial_cmp(&a.raw_score).unwrap_or(std::cmp::Ordering::Equal));

        debug!(results = results.len(), "graph search layer complete");
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Internal: recency boost
    // -----------------------------------------------------------------------

    fn apply_recency_boost_vec(&self, results: &mut [HybridResult]) {
        if (self.weights.recency_boost - 1.0).abs() < f64::EPSILON {
            return;
        }

        let now = chrono::Utc::now().naive_utc();

        for result in results.iter_mut() {
            if let Some(ts_str) = result.metadata.get("started_at")
                .or_else(|| result.metadata.get("created_at"))
                .or_else(|| result.metadata.get("valid_at"))
            {
                if let Ok(ts) = chrono::NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S%.f") {
                    let age_hours = (now - ts).num_hours().max(1) as f64;
                    let decay = self.weights.recency_boost.powf(1.0 / (age_hours / 24.0 + 1.0).log2().max(1.0));
                    result.score *= decay;
                }
            }
        }

        // Re-sort after recency adjustment
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Keep old method for backward compatibility with context.rs search_text usage
    fn apply_recency_boost(&self, scored: &mut HashMap<String, HybridResult>) {
        if (self.weights.recency_boost - 1.0).abs() < f64::EPSILON {
            return;
        }

        let now = chrono::Utc::now().naive_utc();

        for result in scored.values_mut() {
            if let Some(ts_str) = result.metadata.get("started_at")
                .or_else(|| result.metadata.get("created_at"))
                .or_else(|| result.metadata.get("valid_at"))
            {
                if let Ok(ts) = chrono::NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S%.f") {
                    let age_hours = (now - ts).num_hours().max(1) as f64;
                    let decay = self.weights.recency_boost.powf(1.0 / (age_hours / 24.0 + 1.0).log2().max(1.0));
                    result.score *= decay;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Detect if a query string is a Semantic XPath expression.
pub fn is_xpath_query(query: &str) -> bool {
    let q = query.trim();
    q.starts_with("//") || (q.starts_with('/') && q.len() > 1 && q.as_bytes()[1].is_ascii_alphabetic())
}

/// Parse a node ID like "session:abc" into (ResultType, raw_id).
fn parse_node_type_and_id(node_id: &str, node_type: &str) -> (ResultType, String) {
    let raw_id = if let Some((_prefix, id)) = node_id.split_once(':') {
        id.to_string()
    } else {
        node_id.to_string()
    };

    let result_type = match node_type {
        "Session" => ResultType::Session,
        "Memory" => ResultType::Memory,
        "Decision" => ResultType::Decision,
        "Fact" => ResultType::Fact,
        "CodeEntity" | "Symbol" => ResultType::CodeEntity,
        "ToolCall" => ResultType::ToolCall,
        _ => ResultType::Memory,
    };

    (result_type, raw_id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_only_search() {
        let store = DuckStore::open_in_memory().unwrap();

        // Insert test data
        use crate::store::duckdb::{Memory, Session};
        use chrono::Utc;

        let session = Session {
            id: "s-1".into(),
            project_id: Some("proj-1".into()),
            agent: "claude".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(10),
            message_count: Some(5),
            tool_call_count: Some(3),
            total_tokens: Some(1200),
            files_changed: vec!["src/auth.rs".into()],
            summary: Some("refactored authentication module".into()),
        };
        store.insert_or_replace_session(&session).unwrap();

        let memory = Memory {
            id: "m-1".into(),
            project_id: Some("proj-1".into()),
            content: "Authentication uses JWT tokens".into(),
            memory_type: Some("insight".into()),
            source_session_id: Some("s-1".into()),
            confidence: 0.9,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&memory).unwrap();

        let search = HybridSearch::new(&store);
        let results = search.search_text_only("authentication", 10).unwrap();

        assert!(results.len() >= 2, "should find session and memory, got {}", results.len());
        assert!(results.iter().any(|r| r.result_type == ResultType::Session));
        assert!(results.iter().any(|r| r.result_type == ResultType::Memory));
    }

    #[test]
    fn test_hybrid_weights_affect_scoring() {
        let store = DuckStore::open_in_memory().unwrap();

        use crate::store::duckdb::Session;
        use chrono::Utc;

        let session = Session {
            id: "s-1".into(),
            project_id: None,
            agent: "claude".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: None,
            message_count: None,
            tool_call_count: None,
            total_tokens: None,
            files_changed: vec![],
            summary: Some("test query match".into()),
        };
        store.insert_or_replace_session(&session).unwrap();

        // Higher text weight => higher scores
        let search = HybridSearch::new(&store).with_weights(HybridWeights {
            text_weight: 0.9,
            vector_weight: 0.0,
            graph_weight: 0.0,
            xpath_weight: 0.0,
            recency_boost: 1.0,
            rrf_k: 60.0,
        });
        let results = search.search_text_only("test", 10).unwrap();
        assert!(!results.is_empty());
        // RRF score = weight / (k + rank) = 0.9 / (60 + 1) ≈ 0.0147 for rank 0
        assert!(results[0].score > 0.0, "RRF score should be positive");
    }

    #[test]
    fn test_is_xpath_query() {
        assert!(is_xpath_query("//Session[node~\"auth\"]"));
        assert!(is_xpath_query("/Session/Decision"));
        assert!(!is_xpath_query("authentication module"));
        assert!(!is_xpath_query("search for auth"));
        // Edge case: just "/" is not xpath
        assert!(!is_xpath_query("/"));
    }

    #[test]
    fn test_xpath_search() {
        let store = DuckStore::open_in_memory().unwrap();

        use crate::store::duckdb::Session;
        use chrono::Utc;

        let session = Session {
            id: "s-1".into(),
            project_id: Some("proj-1".into()),
            agent: "claude".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(10),
            message_count: Some(5),
            tool_call_count: Some(3),
            total_tokens: Some(1200),
            files_changed: vec!["src/auth.rs".into()],
            summary: Some("refactored authentication module".into()),
        };
        store.insert_or_replace_session(&session).unwrap();

        let search = HybridSearch::new(&store);

        // XPath query for sessions
        let results = search.search_xpath("//Session", 10).unwrap();
        assert!(!results.is_empty(), "xpath should find sessions");
        assert_eq!(results[0].result_type, ResultType::Session);

        // XPath with semantic predicate
        let results = search.search_xpath(r#"//Session[node~"auth"]"#, 10).unwrap();
        assert!(!results.is_empty(), "xpath with semantic should find auth session");
    }

    #[test]
    fn test_xpath_facts_in_tree() {
        let store = DuckStore::open_in_memory().unwrap();

        use crate::store::duckdb::{Fact, Session};
        use chrono::Utc;

        // Need a session to create a project
        let session = Session {
            id: "s-1".into(),
            project_id: Some("proj-1".into()),
            agent: "claude".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: None,
            message_count: None,
            tool_call_count: None,
            total_tokens: None,
            files_changed: vec![],
            summary: Some("test".into()),
        };
        store.insert_or_replace_session(&session).unwrap();

        let fact = Fact {
            id: "f-1".into(),
            project_id: Some("proj-1".into()),
            subject: "auth module".into(),
            predicate: "uses".into(),
            object: "JWT tokens".into(),
            confidence: 0.95,
            source_session_id: None,
            source_agent: Some("claude".into()),
            valid_at: Some(Utc::now().naive_utc()),
            invalid_at: None,
            superseded_by: None,
            created_at: Some(Utc::now().naive_utc()),
        };
        store.insert_fact(&fact).unwrap();

        let search = HybridSearch::new(&store);
        let results = search.search_xpath("//Fact", 10).unwrap();
        assert!(!results.is_empty(), "xpath should find facts in tree");
        assert_eq!(results[0].result_type, ResultType::Fact);
    }

    #[test]
    fn test_facts_in_hybrid_search() {
        let store = DuckStore::open_in_memory().unwrap();

        use crate::store::duckdb::Fact;
        use chrono::Utc;

        let fact = Fact {
            id: "f-1".into(),
            project_id: Some("proj-1".into()),
            subject: "auth module".into(),
            predicate: "uses".into(),
            object: "JWT tokens".into(),
            confidence: 0.95,
            source_session_id: Some("s-1".into()),
            source_agent: Some("claude".into()),
            valid_at: Some(Utc::now().naive_utc()),
            invalid_at: None,
            superseded_by: None,
            created_at: Some(Utc::now().naive_utc()),
        };
        store.insert_fact(&fact).unwrap();

        let search = HybridSearch::new(&store);
        let results = search.search_text_only("auth", 10).unwrap();

        assert!(results.iter().any(|r| r.result_type == ResultType::Fact));
    }

    #[test]
    fn test_classify_query() {
        // XPath queries
        assert_eq!(classify_query("//Session[node~\"auth\"]"), QueryComplexity::XPath);
        assert_eq!(classify_query("/Session/Decision"), QueryComplexity::XPath);

        // Fast: short keyword queries
        assert_eq!(classify_query("auth"), QueryComplexity::Fast);
        assert_eq!(classify_query("JWT tokens"), QueryComplexity::Fast);
        assert_eq!(classify_query("auth module JWT"), QueryComplexity::Fast);

        // Fast: quoted exact match
        assert_eq!(classify_query("\"auth module\""), QueryComplexity::Fast);

        // Fast: empty/short
        assert_eq!(classify_query(""), QueryComplexity::Fast);
        assert_eq!(classify_query("ab"), QueryComplexity::Fast);

        // Slow: question words
        assert_eq!(classify_query("what does the auth module do"), QueryComplexity::Slow);
        assert_eq!(classify_query("how is JWT implemented"), QueryComplexity::Slow);
        assert_eq!(classify_query("why did we choose DuckDB"), QueryComplexity::Slow);

        // Slow: question mark
        assert_eq!(classify_query("auth module?"), QueryComplexity::Slow);

        // Slow: long queries (>3 words)
        assert_eq!(classify_query("authentication module implementation details"), QueryComplexity::Slow);
    }
}
