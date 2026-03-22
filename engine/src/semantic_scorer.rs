//! Semantic scoring bridge for XPath `[node≈"..."]` predicates.
//!
//! Provides a `SemanticScorer` that can precompute query embeddings for batch
//! efficiency, and a simple `keyword_scorer` fallback that works offline without
//! any embedding provider.

use std::collections::HashMap;

use anyhow::Result;

use crate::embedding::EmbedProvider;
use crate::xpath_query::{Predicate, XPathQuery};

// ---------------------------------------------------------------------------
// SemanticScorer
// ---------------------------------------------------------------------------

/// Semantic scorer that uses embedding similarity for `[node≈"query"]` predicates.
///
/// For the MVP the actual scoring falls back to keyword overlap. The cache
/// infrastructure is in place so that embedding-based cosine similarity can be
/// plugged in later when LanceDB node embeddings are available.
pub struct SemanticScorer {
    /// Precomputed query/node embeddings cache: text -> embedding vector.
    pub(crate) cache: HashMap<String, Vec<f32>>,
}

impl SemanticScorer {
    /// Create a new (empty) semantic scorer.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Precompute embeddings for all semantic query strings found in an `XPathQuery`.
    ///
    /// Call this before `evaluate()` to batch-embed all queries efficiently.
    pub async fn precompute<P: EmbedProvider>(
        &mut self,
        query: &XPathQuery,
        provider: &P,
    ) -> Result<()> {
        let mut texts: Vec<String> = Vec::new();
        for step in &query.steps {
            collect_semantic_strings(&step.predicates, &mut texts);
        }

        // Deduplicate
        texts.sort();
        texts.dedup();

        if texts.is_empty() {
            return Ok(());
        }

        // Batch embed
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = provider.embed_texts(&refs).await?;

        for (text, embedding) in texts.into_iter().zip(embeddings.into_iter()) {
            self.cache.insert(text, embedding);
        }

        Ok(())
    }

    /// Score a node's text against a query string.
    ///
    /// Returns a similarity score in `[0.0, 1.0]`. If embeddings are cached for
    /// both the query and node text, uses cosine similarity. Otherwise falls back
    /// to `keyword_scorer`.
    pub fn score(&self, node_text: &str, query_text: &str) -> f64 {
        if let (Some(query_emb), Some(node_emb)) =
            (self.cache.get(query_text), self.cache.get(node_text))
        {
            // Both embeddings available: use cosine similarity
            let sim = cosine_similarity(query_emb, node_emb);
            // Cosine similarity is in [-1, 1]; clamp to [0, 1]
            sim.max(0.0)
        } else if let Some(_query_emb) = self.cache.get(query_text) {
            // Query embedding available but no node embedding: keyword fallback
            keyword_scorer(node_text, query_text)
        } else {
            keyword_scorer(node_text, query_text)
        }
    }

    /// Precompute embeddings for node texts (used to enable cosine scoring).
    pub async fn precompute_nodes<P: EmbedProvider>(
        &mut self,
        node_texts: &[&str],
        provider: &P,
    ) -> Result<()> {
        // Filter out already-cached texts
        let new_texts: Vec<&str> = node_texts
            .iter()
            .filter(|t| !self.cache.contains_key(**t))
            .copied()
            .collect();

        if new_texts.is_empty() {
            return Ok(());
        }

        let refs: Vec<&str> = new_texts.iter().copied().collect();
        let embeddings = provider.embed_texts(&refs).await?;

        for (text, embedding) in new_texts.into_iter().zip(embeddings.into_iter()) {
            self.cache.insert(text.to_string(), embedding);
        }

        Ok(())
    }

    /// Create a scorer closure suitable for use with `evaluate()`.
    pub fn as_scorer(&self) -> impl Fn(&str, &str) -> f64 + '_ {
        |node_text, query_text| self.score(node_text, query_text)
    }
}

impl Default for SemanticScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// keyword_scorer
// ---------------------------------------------------------------------------

/// Simple keyword-based scorer (no embeddings needed, good for testing and fallback).
///
/// Lowercases both texts, splits the query into words, counts how many query
/// words appear anywhere in the node text, and returns the ratio
/// `matching_words / total_query_words`.
pub fn keyword_scorer(node_text: &str, query_text: &str) -> f64 {
    let node_lower = node_text.to_lowercase();
    let query_lower = query_text.to_lowercase();

    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    if query_words.is_empty() {
        return 0.0;
    }

    let matching = query_words
        .iter()
        .filter(|w| node_lower.contains(*w))
        .count();

    matching as f64 / query_words.len() as f64
}

// ---------------------------------------------------------------------------
// BenchmarkResult & benchmark_xpath_vs_flat
// ---------------------------------------------------------------------------

/// Results from comparing Semantic XPath search vs flat text search.
#[derive(Debug)]
pub struct BenchmarkResult {
    /// Number of results from XPath query.
    pub xpath_results: usize,
    /// Number of results from flat search.
    pub flat_results: usize,
    /// Total characters in XPath result names (proxy for tokens).
    pub xpath_chars: usize,
    /// Total characters in flat search results (proxy for tokens).
    pub flat_chars: usize,
    /// Time to parse the XPath query (milliseconds).
    pub xpath_parse_ms: u64,
    /// Time to build tree + evaluate the XPath query (milliseconds).
    pub xpath_eval_ms: u64,
}

/// Compare Semantic XPath vs flat search on a `DuckStore`.
///
/// Returns a `BenchmarkResult` with counts and timing for both approaches.
pub fn benchmark_xpath_vs_flat(
    store: &crate::store::duckdb::DuckStore,
    xpath_query: &str,
    flat_query: &str,
    depth: usize,
) -> Result<BenchmarkResult> {
    use std::time::Instant;

    // --- XPath path ---
    let t0 = Instant::now();
    let parsed = crate::xpath_query::parse(xpath_query)
        .map_err(|e| anyhow::anyhow!("XPath parse error at {}: {}", e.position, e.message))?;
    let xpath_parse_ms = t0.elapsed().as_millis() as u64;

    let t1 = Instant::now();
    let builder = crate::semantic_tree::TreeBuilder::new(store);
    let root = builder.build_tree(depth)?;
    let results = crate::xpath_query::evaluate(&parsed, &root, &keyword_scorer);
    let xpath_eval_ms = t1.elapsed().as_millis() as u64;

    let xpath_results = results.len();
    let xpath_chars: usize = results.iter().map(|r| r.name.len()).sum();

    // --- Flat search path ---
    let flat_sessions = store.search_sessions_by_summary(flat_query)?;
    let flat_results = flat_sessions.len();
    let flat_chars: usize = flat_sessions
        .iter()
        .map(|s| s.summary.as_deref().unwrap_or("").len())
        .sum();

    Ok(BenchmarkResult {
        xpath_results,
        flat_results,
        xpath_chars,
        flat_chars,
        xpath_parse_ms,
        xpath_eval_ms,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors. Returns a value in `[-1.0, 1.0]`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Recursively collect all semantic query strings from a list of predicates.
fn collect_semantic_strings(predicates: &[Predicate], out: &mut Vec<String>) {
    for pred in predicates {
        collect_semantic_strings_from_predicate(pred, out);
    }
}

/// Recursively collect semantic strings from a single predicate.
fn collect_semantic_strings_from_predicate(pred: &Predicate, out: &mut Vec<String>) {
    match pred {
        Predicate::Semantic(s) => {
            out.push(s.clone());
        }
        Predicate::Aggregate(_op, subquery, _semantic_text) => {
            // Walk the subquery's steps for any semantic predicates
            for step in &subquery.steps {
                collect_semantic_strings(&step.predicates, out);
            }
        }
        Predicate::And(a, b) | Predicate::Or(a, b) => {
            collect_semantic_strings_from_predicate(a, out);
            collect_semantic_strings_from_predicate(b, out);
        }
        Predicate::Not(inner) => {
            collect_semantic_strings_from_predicate(inner, out);
        }
        // No semantic strings in these variants
        Predicate::Position(_)
        | Predicate::Range(_, _)
        | Predicate::AttrEquals(_, _)
        | Predicate::Comparison(_, _, _) => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_scorer_full_match() {
        let score = keyword_scorer("Authentication and login session", "authentication login");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_keyword_scorer_partial_match() {
        let score = keyword_scorer("Authentication session", "authentication login");
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_keyword_scorer_no_match() {
        let score = keyword_scorer("DuckDB schema", "authentication login");
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_keyword_scorer_empty_query() {
        let score = keyword_scorer("some text", "");
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_keyword_scorer_case_insensitive() {
        let score = keyword_scorer("AUTHENTICATION Login", "authentication LOGIN");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_keyword_scorer_single_word() {
        let score = keyword_scorer("refactored the auth module", "auth");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_semantic_scorer_fallback() {
        let scorer = SemanticScorer::new();
        let score = scorer.score("Authentication session", "authentication");
        assert!(score > 0.0);
    }

    #[test]
    fn test_semantic_scorer_as_closure() {
        let scorer = SemanticScorer::new();
        let f = scorer.as_scorer();
        let score = f("Authentication session", "authentication");
        assert!(score > 0.0);
    }

    #[test]
    fn test_collect_semantic_strings_simple() {
        let query = crate::xpath_query::parse(r#"//Session[node~"auth"]/Decision"#).unwrap();
        let mut strings = Vec::new();
        for step in &query.steps {
            collect_semantic_strings(&step.predicates, &mut strings);
        }
        assert_eq!(strings, vec!["auth"]);
    }

    #[test]
    fn test_collect_semantic_strings_aggregate() {
        let query =
            crate::xpath_query::parse(r#"//Session[avg(/ToolCall[node~"auth"])]/Decision"#)
                .unwrap();
        let mut strings = Vec::new();
        for step in &query.steps {
            collect_semantic_strings(&step.predicates, &mut strings);
        }
        assert_eq!(strings, vec!["auth"]);
    }

    #[test]
    fn test_collect_semantic_strings_and_or() {
        let query =
            crate::xpath_query::parse(r#"//Session[node~"auth" or node~"login"]"#).unwrap();
        let mut strings = Vec::new();
        for step in &query.steps {
            collect_semantic_strings(&step.predicates, &mut strings);
        }
        assert!(strings.contains(&"auth".to_string()));
        assert!(strings.contains(&"login".to_string()));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_semantic_scorer_with_cached_embeddings() {
        let mut scorer = SemanticScorer::new();
        // Manually insert embeddings to test cosine path
        scorer.cache.insert("auth login".to_string(), vec![1.0, 0.0, 0.0]);
        scorer.cache.insert("authentication session".to_string(), vec![0.9, 0.1, 0.0]);
        let score = scorer.score("authentication session", "auth login");
        // Should use cosine similarity, not keyword scorer
        assert!(score > 0.9, "cosine similarity should be high, got {score}");
    }

    #[test]
    fn test_collect_semantic_strings_no_semantic() {
        let query = crate::xpath_query::parse(r#"//Session[agent="claude"]"#).unwrap();
        let mut strings = Vec::new();
        for step in &query.steps {
            collect_semantic_strings(&step.predicates, &mut strings);
        }
        assert!(strings.is_empty());
    }
}
