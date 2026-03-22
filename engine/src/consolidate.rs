//! Memory consolidation, decay, and lifecycle management.
//!
//! Implements:
//! - **Decay scoring**: Relevance = f(confidence, access_count, age, recency)
//! - **Duplicate detection**: Find near-identical memories for merging
//! - **TTL enforcement**: Invalidate memories past their valid_until date
//! - **Consolidation**: Merge similar memories, boosting confidence

use anyhow::Result;
use chrono::Utc;

use crate::store::duckdb::DuckStore;

/// Decay score for a memory, combining multiple signals into [0, 1].
#[derive(Debug, Clone)]
pub struct DecayScore {
    pub memory_id: String,
    pub score: f64,
    /// Component breakdown for debugging.
    pub components: DecayComponents,
}

#[derive(Debug, Clone)]
pub struct DecayComponents {
    pub confidence: f64,
    pub access_frequency: f64,
    pub recency: f64,
    pub age_penalty: f64,
}

/// Consolidation result for a pair of memories.
#[derive(Debug)]
pub struct MergeCandidate {
    pub memory_a: String,
    pub memory_b: String,
    /// Jaccard similarity of content tokens.
    pub similarity: f64,
}

/// Stats from a consolidation run.
#[derive(Debug, Default)]
pub struct ConsolidationStats {
    pub expired_count: usize,
    pub merged_count: usize,
    pub scored_count: usize,
}

/// Compute decay scores for all memories in a project.
///
/// Score formula:
///   score = confidence * access_boost * recency_factor * (1 - age_penalty)
///
/// Where:
///   access_boost = min(1.0, 0.3 + 0.7 * log2(access_count + 1) / 10)
///   recency_factor = 1 / (1 + days_since_last_access / 30)
///   age_penalty = min(0.5, days_since_creation / 365)
pub fn compute_decay_scores(store: &DuckStore, project: Option<&str>) -> Result<Vec<DecayScore>> {
    let memories = store.get_memories(project, 10_000)?;
    let now = Utc::now().naive_utc();

    let mut scores = Vec::with_capacity(memories.len());

    for m in &memories {
        let confidence = m.confidence as f64;

        let access_boost = (0.3 + 0.7 * ((m.access_count as f64 + 1.0).log2() / 10.0)).min(1.0);

        let days_since_update = m
            .updated_at
            .map(|t| (now - t).num_hours().max(0) as f64 / 24.0)
            .unwrap_or(30.0);
        let recency = 1.0 / (1.0 + days_since_update / 30.0);

        let days_since_creation = m
            .created_at
            .map(|t| (now - t).num_hours().max(0) as f64 / 24.0)
            .unwrap_or(90.0);
        let age_penalty = (days_since_creation / 365.0).min(0.5);

        let score = confidence * access_boost * recency * (1.0 - age_penalty);

        scores.push(DecayScore {
            memory_id: m.id.clone(),
            score: score.clamp(0.0, 1.0),
            components: DecayComponents {
                confidence,
                access_frequency: access_boost,
                recency,
                age_penalty,
            },
        });
    }

    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(scores)
}

/// Find pairs of memories with high content overlap (candidates for merging).
/// Uses Jaccard similarity on word tokens.
pub fn find_merge_candidates(
    store: &DuckStore,
    project: Option<&str>,
    threshold: f64,
) -> Result<Vec<MergeCandidate>> {
    let memories = store.get_memories(project, 5_000)?;
    let mut candidates = Vec::new();

    // Tokenize all memories
    let tokenized: Vec<(String, std::collections::HashSet<String>)> = memories
        .iter()
        .map(|m| {
            let tokens: std::collections::HashSet<String> = m
                .content
                .to_lowercase()
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| w.len() > 2)
                .collect();
            (m.id.clone(), tokens)
        })
        .collect();

    // Pairwise comparison (O(n^2) but memories are typically < 1000)
    for i in 0..tokenized.len() {
        for j in (i + 1)..tokenized.len() {
            let (ref id_a, ref tokens_a) = tokenized[i];
            let (ref id_b, ref tokens_b) = tokenized[j];

            if tokens_a.is_empty() || tokens_b.is_empty() {
                continue;
            }

            let intersection = tokens_a.intersection(tokens_b).count();
            let union = tokens_a.union(tokens_b).count();
            let similarity = intersection as f64 / union as f64;

            if similarity >= threshold {
                candidates.push(MergeCandidate {
                    memory_a: id_a.clone(),
                    memory_b: id_b.clone(),
                    similarity,
                });
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(candidates)
}

/// Expire memories that are past their valid_until date.
pub fn expire_stale_memories(store: &DuckStore) -> Result<usize> {
    let conn = store.connection().lock().expect("lock poisoned");
    let now = Utc::now().naive_utc();
    let count = conn.execute(
        "DELETE FROM memories WHERE valid_until IS NOT NULL AND valid_until < ?",
        duckdb::params![now],
    )?;
    Ok(count)
}

/// Run full consolidation: expire, find duplicates, compute scores.
pub fn consolidate(
    store: &DuckStore,
    project: Option<&str>,
    merge_threshold: f64,
) -> Result<(ConsolidationStats, Vec<DecayScore>, Vec<MergeCandidate>)> {
    let expired_count = expire_stale_memories(store)?;
    let candidates = find_merge_candidates(store, project, merge_threshold)?;
    let scores = compute_decay_scores(store, project)?;

    let stats = ConsolidationStats {
        expired_count,
        merged_count: candidates.len(),
        scored_count: scores.len(),
    };

    Ok((stats, scores, candidates))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::duckdb::Memory;

    fn make_memory(id: &str, content: &str, confidence: f32, access_count: i32) -> Memory {
        Memory {
            id: id.into(),
            project_id: Some("test".into()),
            content: content.into(),
            memory_type: Some("insight".into()),
            source_session_id: None,
            confidence,
            access_count,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        }
    }

    #[test]
    fn test_decay_scores() {
        let store = DuckStore::open_in_memory().unwrap();

        store
            .insert_memory(&make_memory("m-1", "auth uses JWT tokens", 0.95, 10))
            .unwrap();
        store
            .insert_memory(&make_memory("m-2", "old forgotten note", 0.5, 0))
            .unwrap();

        let scores = compute_decay_scores(&store, Some("test")).unwrap();
        assert_eq!(scores.len(), 2);
        // m-1 should score higher (high confidence + high access count)
        assert!(
            scores[0].memory_id == "m-1",
            "high confidence+access should rank first"
        );
        assert!(scores[0].score > scores[1].score);
    }

    #[test]
    fn test_find_merge_candidates() {
        let store = DuckStore::open_in_memory().unwrap();

        store
            .insert_memory(&make_memory(
                "m-1",
                "authentication module uses JWT tokens for auth",
                0.9,
                1,
            ))
            .unwrap();
        store
            .insert_memory(&make_memory(
                "m-2",
                "authentication module uses JWT tokens for authorization",
                0.8,
                2,
            ))
            .unwrap();
        store
            .insert_memory(&make_memory(
                "m-3",
                "database uses PostgreSQL for storage",
                0.9,
                1,
            ))
            .unwrap();

        let candidates = find_merge_candidates(&store, Some("test"), 0.5).unwrap();
        // m-1 and m-2 should be candidates (high overlap), m-3 should not match either
        assert!(
            !candidates.is_empty(),
            "should find at least one merge candidate"
        );
        assert!(candidates[0].similarity > 0.5);
        // Should be m-1/m-2 pair
        let pair = &candidates[0];
        assert!(
            (pair.memory_a == "m-1" && pair.memory_b == "m-2")
                || (pair.memory_a == "m-2" && pair.memory_b == "m-1")
        );
    }

    #[test]
    fn test_expire_stale_memories() {
        let store = DuckStore::open_in_memory().unwrap();

        let mut expired = make_memory("m-expired", "old stuff", 0.5, 0);
        expired.valid_until = Some(Utc::now().naive_utc() - chrono::Duration::days(1));
        store.insert_memory(&expired).unwrap();

        let mut fresh = make_memory("m-fresh", "new stuff", 0.9, 5);
        fresh.valid_until = Some(Utc::now().naive_utc() + chrono::Duration::days(30));
        store.insert_memory(&fresh).unwrap();

        let count = expire_stale_memories(&store).unwrap();
        assert_eq!(count, 1, "should expire 1 stale memory");

        let remaining = store.get_memories(Some("test"), 100).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, "m-fresh");
    }

    #[test]
    fn test_consolidate_full() {
        let store = DuckStore::open_in_memory().unwrap();

        store
            .insert_memory(&make_memory("m-1", "auth uses JWT", 0.9, 5))
            .unwrap();
        store
            .insert_memory(&make_memory("m-2", "auth uses JWT tokens", 0.8, 2))
            .unwrap();

        let (stats, scores, candidates) = consolidate(&store, Some("test"), 0.5).unwrap();
        assert_eq!(stats.expired_count, 0);
        assert_eq!(stats.scored_count, 2);
        assert!(scores.len() == 2);
        assert!(!candidates.is_empty());
    }
}
