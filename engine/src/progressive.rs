//! Progressive disclosure: 3-layer retrieval that saves 10x tokens.
//!
//! Layer 1 (Index): Returns only titles/IDs — minimal tokens
//! Layer 2 (Summary): Returns summaries — moderate tokens
//! Layer 3 (Detail): Returns full content — maximum tokens
//!
//! Agents start with Layer 1, then drill down only into relevant results,
//! dramatically reducing context window usage.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::store::duckdb::DuckStore;

// ---------------------------------------------------------------------------
// Disclosure levels
// ---------------------------------------------------------------------------

/// Progressive disclosure depth level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosureLevel {
    /// Layer 1: IDs and titles only (~10 tokens per result).
    Index,
    /// Layer 2: Summaries with key metadata (~50 tokens per result).
    Summary,
    /// Layer 3: Full content with all metadata (~200+ tokens per result).
    Detail,
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Index-level result (Layer 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub id: String,
    pub title: String,
    pub entry_type: String,
    pub relevance_hint: Option<String>,
}

/// Summary-level result (Layer 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryEntry {
    pub id: String,
    pub title: String,
    pub entry_type: String,
    pub summary: String,
    pub agent: Option<String>,
    pub project: Option<String>,
    pub timestamp: Option<String>,
}

/// Detail-level result (Layer 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailEntry {
    pub id: String,
    pub title: String,
    pub entry_type: String,
    pub content: String,
    pub agent: Option<String>,
    pub project: Option<String>,
    pub timestamp: Option<String>,
    pub metadata: std::collections::HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// ProgressiveRetriever
// ---------------------------------------------------------------------------

/// Progressive disclosure retriever. Agents use this to efficiently explore
/// the knowledge base without loading full content upfront.
pub struct ProgressiveRetriever<'a> {
    duck: &'a DuckStore,
}

impl<'a> ProgressiveRetriever<'a> {
    pub fn new(duck: &'a DuckStore) -> Self {
        Self { duck }
    }

    /// Layer 1: Search and return index entries (titles + IDs only).
    /// Costs ~10 tokens per result. Use this for initial exploration.
    pub fn search_index(&self, query: &str, limit: usize) -> Result<Vec<IndexEntry>> {
        let mut entries = Vec::new();

        // Sessions
        let sessions = self.duck.search_sessions_by_summary(query)?;
        for s in sessions.iter().take(limit) {
            entries.push(IndexEntry {
                id: format!("session:{}", s.id),
                title: truncate(s.summary.as_deref().unwrap_or("(no summary)"), 80),
                entry_type: "session".to_string(),
                relevance_hint: Some(s.agent.clone()),
            });
        }

        // Memories
        let memories = self.duck.search_memories(query)?;
        for m in memories.iter().take(limit) {
            entries.push(IndexEntry {
                id: format!("memory:{}", m.id),
                title: truncate(&m.content, 80),
                entry_type: m.memory_type.as_deref().unwrap_or("memory").to_string(),
                relevance_hint: None,
            });
        }

        // Facts
        let facts = self.duck.search_facts(query)?;
        for f in facts.iter().take(limit) {
            if f.invalid_at.is_some() {
                continue;
            }
            entries.push(IndexEntry {
                id: format!("fact:{}", f.id),
                title: format!("{} {} {}", f.subject, f.predicate, f.object),
                entry_type: "fact".to_string(),
                relevance_hint: None,
            });
        }

        entries.truncate(limit);
        Ok(entries)
    }

    /// Layer 2: Get summary for a specific entry or list of IDs.
    /// Costs ~50 tokens per result. Use after Layer 1 to narrow down.
    pub fn get_summaries(&self, ids: &[&str]) -> Result<Vec<SummaryEntry>> {
        let mut entries = Vec::new();

        for id in ids {
            if let Some(entry) = self.get_summary(id)? {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Layer 3: Get full detail for a specific entry.
    /// Costs ~200+ tokens. Use only for results the agent needs to fully understand.
    pub fn get_detail(&self, id: &str) -> Result<Option<DetailEntry>> {
        let (kind, raw_id) = split_typed_id(id);

        match kind {
            "session" => {
                let sessions = self.duck.search_sessions(None, None, None, 10000)?;
                if let Some(s) = sessions.iter().find(|s| s.id == raw_id) {
                    let tool_calls = self.duck.get_tool_calls_for_session(&s.id)?;
                    let mut meta = std::collections::HashMap::new();
                    meta.insert("messages".to_string(), s.message_count.unwrap_or(0).to_string());
                    meta.insert("tool_calls".to_string(), s.tool_call_count.unwrap_or(0).to_string());
                    meta.insert("tokens".to_string(), s.total_tokens.unwrap_or(0).to_string());
                    meta.insert("files".to_string(), s.files_changed.join(", "));

                    let tool_summary: Vec<String> = tool_calls
                        .iter()
                        .take(20)
                        .map(|tc| {
                            format!(
                                "{}: {}",
                                tc.tool_name.as_deref().unwrap_or("?"),
                                truncate(tc.command.as_deref().unwrap_or(""), 60)
                            )
                        })
                        .collect();
                    meta.insert("tool_log".to_string(), tool_summary.join("\n"));

                    Ok(Some(DetailEntry {
                        id: id.to_string(),
                        title: s.summary.clone().unwrap_or_default(),
                        entry_type: "session".to_string(),
                        content: format!(
                            "Session {} ({} agent)\n\
                             Summary: {}\n\
                             Duration: {} min\n\
                             Files: {}\n\
                             Tool calls:\n{}",
                            s.id,
                            s.agent,
                            s.summary.as_deref().unwrap_or(""),
                            s.duration_minutes.unwrap_or(0),
                            s.files_changed.join(", "),
                            tool_summary.join("\n"),
                        ),
                        agent: Some(s.agent.clone()),
                        project: s.project_id.clone(),
                        timestamp: s.started_at.map(|t| t.to_string()),
                        metadata: meta,
                    }))
                } else {
                    Ok(None)
                }
            }
            "memory" => {
                let memories = self.duck.search_memories("")?;
                if let Some(m) = memories.iter().find(|m| m.id == raw_id) {
                    let mut meta = std::collections::HashMap::new();
                    meta.insert("confidence".to_string(), m.confidence.to_string());
                    meta.insert("access_count".to_string(), m.access_count.to_string());
                    if let Some(ref t) = m.memory_type {
                        meta.insert("type".to_string(), t.clone());
                    }

                    Ok(Some(DetailEntry {
                        id: id.to_string(),
                        title: truncate(&m.content, 80),
                        entry_type: m.memory_type.as_deref().unwrap_or("memory").to_string(),
                        content: m.content.clone(),
                        agent: None,
                        project: m.project_id.clone(),
                        timestamp: m.created_at.map(|t| t.to_string()),
                        metadata: meta,
                    }))
                } else {
                    Ok(None)
                }
            }
            "fact" => {
                // Search all facts (including history)
                let facts = self.duck.search_facts("")?;
                if let Some(f) = facts.iter().find(|f| f.id == raw_id) {
                    let mut meta = std::collections::HashMap::new();
                    meta.insert("subject".to_string(), f.subject.clone());
                    meta.insert("predicate".to_string(), f.predicate.clone());
                    meta.insert("object".to_string(), f.object.clone());
                    meta.insert("confidence".to_string(), f.confidence.to_string());
                    if let Some(ref a) = f.source_agent {
                        meta.insert("source_agent".to_string(), a.clone());
                    }
                    let status = if f.invalid_at.is_some() { "invalidated" } else { "active" };
                    meta.insert("status".to_string(), status.to_string());

                    // Get history for this subject
                    let history = self.duck.get_fact_history(&f.subject)?;
                    let history_str: Vec<String> = history
                        .iter()
                        .map(|h| {
                            let st = if h.invalid_at.is_some() { "X" } else { ">" };
                            format!("{st} {} {} {} ({})",
                                h.subject, h.predicate, h.object,
                                h.valid_at.map(|t| t.to_string()).unwrap_or_default(),
                            )
                        })
                        .collect();

                    Ok(Some(DetailEntry {
                        id: id.to_string(),
                        title: format!("{} {} {}", f.subject, f.predicate, f.object),
                        entry_type: "fact".to_string(),
                        content: format!(
                            "Fact: {} {} {}\nConfidence: {}\nStatus: {}\n\nHistory:\n{}",
                            f.subject, f.predicate, f.object,
                            f.confidence, status,
                            history_str.join("\n"),
                        ),
                        agent: f.source_agent.clone(),
                        project: f.project_id.clone(),
                        timestamp: f.valid_at.map(|t| t.to_string()),
                        metadata: meta,
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    // Internal helper for Layer 2
    fn get_summary(&self, id: &str) -> Result<Option<SummaryEntry>> {
        let (kind, raw_id) = split_typed_id(id);

        match kind {
            "session" => {
                let sessions = self.duck.search_sessions(None, None, None, 10000)?;
                if let Some(s) = sessions.iter().find(|s| s.id == raw_id) {
                    Ok(Some(SummaryEntry {
                        id: id.to_string(),
                        title: truncate(s.summary.as_deref().unwrap_or(""), 80),
                        entry_type: "session".to_string(),
                        summary: format!(
                            "{} | {} msgs, {} tools, {} files",
                            s.summary.as_deref().unwrap_or("(no summary)"),
                            s.message_count.unwrap_or(0),
                            s.tool_call_count.unwrap_or(0),
                            s.files_changed.len(),
                        ),
                        agent: Some(s.agent.clone()),
                        project: s.project_id.clone(),
                        timestamp: s.started_at.map(|t| t.to_string()),
                    }))
                } else {
                    Ok(None)
                }
            }
            "memory" => {
                let memories = self.duck.search_memories("")?;
                if let Some(m) = memories.iter().find(|m| m.id == raw_id) {
                    Ok(Some(SummaryEntry {
                        id: id.to_string(),
                        title: truncate(&m.content, 80),
                        entry_type: m.memory_type.as_deref().unwrap_or("memory").to_string(),
                        summary: truncate(&m.content, 200),
                        agent: None,
                        project: m.project_id.clone(),
                        timestamp: m.created_at.map(|t| t.to_string()),
                    }))
                } else {
                    Ok(None)
                }
            }
            "fact" => {
                let facts = self.duck.search_facts("")?;
                if let Some(f) = facts.iter().find(|f| f.id == raw_id) {
                    let status = if f.invalid_at.is_some() { " [invalidated]" } else { "" };
                    Ok(Some(SummaryEntry {
                        id: id.to_string(),
                        title: format!("{} {} {}{}", f.subject, f.predicate, f.object, status),
                        entry_type: "fact".to_string(),
                        summary: format!(
                            "{} {} {} (confidence: {}){}",
                            f.subject, f.predicate, f.object, f.confidence, status,
                        ),
                        agent: f.source_agent.clone(),
                        project: f.project_id.clone(),
                        timestamp: f.valid_at.map(|t| t.to_string()),
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

fn split_typed_id(id: &str) -> (&str, &str) {
    if let Some((kind, raw)) = id.split_once(':') {
        (kind, raw)
    } else {
        ("unknown", id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::duckdb::{Memory, Session};
    use chrono::Utc;

    fn setup_store() -> DuckStore {
        let store = DuckStore::open_in_memory().unwrap();

        let session = Session {
            id: "s-test".into(),
            project_id: Some("proj-1".into()),
            agent: "claude".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(15),
            message_count: Some(20),
            tool_call_count: Some(8),
            total_tokens: Some(5000),
            files_changed: vec!["src/auth.rs".into(), "src/main.rs".into()],
            summary: Some("Refactored authentication to use JWT".into()),
        };
        store.insert_or_replace_session(&session).unwrap();

        let memory = Memory {
            id: "m-test".into(),
            project_id: Some("proj-1".into()),
            content: "Authentication module uses JWT tokens with RS256 algorithm".into(),
            memory_type: Some("insight".into()),
            source_session_id: Some("s-test".into()),
            confidence: 0.95,
            access_count: 3,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&memory).unwrap();

        store
    }

    #[test]
    fn test_layer1_index_search() {
        let store = setup_store();
        let retriever = ProgressiveRetriever::new(&store);
        let results = retriever.search_index("auth", 10).unwrap();

        assert!(!results.is_empty());
        // Should find both session and memory
        assert!(results.iter().any(|r| r.entry_type == "session"));
        assert!(results.iter().any(|r| r.id.starts_with("memory:")));

        // Index entries should be concise
        for entry in &results {
            assert!(entry.title.len() <= 83); // 80 + "..."
        }
    }

    #[test]
    fn test_layer2_summaries() {
        let store = setup_store();
        let retriever = ProgressiveRetriever::new(&store);
        let summaries = retriever.get_summaries(&["session:s-test", "memory:m-test"]).unwrap();

        assert_eq!(summaries.len(), 2);

        let session_summary = summaries.iter().find(|s| s.entry_type == "session").unwrap();
        assert!(session_summary.summary.contains("20 msgs"));
        assert!(session_summary.summary.contains("8 tools"));

        let memory_summary = summaries.iter().find(|s| s.entry_type == "insight").unwrap();
        assert!(memory_summary.summary.contains("JWT"));
    }

    #[test]
    fn test_layer3_detail() {
        let store = setup_store();
        let retriever = ProgressiveRetriever::new(&store);

        let detail = retriever.get_detail("session:s-test").unwrap().unwrap();
        assert!(detail.content.contains("Refactored authentication"));
        assert!(detail.content.contains("src/auth.rs"));
        assert!(detail.metadata.contains_key("messages"));
        assert!(detail.metadata.contains_key("files"));

        let mem_detail = retriever.get_detail("memory:m-test").unwrap().unwrap();
        assert!(mem_detail.content.contains("RS256"));
    }

    #[test]
    fn test_progressive_token_savings() {
        let store = setup_store();
        let retriever = ProgressiveRetriever::new(&store);

        // Layer 1 tokens (index)
        let index = retriever.search_index("auth", 10).unwrap();
        let index_chars: usize = index.iter().map(|e| e.title.len() + e.entry_type.len()).sum();

        // Layer 2 tokens (summary)
        let ids: Vec<String> = index.iter().map(|e| e.id.clone()).collect();
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let summaries = retriever.get_summaries(&id_refs).unwrap();
        let summary_chars: usize = summaries.iter().map(|s| s.summary.len()).sum();

        // Layer 3 tokens (detail) - just one result
        let detail = retriever.get_detail(&index[0].id).unwrap().unwrap();
        let detail_chars = detail.content.len();

        // Index should be much smaller than detail
        assert!(
            index_chars < detail_chars,
            "index ({index_chars} chars) should be smaller than detail ({detail_chars} chars)"
        );
        assert!(
            summary_chars < detail_chars * 2,
            "summary ({summary_chars} chars) should be smaller than 2x detail"
        );
    }
}
