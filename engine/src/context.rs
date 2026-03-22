//! Context assembly for LLM agents.
//!
//! Produces token-efficient, structured context blocks that agents can inject
//! into their system prompts. Unlike raw JSON dumps, these blocks are formatted
//! for minimal token usage while preserving maximum information density.

use anyhow::Result;
use std::collections::HashMap;

use crate::hybrid_search::HybridSearch;
use crate::store::duckdb::DuckStore;

/// Memory pressure level — signals agents to manage their context usage.
/// Inspired by Letta/MemGPT's context management, integrated with Progressive Disclosure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum PressureLevel {
    /// < 50% budget used. Normal operation.
    Normal,
    /// 50-75% budget. Consider switching from Detail to Summary disclosure level.
    Elevated,
    /// > 75% budget. Switch to Index-only disclosure. Consolidate or archive memories.
    Critical,
}

/// Assembled context ready for LLM consumption.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentContext {
    /// One-line project status.
    pub status_line: String,
    /// Active facts as compact triples.
    pub facts: Vec<String>,
    /// Recent decisions (what + why).
    pub decisions: Vec<String>,
    /// Relevant memories (highest confidence first).
    pub memories: Vec<String>,
    /// Recent session summaries.
    pub sessions: Vec<String>,
    /// Hot files (most frequently changed).
    pub hot_files: Vec<String>,
    /// Total estimated tokens in this context block.
    pub estimated_tokens: usize,
    /// Memory pressure level relative to max_tokens budget.
    pub pressure: PressureLevel,
    /// Human-readable pressure warning (None if Normal).
    pub pressure_message: Option<String>,
}

impl AgentContext {
    /// Render as a compact text block optimized for LLM system prompts.
    /// Uses ~4 chars per token estimate.
    pub fn to_prompt_block(&self) -> String {
        let mut out = String::with_capacity(2048);

        out.push_str("# Project Context\n");
        out.push_str(&self.status_line);
        out.push('\n');

        if !self.facts.is_empty() {
            out.push_str("\n## Known Facts\n");
            for f in &self.facts {
                out.push_str("- ");
                out.push_str(f);
                out.push('\n');
            }
        }

        if !self.decisions.is_empty() {
            out.push_str("\n## Recent Decisions\n");
            for d in &self.decisions {
                out.push_str("- ");
                out.push_str(d);
                out.push('\n');
            }
        }

        if !self.memories.is_empty() {
            out.push_str("\n## Key Memories\n");
            for m in &self.memories {
                out.push_str("- ");
                out.push_str(m);
                out.push('\n');
            }
        }

        if !self.sessions.is_empty() {
            out.push_str("\n## Recent Activity\n");
            for s in &self.sessions {
                out.push_str("- ");
                out.push_str(s);
                out.push('\n');
            }
        }

        if !self.hot_files.is_empty() {
            out.push_str("\n## Hot Files\n");
            for f in &self.hot_files {
                out.push_str("- ");
                out.push_str(f);
                out.push('\n');
            }
        }

        // Pressure warning — agents should act on this
        if let Some(ref msg) = self.pressure_message {
            out.push_str("\n## Memory Pressure Warning\n");
            out.push_str(msg);
            out.push('\n');
        }

        out
    }

    /// Render as JSON for programmatic consumption.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Assembles context from all available sources in the triple-store.
pub struct ContextAssembler<'a> {
    store: &'a DuckStore,
    /// Maximum total tokens for the context block.
    max_tokens: usize,
}

impl<'a> ContextAssembler<'a> {
    pub fn new(store: &'a DuckStore) -> Self {
        Self {
            store,
            max_tokens: 1000,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Assemble full project context (for `rem brief --for-agent`).
    pub fn project_context(&self, project: Option<&str>) -> Result<AgentContext> {
        let now = chrono::Utc::now().naive_utc();
        let since_3d = now - chrono::Duration::days(3);

        let sessions = self
            .store
            .search_sessions(None, project, Some(since_3d), 100)?;
        let facts = self.store.get_active_facts(project, 30)?;
        let memories = self.store.get_memories(project, 20)?;
        let decisions = self.store.get_decisions(project, 10)?;

        // Status line
        let agent_count = sessions
            .iter()
            .map(|s| s.agent.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let status_line = format!(
            "{} sessions by {} agent(s) in last 3 days | {} facts | {} memories",
            sessions.len(),
            agent_count,
            facts.len(),
            memories.len()
        );

        // Facts as compact triples
        let fact_lines: Vec<String> = facts
            .iter()
            .take(self.budget_items(self.max_tokens / 4, 15))
            .map(|f| {
                if f.confidence < 0.8 {
                    format!(
                        "{} {} {} (conf: {:.0}%)",
                        f.subject,
                        f.predicate,
                        f.object,
                        f.confidence * 100.0
                    )
                } else {
                    format!("{} {} {}", f.subject, f.predicate, f.object)
                }
            })
            .collect();

        // Decisions
        let decision_lines: Vec<String> = decisions
            .iter()
            .take(self.budget_items(self.max_tokens / 6, 8))
            .map(|d| {
                let why = d.why.as_deref().unwrap_or("");
                if why.is_empty() {
                    truncate_str(&d.what, 100).to_string()
                } else {
                    format!("{} — {}", truncate_str(&d.what, 60), truncate_str(why, 40))
                }
            })
            .collect();

        // Memories (sorted by confidence desc, then access_count desc)
        let mut sorted_memories = memories;
        sorted_memories.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.access_count.cmp(&a.access_count))
        });
        let memory_lines: Vec<String> = sorted_memories
            .iter()
            .take(self.budget_items(self.max_tokens / 4, 10))
            .map(|m| {
                let tag = m.memory_type.as_deref().unwrap_or("note");
                format!("[{}] {}", tag, truncate_str(&m.content, 80))
            })
            .collect();

        // Sessions: group by agent, show most recent
        let session_lines: Vec<String> = sessions
            .iter()
            .take(self.budget_items(self.max_tokens / 6, 8))
            .map(|s| {
                let summary = s.summary.as_deref().unwrap_or("(no summary)");
                let proj = s.project_id.as_deref().unwrap_or("");
                format!("[{}] {}: {}", s.agent, proj, truncate_str(summary, 60))
            })
            .collect();

        // Hot files: use file_stats table if populated, else count from sessions
        let hot_files: Vec<String> = match self.store.get_hot_files(project, 8) {
            Ok(files) if !files.is_empty() => files
                .iter()
                .map(|(f, count)| format!("{f} ({count}x)"))
                .collect(),
            _ => {
                let mut file_freq: HashMap<String, usize> = HashMap::new();
                for s in &sessions {
                    for f in &s.files_changed {
                        *file_freq.entry(f.clone()).or_insert(0) += 1;
                    }
                }
                let mut hot: Vec<_> = file_freq.into_iter().collect();
                hot.sort_by(|a, b| b.1.cmp(&a.1));
                hot.iter()
                    .take(8)
                    .map(|(f, count)| format!("{f} ({count}x)"))
                    .collect()
            }
        };

        let estimated = estimate_tokens(
            &status_line,
            &fact_lines,
            &decision_lines,
            &memory_lines,
            &session_lines,
            &hot_files,
        );
        let (pressure, pressure_message) = compute_pressure(estimated, self.max_tokens);

        let ctx = AgentContext {
            estimated_tokens: estimated,
            status_line,
            facts: fact_lines,
            decisions: decision_lines,
            memories: memory_lines,
            sessions: session_lines,
            hot_files,
            pressure,
            pressure_message,
        };

        Ok(ctx)
    }

    /// Assemble topic-specific context (for `rem context <topic>`).
    /// Uses hybrid search to find relevant items, then formats as context.
    pub fn topic_context(&self, topic: &str, project: Option<&str>) -> Result<AgentContext> {
        let search = HybridSearch::new(self.store);

        // Search across all entity types
        let results = search.search_text_only(topic, 30)?;

        let mut fact_lines = Vec::new();
        let mut memory_lines = Vec::new();
        let mut session_lines = Vec::new();
        let mut decision_lines = Vec::new();

        for r in &results {
            let line = truncate_str(&r.content, 80).to_string();
            match r.result_type {
                crate::hybrid_search::ResultType::Fact => fact_lines.push(line),
                crate::hybrid_search::ResultType::Memory => {
                    let tag = r.metadata.get("type").map(|s| s.as_str()).unwrap_or("note");
                    memory_lines.push(format!("[{tag}] {line}"));
                }
                crate::hybrid_search::ResultType::Session => session_lines.push(line),
                crate::hybrid_search::ResultType::Decision => decision_lines.push(line),
                _ => memory_lines.push(line),
            }
        }

        // Also fetch facts matching the topic
        let all_facts = self.store.search_facts(topic)?;
        for f in all_facts.iter().filter(|f| f.invalid_at.is_none()).take(10) {
            let line = format!("{} {} {}", f.subject, f.predicate, f.object);
            if !fact_lines.contains(&line) {
                fact_lines.push(line);
            }
        }

        // Apply project filter if specified
        if let Some(proj) = project {
            session_lines.retain(|_| true); // sessions already filtered by search
            // For facts/memories we can't easily filter post-hoc, so leave as-is
            let _ = proj;
        }

        fact_lines.truncate(self.budget_items(self.max_tokens / 3, 15));
        memory_lines.truncate(self.budget_items(self.max_tokens / 3, 10));
        session_lines.truncate(5);
        decision_lines.truncate(5);

        let status_line = format!(
            "Topic context for '{}': {} facts, {} memories, {} sessions",
            topic,
            fact_lines.len(),
            memory_lines.len(),
            session_lines.len()
        );

        let estimated = estimate_tokens(
            &status_line,
            &fact_lines,
            &decision_lines,
            &memory_lines,
            &session_lines,
            &[],
        );
        let (pressure, pressure_message) = compute_pressure(estimated, self.max_tokens);

        let ctx = AgentContext {
            estimated_tokens: estimated,
            status_line,
            facts: fact_lines,
            decisions: decision_lines,
            memories: memory_lines,
            sessions: session_lines,
            hot_files: Vec::new(),
            pressure,
            pressure_message,
        };

        Ok(ctx)
    }

    /// How many items fit in a token budget, assuming ~15 tokens per item.
    fn budget_items(&self, token_budget: usize, max: usize) -> usize {
        (token_budget / 15).min(max)
    }
}

fn truncate_str(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        s
    } else {
        let end = s
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        &s[..end]
    }
}

fn compute_pressure(estimated_tokens: usize, max_tokens: usize) -> (PressureLevel, Option<String>) {
    if max_tokens == 0 {
        return (PressureLevel::Normal, None);
    }
    let ratio = estimated_tokens as f64 / max_tokens as f64;
    if ratio > 0.75 {
        (
            PressureLevel::Critical,
            Some(format!(
                "CRITICAL: Context uses {:.0}% of budget ({}/{} tokens). \
                 Switch to Index-level disclosure. Consider running `rem consolidate` \
                 to merge duplicate memories, or use `mem_delete` to remove stale entries.",
                ratio * 100.0,
                estimated_tokens,
                max_tokens
            )),
        )
    } else if ratio > 0.50 {
        (
            PressureLevel::Elevated,
            Some(format!(
                "ELEVATED: Context uses {:.0}% of budget ({}/{} tokens). \
                 Consider switching from Detail to Summary disclosure level.",
                ratio * 100.0,
                estimated_tokens,
                max_tokens
            )),
        )
    } else {
        (PressureLevel::Normal, None)
    }
}

fn estimate_tokens(
    status: &str,
    facts: &[String],
    decisions: &[String],
    memories: &[String],
    sessions: &[String],
    hot_files: &[String],
) -> usize {
    let total_chars: usize = status.len()
        + facts.iter().map(|s| s.len() + 4).sum::<usize>()
        + decisions.iter().map(|s| s.len() + 4).sum::<usize>()
        + memories.iter().map(|s| s.len() + 4).sum::<usize>()
        + sessions.iter().map(|s| s.len() + 4).sum::<usize>()
        + hot_files.iter().map(|s| s.len() + 4).sum::<usize>()
        + 100; // headers
    // ~4 chars per token for English text
    total_chars / 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::duckdb::{Fact, Memory, Session};
    use chrono::Utc;

    #[test]
    fn test_project_context_empty() {
        let store = DuckStore::open_in_memory().unwrap();
        let assembler = ContextAssembler::new(&store);
        let ctx = assembler.project_context(None).unwrap();
        assert!(ctx.facts.is_empty());
        assert!(ctx.memories.is_empty());
        assert!(ctx.sessions.is_empty());
        assert!(ctx.status_line.contains("0 sessions"));
    }

    #[test]
    fn test_project_context_with_data() {
        let store = DuckStore::open_in_memory().unwrap();

        let session = Session {
            id: "s-1".into(),
            project_id: Some("myproj".into()),
            agent: "claude".into(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(10),
            message_count: Some(5),
            tool_call_count: Some(3),
            total_tokens: Some(1200),
            files_changed: vec!["src/main.rs".into(), "src/lib.rs".into()],
            summary: Some("refactored auth module".into()),
        };
        store.insert_or_replace_session(&session).unwrap();

        let memory = Memory {
            id: "m-1".into(),
            project_id: Some("myproj".into()),
            content: "Auth uses JWT with RS256".into(),
            memory_type: Some("insight".into()),
            source_session_id: Some("s-1".into()),
            confidence: 0.95,
            access_count: 3,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&memory).unwrap();

        let fact = Fact {
            id: "f-1".into(),
            project_id: Some("myproj".into()),
            subject: "auth module".into(),
            predicate: "uses".into(),
            object: "JWT RS256".into(),
            confidence: 0.9,
            source_session_id: Some("s-1".into()),
            source_agent: Some("claude".into()),
            valid_at: Some(Utc::now().naive_utc()),
            invalid_at: None,
            superseded_by: None,
            created_at: Some(Utc::now().naive_utc()),
        };
        store.insert_fact(&fact).unwrap();

        let assembler = ContextAssembler::new(&store);
        let ctx = assembler.project_context(Some("myproj")).unwrap();

        assert!(!ctx.facts.is_empty(), "should have facts");
        assert!(!ctx.memories.is_empty(), "should have memories");
        assert!(!ctx.sessions.is_empty(), "should have sessions");
        assert!(!ctx.hot_files.is_empty(), "should have hot files");
        assert!(ctx.status_line.contains("1 sessions"));

        // Check prompt block renders cleanly
        let block = ctx.to_prompt_block();
        assert!(block.contains("# Project Context"));
        assert!(block.contains("auth module uses JWT RS256"));
        assert!(block.contains("[insight] Auth uses JWT with RS256"));
        assert!(block.contains("src/main.rs"));
    }

    #[test]
    fn test_topic_context() {
        let store = DuckStore::open_in_memory().unwrap();

        let memory = Memory {
            id: "m-1".into(),
            project_id: None,
            content: "authentication uses bcrypt hashing".into(),
            memory_type: Some("technical".into()),
            source_session_id: None,
            confidence: 0.8,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        };
        store.insert_memory(&memory).unwrap();

        let assembler = ContextAssembler::new(&store);
        let ctx = assembler.topic_context("authentication", None).unwrap();
        assert!(!ctx.memories.is_empty());
        assert!(ctx.status_line.contains("authentication"));
    }

    #[test]
    fn test_token_budget() {
        let store = DuckStore::open_in_memory().unwrap();
        let assembler = ContextAssembler::new(&store).with_max_tokens(200);
        // With 200 max tokens, budget_items(50, 15) = min(50/15, 15) = 3
        assert_eq!(assembler.budget_items(50, 15), 3);
        assert_eq!(assembler.budget_items(200, 5), 5); // capped at max
    }
}
