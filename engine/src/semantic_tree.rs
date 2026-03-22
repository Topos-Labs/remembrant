//! Semantic tree model for XPath-like queries over Remembrant's conversational data.
//!
//! Based on "Semantic XPath: Structured Agentic Memory Access for Conversational AI"
//! (arXiv:2603.01160). The tree is materialized lazily from DuckDB data on demand.
//!
//! # Hierarchy
//!
//! ```text
//! Root
//! ├── Project (from: sessions.project_id distinct)
//! │   ├── Session (from: sessions WHERE project_id = ?)
//! │   │   ├── Decision (from: decisions WHERE session_id = ?)
//! │   │   ├── Memory (from: memories WHERE source_session_id = ?)
//! │   │   ├── ToolCall (from: tool_calls WHERE session_id = ?)
//! │   │   └── CodeEntity (from: files_changed in session)
//! │   │       └── Symbol (from: code_symbols WHERE file_path = ? AND project_id = ?)
//! │   └── Memory (project-level memories without session)
//! ```

use anyhow::{Context, Result};
use duckdb::params;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::store::duckdb::DuckStore;

// ---------------------------------------------------------------------------
// TreeNodeType
// ---------------------------------------------------------------------------

/// Node types in the semantic tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TreeNodeType {
    Root,
    Project,
    Session,
    Decision,
    Memory,
    ToolCall,
    CodeEntity,
    Symbol,
}

impl TreeNodeType {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Root => "Root",
            Self::Project => "Project",
            Self::Session => "Session",
            Self::Decision => "Decision",
            Self::Memory => "Memory",
            Self::ToolCall => "ToolCall",
            Self::CodeEntity => "CodeEntity",
            Self::Symbol => "Symbol",
        }
    }

    /// Alias for `name()` (compatibility with xpath_query evaluator).
    pub fn as_str(&self) -> &'static str {
        self.name()
    }

    /// Parse from an exact (case-sensitive) string.
    pub fn from_str_exact(s: &str) -> Option<Self> {
        match s {
            "Root" => Some(Self::Root),
            "Project" => Some(Self::Project),
            "Session" => Some(Self::Session),
            "Decision" => Some(Self::Decision),
            "Memory" => Some(Self::Memory),
            "ToolCall" => Some(Self::ToolCall),
            "CodeEntity" => Some(Self::CodeEntity),
            "Symbol" => Some(Self::Symbol),
            _ => None,
        }
    }

    /// Parse from string (case-insensitive).
    pub fn from_str_ci(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "root" => Some(Self::Root),
            "project" => Some(Self::Project),
            "session" => Some(Self::Session),
            "decision" => Some(Self::Decision),
            "memory" => Some(Self::Memory),
            "toolcall" | "tool_call" => Some(Self::ToolCall),
            "codeentity" | "code_entity" => Some(Self::CodeEntity),
            "symbol" => Some(Self::Symbol),
            _ => None,
        }
    }

    /// What child types can this node have?
    pub fn child_types(&self) -> &[TreeNodeType] {
        match self {
            Self::Root => &[TreeNodeType::Project],
            Self::Project => &[TreeNodeType::Session, TreeNodeType::Memory],
            Self::Session => &[
                TreeNodeType::Decision,
                TreeNodeType::Memory,
                TreeNodeType::ToolCall,
                TreeNodeType::CodeEntity,
            ],
            Self::CodeEntity => &[TreeNodeType::Symbol],
            // Leaf nodes
            Self::Decision | Self::Memory | Self::ToolCall | Self::Symbol => &[],
        }
    }
}

impl std::fmt::Display for TreeNodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// TreeNode
// ---------------------------------------------------------------------------

/// A node in the semantic tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Unique ID (e.g., "project:infiniloom", "session:abc123").
    pub id: String,
    /// Node type.
    pub node_type: TreeNodeType,
    /// Display name / summary.
    pub name: String,
    /// Attributes as key-value pairs (for semantic matching and display).
    pub attributes: HashMap<String, String>,
    /// Text representation for semantic matching (concatenation of meaningful attributes).
    pub text_repr: String,
    /// Children (lazily populated).
    pub children: Vec<TreeNode>,
    /// Weight/score from query evaluation (default 1.0).
    pub weight: f64,
}

impl TreeNode {
    /// Create a new tree node with defaults.
    pub fn new(id: impl Into<String>, node_type: TreeNodeType, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            node_type,
            name: name.into(),
            attributes: HashMap::new(),
            text_repr: String::new(),
            children: Vec::new(),
            weight: 1.0,
        }
    }

    /// Set an attribute (builder pattern).
    pub fn with_attr(mut self, key: &str, value: impl Into<String>) -> Self {
        self.attributes.insert(key.to_string(), value.into());
        self
    }

    /// Build the text representation from attributes.
    ///
    /// Concatenates all non-empty attribute values separated by " | ".
    pub fn build_text_repr(&mut self) {
        let parts: Vec<&str> = self
            .attributes
            .values()
            .filter(|v| !v.is_empty())
            .map(|v| v.as_str())
            .collect();
        self.text_repr = if parts.is_empty() {
            self.name.clone()
        } else {
            parts.join(" | ")
        };
    }

    /// Check if children are loaded.
    ///
    /// Symbol nodes are always considered loaded (they are leaf nodes).
    pub fn is_loaded(&self) -> bool {
        !self.children.is_empty() || self.node_type == TreeNodeType::Symbol
    }

    /// Count all nodes in this subtree (including self).
    pub fn subtree_size(&self) -> usize {
        1 + self.children.iter().map(|c| c.subtree_size()).sum::<usize>()
    }

    /// Recursively find a node by ID within this subtree.
    pub fn find_by_id(&self, id: &str) -> Option<&TreeNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_by_id(id) {
                return Some(found);
            }
        }
        None
    }

    /// Recursively find a mutable node by ID within this subtree.
    pub fn find_by_id_mut(&mut self, id: &str) -> Option<&mut TreeNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_by_id_mut(id) {
                return Some(found);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// TreeSchema
// ---------------------------------------------------------------------------

/// Schema definition for the tree. Validates parent-child relationships.
pub struct TreeSchema {
    /// Maps parent type -> allowed child types.
    hierarchy: HashMap<TreeNodeType, Vec<TreeNodeType>>,
}

impl TreeSchema {
    /// Create the default Remembrant schema.
    pub fn default_schema() -> Self {
        let mut hierarchy = HashMap::new();
        hierarchy.insert(TreeNodeType::Root, vec![TreeNodeType::Project]);
        hierarchy.insert(
            TreeNodeType::Project,
            vec![TreeNodeType::Session, TreeNodeType::Memory],
        );
        hierarchy.insert(
            TreeNodeType::Session,
            vec![
                TreeNodeType::Decision,
                TreeNodeType::Memory,
                TreeNodeType::ToolCall,
                TreeNodeType::CodeEntity,
            ],
        );
        hierarchy.insert(TreeNodeType::CodeEntity, vec![TreeNodeType::Symbol]);
        // Leaf nodes have no children
        hierarchy.insert(TreeNodeType::Decision, vec![]);
        hierarchy.insert(TreeNodeType::Memory, vec![]);
        hierarchy.insert(TreeNodeType::ToolCall, vec![]);
        hierarchy.insert(TreeNodeType::Symbol, vec![]);

        Self { hierarchy }
    }

    /// Validate that a parent-child relationship is valid.
    pub fn is_valid_child(&self, parent: TreeNodeType, child: TreeNodeType) -> bool {
        self.hierarchy
            .get(&parent)
            .map(|children| children.contains(&child))
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// TreeBuilder
// ---------------------------------------------------------------------------

/// Builds semantic trees from DuckDB data with lazy loading.
pub struct TreeBuilder<'a> {
    store: &'a DuckStore,
    schema: TreeSchema,
}

impl<'a> TreeBuilder<'a> {
    /// Create a new tree builder backed by the given DuckStore.
    pub fn new(store: &'a DuckStore) -> Self {
        Self {
            store,
            schema: TreeSchema::default_schema(),
        }
    }

    /// Access the schema.
    pub fn schema(&self) -> &TreeSchema {
        &self.schema
    }

    /// Build the root node with project children (level 1 only).
    pub fn build_root(&self) -> Result<TreeNode> {
        let mut root = TreeNode::new("root", TreeNodeType::Root, "Remembrant");
        root.text_repr = "Remembrant Memory Store".to_string();

        let project_ids = self.store.get_project_ids()?;
        for pid in project_ids {
            let mut project = TreeNode::new(
                format!("project:{pid}"),
                TreeNodeType::Project,
                pid.clone(),
            )
            .with_attr("project_id", &pid);
            project.text_repr = pid;
            root.children.push(project);
        }

        Ok(root)
    }

    /// Load children for a given node (one level deep).
    pub fn load_children(&self, node: &mut TreeNode) -> Result<()> {
        // Skip if already loaded or is a leaf
        if node.is_loaded() {
            return Ok(());
        }

        match node.node_type {
            TreeNodeType::Root => self.load_root_children(node),
            TreeNodeType::Project => self.load_project_children(node),
            TreeNodeType::Session => self.load_session_children(node),
            TreeNodeType::CodeEntity => self.load_code_entity_children(node),
            // Leaf node types have no children to load
            TreeNodeType::Decision
            | TreeNodeType::Memory
            | TreeNodeType::ToolCall
            | TreeNodeType::Symbol => Ok(()),
        }
    }

    /// Load the full subtree for a node up to `max_depth` levels.
    pub fn load_subtree(&self, node: &mut TreeNode, max_depth: usize) -> Result<()> {
        if max_depth == 0 {
            return Ok(());
        }

        self.load_children(node)?;

        for child in &mut node.children {
            self.load_subtree(child, max_depth - 1)?;
        }

        Ok(())
    }

    /// Build a complete tree from root to specified depth.
    ///
    /// `max_depth` counts levels below root: 0 = root + projects only (from `build_root`),
    /// 1 = also load each project's children (sessions, memories), 2 = also load
    /// session children (decisions, tool calls, code entities), etc.
    pub fn build_tree(&self, max_depth: usize) -> Result<TreeNode> {
        let mut root = self.build_root()?;
        for child in &mut root.children {
            self.load_subtree(child, max_depth)?;
        }
        Ok(root)
    }

    /// Find a node by ID and return it with its subtree loaded to one level.
    ///
    /// Parses the node ID prefix to determine type (e.g., "project:", "session:")
    /// and queries accordingly.
    pub fn find_node(&self, node_id: &str) -> Result<Option<TreeNode>> {
        if node_id == "root" {
            let root = self.build_root()?;
            return Ok(Some(root));
        }

        let (prefix, id) = match node_id.split_once(':') {
            Some(pair) => pair,
            None => return Ok(None),
        };

        match prefix {
            "project" => {
                let project_ids = self.store.get_project_ids()?;
                if project_ids.contains(&id.to_string()) {
                    let mut node = TreeNode::new(
                        node_id.to_string(),
                        TreeNodeType::Project,
                        id.to_string(),
                    )
                    .with_attr("project_id", id);
                    node.text_repr = id.to_string();
                    self.load_children(&mut node)?;
                    Ok(Some(node))
                } else {
                    Ok(None)
                }
            }
            "session" => {
                let sessions = self.store.search_sessions(None, None, None, 100_000)?;
                if let Some(session) = sessions.into_iter().find(|s| s.id == id) {
                    let mut node = self.session_to_node(&session);
                    self.load_children(&mut node)?;
                    Ok(Some(node))
                } else {
                    Ok(None)
                }
            }
            "decision" => {
                let decisions = self.store.get_decisions(None, 100_000)?;
                if let Some(decision) = decisions.into_iter().find(|d| d.id == id) {
                    Ok(Some(self.decision_to_node(&decision)))
                } else {
                    Ok(None)
                }
            }
            "memory" => {
                let memories = self.store.get_memories(None, 100_000)?;
                if let Some(memory) = memories.into_iter().find(|m| m.id == id) {
                    Ok(Some(self.memory_to_node(&memory)))
                } else {
                    Ok(None)
                }
            }
            "toolcall" => {
                // ToolCalls don't have a global search, so we can't find by ID
                // without knowing the session. Return None.
                Ok(None)
            }
            "codeentity" => {
                // CodeEntity is a virtual node derived from session files_changed.
                // The rest of the id is "project_id:file_path".
                if let Some((proj, file_path)) = id.split_once(':') {
                    let mut node = TreeNode::new(
                        node_id.to_string(),
                        TreeNodeType::CodeEntity,
                        file_path.to_string(),
                    )
                    .with_attr("file_path", file_path)
                    .with_attr("project_id", proj);
                    node.text_repr = file_path.to_string();
                    self.load_children(&mut node)?;
                    Ok(Some(node))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    // -----------------------------------------------------------------------
    // Internal: load children by parent type
    // -----------------------------------------------------------------------

    fn load_root_children(&self, node: &mut TreeNode) -> Result<()> {
        let project_ids = self.store.get_project_ids()?;
        for pid in project_ids {
            let mut project = TreeNode::new(
                format!("project:{pid}"),
                TreeNodeType::Project,
                pid.clone(),
            )
            .with_attr("project_id", &pid);
            project.text_repr = pid;
            node.children.push(project);
        }
        Ok(())
    }

    fn load_project_children(&self, node: &mut TreeNode) -> Result<()> {
        let project_id = node
            .attributes
            .get("project_id")
            .cloned()
            .unwrap_or_default();

        if project_id.is_empty() {
            return Ok(());
        }

        // Sessions for this project
        let sessions =
            self.store
                .search_sessions(None, Some(&project_id), None, 10_000)?;
        for session in &sessions {
            let session_node = self.session_to_node(session);
            node.children.push(session_node);
        }

        // Project-level memories (no source_session_id)
        let all_memories = self.query_project_memories_without_session(&project_id)?;
        for memory in &all_memories {
            let memory_node = self.memory_to_node(memory);
            node.children.push(memory_node);
        }

        Ok(())
    }

    fn load_session_children(&self, node: &mut TreeNode) -> Result<()> {
        let session_id = node
            .attributes
            .get("session_id")
            .cloned()
            .unwrap_or_default();
        let project_id = node
            .attributes
            .get("project_id")
            .cloned()
            .unwrap_or_default();

        if session_id.is_empty() {
            return Ok(());
        }

        // Decisions for this session
        let decisions = self.query_decisions_for_session(&session_id)?;
        for decision in &decisions {
            node.children.push(self.decision_to_node(decision));
        }

        // Memories linked to this session
        let memories = self.query_memories_for_session(&session_id)?;
        for memory in &memories {
            node.children.push(self.memory_to_node(memory));
        }

        // Tool calls for this session
        let tool_calls = self.store.get_tool_calls_for_session(&session_id)?;
        for tc in &tool_calls {
            node.children.push(self.tool_call_to_node(tc));
        }

        // CodeEntity nodes from files_changed
        let files_json = node
            .attributes
            .get("files_changed")
            .cloned()
            .unwrap_or_else(|| "[]".to_string());
        let files: Vec<String> = serde_json::from_str(&files_json).unwrap_or_default();
        for file_path in &files {
            let ce_id = format!(
                "codeentity:{}:{}",
                if project_id.is_empty() {
                    "unknown"
                } else {
                    &project_id
                },
                file_path
            );
            let mut ce = TreeNode::new(ce_id, TreeNodeType::CodeEntity, file_path.clone())
                .with_attr("file_path", file_path)
                .with_attr("project_id", &project_id);
            ce.text_repr = file_path.clone();
            node.children.push(ce);
        }

        Ok(())
    }

    fn load_code_entity_children(&self, node: &mut TreeNode) -> Result<()> {
        let file_path = node
            .attributes
            .get("file_path")
            .cloned()
            .unwrap_or_default();
        let project_id = node
            .attributes
            .get("project_id")
            .cloned()
            .unwrap_or_default();

        if file_path.is_empty() || project_id.is_empty() {
            return Ok(());
        }

        let symbols = self.store.get_symbols_in_file(&file_path, &project_id)?;
        for sym in &symbols {
            let mut sym_node = TreeNode::new(
                format!("symbol:{}", sym.id),
                TreeNodeType::Symbol,
                sym.symbol_name.clone(),
            )
            .with_attr("symbol_name", &sym.symbol_name)
            .with_attr("symbol_kind", &sym.symbol_kind)
            .with_attr("file_path", &sym.file_path);

            if let Some(ref sig) = sym.signature {
                sym_node = sym_node.with_attr("signature", sig);
            }

            sym_node.text_repr = format!(
                "{} ({}) in {}",
                sym.symbol_name, sym.symbol_kind, sym.file_path
            );
            node.children.push(sym_node);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal: domain struct -> TreeNode conversions
    // -----------------------------------------------------------------------

    fn session_to_node(&self, session: &crate::store::duckdb::Session) -> TreeNode {
        let files_json = serde_json::to_string(&session.files_changed).unwrap_or_default();
        let mut node = TreeNode::new(
            format!("session:{}", session.id),
            TreeNodeType::Session,
            session
                .summary
                .clone()
                .unwrap_or_else(|| format!("Session {}", session.id)),
        )
        .with_attr("session_id", &session.id)
        .with_attr(
            "project_id",
            session.project_id.as_deref().unwrap_or(""),
        )
        .with_attr("agent", &session.agent)
        .with_attr("files_changed", &files_json)
        .with_attr(
            "summary",
            session.summary.as_deref().unwrap_or(""),
        );

        node.text_repr = format!(
            "{} | agent: {} | files: {}",
            session.summary.as_deref().unwrap_or("(no summary)"),
            session.agent,
            session.files_changed.join(", ")
        );

        node
    }

    fn decision_to_node(&self, decision: &crate::store::duckdb::Decision) -> TreeNode {
        let mut node = TreeNode::new(
            format!("decision:{}", decision.id),
            TreeNodeType::Decision,
            decision.what.clone(),
        )
        .with_attr("what", &decision.what)
        .with_attr("why", decision.why.as_deref().unwrap_or(""))
        .with_attr(
            "decision_type",
            decision.decision_type.as_deref().unwrap_or(""),
        );

        node.text_repr = format!(
            "{} | why: {}",
            decision.what,
            decision.why.as_deref().unwrap_or("(no reason)")
        );

        node
    }

    fn memory_to_node(&self, memory: &crate::store::duckdb::Memory) -> TreeNode {
        let mut node = TreeNode::new(
            format!("memory:{}", memory.id),
            TreeNodeType::Memory,
            memory.content.clone(),
        )
        .with_attr("content", &memory.content)
        .with_attr(
            "memory_type",
            memory.memory_type.as_deref().unwrap_or(""),
        );

        node.text_repr = format!(
            "{} | type: {}",
            memory.content,
            memory.memory_type.as_deref().unwrap_or("(untyped)")
        );

        node
    }

    fn tool_call_to_node(&self, tc: &crate::store::duckdb::ToolCall) -> TreeNode {
        let mut node = TreeNode::new(
            format!("toolcall:{}", tc.id),
            TreeNodeType::ToolCall,
            tc.tool_name
                .clone()
                .unwrap_or_else(|| "(unknown tool)".to_string()),
        )
        .with_attr(
            "tool_name",
            tc.tool_name.as_deref().unwrap_or(""),
        )
        .with_attr("command", tc.command.as_deref().unwrap_or(""));

        node.text_repr = format!(
            "{}: {}",
            tc.tool_name.as_deref().unwrap_or("(unknown)"),
            tc.command.as_deref().unwrap_or("(no command)")
        );

        node
    }

    // -----------------------------------------------------------------------
    // Internal: queries not directly available on DuckStore
    // -----------------------------------------------------------------------

    /// Query decisions belonging to a specific session.
    fn query_decisions_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::store::duckdb::Decision>> {
        let conn = self.store.connection();
        let conn = conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, project_id, decision_type, what,
                        why, alternatives, outcome, created_at, valid_until
                 FROM decisions
                 WHERE session_id = ?
                 ORDER BY created_at ASC NULLS LAST",
            )
            .context("failed to prepare query_decisions_for_session")?;

        let rows = stmt
            .query_map(params![session_id], |row| {
                let alts_str: String = row.get::<_, String>(6).unwrap_or_default();
                let alts: Vec<String> = serde_json::from_str(&alts_str).unwrap_or_default();
                Ok(crate::store::duckdb::Decision {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    project_id: row.get(2)?,
                    decision_type: row.get(3)?,
                    what: row.get(4)?,
                    why: row.get(5)?,
                    alternatives: alts,
                    outcome: row.get(7)?,
                    created_at: row.get(8)?,
                    valid_until: row.get(9)?,
                })
            })
            .context("failed to query decisions for session")?;

        let mut decisions = Vec::new();
        for row in rows {
            decisions.push(row.context("failed to read decision row")?);
        }
        Ok(decisions)
    }

    /// Query memories linked to a specific session (source_session_id = session_id).
    fn query_memories_for_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::store::duckdb::Memory>> {
        let conn = self.store.connection();
        let conn = conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, content, memory_type, source_session_id,
                        confidence, access_count, created_at, updated_at, valid_until
                 FROM memories
                 WHERE source_session_id = ?
                 ORDER BY created_at ASC NULLS LAST",
            )
            .context("failed to prepare query_memories_for_session")?;

        let rows = stmt
            .query_map(params![session_id], |row| {
                Ok(crate::store::duckdb::Memory {
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
            .context("failed to query memories for session")?;

        let mut memories = Vec::new();
        for row in rows {
            memories.push(row.context("failed to read memory row")?);
        }
        Ok(memories)
    }

    /// Query project-level memories (where source_session_id IS NULL).
    fn query_project_memories_without_session(
        &self,
        project_id: &str,
    ) -> Result<Vec<crate::store::duckdb::Memory>> {
        let conn = self.store.connection();
        let conn = conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT id, project_id, content, memory_type, source_session_id,
                        confidence, access_count, created_at, updated_at, valid_until
                 FROM memories
                 WHERE project_id = ? AND source_session_id IS NULL
                 ORDER BY created_at ASC NULLS LAST",
            )
            .context("failed to prepare query_project_memories_without_session")?;

        let rows = stmt
            .query_map(params![project_id], |row| {
                Ok(crate::store::duckdb::Memory {
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
            .context("failed to query project memories without session")?;

        let mut memories = Vec::new();
        for row in rows {
            memories.push(row.context("failed to read memory row")?);
        }
        Ok(memories)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // -----------------------------------------------------------------------
    // TreeNodeType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_node_type_name() {
        assert_eq!(TreeNodeType::Root.name(), "Root");
        assert_eq!(TreeNodeType::Project.name(), "Project");
        assert_eq!(TreeNodeType::Symbol.name(), "Symbol");
    }

    #[test]
    fn test_node_type_from_str_ci() {
        assert_eq!(
            TreeNodeType::from_str_ci("root"),
            Some(TreeNodeType::Root)
        );
        assert_eq!(
            TreeNodeType::from_str_ci("PROJECT"),
            Some(TreeNodeType::Project)
        );
        assert_eq!(
            TreeNodeType::from_str_ci("ToolCall"),
            Some(TreeNodeType::ToolCall)
        );
        assert_eq!(
            TreeNodeType::from_str_ci("tool_call"),
            Some(TreeNodeType::ToolCall)
        );
        assert_eq!(
            TreeNodeType::from_str_ci("code_entity"),
            Some(TreeNodeType::CodeEntity)
        );
        assert_eq!(TreeNodeType::from_str_ci("nonexistent"), None);
    }

    #[test]
    fn test_node_type_child_types() {
        let root_children = TreeNodeType::Root.child_types();
        assert_eq!(root_children, &[TreeNodeType::Project]);

        let session_children = TreeNodeType::Session.child_types();
        assert!(session_children.contains(&TreeNodeType::Decision));
        assert!(session_children.contains(&TreeNodeType::Memory));
        assert!(session_children.contains(&TreeNodeType::ToolCall));
        assert!(session_children.contains(&TreeNodeType::CodeEntity));

        // Leaf nodes
        assert!(TreeNodeType::Symbol.child_types().is_empty());
        assert!(TreeNodeType::Decision.child_types().is_empty());
    }

    // -----------------------------------------------------------------------
    // TreeNode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tree_node_new() {
        let node = TreeNode::new("test:1", TreeNodeType::Session, "Test Session");
        assert_eq!(node.id, "test:1");
        assert_eq!(node.node_type, TreeNodeType::Session);
        assert_eq!(node.name, "Test Session");
        assert!(node.children.is_empty());
        assert_eq!(node.weight, 1.0);
    }

    #[test]
    fn test_tree_node_with_attr() {
        let node = TreeNode::new("test:1", TreeNodeType::Session, "Test")
            .with_attr("agent", "claude")
            .with_attr("summary", "did things");
        assert_eq!(node.attributes.get("agent").unwrap(), "claude");
        assert_eq!(node.attributes.get("summary").unwrap(), "did things");
    }

    #[test]
    fn test_tree_node_build_text_repr() {
        let mut node = TreeNode::new("test:1", TreeNodeType::Session, "Test")
            .with_attr("summary", "refactored module")
            .with_attr("agent", "claude");
        node.build_text_repr();
        // text_repr should contain both attribute values
        assert!(node.text_repr.contains("refactored module"));
        assert!(node.text_repr.contains("claude"));
    }

    #[test]
    fn test_tree_node_build_text_repr_empty_attrs() {
        let mut node = TreeNode::new("test:1", TreeNodeType::Session, "Fallback Name");
        node.build_text_repr();
        assert_eq!(node.text_repr, "Fallback Name");
    }

    #[test]
    fn test_tree_node_is_loaded() {
        let mut node = TreeNode::new("test:1", TreeNodeType::Session, "Test");
        assert!(!node.is_loaded());

        node.children.push(TreeNode::new(
            "child:1",
            TreeNodeType::Decision,
            "Child",
        ));
        assert!(node.is_loaded());

        // Symbol is always loaded (leaf)
        let symbol = TreeNode::new("sym:1", TreeNodeType::Symbol, "func");
        assert!(symbol.is_loaded());
    }

    #[test]
    fn test_tree_node_subtree_size() {
        let mut root = TreeNode::new("root", TreeNodeType::Root, "Root");
        let mut project = TreeNode::new("p:1", TreeNodeType::Project, "P1");
        project.children.push(TreeNode::new(
            "s:1",
            TreeNodeType::Session,
            "S1",
        ));
        project.children.push(TreeNode::new(
            "s:2",
            TreeNodeType::Session,
            "S2",
        ));
        root.children.push(project);
        assert_eq!(root.subtree_size(), 4);
    }

    #[test]
    fn test_tree_node_find_by_id() {
        let mut root = TreeNode::new("root", TreeNodeType::Root, "Root");
        let mut project = TreeNode::new("p:1", TreeNodeType::Project, "P1");
        project.children.push(TreeNode::new(
            "s:1",
            TreeNodeType::Session,
            "S1",
        ));
        root.children.push(project);

        assert!(root.find_by_id("s:1").is_some());
        assert_eq!(root.find_by_id("s:1").unwrap().name, "S1");
        assert!(root.find_by_id("nonexistent").is_none());
    }

    // -----------------------------------------------------------------------
    // TreeSchema tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tree_schema_valid_children() {
        let schema = TreeSchema::default_schema();
        assert!(schema.is_valid_child(TreeNodeType::Root, TreeNodeType::Project));
        assert!(schema.is_valid_child(TreeNodeType::Project, TreeNodeType::Session));
        assert!(schema.is_valid_child(TreeNodeType::Project, TreeNodeType::Memory));
        assert!(schema.is_valid_child(TreeNodeType::Session, TreeNodeType::Decision));
        assert!(schema.is_valid_child(TreeNodeType::Session, TreeNodeType::ToolCall));
        assert!(schema.is_valid_child(TreeNodeType::Session, TreeNodeType::CodeEntity));
        assert!(schema.is_valid_child(TreeNodeType::CodeEntity, TreeNodeType::Symbol));

        // Invalid relationships
        assert!(!schema.is_valid_child(TreeNodeType::Root, TreeNodeType::Session));
        assert!(!schema.is_valid_child(TreeNodeType::Session, TreeNodeType::Project));
        assert!(!schema.is_valid_child(TreeNodeType::Symbol, TreeNodeType::Decision));
    }

    // -----------------------------------------------------------------------
    // TreeBuilder integration tests (require DuckDB in-memory)
    // -----------------------------------------------------------------------

    fn make_test_session(id: &str, project_id: &str) -> crate::store::duckdb::Session {
        crate::store::duckdb::Session {
            id: id.to_string(),
            project_id: Some(project_id.to_string()),
            agent: "claude".to_string(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(10),
            message_count: Some(5),
            tool_call_count: Some(3),
            total_tokens: Some(1200),
            files_changed: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            summary: Some("refactored module".to_string()),
        }
    }

    fn make_test_decision(
        id: &str,
        session_id: &str,
        project_id: &str,
    ) -> crate::store::duckdb::Decision {
        crate::store::duckdb::Decision {
            id: id.to_string(),
            session_id: Some(session_id.to_string()),
            project_id: Some(project_id.to_string()),
            decision_type: Some("architecture".to_string()),
            what: "use DuckDB for structured store".to_string(),
            why: Some("embedded, fast analytics".to_string()),
            alternatives: vec!["SQLite".to_string()],
            outcome: None,
            created_at: None,
            valid_until: None,
        }
    }

    fn make_test_memory(
        id: &str,
        project_id: &str,
        session_id: Option<&str>,
    ) -> crate::store::duckdb::Memory {
        crate::store::duckdb::Memory {
            id: id.to_string(),
            project_id: Some(project_id.to_string()),
            content: format!("Memory content for {id}"),
            memory_type: Some("insight".to_string()),
            source_session_id: session_id.map(|s| s.to_string()),
            confidence: 0.9,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        }
    }

    fn make_test_tool_call(id: &str, session_id: &str) -> crate::store::duckdb::ToolCall {
        crate::store::duckdb::ToolCall {
            id: id.to_string(),
            session_id: Some(session_id.to_string()),
            tool_name: Some("bash".to_string()),
            command: Some("cargo build".to_string()),
            success: Some(true),
            error_message: None,
            duration_ms: Some(1500),
            timestamp: Some(Utc::now().naive_utc()),
        }
    }

    fn make_test_symbol(
        id: &str,
        project_id: &str,
        file_path: &str,
    ) -> crate::store::duckdb::CodeSymbol {
        crate::store::duckdb::CodeSymbol {
            id: id.to_string(),
            project_id: project_id.to_string(),
            file_path: file_path.to_string(),
            symbol_name: format!("func_{id}"),
            symbol_kind: "function".to_string(),
            signature: Some("fn func() -> Result<()>".to_string()),
            docstring: None,
            start_line: 1,
            end_line: 10,
            visibility: Some("public".to_string()),
            parent_symbol: None,
            pagerank_score: 0.5,
            reference_count: 3,
            language: Some("rust".to_string()),
            content_hash: None,
            indexed_at: Some(Utc::now().naive_utc()),
        }
    }

    fn setup_test_store() -> DuckStore {
        let store = DuckStore::open_in_memory().expect("open in-memory DuckDB");

        // Insert test data
        store
            .insert_session(&make_test_session("s-1", "proj-alpha"))
            .unwrap();
        store
            .insert_session(&make_test_session("s-2", "proj-alpha"))
            .unwrap();
        store
            .insert_session(&make_test_session("s-3", "proj-beta"))
            .unwrap();

        store
            .insert_decision(&make_test_decision("d-1", "s-1", "proj-alpha"))
            .unwrap();
        store
            .insert_decision(&make_test_decision("d-2", "s-1", "proj-alpha"))
            .unwrap();

        store
            .insert_memory(&make_test_memory("m-1", "proj-alpha", Some("s-1")))
            .unwrap();
        store
            .insert_memory(&make_test_memory("m-2", "proj-alpha", None)) // project-level
            .unwrap();
        store
            .insert_memory(&make_test_memory("m-3", "proj-beta", Some("s-3")))
            .unwrap();

        store
            .insert_tool_call(&make_test_tool_call("tc-1", "s-1"))
            .unwrap();
        store
            .insert_tool_call(&make_test_tool_call("tc-2", "s-1"))
            .unwrap();

        store
            .insert_code_symbol(&make_test_symbol("sym-1", "proj-alpha", "src/main.rs"))
            .unwrap();
        store
            .insert_code_symbol(&make_test_symbol("sym-2", "proj-alpha", "src/main.rs"))
            .unwrap();

        store
    }

    #[test]
    fn test_build_root() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        let root = builder.build_root().unwrap();
        assert_eq!(root.node_type, TreeNodeType::Root);
        assert_eq!(root.children.len(), 2); // proj-alpha, proj-beta

        let project_names: Vec<&str> = root.children.iter().map(|c| c.name.as_str()).collect();
        assert!(project_names.contains(&"proj-alpha"));
        assert!(project_names.contains(&"proj-beta"));
    }

    #[test]
    fn test_load_project_children() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        let mut root = builder.build_root().unwrap();

        // Find proj-alpha and load its children
        let alpha = root
            .children
            .iter_mut()
            .find(|c| c.name == "proj-alpha")
            .unwrap();
        builder.load_children(alpha).unwrap();

        // Should have 2 sessions + 1 project-level memory
        let session_count = alpha
            .children
            .iter()
            .filter(|c| c.node_type == TreeNodeType::Session)
            .count();
        let memory_count = alpha
            .children
            .iter()
            .filter(|c| c.node_type == TreeNodeType::Memory)
            .count();

        assert_eq!(session_count, 2);
        assert_eq!(memory_count, 1); // m-2 is project-level
    }

    #[test]
    fn test_load_session_children() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        let mut root = builder.build_root().unwrap();
        let alpha = root
            .children
            .iter_mut()
            .find(|c| c.name == "proj-alpha")
            .unwrap();
        builder.load_children(alpha).unwrap();

        // Find session s-1 and load its children
        let s1 = alpha
            .children
            .iter_mut()
            .find(|c| c.id == "session:s-1")
            .unwrap();
        builder.load_children(s1).unwrap();

        let decision_count = s1
            .children
            .iter()
            .filter(|c| c.node_type == TreeNodeType::Decision)
            .count();
        let memory_count = s1
            .children
            .iter()
            .filter(|c| c.node_type == TreeNodeType::Memory)
            .count();
        let tc_count = s1
            .children
            .iter()
            .filter(|c| c.node_type == TreeNodeType::ToolCall)
            .count();
        let ce_count = s1
            .children
            .iter()
            .filter(|c| c.node_type == TreeNodeType::CodeEntity)
            .count();

        assert_eq!(decision_count, 2); // d-1, d-2
        assert_eq!(memory_count, 1); // m-1
        assert_eq!(tc_count, 2); // tc-1, tc-2
        assert_eq!(ce_count, 2); // src/main.rs, src/lib.rs
    }

    #[test]
    fn test_load_code_entity_children() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        // Build tree 3 levels deep to get to CodeEntity
        let mut tree = builder.build_tree(3).unwrap();

        // Navigate: root -> proj-alpha -> session:s-1 -> codeentity for src/main.rs
        let alpha = tree
            .children
            .iter_mut()
            .find(|c| c.name == "proj-alpha")
            .unwrap();
        let s1 = alpha
            .children
            .iter_mut()
            .find(|c| c.id == "session:s-1")
            .unwrap();
        let main_rs = s1
            .children
            .iter_mut()
            .find(|c| {
                c.node_type == TreeNodeType::CodeEntity
                    && c.attributes.get("file_path").map(|s| s.as_str()) == Some("src/main.rs")
            })
            .unwrap();

        // Load symbols for src/main.rs
        builder.load_children(main_rs).unwrap();

        assert_eq!(main_rs.children.len(), 2); // sym-1, sym-2
        assert!(main_rs
            .children
            .iter()
            .all(|c| c.node_type == TreeNodeType::Symbol));
    }

    #[test]
    fn test_build_tree_depth() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        // Depth 0: only root, no children loaded
        let tree = builder.build_tree(0).unwrap();
        assert_eq!(tree.node_type, TreeNodeType::Root);
        // build_root always populates projects, so children exist
        assert!(!tree.children.is_empty());

        // Depth 1: root -> projects -> (sessions + memories loaded)
        let tree = builder.build_tree(1).unwrap();
        let alpha = tree
            .children
            .iter()
            .find(|c| c.name == "proj-alpha")
            .unwrap();
        assert!(alpha.is_loaded());

        // Depth 2: sessions should have their children loaded
        let tree = builder.build_tree(2).unwrap();
        let alpha = tree
            .children
            .iter()
            .find(|c| c.name == "proj-alpha")
            .unwrap();
        let s1 = alpha
            .children
            .iter()
            .find(|c| c.id == "session:s-1")
            .unwrap();
        assert!(s1.is_loaded());
    }

    #[test]
    fn test_find_node_project() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        let node = builder.find_node("project:proj-alpha").unwrap();
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node_type, TreeNodeType::Project);
        assert!(node.is_loaded()); // children should be loaded
    }

    #[test]
    fn test_find_node_session() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        let node = builder.find_node("session:s-1").unwrap();
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.node_type, TreeNodeType::Session);
        assert!(node.is_loaded());
    }

    #[test]
    fn test_find_node_root() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        let node = builder.find_node("root").unwrap();
        assert!(node.is_some());
        assert_eq!(node.unwrap().node_type, TreeNodeType::Root);
    }

    #[test]
    fn test_find_node_nonexistent() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        assert!(builder.find_node("project:nonexistent").unwrap().is_none());
        assert!(builder.find_node("session:nonexistent").unwrap().is_none());
        assert!(builder.find_node("garbage").unwrap().is_none());
    }

    #[test]
    fn test_text_repr_session() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        let tree = builder.build_tree(1).unwrap();
        let alpha = tree
            .children
            .iter()
            .find(|c| c.name == "proj-alpha")
            .unwrap();
        let s1 = alpha
            .children
            .iter()
            .find(|c| c.id == "session:s-1")
            .unwrap();

        assert!(s1.text_repr.contains("refactored module"));
        assert!(s1.text_repr.contains("claude"));
        assert!(s1.text_repr.contains("src/main.rs"));
    }

    #[test]
    fn test_text_repr_decision() {
        let store = setup_test_store();
        let builder = TreeBuilder::new(&store);

        let tree = builder.build_tree(2).unwrap();
        let alpha = tree
            .children
            .iter()
            .find(|c| c.name == "proj-alpha")
            .unwrap();
        let s1 = alpha
            .children
            .iter()
            .find(|c| c.id == "session:s-1")
            .unwrap();
        let d1 = s1
            .children
            .iter()
            .find(|c| c.id == "decision:d-1")
            .unwrap();

        assert!(d1.text_repr.contains("use DuckDB"));
        assert!(d1.text_repr.contains("embedded, fast analytics"));
    }

    #[test]
    fn test_empty_store_builds_empty_tree() {
        let store = DuckStore::open_in_memory().unwrap();
        let builder = TreeBuilder::new(&store);

        let root = builder.build_root().unwrap();
        assert_eq!(root.children.len(), 0);
    }
}
