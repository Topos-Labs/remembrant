use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Result;
use tracing::{debug, info, warn};

use crate::store::duckdb::{Decision, DuckStore, Memory, Session, ToolCall};
use crate::store::graph::*;

// ---------------------------------------------------------------------------
// GraphBackend trait — abstracts in-memory vs persistent graph storage
// ---------------------------------------------------------------------------

/// Information about a neighboring node returned from a graph traversal.
#[derive(Debug, Clone)]
pub struct NeighborInfo {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub properties: String,
    pub edge_kind: String,
    pub direction: String,
}

/// Trait abstracting the operations GraphBuilder needs from a graph backend.
///
/// All identifiers are strings:
/// - `kind` for nodes maps to `NodeKind::table_name()` (e.g. "CodeEntity", "Memory")
/// - `kind` for edges maps to `EdgeKind::name()` (e.g. "CALLS", "ABOUT")
/// - `properties` is a JSON-serialized string
/// - `direction` is "outgoing" or "incoming"
pub trait GraphBackend {
    fn add_node(&self, id: &str, kind: &str, name: &str, properties: &str) -> Result<()>;
    fn add_edge(&self, from_id: &str, to_id: &str, kind: &str, properties: &str) -> Result<()>;
    fn get_node(&self, id: &str) -> Result<Option<(String, String, String, String)>>; // (id, kind, name, props)
    fn delete_node(&self, id: &str) -> Result<bool>;
    fn query_neighbors(&self, id: &str, edge_kind: Option<&str>) -> Result<Vec<NeighborInfo>>;
    fn node_count(&self) -> Result<usize>;
    fn edge_count(&self) -> Result<usize>;
}

// ---------------------------------------------------------------------------
// GraphBackend implementation for GraphStore (in-memory)
// ---------------------------------------------------------------------------

fn props_to_json(props: &HashMap<String, String>) -> String {
    serde_json::to_string(props).unwrap_or_else(|_| "{}".to_string())
}

fn json_to_props(json: &str) -> HashMap<String, String> {
    serde_json::from_str(json).unwrap_or_default()
}

fn node_kind_from_str(s: &str) -> NodeKind {
    match s {
        "CodeEntity" => NodeKind::CodeEntity,
        "Concept" => NodeKind::Concept,
        "Memory" => NodeKind::Memory,
        "Pattern" => NodeKind::Pattern,
        "Problem" => NodeKind::Problem,
        "Solution" => NodeKind::Solution,
        "Symbol" => NodeKind::Symbol,
        "Module" => NodeKind::Module,
        _ => NodeKind::Concept, // fallback
    }
}

fn edge_kind_from_str(s: &str) -> EdgeKind {
    match s {
        "CALLS" => EdgeKind::Calls,
        "IMPORTS" => EdgeKind::Imports,
        "INHERITS" => EdgeKind::Inherits,
        "IMPLEMENTS" => EdgeKind::Implements,
        "USED_IN" => EdgeKind::UsedIn,
        "SOLVES" => EdgeKind::Solves,
        "SOLVED_BY" => EdgeKind::SolvedBy,
        "RELATES_TO" => EdgeKind::RelatesTo,
        "DERIVED_FROM" => EdgeKind::DerivedFrom,
        "ABOUT" => EdgeKind::About,
        "SUPERSEDES" => EdgeKind::Supersedes,
        "DEFINES" => EdgeKind::Defines,
        "CONTAINED_IN" => EdgeKind::ContainedIn,
        "DEPENDS_ON" => EdgeKind::DependsOn,
        "REFERENCES" => EdgeKind::References,
        _ => EdgeKind::RelatesTo, // fallback
    }
}

impl GraphBackend for GraphStore {
    fn add_node(&self, id: &str, kind: &str, name: &str, properties: &str) -> Result<()> {
        let node = GraphNode {
            id: id.to_string(),
            kind: node_kind_from_str(kind),
            name: name.to_string(),
            properties: json_to_props(properties),
        };
        GraphStoreBackend::add_node(self, &node)
    }

    fn add_edge(&self, from_id: &str, to_id: &str, kind: &str, properties: &str) -> Result<()> {
        let edge = GraphEdge {
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
            kind: edge_kind_from_str(kind),
            properties: json_to_props(properties),
        };
        GraphStoreBackend::add_edge(self, &edge)
    }

    fn get_node(&self, id: &str) -> Result<Option<(String, String, String, String)>> {
        let node = GraphStoreBackend::get_node(self, id)?;
        Ok(node.map(|n| {
            (
                n.id,
                n.kind.table_name().to_string(),
                n.name,
                props_to_json(&n.properties),
            )
        }))
    }

    fn delete_node(&self, id: &str) -> Result<bool> {
        GraphStoreBackend::delete_node(self, id)
    }

    fn query_neighbors(&self, id: &str, edge_kind: Option<&str>) -> Result<Vec<NeighborInfo>> {
        let ek = edge_kind.map(edge_kind_from_str);
        let neighbors = GraphStoreBackend::query_neighbors(self, id, ek)?;
        Ok(neighbors
            .into_iter()
            .map(|n| NeighborInfo {
                id: n.node.id,
                kind: n.node.kind.table_name().to_string(),
                name: n.node.name,
                properties: props_to_json(&n.node.properties),
                edge_kind: n.edge_kind.name().to_string(),
                direction: match n.direction {
                    Direction::Outgoing => "outgoing".to_string(),
                    Direction::Incoming => "incoming".to_string(),
                },
            })
            .collect())
    }

    fn node_count(&self) -> Result<usize> {
        GraphStoreBackend::node_count(self)
    }

    fn edge_count(&self) -> Result<usize> {
        GraphStoreBackend::edge_count(self)
    }
}

// ---------------------------------------------------------------------------
// GraphBackend implementation for DuckStore (persistent)
// ---------------------------------------------------------------------------

impl GraphBackend for DuckStore {
    fn add_node(&self, id: &str, kind: &str, name: &str, properties: &str) -> Result<()> {
        self.insert_graph_node(id, kind, name, properties)
    }

    fn add_edge(&self, from_id: &str, to_id: &str, kind: &str, properties: &str) -> Result<()> {
        self.insert_graph_edge(from_id, to_id, kind, properties)
    }

    fn get_node(&self, id: &str) -> Result<Option<(String, String, String, String)>> {
        let row = self.get_graph_node(id)?;
        Ok(row.map(|r| (r.id, r.kind, r.name, r.properties)))
    }

    fn delete_node(&self, id: &str) -> Result<bool> {
        self.delete_graph_node(id)
    }

    fn query_neighbors(&self, id: &str, edge_kind: Option<&str>) -> Result<Vec<NeighborInfo>> {
        let rows = self.query_graph_neighbors(id, edge_kind)?;
        Ok(rows
            .into_iter()
            .map(|r| NeighborInfo {
                id: r.node.id,
                kind: r.node.kind,
                name: r.node.name,
                properties: r.node.properties,
                edge_kind: r.edge_kind,
                direction: r.direction,
            })
            .collect())
    }

    fn node_count(&self) -> Result<usize> {
        self.count_graph_nodes()
    }

    fn edge_count(&self) -> Result<usize> {
        self.count_graph_edges()
    }
}

impl GraphBackend for &DuckStore {
    fn add_node(&self, id: &str, kind: &str, name: &str, properties: &str) -> Result<()> {
        (*self).insert_graph_node(id, kind, name, properties)
    }
    fn add_edge(&self, from_id: &str, to_id: &str, kind: &str, properties: &str) -> Result<()> {
        (*self).insert_graph_edge(from_id, to_id, kind, properties)
    }
    fn get_node(&self, id: &str) -> Result<Option<(String, String, String, String)>> {
        let row = (*self).get_graph_node(id)?;
        Ok(row.map(|r| (r.id, r.kind, r.name, r.properties)))
    }
    fn delete_node(&self, id: &str) -> Result<bool> {
        (*self).delete_graph_node(id)
    }
    fn query_neighbors(&self, id: &str, edge_kind: Option<&str>) -> Result<Vec<NeighborInfo>> {
        let rows = (*self).query_graph_neighbors(id, edge_kind)?;
        Ok(rows
            .into_iter()
            .map(|r| NeighborInfo {
                id: r.node.id,
                kind: r.node.kind,
                name: r.node.name,
                properties: r.node.properties,
                edge_kind: r.edge_kind,
                direction: r.direction,
            })
            .collect())
    }
    fn node_count(&self) -> Result<usize> {
        (*self).count_graph_nodes()
    }
    fn edge_count(&self) -> Result<usize> {
        (*self).count_graph_edges()
    }
}

// ---------------------------------------------------------------------------
// Neighbor with depth (for BFS traversal results)
// ---------------------------------------------------------------------------

/// A neighbor at a specific depth in a BFS traversal.
#[derive(Debug, Clone)]
pub struct DepthNeighbor {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub properties: String,
    pub edge_kind: String,
    pub direction: String,
    pub depth: usize,
    /// The node ID through which this neighbor was reached (for 2+ hop).
    pub via_node_id: Option<String>,
}

// ---------------------------------------------------------------------------
// GraphBuilder
// ---------------------------------------------------------------------------

/// Builds and queries a knowledge graph from ingested data.
///
/// Generic over backend: use `GraphStore` for in-memory (tests/backward compat)
/// or `DuckStore` for persistent storage.
pub struct GraphBuilder<B: GraphBackend> {
    backend: B,
}

/// Convenience type alias for the in-memory variant.
pub type InMemoryGraphBuilder = GraphBuilder<GraphStore>;

impl GraphBuilder<GraphStore> {
    /// Create a new GraphBuilder with the default in-memory GraphStore backend.
    pub fn new() -> Self {
        Self {
            backend: GraphStore::new(),
        }
    }

    /// Access the underlying GraphStore (only available for in-memory variant).
    pub fn store(&self) -> &GraphStore {
        &self.backend
    }
}

impl Default for GraphBuilder<GraphStore> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: GraphBackend> GraphBuilder<B> {
    /// Create a GraphBuilder with a specific backend.
    pub fn with_backend(backend: B) -> Self {
        Self { backend }
    }

    /// Access the underlying backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn node_count(&self) -> usize {
        self.backend.node_count().unwrap_or(0)
    }

    pub fn edge_count(&self) -> usize {
        self.backend.edge_count().unwrap_or(0)
    }

    /// Build graph from all ingested data.
    pub fn build_from_data(
        &self,
        sessions: &[Session],
        memories: &[Memory],
        decisions: &[Decision],
        tool_calls: &[ToolCall],
    ) -> Result<()> {
        info!(
            "Building graph from {} sessions, {} memories, {} decisions, {} tool_calls",
            sessions.len(),
            memories.len(),
            decisions.len(),
            tool_calls.len(),
        );

        for session in sessions {
            if let Err(e) = self.add_session(session) {
                warn!("Failed to add session {}: {e}", session.id);
            }
        }

        for memory in memories {
            if let Err(e) = self.add_memory(memory) {
                warn!("Failed to add memory {}: {e}", memory.id);
            }
        }

        for decision in decisions {
            if let Err(e) = self.add_decision(decision) {
                warn!("Failed to add decision {}: {e}", decision.id);
            }
        }

        // Extract code entities and cross-entity relationships
        self.extract_code_entities(sessions)?;
        self.link_shared_files(sessions)?;
        self.link_memories_to_code_entities(memories)?;

        let (nodes, edges) = self.stats()?;
        info!("Graph built: {nodes} nodes, {edges} edges");

        Ok(())
    }

    /// Add a session as a node with edges to its project.
    pub fn add_session(&self, session: &Session) -> Result<()> {
        let fallback = format!("Session {}", &session.id);
        let name = session.summary.as_deref().unwrap_or(&fallback);
        let name = truncate_str(name, 80);

        let mut props = HashMap::new();
        props.insert("agent".to_string(), session.agent.clone());
        if let Some(ref pid) = session.project_id {
            props.insert("project_id".to_string(), pid.clone());
        }
        if let Some(ref dt) = session.started_at {
            props.insert(
                "started_at".to_string(),
                dt.format("%Y-%m-%d %H:%M").to_string(),
            );
        }
        if let Some(msgs) = session.message_count {
            props.insert("message_count".to_string(), msgs.to_string());
        }

        let node_id = format!("session:{}", session.id);
        self.backend.add_node(
            &node_id,
            NodeKind::Memory.table_name(),
            &name,
            &props_to_json(&props),
        )?;

        // Create project concept node and link
        if let Some(ref project_id) = session.project_id {
            let proj_node_id = format!("project:{project_id}");
            self.ensure_project_node(&proj_node_id, project_id)?;
            self.backend
                .add_edge(&node_id, &proj_node_id, EdgeKind::About.name(), "{}")?;
        }

        debug!("Added session node: {}", node_id);
        Ok(())
    }

    /// Add a memory as a node, link to source session.
    pub fn add_memory(&self, memory: &Memory) -> Result<()> {
        let name = truncate_str(&memory.content, 80);

        let mut props = HashMap::new();
        if let Some(ref mtype) = memory.memory_type {
            props.insert("memory_type".to_string(), mtype.clone());
        }
        if let Some(ref pid) = memory.project_id {
            props.insert("project_id".to_string(), pid.clone());
        }
        if let Some(ref dt) = memory.created_at {
            props.insert(
                "created_at".to_string(),
                dt.format("%Y-%m-%d %H:%M").to_string(),
            );
        }
        props.insert(
            "confidence".to_string(),
            format!("{:.2}", memory.confidence),
        );

        let node_id = format!("memory:{}", memory.id);
        self.backend.add_node(
            &node_id,
            NodeKind::Memory.table_name(),
            &name,
            &props_to_json(&props),
        )?;

        // Link to source session
        if let Some(ref session_id) = memory.source_session_id {
            self.backend.add_edge(
                &node_id,
                &format!("session:{session_id}"),
                EdgeKind::DerivedFrom.name(),
                "{}",
            )?;
        }

        // Link to project
        if let Some(ref project_id) = memory.project_id {
            let proj_node_id = format!("project:{project_id}");
            self.ensure_project_node(&proj_node_id, project_id)?;
            self.backend
                .add_edge(&node_id, &proj_node_id, EdgeKind::About.name(), "{}")?;
        }

        debug!("Added memory node: {}", node_id);
        Ok(())
    }

    /// Add a decision as a node, link to session.
    pub fn add_decision(&self, decision: &Decision) -> Result<()> {
        let mut props = HashMap::new();
        if let Some(ref dtype) = decision.decision_type {
            props.insert("decision_type".to_string(), dtype.clone());
        }
        if let Some(ref why) = decision.why {
            props.insert("why".to_string(), truncate_str(why, 120));
        }
        if let Some(ref outcome) = decision.outcome {
            props.insert("outcome".to_string(), outcome.clone());
        }
        if let Some(ref dt) = decision.created_at {
            props.insert(
                "created_at".to_string(),
                dt.format("%Y-%m-%d %H:%M").to_string(),
            );
        }

        let node_id = format!("decision:{}", decision.id);
        self.backend.add_node(
            &node_id,
            NodeKind::Concept.table_name(),
            &truncate_str(&decision.what, 80),
            &props_to_json(&props),
        )?;

        // Link to session
        if let Some(ref session_id) = decision.session_id {
            self.backend.add_edge(
                &node_id,
                &format!("session:{session_id}"),
                EdgeKind::DerivedFrom.name(),
                "{}",
            )?;
        }

        // Link to project
        if let Some(ref project_id) = decision.project_id {
            let proj_node_id = format!("project:{project_id}");
            self.ensure_project_node(&proj_node_id, project_id)?;
            self.backend
                .add_edge(&node_id, &proj_node_id, EdgeKind::About.name(), "{}")?;
        }

        debug!("Added decision node: {}", node_id);
        Ok(())
    }

    /// Extract code entities from file paths in sessions and create nodes + edges.
    pub fn extract_code_entities(&self, sessions: &[Session]) -> Result<()> {
        for session in sessions {
            let session_node_id = format!("session:{}", session.id);
            for file_path in &session.files_changed {
                let file_node_id = format!("file:{file_path}");

                // Create the code entity node (idempotent via upsert)
                let file_name = file_path
                    .rsplit('/')
                    .next()
                    .unwrap_or(file_path)
                    .to_string();

                let mut props = HashMap::new();
                props.insert("path".to_string(), file_path.clone());

                self.backend.add_node(
                    &file_node_id,
                    NodeKind::CodeEntity.table_name(),
                    &file_name,
                    &props_to_json(&props),
                )?;

                // Edge: code entity -> used in session
                self.backend.add_edge(
                    &file_node_id,
                    &session_node_id,
                    EdgeKind::UsedIn.name(),
                    "{}",
                )?;
            }
        }
        Ok(())
    }

    /// If two sessions share the same file, add RelatesTo edge between them.
    fn link_shared_files(&self, sessions: &[Session]) -> Result<()> {
        // Build file -> session IDs map
        let mut file_sessions: HashMap<&str, Vec<&str>> = HashMap::new();
        for session in sessions {
            for file_path in &session.files_changed {
                file_sessions
                    .entry(file_path.as_str())
                    .or_default()
                    .push(&session.id);
            }
        }

        // For each file with multiple sessions, link the sessions pairwise
        let mut linked: HashSet<(String, String)> = HashSet::new();
        for session_ids in file_sessions.values() {
            if session_ids.len() < 2 {
                continue;
            }
            for i in 0..session_ids.len() {
                for j in (i + 1)..session_ids.len() {
                    let a = format!("session:{}", session_ids[i]);
                    let b = format!("session:{}", session_ids[j]);
                    let key = if a < b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    if linked.insert(key) {
                        self.backend
                            .add_edge(&a, &b, EdgeKind::RelatesTo.name(), "{}")?;
                    }
                }
            }
        }
        Ok(())
    }

    /// If a memory mentions a file path that exists as a CodeEntity, add About edge.
    fn link_memories_to_code_entities(&self, memories: &[Memory]) -> Result<()> {
        for memory in memories {
            let memory_node_id = format!("memory:{}", memory.id);
            // Look for file-like patterns in content (e.g., paths with extensions)
            for word in memory.content.split_whitespace() {
                let cleaned = word.trim_matches(|c: char| {
                    !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-'
                });
                if looks_like_file_path(cleaned) {
                    let file_node_id = format!("file:{cleaned}");
                    if self.backend.get_node(&file_node_id)?.is_some() {
                        self.backend.add_edge(
                            &memory_node_id,
                            &file_node_id,
                            EdgeKind::About.name(),
                            "{}",
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Ensure a project concept node exists.
    fn ensure_project_node(&self, node_id: &str, project_id: &str) -> Result<()> {
        if self.backend.get_node(node_id)?.is_none() {
            self.backend.add_node(
                node_id,
                NodeKind::Concept.table_name(),
                &format!("Project \"{project_id}\""),
                "{}",
            )?;
        }
        Ok(())
    }

    /// Find related nodes to a given node (BFS up to `max_depth`).
    pub fn find_related(&self, id: &str, max_depth: usize) -> Result<Vec<DepthNeighbor>> {
        let mut results: Vec<DepthNeighbor> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        // Queue entries: (node_id, depth, via_node_id)
        let mut queue: VecDeque<(String, usize, Option<String>)> = VecDeque::new();

        visited.insert(id.to_string());
        queue.push_back((id.to_string(), 0, None));

        while let Some((current_id, depth, via)) = queue.pop_front() {
            if depth > max_depth {
                break;
            }

            let neighbors = self.backend.query_neighbors(&current_id, None)?;
            for neighbor in neighbors {
                if visited.contains(&neighbor.id) {
                    continue;
                }
                visited.insert(neighbor.id.clone());

                let via_for_result = if depth == 0 {
                    None
                } else {
                    Some(via.clone().unwrap_or_else(|| current_id.clone()))
                };

                results.push(DepthNeighbor {
                    id: neighbor.id.clone(),
                    kind: neighbor.kind.clone(),
                    name: neighbor.name.clone(),
                    properties: neighbor.properties.clone(),
                    edge_kind: neighbor.edge_kind.clone(),
                    direction: neighbor.direction.clone(),
                    depth: depth + 1,
                    via_node_id: via_for_result,
                });

                if depth + 1 < max_depth {
                    queue.push_back((
                        neighbor.id.clone(),
                        depth + 1,
                        Some(via.clone().unwrap_or_else(|| current_id.clone())),
                    ));
                }
            }
        }

        Ok(results)
    }

    /// Find a node by a query string. Tries exact ID match first, then
    /// searches by file path, then by name substring.
    pub fn find_node_id(&self, query: &str) -> Result<Option<String>> {
        // Try exact ID
        if self.backend.get_node(query)?.is_some() {
            return Ok(Some(query.to_string()));
        }

        // Try as file path
        let file_id = format!("file:{query}");
        if self.backend.get_node(&file_id)?.is_some() {
            return Ok(Some(file_id));
        }

        // Try as session
        let session_id = format!("session:{query}");
        if self.backend.get_node(&session_id)?.is_some() {
            return Ok(Some(session_id));
        }

        // Try as project
        let project_id = format!("project:{query}");
        if self.backend.get_node(&project_id)?.is_some() {
            return Ok(Some(project_id));
        }

        Ok(None)
    }

    /// Get the graph stats: (node_count, edge_count).
    pub fn stats(&self) -> Result<(usize, usize)> {
        Ok((self.backend.node_count()?, self.backend.edge_count()?))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Truncate a string, appending "..." if it exceeds `max_chars`.
fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

/// Heuristic: does this look like a file path?
fn looks_like_file_path(s: &str) -> bool {
    if s.len() < 3 {
        return false;
    }
    // Must contain a dot (for extension) or a slash (for path)
    let has_slash = s.contains('/');
    let has_ext = s.rsplit('.').next().is_some_and(|ext| {
        matches!(
            ext,
            "rs" | "py"
                | "js"
                | "ts"
                | "go"
                | "java"
                | "c"
                | "cpp"
                | "h"
                | "rb"
                | "php"
                | "swift"
                | "kt"
                | "toml"
                | "yaml"
                | "yml"
                | "json"
                | "md"
                | "txt"
                | "sh"
                | "css"
                | "html"
                | "sql"
        )
    });
    has_slash || has_ext
}

// ---------------------------------------------------------------------------
// Display helpers for CLI
// ---------------------------------------------------------------------------

/// Render a flat list of depth-neighbors as "Related to" output.
pub fn format_related(query: &str, neighbors: &[DepthNeighbor]) -> String {
    let mut out = String::new();
    out.push_str(&format!("Related to: {query}\n\n"));

    let direct: Vec<_> = neighbors.iter().filter(|n| n.depth == 1).collect();
    let indirect: Vec<_> = neighbors.iter().filter(|n| n.depth > 1).collect();

    if direct.is_empty() && indirect.is_empty() {
        out.push_str("  (no connections found)\n");
        return out;
    }

    if !direct.is_empty() {
        out.push_str("Direct connections:\n");
        for n in &direct {
            let arrow = match n.direction.as_str() {
                "outgoing" => "->",
                _ => "<-",
            };
            out.push_str(&format!(
                "  {} [{}] {} ({})\n",
                arrow, n.edge_kind, n.name, n.kind
            ));
        }
    }

    if !indirect.is_empty() {
        out.push_str("\n2-hop connections:\n");
        for n in &indirect {
            let arrow = match n.direction.as_str() {
                "outgoing" => "->",
                _ => "<-",
            };
            let via = n.via_node_id.as_deref().unwrap_or("?");
            out.push_str(&format!(
                "  {} [{}] {} ({}) (via {})\n",
                arrow, n.edge_kind, n.name, n.kind, via
            ));
        }
    }

    out
}

/// Render neighbors as an ASCII tree.
pub fn format_graph_tree(_query: &str, node_name: &str, neighbors: &[DepthNeighbor]) -> String {
    let mut out = String::new();
    out.push_str(&format!("{node_name}\n"));

    if neighbors.is_empty() {
        out.push_str("  (no connections)\n");
        return out;
    }

    // Group depth-1 neighbors; for each, find their depth-2 children
    let direct: Vec<_> = neighbors.iter().filter(|n| n.depth == 1).collect();

    for (i, d1) in direct.iter().enumerate() {
        let is_last_d1 = i == direct.len() - 1;
        let prefix = if is_last_d1 {
            "\u{2514}\u{2500}\u{2500} "
        } else {
            "\u{251c}\u{2500}\u{2500} "
        };
        out.push_str(&format!("{prefix}[{}] {}\n", d1.edge_kind, d1.name));

        // Find depth-2 neighbors that came via this node
        let children: Vec<_> = neighbors
            .iter()
            .filter(|n| n.depth == 2 && n.via_node_id.as_deref() == Some(&d1.id))
            .collect();

        let child_prefix = if is_last_d1 { "    " } else { "\u{2502}   " };
        for (j, d2) in children.iter().enumerate() {
            let is_last_d2 = j == children.len() - 1;
            let child_branch = if is_last_d2 {
                "\u{2514}\u{2500}\u{2500} "
            } else {
                "\u{251c}\u{2500}\u{2500} "
            };
            out.push_str(&format!(
                "{child_prefix}{child_branch}[{}] {}\n",
                d2.edge_kind, d2.name
            ));
        }
    }

    out
}

/// Format timeline entries chronologically.
pub fn format_timeline(topic: &str, sessions: &[Session], memories: &[Memory]) -> String {
    let mut out = String::new();
    out.push_str(&format!("Timeline: {topic}\n\n"));

    // Collect all entries with timestamps
    struct Entry {
        timestamp: String,
        kind: &'static str,
        description: String,
        agent: String,
    }

    let mut entries: Vec<Entry> = Vec::new();

    for s in sessions {
        let ts = s
            .started_at
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "????-??-?? ??:??".to_string());
        let summary = s.summary.as_deref().unwrap_or("(no summary)");
        entries.push(Entry {
            timestamp: ts,
            kind: "session",
            description: summary.to_string(),
            agent: s.agent.clone(),
        });
    }

    for m in memories {
        let ts = m
            .created_at
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "????-??-?? ??:??".to_string());
        entries.push(Entry {
            timestamp: ts,
            kind: "memory",
            description: truncate_str(&m.content, 60),
            agent: String::new(),
        });
    }

    // Sort chronologically
    entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    if entries.is_empty() {
        out.push_str("  (no matching entries found)\n");
        return out;
    }

    for entry in &entries {
        let agent_suffix = if entry.agent.is_empty() {
            String::new()
        } else {
            format!(" ({})", entry.agent)
        };
        out.push_str(&format!(
            "{}  [{}]  {}{}\n",
            entry.timestamp, entry.kind, entry.description, agent_suffix
        ));
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_session(id: &str, project: &str, files: Vec<&str>, summary: &str) -> Session {
        Session {
            id: id.to_string(),
            project_id: Some(project.to_string()),
            agent: "claude".to_string(),
            started_at: Some(Utc::now().naive_utc()),
            ended_at: None,
            duration_minutes: Some(10),
            message_count: Some(5),
            tool_call_count: Some(3),
            total_tokens: Some(1200),
            files_changed: files.into_iter().map(String::from).collect(),
            summary: Some(summary.to_string()),
        }
    }

    fn make_memory(id: &str, content: &str, session_id: Option<&str>) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: Some("proj-1".to_string()),
            content: content.to_string(),
            memory_type: Some("insight".to_string()),
            source_session_id: session_id.map(String::from),
            confidence: 0.9,
            access_count: 0,
            created_at: Some(Utc::now().naive_utc()),
            updated_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        }
    }

    fn make_decision(id: &str, what: &str, session_id: Option<&str>) -> Decision {
        Decision {
            id: id.to_string(),
            session_id: session_id.map(String::from),
            project_id: Some("proj-1".to_string()),
            decision_type: Some("architecture".to_string()),
            what: what.to_string(),
            why: Some("good reasons".to_string()),
            alternatives: vec!["alt-A".to_string()],
            outcome: None,
            created_at: Some(Utc::now().naive_utc()),
            valid_until: None,
        }
    }

    /// Helper to call GraphBackend::get_node unambiguously (avoids clash with GraphStoreBackend).
    fn backend_get_node(
        b: &impl GraphBackend,
        id: &str,
    ) -> Option<(String, String, String, String)> {
        GraphBackend::get_node(b, id).unwrap()
    }

    /// Helper to call GraphBackend::query_neighbors unambiguously.
    fn backend_query_neighbors(
        b: &impl GraphBackend,
        id: &str,
        edge_kind: Option<&str>,
    ) -> Vec<NeighborInfo> {
        GraphBackend::query_neighbors(b, id, edge_kind).unwrap()
    }

    #[test]
    fn test_add_session_creates_nodes_and_edges() {
        let builder = GraphBuilder::new();
        let session = make_session(
            "s-1",
            "proj-1",
            vec!["src/main.rs", "src/lib.rs"],
            "refactored module",
        );

        builder.add_session(&session).unwrap();
        builder.extract_code_entities(&[session]).unwrap();

        // Session node exists
        let session_node = backend_get_node(builder.backend(), "session:s-1");
        assert!(session_node.is_some());
        let (_, _, name, _) = session_node.unwrap();
        assert_eq!(name, "refactored module");

        // Project node exists
        let proj_node = backend_get_node(builder.backend(), "project:proj-1");
        assert!(proj_node.is_some());

        // Code entity nodes exist
        let file1 = backend_get_node(builder.backend(), "file:src/main.rs");
        assert!(file1.is_some());
        let (_, _, name1, _) = file1.unwrap();
        assert_eq!(name1, "main.rs");

        let file2 = backend_get_node(builder.backend(), "file:src/lib.rs");
        assert!(file2.is_some());

        // Edges: session -> project (About), file -> session (UsedIn)
        let (nodes, edges) = builder.stats().unwrap();
        assert_eq!(nodes, 4); // session + project + 2 files
        assert!(edges >= 3); // About + 2 UsedIn
    }

    #[test]
    fn test_add_memory_links_to_session() {
        let builder = GraphBuilder::new();

        // First add the session so the DerivedFrom edge target exists
        let session = make_session("s-1", "proj-1", vec![], "test session");
        builder.add_session(&session).unwrap();

        let memory = make_memory("m-1", "Always validate tokens server-side", Some("s-1"));
        builder.add_memory(&memory).unwrap();

        // Memory node exists
        let mem_node = backend_get_node(builder.backend(), "memory:m-1");
        assert!(mem_node.is_some());

        // Check DerivedFrom edge exists (memory -> session)
        let neighbors = backend_query_neighbors(
            builder.backend(),
            "memory:m-1",
            Some(EdgeKind::DerivedFrom.name()),
        );
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, "session:s-1");
    }

    #[test]
    fn test_find_related_returns_neighbors() {
        let builder = GraphBuilder::new();

        let s1 = make_session("s-1", "proj-1", vec!["src/main.rs"], "session one");
        let s2 = make_session(
            "s-2",
            "proj-1",
            vec!["src/main.rs", "src/auth.rs"],
            "session two",
        );

        builder.add_session(&s1).unwrap();
        builder.add_session(&s2).unwrap();
        builder.extract_code_entities(&[s1, s2]).unwrap();

        // Find related to file:src/main.rs
        let related = builder.find_related("file:src/main.rs", 2).unwrap();
        assert!(!related.is_empty());

        // Should find at least the two sessions via UsedIn edges
        let session_neighbors: Vec<_> = related
            .iter()
            .filter(|n| n.id.starts_with("session:"))
            .collect();
        assert!(session_neighbors.len() >= 2);
    }

    #[test]
    fn test_build_from_data_full() {
        let builder = GraphBuilder::new();

        let sessions = vec![
            make_session("s-1", "proj-1", vec!["src/main.rs"], "added auth flow"),
            make_session(
                "s-2",
                "proj-1",
                vec!["src/main.rs", "src/auth.rs"],
                "fixed auth bug",
            ),
        ];
        let memories = vec![
            make_memory("m-1", "JWT tokens expire after 24h", Some("s-1")),
            make_memory("m-2", "Always validate on server side", None),
        ];
        let decisions = vec![make_decision("d-1", "use JWT for auth", Some("s-1"))];
        let tool_calls: Vec<ToolCall> = vec![];

        builder
            .build_from_data(&sessions, &memories, &decisions, &tool_calls)
            .unwrap();

        let (nodes, edges) = builder.stats().unwrap();

        // Expected nodes: 2 sessions + 1 project + 2 files + 2 memories + 1 decision = 8
        assert!(nodes >= 8, "Expected at least 8 nodes, got {nodes}");
        // Expected edges: many (About, UsedIn, DerivedFrom, RelatesTo for shared files)
        assert!(edges >= 5, "Expected at least 5 edges, got {edges}");

        // Verify cross-entity relationship: sessions share src/main.rs -> RelatesTo
        let s1_neighbors = backend_query_neighbors(
            builder.backend(),
            "session:s-1",
            Some(EdgeKind::RelatesTo.name()),
        );
        assert!(
            !s1_neighbors.is_empty(),
            "Sessions sharing a file should be linked via RelatesTo"
        );
    }

    #[test]
    fn test_find_node_id_resolution() {
        let builder = GraphBuilder::new();
        let session = make_session("s-1", "proj-1", vec!["src/main.rs"], "test");
        builder.add_session(&session).unwrap();
        builder.extract_code_entities(&[session]).unwrap();

        // Find by file path
        assert_eq!(
            builder.find_node_id("src/main.rs").unwrap(),
            Some("file:src/main.rs".to_string())
        );

        // Find by session id
        assert_eq!(
            builder.find_node_id("s-1").unwrap(),
            Some("session:s-1".to_string())
        );

        // Find by project id
        assert_eq!(
            builder.find_node_id("proj-1").unwrap(),
            Some("project:proj-1".to_string())
        );

        // Not found
        assert_eq!(builder.find_node_id("nonexistent").unwrap(), None);
    }

    #[test]
    fn test_empty_data() {
        let builder = GraphBuilder::new();
        builder.build_from_data(&[], &[], &[], &[]).unwrap();
        let (nodes, edges) = builder.stats().unwrap();
        assert_eq!(nodes, 0);
        assert_eq!(edges, 0);
    }

    #[test]
    fn test_format_timeline() {
        let sessions = vec![make_session("s-1", "proj-1", vec![], "Added auth")];
        let memories = vec![make_memory("m-1", "JWT tokens expire", None)];

        let output = format_timeline("auth", &sessions, &memories);
        assert!(output.contains("Timeline: auth"));
        assert!(output.contains("[session]"));
        assert!(output.contains("[memory]"));
    }
}
