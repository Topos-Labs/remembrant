use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Node & edge types
// ---------------------------------------------------------------------------

/// Categories of nodes in the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    CodeEntity,
    Concept,
    Memory,
    Pattern,
    Problem,
    Solution,
    /// An AST-parsed code symbol (function, class, struct, etc.)
    Symbol,
    /// A module or directory in the codebase
    Module,
}

impl NodeKind {
    pub fn table_name(&self) -> &'static str {
        match self {
            Self::CodeEntity => "CodeEntity",
            Self::Concept => "Concept",
            Self::Memory => "Memory",
            Self::Pattern => "Pattern",
            Self::Problem => "Problem",
            Self::Solution => "Solution",
            Self::Symbol => "Symbol",
            Self::Module => "Module",
        }
    }
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.table_name())
    }
}

/// Relationship types between nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    Calls,
    Imports,
    Inherits,
    Implements,
    UsedIn,
    Solves,
    SolvedBy,
    RelatesTo,
    DerivedFrom,
    About,
    Supersedes,
    /// File defines a symbol
    Defines,
    /// Symbol is contained in a file/module
    ContainedIn,
    /// File depends on another file (import)
    DependsOn,
    /// Symbol references another symbol (non-call)
    References,
}

impl EdgeKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Calls => "CALLS",
            Self::Imports => "IMPORTS",
            Self::Inherits => "INHERITS",
            Self::Implements => "IMPLEMENTS",
            Self::UsedIn => "USED_IN",
            Self::Solves => "SOLVES",
            Self::SolvedBy => "SOLVED_BY",
            Self::RelatesTo => "RELATES_TO",
            Self::DerivedFrom => "DERIVED_FROM",
            Self::About => "ABOUT",
            Self::Supersedes => "SUPERSEDES",
            Self::Defines => "DEFINES",
            Self::ContainedIn => "CONTAINED_IN",
            Self::DependsOn => "DEPENDS_ON",
            Self::References => "REFERENCES",
        }
    }
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Graph data types
// ---------------------------------------------------------------------------

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: NodeKind,
    pub name: String,
    pub properties: HashMap<String, String>,
}

/// An edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from_id: String,
    pub to_id: String,
    pub kind: EdgeKind,
    pub properties: HashMap<String, String>,
}

/// A neighbor returned from a graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbor {
    pub node: GraphNode,
    pub edge_kind: EdgeKind,
    pub direction: Direction,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Direction {
    Outgoing,
    Incoming,
}

// ---------------------------------------------------------------------------
// GraphStore trait
// ---------------------------------------------------------------------------

/// Trait for graph database backends.
/// Implementations can use Kuzu, LadybugDB, or an in-memory store.
pub trait GraphStoreBackend: Send + Sync {
    fn add_node(&self, node: &GraphNode) -> Result<()>;
    fn add_edge(&self, edge: &GraphEdge) -> Result<()>;
    fn get_node(&self, id: &str) -> Result<Option<GraphNode>>;
    fn delete_node(&self, id: &str) -> Result<bool>;
    fn query_neighbors(&self, id: &str, edge_kind: Option<EdgeKind>) -> Result<Vec<Neighbor>>;

    /// Find a shortest path between two nodes (up to `max_depth` hops).
    /// Returns the sequence of node IDs along the path, or an empty vec if
    /// no path exists within the depth limit.
    fn query_path(&self, from_id: &str, to_id: &str, max_depth: usize) -> Result<Vec<String>>;

    fn node_count(&self) -> Result<usize>;
    fn edge_count(&self) -> Result<usize>;
}

// ---------------------------------------------------------------------------
// In-memory implementation (default, for MVP and tests)
// ---------------------------------------------------------------------------

/// In-memory graph store. Suitable for development and testing.
/// Will be replaced by Kuzu or LadybugDB when their Rust SDKs mature.
pub struct GraphStore {
    nodes: Mutex<HashMap<String, GraphNode>>,
    edges: Mutex<Vec<GraphEdge>>,
}

impl GraphStore {
    pub fn new() -> Self {
        Self {
            nodes: Mutex::new(HashMap::new()),
            edges: Mutex::new(Vec::new()),
        }
    }

    /// Return all nodes in the graph (for search/iteration).
    pub fn all_nodes(&self) -> Result<Vec<GraphNode>> {
        Ok(self
            .nodes
            .lock()
            .expect("lock poisoned")
            .values()
            .cloned()
            .collect())
    }
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStoreBackend for GraphStore {
    fn add_node(&self, node: &GraphNode) -> Result<()> {
        self.nodes
            .lock()
            .expect("lock poisoned")
            .insert(node.id.clone(), node.clone());
        Ok(())
    }

    fn add_edge(&self, edge: &GraphEdge) -> Result<()> {
        self.edges.lock().expect("lock poisoned").push(edge.clone());
        Ok(())
    }

    fn get_node(&self, id: &str) -> Result<Option<GraphNode>> {
        Ok(self.nodes.lock().expect("lock poisoned").get(id).cloned())
    }

    fn delete_node(&self, id: &str) -> Result<bool> {
        let mut nodes = self.nodes.lock().expect("lock poisoned");
        let removed = nodes.remove(id).is_some();
        if removed {
            // Also remove all incident edges.
            let mut edges = self.edges.lock().expect("lock poisoned");
            edges.retain(|e| e.from_id != id && e.to_id != id);
        }
        Ok(removed)
    }

    fn query_neighbors(&self, id: &str, edge_kind: Option<EdgeKind>) -> Result<Vec<Neighbor>> {
        let nodes = self.nodes.lock().expect("lock poisoned");
        let edges = self.edges.lock().expect("lock poisoned");
        let mut results = Vec::new();

        for edge in edges.iter() {
            if let Some(filter) = edge_kind
                && edge.kind != filter
            {
                continue;
            }

            if edge.from_id == id {
                if let Some(node) = nodes.get(&edge.to_id) {
                    results.push(Neighbor {
                        node: node.clone(),
                        edge_kind: edge.kind,
                        direction: Direction::Outgoing,
                    });
                }
            } else if edge.to_id == id
                && let Some(node) = nodes.get(&edge.from_id)
            {
                results.push(Neighbor {
                    node: node.clone(),
                    edge_kind: edge.kind,
                    direction: Direction::Incoming,
                });
            }
        }

        Ok(results)
    }

    fn query_path(&self, from_id: &str, to_id: &str, max_depth: usize) -> Result<Vec<String>> {
        if from_id == to_id {
            return Ok(vec![from_id.to_string()]);
        }
        if max_depth == 0 {
            return Ok(Vec::new());
        }

        let nodes = self.nodes.lock().expect("lock poisoned");
        let edges = self.edges.lock().expect("lock poisoned");

        // BFS to find shortest path.
        use std::collections::{HashSet, VecDeque};

        // Each entry: (current_node_id, path_so_far)
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
        let mut visited: HashSet<String> = HashSet::new();

        queue.push_back((from_id.to_string(), vec![from_id.to_string()]));
        visited.insert(from_id.to_string());

        while let Some((current, path)) = queue.pop_front() {
            if path.len() > max_depth + 1 {
                break;
            }

            for edge in edges.iter() {
                let next = if edge.from_id == current {
                    &edge.to_id
                } else if edge.to_id == current {
                    &edge.from_id
                } else {
                    continue;
                };

                if !nodes.contains_key(next.as_str()) || visited.contains(next.as_str()) {
                    continue;
                }

                let mut new_path = path.clone();
                new_path.push(next.clone());

                if next == to_id {
                    return Ok(new_path);
                }

                if new_path.len() <= max_depth {
                    visited.insert(next.clone());
                    queue.push_back((next.clone(), new_path));
                }
            }
        }

        Ok(Vec::new())
    }

    fn node_count(&self) -> Result<usize> {
        Ok(self.nodes.lock().expect("lock poisoned").len())
    }

    fn edge_count(&self) -> Result<usize> {
        Ok(self.edges.lock().expect("lock poisoned").len())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_node() {
        let store = GraphStore::new();
        let node = GraphNode {
            id: "fn-1".into(),
            kind: NodeKind::CodeEntity,
            name: "authenticate".into(),
            properties: HashMap::new(),
        };
        store.add_node(&node).unwrap();
        let found = store.get_node("fn-1").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "authenticate");
    }

    #[test]
    fn test_add_edge_and_query_neighbors() {
        let store = GraphStore::new();

        let n1 = GraphNode {
            id: "fn-1".into(),
            kind: NodeKind::CodeEntity,
            name: "authenticate".into(),
            properties: HashMap::new(),
        };
        let n2 = GraphNode {
            id: "fn-2".into(),
            kind: NodeKind::CodeEntity,
            name: "verify_jwt".into(),
            properties: HashMap::new(),
        };
        store.add_node(&n1).unwrap();
        store.add_node(&n2).unwrap();

        let edge = GraphEdge {
            from_id: "fn-1".into(),
            to_id: "fn-2".into(),
            kind: EdgeKind::Calls,
            properties: HashMap::new(),
        };
        store.add_edge(&edge).unwrap();

        let neighbors = store.query_neighbors("fn-1", None).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].node.name, "verify_jwt");

        let filtered = store
            .query_neighbors("fn-1", Some(EdgeKind::Imports))
            .unwrap();
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_counts() {
        let store = GraphStore::new();
        assert_eq!(store.node_count().unwrap(), 0);
        assert_eq!(store.edge_count().unwrap(), 0);

        store
            .add_node(&GraphNode {
                id: "n1".into(),
                kind: NodeKind::Concept,
                name: "auth".into(),
                properties: HashMap::new(),
            })
            .unwrap();

        assert_eq!(store.node_count().unwrap(), 1);
    }

    #[test]
    fn test_delete_node() {
        let store = GraphStore::new();
        let n1 = GraphNode {
            id: "n1".into(),
            kind: NodeKind::CodeEntity,
            name: "foo".into(),
            properties: HashMap::new(),
        };
        let n2 = GraphNode {
            id: "n2".into(),
            kind: NodeKind::CodeEntity,
            name: "bar".into(),
            properties: HashMap::new(),
        };
        store.add_node(&n1).unwrap();
        store.add_node(&n2).unwrap();
        store
            .add_edge(&GraphEdge {
                from_id: "n1".into(),
                to_id: "n2".into(),
                kind: EdgeKind::Calls,
                properties: HashMap::new(),
            })
            .unwrap();

        assert!(store.delete_node("n1").unwrap());
        assert!(!store.delete_node("n1").unwrap()); // already gone
        assert!(store.get_node("n1").unwrap().is_none());
        assert_eq!(store.edge_count().unwrap(), 0); // incident edge removed
    }

    #[test]
    fn test_query_path() {
        let store = GraphStore::new();
        // Build a chain: a -> b -> c -> d
        for id in &["a", "b", "c", "d"] {
            store
                .add_node(&GraphNode {
                    id: id.to_string(),
                    kind: NodeKind::Concept,
                    name: id.to_string(),
                    properties: HashMap::new(),
                })
                .unwrap();
        }
        for (from, to) in &[("a", "b"), ("b", "c"), ("c", "d")] {
            store
                .add_edge(&GraphEdge {
                    from_id: from.to_string(),
                    to_id: to.to_string(),
                    kind: EdgeKind::RelatesTo,
                    properties: HashMap::new(),
                })
                .unwrap();
        }

        let path = store.query_path("a", "d", 5).unwrap();
        assert_eq!(path, vec!["a", "b", "c", "d"]);

        // Too shallow
        let no_path = store.query_path("a", "d", 2).unwrap();
        assert!(no_path.is_empty());

        // Same node
        let self_path = store.query_path("b", "b", 1).unwrap();
        assert_eq!(self_path, vec!["b"]);

        // No connection
        store
            .add_node(&GraphNode {
                id: "isolated".into(),
                kind: NodeKind::Concept,
                name: "island".into(),
                properties: HashMap::new(),
            })
            .unwrap();
        let no_path = store.query_path("a", "isolated", 10).unwrap();
        assert!(no_path.is_empty());
    }

    #[test]
    fn test_cross_kind_edges() {
        let store = GraphStore::new();
        store
            .add_node(&GraphNode {
                id: "prob-1".into(),
                kind: NodeKind::Problem,
                name: "memory leak".into(),
                properties: HashMap::new(),
            })
            .unwrap();
        store
            .add_node(&GraphNode {
                id: "sol-1".into(),
                kind: NodeKind::Solution,
                name: "use arena allocator".into(),
                properties: HashMap::new(),
            })
            .unwrap();
        store
            .add_edge(&GraphEdge {
                from_id: "sol-1".into(),
                to_id: "prob-1".into(),
                kind: EdgeKind::Solves,
                properties: HashMap::new(),
            })
            .unwrap();

        let neighbors = store
            .query_neighbors("prob-1", Some(EdgeKind::Solves))
            .unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].node.name, "use arena allocator");
    }
}
