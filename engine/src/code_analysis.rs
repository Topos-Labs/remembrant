//! Bridge between Infiniloom's code analysis and Remembrant's storage.
//! Gated behind the `code-analysis` feature flag.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info};

use crate::store::duckdb::DuckStore;
use crate::store::graph::{EdgeKind, GraphEdge, GraphNode, GraphStoreBackend, NodeKind};

// Import Infiniloom types
use infiniloom_engine::index::builder::IndexBuilder;
use infiniloom_engine::index::types::{DepGraph, IndexSymbol, SymbolIndex};

// Re-export for convenience
pub use infiniloom_engine::index::types::Language;

/// Code analyzer that bridges Infiniloom's AST parsing into Remembrant's storage
pub struct CodeAnalyzer {
    project_id: String,
    repo_path: PathBuf,
}

/// Result of a code analysis operation
pub struct AnalysisResult {
    pub files_analyzed: usize,
    pub symbols_extracted: usize,
    pub dependencies_found: usize,
    pub duration_ms: u64,
}

/// Code symbol stored in DuckDB
#[derive(Debug, Clone)]
pub struct CodeSymbol {
    pub id: String,
    pub project_id: String,
    pub file_path: String,
    pub name: String,
    pub kind: String,
    pub start_line: u32,
    pub end_line: u32,
    pub signature: Option<String>,
    pub visibility: String,
    pub parent_symbol: Option<String>,
    pub language: Option<String>,
}

/// Code dependency edge
#[derive(Debug, Clone)]
pub struct CodeDependency {
    pub id: String,
    pub project_id: String,
    pub from_symbol: String,
    pub to_symbol: String,
    pub dependency_type: String,
}

/// Analysis run metadata
#[derive(Debug, Clone)]
pub struct AnalysisRun {
    pub id: String,
    pub project_id: String,
    pub files_analyzed: i32,
    pub symbols_extracted: i32,
    pub dependencies_found: i32,
    pub duration_ms: i64,
}

impl CodeAnalyzer {
    /// Create a new code analyzer
    pub fn new(project_id: &str, repo_path: impl AsRef<Path>) -> Self {
        Self {
            project_id: project_id.to_string(),
            repo_path: repo_path.as_ref().to_path_buf(),
        }
    }

    /// Analyze the repository and store results
    pub fn analyze(
        &self,
        store: &DuckStore,
        graph_store: &impl GraphStoreBackend,
    ) -> Result<AnalysisResult> {
        let start = Instant::now();

        info!("Starting code analysis for project: {}", self.project_id);
        info!("Repository path: {}", self.repo_path.display());

        // Step 1: Clear existing symbols for this project (idempotent re-analysis)
        debug!("Clearing existing symbols for project: {}", self.project_id);
        self.clear_existing_symbols(store)?;

        // Step 2: Build index using Infiniloom's IndexBuilder
        info!("Building symbol index...");
        let builder = IndexBuilder::new(&self.repo_path);
        let (index, dep_graph) = builder
            .build()
            .context("Failed to build Infiniloom index")?;

        info!(
            "Index built: {} files, {} symbols",
            index.files.len(),
            index.symbols.len()
        );

        // Step 3: Convert and store symbols in DuckDB
        debug!("Storing symbols in DuckDB...");
        let symbols_stored = self.store_symbols(store, &index)?;

        // Step 4: Convert and store dependencies in DuckDB
        debug!("Storing dependencies in DuckDB...");
        let deps_stored = self.store_dependencies(store, &index, &dep_graph)?;

        // Step 5: Populate graph store
        debug!("Populating graph store...");
        let (nodes_added, edges_added) = self.populate_graph(&index, &dep_graph, graph_store)?;

        info!(
            "Graph populated: {} nodes, {} edges",
            nodes_added, edges_added
        );

        // Step 6: Record analysis run
        let duration_ms = start.elapsed().as_millis() as u64;
        self.record_analysis_run(
            store,
            index.files.len(),
            symbols_stored,
            deps_stored,
            duration_ms,
        )?;

        Ok(AnalysisResult {
            files_analyzed: index.files.len(),
            symbols_extracted: symbols_stored,
            dependencies_found: deps_stored,
            duration_ms,
        })
    }

    /// Clear existing symbols for this project
    fn clear_existing_symbols(&self, _store: &DuckStore) -> Result<()> {
        // Since the tables don't exist yet, we'll implement this when DuckDB schema is extended
        // For now, this is a no-op
        Ok(())
    }

    /// Convert Infiniloom symbols to Remembrant CodeSymbol and store in DuckDB
    fn store_symbols(&self, _store: &DuckStore, index: &SymbolIndex) -> Result<usize> {
        let mut count = 0;

        for symbol in &index.symbols {
            let file = index
                .get_file_by_id(symbol.file_id.as_u32())
                .context("Symbol references non-existent file")?;

            let _code_symbol =
                self.convert_symbol(&self.project_id, &file.path, symbol, Some(file.language));

            // Store in DuckDB (when tables are added by other agent)
            // For now, we just count
            count += 1;
        }

        Ok(count)
    }

    /// Convert Infiniloom dependencies to Remembrant CodeDependency and store
    fn store_dependencies(
        &self,
        _store: &DuckStore,
        index: &SymbolIndex,
        dep_graph: &DepGraph,
    ) -> Result<usize> {
        let mut count = 0;

        // Store function call edges
        for (caller_id, callee_id) in &dep_graph.calls {
            if let (Some(caller), Some(callee)) =
                (index.get_symbol(*caller_id), index.get_symbol(*callee_id))
            {
                let _dep = CodeDependency {
                    id: format!("{}:{}:{}", self.project_id, caller_id, callee_id),
                    project_id: self.project_id.clone(),
                    from_symbol: format!("{}:{}", caller.file_id.as_u32(), caller.name),
                    to_symbol: format!("{}:{}", callee.file_id.as_u32(), callee.name),
                    dependency_type: "call".to_string(),
                };
                // Store in DuckDB (when tables are added)
                count += 1;
            }
        }

        // Store file import edges
        for (_from_file, _to_file) in &dep_graph.file_imports {
            // Create dependency record
            count += 1;
        }

        Ok(count)
    }

    /// Convert an Infiniloom IndexSymbol to Remembrant's CodeSymbol
    fn convert_symbol(
        &self,
        project_id: &str,
        file_path: &str,
        symbol: &IndexSymbol,
        language: Option<Language>,
    ) -> CodeSymbol {
        CodeSymbol {
            id: format!(
                "symbol:{}:{}:{}:{}",
                project_id, file_path, symbol.name, symbol.span.start_line
            ),
            project_id: project_id.to_string(),
            file_path: file_path.to_string(),
            name: symbol.name.clone(),
            kind: symbol.kind.name().to_string(),
            start_line: symbol.span.start_line,
            end_line: symbol.span.end_line,
            signature: symbol.signature.clone(),
            visibility: format!("{:?}", symbol.visibility).to_lowercase(),
            parent_symbol: symbol.parent.map(|p| p.to_string()),
            language: language.map(|l| l.name().to_string()),
        }
    }

    /// Populate the graph store with nodes and edges
    fn populate_graph(
        &self,
        index: &SymbolIndex,
        dep_graph: &DepGraph,
        graph_store: &impl GraphStoreBackend,
    ) -> Result<(usize, usize)> {
        let mut nodes_added = 0;
        let mut edges_added = 0;

        // Create Symbol nodes for each symbol
        for symbol in &index.symbols {
            let file = index
                .get_file_by_id(symbol.file_id.as_u32())
                .context("Symbol references non-existent file")?;

            let node_id = format!(
                "symbol:{}:{}:{}:{}",
                self.project_id, file.path, symbol.name, symbol.span.start_line
            );

            let mut properties = HashMap::new();
            properties.insert("file_path".to_string(), file.path.clone());
            properties.insert("kind".to_string(), symbol.kind.name().to_string());
            properties.insert("start_line".to_string(), symbol.span.start_line.to_string());
            properties.insert("end_line".to_string(), symbol.span.end_line.to_string());
            properties.insert(
                "visibility".to_string(),
                format!("{:?}", symbol.visibility).to_lowercase(),
            );

            if let Some(sig) = &symbol.signature {
                properties.insert("signature".to_string(), sig.clone());
            }

            let node = GraphNode {
                id: node_id.clone(),
                kind: NodeKind::Symbol,
                name: symbol.name.clone(),
                properties,
            };

            graph_store.add_node(&node)?;
            nodes_added += 1;
        }

        // Create CodeEntity nodes for each file
        for file in &index.files {
            let node_id = format!("file:{}", file.path);

            let mut properties = HashMap::new();
            properties.insert("path".to_string(), file.path.clone());
            properties.insert("language".to_string(), file.language.name().to_string());
            properties.insert("lines".to_string(), file.lines.to_string());
            properties.insert("tokens".to_string(), file.tokens.to_string());

            let node = GraphNode {
                id: node_id.clone(),
                kind: NodeKind::CodeEntity,
                name: file.path.clone(),
                properties,
            };

            graph_store.add_node(&node)?;
            nodes_added += 1;

            // Create Defines edges from file to symbols
            for symbol_idx in file.symbols.clone() {
                if let Some(symbol) = index.get_symbol(symbol_idx) {
                    let symbol_id = format!(
                        "symbol:{}:{}:{}:{}",
                        self.project_id, file.path, symbol.name, symbol.span.start_line
                    );

                    let edge = GraphEdge {
                        from_id: node_id.clone(),
                        to_id: symbol_id,
                        kind: EdgeKind::Defines,
                        properties: HashMap::new(),
                    };

                    graph_store.add_edge(&edge)?;
                    edges_added += 1;
                }
            }
        }

        // Create Calls edges between symbols
        for (caller_id, callee_id) in &dep_graph.calls {
            if let (Some(caller), Some(callee)) =
                (index.get_symbol(*caller_id), index.get_symbol(*callee_id))
            {
                let caller_file = index
                    .get_file_by_id(caller.file_id.as_u32())
                    .context("Caller symbol references non-existent file")?;
                let callee_file = index
                    .get_file_by_id(callee.file_id.as_u32())
                    .context("Callee symbol references non-existent file")?;

                let caller_node_id = format!(
                    "symbol:{}:{}:{}:{}",
                    self.project_id, caller_file.path, caller.name, caller.span.start_line
                );
                let callee_node_id = format!(
                    "symbol:{}:{}:{}:{}",
                    self.project_id, callee_file.path, callee.name, callee.span.start_line
                );

                let edge = GraphEdge {
                    from_id: caller_node_id,
                    to_id: callee_node_id,
                    kind: EdgeKind::Calls,
                    properties: HashMap::new(),
                };

                graph_store.add_edge(&edge)?;
                edges_added += 1;
            }
        }

        // Create DependsOn edges for file imports
        for (from_file_id, to_file_id) in &dep_graph.file_imports {
            if let (Some(from_file), Some(to_file)) = (
                index.get_file_by_id(*from_file_id),
                index.get_file_by_id(*to_file_id),
            ) {
                let from_node_id = format!("file:{}", from_file.path);
                let to_node_id = format!("file:{}", to_file.path);

                let edge = GraphEdge {
                    from_id: from_node_id,
                    to_id: to_node_id,
                    kind: EdgeKind::DependsOn,
                    properties: HashMap::new(),
                };

                graph_store.add_edge(&edge)?;
                edges_added += 1;
            }
        }

        Ok((nodes_added, edges_added))
    }

    /// Record the analysis run in DuckDB
    fn record_analysis_run(
        &self,
        _store: &DuckStore,
        files_analyzed: usize,
        symbols_extracted: usize,
        dependencies_found: usize,
        duration_ms: u64,
    ) -> Result<()> {
        // When DuckDB tables are added, store the analysis run record
        info!(
            "Analysis complete: {} files, {} symbols, {} deps in {}ms",
            files_analyzed, symbols_extracted, dependencies_found, duration_ms
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = CodeAnalyzer::new("test-project", "/tmp/test");
        assert_eq!(analyzer.project_id, "test-project");
    }

    #[test]
    fn test_convert_symbol() {
        use infiniloom_engine::index::types::{
            FileId, IndexSymbol, IndexSymbolKind, Span, SymbolId, Visibility,
        };

        let analyzer = CodeAnalyzer::new("test-project", "/tmp/test");

        let symbol = IndexSymbol {
            id: SymbolId::new(0),
            name: "test_function".to_string(),
            kind: IndexSymbolKind::Function,
            file_id: FileId::new(0),
            span: Span::new(10, 0, 20, 0),
            signature: Some("fn test_function()".to_string()),
            parent: None,
            visibility: Visibility::Public,
            docstring: None,
        };

        let code_symbol =
            analyzer.convert_symbol("test-project", "src/main.rs", &symbol, Some(Language::Rust));

        assert_eq!(code_symbol.name, "test_function");
        assert_eq!(code_symbol.kind, "function");
        assert_eq!(code_symbol.start_line, 10);
        assert_eq!(code_symbol.end_line, 20);
        assert_eq!(code_symbol.language, Some("rust".to_string()));
    }
}
