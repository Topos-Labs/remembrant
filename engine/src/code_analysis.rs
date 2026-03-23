//! Bridge between Infiniloom's code analysis and Remembrant's storage.
//! Gated behind the `code-analysis` feature flag.

use anyhow::{Context, Result};
use chrono::Utc;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::store::duckdb::{
    AnalysisRun, CodeDependency, CodeSymbol as DuckCodeSymbol, DuckStore,
};
use crate::store::graph::{EdgeKind, GraphEdge, GraphNode, GraphStoreBackend, NodeKind};

// Import Infiniloom types
use infiniloom_engine::index::builder::IndexBuilder;
use infiniloom_engine::index::types::{DepGraph, SymbolIndex};

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
        store.clear_symbols_for_project(&self.project_id)?;

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

        // Step 3: Store symbols in DuckDB
        debug!("Storing symbols in DuckDB...");
        let symbols_stored = self.store_symbols(store, &index)?;

        // Step 4: Store dependencies in DuckDB
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
        self.record_analysis_run(store, &index, symbols_stored, deps_stored, duration_ms)?;

        Ok(AnalysisResult {
            files_analyzed: index.files.len(),
            symbols_extracted: symbols_stored,
            dependencies_found: deps_stored,
            duration_ms,
        })
    }

    /// Convert Infiniloom symbols to DuckDB CodeSymbol and store
    fn store_symbols(&self, store: &DuckStore, index: &SymbolIndex) -> Result<usize> {
        let mut count = 0;

        for symbol in &index.symbols {
            let file = match index.get_file_by_id(symbol.file_id.as_u32()) {
                Some(f) => f,
                None => {
                    warn!(symbol = %symbol.name, "symbol references non-existent file, skipping");
                    continue;
                }
            };

            let duck_symbol = DuckCodeSymbol {
                id: format!(
                    "symbol:{}:{}:{}:{}",
                    self.project_id, file.path, symbol.name, symbol.span.start_line
                ),
                project_id: self.project_id.clone(),
                file_path: file.path.clone(),
                symbol_name: symbol.name.clone(),
                symbol_kind: symbol.kind.name().to_string(),
                signature: symbol.signature.clone(),
                docstring: symbol.docstring.clone(),
                start_line: symbol.span.start_line as i32,
                end_line: symbol.span.end_line as i32,
                visibility: Some(format!("{:?}", symbol.visibility).to_lowercase()),
                parent_symbol: symbol.parent.map(|p| p.to_string()),
                pagerank_score: 0.0, // Will be updated by PageRank pass
                reference_count: 0,  // Will be updated after dep graph analysis
                language: Some(file.language.name().to_string()),
                content_hash: None, // Set during embedding pass
                indexed_at: Some(Utc::now().naive_utc()),
            };

            if let Err(e) = store.insert_code_symbol(&duck_symbol) {
                warn!(symbol = %symbol.name, error = %e, "failed to insert symbol");
                continue;
            }
            count += 1;
        }

        info!(count, "symbols stored in DuckDB");
        Ok(count)
    }

    /// Convert Infiniloom dependencies to DuckDB CodeDependency and store.
    /// Handles calls, file imports, and symbol-level extends/implements relationships.
    fn store_dependencies(
        &self,
        store: &DuckStore,
        index: &SymbolIndex,
        dep_graph: &DepGraph,
    ) -> Result<usize> {
        let mut count = 0;

        // Store function call edges
        for (caller_id, callee_id) in &dep_graph.calls {
            if let (Some(caller), Some(callee)) =
                (index.get_symbol(*caller_id), index.get_symbol(*callee_id))
            {
                let caller_file = match index.get_file_by_id(caller.file_id.as_u32()) {
                    Some(f) => f,
                    None => continue,
                };
                let callee_file = match index.get_file_by_id(callee.file_id.as_u32()) {
                    Some(f) => f,
                    None => continue,
                };

                let dep = CodeDependency {
                    id: format!("dep:call:{}:{}:{}", self.project_id, caller_id, callee_id),
                    project_id: self.project_id.clone(),
                    from_symbol: format!("{}:{}", caller_file.path, caller.name),
                    to_symbol: format!("{}:{}", callee_file.path, callee.name),
                    relationship: "calls".to_string(),
                    from_file: caller_file.path.clone(),
                    to_file: callee_file.path.clone(),
                };

                if let Err(e) = store.insert_code_dependency(&dep) {
                    debug!(error = %e, "failed to insert call dependency");
                }
                count += 1;
            }
        }

        // Store file import edges
        for (from_file_id, to_file_id) in &dep_graph.file_imports {
            if let (Some(from_file), Some(to_file)) = (
                index.get_file_by_id(*from_file_id),
                index.get_file_by_id(*to_file_id),
            ) {
                let dep = CodeDependency {
                    id: format!(
                        "dep:import:{}:{}:{}",
                        self.project_id, from_file_id, to_file_id
                    ),
                    project_id: self.project_id.clone(),
                    from_symbol: from_file.path.clone(),
                    to_symbol: to_file.path.clone(),
                    relationship: "imports".to_string(),
                    from_file: from_file.path.clone(),
                    to_file: to_file.path.clone(),
                };

                if let Err(e) = store.insert_code_dependency(&dep) {
                    debug!(error = %e, "failed to insert import dependency");
                }
                count += 1;
            }
        }

        // NOTE: IndexSymbol (v0.7) doesn't have extends/implements fields.
        // Those are in the higher-level Symbol type. When Infiniloom exposes them
        // via IndexBuilder, we can wire inherits/implements edges here.

        info!(count, "dependencies stored in DuckDB");
        Ok(count)
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
            let file = match index.get_file_by_id(symbol.file_id.as_u32()) {
                Some(f) => f,
                None => continue,
            };

            let node_id = format!(
                "symbol:{}:{}:{}:{}",
                self.project_id, file.path, symbol.name, symbol.span.start_line
            );

            let mut properties = std::collections::HashMap::new();
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
            if let Some(doc) = &symbol.docstring {
                properties.insert("docstring".to_string(), doc.clone());
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

            let mut properties = std::collections::HashMap::new();
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
                        properties: std::collections::HashMap::new(),
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
                let caller_file = match index.get_file_by_id(caller.file_id.as_u32()) {
                    Some(f) => f,
                    None => continue,
                };
                let callee_file = match index.get_file_by_id(callee.file_id.as_u32()) {
                    Some(f) => f,
                    None => continue,
                };

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
                    properties: std::collections::HashMap::new(),
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
                    properties: std::collections::HashMap::new(),
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
        store: &DuckStore,
        index: &SymbolIndex,
        symbols_extracted: usize,
        dependencies_found: usize,
        duration_ms: u64,
    ) -> Result<()> {
        let run = AnalysisRun {
            project_id: self.project_id.clone(),
            commit_hash: None, // TODO: extract from git
            files_analyzed: index.files.len() as i32,
            symbols_extracted: symbols_extracted as i32,
            dependencies_found: dependencies_found as i32,
            chunks_generated: 0,
            duration_ms: duration_ms as i32,
            analyzed_at: Some(Utc::now().naive_utc()),
        };

        store.insert_analysis_run(&run)?;

        info!(
            "Analysis complete: {} files, {} symbols, {} deps in {}ms",
            run.files_analyzed, symbols_extracted, dependencies_found, duration_ms
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = CodeAnalyzer::new("test-project", "/tmp/test");
        assert_eq!(analyzer.project_id, "test-project");
    }
}
