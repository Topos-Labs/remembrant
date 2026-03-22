use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::database::CreateTableMode;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Result structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSearchResult {
    pub id: String,
    pub content: String,
    pub granularity: String,
    pub project_id: String,
    pub file_path: Option<String>,
    pub language: Option<String>,
    pub distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub project_id: String,
    pub distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolSearchResult {
    pub id: String,
    pub content: String,
    pub project_id: String,
    pub file_path: String,
    pub language: Option<String>,
    pub symbol_name: String,
    pub symbol_kind: String,
    pub signature: Option<String>,
    pub pagerank: Option<f32>,
    pub start_line: u32,
    pub end_line: u32,
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// LanceStore
// ---------------------------------------------------------------------------

/// Persistent vector store backed by LanceDB.
pub struct LanceStore {
    conn: lancedb::Connection,
    embedding_dim: i32,
}

impl LanceStore {
    /// Open (or create) a LanceDB database at the given directory path.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_dim(path, 1024).await
    }

    /// Open with a custom embedding dimension.
    pub async fn open_with_dim(path: impl AsRef<Path>, embedding_dim: i32) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let conn = lancedb::connect(&path_str)
            .execute()
            .await
            .with_context(|| format!("failed to open LanceDB at {path_str}"))?;
        let store = Self {
            conn,
            embedding_dim,
        };
        store.init_tables().await?;
        Ok(store)
    }

    // -----------------------------------------------------------------------
    // Schema helpers
    // -----------------------------------------------------------------------

    fn code_embeddings_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dim,
                ),
                true,
            ),
            Field::new("content", DataType::Utf8, false),
            Field::new("granularity", DataType::Utf8, false),
            Field::new("project_id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, true),
            Field::new("language", DataType::Utf8, true),
            Field::new("created_at", DataType::Utf8, false),
        ]))
    }

    fn memory_embeddings_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dim,
                ),
                true,
            ),
            Field::new("content", DataType::Utf8, false),
            Field::new("memory_type", DataType::Utf8, false),
            Field::new("project_id", DataType::Utf8, false),
            Field::new("created_at", DataType::Utf8, false),
        ]))
    }

    fn symbol_embeddings_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dim,
                ),
                true,
            ),
            Field::new("content", DataType::Utf8, false),
            Field::new("project_id", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("language", DataType::Utf8, true),
            Field::new("symbol_name", DataType::Utf8, false),
            Field::new("symbol_kind", DataType::Utf8, false),
            Field::new("signature", DataType::Utf8, true),
            Field::new("pagerank", DataType::Float32, true),
            Field::new("start_line", DataType::UInt32, false),
            Field::new("end_line", DataType::UInt32, false),
            Field::new("created_at", DataType::Utf8, false),
        ]))
    }

    // -----------------------------------------------------------------------
    // Table initialisation
    // -----------------------------------------------------------------------

    /// Create the `code_embeddings` and `memory_embeddings` tables if they do
    /// not already exist.
    pub async fn init_tables(&self) -> Result<()> {
        // code_embeddings
        let code_schema = self.code_embeddings_schema();
        self.conn
            .create_empty_table("code_embeddings", code_schema)
            .mode(CreateTableMode::exist_ok(|req| req))
            .execute()
            .await
            .context("failed to create code_embeddings table")?;

        // memory_embeddings
        let mem_schema = self.memory_embeddings_schema();
        self.conn
            .create_empty_table("memory_embeddings", mem_schema)
            .mode(CreateTableMode::exist_ok(|req| req))
            .execute()
            .await
            .context("failed to create memory_embeddings table")?;

        // symbol_embeddings
        let sym_schema = self.symbol_embeddings_schema();
        self.conn
            .create_empty_table("symbol_embeddings", sym_schema)
            .mode(CreateTableMode::exist_ok(|req| req))
            .execute()
            .await
            .context("failed to create symbol_embeddings table")?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Inserts
    // -----------------------------------------------------------------------

    /// Insert a single code embedding record.
    pub async fn insert_code_embedding(
        &self,
        id: &str,
        embedding: &[f32],
        content: &str,
        granularity: &str,
        project_id: &str,
        file_path: Option<&str>,
        language: Option<&str>,
    ) -> Result<()> {
        let schema = self.code_embeddings_schema();
        let now = chrono::Utc::now().to_rfc3339();
        let dim = self.embedding_dim;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id])),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        vec![Some(embedding.iter().map(|v| Some(*v)).collect::<Vec<_>>())],
                        dim,
                    ),
                ),
                Arc::new(StringArray::from(vec![content])),
                Arc::new(StringArray::from(vec![granularity])),
                Arc::new(StringArray::from(vec![project_id])),
                Arc::new(StringArray::from(vec![file_path])),
                Arc::new(StringArray::from(vec![language])),
                Arc::new(StringArray::from(vec![now.as_str()])),
            ],
        )
        .context("failed to build code_embeddings record batch")?;

        let table = self
            .conn
            .open_table("code_embeddings")
            .execute()
            .await
            .context("failed to open code_embeddings table")?;

        table
            .add(vec![batch])
            .execute()
            .await
            .context("failed to insert code embedding")?;

        Ok(())
    }

    /// Insert a single memory embedding record.
    pub async fn insert_memory_embedding(
        &self,
        id: &str,
        embedding: &[f32],
        content: &str,
        memory_type: &str,
        project_id: &str,
    ) -> Result<()> {
        let schema = self.memory_embeddings_schema();
        let now = chrono::Utc::now().to_rfc3339();
        let dim = self.embedding_dim;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id])),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        vec![Some(embedding.iter().map(|v| Some(*v)).collect::<Vec<_>>())],
                        dim,
                    ),
                ),
                Arc::new(StringArray::from(vec![content])),
                Arc::new(StringArray::from(vec![memory_type])),
                Arc::new(StringArray::from(vec![project_id])),
                Arc::new(StringArray::from(vec![now.as_str()])),
            ],
        )
        .context("failed to build memory_embeddings record batch")?;

        let table = self
            .conn
            .open_table("memory_embeddings")
            .execute()
            .await
            .context("failed to open memory_embeddings table")?;

        table
            .add(vec![batch])
            .execute()
            .await
            .context("failed to insert memory embedding")?;

        Ok(())
    }

    /// Insert a single symbol embedding record.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_symbol_embedding(
        &self,
        id: &str,
        embedding: &[f32],
        content: &str,
        project_id: &str,
        file_path: &str,
        language: Option<&str>,
        symbol_name: &str,
        symbol_kind: &str,
        signature: Option<&str>,
        pagerank: Option<f32>,
        start_line: u32,
        end_line: u32,
    ) -> Result<()> {
        let schema = self.symbol_embeddings_schema();
        let now = chrono::Utc::now().to_rfc3339();
        let dim = self.embedding_dim;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id])),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                        vec![Some(embedding.iter().map(|v| Some(*v)).collect::<Vec<_>>())],
                        dim,
                    ),
                ),
                Arc::new(StringArray::from(vec![content])),
                Arc::new(StringArray::from(vec![project_id])),
                Arc::new(StringArray::from(vec![file_path])),
                Arc::new(StringArray::from(vec![language])),
                Arc::new(StringArray::from(vec![symbol_name])),
                Arc::new(StringArray::from(vec![symbol_kind])),
                Arc::new(StringArray::from(vec![signature])),
                Arc::new(Float32Array::from(vec![pagerank])),
                Arc::new(UInt32Array::from(vec![start_line])),
                Arc::new(UInt32Array::from(vec![end_line])),
                Arc::new(StringArray::from(vec![now.as_str()])),
            ],
        )
        .context("failed to build symbol_embeddings record batch")?;

        let table = self
            .conn
            .open_table("symbol_embeddings")
            .execute()
            .await
            .context("failed to open symbol_embeddings table")?;

        table
            .add(vec![batch])
            .execute()
            .await
            .context("failed to insert symbol embedding")?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Vector search
    // -----------------------------------------------------------------------

    /// Search code embeddings by vector similarity.
    pub async fn search_code(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<CodeSearchResult>> {
        let table = self
            .conn
            .open_table("code_embeddings")
            .execute()
            .await
            .context("failed to open code_embeddings for search")?;

        let batches = table
            .query()
            .nearest_to(query_embedding)
            .context("failed to build vector query for code_embeddings")?
            .limit(limit)
            .execute()
            .await
            .context("failed to execute code search")?
            .try_collect::<Vec<_>>()
            .await
            .context("failed to collect code search results")?;

        let mut results = Vec::new();
        for batch in &batches {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let contents = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let granularities = batch
                .column_by_name("granularity")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let project_ids = batch
                .column_by_name("project_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let file_paths = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let languages = batch
                .column_by_name("language")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            let (Some(ids), Some(contents), Some(granularities), Some(project_ids)) =
                (ids, contents, granularities, project_ids)
            else {
                continue;
            };

            for i in 0..batch.num_rows() {
                results.push(CodeSearchResult {
                    id: ids.value(i).to_string(),
                    content: contents.value(i).to_string(),
                    granularity: granularities.value(i).to_string(),
                    project_id: project_ids.value(i).to_string(),
                    file_path: file_paths.and_then(|a| {
                        if a.is_null(i) {
                            None
                        } else {
                            Some(a.value(i).to_string())
                        }
                    }),
                    language: languages.and_then(|a| {
                        if a.is_null(i) {
                            None
                        } else {
                            Some(a.value(i).to_string())
                        }
                    }),
                    distance: distances.map(|d| d.value(i)).unwrap_or(f32::MAX),
                });
            }
        }

        Ok(results)
    }

    /// Search symbol embeddings by vector similarity.
    pub async fn search_symbols(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<SymbolSearchResult>> {
        self.search_symbols_internal(query_embedding, limit, None)
            .await
    }

    /// Search symbol embeddings filtered by symbol kind.
    pub async fn search_symbols_by_kind(
        &self,
        query_embedding: &[f32],
        kind: &str,
        limit: usize,
    ) -> Result<Vec<SymbolSearchResult>> {
        let filter = format!("symbol_kind = '{}'", kind.replace('\'', "''"));
        self.search_symbols_internal(query_embedding, limit, Some(&filter))
            .await
    }

    /// Search symbol embeddings filtered by project ID.
    pub async fn search_symbols_in_project(
        &self,
        query_embedding: &[f32],
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<SymbolSearchResult>> {
        let filter = format!("project_id = '{}'", project_id.replace('\'', "''"));
        self.search_symbols_internal(query_embedding, limit, Some(&filter))
            .await
    }

    /// Internal helper for symbol search with optional filter.
    async fn search_symbols_internal(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: Option<&str>,
    ) -> Result<Vec<SymbolSearchResult>> {
        let table = self
            .conn
            .open_table("symbol_embeddings")
            .execute()
            .await
            .context("failed to open symbol_embeddings for search")?;

        let mut query = table
            .query()
            .nearest_to(query_embedding)
            .context("failed to build vector query for symbol_embeddings")?
            .limit(limit);

        if let Some(f) = filter {
            query = query.only_if(f);
        }

        let batches = query
            .execute()
            .await
            .context("failed to execute symbol search")?
            .try_collect::<Vec<_>>()
            .await
            .context("failed to collect symbol search results")?;

        let mut results = Vec::new();
        for batch in &batches {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let contents = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let project_ids = batch
                .column_by_name("project_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let file_paths = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let languages = batch
                .column_by_name("language")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let symbol_names = batch
                .column_by_name("symbol_name")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let symbol_kinds = batch
                .column_by_name("symbol_kind")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let signatures = batch
                .column_by_name("signature")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let pageranks = batch
                .column_by_name("pagerank")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            let start_lines = batch
                .column_by_name("start_line")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let end_lines = batch
                .column_by_name("end_line")
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            let (
                Some(ids),
                Some(contents),
                Some(project_ids),
                Some(file_paths),
                Some(symbol_names),
                Some(symbol_kinds),
                Some(start_lines),
                Some(end_lines),
            ) = (
                ids,
                contents,
                project_ids,
                file_paths,
                symbol_names,
                symbol_kinds,
                start_lines,
                end_lines,
            )
            else {
                continue;
            };

            for i in 0..batch.num_rows() {
                results.push(SymbolSearchResult {
                    id: ids.value(i).to_string(),
                    content: contents.value(i).to_string(),
                    project_id: project_ids.value(i).to_string(),
                    file_path: file_paths.value(i).to_string(),
                    language: languages.and_then(|a| {
                        if a.is_null(i) {
                            None
                        } else {
                            Some(a.value(i).to_string())
                        }
                    }),
                    symbol_name: symbol_names.value(i).to_string(),
                    symbol_kind: symbol_kinds.value(i).to_string(),
                    signature: signatures.and_then(|a| {
                        if a.is_null(i) {
                            None
                        } else {
                            Some(a.value(i).to_string())
                        }
                    }),
                    pagerank: pageranks
                        .and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) }),
                    start_line: start_lines.value(i),
                    end_line: end_lines.value(i),
                    distance: distances.map(|d| d.value(i)).unwrap_or(f32::MAX),
                });
            }
        }

        Ok(results)
    }

    /// Search memory embeddings by vector similarity.
    pub async fn search_memories(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>> {
        let table = self
            .conn
            .open_table("memory_embeddings")
            .execute()
            .await
            .context("failed to open memory_embeddings for search")?;

        let batches = table
            .query()
            .nearest_to(query_embedding)
            .context("failed to build vector query for memory_embeddings")?
            .limit(limit)
            .execute()
            .await
            .context("failed to execute memory search")?
            .try_collect::<Vec<_>>()
            .await
            .context("failed to collect memory search results")?;

        let mut results = Vec::new();
        for batch in &batches {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let contents = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let memory_types = batch
                .column_by_name("memory_type")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let project_ids = batch
                .column_by_name("project_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            let (Some(ids), Some(contents), Some(memory_types), Some(project_ids)) =
                (ids, contents, memory_types, project_ids)
            else {
                continue;
            };

            for i in 0..batch.num_rows() {
                results.push(MemorySearchResult {
                    id: ids.value(i).to_string(),
                    content: contents.value(i).to_string(),
                    memory_type: memory_types.value(i).to_string(),
                    project_id: project_ids.value(i).to_string(),
                    distance: distances.map(|d| d.value(i)).unwrap_or(f32::MAX),
                });
            }
        }

        Ok(results)
    }
}
