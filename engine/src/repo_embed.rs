use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

// Conditional imports for code-analysis feature
#[cfg(feature = "code-analysis")]
use infiniloom_engine::chunking::{ChunkStrategy, Chunker};
#[cfg(feature = "code-analysis")]
use infiniloom_engine::incremental::IncrementalScanner;
#[cfg(feature = "code-analysis")]
use infiniloom_engine::security::SecurityScanner;
#[cfg(feature = "code-analysis")]
use infiniloom_engine::types::{RepoFile, Repository, TokenCounts};

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

/// Languages we can detect by extension.
fn detect_language(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "js" => Some("javascript"),
        "ts" => Some("typescript"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "hpp" | "cc" => Some("cpp"),
        "rb" => Some("ruby"),
        "php" => Some("php"),
        "swift" => Some("swift"),
        "kt" => Some("kotlin"),
        "scala" => Some("scala"),
        "ex" | "exs" => Some("elixir"),
        "hs" => Some("haskell"),
        "ml" | "mli" => Some("ocaml"),
        "lua" => Some("lua"),
        "r" | "R" => Some("r"),
        "tf" | "hcl" => Some("hcl"),
        "sh" | "bash" => Some("bash"),
        "zig" => Some("zig"),
        "dart" => Some("dart"),
        "cs" => Some("csharp"),
        "md" => Some("markdown"),
        "yaml" | "yml" => Some("yaml"),
        "toml" => Some("toml"),
        "json" => Some("json"),
        _ => None,
    }
}

/// Directories to skip during traversal.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".cache",
    "vendor",
];

/// Returns `true` if the first `n` bytes of `buf` contain a null byte,
/// indicating likely binary content.
fn looks_binary(buf: &[u8]) -> bool {
    buf.contains(&0)
}

// ---------------------------------------------------------------------------
// CodeChunk
// ---------------------------------------------------------------------------

/// A chunk of source code ready to be embedded.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub file_path: String,
    pub language: Option<String>,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub granularity: String, // "file", "chunk", "symbol"
    /// BLAKE3 content-addressable ID (populated when `code-analysis` feature is enabled).
    pub blake3_id: Option<String>,
}

// ---------------------------------------------------------------------------
// SecretScanResult
// ---------------------------------------------------------------------------

/// Summary of secret scanning findings for a file.
#[derive(Debug, Clone, Default)]
pub struct SecretScanResult {
    /// Number of secrets found.
    pub findings_count: usize,
    /// Number of critical-severity findings.
    pub critical_count: usize,
    /// Number of high-severity findings.
    pub high_count: usize,
}

// ---------------------------------------------------------------------------
// EmbedResult
// ---------------------------------------------------------------------------

/// Summary of an embed-and-store operation.
#[derive(Debug, Default)]
pub struct EmbedResult {
    pub files_found: usize,
    pub chunks_created: usize,
    pub chunks_embedded: usize,
    pub errors: usize,
}

// ---------------------------------------------------------------------------
// RepoEmbedder
// ---------------------------------------------------------------------------

/// Walks a repository, chunks source files, and stores embeddings.
pub struct RepoEmbedder {
    root: PathBuf,
    project_id: String,
    max_file_size: usize,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl RepoEmbedder {
    /// Create a new `RepoEmbedder` rooted at the given directory.
    pub fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        let project_id = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        Self {
            root,
            project_id,
            max_file_size: 100_000, // 100 KB
            chunk_size: 50,         // 50 lines per chunk
            chunk_overlap: 10,      // 10 lines overlap
        }
    }

    /// Override the project ID (default: directory name).
    pub fn with_project_id(mut self, id: &str) -> Self {
        self.project_id = id.to_string();
        self
    }

    /// Return the project ID.
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    // -------------------------------------------------------------------
    // File discovery
    // -------------------------------------------------------------------

    /// Walk the repo and collect all embeddable source files.
    pub fn discover_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        self.walk_dir(&self.root, &mut files)?;
        files.sort();
        Ok(files)
    }

    /// Recursive directory walker.
    fn walk_dir(&self, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        let entries = fs::read_dir(dir)
            .with_context(|| format!("failed to read directory {}", dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;

            if file_type.is_dir() {
                let dir_name = entry.file_name();
                let dir_name_str = dir_name.to_string_lossy();
                if SKIP_DIRS.contains(&dir_name_str.as_ref()) {
                    debug!(dir = %dir_name_str, "skipping directory");
                    continue;
                }
                self.walk_dir(&path, out)?;
            } else if file_type.is_file() {
                // Skip files with no recognised language extension.
                if detect_language(&path).is_none() {
                    continue;
                }

                // Skip files that are too large.
                let meta = fs::metadata(&path)?;
                if meta.len() as usize > self.max_file_size {
                    debug!(path = %path.display(), size = meta.len(), "skipping large file");
                    continue;
                }

                // Skip binary files (check first 512 bytes).
                if self.is_binary(&path)? {
                    debug!(path = %path.display(), "skipping binary file");
                    continue;
                }

                out.push(path);
            }
        }

        Ok(())
    }

    /// Returns `true` if the file appears to be binary.
    fn is_binary(&self, path: &Path) -> Result<bool> {
        let mut file = fs::File::open(path)?;
        let mut buf = [0u8; 512];
        let n = file.read(&mut buf)?;
        Ok(looks_binary(&buf[..n]))
    }

    // -------------------------------------------------------------------
    // Secret scanning (code-analysis feature)
    // -------------------------------------------------------------------

    /// Scan file content for secrets and return redacted content.
    ///
    /// When the `code-analysis` feature is enabled, this uses Infiniloom's
    /// `SecurityScanner` to detect and redact secrets (API keys, tokens, etc.)
    /// before the content reaches the embedding pipeline.
    #[cfg(feature = "code-analysis")]
    pub fn redact_secrets(content: &str, file_path: &str) -> (String, SecretScanResult) {
        let scanner = SecurityScanner::new();
        let (redacted, findings) = scanner.scan_and_redact(content, file_path);

        let critical_count = findings
            .iter()
            .filter(|f| f.severity == infiniloom_engine::security::Severity::Critical)
            .count();
        let high_count = findings
            .iter()
            .filter(|f| f.severity == infiniloom_engine::security::Severity::High)
            .count();

        let result = SecretScanResult {
            findings_count: findings.len(),
            critical_count,
            high_count,
        };

        if !findings.is_empty() {
            warn!(
                file = %file_path,
                total = findings.len(),
                critical = critical_count,
                high = high_count,
                "secrets detected and redacted before embedding"
            );
        }

        (redacted, result)
    }

    // -------------------------------------------------------------------
    // Chunking
    // -------------------------------------------------------------------

    /// Naive line-based chunker (fallback when `code-analysis` is disabled).
    ///
    /// Splits a file into fixed-size chunks of `self.chunk_size` lines with
    /// `self.chunk_overlap` lines of overlap.
    fn chunk_file_naive(
        &self,
        content: &str,
        rel_path: &str,
        language: Option<String>,
    ) -> Vec<CodeChunk> {
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        if total_lines == 0 {
            return Vec::new();
        }

        // Small files: single chunk with "file" granularity.
        if total_lines <= self.chunk_size {
            return vec![CodeChunk {
                file_path: rel_path.to_string(),
                language,
                content: content.to_string(),
                start_line: 1,
                end_line: total_lines,
                granularity: "file".to_string(),
                blake3_id: None,
            }];
        }

        // Large files: overlapping chunks.
        let mut chunks = Vec::new();
        let step = self.chunk_size.saturating_sub(self.chunk_overlap).max(1);
        let mut start = 0usize;

        while start < total_lines {
            let end = (start + self.chunk_size).min(total_lines);
            let chunk_lines = &lines[start..end];

            // Prepend a header so the embedding model has file context.
            let header = format!("// File: {} (lines {}-{})", rel_path, start + 1, end);
            let mut chunk_content = String::with_capacity(
                header.len() + 1 + chunk_lines.iter().map(|l| l.len() + 1).sum::<usize>(),
            );
            chunk_content.push_str(&header);
            chunk_content.push('\n');
            for line in chunk_lines {
                chunk_content.push_str(line);
                chunk_content.push('\n');
            }

            chunks.push(CodeChunk {
                file_path: rel_path.to_string(),
                language: language.clone(),
                content: chunk_content,
                start_line: start + 1,
                end_line: end,
                granularity: "chunk".to_string(),
                blake3_id: None,
            });

            // Advance by step; if we've reached the end, stop.
            if end == total_lines {
                break;
            }
            start += step;
        }

        chunks
    }

    /// AST-aware chunker using Infiniloom's `Chunker` with `ChunkStrategy::Semantic`.
    ///
    /// Splits at semantic boundaries (function/class declarations, module-level
    /// statements) for more meaningful chunks than fixed-line splitting. Falls
    /// back to the naive chunker if the AST chunker produces no results.
    #[cfg(feature = "code-analysis")]
    fn chunk_file_with_ast(
        &self,
        content: &str,
        path: &Path,
        rel_path: &str,
        language: Option<String>,
    ) -> Vec<CodeChunk> {
        // Build a minimal Infiniloom Repository with a single file for chunking.
        let repo_name = self
            .root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "repo".to_string());

        let mut repo = Repository::new(&repo_name, &self.root);

        let file_size = content.len() as u64;
        let repo_file = RepoFile {
            path: path.to_path_buf(),
            relative_path: rel_path.to_string(),
            language: language.clone(),
            size_bytes: file_size,
            token_count: TokenCounts::default(),
            symbols: Vec::new(),
            importance: 0.5,
            content: Some(content.to_string()),
        };
        repo.files.push(repo_file);

        // Use Semantic strategy for AST-aware chunking (max 2000 tokens per chunk).
        // Semantic splits at declaration boundaries (functions, classes, modules)
        // which produces better retrieval quality than Symbol-only splitting.
        let chunker = Chunker::new(ChunkStrategy::Semantic, 2000);
        let il_chunks = chunker.chunk(&repo);

        if il_chunks.is_empty() {
            // Fallback to naive chunker if AST produces nothing.
            return self.chunk_file_naive(content, rel_path, language);
        }

        let mut result = Vec::new();
        for il_chunk in &il_chunks {
            for chunk_file in &il_chunk.files {
                let chunk_content = &chunk_file.content;
                if chunk_content.is_empty() {
                    continue;
                }

                // Determine line range from the chunk content.
                let line_count = chunk_content.lines().count();
                // Find the start line within the original content.
                let start_line =
                    if let Some(pos) = content.find(chunk_content.lines().next().unwrap_or("")) {
                        content[..pos].lines().count().max(1)
                    } else {
                        1
                    };
                let end_line = start_line + line_count.saturating_sub(1);

                // Compute BLAKE3 content-addressable ID.
                let hash = blake3::hash(chunk_content.as_bytes());
                let blake3_id = Some(hash.to_hex().to_string());

                result.push(CodeChunk {
                    file_path: chunk_file.path.clone(),
                    language: language.clone(),
                    content: chunk_content.clone(),
                    start_line,
                    end_line,
                    granularity: "semantic".to_string(),
                    blake3_id,
                });
            }
        }

        if result.is_empty() {
            // Fallback if AST chunks had no usable content.
            self.chunk_file_naive(content, rel_path, language)
        } else {
            result
        }
    }

    /// Read a file and split into chunks.
    pub fn chunk_file(&self, path: &Path) -> Result<Vec<CodeChunk>> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;

        let rel_path = path
            .strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let language = detect_language(path).map(|s| s.to_string());

        if content.is_empty() {
            return Ok(Vec::new());
        }

        // When code-analysis is enabled: redact secrets first, then use AST chunker.
        #[cfg(feature = "code-analysis")]
        {
            let (redacted_content, _scan_result) = Self::redact_secrets(&content, &rel_path);
            let mut chunks = self.chunk_file_with_ast(&redacted_content, path, &rel_path, language);

            // Ensure all chunks have BLAKE3 IDs (the AST path sets them,
            // but add them for any that might be missing).
            for chunk in &mut chunks {
                if chunk.blake3_id.is_none() {
                    let hash = blake3::hash(chunk.content.as_bytes());
                    chunk.blake3_id = Some(hash.to_hex().to_string());
                }
            }

            return Ok(chunks);
        }

        // Fallback: naive line-based chunker (no feature flag).
        #[cfg(not(feature = "code-analysis"))]
        {
            Ok(self.chunk_file_naive(&content, &rel_path, language))
        }
    }

    /// Discover files and chunk them all.
    ///
    /// When the `code-analysis` feature is enabled, uses Infiniloom's
    /// `IncrementalScanner` to skip files that haven't changed since the last
    /// run (based on mtime + size + content hash).
    pub fn chunk_all(&self) -> Result<(Vec<CodeChunk>, usize)> {
        let files = self.discover_files()?;
        let file_count = files.len();
        info!(files = file_count, root = %self.root.display(), "discovered files");

        let mut all_chunks = Vec::new();

        #[cfg(feature = "code-analysis")]
        {
            let mut scanner = IncrementalScanner::new(&self.root);
            let mut skipped = 0usize;

            for file in &files {
                // Skip files that haven't changed since last scan.
                if !scanner.needs_rescan(file) {
                    skipped += 1;
                    continue;
                }

                match self.chunk_file(file) {
                    Ok(chunks) => {
                        // Update the scanner cache for this file.
                        if let Ok(metadata) = std::fs::metadata(file) {
                            let mtime = metadata
                                .modified()
                                .ok()
                                .and_then(|t| {
                                    t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok()
                                })
                                .map_or(0, |d| d.as_secs());

                            let rel_path = file
                                .strip_prefix(&self.root)
                                .unwrap_or(file)
                                .to_string_lossy()
                                .to_string();

                            scanner.update(infiniloom_engine::incremental::CachedFile {
                                path: rel_path,
                                mtime,
                                size: metadata.len(),
                                hash: 0, // Could compute BLAKE3 hash here for extra accuracy
                                tokens: TokenCounts::default(),
                                symbols: vec![],
                                symbols_extracted: false,
                                language: detect_language(file).map(|s| s.to_string()),
                                lines: chunks
                                    .last()
                                    .map(|c| c.end_line)
                                    .unwrap_or(0),
                            });
                        }

                        all_chunks.extend(chunks);
                    }
                    Err(e) => {
                        warn!(path = %file.display(), error = %e, "failed to chunk file");
                    }
                }
            }

            // Persist the cache for next run.
            if let Err(e) = scanner.save() {
                warn!(error = %e, "failed to save incremental scanner cache");
            }

            if skipped > 0 {
                info!(
                    skipped,
                    changed = file_count - skipped,
                    "incremental scan: skipped unchanged files"
                );
            }
        }

        #[cfg(not(feature = "code-analysis"))]
        {
            for file in &files {
                match self.chunk_file(file) {
                    Ok(chunks) => all_chunks.extend(chunks),
                    Err(e) => {
                        warn!(path = %file.display(), error = %e, "failed to chunk file");
                    }
                }
            }
        }

        info!(
            chunks = all_chunks.len(),
            files = file_count,
            "chunking complete"
        );
        Ok((all_chunks, file_count))
    }

    // -------------------------------------------------------------------
    // Embed + store
    // -------------------------------------------------------------------

    /// Embed and store all chunks using the given provider and LanceDB store.
    pub async fn embed_and_store<P: crate::embedding::EmbedProvider>(
        &self,
        provider: &P,
        lance_store: &crate::store::LanceStore,
        batch_size: usize,
    ) -> Result<EmbedResult> {
        let (chunks, file_count) = self.chunk_all()?;
        let total_chunks = chunks.len();

        let mut result = EmbedResult {
            files_found: file_count,
            chunks_created: total_chunks,
            ..Default::default()
        };

        if chunks.is_empty() {
            return Ok(result);
        }

        // Collect text references for batch embedding.
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();

        // Embed in batches.
        let batch_size = batch_size.max(1);
        let n_batches = texts.len().div_ceil(batch_size);
        info!(total_chunks, batch_size, n_batches, "embedding chunks");

        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(total_chunks);

        for (i, text_batch) in texts.chunks(batch_size).enumerate() {
            debug!(batch = i + 1, size = text_batch.len(), "embedding batch");
            match provider.embed_texts(text_batch).await {
                Ok(vectors) => {
                    all_vectors.extend(vectors);
                }
                Err(e) => {
                    warn!(batch = i + 1, error = %e, "batch embedding failed");
                    result.errors += text_batch.len();
                    // Push empty vectors as placeholders so indices stay aligned.
                    for _ in 0..text_batch.len() {
                        all_vectors.push(Vec::new());
                    }
                }
            }
        }

        // Store each chunk with its embedding.
        for (idx, chunk) in chunks.iter().enumerate() {
            let embedding = &all_vectors[idx];
            if embedding.is_empty() {
                // Embedding failed for this chunk.
                continue;
            }

            // Use BLAKE3 ID when available, otherwise fall back to positional ID.
            let id = chunk.blake3_id.clone().unwrap_or_else(|| {
                format!(
                    "{}:{}:{}:{}",
                    self.project_id, chunk.file_path, chunk.start_line, chunk.end_line
                )
            });

            match lance_store
                .insert_code_embedding(
                    &id,
                    embedding,
                    &chunk.content,
                    &chunk.granularity,
                    &self.project_id,
                    Some(&chunk.file_path),
                    chunk.language.as_deref(),
                )
                .await
            {
                Ok(()) => {
                    result.chunks_embedded += 1;
                }
                Err(e) => {
                    warn!(
                        id = %id,
                        error = %e,
                        "failed to store chunk embedding"
                    );
                    result.errors += 1;
                }
            }
        }

        info!(
            embedded = result.chunks_embedded,
            errors = result.errors,
            "embed-and-store complete"
        );
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Path::new("main.rs")), Some("rust"));
        assert_eq!(detect_language(Path::new("app.py")), Some("python"));
        assert_eq!(detect_language(Path::new("index.js")), Some("javascript"));
        assert_eq!(detect_language(Path::new("lib.ts")), Some("typescript"));
        assert_eq!(detect_language(Path::new("main.go")), Some("go"));
        assert_eq!(detect_language(Path::new("App.java")), Some("java"));
        assert_eq!(detect_language(Path::new("foo.c")), Some("c"));
        assert_eq!(detect_language(Path::new("bar.h")), Some("c"));
        assert_eq!(detect_language(Path::new("baz.cpp")), Some("cpp"));
        assert_eq!(detect_language(Path::new("x.rb")), Some("ruby"));
        assert_eq!(detect_language(Path::new("y.swift")), Some("swift"));
        assert_eq!(detect_language(Path::new("z.zig")), Some("zig"));
        assert_eq!(detect_language(Path::new("config.yaml")), Some("yaml"));
        assert_eq!(detect_language(Path::new("config.yml")), Some("yaml"));
        assert_eq!(detect_language(Path::new("Cargo.toml")), Some("toml"));
        assert_eq!(detect_language(Path::new("data.json")), Some("json"));
        assert_eq!(detect_language(Path::new("README.md")), Some("markdown"));
        // Unknown extension returns None.
        assert_eq!(detect_language(Path::new("image.png")), None);
        assert_eq!(detect_language(Path::new("no_extension")), None);
    }

    #[test]
    fn test_discover_files_skips_ignored_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Create some source files.
        fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("lib.py"), "print('hi')").unwrap();

        // Create directories that should be skipped.
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::write(root.join(".git/config"), "gitconfig").unwrap();

        fs::create_dir_all(root.join("node_modules/foo")).unwrap();
        fs::write(
            root.join("node_modules/foo/index.js"),
            "module.exports = {}",
        )
        .unwrap();

        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::write(root.join("target/debug/main.rs"), "// build artifact").unwrap();

        fs::create_dir_all(root.join("__pycache__")).unwrap();
        fs::write(root.join("__pycache__/mod.py"), "cached").unwrap();

        // Create a nested source file that *should* be found.
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/util.rs"), "pub fn util() {}").unwrap();

        let embedder = RepoEmbedder::new(root);
        let files = embedder.discover_files().unwrap();

        let rel_paths: Vec<String> = files
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().to_string())
            .collect();

        assert!(rel_paths.contains(&"main.rs".to_string()));
        assert!(rel_paths.contains(&"lib.py".to_string()));
        assert!(rel_paths.contains(&"src/util.rs".to_string()));

        // Ignored directories should not appear.
        for p in &rel_paths {
            assert!(!p.starts_with(".git"), "should skip .git: {p}");
            assert!(
                !p.starts_with("node_modules"),
                "should skip node_modules: {p}"
            );
            assert!(!p.starts_with("target"), "should skip target: {p}");
            assert!(
                !p.starts_with("__pycache__"),
                "should skip __pycache__: {p}"
            );
        }
    }

    #[test]
    fn test_chunk_small_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // A small file with 10 lines (well under the default chunk_size of 50).
        let content: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        fs::write(root.join("small.rs"), &content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file(&root.join("small.rs")).unwrap();

        assert_eq!(chunks.len(), 1, "small file should produce exactly 1 chunk");
        // With code-analysis, AST chunker may produce "symbol" granularity.
        #[cfg(not(feature = "code-analysis"))]
        assert_eq!(chunks[0].granularity, "file");
        assert_eq!(chunks[0].start_line, 1);
        #[cfg(not(feature = "code-analysis"))]
        assert_eq!(chunks[0].end_line, 10);
        assert_eq!(chunks[0].file_path, "small.rs");
        assert_eq!(chunks[0].language.as_deref(), Some("rust"));
    }

    #[test]
    #[cfg(not(feature = "code-analysis"))]
    fn test_chunk_large_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // A file with 120 lines (larger than the default chunk_size of 50).
        let content: String = (1..=120).map(|i| format!("line {i}\n")).collect();
        fs::write(root.join("large.py"), &content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file(&root.join("large.py")).unwrap();

        // With chunk_size=50, overlap=10, step=40:
        // Chunk 1: lines 1-50
        // Chunk 2: lines 41-90
        // Chunk 3: lines 81-120
        assert!(
            chunks.len() >= 3,
            "expected at least 3 chunks, got {}",
            chunks.len()
        );

        // All chunks should have "chunk" granularity.
        for chunk in &chunks {
            assert_eq!(chunk.granularity, "chunk");
            assert_eq!(chunk.language.as_deref(), Some("python"));
        }

        // First chunk should start at line 1.
        assert_eq!(chunks[0].start_line, 1);

        // Last chunk should end at line 120.
        let last = chunks.last().unwrap();
        assert_eq!(last.end_line, 120);

        // Chunks should have the header with file path.
        assert!(
            chunks[0].content.starts_with("// File: large.py"),
            "chunk should start with file path header"
        );

        // Verify overlap: second chunk should start before first chunk ends.
        assert!(
            chunks[1].start_line < chunks[0].end_line,
            "chunks should overlap"
        );
    }

    #[test]
    fn test_empty_file_produces_no_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        fs::write(root.join("empty.rs"), "").unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file(&root.join("empty.rs")).unwrap();

        assert!(chunks.is_empty(), "empty file should produce 0 chunks");
    }

    #[test]
    fn test_binary_file_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // Write a file with null bytes (binary).
        let mut data = b"fn main() {}\0\0\0binary data".to_vec();
        data.extend_from_slice(&[0u8; 100]);
        // Give it a .rs extension so it would be detected by language.
        fs::write(root.join("binary.rs"), &data).unwrap();

        // Also create a valid text file.
        fs::write(root.join("valid.rs"), "fn main() {}").unwrap();

        let embedder = RepoEmbedder::new(root);
        let files = embedder.discover_files().unwrap();

        let rel_paths: Vec<String> = files
            .iter()
            .map(|p| p.strip_prefix(root).unwrap().to_string_lossy().to_string())
            .collect();

        assert!(rel_paths.contains(&"valid.rs".to_string()));
        assert!(
            !rel_paths.contains(&"binary.rs".to_string()),
            "binary file should be skipped"
        );
    }

    #[test]
    fn test_with_project_id() {
        let tmp = tempfile::tempdir().unwrap();
        let embedder = RepoEmbedder::new(tmp.path()).with_project_id("my-project");
        assert_eq!(embedder.project_id(), "my-project");
    }

    #[test]
    fn test_blake3_id_is_none_without_feature() {
        // Without the code-analysis feature, blake3_id should be None.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let content: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        fs::write(root.join("test.rs"), &content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file(&root.join("test.rs")).unwrap();

        // Without code-analysis, blake3_id should be None.
        #[cfg(not(feature = "code-analysis"))]
        {
            assert!(chunks[0].blake3_id.is_none());
        }

        // With code-analysis, blake3_id should be Some.
        #[cfg(feature = "code-analysis")]
        {
            assert!(chunks[0].blake3_id.is_some());
            // BLAKE3 hex is 64 characters.
            assert_eq!(chunks[0].blake3_id.as_ref().unwrap().len(), 64);
        }
    }

    #[test]
    fn test_naive_chunker_directly() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let embedder = RepoEmbedder::new(root);

        // Test empty content.
        let chunks = embedder.chunk_file_naive("", "empty.rs", Some("rust".to_string()));
        assert!(chunks.is_empty());

        // Test small content.
        let content = "line 1\nline 2\nline 3\n";
        let chunks = embedder.chunk_file_naive(content, "small.rs", Some("rust".to_string()));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].granularity, "file");
        assert!(chunks[0].blake3_id.is_none());

        // Test large content (>50 lines).
        let large: String = (1..=120).map(|i| format!("line {i}\n")).collect();
        let chunks = embedder.chunk_file_naive(&large, "large.py", Some("python".to_string()));
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert_eq!(chunk.granularity, "chunk");
        }
    }

    #[test]
    fn test_secret_scan_result_default() {
        let result = SecretScanResult::default();
        assert_eq!(result.findings_count, 0);
        assert_eq!(result.critical_count, 0);
        assert_eq!(result.high_count, 0);
    }
}

// ---------------------------------------------------------------------------
// Tests requiring code-analysis feature
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "code-analysis")]
mod code_analysis_tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_redact_secrets_detects_aws_key() {
        let content = r#"const AWS_KEY = "AKIAIOSFODNN7PRODKEY";"#;
        let (redacted, result) = RepoEmbedder::redact_secrets(content, "config.js");

        assert!(result.findings_count > 0, "should detect AWS key");
        assert!(result.critical_count > 0, "AWS key should be critical");
        assert!(
            !redacted.contains("AKIAIOSFODNN7PRODKEY"),
            "secret should be redacted from content"
        );
    }

    #[test]
    fn test_redact_secrets_clean_file() {
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        let (redacted, result) = RepoEmbedder::redact_secrets(content, "main.rs");

        assert_eq!(result.findings_count, 0);
        assert_eq!(redacted, content, "clean file should be unchanged");
    }

    #[test]
    fn test_chunk_file_with_ast_produces_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let content =
            "fn foo() {\n    println!(\"foo\");\n}\n\nfn bar() {\n    println!(\"bar\");\n}\n";
        fs::write(root.join("funcs.rs"), content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file_with_ast(
            content,
            &root.join("funcs.rs"),
            "funcs.rs",
            Some("rust".to_string()),
        );

        assert!(!chunks.is_empty(), "AST chunker should produce chunks");
        for chunk in &chunks {
            assert!(
                chunk.blake3_id.is_some(),
                "AST chunks should have BLAKE3 IDs"
            );
            assert_eq!(chunk.blake3_id.as_ref().unwrap().len(), 64);
        }
    }

    #[test]
    fn test_chunk_file_with_ast_fallback() {
        // An empty file should fall back gracefully.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let content = "// just a comment\n";
        fs::write(root.join("comment.rs"), content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file_with_ast(
            content,
            &root.join("comment.rs"),
            "comment.rs",
            Some("rust".to_string()),
        );

        // Should produce at least one chunk (either from AST or fallback).
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_file_integration_with_secrets() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // File containing a secret.
        let content = "fn config() {\n    let key = \"AKIAIOSFODNN7PRODKEY\";\n}\n";
        fs::write(root.join("secret.rs"), content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks = embedder.chunk_file(&root.join("secret.rs")).unwrap();

        assert!(!chunks.is_empty());
        // The secret should be redacted in all chunks.
        for chunk in &chunks {
            assert!(
                !chunk.content.contains("AKIAIOSFODNN7PRODKEY"),
                "secret should be redacted in chunk content"
            );
            assert!(chunk.blake3_id.is_some(), "chunks should have BLAKE3 IDs");
        }
    }

    #[test]
    fn test_blake3_id_is_content_addressable() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let content = "fn hello() {}\n";
        fs::write(root.join("a.rs"), content).unwrap();
        fs::write(root.join("b.rs"), content).unwrap();

        let embedder = RepoEmbedder::new(root);
        let chunks_a = embedder.chunk_file(&root.join("a.rs")).unwrap();
        let chunks_b = embedder.chunk_file(&root.join("b.rs")).unwrap();

        // Same content should produce the same BLAKE3 hash.
        assert_eq!(
            chunks_a[0].blake3_id, chunks_b[0].blake3_id,
            "identical content should produce identical BLAKE3 IDs"
        );
    }
}
