use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

/// Trait for embedding providers (LM Studio, OpenAI, Voyage, etc.)
pub trait EmbedProvider: Send + Sync {
    /// Embed a batch of texts, returning one vector per input text.
    fn embed_texts(
        &self,
        texts: &[&str],
    ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send;

    /// The dimensionality of the embedding vectors produced by this provider.
    fn dimensions(&self) -> usize;

    /// The model name used by this provider.
    fn model_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// LM Studio API types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    encoding_format: &'a str,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDataItem>,
}

#[derive(Debug, Deserialize)]
struct EmbedDataItem {
    embedding: Vec<f32>,
    index: usize,
}

// ---------------------------------------------------------------------------
// LmStudioEmbedder
// ---------------------------------------------------------------------------

/// LM Studio embedding client (OpenAI-compatible local API).
pub struct LmStudioEmbedder {
    base_url: String,
    model: String,
    dimensions: usize,
    client: reqwest::Client,
}

impl LmStudioEmbedder {
    /// Create a new embedder with the default LM Studio base URL
    /// (`http://localhost:1234/v1`).
    pub fn new(model: &str, dimensions: usize) -> Self {
        Self::with_base_url("http://localhost:1234/v1", model, dimensions)
    }

    /// Create an embedder pointing at a custom base URL.
    pub fn with_base_url(base_url: &str, model: &str, dimensions: usize) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dimensions,
            client: reqwest::Client::new(),
        }
    }

    /// Build an embedder from the application's [`EmbeddingConfig`](crate::config::EmbeddingConfig).
    pub fn from_config(config: &crate::config::EmbeddingConfig) -> Self {
        Self::new(&config.model, config.dimensions)
    }
}

impl EmbedProvider for LmStudioEmbedder {
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url);
        debug!(
            model = %self.model,
            count = texts.len(),
            "requesting embeddings from LM Studio"
        );

        let body = EmbedRequest {
            model: &self.model,
            input: texts,
            encoding_format: "float",
        };

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| {
                format!(
                    "failed to connect to LM Studio at {url} \
                     -- is LM Studio running?"
                )
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            anyhow::bail!("LM Studio returned HTTP {status}: {body_text}");
        }

        let mut embed_resp: EmbedResponse = response
            .json()
            .await
            .context("failed to parse LM Studio embedding response")?;

        // Sort by index to guarantee order matches input order.
        embed_resp.data.sort_by_key(|item| item.index);

        let vectors: Vec<Vec<f32>> = embed_resp
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect();

        if vectors.len() != texts.len() {
            warn!(
                expected = texts.len(),
                got = vectors.len(),
                "LM Studio returned a different number of embeddings than requested"
            );
        }

        Ok(vectors)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// MockEmbedder
// ---------------------------------------------------------------------------

/// Zero-vector embedder for testing (no LM Studio required).
pub struct MockEmbedder {
    dimensions: usize,
}

impl MockEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl EmbedProvider for MockEmbedder {
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.dimensions]).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        "mock"
    }
}

// ---------------------------------------------------------------------------
// Batch helper
// ---------------------------------------------------------------------------

/// Splits `texts` into batches of `batch_size` and embeds them all,
/// returning the concatenated result vectors in order.
pub async fn batch_embed<P: EmbedProvider>(
    provider: &P,
    texts: &[&str],
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = batch_size.max(1);
    let n_batches = (texts.len() + batch_size - 1) / batch_size;
    info!(
        total = texts.len(),
        batch_size, n_batches, "batch embedding texts"
    );

    let mut all_vectors = Vec::with_capacity(texts.len());

    for (i, chunk) in texts.chunks(batch_size).enumerate() {
        debug!(batch = i + 1, size = chunk.len(), "embedding batch");
        let vectors = provider
            .embed_texts(chunk)
            .await
            .with_context(|| format!("batch {}/{n_batches} failed", i + 1))?;
        all_vectors.extend(vectors);
    }

    Ok(all_vectors)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedder_returns_correct_dimensions() {
        let embedder = MockEmbedder::new(768);
        let texts: Vec<&str> = vec!["hello world", "foo bar baz"];
        let vectors = embedder.embed_texts(&texts).await.unwrap();

        assert_eq!(vectors.len(), 2);
        for v in &vectors {
            assert_eq!(v.len(), 768);
            assert!(v.iter().all(|&x| x == 0.0));
        }
        assert_eq!(embedder.dimensions(), 768);
        assert_eq!(embedder.model_name(), "mock");
    }

    #[tokio::test]
    async fn test_batch_embed_splits_correctly() {
        let embedder = MockEmbedder::new(4);
        let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
        let vectors = batch_embed(&embedder, &texts, 2).await.unwrap();

        // 5 texts, batch_size=2 => 3 batches (2+2+1). All results present.
        assert_eq!(vectors.len(), 5);
        for v in &vectors {
            assert_eq!(v.len(), 4);
        }
    }

    #[tokio::test]
    async fn test_batch_embed_empty_input() {
        let embedder = MockEmbedder::new(4);
        let texts: Vec<&str> = vec![];
        let vectors = batch_embed(&embedder, &texts, 10).await.unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn test_from_config() {
        let config = crate::config::EmbeddingConfig {
            model: "test-model".to_string(),
            api_key_env: String::new(),
            batch_size: 50,
            dimensions: 384,
        };
        let embedder = LmStudioEmbedder::from_config(&config);
        assert_eq!(embedder.model_name(), "test-model");
        assert_eq!(embedder.dimensions(), 384);
        assert_eq!(embedder.base_url, "http://localhost:1234/v1");
    }
}
