pub mod config;
pub mod consolidate;
pub mod context;
pub mod detect;
pub mod distill;
pub mod embed_pipeline;
pub mod embedding;
pub mod graph_builder;
pub mod hybrid_search;
pub mod ingest;
pub mod pipeline;
pub mod progressive;
pub mod repo_embed;
pub mod semantic_scorer;
pub mod semantic_tree;
pub mod store;
pub mod watcher;
pub mod xpath_query;

#[cfg(feature = "code-analysis")]
pub mod code_analysis;

pub use config::AppConfig;
pub use consolidate::{
    ConsolidationStats, DecayScore, MergeCandidate, compute_decay_scores, consolidate,
};
pub use context::{AgentContext, ContextAssembler, PressureLevel};
pub use detect::{AgentDetection, AgentInfo, detect_agents};
pub use distill::{DistilledSession, Distiller, LlmClient};
pub use embed_pipeline::{EmbedChunk, EmbedPipeline, EmbedStats, Granularity};
pub use embedding::{EmbedProvider, LmStudioEmbedder, MockEmbedder, batch_embed};
pub use graph_builder::{GraphBackend, GraphBuilder, InMemoryGraphBuilder};
pub use hybrid_search::{
    HybridResult, HybridSearch, HybridWeights, QueryComplexity, ResultType, classify_query,
    is_xpath_query,
};
pub use ingest::{ClaudeIngester, CodexIngester, GeminiIngester, IngestResult};
pub use pipeline::IngestPipeline;
pub use progressive::{
    DetailEntry, DisclosureLevel, IndexEntry, ProgressiveRetriever, SummaryEntry,
};
pub use repo_embed::{CodeChunk, EmbedResult as RepoEmbedResult, RepoEmbedder};
pub use semantic_scorer::{
    BenchmarkResult, SemanticScorer, benchmark_xpath_vs_flat, cosine_similarity, keyword_scorer,
};
pub use semantic_tree::{TreeBuilder, TreeNode, TreeNodeType, TreeSchema};
pub use watcher::FileWatcher;
pub use xpath_query::{
    AggOp, Axis, CompOp, NodeSelect, ParseError, Predicate, QueryStep, WeightedNode, XPathQuery,
    evaluate as evaluate_xpath, parse as parse_xpath,
};

#[cfg(feature = "code-analysis")]
pub use code_analysis::{AnalysisResult, CodeAnalyzer};
