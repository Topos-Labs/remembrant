use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub agents: AgentsConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub distillation: DistillationConfig,
    #[serde(default)]
    pub retention: RetentionConfig,
    #[serde(default)]
    pub cross_project: CrossProjectConfig,
    #[serde(default)]
    pub watch: WatchConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            agents: AgentsConfig::default(),
            storage: StorageConfig::default(),
            embedding: EmbeddingConfig::default(),
            distillation: DistillationConfig::default(),
            retention: RetentionConfig::default(),
            cross_project: CrossProjectConfig::default(),
            watch: WatchConfig::default(),
        }
    }
}

impl AppConfig {
    /// Return the Remembrant configuration directory (`~/.remembrant/`).
    pub fn config_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("could not determine home directory")?;
        Ok(home.join(".remembrant"))
    }

    /// Path to the config file (`~/.remembrant/config.yaml`).
    fn config_path() -> Result<PathBuf> {
        Ok(Self::config_dir()?.join("config.yaml"))
    }

    /// Load configuration from `~/.remembrant/config.yaml`.
    ///
    /// If the file does not exist a default configuration is written first,
    /// then returned.
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;

        if !path.exists() {
            let config = Self::default();
            config.save().context("failed to write default config")?;
            return Ok(config);
        }

        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self = serde_yaml::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(config)
    }

    /// Persist the current configuration to `~/.remembrant/config.yaml`.
    pub fn save(&self) -> Result<()> {
        let dir = Self::config_dir()?;
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;

        let path = Self::config_path()?;
        let yaml = serde_yaml::to_string(self).context("failed to serialize config")?;
        fs::write(&path, yaml).with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------

/// Configuration for supported AI coding agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentsConfig {
    #[serde(default)]
    pub claude_code: AgentEntry,
    #[serde(default)]
    pub codex: AgentEntry,
    #[serde(default)]
    pub gemini: AgentEntry,
}

impl Default for AgentsConfig {
    fn default() -> Self {
        Self {
            claude_code: AgentEntry::new(true, "~/.claude"),
            codex: AgentEntry::new(true, "~/.codex"),
            gemini: AgentEntry::new(true, "~/.gemini"),
        }
    }
}

/// A single agent entry (enabled flag + transcript directory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEntry {
    pub enabled: bool,
    pub path: String,
}

impl AgentEntry {
    fn new(enabled: bool, path: &str) -> Self {
        Self {
            enabled,
            path: path.to_string(),
        }
    }
}

impl Default for AgentEntry {
    fn default() -> Self {
        Self {
            enabled: false,
            path: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Paths for persistent stores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub base_dir: String,
    pub duckdb_path: String,
    pub lancedb_path: String,
    pub graph_db_path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_dir: "~/.remembrant".to_string(),
            duckdb_path: "~/.remembrant/remembrant.duckdb".to_string(),
            lancedb_path: "~/.remembrant/lancedb".to_string(),
            graph_db_path: "~/.remembrant/graph".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Configuration for the embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub api_key_env: String,
    pub batch_size: usize,
    pub dimensions: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "text-embedding-nomic-embed-text-v1.5@q8_0".to_string(),
            api_key_env: String::new(),
            batch_size: 100,
            dimensions: 768,
        }
    }
}

// ---------------------------------------------------------------------------
// Distillation
// ---------------------------------------------------------------------------

/// How aggressively to distil ingested transcripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistillationLevel {
    None,
    Minimal,
    Balanced,
    Aggressive,
    Full,
}

impl Default for DistillationLevel {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Distillation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    #[serde(default)]
    pub level: DistillationLevel,
    pub llm_provider: String,
    pub llm_model: String,
    pub api_key_env: String,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            level: DistillationLevel::default(),
            llm_provider: "http://localhost:1234/v1".to_string(),
            llm_model: "qwen/qwen3-30b-a3b".to_string(),
            api_key_env: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Retention
// ---------------------------------------------------------------------------

/// Data retention policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    pub raw_transcripts_days: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            raw_transcripts_days: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-project
// ---------------------------------------------------------------------------

/// Default visibility for cross-project memory items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    Public,
    Private,
}

impl Default for Visibility {
    fn default() -> Self {
        Self::Public
    }
}

/// Cross-project sharing settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossProjectConfig {
    pub enabled: bool,
    #[serde(default)]
    pub default_visibility: Visibility,
}

impl Default for CrossProjectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_visibility: Visibility::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Watch
// ---------------------------------------------------------------------------

/// File-watcher settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig {
    pub debounce_ms: u64,
    pub auto_start: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 5000,
            auto_start: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_round_trips_through_yaml() {
        let config = AppConfig::default();
        let yaml = serde_yaml::to_string(&config).expect("serialize");
        let parsed: AppConfig = serde_yaml::from_str(&yaml).expect("deserialize");

        assert_eq!(parsed.embedding.model, "text-embedding-nomic-embed-text-v1.5@q8_0");
        assert_eq!(parsed.embedding.batch_size, 100);
        assert_eq!(parsed.embedding.dimensions, 768);
        assert_eq!(parsed.distillation.level, DistillationLevel::Balanced);
        assert_eq!(parsed.retention.raw_transcripts_days, 30);
        assert!(parsed.cross_project.enabled);
        assert_eq!(parsed.cross_project.default_visibility, Visibility::Public);
        assert_eq!(parsed.watch.debounce_ms, 5000);
        assert!(!parsed.watch.auto_start);
        assert!(parsed.agents.claude_code.enabled);
        assert_eq!(parsed.agents.claude_code.path, "~/.claude");
    }

    #[test]
    fn partial_yaml_fills_defaults() {
        let yaml = "retention:\n  raw_transcripts_days: 90\n";
        let config: AppConfig = serde_yaml::from_str(yaml).expect("deserialize");

        // Overridden field
        assert_eq!(config.retention.raw_transcripts_days, 90);
        // Everything else should be default
        assert_eq!(config.embedding.dimensions, 768);
        assert_eq!(config.distillation.level, DistillationLevel::Balanced);
    }

    #[test]
    fn config_dir_is_under_home() {
        let dir = AppConfig::config_dir().expect("config_dir");
        assert!(dir.ends_with(".remembrant"));
    }
}
