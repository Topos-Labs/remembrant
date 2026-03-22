use std::path::PathBuf;

/// Information about a detected coding agent installation.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Human-readable agent name (e.g. "Claude Code").
    pub name: String,
    /// Root directory of the agent's local data.
    pub base_path: PathBuf,
    /// Approximate count of session files found.
    pub session_count: usize,
    /// Whether a configuration file was found.
    pub config_found: bool,
}

/// Results of scanning the filesystem for known coding agents.
#[derive(Debug, Clone, Default)]
pub struct AgentDetection {
    pub claude_code: Option<AgentInfo>,
    pub codex: Option<AgentInfo>,
    pub gemini: Option<AgentInfo>,
}

/// Scan the filesystem for installed coding agents and return detection results.
///
/// Checks the following locations under the user's home directory:
/// - Claude Code: `~/.claude/` with `projects/` subdirectory
/// - Codex CLI:   `~/.codex/`  with `sessions/` and `config.toml`
/// - Gemini CLI:  `~/.gemini/` with `tmp/` and `projects.json`
pub fn detect_agents() -> AgentDetection {
    let Some(home) = home_dir() else {
        return AgentDetection::default();
    };

    AgentDetection {
        claude_code: detect_claude_code(&home),
        codex: detect_codex(&home),
        gemini: detect_gemini(&home),
    }
}

/// Resolve the user's home directory.
fn home_dir() -> Option<PathBuf> {
    // Prefer $HOME; fall back to std::env::home_dir (deprecated but still useful).
    std::env::var_os("HOME").map(PathBuf::from).or_else(|| {
        #[allow(deprecated)]
        std::env::home_dir()
    })
}

// ---------------------------------------------------------------------------
// Claude Code
// ---------------------------------------------------------------------------

/// Detect Claude Code by looking for `~/.claude/` with a `projects/` subdirectory.
/// Session count: number of entries in `~/.claude/projects/*/`.
fn detect_claude_code(home: &PathBuf) -> Option<AgentInfo> {
    let base = home.join(".claude");
    if !base.is_dir() {
        return None;
    }

    let projects_dir = base.join("projects");
    let has_projects = projects_dir.is_dir();

    let session_count = if has_projects {
        count_entries(&projects_dir)
    } else {
        0
    };

    Some(AgentInfo {
        name: "Claude Code".to_string(),
        base_path: base,
        session_count,
        config_found: has_projects,
    })
}

// ---------------------------------------------------------------------------
// Codex CLI
// ---------------------------------------------------------------------------

/// Detect Codex CLI by looking for `~/.codex/` with `sessions/` and `config.toml`.
/// Session count: number of directories in `~/.codex/sessions/`.
fn detect_codex(home: &PathBuf) -> Option<AgentInfo> {
    let base = home.join(".codex");
    if !base.is_dir() {
        return None;
    }

    let sessions_dir = base.join("sessions");
    let config_path = base.join("config.toml");

    let session_count = if sessions_dir.is_dir() {
        count_dirs(&sessions_dir)
    } else {
        0
    };

    Some(AgentInfo {
        name: "Codex CLI".to_string(),
        base_path: base,
        session_count,
        config_found: config_path.is_file(),
    })
}

// ---------------------------------------------------------------------------
// Gemini CLI
// ---------------------------------------------------------------------------

/// Detect Gemini CLI by looking for `~/.gemini/` with `tmp/` and `projects.json`.
/// Session count: number of JSON files across `~/.gemini/tmp/*/chats/`.
fn detect_gemini(home: &PathBuf) -> Option<AgentInfo> {
    let base = home.join(".gemini");
    if !base.is_dir() {
        return None;
    }

    let tmp_dir = base.join("tmp");
    let projects_json = base.join("projects.json");

    let session_count = if tmp_dir.is_dir() {
        count_gemini_chats(&tmp_dir)
    } else {
        0
    };

    Some(AgentInfo {
        name: "Gemini CLI".to_string(),
        base_path: base,
        session_count,
        config_found: projects_json.is_file(),
    })
}

/// Count JSON files inside `tmp/*/chats/` directories.
fn count_gemini_chats(tmp_dir: &PathBuf) -> usize {
    let Ok(entries) = std::fs::read_dir(tmp_dir) else {
        return 0;
    };

    let mut total = 0;
    for entry in entries.flatten() {
        let chats_dir = entry.path().join("chats");
        if chats_dir.is_dir() {
            if let Ok(chat_files) = std::fs::read_dir(&chats_dir) {
                total += chat_files
                    .flatten()
                    .filter(|f| f.path().extension().map_or(false, |ext| ext == "json"))
                    .count();
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count all entries (files and directories) directly inside `dir`.
fn count_entries(dir: &PathBuf) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| rd.flatten().count())
        .unwrap_or(0)
}

/// Count only sub-directories directly inside `dir`.
fn count_dirs(dir: &PathBuf) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_type().map_or(false, |ft| ft.is_dir()))
                .count()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_agents_returns_without_panic() {
        // Smoke test: must not panic regardless of what's on disk.
        let detection = detect_agents();
        // At minimum the struct should be constructable.
        let _ = format!("{detection:?}");
    }

    #[test]
    fn default_detection_is_all_none() {
        let d = AgentDetection::default();
        assert!(d.claude_code.is_none());
        assert!(d.codex.is_none());
        assert!(d.gemini.is_none());
    }
}
