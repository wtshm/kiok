use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Memory visibility scope for a project or chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Global,
    Project,
    Isolated,
}

impl Scope {
    /// Return the canonical string representation of the scope.
    pub fn as_str(&self) -> &'static str {
        match self {
            Scope::Global => "global",
            Scope::Project => "project",
            Scope::Isolated => "isolated",
        }
    }
}

impl Default for Scope {
    fn default() -> Self {
        Scope::Global
    }
}

// ---------------------------------------------------------------------------
// Internal deserialisation helper
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PolicyFile {
    scope: Scope,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read the memory policy for `project_path`.
///
/// Looks for `<project_path>/.claude/memory-policy.json`.  If the file does
/// not exist, returns [`Scope::Global`].
pub fn load_scope(project_path: &str) -> Result<Scope> {
    let policy_path = Path::new(project_path)
        .join(".claude")
        .join("memory-policy.json");

    if !policy_path.exists() {
        return Ok(Scope::Global);
    }

    let content = std::fs::read_to_string(&policy_path)?;
    let pf: PolicyFile = serde_json::from_str(&content)?;
    Ok(pf.scope)
}

/// Determine whether a chunk is visible to a given viewer.
///
/// Rules:
/// - A chunk from the **same project** as the viewer is always visible.
/// - A viewer with `Isolated` scope can only see its own project's chunks.
/// - A viewer with `Project` or `Global` scope can also see chunks from
///   other projects if those chunks have a `"global"` scope.
pub fn is_visible(
    chunk_scope: &str,
    chunk_project: &str,
    viewer_scope: Scope,
    viewer_project: &str,
) -> bool {
    // Same project is always visible regardless of scope.
    if chunk_project == viewer_project {
        return true;
    }

    // Cross-project visibility.
    match viewer_scope {
        // Isolated viewers cannot see any other project.
        Scope::Isolated => false,
        // Project and Global viewers can see other projects' global chunks.
        Scope::Project | Scope::Global => chunk_scope == "global",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_default_scope_is_global() {
        assert_eq!(Scope::default(), Scope::Global);
    }

    #[test]
    fn test_same_project_always_visible() {
        // Same project, isolated viewer — still visible.
        assert!(is_visible("isolated", "my-project", Scope::Isolated, "my-project"));
        // Same project, global viewer.
        assert!(is_visible("global", "my-project", Scope::Global, "my-project"));
    }

    #[test]
    fn test_isolated_blocks_cross_project() {
        assert!(!is_visible("global", "other-project", Scope::Isolated, "my-project"));
        assert!(!is_visible("project", "other-project", Scope::Isolated, "my-project"));
        assert!(!is_visible("isolated", "other-project", Scope::Isolated, "my-project"));
    }

    #[test]
    fn test_project_scope_sees_global() {
        // Cross-project, global chunk — visible to project-scoped viewer.
        assert!(is_visible("global", "other-project", Scope::Project, "my-project"));
        // Cross-project, non-global chunk — not visible.
        assert!(!is_visible("project", "other-project", Scope::Project, "my-project"));
    }

    #[test]
    fn test_global_scope_sees_global() {
        assert!(is_visible("global", "other-project", Scope::Global, "my-project"));
        assert!(!is_visible("isolated", "other-project", Scope::Global, "my-project"));
    }

    #[test]
    fn test_parse_policy_json() {
        let tmp = TempDir::new().expect("tempdir failed");
        let claude_dir = tmp.path().join(".claude");
        std::fs::create_dir_all(&claude_dir).expect("mkdir failed");

        let policy_path = claude_dir.join("memory-policy.json");
        let mut f = std::fs::File::create(&policy_path).expect("create failed");
        writeln!(f, r#"{{"scope": "isolated"}}"#).expect("write failed");

        let scope = load_scope(tmp.path().to_str().unwrap()).expect("load_scope failed");
        assert_eq!(scope, Scope::Isolated);
    }
}
