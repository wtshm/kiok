//! End-to-end tests that run the `kiok` binary as a subprocess.
//!
//! Each test gets its own temp directory used as `$HOME`, so the
//! database, model paths, and Claude project directories are fully
//! isolated from the real environment.
//!
//! Project paths are also placed inside the temp directory so that
//! policy files (read from `<project>/.claude/memory-policy.json`)
//! don't leak onto the real filesystem.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn kiok_bin() -> PathBuf {
    let mut path = std::env::current_exe()
        .expect("current_exe failed")
        .parent()
        .expect("no parent")
        .parent()
        .expect("no grandparent")
        .to_path_buf();
    path.push("kiok");
    path
}

fn run(args: &[&str], home: &Path) -> Output {
    Command::new(kiok_bin())
        .args(args)
        .env("HOME", home)
        .output()
        .expect("failed to execute kiok")
}

fn encode_path(path: &str) -> String {
    path.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sample_session.jsonl")
}

/// Set up a fake Claude project directory with a JSONL session file.
/// The project path is placed inside the temp directory for full isolation.
fn setup_project(home: &Path, name: &str, session_id: &str) -> String {
    // Use a path inside the tempdir so policy files are also isolated.
    let project_path = home.join("projects").join(name);
    fs::create_dir_all(&project_path).expect("create project dir failed");
    let project_str = project_path.to_str().expect("non-utf8 path").to_owned();

    let encoded = encode_path(&project_str);
    let session_dir = home.join(".claude").join("projects").join(&encoded);
    fs::create_dir_all(&session_dir).expect("create session dir failed");

    let content = fs::read_to_string(fixture_path()).expect("read fixture failed");
    let suffix = session_id.replace("test-session-", "");

    let modified = content
        .replace("test-session-1", session_id)
        .replace("\"u1\"", &format!("\"u1{}\"", suffix))
        .replace("\"u2\"", &format!("\"u2{}\"", suffix))
        .replace("\"a1\"", &format!("\"a1{}\"", suffix))
        .replace("\"a2\"", &format!("\"a2{}\"", suffix))
        .replace("\"a3\"", &format!("\"a3{}\"", suffix))
        .replace("\"a4\"", &format!("\"a4{}\"", suffix))
        .replace("\"p1\"", &format!("\"p1{}\"", suffix));

    fs::write(
        session_dir.join(format!("{}.jsonl", session_id)),
        modified,
    )
    .expect("write fixture failed");

    project_str
}

fn set_policy(project_path: &str, scope: &str) {
    let policy_dir = Path::new(project_path).join(".claude");
    fs::create_dir_all(&policy_dir).expect("create policy dir failed");
    fs::write(
        policy_dir.join("memory-policy.json"),
        format!("{{\"scope\":\"{}\"}}", scope),
    )
    .expect("write policy failed");
}

fn db_path(home: &Path) -> PathBuf {
    home.join(".kiok").join("memory.db")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_save_then_keyword_recall() {
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let home = tmp.path();
    let project = setup_project(home, "test-project", "test-session-1");

    let out = run(&["save", "--project", &project], home);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "save failed: {}", stderr);
    assert!(stderr.contains("save: session="), "should print save summary");
    assert!(db_path(home).exists(), "database should be created");

    let out = run(&["recall", "Docker", "--project", &project, "--count", "5"], home);
    assert!(out.status.success(), "recall failed: {}", String::from_utf8_lossy(&out.stderr));

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("<kiok>"), "output should contain <kiok> tag");
    assert!(stdout.contains("</kiok>"), "output should contain closing tag");
    assert!(
        stdout.contains("Docker") || stdout.contains("docker"),
        "output should contain Docker-related result, got:\n{}",
        stdout
    );
}

#[test]
fn test_save_idempotency() {
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let home = tmp.path();
    let project = setup_project(home, "idempotent-project", "test-session-1");

    let out1 = run(&["save", "--project", &project], home);
    assert!(out1.status.success());

    let out2 = run(&["save", "--project", &project], home);
    assert!(out2.status.success());
    let stderr2 = String::from_utf8_lossy(&out2.stderr);
    assert!(
        stderr2.contains("already saved"),
        "second save should report session already saved, got: {}",
        stderr2
    );
}

#[test]
fn test_recall_empty_db_is_silent() {
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let home = tmp.path();
    let fake_project = home.join("projects").join("no-project");
    fs::create_dir_all(&fake_project).unwrap();

    let out = run(
        &["recall", "anything", "--project", fake_project.to_str().unwrap(), "--count", "5"],
        home,
    );
    assert!(out.status.success(), "recall should succeed silently with no DB");
    assert!(out.stdout.is_empty(), "stdout should be empty when no DB exists");
}

#[test]
fn test_stats_after_save() {
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let home = tmp.path();
    let project = setup_project(home, "stats-project", "test-session-1");

    run(&["save", "--project", &project], home);

    let out = run(&["stats"], home);
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Sessions:   1"), "should show 1 session, got: {}", stdout);
    assert!(stdout.contains("Chunks:     2"), "should show 2 chunks, got: {}", stdout);
    assert!(stdout.contains("Embeddings: 0 / 2"), "should show 0/2 embeddings, got: {}", stdout);
}

#[test]
fn test_recall_no_match_is_silent() {
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let home = tmp.path();
    let project = setup_project(home, "nomatch-project", "test-session-1");

    run(&["save", "--project", &project], home);

    let out = run(
        &["recall", "xyzzy_nonexistent_term", "--project", &project, "--count", "5"],
        home,
    );
    assert!(out.status.success());
    assert!(out.stdout.is_empty(), "no-match recall should produce empty stdout");
}

/// Global-scope projects can see each other's results.
#[test]
fn test_global_scope_sees_multiple_projects() {
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let home = tmp.path();

    let project_a = setup_project(home, "project-a", "test-session-a");
    let out_a = run(&["save", "--project", &project_a], home);
    assert!(out_a.status.success(), "save A failed: {}", String::from_utf8_lossy(&out_a.stderr));

    let project_b = setup_project(home, "project-b", "test-session-b");
    let out_b = run(&["save", "--project", &project_b], home);
    assert!(out_b.status.success(), "save B failed: {}", String::from_utf8_lossy(&out_b.stderr));

    let out = run(
        &["recall", "Docker", "--project", &project_a, "--count", "10"],
        home,
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains("project-a") && stdout.contains("project-b"),
        "global scope should see both projects, got:\n{}",
        stdout
    );
}

/// Isolated-scope project cannot see other projects' chunks.
#[test]
fn test_isolated_scope_blocks_cross_project() {
    let tmp = tempfile::tempdir().expect("tempdir failed");
    let home = tmp.path();

    let project_a = setup_project(home, "project-a", "test-session-a");
    run(&["save", "--project", &project_a], home);

    let project_b = setup_project(home, "project-b", "test-session-b");
    run(&["save", "--project", &project_b], home);

    set_policy(&project_a, "isolated");

    // Isolated viewer should only see own project's chunks.
    let out = run(
        &["recall", "Docker", "--project", &project_a, "--count", "10"],
        home,
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(stdout.contains("project-a"), "should see own project");
    assert!(
        !stdout.contains("project-b"),
        "isolated scope should NOT see project-b, got:\n{}",
        stdout
    );
}
