/// Returns true if the record type should be included in processing.
/// Only "user" and "assistant" records carry conversation content.
pub fn is_relevant_record_type(record_type: &str) -> bool {
    matches!(record_type, "user" | "assistant")
}

/// Returns true if a tool_use block should be preserved in the output.
/// Read-only / informational tools are excluded because they add noise
/// without conveying the model's reasoning or actions.
pub fn is_preserved_tool(tool_name: &str) -> bool {
    !matches!(
        tool_name,
        "Read" | "Glob" | "Grep" | "LSP" | "ToolSearch" | "TaskGet" | "TaskOutput" | "TaskList"
    )
}

/// Remove XML-style system tag blocks from `content`.
///
/// The following tags (and everything between them) are stripped:
/// - system-reminder
/// - local-command-caveat
/// - local-command-stdout
/// - command-name
/// - command-message
/// - command-args
/// - task-notification
///
/// Tags may span multiple lines.
pub fn strip_system_tags(content: &str) -> String {
    const TAGS: &[&str] = &[
        "system-reminder",
        "local-command-caveat",
        "local-command-stdout",
        "command-name",
        "command-message",
        "command-args",
        "task-notification",
    ];

    let mut result = content.to_owned();
    for tag in TAGS {
        result = remove_tag(&result, tag);
    }
    result
}

/// Remove all occurrences of `<tag>...</tag>` (including multiline) from `s`.
fn remove_tag(s: &str, tag: &str) -> String {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let mut out = String::with_capacity(s.len());
    let mut rest = s;

    loop {
        match rest.find(open.as_str()) {
            None => {
                out.push_str(rest);
                break;
            }
            Some(start) => {
                out.push_str(&rest[..start]);
                let after_open = &rest[start + open.len()..];
                match after_open.find(close.as_str()) {
                    None => {
                        // Malformed: no closing tag — leave the rest as-is.
                        out.push_str(after_open);
                        break;
                    }
                    Some(end) => {
                        rest = &after_open[end + close.len()..];
                    }
                }
            }
        }
    }

    out
}

/// Returns true if `content` is empty or contains only whitespace.
pub fn is_empty_content(content: &str) -> bool {
    content.trim().is_empty()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevant_record_types() {
        assert!(is_relevant_record_type("user"));
        assert!(is_relevant_record_type("assistant"));
        assert!(!is_relevant_record_type("progress"));
        assert!(!is_relevant_record_type("queue-operation"));
        assert!(!is_relevant_record_type("system"));
        assert!(!is_relevant_record_type(""));
    }

    #[test]
    fn test_preserved_tools() {
        // Tools that should be preserved (write / action tools).
        assert!(is_preserved_tool("Edit"));
        assert!(is_preserved_tool("Write"));
        assert!(is_preserved_tool("Bash"));
        assert!(is_preserved_tool("Agent"));

        // Read-only tools that should be excluded.
        assert!(!is_preserved_tool("Read"));
        assert!(!is_preserved_tool("Glob"));
        assert!(!is_preserved_tool("Grep"));
        assert!(!is_preserved_tool("LSP"));
        assert!(!is_preserved_tool("ToolSearch"));
        assert!(!is_preserved_tool("TaskGet"));
        assert!(!is_preserved_tool("TaskOutput"));
        assert!(!is_preserved_tool("TaskList"));
    }

    #[test]
    fn test_strip_system_tags() {
        let input = "Hello <system-reminder>hidden stuff</system-reminder> World";
        let expected = "Hello  World";
        assert_eq!(strip_system_tags(input), expected);
    }

    #[test]
    fn test_strip_multiple_tags() {
        let input = "A <system-reminder>secret</system-reminder> B <local-command-caveat>note</local-command-caveat> C";
        let expected = "A  B  C";
        assert_eq!(strip_system_tags(input), expected);
    }

    #[test]
    fn test_strip_nested_content() {
        let input = "Before\n<system-reminder>\nLine 1\nLine 2\nLine 3\n</system-reminder>\nAfter";
        let expected = "Before\n\nAfter";
        assert_eq!(strip_system_tags(input), expected);
    }

    #[test]
    fn test_empty_content() {
        assert!(is_empty_content(""));
        assert!(is_empty_content("   "));
        assert!(is_empty_content("\t\n"));
        assert!(!is_empty_content("hello"));
        assert!(!is_empty_content("  x  "));
    }
}
