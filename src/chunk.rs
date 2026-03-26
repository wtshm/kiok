use anyhow::{Context, Result};
use serde_json::Value;

use crate::filter;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A Q&A pair extracted from a conversation session.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub session_id: String,
    pub uuid: Option<String>,
    pub question: String,
    pub answer: String,
    pub timestamp: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A single conversation turn (user or assistant) after filtering and
/// content extraction.
struct Message {
    role: String,
    content: String,
    uuid: Option<String>,
    timestamp: Option<String>,
    session_id: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a JSONL session file and return a list of Q&A chunks.
///
/// This is the main entry point used by the save command.
pub fn parse_session(jsonl_content: &str) -> Result<Vec<Chunk>> {
    let messages = extract_messages(jsonl_content)?;
    Ok(pair_into_chunks(messages))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Read each non-empty line of `jsonl_content` as a JSON object,
/// filter by record type, extract the conversation content, and return
/// a `Message` for every line that passes the filter.
fn extract_messages(jsonl_content: &str) -> Result<Vec<Message>> {
    let mut messages = Vec::new();

    for (line_no, line) in jsonl_content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let record: Value = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse JSON at line {}", line_no + 1))?;

        // Filter by record type.
        let record_type = record["type"].as_str().unwrap_or("");
        if !filter::is_relevant_record_type(record_type) {
            continue;
        }

        // Extract metadata.
        let uuid = record["uuid"].as_str().map(str::to_owned);
        let timestamp = record["timestamp"].as_str().map(str::to_owned);
        let session_id = record["sessionId"]
            .as_str()
            .unwrap_or("")
            .to_owned();

        // Extract content from the nested message object.
        let message_obj = &record["message"];
        let role = message_obj["role"].as_str().unwrap_or("").to_owned();

        let raw_content = extract_content(message_obj)
            .with_context(|| format!("Failed to extract content at line {}", line_no + 1))?;

        // Strip system tags and skip blank messages.
        let content = filter::strip_system_tags(&raw_content);
        if filter::is_empty_content(&content) {
            continue;
        }

        messages.push(Message {
            role,
            content,
            uuid,
            timestamp,
            session_id,
        });
    }

    Ok(messages)
}

/// Extract a plain-text representation of a message's `content` field.
///
/// - User messages: `content` is a plain string.
/// - Assistant messages: `content` is an array of typed blocks.
///   - `text` blocks → include the text.
///   - `tool_use` blocks → include only if `filter::is_preserved_tool`.
///   - `thinking`, `tool_result`, and other block types → skip.
fn extract_content(message: &Value) -> Result<String> {
    let content = &message["content"];

    // User messages carry content as a plain string.
    if let Some(s) = content.as_str() {
        return Ok(s.to_owned());
    }

    // Assistant messages carry content as an array of blocks.
    if let Some(blocks) = content.as_array() {
        let mut parts: Vec<String> = Vec::new();

        for block in blocks {
            let block_type = block["type"].as_str().unwrap_or("");
            match block_type {
                "text" => {
                    if let Some(text) = block["text"].as_str() {
                        parts.push(text.to_owned());
                    }
                }
                "tool_use" => {
                    let name = block["name"].as_str().unwrap_or("");
                    if filter::is_preserved_tool(name) {
                        parts.push(format!("[tool_use: {}]", name));
                    }
                }
                // Skip "thinking", "tool_result", and any other block types.
                _ => {}
            }
        }

        return Ok(parts.join("\n"));
    }

    Ok(String::new())
}

/// Pair consecutive user/assistant messages into Q&A chunks.
///
/// A "user" message becomes the `question`.  All subsequent "assistant"
/// messages (until the next "user" message) are joined with a newline and
/// become the `answer`.  The chunk's metadata (uuid, timestamp, session_id)
/// is taken from the first assistant message in each group.
fn pair_into_chunks(messages: Vec<Message>) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();

    // Accumulated assistant content for the current question.
    let mut answer_parts: Vec<String> = Vec::new();
    let mut answer_meta: Option<(Option<String>, Option<String>, String)> = None; // (uuid, timestamp, session_id)

    // We need to keep track of the question message so we can flush it when
    // the next user message arrives or at the end.  Store indices into the
    // original `messages` slice.
    let mut pending_question: Option<usize> = None;

    for (i, msg) in messages.iter().enumerate() {
        if msg.role == "user" {
            // Flush the previous Q&A pair if there is one.
            if let Some(qi) = pending_question {
                if !answer_parts.is_empty() {
                    let qm = &messages[qi];
                    let (uuid, timestamp, session_id) =
                        answer_meta.take().unwrap_or((None, None, qm.session_id.clone()));
                    chunks.push(Chunk {
                        session_id,
                        uuid,
                        question: qm.content.clone(),
                        answer: answer_parts.join("\n"),
                        timestamp,
                    });
                    answer_parts.clear();
                }
            }
            pending_question = Some(i);
        } else if msg.role == "assistant" {
            if pending_question.is_some() {
                // Record metadata from the first assistant message in this group.
                if answer_parts.is_empty() {
                    answer_meta = Some((
                        msg.uuid.clone(),
                        msg.timestamp.clone(),
                        msg.session_id.clone(),
                    ));
                }
                if !filter::is_empty_content(&msg.content) {
                    answer_parts.push(msg.content.clone());
                }
            }
        }
    }

    // Flush the last pending Q&A pair.
    if let Some(qi) = pending_question {
        if !answer_parts.is_empty() {
            let qm = &messages[qi];
            let (uuid, timestamp, session_id) =
                answer_meta.take().unwrap_or((None, None, qm.session_id.clone()));
            chunks.push(Chunk {
                session_id,
                uuid,
                question: qm.content.clone(),
                answer: answer_parts.join("\n"),
                timestamp,
            });
        }
    }

    chunks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Load the sample fixture and parse it.
    fn load_fixture() -> Vec<Chunk> {
        let content = include_str!("../tests/fixtures/sample_session.jsonl");
        parse_session(content).expect("parse_session failed")
    }

    #[test]
    fn test_parse_session_from_fixture() {
        let chunks = load_fixture();
        assert_eq!(chunks.len(), 2, "expected exactly 2 Q&A chunks");

        // First chunk: about Docker setup.
        assert!(
            chunks[0].question.contains("Docker"),
            "first question should be about Docker"
        );
        assert!(
            chunks[0].answer.contains("docker-compose"),
            "first answer should mention docker-compose"
        );

        // Second chunk: about build cache.
        assert!(
            chunks[1].question.contains("ビルドキャッシュ"),
            "second question should be about build cache"
        );
        assert!(
            chunks[1].answer.contains("BuildKit"),
            "second answer should mention BuildKit"
        );
    }

    #[test]
    fn test_system_tags_stripped() {
        let chunks = load_fixture();
        for chunk in &chunks {
            assert!(
                !chunk.answer.contains("system-reminder"),
                "answer must not contain 'system-reminder'"
            );
            assert!(
                !chunk.answer.contains("This is hidden"),
                "answer must not contain hidden tag content"
            );
            assert!(
                !chunk.question.contains("system-reminder"),
                "question must not contain 'system-reminder'"
            );
        }
    }

    #[test]
    fn test_tool_result_excluded() {
        let chunks = load_fixture();
        for chunk in &chunks {
            assert!(
                !chunk.answer.contains("file content here"),
                "tool_result content must not appear in any chunk"
            );
        }
    }

    #[test]
    fn test_read_tool_excluded() {
        let chunks = load_fixture();
        for chunk in &chunks {
            assert!(
                !chunk.answer.contains("[tool_use: Read]"),
                "Read tool_use must not appear in any chunk"
            );
        }
    }

    #[test]
    fn test_pair_chunks_handles_multiple_assistant_messages() {
        // Q1 → A1_part1 + A1_part2 (joined), Q2 → A2
        let content = r#"{"type":"user","message":{"role":"user","content":"Q1"},"timestamp":"2026-01-01T00:00:00Z","sessionId":"s1","uuid":"u1"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"A1_part1"}]},"timestamp":"2026-01-01T00:00:01Z","sessionId":"s1","uuid":"a1"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"A1_part2"}]},"timestamp":"2026-01-01T00:00:02Z","sessionId":"s1","uuid":"a2"}
{"type":"user","message":{"role":"user","content":"Q2"},"timestamp":"2026-01-01T00:00:03Z","sessionId":"s1","uuid":"u2"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"A2"}]},"timestamp":"2026-01-01T00:00:04Z","sessionId":"s1","uuid":"a3"}
"#;
        let chunks = parse_session(content).expect("parse_session failed");
        assert_eq!(chunks.len(), 2);

        // First chunk should have both parts joined.
        assert_eq!(chunks[0].question, "Q1");
        assert!(
            chunks[0].answer.contains("A1_part1"),
            "first answer must contain A1_part1"
        );
        assert!(
            chunks[0].answer.contains("A1_part2"),
            "first answer must contain A1_part2"
        );

        // Second chunk.
        assert_eq!(chunks[1].question, "Q2");
        assert_eq!(chunks[1].answer, "A2");
    }
}
