//! Canned response generators used by the mock backend.
//!
//! Keeping these in one place makes it easy to update the fixture bodies
//! without touching routing or streaming logic.

use serde_json::{json, Value};

/// Fixed text returned by every non-streaming chat completion.
pub const CANNED_BODY: &str = "This is a mock response from a fake inference backend.";

/// The tokens we pretend to generate when streaming.
///
/// A deliberately small, fixed set so tests can rely on the chunk count.
const TOKENS: &[&str] = &[
    "This ",
    "is ",
    "a ",
    "mock ",
    "streaming ",
    "response ",
    "from ",
    "a ",
    "fake ",
    "backend.",
];

/// A realistic, fully-formed OpenAI chat completion response.
pub fn canned_chat_response(model: &str) -> Value {
    json!({
        "id": "chatcmpl-mock-0001",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": CANNED_BODY,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 12,
            "total_tokens": 20,
        }
    })
}

/// SSE event payload strings for a streamed chat completion.
///
/// Each element is a complete `data:` line body (without the trailing
/// double-newline). The final `[DONE]` marker is appended by the server.
pub fn streaming_chunks(model: &str) -> Vec<String> {
    const CREATED: u64 = 1_700_000_000;
    const ID: &str = "chatcmpl-mock-stream-0001";

    let chunk = |delta: Value, finish_reason: Option<&str>| {
        json!({
            "id": ID,
            "object": "chat.completion.chunk",
            "created": CREATED,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        })
        .to_string()
    };

    let mut chunks = Vec::with_capacity(TOKENS.len() + 2);
    chunks.push(chunk(json!({ "role": "assistant", "content": "" }), None));
    for token in TOKENS {
        chunks.push(chunk(json!({ "content": token }), None));
    }
    chunks.push(chunk(json!({}), Some("stop")));
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canned_response_contains_required_fields() {
        let value = canned_chat_response("qwen2.5-7b");
        assert_eq!(value["model"], "qwen2.5-7b");
        assert_eq!(value["object"], "chat.completion");
        assert!(value["choices"].is_array());
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert!(value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("mock"));
    }

    #[test]
    fn streaming_chunk_count_is_reasonable() {
        let chunks = streaming_chunks("phi-3-mini");
        // role + 10 tokens + stop = 12. CLAUDE.md asks for ~8-12.
        assert!((8..=14).contains(&chunks.len()));
    }
}
