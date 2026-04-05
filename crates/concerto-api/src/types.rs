//! OpenAI-compatible request/response types used at the HTTP boundary.
//!
//! These are intentionally minimal — just enough shape for non-streaming and
//! streaming `/v1/chat/completions` and the `/v1/models` listing. Any extra
//! OpenAI fields (temperature, max_tokens, tools, etc.) are captured by
//! `#[serde(flatten)]` and forwarded verbatim to the upstream backend.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    /// Any other OpenAI fields (temperature, max_tokens, tools, …) are captured
    /// here and forwarded to the backend verbatim.
    #[serde(flatten)]
    pub extra: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Value,
}

/// Single entry in the `/v1/models` listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Top-level payload returned by `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsList {
    pub object: String,
    pub data: Vec<Model>,
}
