//! Cross-cutting axum middleware applied to specific route subsets.
//!
//! Currently exposes a single middleware: [`request_timeout`], applied to
//! `POST /v1/chat/completions` only and gated on
//! `routing.request_timeout_secs > 0` per ROADMAP §8.
//!
//! ## Streaming exemption
//!
//! `tokio::time::timeout` wraps the *response future* — i.e. the time until
//! the handler returns a [`Response`]. For streaming completions the handler
//! resolves quickly with a `Body::from_stream(...)` whose body is sent
//! incrementally; only the time-to-first-byte is bounded by this middleware.
//! That matches the design intent: streaming is by nature long-running and
//! must not be killed by a per-request wall-clock budget. Documented in
//! `docs/troubleshooting.md`.

use std::time::Duration;

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::app::AppState;
use crate::error::ApiError;

/// Wrap the inner handler in `tokio::time::timeout` using
/// `state.config.routing.request_timeout_secs`. A value of `0` disables the
/// timeout (the middleware becomes a no-op).
pub async fn request_timeout(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let secs = state.config.routing.request_timeout_secs;
    if secs == 0 {
        return next.run(req).await;
    }
    match tokio::time::timeout(Duration::from_secs(secs), next.run(req)).await {
        Ok(resp) => resp,
        Err(_) => ApiError::RequestTimeout.into_response(),
    }
}
