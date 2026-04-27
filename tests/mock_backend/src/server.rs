//! HTTP server implementation for the mock inference backend.

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::stream::Stream;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::responses::{canned_chat_response, streaming_chunks};

/// CLI arguments for the mock backend binary.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "mock-inference-backend",
    about = "A tiny HTTP server that pretends to be a real inference engine."
)]
pub struct Args {
    /// Port to bind on.
    #[arg(long, default_value_t = 18080)]
    pub port: u16,

    /// Host to bind on.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Seconds to sleep before binding — simulates model loading time.
    #[arg(long, default_value_t = 0)]
    pub startup_delay_secs: u64,

    /// Artificial latency added before the response is produced, in
    /// milliseconds. For streaming requests this delays the response
    /// *headers* (and therefore time-to-first-byte); for non-streaming
    /// requests it delays the entire response.
    #[arg(long, default_value_t = 10)]
    pub response_latency_ms: u64,

    /// Per-chunk delay applied to streaming responses (ms between SSE
    /// events). Lets a test exercise long-running streaming bodies that
    /// outlive `request_timeout_secs` without delaying headers.
    #[arg(long, default_value_t = 2)]
    pub stream_chunk_delay_ms: u64,

    /// Probability of returning a 500 error for a given request (0.0 - 1.0).
    #[arg(long, default_value_t = 0.0)]
    pub fail_probability: f64,

    /// If set, the process exits with status 1 after serving this many requests.
    #[arg(long)]
    pub crash_after: Option<usize>,
}

/// Shared application state available to all handlers.
#[derive(Debug, Clone)]
pub struct AppState {
    pub request_counter: Arc<AtomicUsize>,
    pub response_latency_ms: u64,
    pub stream_chunk_delay_ms: u64,
    pub fail_probability: f64,
    pub crash_after: Option<usize>,
}

impl AppState {
    pub fn from_args(args: &Args) -> Self {
        Self {
            request_counter: Arc::new(AtomicUsize::new(0)),
            response_latency_ms: args.response_latency_ms,
            stream_chunk_delay_ms: args.stream_chunk_delay_ms,
            fail_probability: args.fail_probability,
            crash_after: args.crash_after,
        }
    }
}

/// Build an axum [`Router`] with all mock endpoints wired in.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

/// Run the server until the process is killed.
pub async fn run(args: Args) -> Result<()> {
    if args.startup_delay_secs > 0 {
        tracing::info!(
            delay_secs = args.startup_delay_secs,
            "simulating model load startup delay"
        );
        tokio::time::sleep(Duration::from_secs(args.startup_delay_secs)).await;
    }

    let state = AppState::from_args(&args);
    let app = build_router(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;

    tracing::info!(%addr, "mock inference backend listening");
    axum::serve(listener, app)
        .await
        .context("axum server exited with error")?;

    Ok(())
}

/// Liveness probe endpoint.
async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

/// Prometheus-compatible metrics endpoint with a few fake counters.
async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let requests = state.request_counter.load(Ordering::Relaxed);
    let body = format!(
        "# HELP mock_backend_requests_total Total requests served by the mock backend.\n\
         # TYPE mock_backend_requests_total counter\n\
         mock_backend_requests_total {requests}\n\
         # HELP mock_backend_memory_bytes Fake VRAM usage reported by the mock backend.\n\
         # TYPE mock_backend_memory_bytes gauge\n\
         mock_backend_memory_bytes 8589934592\n\
         # HELP mock_backend_up 1 if the mock backend is running.\n\
         # TYPE mock_backend_up gauge\n\
         mock_backend_up 1\n"
    );

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
}

/// Minimal subset of the OpenAI chat completion request used by the mock.
#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    stream: bool,
}

fn default_model() -> String {
    "mock-model".to_string()
}

/// Handler for `POST /v1/chat/completions`.
async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    // Increment the request counter up front so crash-after / metrics are
    // consistent across the streaming and non-streaming branches.
    let count = state.request_counter.fetch_add(1, Ordering::Relaxed) + 1;

    if let Some(limit) = state.crash_after {
        if count >= limit {
            tracing::warn!(count, "crash-after threshold reached, exiting");
            std::process::exit(1);
        }
    }

    if state.response_latency_ms > 0 {
        tokio::time::sleep(Duration::from_millis(state.response_latency_ms)).await;
    }

    if state.fail_probability > 0.0 && rand::random::<f64>() < state.fail_probability {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "simulated failure" })),
        ));
    }

    if payload.stream {
        Ok(streaming_response(payload.model, state.stream_chunk_delay_ms).into_response())
    } else {
        Ok(Json(canned_chat_response(&payload.model)).into_response())
    }
}

/// Build an SSE response containing the canned streaming chunks plus `[DONE]`.
fn streaming_response(
    model: String,
    chunk_delay_ms: u64,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        for chunk in streaming_chunks(&model) {
            yield Ok(Event::default().data(chunk));
            // Per-chunk delay (default 2ms) so clients see distinct events;
            // tests can crank this up to exercise long-running bodies.
            tokio::time::sleep(Duration::from_millis(chunk_delay_ms)).await;
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}
