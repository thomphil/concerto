//! `POST /v1/chat/completions` — deserialise, orchestrate, forward.
//!
//! The body is parsed as a [`ChatCompletionRequest`] but forwarded to the
//! selected backend as raw JSON (via the `extra: serde_json::Value` capture)
//! so any upstream OpenAI fields we don't explicitly model are preserved.
//!
//! For streaming requests (`stream: true`), the upstream `reqwest::Response`
//! byte stream is forwarded verbatim into the axum response — no SSE
//! re-parsing. This is T5 in ROADMAP §5.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use concerto_core::ModelId;
use futures::StreamExt;
use metrics::counter;
use serde_json::Value;
use tokio::sync::Notify;

use crate::app::InFlightGuard;
use crate::error::ApiError;
use crate::metrics::REQUESTS_TOTAL;
use crate::orchestrator::{route_and_dispatch, RoutingKind};
use crate::types::ChatCompletionRequest;
use crate::AppState;

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    // Track this request for the graceful-shutdown drain. The guard moves
    // into completions_inner so that streaming bodies can transfer it into
    // the body stream and decrement only when the stream completes.
    let guard = InFlightGuard::new(state.in_flight.clone());
    let outcome = completions_inner(&state, req, guard).await;
    let decision_label = match &outcome {
        Ok((_, kind)) => kind.label(),
        Err(e) => decision_label_for_error(e),
    };
    counter!(REQUESTS_TOTAL, "decision" => decision_label).increment(1);
    outcome.map(|(resp, _)| resp)
}

/// Inner handler body. Separated from [`completions`] so the outer
/// function owns the metric emission regardless of which exit path the
/// request takes (including `?`-propagation of upstream errors).
async fn completions_inner(
    state: &AppState,
    req: ChatCompletionRequest,
    guard: InFlightGuard,
) -> Result<(Response, RoutingKind), ApiError> {
    let model_id = ModelId(req.model.clone());
    tracing::info!(model = %model_id, stream = req.stream, "incoming chat completion");

    // Decide where the request should go. This is the orchestrator state
    // machine — may launch a new backend, subscribe to an in-flight load,
    // or evict a stale model.
    let target = route_and_dispatch(state, model_id.clone()).await?;

    // Rebuild the JSON body exactly as the client sent it (model + messages
    // + any flattened extras).
    let forwarded_body = upstream_body(&req)?;

    let upstream_url = format!("http://127.0.0.1:{}/v1/chat/completions", target.port);
    let client = reqwest::Client::new();
    let upstream = client
        .post(&upstream_url)
        .json(&forwarded_body)
        .send()
        .await
        .map_err(|e| {
            ApiError::BackendCrashed(format!(
                "failed to reach backend on port {}: {e}",
                target.port
            ))
        })?;

    let status_code =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    // Preserve content-type so the client can tell streaming from JSON.
    let mut headers = HeaderMap::new();
    if let Some(ct) = upstream.headers().get(reqwest::header::CONTENT_TYPE) {
        if let Ok(v) = HeaderValue::from_bytes(ct.as_bytes()) {
            headers.insert(HeaderName::from_static("content-type"), v);
        }
    }

    if req.stream {
        // Byte-forward the stream. Do NOT re-parse SSE events — the upstream
        // backend already produces them in the correct shape (including the
        // `data: [DONE]` terminator) and we are a transparent proxy.
        //
        // Two enhancements over the trivial proxy: (1) the [`InFlightGuard`]
        // is moved into the stream so it drops only when the body finishes
        // or is cancelled, keeping `state.in_flight` accurate during
        // graceful shutdown; (2) `state.shutdown` is selected against
        // each chunk so SIGTERM mid-stream emits a final `data: [DONE]`
        // and closes cleanly rather than blocking forever.
        let upstream_stream = upstream.bytes_stream();
        let body = Body::from_stream(shutdown_aware_stream(
            upstream_stream,
            state.shutdown.clone(),
            guard,
        ));
        let mut response = Response::builder()
            .status(status_code)
            .body(body)
            .map_err(|e| ApiError::Internal(format!("building stream response: {e}")))?;
        *response.headers_mut() = headers;
        Ok((response, target.kind))
    } else {
        // Non-streaming: read the whole body into memory and forward.
        // The [`InFlightGuard`] held by this function drops on return,
        // decrementing the in-flight counter once the body has been read.
        let bytes = upstream
            .bytes()
            .await
            .map_err(|e| ApiError::BackendCrashed(format!("reading upstream body: {e}")))?;
        let mut response = (status_code, bytes).into_response();
        let response_headers = response.headers_mut();
        for (k, v) in headers.iter() {
            response_headers.insert(k, v.clone());
        }
        drop(guard);
        Ok((response, target.kind))
    }
}

/// Wrap an upstream byte stream so:
///
/// 1. The supplied [`InFlightGuard`] drops when the body finishes or the
///    consumer cancels — keeping `AppState::in_flight` accurate during the
///    shutdown drain.
/// 2. A `state.shutdown` notification short-circuits the proxy with a final
///    `data: [DONE]\n\n` event, so SIGTERM mid-stream returns to the client
///    cleanly instead of hanging until the connection is force-closed.
fn shutdown_aware_stream<S>(
    mut upstream: S,
    shutdown: std::sync::Arc<Notify>,
    guard: InFlightGuard,
) -> impl futures::Stream<Item = std::result::Result<bytes::Bytes, std::io::Error>>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>>
        + Unpin
        + Send
        + 'static,
{
    async_stream::stream! {
        // Hold the guard for the lifetime of this generator so it drops
        // on completion *or* cancellation. `_`-prefix avoids the unused-
        // binding warning.
        let _guard = guard;

        loop {
            let notified = shutdown.notified();
            tokio::pin!(notified);
            tokio::select! {
                chunk = upstream.next() => match chunk {
                    Some(Ok(bytes)) => yield Ok(bytes),
                    Some(Err(e)) => {
                        yield Err(std::io::Error::other(
                            format!("upstream stream error: {e}"),
                        ));
                        break;
                    }
                    None => break,
                },
                _ = &mut notified => {
                    tracing::info!(
                        "shutdown notified during streaming response; \
                         emitting final data: [DONE] and closing"
                    );
                    yield Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n"));
                    break;
                }
            }
        }
    }
}

/// Map an [`ApiError`] into the `decision` label used by
/// `concerto_requests_total`. The label set is deliberately small so
/// dashboards can reason about it without having to enumerate every
/// internal error variant.
fn decision_label_for_error(e: &ApiError) -> &'static str {
    match e {
        ApiError::AllGpusUnhealthy => "rejected_all_unhealthy",
        ApiError::BackendUnavailable(_) => "rejected_backend_unavailable",
        _ => "error",
    }
}

/// Reassemble a JSON body out of a [`ChatCompletionRequest`] that preserves
/// every extra field. We use the `#[serde(flatten)]` capture directly as the
/// object map if it happens to be an object; otherwise we construct a fresh
/// object with `model` + `messages` + `stream`.
fn upstream_body(req: &ChatCompletionRequest) -> Result<Value, ApiError> {
    let mut body = match &req.extra {
        Value::Object(map) => Value::Object(map.clone()),
        Value::Null => Value::Object(Default::default()),
        _ => {
            return Err(ApiError::BadRequest(
                "request body must be a JSON object".into(),
            ));
        }
    };
    if let Value::Object(map) = &mut body {
        map.insert("model".to_string(), Value::String(req.model.clone()));
        map.insert("messages".to_string(), serde_json::to_value(&req.messages)?);
        map.insert("stream".to_string(), Value::Bool(req.stream));
    }
    Ok(body)
}
