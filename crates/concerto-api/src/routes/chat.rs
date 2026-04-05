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

use crate::error::ApiError;
use crate::metrics::REQUESTS_TOTAL;
use crate::orchestrator::{route_and_dispatch, RoutingKind};
use crate::types::ChatCompletionRequest;
use crate::AppState;

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let outcome = completions_inner(&state, req).await;
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
        let stream = upstream.bytes_stream().map(|chunk| {
            chunk.map_err(|e| std::io::Error::other(format!("upstream stream error: {e}")))
        });
        let body = Body::from_stream(stream);
        let mut response = Response::builder()
            .status(status_code)
            .body(body)
            .map_err(|e| ApiError::Internal(format!("building stream response: {e}")))?;
        *response.headers_mut() = headers;
        Ok((response, target.kind))
    } else {
        // Non-streaming: read the whole body into memory and forward.
        let bytes = upstream
            .bytes()
            .await
            .map_err(|e| ApiError::BackendCrashed(format!("reading upstream body: {e}")))?;
        let mut response = (status_code, bytes).into_response();
        let response_headers = response.headers_mut();
        for (k, v) in headers.iter() {
            response_headers.insert(k, v.clone());
        }
        Ok((response, target.kind))
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
