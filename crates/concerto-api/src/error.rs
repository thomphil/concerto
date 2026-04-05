//! The canonical [`ApiError`] type returned by every HTTP handler.
//!
//! Every downstream crate's error type has a `From` impl here, so handlers can
//! simply `?`-propagate and get a sensible HTTP response shape out the other
//! side. The axum `IntoResponse` impl picks an appropriate status code and
//! serialises an [`ErrorBody`] JSON payload.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use concerto_backend::BackendError;
use concerto_config::ConfigError;
use concerto_core::{CoreError, ModelId};
use concerto_gpu::GpuMonitorError;
use serde::Serialize;
use thiserror::Error;

/// JSON envelope returned for every error response.
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: String,
    pub kind: &'static str,
}

/// Top-level API error. Each variant maps to a specific HTTP status code via
/// the [`IntoResponse`] impl below.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("model '{0}' is not in the configured registry")]
    ModelNotFound(ModelId),

    #[error("no healthy GPU is available to serve requests")]
    AllGpusUnhealthy,

    #[error("timed out waiting for model '{0}' to become ready")]
    LoadTimeout(ModelId),

    #[error("backend crashed: {0}")]
    BackendCrashed(String),

    #[error("request cannot be served: {0}")]
    BackendUnavailable(String),

    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    Backend(#[from] BackendError),

    #[error(transparent)]
    Gpu(#[from] GpuMonitorError),

    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Http(#[from] reqwest::Error),
}

impl ApiError {
    /// Short machine-readable error kind included in [`ErrorBody`] and useful
    /// for log correlation.
    pub fn kind(&self) -> &'static str {
        match self {
            ApiError::ModelNotFound(_) => "model_not_found",
            ApiError::AllGpusUnhealthy => "all_gpus_unhealthy",
            ApiError::LoadTimeout(_) => "load_timeout",
            ApiError::BackendCrashed(_) => "backend_crashed",
            ApiError::BackendUnavailable(_) => "backend_unavailable",
            ApiError::BadRequest(_) => "bad_request",
            ApiError::Internal(_) => "internal",
            ApiError::Core(_) => "core",
            ApiError::Backend(_) => "backend",
            ApiError::Gpu(_) => "gpu",
            ApiError::Config(_) => "config",
            ApiError::Json(_) => "bad_request",
            ApiError::Http(_) => "upstream_http",
        }
    }

    fn status_code(&self) -> StatusCode {
        match self {
            ApiError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            ApiError::AllGpusUnhealthy => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::LoadTimeout(_) => StatusCode::GATEWAY_TIMEOUT,
            ApiError::BackendCrashed(_) => StatusCode::BAD_GATEWAY,
            ApiError::BackendUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::BadRequest(_) | ApiError::Json(_) => StatusCode::BAD_REQUEST,
            ApiError::Core(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Backend(_) => StatusCode::BAD_GATEWAY,
            ApiError::Gpu(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Http(_) => StatusCode::BAD_GATEWAY,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = ErrorBody {
            error: self.to_string(),
            kind: self.kind(),
        };
        if status.is_server_error() {
            tracing::error!(kind = body.kind, error = %body.error, "api error");
        } else {
            tracing::warn!(kind = body.kind, error = %body.error, "api error");
        }
        (status, Json(body)).into_response()
    }
}
