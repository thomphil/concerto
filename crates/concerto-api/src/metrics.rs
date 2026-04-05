//! Prometheus metrics facade (ROADMAP §5 T10, §8).
//!
//! Concerto exposes a small, stable set of Prometheus metrics via
//! `GET /metrics`. This module owns:
//!
//! - the metric **name constants** so emit-sites and scrape-time consumers
//!   agree on spelling,
//! - a **process-global recorder installer** so `concerto-cli`, the
//!   scenario harness, and any future embedder share a single
//!   `PrometheusHandle` without racing on the `metrics` crate's global
//!   recorder slot,
//! - the gauge refresh helper that samples live cluster / GPU state so
//!   gauges rendered by `/metrics` reflect the current truth rather than
//!   the last event-driven write.
//!
//! Counters and histograms are emitted inline at their call sites (chat
//! route, orchestrator, health loop) via the `metrics::counter!` /
//! `metrics::histogram!` macros. Only gauges need the refresh pass here,
//! because they're time-based rather than event-driven.

use std::sync::{Arc, OnceLock};

use metrics::{gauge, Gauge};
use metrics_exporter_prometheus::{BuildError, PrometheusBuilder, PrometheusHandle};

use crate::app::AppState;

// ---- Metric name constants -------------------------------------------------
//
// Names match ROADMAP §8. Labels are applied at the emit site. Keep this
// list as the single source of truth — scrapers, dashboards, and tests all
// key off these strings.

/// Counter incremented once per incoming chat completion, labelled by
/// `decision` (`loaded`, `loaded_after_load`, `rejected_backend_unavailable`,
/// `rejected_all_unhealthy`, `error`).
pub const REQUESTS_TOTAL: &str = "concerto_requests_total";

/// Histogram of the pure-logic routing decision phase (seconds).
pub const ROUTING_DECISION_SECONDS: &str = "concerto_routing_decision_seconds";

/// Histogram of cold-start + launch duration (seconds), recorded only on
/// the launcher path (subscribers waiting on a dedup channel don't
/// double-count).
pub const MODEL_LOAD_DURATION_SECONDS: &str = "concerto_model_load_duration_seconds";

/// Counter of successful backend launches.
pub const BACKEND_LAUNCHES_TOTAL: &str = "concerto_backend_launches_total";

/// Counter of evictions performed while serving a cold-start.
pub const EVICTION_TOTAL: &str = "concerto_eviction_total";

/// Counter of backends removed by the background health loop after a
/// failed `/health` probe.
pub const BACKEND_HEALTH_CHECK_FAILURES_TOTAL: &str =
    "concerto_backend_health_check_failures_total";

/// Gauge: number of live backends currently tracked by the orchestrator.
pub const ACTIVE_BACKENDS: &str = "concerto_active_backends";

/// Gauge: GPU memory currently in use (bytes), labelled by `gpu` id.
pub const GPU_MEMORY_USED_BYTES: &str = "concerto_gpu_memory_used_bytes";

/// Gauge: GPU total physical memory (bytes), labelled by `gpu` id.
pub const GPU_MEMORY_TOTAL_BYTES: &str = "concerto_gpu_memory_total_bytes";

// ---- Recorder installation -------------------------------------------------

/// Process-global cache of the install result. `Ok` on success, `Err` with
/// a stringified build error on failure — stored either way so retries
/// surface the same outcome instead of trying to reinstall.
static HANDLE: OnceLock<Result<Arc<PrometheusHandle>, String>> = OnceLock::new();

/// Error returned by [`install`] when the Prometheus recorder cannot be
/// installed.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("installing Prometheus recorder: {0}")]
    Install(String),
}

impl From<BuildError> for MetricsError {
    fn from(e: BuildError) -> Self {
        MetricsError::Install(e.to_string())
    }
}

/// Install the process-global Prometheus recorder and return a cloneable
/// handle suitable for storing in [`AppState`].
///
/// Idempotent: the first call installs the recorder and caches the handle;
/// every subsequent call returns a clone of the cached handle. A failure
/// on the first call is also cached and returned on every subsequent call
/// so a broken install doesn't mask itself behind a second attempt.
pub fn install() -> Result<Arc<PrometheusHandle>, MetricsError> {
    let cached = HANDLE.get_or_init(|| {
        PrometheusBuilder::new()
            .install_recorder()
            .map(Arc::new)
            .map_err(|e| e.to_string())
    });
    match cached {
        Ok(handle) => Ok(handle.clone()),
        Err(e) => Err(MetricsError::Install(e.clone())),
    }
}

// ---- Gauge refresh ---------------------------------------------------------

/// Sample live state and write it into the Prometheus gauges.
///
/// Called at the top of `GET /metrics` so a scrape always sees fresh
/// values. Cheap enough to run per scrape (two short-lived mutex
/// acquires) — scrape frequency is capped by the caller (the bench rig
/// samples at 1 Hz).
///
/// The GPU memory gauges reflect concerto's **cluster bookkeeping**, not
/// the driver's ground truth. That's deliberate: the bench rig's VRAM
/// drift check (ROADMAP M4) compares this tracked value to an independent
/// `nvidia-smi` sample to detect divergence. Scrapers that want driver
/// ground truth should use `nvidia-smi` or a node exporter directly.
pub async fn refresh_state_gauges(state: &AppState) {
    // Active backend count — snapshot under a short critical section.
    let active = state.backends.lock().await.len();
    gauge!(ACTIVE_BACKENDS).set(active as f64);

    // GPU memory gauges — one set per GPU, labelled by id. Read from
    // `ClusterState` so the values match what `/status` renders.
    let cluster = state.cluster.lock().await;
    for gpu in &cluster.gpus {
        let gpu_label = gpu.id.0.to_string();
        gpu_gauge(GPU_MEMORY_USED_BYTES, &gpu_label).set(gpu.memory_used.as_u64() as f64);
        gpu_gauge(GPU_MEMORY_TOTAL_BYTES, &gpu_label).set(gpu.memory_total.as_u64() as f64);
    }
}

/// Helper for constructing a GPU-labelled gauge. `metrics::gauge!` insists
/// on a literal name and static label keys; we pass through a pre-built
/// name and dynamic `gpu` id as the label value.
fn gpu_gauge(name: &'static str, gpu_id: &str) -> Gauge {
    gauge!(name, "gpu" => gpu_id.to_string())
}
