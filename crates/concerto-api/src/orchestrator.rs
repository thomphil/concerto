//! The orchestrator state machine.
//!
//! This module bridges the pure routing logic in [`concerto_core`] with the
//! side-effectful backend lifecycle management in [`concerto_backend`]. It's
//! the piece that every "hidden hard problem" in ROADMAP §3 converges on.
//!
//! The design (all decided in ROADMAP §5 T1–T10) is:
//!
//! - A single `Mutex<ClusterState>` guards all cluster mutations. Decisions
//!   are made under the lock; launches happen outside it. No `.await` while
//!   the lock is held.
//! - Cold-start dedup: before launching, the orchestrator checks an
//!   in-memory map keyed by `ModelId` for an existing `broadcast::Sender`.
//!   Concurrent requesters for the same cold model subscribe rather than
//!   triggering a second launch (ROADMAP §3 problem 1, T3).
//! - Failed loads and evicted-then-failed paths clean up after themselves:
//!   the dedup sender is always removed, waiters are always notified, and
//!   the `state.backends` bookkeeping map is always kept in sync with the
//!   cluster snapshot.

use std::time::{Duration, Instant};

use concerto_core::{route_request, GpuId, ModelId, RoutingDecision};
use metrics::{counter, histogram};
use tokio::sync::broadcast;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::app::{AppState, LoadResult};
use crate::error::ApiError;
use crate::metrics::{
    BACKEND_LAUNCHES_TOTAL, EVICTION_TOTAL, MODEL_LOAD_DURATION_SECONDS, ROUTING_DECISION_SECONDS,
};

/// How the orchestrator satisfied a request. Used to label the
/// `concerto_requests_total` metric at the chat handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingKind {
    /// Served from a backend that was already warm.
    Loaded,
    /// Required a cold-start (launcher or dedup subscriber path). The
    /// caller-visible outcome is identical; only the orchestrator work
    /// differs.
    LoadedAfterLoad,
}

impl RoutingKind {
    /// Prometheus label value for the `decision` dimension.
    pub fn label(&self) -> &'static str {
        match self {
            RoutingKind::Loaded => "loaded",
            RoutingKind::LoadedAfterLoad => "loaded_after_load",
        }
    }
}

/// Resolved dispatch target for an incoming request.
#[derive(Debug, Clone)]
pub struct BackendTarget {
    pub gpu_id: GpuId,
    pub port: u16,
    pub kind: RoutingKind,
}

/// Route an incoming request and make the target backend ready.
///
/// This is the public entry point used by `routes::chat::completions`. On
/// success it returns a [`BackendTarget`] the caller can forward to; on
/// failure it returns an [`ApiError`] with a specific kind that the response
/// layer maps to a status code.
pub async fn route_and_dispatch(
    state: &AppState,
    model_id: ModelId,
) -> Result<BackendTarget, ApiError> {
    let routing_config = state.config.routing_config();

    // --- 1. Make the routing decision under a short critical section. -----
    //
    // Time only the pure-logic phase (lock acquire + `route_request`). The
    // cold-start / launch phase is measured separately by
    // `concerto_model_load_duration_seconds` so dashboards can see the two
    // independently.
    let decision_start = Instant::now();
    let decision = {
        let cluster = state.cluster.lock().await;
        route_request(&model_id, &cluster, &routing_config)
    };
    histogram!(ROUTING_DECISION_SECONDS).record(decision_start.elapsed().as_secs_f64());

    match decision {
        RoutingDecision::RouteToLoaded { gpu_id, port } => {
            // Already loaded. Bump bookkeeping and return.
            touch_loaded_model(state, &model_id).await;
            debug!(model = %model_id, %gpu_id, port, "serving from warm backend");
            Ok(BackendTarget {
                gpu_id,
                port,
                kind: RoutingKind::Loaded,
            })
        }
        RoutingDecision::Reject { reason } => {
            warn!(model = %model_id, reason, "routing rejected");
            Err(ApiError::BackendUnavailable(reason))
        }
        RoutingDecision::LoadModel { gpu_id, evict } => {
            cold_start(state, &model_id, gpu_id, evict).await
        }
    }
}

/// Bump `last_request_at` + `request_count` on a loaded model.
async fn touch_loaded_model(state: &AppState, model_id: &ModelId) {
    let now = chrono::Utc::now();
    let mut cluster = state.cluster.lock().await;
    for gpu in &mut cluster.gpus {
        if let Some(m) = gpu
            .loaded_models
            .iter_mut()
            .find(|m| m.model_id == *model_id)
        {
            m.last_request_at = now;
            m.request_count += 1;
            return;
        }
    }
}

/// Cold-start path with dedup and rollback.
///
/// One of two things happens here: either this is the first request for the
/// cold model and we take ownership of the launch, or another task is
/// already launching and we subscribe to its broadcast channel and wait.
async fn cold_start(
    state: &AppState,
    model_id: &ModelId,
    gpu_id: GpuId,
    evict: Vec<ModelId>,
) -> Result<BackendTarget, ApiError> {
    // --- 2. Dedup lookup: am I the launcher, or a subscriber? -------------
    let role = {
        let mut loading = state.loading.lock().await;
        if let Some(sender) = loading.get(model_id) {
            LoadRole::Subscriber(sender.subscribe())
        } else {
            let (tx, _rx) = broadcast::channel(16);
            loading.insert(model_id.clone(), tx);
            LoadRole::Launcher
        }
    };

    match role {
        LoadRole::Subscriber(mut rx) => {
            let cold_start_timeout =
                Duration::from_secs(state.config.routing.cold_start_timeout_secs);
            debug!(model = %model_id, "subscribing to in-flight load");
            match timeout(cold_start_timeout, rx.recv()).await {
                Ok(Ok(LoadResult::Ok(handle))) => Ok(BackendTarget {
                    gpu_id: handle.gpu_id,
                    port: handle.port,
                    kind: RoutingKind::LoadedAfterLoad,
                }),
                Ok(Ok(LoadResult::Err(reason))) => Err(ApiError::BackendCrashed(reason)),
                Ok(Err(_)) => Err(ApiError::BackendCrashed(
                    "dedup broadcast dropped before delivering load result".into(),
                )),
                Err(_) => Err(ApiError::LoadTimeout(model_id.clone())),
            }
        }
        LoadRole::Launcher => {
            // We own the launch. Every exit path from here must:
            // - broadcast the outcome to any subscribers
            // - remove the dedup sender
            // - ensure cluster/backends bookkeeping is consistent
            //
            // Only the launcher path records the model-load histogram;
            // subscribers wait on the same event but double-counting would
            // skew p95/p99 under concurrent requests for the same cold
            // model.
            let launch_start = Instant::now();
            let outcome = do_launch(state, model_id, gpu_id, evict).await;
            histogram!(MODEL_LOAD_DURATION_SECONDS).record(launch_start.elapsed().as_secs_f64());
            finalise_launch(state, model_id, &outcome).await;
            outcome
        }
    }
}

enum LoadRole {
    Launcher,
    Subscriber(broadcast::Receiver<LoadResult>),
}

/// Perform eviction(s) and launch a fresh backend on the selected GPU.
///
/// This function is the one that actually calls `BackendManager::stop` and
/// `BackendManager::launch`. Every mutation to `state.cluster` and
/// `state.backends` goes through short critical sections — no `.await` while
/// holding the cluster mutex.
async fn do_launch(
    state: &AppState,
    model_id: &ModelId,
    gpu_id: GpuId,
    evict: Vec<ModelId>,
) -> Result<BackendTarget, ApiError> {
    // --- 3. Evict any models the router asked us to evict. ---------------
    let mut evicted_any = false;
    for victim in &evict {
        let handle = {
            let mut backends = state.backends.lock().await;
            backends.remove(victim)
        };
        if let Some(handle) = handle {
            info!(%victim, pid = handle.pid, port = handle.port, "evicting backend");
            if let Err(e) = state.backend.stop(&handle).await {
                warn!(%victim, error = %e, "stop failed during eviction; continuing");
            }
            evicted_any = true;
        }
        remove_loaded_model(state, victim).await;
        counter!(EVICTION_TOTAL).increment(1);
    }

    // Inference engines (especially vLLM) spawn child processes that may
    // hold GPU memory briefly after the main process is killed. Give the
    // OS a moment to reclaim CUDA resources before launching a new
    // backend on the same GPU.
    // TODO: replace with process-group kill (nix::sys::signal::killpg)
    // so we don't need a fixed delay.
    if evicted_any {
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    // --- 4. Look up the model spec we're about to launch. ----------------
    let spec = {
        let cluster = state.cluster.lock().await;
        cluster
            .get_model_spec(model_id)
            .cloned()
            .ok_or_else(|| ApiError::ModelNotFound(model_id.clone()))?
    };

    // --- 5. Launch the backend, bounded by cold_start_timeout. -----------
    let cold_start_timeout = Duration::from_secs(state.config.routing.cold_start_timeout_secs);
    info!(model = %model_id, %gpu_id, "launching backend");
    let handle = match timeout(cold_start_timeout, state.backend.launch(&spec, gpu_id)).await {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => {
            return Err(ApiError::from(e));
        }
        Err(_) => {
            return Err(ApiError::LoadTimeout(model_id.clone()));
        }
    };

    // --- 6. Commit: insert into cluster + backends map. ------------------
    {
        let mut cluster = state.cluster.lock().await;
        if let Some(gpu) = cluster.gpus.iter_mut().find(|g| g.id == gpu_id) {
            let vram = spec.vram_required;
            let used = bytesize::ByteSize::b(gpu.memory_used.as_u64() + vram.as_u64());
            let available =
                bytesize::ByteSize::b(gpu.memory_available.as_u64().saturating_sub(vram.as_u64()));
            gpu.memory_used = used;
            gpu.memory_available = available;
            gpu.loaded_models.push(concerto_core::LoadedModel {
                model_id: model_id.clone(),
                vram_usage: vram,
                last_request_at: chrono::Utc::now(),
                request_count: 1,
                backend_port: handle.port,
            });
        }
    }
    state
        .backends
        .lock()
        .await
        .insert(model_id.clone(), handle.clone());

    counter!(BACKEND_LAUNCHES_TOTAL).increment(1);
    info!(model = %model_id, pid = handle.pid, port = handle.port, "backend ready");
    Ok(BackendTarget {
        gpu_id: handle.gpu_id,
        port: handle.port,
        kind: RoutingKind::LoadedAfterLoad,
    })
}

/// After [`do_launch`] returns, broadcast the outcome to any subscribers and
/// remove the dedup sender so future requests start fresh.
async fn finalise_launch(
    state: &AppState,
    model_id: &ModelId,
    outcome: &Result<BackendTarget, ApiError>,
) {
    let sender = state.loading.lock().await.remove(model_id);
    if let Some(sender) = sender {
        let msg = match outcome {
            Ok(target) => {
                // Re-wrap the target into a BackendHandle-shaped broadcast.
                // Subscribers only look at port + gpu_id, so we can reuse the
                // handle stored in state.backends.
                if let Some(handle) = state.backends.lock().await.get(model_id).cloned() {
                    LoadResult::Ok(handle)
                } else {
                    LoadResult::Ok(concerto_backend::BackendHandle {
                        pid: 0,
                        port: target.port,
                        model_id: model_id.clone(),
                        gpu_id: target.gpu_id,
                        health_path: "/health".to_string(),
                    })
                }
            }
            Err(e) => LoadResult::Err(e.to_string()),
        };
        // A send error just means no subscribers were waiting; that's fine.
        let _ = sender.send(msg);
    }
}

/// Remove a model from cluster.loaded_models and reclaim its VRAM bookkeeping.
async fn remove_loaded_model(state: &AppState, model_id: &ModelId) {
    let mut cluster = state.cluster.lock().await;
    for gpu in &mut cluster.gpus {
        let before = gpu.loaded_models.len();
        let freed: u64 = gpu
            .loaded_models
            .iter()
            .filter(|m| m.model_id == *model_id)
            .map(|m| m.vram_usage.as_u64())
            .sum();
        gpu.loaded_models.retain(|m| m.model_id != *model_id);
        if gpu.loaded_models.len() != before {
            gpu.memory_used = bytesize::ByteSize::b(gpu.memory_used.as_u64().saturating_sub(freed));
            gpu.memory_available = bytesize::ByteSize::b(gpu.memory_available.as_u64() + freed);
        }
    }
}
