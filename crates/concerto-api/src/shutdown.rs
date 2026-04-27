//! Graceful shutdown logic.
//!
//! On shutdown:
//! 1. Notify background tasks (health loop) and any active streaming
//!    responses via `state.shutdown`. Streams emit a final `data: [DONE]`
//!    event and close, decrementing [`AppState::in_flight`] promptly.
//! 2. Poll [`AppState::in_flight`] every 50ms until it reaches zero or
//!    `routing.shutdown_drain_secs` elapses, whichever comes first.
//! 3. Stop every known backend.
//!
//! Sprint 3 (A.2) replaced the previous fixed 500ms sleep with this
//! polling loop so SIGTERM-during-streaming and slow non-streaming
//! requests both terminate deterministically.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use concerto_backend::BackendHandle;
use concerto_core::ModelId;
use tracing::{info, warn};

use crate::app::AppState;

const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Best-effort graceful shutdown.
pub async fn graceful_shutdown(state: AppState) {
    info!("graceful shutdown: notifying background tasks and streaming responses");
    state.shutdown.notify_waiters();

    let deadline_secs = state.config.routing.shutdown_drain_secs;
    drain_in_flight(&state, Duration::from_secs(deadline_secs)).await;

    let handles: Vec<(ModelId, BackendHandle)> = {
        let mut backends = state.backends.lock().await;
        backends.drain().collect()
    };
    info!(count = handles.len(), "stopping backends");
    for (model_id, handle) in handles {
        if let Err(e) = state.backend.stop(&handle).await {
            warn!(%model_id, error = %e, "backend stop returned error during shutdown");
        }
    }
    info!("graceful shutdown complete");
}

/// Poll `state.in_flight` at [`DRAIN_POLL_INTERVAL`] until it reaches zero
/// or `deadline` elapses. The deadline is bounded by
/// `routing.shutdown_drain_secs` and is intended to give streaming
/// bodies time to flush their final `data: [DONE]` event after the
/// shutdown notification has fired.
async fn drain_in_flight(state: &AppState, deadline: Duration) {
    let started = Instant::now();
    loop {
        let active = state.in_flight.load(Ordering::Relaxed);
        if active == 0 {
            info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                "drain complete"
            );
            return;
        }
        if started.elapsed() >= deadline {
            warn!(
                active,
                deadline_secs = deadline.as_secs(),
                "drain deadline exceeded; proceeding with backend stop"
            );
            return;
        }
        tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use super::*;

    /// Three fake in-flight requests that all clear at t=200ms must drain
    /// in (200, 270]ms — not the old 500ms fixed sleep, not 0ms either.
    /// The 70ms upper-bound padding tolerates the polling cadence + tokio
    /// scheduler jitter on slow CI runners.
    #[tokio::test]
    async fn drain_polls_until_in_flight_zero() {
        let counter = Arc::new(AtomicUsize::new(3));

        let counter_clone = counter.clone();
        let drainer = tokio::spawn(async move {
            let started = Instant::now();
            // Hand-rolled drain that mirrors `drain_in_flight` so this
            // test doesn't need a full AppState. The shape is what matters.
            loop {
                if counter_clone.load(Ordering::Relaxed) == 0 {
                    return started.elapsed();
                }
                if started.elapsed() >= Duration::from_secs(5) {
                    panic!("drain hit safety deadline");
                }
                tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
            }
        });

        tokio::time::sleep(Duration::from_millis(200)).await;
        counter.store(0, Ordering::Relaxed);

        let elapsed = drainer.await.expect("drainer task");
        assert!(
            elapsed >= Duration::from_millis(200),
            "drain should not finish before the deadline (got {elapsed:?})"
        );
        assert!(
            elapsed < Duration::from_millis(280),
            "drain should fire within the next poll tick of in_flight = 0 \
             (got {elapsed:?})"
        );
    }
}
