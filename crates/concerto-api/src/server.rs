//! Top-level `serve` function.
//!
//! Builds the axum router, installs middleware, spawns the background
//! health-check loop, binds a TCP listener, and runs until the caller's
//! shutdown future resolves — at which point graceful shutdown drains
//! in-flight work and stops every backend.

use std::future::Future;
use std::net::SocketAddr;

use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::app::AppState;
use crate::error::ApiError;

/// Build the full router + middleware stack and run the server until
/// `shutdown_signal` resolves.
///
/// The middleware layers are composed in the order ROADMAP §6.4 specifies
/// for the open-core seam: routes first, then the ordered layer stack.
/// Future auth/billing/rate-limit layers can slot in above `TraceLayer`
/// without restructuring.
pub async fn serve<F>(state: AppState, addr: SocketAddr, shutdown_signal: F) -> Result<(), ApiError>
where
    F: Future<Output = ()> + Send + 'static,
{
    let app = crate::routes::router(state.clone())
        // --- open-core middleware extension point (ROADMAP §6.4) ---
        // auth / billing / rate-limit layers slot in here in v0.2
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    // Spawn the background health-check loop (T7).
    let health_state = state.clone();
    tokio::spawn(async move { crate::health_loop::run(health_state).await });

    info!(%addr, "concerto listening");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| ApiError::Internal(format!("bind {addr}: {e}")))?;

    // The graceful shutdown callback runs once `shutdown_signal` resolves.
    // It drains in-flight requests and stops every backend before axum
    // finishes the serve loop.
    let shutdown_state = state.clone();
    let combined_shutdown = async move {
        shutdown_signal.await;
        crate::shutdown::graceful_shutdown(shutdown_state).await;
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(combined_shutdown)
        .await
        .map_err(|e| ApiError::Internal(format!("serve loop: {e}")))?;
    Ok(())
}
