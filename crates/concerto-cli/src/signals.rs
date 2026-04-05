//! POSIX + Ctrl-C signal handling. Fires `state.shutdown` on SIGINT/SIGTERM.

use std::sync::Arc;

use tokio::sync::Notify;
use tracing::info;

pub async fn shutdown_signal(notify: Arc<Notify>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("received SIGINT"),
        _ = terminate => info!("received SIGTERM"),
    }

    notify.notify_waiters();
}
