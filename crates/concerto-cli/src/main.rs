//! Concerto CLI binary (`concerto`).
//!
//! Thin wrapper that parses args, wires the API state together, installs
//! signal handlers, and hands off to [`concerto_api::serve`].

mod cli;
mod reconcile;
mod setup;
mod signals;

use anyhow::Context;
use clap::Parser;
use concerto_api::state_file::default_state_file_path;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    setup::init_tracing(&args)?;

    // Sprint 3 §A.3: clean up any backends a previous Concerto crash
    // left running before we start a fresh batch on the same port
    // range. Best-effort — failures here are logged and swallowed so a
    // cosmetic state-file problem can't keep the server from starting.
    let state_path =
        default_state_file_path().context("resolving state-file path for reconcile")?;
    match reconcile::reconcile(&state_path).await {
        Ok(report) if report == reconcile::ReconcileReport::default() => {}
        Ok(report) => info!(
            killed = report.killed,
            stale = report.stale,
            failed = report.failed,
            "startup reconcile complete"
        ),
        Err(e) => tracing::warn!(error = %e, "startup reconcile failed; continuing"),
    }

    let (state, addr) = setup::build_app_state(&args).await?;
    let shutdown = signals::shutdown_signal(state.shutdown.clone());
    concerto_api::serve(state, addr, shutdown).await?;
    Ok(())
}
