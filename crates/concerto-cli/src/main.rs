//! Concerto CLI binary (`concerto`).
//!
//! Thin wrapper that parses args, wires the API state together, installs
//! signal handlers, and hands off to [`concerto_api::serve`].

mod cli;
mod setup;
mod signals;

use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    setup::init_tracing(&args)?;
    let (state, addr) = setup::build_app_state(&args).await?;
    let shutdown = signals::shutdown_signal(state.shutdown.clone());
    concerto_api::serve(state, addr, shutdown).await?;
    Ok(())
}
