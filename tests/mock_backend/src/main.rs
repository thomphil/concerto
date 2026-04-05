//! Binary entrypoint for the mock inference backend.
//!
//! A tiny axum server that pretends to be a real inference engine (vLLM).
//! Used by Concerto integration tests to exercise routing without GPUs.

use anyhow::Result;
use clap::Parser;
use mock_inference_backend::{run, Args};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    run(args).await
}
