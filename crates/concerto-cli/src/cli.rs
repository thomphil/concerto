//! CLI argument parsing via `clap` derive.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "concerto",
    version,
    about = "Self-hosted LLM inference multiplexer"
)]
pub struct Cli {
    /// Path to concerto.toml configuration.
    #[arg(long, short = 'c')]
    pub config: PathBuf,

    /// Override the log filter (e.g. `info`, `debug`, `concerto=trace`).
    /// Falls back to `RUST_LOG` then `info`.
    #[arg(long)]
    pub log_level: Option<String>,

    /// Log output format.
    #[arg(long, value_enum, default_value_t = LogFormat::Pretty)]
    pub log_format: LogFormat,

    /// Override the server bind port from the config file.
    #[arg(long)]
    pub port_override: Option<u16>,

    /// Use N synthetic mock GPUs instead of real NVML-backed ones. This
    /// rewrites every model in the config to use the bundled
    /// `mock-inference-backend` binary, so `cargo run -p concerto-cli --
    /// --mock-gpus 2` works end-to-end without any real inference engine
    /// installed.
    #[arg(long)]
    pub mock_gpus: Option<usize>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LogFormat {
    Pretty,
    Json,
}
