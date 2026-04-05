//! Error types for configuration parsing and validation.

use thiserror::Error;

/// Errors that can occur while loading or validating a Concerto configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Failed to read a config file from disk.
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to parse the TOML document (syntax or schema error).
    #[error("failed to parse config TOML: {0}")]
    Parse(#[from] toml::de::Error),

    /// Two `[[models]]` entries share the same `id`.
    #[error("duplicate model id in config: {0}")]
    DuplicateModelId(String),

    /// Two `[[gpus]]` entries share the same `id`.
    #[error("duplicate gpu id in config: {0}")]
    DuplicateGpuId(usize),

    /// The `[[models]]` section is empty — at least one model is required.
    #[error("config must contain at least one model in [[models]]")]
    EmptyModels,

    /// The `[[gpus]]` section is empty — at least one GPU is required.
    #[error("config must contain at least one GPU in [[gpus]]")]
    EmptyGpus,

    /// `server.port` is outside the valid range (1..=65535).
    ///
    /// Note: `u16` already constrains the upper bound, so in practice this
    /// only fires for port `0`.
    #[error("invalid server port: {0} (must be 1-65535)")]
    InvalidPort(u16),
}
