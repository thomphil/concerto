//! # concerto-config
//!
//! TOML configuration parsing for Concerto.
//!
//! This crate is responsible for loading and validating the `concerto.toml`
//! configuration file and converting it into the canonical types defined in
//! [`concerto_core`].
//!
//! The entry point is [`ConcertoConfig`], which can be parsed from a string
//! with [`ConcertoConfig::from_toml_str`] or from a path on disk with
//! [`ConcertoConfig::from_path`].

pub mod config;
pub mod error;
pub mod gpus;
pub mod models;
pub mod routing;
pub mod server;

pub use config::ConcertoConfig;
pub use error::ConfigError;
pub use gpus::GpuConfigEntry;
pub use models::ModelConfigEntry;
pub use routing::RoutingSection;
pub use server::ServerConfig;
