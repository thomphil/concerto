//! Concerto HTTP API.
//!
//! This crate hosts the axum-based HTTP server that fronts the concerto
//! inference multiplexer, together with the orchestrator that turns pure
//! routing decisions from [`concerto_core`] into side-effectful backend
//! lifecycle management from [`concerto_backend`].
//!
//! The entry point for consumers is [`serve`], which binds a listener,
//! installs middleware, starts the background health-check loop, and awaits
//! a graceful-shutdown signal supplied by the caller.

pub mod app;
pub mod error;
pub mod health_loop;
pub mod metrics;
pub mod orchestrator;
pub mod routes;
pub mod server;
pub mod shutdown;
pub mod types;

pub use app::{AppState, LoadResult};
pub use error::ApiError;
pub use server::serve;
