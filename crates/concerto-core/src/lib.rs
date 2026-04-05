//! # concerto-core
//!
//! Pure routing logic, eviction policies, and memory accounting for Concerto.
//!
//! This crate has ZERO IO dependencies. It takes state in, returns decisions out.
//! All the interesting logic lives here and is trivially unit-testable.

pub mod eviction;
pub mod routing;
pub mod state;
pub mod types;

pub use routing::route_request;
pub use state::ClusterState;
pub use types::*;
