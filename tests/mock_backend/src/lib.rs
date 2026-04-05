//! Library surface for the mock inference backend.
//!
//! The crate is primarily a binary, but exposing a library allows integration
//! tests to spin the server up in-process.

pub mod responses;
pub mod server;

pub use server::{build_router, run, AppState, Args};
