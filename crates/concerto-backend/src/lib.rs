//! # concerto-backend
//!
//! Backend process lifecycle management for Concerto.
//!
//! This crate provides the [`BackendManager`] trait, which abstracts over how
//! inference-engine backends are started, stopped, and health-checked, plus two
//! implementations:
//!
//! - [`MockBackendManager`] — an in-memory manager that pretends to launch
//!   backends without spawning any real processes. Used for unit and
//!   integration tests, and for development without GPUs.
//! - [`ProcessBackendManager`] — a real manager that spawns child processes
//!   (vLLM, llama.cpp, SGLang, or our mock inference binary) and waits for
//!   them to become healthy before returning a handle.
//!
//! Both implementations share a [`PortAllocator`] abstraction for handing out
//! unique backend ports from a configurable range.

pub mod manager;
pub mod mock;
pub mod port_alloc;
pub mod process;

pub use manager::{BackendError, BackendHandle, BackendManager};
pub use mock::MockBackendManager;
pub use port_alloc::PortAllocator;
pub use process::ProcessBackendManager;
