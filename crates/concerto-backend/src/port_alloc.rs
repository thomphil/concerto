//! Port allocation for backend processes.
//!
//! Backends need a unique TCP port to listen on. [`PortAllocator`] hands out
//! ports from a configurable range and lets them be released again when a
//! backend is stopped. It is entirely in-memory — it does not actually bind
//! sockets or consult the operating system — which keeps it deterministic for
//! tests. Higher layers are responsible for not stepping on ports that other
//! processes on the same host are already using.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Mutex;

/// Default port range for newly allocated backends.
pub const DEFAULT_PORT_RANGE: Range<u16> = 8100..9000;

/// A simple in-memory allocator handing out unique ports from a range.
#[derive(Debug)]
pub struct PortAllocator {
    range: Range<u16>,
    in_use: Mutex<HashSet<u16>>,
}

impl PortAllocator {
    /// Create a new allocator over the default range (8100..9000).
    pub fn new() -> Self {
        Self::with_range(DEFAULT_PORT_RANGE)
    }

    /// Create a new allocator over an explicit port range.
    pub fn with_range(range: Range<u16>) -> Self {
        Self {
            range,
            in_use: Mutex::new(HashSet::new()),
        }
    }

    /// Allocate the next free port in the range, or `None` if the range is
    /// exhausted.
    pub fn allocate(&self) -> Option<u16> {
        let mut in_use = self.lock();
        self.range.clone().find(|port| in_use.insert(*port))
    }

    /// Release a previously allocated port, making it available again.
    ///
    /// Releasing a port that was never allocated is a no-op.
    pub fn release(&self, port: u16) {
        self.lock().remove(&port);
    }

    /// Number of ports currently allocated.
    pub fn in_use_count(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashSet<u16>> {
        // The only operations we perform under this lock are infallible
        // HashSet ops, so poisoning is impossible in practice. Panicking
        // here would indicate a genuine bug, not a recoverable condition.
        self.in_use
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Default for PortAllocator {
    fn default() -> Self {
        Self::new()
    }
}
