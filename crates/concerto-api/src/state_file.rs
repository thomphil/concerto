//! Persistent record of every backend process Concerto currently manages.
//!
//! Sprint 3 §A.3. The state file lives at
//! `$CONCERTO_STATE_DIR/state.json` (default
//! `~/.local/share/concerto/state.json`) and is updated atomically on
//! every launch / stop. On startup, [`reconcile`] (in `concerto-cli`)
//! reads the file, kills any process whose pid is still alive (best-
//! effort — see `concerto-cli::reconcile` for the kill/killpg dance),
//! and truncates it.
//!
//! The format is versioned by an explicit `schema_version` field so a
//! v2 file written by a future Concerto can be skipped with a warning
//! rather than panicking. Compatibility is read-side only; the writer
//! always emits the version it was compiled with.
//!
//! ## Why a wrapper, not a hook
//!
//! The recording happens in a [`RecordingBackendManager`] that wraps
//! the production `BackendManager`. Every call to `launch()` /
//! `stop()` flows through it — including the orchestrator and the
//! shutdown path — so we don't have to scatter recorder calls across
//! call sites and risk forgetting one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use concerto_backend::{BackendError, BackendHandle, BackendManager};
use concerto_core::{GpuId, ModelSpec};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::warn;

/// Current state-file schema version. Bumped when the on-disk shape
/// changes incompatibly.
pub const STATE_FILE_SCHEMA_VERSION: u32 = 1;

/// Default subdirectory under `$XDG_STATE_HOME` (or `~/.local/share`)
/// for the Concerto state file.
const DEFAULT_STATE_SUBDIR: &str = "concerto";

/// Filename inside the state directory.
pub const STATE_FILE_NAME: &str = "state.json";

/// Resolve the directory the state file lives in. `$CONCERTO_STATE_DIR`
/// wins; otherwise `$XDG_STATE_HOME/concerto`; otherwise
/// `~/.local/share/concerto`. The directory is created if it does not
/// exist.
pub fn state_dir() -> std::io::Result<PathBuf> {
    if let Ok(p) = std::env::var("CONCERTO_STATE_DIR") {
        let dir = PathBuf::from(p);
        std::fs::create_dir_all(&dir)?;
        return Ok(dir);
    }
    let base = if let Ok(p) = std::env::var("XDG_STATE_HOME") {
        PathBuf::from(p)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".local").join("share")
    } else {
        // Fallback for builds without HOME (CI, sandboxes).
        std::env::temp_dir()
    };
    let dir = base.join(DEFAULT_STATE_SUBDIR);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// One backend's entry in the state file. Recorded at launch time,
/// removed at stop time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateEntry {
    pub pid: u32,
    /// The process-group leader for the spawned backend. After Sprint 3
    /// A.1 lands, `pgid == pid` because the child runs `setsid(2)` in
    /// its pre-exec hook. Optional so older Concerto versions writing
    /// the file (before A.1) round-trip cleanly.
    #[serde(default)]
    pub pgid: Option<u32>,
    pub port: u16,
    pub model_id: String,
    pub gpu_id: u32,
    pub started_at: DateTime<Utc>,
}

impl StateEntry {
    pub fn from_handle(handle: &BackendHandle) -> Self {
        Self {
            pid: handle.pid,
            // After A.1 lands, every backend is its own pgid leader.
            // Recording it explicitly is a forward-compat hint for a
            // reconcile pass that wants to `killpg` rather than `kill`.
            pgid: Some(handle.pid),
            port: handle.port,
            model_id: handle.model_id.0.clone(),
            gpu_id: handle.gpu_id.0 as u32,
            started_at: Utc::now(),
        }
    }
}

/// On-disk root document. Versioned so future schema changes don't
/// force a hard panic on older readers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateFile {
    pub schema_version: u32,
    pub entries: Vec<StateEntry>,
}

impl StateFile {
    pub fn empty() -> Self {
        Self {
            schema_version: STATE_FILE_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }

    /// Read and parse the state file at `path`. Returns
    /// `Ok(StateFile::empty())` if the file does not exist (a fresh
    /// install) or has an unknown schema version (a future Concerto
    /// wrote it; skip with a warning rather than crash).
    pub async fn read_from(path: &Path) -> std::io::Result<Self> {
        match fs::read(path).await {
            Ok(bytes) => match serde_json::from_slice::<StateFile>(&bytes) {
                Ok(file) if file.schema_version == STATE_FILE_SCHEMA_VERSION => Ok(file),
                Ok(file) => {
                    warn!(
                        path = %path.display(),
                        version = file.schema_version,
                        expected = STATE_FILE_SCHEMA_VERSION,
                        "state file has unknown schema_version; ignoring"
                    );
                    Ok(Self::empty())
                }
                Err(e) => {
                    warn!(
                        path = %path.display(),
                        error = %e,
                        "state file is corrupt; ignoring and starting fresh"
                    );
                    Ok(Self::empty())
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::empty()),
            Err(e) => Err(e),
        }
    }

    /// Write `self` to `path` atomically — serialize to a sibling
    /// tempfile, then rename. A crash mid-write either keeps the old
    /// file intact or replaces it with the new one; never half-written
    /// JSON.
    pub async fn write_to(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        let parent = path.parent().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "state file has no parent")
        })?;

        // tempfile::NamedTempFile is sync; do the swap on a blocking
        // task so we don't block the runtime worker.
        let parent = parent.to_path_buf();
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            let mut tmp = tempfile::NamedTempFile::new_in(&parent)?;
            std::io::Write::write_all(&mut tmp, &json)?;
            tmp.as_file().sync_all()?;
            tmp.persist(&path)
                .map_err(|e| std::io::Error::other(format!("persist tempfile: {e}")))?;
            Ok(())
        })
        .await
        .map_err(std::io::Error::other)??;
        Ok(())
    }
}

/// Trait for "record this backend launched / this backend stopped".
/// The production implementation is [`JsonStateRecorder`]; tests can
/// supply a no-op or in-memory recorder.
#[async_trait]
pub trait StateRecorder: Send + Sync + 'static {
    async fn record_launch(&self, handle: &BackendHandle);
    async fn record_stop(&self, handle: &BackendHandle);
    /// Drop every entry — called from the graceful-shutdown path so a
    /// clean exit leaves the file empty.
    async fn clear(&self);
}

/// JSON-file implementation of [`StateRecorder`]. Holds the path to
/// `state.json`; every mutation reads + rewrites the whole file
/// atomically. Concerto manages at most a handful of backends so the
/// I/O is cheap.
pub struct JsonStateRecorder {
    path: PathBuf,
}

impl JsonStateRecorder {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    async fn mutate<F>(&self, f: F)
    where
        F: FnOnce(&mut StateFile),
    {
        let mut file = match StateFile::read_from(&self.path).await {
            Ok(f) => f,
            Err(e) => {
                warn!(path = %self.path.display(), error = %e, "state file read failed; starting fresh");
                StateFile::empty()
            }
        };
        f(&mut file);
        if let Err(e) = file.write_to(&self.path).await {
            warn!(path = %self.path.display(), error = %e, "state file write failed; entry may be lost across restart");
        }
    }
}

#[async_trait]
impl StateRecorder for JsonStateRecorder {
    async fn record_launch(&self, handle: &BackendHandle) {
        let entry = StateEntry::from_handle(handle);
        self.mutate(|file| {
            // Drop any previous entry for this pid — defensive against
            // pid reuse across reboots.
            file.entries.retain(|e| e.pid != entry.pid);
            file.entries.push(entry);
        })
        .await;
    }

    async fn record_stop(&self, handle: &BackendHandle) {
        let pid = handle.pid;
        self.mutate(|file| file.entries.retain(|e| e.pid != pid))
            .await;
    }

    async fn clear(&self) {
        let empty = StateFile::empty();
        if let Err(e) = empty.write_to(&self.path).await {
            warn!(path = %self.path.display(), error = %e, "state file truncate on clean shutdown failed");
        }
    }
}

/// Wrap a [`BackendManager`] so every successful launch / stop flows
/// through a [`StateRecorder`]. Drop-in for any place a
/// `Arc<dyn BackendManager>` is expected.
pub struct RecordingBackendManager {
    inner: Arc<dyn BackendManager>,
    recorder: Arc<dyn StateRecorder>,
}

impl RecordingBackendManager {
    pub fn new(inner: Arc<dyn BackendManager>, recorder: Arc<dyn StateRecorder>) -> Self {
        Self { inner, recorder }
    }
}

#[async_trait]
impl BackendManager for RecordingBackendManager {
    async fn launch(&self, spec: &ModelSpec, gpu_id: GpuId) -> Result<BackendHandle, BackendError> {
        let handle = self.inner.launch(spec, gpu_id).await?;
        self.recorder.record_launch(&handle).await;
        Ok(handle)
    }

    async fn stop(&self, handle: &BackendHandle) -> Result<(), BackendError> {
        let result = self.inner.stop(handle).await;
        // Record the stop *intent* even on error — the process is
        // gone either way and a stale entry would survive across
        // restart, which is the worse failure mode.
        self.recorder.record_stop(handle).await;
        result
    }

    async fn health_check(&self, handle: &BackendHandle) -> bool {
        self.inner.health_check(handle).await
    }
}

/// Convenience: helper for tests / `concerto-cli::reconcile` that
/// returns the canonical state file path inside [`state_dir()`].
pub fn default_state_file_path() -> std::io::Result<PathBuf> {
    Ok(state_dir()?.join(STATE_FILE_NAME))
}

#[allow(clippy::ignored_unit_patterns)]
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::SubsecRound;

    fn sample_entry(pid: u32) -> StateEntry {
        StateEntry {
            pid,
            pgid: Some(pid),
            port: 8100,
            model_id: format!("model-{pid}"),
            gpu_id: 0,
            // chrono serialises with microsecond precision in the JSON
            // by default; round so equality after round-trip holds
            // bit-for-bit.
            started_at: Utc::now().trunc_subsecs(0),
        }
    }

    #[tokio::test]
    async fn round_trip_write_read_equality() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");

        let original = StateFile {
            schema_version: STATE_FILE_SCHEMA_VERSION,
            entries: vec![sample_entry(1234), sample_entry(5678)],
        };
        original.write_to(&path).await.expect("write");

        let read = StateFile::read_from(&path).await.expect("read");
        assert_eq!(read, original);
    }

    #[tokio::test]
    async fn read_missing_file_returns_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("does-not-exist.json");
        let read = StateFile::read_from(&path).await.expect("read missing");
        assert_eq!(read, StateFile::empty());
    }

    #[tokio::test]
    async fn unknown_schema_version_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");
        let future = serde_json::json!({
            "schema_version": 999,
            "entries": [{
                "pid": 42,
                "pgid": 42,
                "port": 8100,
                "model_id": "model-x",
                "gpu_id": 0,
                "started_at": "2026-04-27T00:00:00Z",
            }]
        });
        tokio::fs::write(&path, serde_json::to_vec_pretty(&future).unwrap())
            .await
            .expect("write future-version");

        let read = StateFile::read_from(&path).await.expect("read");
        assert_eq!(read, StateFile::empty());
    }

    #[tokio::test]
    async fn corrupt_file_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");
        tokio::fs::write(&path, b"this is not json")
            .await
            .expect("write garbage");
        let read = StateFile::read_from(&path).await.expect("read corrupt");
        assert_eq!(read, StateFile::empty());
    }

    #[tokio::test]
    async fn json_recorder_records_launch_and_stop() {
        use concerto_core::ModelId;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");
        let recorder = JsonStateRecorder::new(path.clone());

        let handle = BackendHandle {
            pid: 4321,
            port: 8100,
            model_id: ModelId("model-a".into()),
            gpu_id: GpuId(0),
            health_path: "/health".into(),
        };
        recorder.record_launch(&handle).await;

        let after_launch = StateFile::read_from(&path).await.expect("read");
        assert_eq!(after_launch.entries.len(), 1);
        assert_eq!(after_launch.entries[0].pid, 4321);
        assert_eq!(after_launch.entries[0].pgid, Some(4321));
        assert_eq!(after_launch.entries[0].port, 8100);

        recorder.record_stop(&handle).await;
        let after_stop = StateFile::read_from(&path).await.expect("read");
        assert!(after_stop.entries.is_empty());
    }

    #[tokio::test]
    async fn json_recorder_clear_truncates_file() {
        use concerto_core::ModelId;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");
        let recorder = JsonStateRecorder::new(path.clone());

        let handle = BackendHandle {
            pid: 1,
            port: 8100,
            model_id: ModelId("a".into()),
            gpu_id: GpuId(0),
            health_path: "/health".into(),
        };
        recorder.record_launch(&handle).await;
        recorder.clear().await;

        let after = StateFile::read_from(&path).await.expect("read");
        assert!(after.entries.is_empty());
    }
}
