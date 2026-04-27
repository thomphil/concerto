//! Startup reconcile: clean up backends left behind by a crashed
//! Concerto process.
//!
//! Sprint 3 §A.3. The state file written by [`JsonStateRecorder`]
//! tracks every backend Concerto currently manages. If we exit cleanly,
//! the file is empty by the time the process dies (the
//! `graceful_shutdown` path calls `recorder.clear()`). If we crash —
//! SIGKILL, OOM, panic in a background task — entries linger.
//!
//! On the next start, [`reconcile`] reads the state file and SIGKILLs
//! every recorded process. Adoption (taking the orphan back into
//! Concerto's management) is **not** in scope for v0.1; the simpler
//! "kill and start fresh" path is what the v0.1 exit criterion calls
//! for and what the docs document. See ROADMAP §11 R13 for the
//! related vLLM EngineCore work that A.1 closes.
//!
//! [`JsonStateRecorder`]: concerto_api::state_file::JsonStateRecorder

use std::path::Path;

use concerto_api::state_file::StateFile;
use tracing::{info, warn};

/// Read the state file at `path`, kill every entry's process group (or
/// process, if pgid wasn't recorded), and truncate the file. Best
/// effort throughout — a missing file is the normal case on a fresh
/// install, and any single kill failure is logged and skipped.
pub async fn reconcile(path: &Path) -> std::io::Result<ReconcileReport> {
    let file = StateFile::read_from(path).await?;
    if file.entries.is_empty() {
        return Ok(ReconcileReport::default());
    }

    let mut report = ReconcileReport::default();
    for entry in &file.entries {
        match kill_orphan(entry.pid, entry.pgid) {
            KillOutcome::WasAlive => {
                warn!(
                    pid = entry.pid,
                    port = entry.port,
                    model_id = %entry.model_id,
                    "reconciled: killed orphan backend left behind by a previous Concerto crash"
                );
                report.killed += 1;
            }
            KillOutcome::AlreadyDead => {
                info!(
                    pid = entry.pid,
                    port = entry.port,
                    model_id = %entry.model_id,
                    "reconciled: stale state-file entry, process already gone"
                );
                report.stale += 1;
            }
            KillOutcome::Failed(e) => {
                warn!(
                    pid = entry.pid,
                    error = %e,
                    "reconcile: failed to signal orphan; leaving entry — manual cleanup may be needed"
                );
                report.failed += 1;
            }
        }
    }

    // Truncate. Even if some kills failed, leaving the entry around
    // means the next start tries again — which is desirable, but the
    // current model is "kill and start fresh", and the file is not
    // adoption-ready in v0.1. We document the manual recovery path in
    // docs/troubleshooting.md.
    StateFile::empty().write_to(path).await?;
    Ok(report)
}

/// Outcome of trying to terminate one orphan.
#[derive(Debug)]
enum KillOutcome {
    WasAlive,
    AlreadyDead,
    #[cfg_attr(not(unix), allow(dead_code))]
    Failed(String),
}

/// What [`reconcile`] did. Useful for tests and for the startup log
/// summary line.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileReport {
    pub killed: usize,
    pub stale: usize,
    pub failed: usize,
}

#[cfg(unix)]
fn kill_orphan(pid: u32, pgid: Option<u32>) -> KillOutcome {
    use nix::errno::Errno;
    use nix::sys::signal::{kill, killpg, Signal};
    use nix::unistd::Pid;

    let pid_t = Pid::from_raw(pid as i32);

    // Probe first — if the pid is gone, we're done. Saves a kill
    // syscall and gives a cleaner log line.
    match kill(pid_t, None) {
        Err(Errno::ESRCH) => return KillOutcome::AlreadyDead,
        Err(e) => return KillOutcome::Failed(format!("probe kill(0): {e}")),
        Ok(()) => {}
    }

    // If pgid was recorded and matches pid, the spawn went through
    // A.1's setsid path and the entire process group is killable in
    // one shot — including any vLLM EngineCore children. Try killpg
    // first; if that fails (e.g. pgid != pid because A.1 hadn't
    // landed when this entry was written) fall back to killing the
    // parent only.
    if let Some(pgid) = pgid {
        let pgid_t = Pid::from_raw(pgid as i32);
        match killpg(pgid_t, Signal::SIGKILL) {
            Ok(()) => return KillOutcome::WasAlive,
            Err(Errno::ESRCH) => {
                // Group already gone — but probe said pid was alive,
                // so the parent isn't a pgid leader. Fall through to
                // direct kill.
            }
            Err(e) => {
                return KillOutcome::Failed(format!("killpg(SIGKILL): {e}"));
            }
        }
    }

    match kill(pid_t, Signal::SIGKILL) {
        Ok(()) => KillOutcome::WasAlive,
        Err(Errno::ESRCH) => KillOutcome::AlreadyDead,
        Err(e) => KillOutcome::Failed(format!("kill(SIGKILL): {e}")),
    }
}

#[cfg(not(unix))]
fn kill_orphan(_pid: u32, _pgid: Option<u32>) -> KillOutcome {
    // Non-Unix platforms aren't supported in production. See
    // `concerto-backend::process::configure_session_leader`.
    KillOutcome::Failed("reconcile is unix-only".into())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use concerto_api::state_file::{StateEntry, StateFile, STATE_FILE_SCHEMA_VERSION};
    use std::time::Duration;
    use tokio::process::Command;

    /// Spawn a long-running `sleep` child, write a state file naming
    /// its pid, run reconcile, assert the child is dead and the file
    /// is empty.
    #[tokio::test]
    async fn reconcile_kills_orphan_and_truncates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");

        let mut child = Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn sleep");
        let pid = child.id().expect("pid");

        let entry = StateEntry {
            pid,
            // pgid is unset here — exercises the kill(pid) fallback,
            // since `sleep` wasn't spawned under setsid.
            pgid: None,
            port: 8100,
            model_id: "model-a".into(),
            gpu_id: 0,
            started_at: chrono::Utc::now(),
        };
        let file = StateFile {
            schema_version: STATE_FILE_SCHEMA_VERSION,
            entries: vec![entry],
        };
        file.write_to(&path).await.expect("write");

        let report = reconcile(&path).await.expect("reconcile");
        assert_eq!(report.killed, 1, "should report 1 kill: {report:?}");
        assert_eq!(report.stale, 0);
        assert_eq!(report.failed, 0);

        // Reap the dead child so it doesn't leave a zombie.
        let exit = tokio::time::timeout(Duration::from_secs(2), child.wait())
            .await
            .expect("child should exit within 2s of SIGKILL");
        assert!(exit.is_ok(), "wait should succeed");

        let after = StateFile::read_from(&path).await.expect("re-read");
        assert!(
            after.entries.is_empty(),
            "state file should be truncated after reconcile"
        );
    }

    #[tokio::test]
    async fn reconcile_handles_stale_dead_pid() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");

        // Spawn `true` — exits immediately. After we wait for it,
        // its pid becomes either reused or unknown — kill(pid, 0)
        // returns ESRCH.
        let mut child = Command::new("true").spawn().expect("spawn true");
        let pid = child.id().expect("pid");
        let _ = child.wait().await;

        // Tight window before the OS recycles the pid; should be
        // ESRCH on most systems for at least a few seconds.
        let entry = StateEntry {
            pid,
            pgid: None,
            port: 8100,
            model_id: "model-a".into(),
            gpu_id: 0,
            started_at: chrono::Utc::now(),
        };
        let file = StateFile {
            schema_version: STATE_FILE_SCHEMA_VERSION,
            entries: vec![entry],
        };
        file.write_to(&path).await.expect("write");

        let report = reconcile(&path).await.expect("reconcile");
        // Either "stale" (kernel said ESRCH) or "killed" (pid got
        // reused by some other process and we just SIGKILLed an
        // unrelated victim — extremely unlikely in a tempdir test
        // but technically possible). Accept both.
        assert_eq!(
            report.killed + report.stale,
            1,
            "exactly one entry should resolve: {report:?}"
        );
        assert_eq!(report.failed, 0);
    }

    #[tokio::test]
    async fn reconcile_no_op_on_empty_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state.json");
        // No file written — reconcile should be a no-op.
        let report = reconcile(&path).await.expect("reconcile");
        assert_eq!(report, ReconcileReport::default());
    }
}
