//! Real process-spawning implementation of [`BackendManager`].
//!
//! [`ProcessBackendManager`] turns a [`ModelSpec`] into an actual child
//! process running the requested inference engine (vLLM, llama.cpp, SGLang,
//! or our mock inference backend for testing). It owns the child handles and
//! is responsible for stopping them again.
//!
//! Tests in this crate never spawn real processes — they use either
//! `MockBackendManager` or exercise the pure [`build_command`] helper, which
//! builds the `Command` that would be spawned but never runs it.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use concerto_core::{EngineType, GpuId, ModelSpec};
use tokio::process::{Child, Command};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::manager::{BackendError, BackendHandle, BackendManager};
use crate::port_alloc::PortAllocator;

/// Default grace period between SIGTERM and SIGKILL when terminating a
/// backend process group. Operators can override via
/// [`ProcessBackendManager::with_termination_grace`] (the CLI wires this
/// from `routing.eviction_grace_period_secs`). Five seconds is enough
/// time for a well-behaved engine to flush KV cache and exit cleanly,
/// while still bounding shutdown latency for misbehaving ones.
pub const DEFAULT_TERMINATION_GRACE: Duration = Duration::from_secs(5);

/// How often `kill_process_group` polls to see if the group has fully
/// exited after SIGTERM, before falling through to SIGKILL.
const TERMINATION_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Default startup timeout — how long we'll wait for a freshly spawned
/// backend to become healthy before giving up.
pub const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

/// How often to poll `/health` during startup.
const STARTUP_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Per-request timeout applied to every HTTP call we make to a backend.
/// Keeps startup polling fast and bounds runtime health checks.
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// A [`BackendManager`] that spawns real inference-engine processes.
pub struct ProcessBackendManager {
    ports: PortAllocator,
    http: reqwest::Client,
    children: RwLock<HashMap<u32, Child>>,
    startup_timeout: Duration,
    termination_grace: Duration,
}

impl ProcessBackendManager {
    /// Create a new `ProcessBackendManager` with the default port allocator
    /// and startup timeout.
    pub fn new() -> Self {
        Self::with_port_allocator(PortAllocator::new())
    }

    /// Create a new `ProcessBackendManager` using a custom [`PortAllocator`].
    pub fn with_port_allocator(ports: PortAllocator) -> Self {
        Self {
            ports,
            http: reqwest::Client::new(),
            children: RwLock::new(HashMap::new()),
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
            termination_grace: DEFAULT_TERMINATION_GRACE,
        }
    }

    /// Set a custom startup timeout (how long `launch` will wait for a newly
    /// spawned backend to become healthy).
    pub fn with_startup_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
        self
    }

    /// Set the SIGTERM → SIGKILL grace period applied when stopping a
    /// backend's process group. Wired from `routing.eviction_grace_period_secs`
    /// in production.
    pub fn with_termination_grace(mut self, grace: Duration) -> Self {
        self.termination_grace = grace;
        self
    }

    /// Poll the backend's health endpoint until it returns 200 or the
    /// startup timeout elapses.
    async fn wait_until_healthy(&self, port: u16, path: &str) -> Result<(), BackendError> {
        let deadline = Instant::now() + self.startup_timeout;
        let url = health_url(port, path);
        loop {
            if Instant::now() >= deadline {
                return Err(BackendError::HealthCheckTimeout);
            }
            // Connection refused / DNS / timeout are all expected during
            // startup — keep polling at `debug` level so a normal cold start
            // doesn't spam `warn`.
            match self
                .http
                .get(&url)
                .timeout(HTTP_REQUEST_TIMEOUT)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                Ok(resp) => {
                    debug!(port, status = %resp.status(), "backend not yet healthy");
                }
                Err(err) => {
                    debug!(port, error = %err, "backend health check error (still starting)");
                }
            }
            tokio::time::sleep(STARTUP_POLL_INTERVAL).await;
        }
    }
}

impl Default for ProcessBackendManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BackendManager for ProcessBackendManager {
    async fn launch(&self, spec: &ModelSpec, gpu_id: GpuId) -> Result<BackendHandle, BackendError> {
        let port = self.ports.allocate().ok_or(BackendError::NoFreePort)?;
        let mut command = build_command(spec, gpu_id, port, None);
        configure_session_leader(&mut command);
        let health_path = default_health_path(&spec.engine);
        info!(model_id = %spec.id, %gpu_id, port, engine = ?spec.engine, health_path = %health_path, "spawning backend process");

        let mut child = command.spawn().map_err(|err| {
            self.ports.release(port);
            BackendError::LaunchFailed(format!("failed to spawn {:?}: {err}", spec.engine))
        })?;

        let Some(pid) = child.id() else {
            // Child exited before we could read its pid — clean up and bail.
            let _ = child.kill().await;
            self.ports.release(port);
            return Err(BackendError::LaunchFailed(format!(
                "child process for model {} exited before reporting a pid",
                spec.id
            )));
        };

        // Wait for the backend to become healthy; on failure, kill the child
        // so we don't leak processes.
        if let Err(err) = self.wait_until_healthy(port, &health_path).await {
            warn!(pid, port, "backend did not become healthy; killing child");
            let _ = child.kill().await;
            self.ports.release(port);
            return Err(err);
        }

        self.children.write().await.insert(pid, child);
        let handle = BackendHandle {
            pid,
            port,
            model_id: spec.id.clone(),
            gpu_id,
            health_path,
        };
        info!(model_id = %spec.id, pid, port, "backend ready");
        Ok(handle)
    }

    async fn stop(&self, handle: &BackendHandle) -> Result<(), BackendError> {
        if let Some(mut child) = self.children.write().await.remove(&handle.pid) {
            info!(pid = handle.pid, port = handle.port, "stopping backend");
            // Send SIGTERM to the entire process group, give the engine
            // up to `termination_grace` to flush KV cache and exit, then
            // SIGKILL anything still running. The child became its own
            // process-group leader at spawn time (see
            // `configure_session_leader`), so `pgid == handle.pid` and
            // `killpg(handle.pid, ...)` reaches the parent and every
            // descendant in one syscall — no `pkill -P` race window.
            kill_process_group(handle.pid, self.termination_grace).await;
            // Wait for the parent to be reaped. SIGKILL was already sent
            // by `kill_process_group` if it didn't exit gracefully, so
            // this should be near-instant.
            let _ = child.wait().await;
        } else {
            warn!(pid = handle.pid, "stop called on unknown backend handle");
        }
        self.ports.release(handle.port);
        Ok(())
    }

    async fn health_check(&self, handle: &BackendHandle) -> bool {
        self.http
            .get(health_url(handle.port, &handle.health_path))
            .timeout(HTTP_REQUEST_TIMEOUT)
            .send()
            .await
            .map(|resp| resp.status().is_success())
            .unwrap_or(false)
    }
}

/// Configure `command` so the spawned child becomes its own session
/// leader (and therefore its own process-group leader) via `setsid(2)`.
/// On non-Unix targets this is a no-op — the whole crate is unix-only
/// in practice.
///
/// The intent is that every descendant of the backend inherits the
/// child's pgid, so `killpg(child_pid, SIGTERM)` reaches the entire
/// tree atomically. This replaces the `pkill -P` workaround that
/// shipped in Sprint 2 (R13 in ROADMAP §11), which had a race window
/// between killing direct children and killing the parent during which
/// new children could spawn and become orphans.
#[cfg(unix)]
fn configure_session_leader(command: &mut Command) {
    // `tokio::process::Command::pre_exec` is an inherent method on Unix
    // (re-exported from std), so no `use CommandExt` is needed.
    //
    // SAFETY: `setsid(2)` is async-signal-safe and standard practice in
    // pre-exec hooks. We do not allocate, lock, or call any
    // non-async-signal-safe libc function. A failure here surfaces as a
    // spawn error to the caller and the child is reaped by the kernel.
    unsafe {
        command.pre_exec(|| {
            nix::unistd::setsid()
                .map(|_| ())
                .map_err(|errno| std::io::Error::from_raw_os_error(errno as i32))
        });
    }
}

#[cfg(not(unix))]
fn configure_session_leader(_command: &mut Command) {
    // Non-Unix targets are not supported in production. The whole crate
    // builds on Linux + macOS dev hosts; Windows would need a different
    // process-group story (job objects).
}

/// Send SIGTERM to the process group led by `pgid` (which equals the
/// pid of the child we spawned, thanks to [`configure_session_leader`]),
/// poll for the group to exit, and fall through to SIGKILL once the
/// grace period elapses.
///
/// All errors are best-effort — a missing process group (ESRCH) is the
/// normal case when a graceful engine has already self-terminated. The
/// goal is "no surviving processes", not "every syscall succeeded".
#[cfg(unix)]
async fn kill_process_group(pgid: u32, grace: Duration) {
    use nix::errno::Errno;
    use nix::sys::signal::{killpg, Signal};
    use nix::unistd::Pid;

    let pgid = Pid::from_raw(pgid as i32);

    // 1. Polite SIGTERM first.
    match killpg(pgid, Signal::SIGTERM) {
        Ok(()) => debug!(?pgid, "sent SIGTERM to backend process group"),
        Err(Errno::ESRCH) => {
            debug!(?pgid, "process group already gone before SIGTERM");
            return;
        }
        Err(e) => {
            warn!(?pgid, error = %e, "killpg(SIGTERM) failed");
        }
    }

    // 2. Poll until the group is gone or the grace period elapses.
    let deadline = Instant::now() + grace;
    loop {
        // `kill(pgid, 0)` is a probe — returns ESRCH if no process in
        // the group still exists. We use the negative-pgid form via
        // killpg with signal 0.
        match killpg(pgid, None) {
            Err(Errno::ESRCH) => {
                debug!(?pgid, "process group exited cleanly within grace period");
                return;
            }
            _ => {
                if Instant::now() >= deadline {
                    break;
                }
                tokio::time::sleep(TERMINATION_POLL_INTERVAL).await;
            }
        }
    }

    // 3. Force-kill anything still running.
    match killpg(pgid, Signal::SIGKILL) {
        Ok(()) => warn!(?pgid, "process group did not exit on SIGTERM; sent SIGKILL"),
        Err(Errno::ESRCH) => {
            // Race: exited between the last probe and the SIGKILL. Fine.
        }
        Err(e) => {
            warn!(?pgid, error = %e, "killpg(SIGKILL) failed; process group may leak");
        }
    }
}

#[cfg(not(unix))]
async fn kill_process_group(_pgid: u32, _grace: Duration) {
    // See `configure_session_leader` — non-Unix is unsupported in production.
}

fn health_url(port: u16, path: &str) -> String {
    format!("http://127.0.0.1:{port}{path}")
}

/// The HTTP path a backend exposes for health probes. Built-in engines all use
/// `/health`; `EngineType::Custom` carries its own `health_endpoint`.
pub(crate) fn default_health_path(engine: &EngineType) -> String {
    match engine {
        EngineType::Custom {
            health_endpoint, ..
        } => health_endpoint.clone(),
        _ => "/health".to_string(),
    }
}

/// Resolve the Python interpreter for engines that launch via `python -m …`.
///
/// If `python_override` is provided, use that directly. Otherwise checks
/// `CONCERTO_PYTHON` first so operators can point at a virtualenv
/// (e.g. `/root/vllm-venv/bin/python`). Falls back to plain `"python"`.
fn python_binary(python_override: Option<&str>) -> String {
    if let Some(path) = python_override {
        return path.to_string();
    }
    std::env::var("CONCERTO_PYTHON").unwrap_or_else(|_| "python".to_string())
}

/// Build (but do not spawn) the [`Command`] that would launch a backend for
/// `spec` on `gpu_id`, listening on `port`.
///
/// This is exposed publicly so tests can assert on the command shape without
/// spawning real processes. The `CUDA_VISIBLE_DEVICES` environment variable
/// is set to `gpu_id` so the child sees exactly one device.
///
/// `python_override` lets callers (including tests) supply the Python
/// interpreter path directly, bypassing the `CONCERTO_PYTHON` env var
/// lookup. Production callers pass `None`.
pub fn build_command(
    spec: &ModelSpec,
    gpu_id: GpuId,
    port: u16,
    python_override: Option<&str>,
) -> Command {
    let port_str = port.to_string();
    let weight = spec.weight_path.as_str();

    let mut command = match &spec.engine {
        EngineType::Vllm => {
            let mut c = Command::new(python_binary(python_override));
            c.args([
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                weight,
                "--served-model-name",
                &spec.id.0,
                "--port",
                &port_str,
            ]);
            c
        }
        EngineType::LlamaCpp => {
            let mut c = Command::new("llama-server");
            c.args(["-m", weight, "--port", &port_str]);
            c
        }
        EngineType::Sglang => {
            let mut c = Command::new(python_binary(python_override));
            c.args([
                "-m",
                "sglang.launch_server",
                "--model-path",
                weight,
                "--served-model-name",
                &spec.id.0,
                "--port",
                &port_str,
            ]);
            c
        }
        EngineType::Mock => {
            let mut c = Command::new("mock-inference-backend");
            c.args(["--port", &port_str]);
            c
        }
        EngineType::Custom {
            command: program,
            args,
            health_endpoint: _,
        } => {
            let mut c = Command::new(program);
            // Substitute `{port}` tokens in the user-supplied args. If the
            // user didn't include a placeholder, append `--port <port>` for
            // them so the common case just works.
            let mut had_port_token = false;
            let substituted: Vec<String> = args
                .iter()
                .map(|a| {
                    if a.contains("{port}") {
                        had_port_token = true;
                        a.replace("{port}", &port_str)
                    } else {
                        a.clone()
                    }
                })
                .collect();
            c.args(substituted);
            if !had_port_token {
                c.args(["--port", &port_str]);
            }
            c
        }
    };

    command.args(&spec.engine_args);
    command.env("CUDA_VISIBLE_DEVICES", gpu_id.0.to_string());
    command
}

#[cfg(all(test, unix))]
mod process_group_tests {
    use super::*;
    use nix::errno::Errno;
    use nix::sys::signal::{kill, killpg};
    use nix::unistd::Pid;
    use tokio::time::sleep;

    fn pid_alive(pid: i32) -> bool {
        // signal 0 = existence check; ESRCH means the pid is gone.
        match kill(Pid::from_raw(pid), None) {
            Ok(()) => true,
            Err(Errno::ESRCH) => false,
            Err(_) => true,
        }
    }

    /// Spawn a small shell pipeline that forks two child sleepers, with
    /// `setsid` applied. Killing the process group must reap all three
    /// pids; the old `pkill -P parent` workaround had a race window where
    /// new children could spawn between the pkill and the parent kill.
    #[tokio::test]
    async fn killpg_reaps_parent_and_children() {
        // Two backgrounded `sleep 60`s under a `bash -c '... wait'` parent.
        // Using `bash -c` (not `sh -c`) because POSIX sh doesn't reliably
        // expose the child pids via `$!` on every distro; bash is
        // ubiquitous on Linux/macOS dev hosts and on the CI runners we
        // care about.
        let mut command = Command::new("bash");
        command
            .arg("-c")
            .arg("sleep 60 & echo $! ; sleep 60 & echo $! ; wait")
            .stdout(std::process::Stdio::piped());
        configure_session_leader(&mut command);

        let mut child = command.spawn().expect("bash spawn");
        let parent_pid = child.id().expect("parent pid") as i32;

        // Read the two child pids that bash echoed on stdout.
        use tokio::io::AsyncReadExt;
        let mut stdout = child.stdout.take().expect("piped stdout");
        let mut buf = String::new();
        // Bash buffers its echo, so give the second sleep time to spawn.
        sleep(Duration::from_millis(150)).await;
        // Read what's available without blocking forever.
        let mut chunk = vec![0u8; 64];
        let n = tokio::time::timeout(Duration::from_secs(2), stdout.read(&mut chunk))
            .await
            .expect("read from bash stdout did not time out")
            .expect("read from bash stdout");
        buf.push_str(&String::from_utf8_lossy(&chunk[..n]));
        let pids: Vec<i32> = buf
            .lines()
            .filter_map(|l| l.trim().parse::<i32>().ok())
            .collect();
        assert_eq!(
            pids.len(),
            2,
            "expected 2 child pids on bash stdout, got: {buf:?}"
        );

        // Sanity check: parent + both children are alive.
        assert!(pid_alive(parent_pid), "parent should be alive");
        for pid in &pids {
            assert!(pid_alive(*pid), "child {pid} should be alive");
        }

        // Confirm the parent is its own pgid leader (setsid worked).
        let pgid = nix::unistd::getpgid(Some(Pid::from_raw(parent_pid)))
            .expect("getpgid(parent)")
            .as_raw();
        assert_eq!(
            pgid, parent_pid,
            "after setsid, parent.pgid should equal parent.pid"
        );

        // Kill the group with a tight grace period.
        kill_process_group(parent_pid as u32, Duration::from_secs(2)).await;

        // Reap the parent so the kernel doesn't leave a zombie.
        let _ = child.wait().await;

        // Give the kernel a beat to finish reaping the children. They
        // were SIGTERMed by killpg and bash's `wait` reaps them, but if
        // bash exited before the children we may need a moment.
        sleep(Duration::from_millis(100)).await;

        for pid in &pids {
            assert!(
                !pid_alive(*pid),
                "child {pid} should be reaped after killpg on its group"
            );
        }
        assert!(
            !pid_alive(parent_pid),
            "parent should be reaped after killpg + wait"
        );

        // Final probe: the entire process group should be ESRCH.
        let pgid_pid = Pid::from_raw(parent_pid);
        assert!(
            matches!(killpg(pgid_pid, None), Err(Errno::ESRCH)),
            "process group should be gone after kill_process_group"
        );
    }

    #[tokio::test]
    async fn kill_process_group_returns_quickly_when_group_already_dead() {
        // If the engine self-terminates before stop() is called, the
        // killpg path must still return promptly without panicking.
        // Spawn a `true` (exits immediately) under setsid, give it a
        // moment to die, then call kill_process_group — it should
        // ESRCH on the SIGTERM and return without sleeping.
        let mut command = Command::new("true");
        configure_session_leader(&mut command);
        let mut child = command.spawn().expect("true spawn");
        let pid = child.id().expect("pid");
        let _ = child.wait().await;
        // Allow kernel to fully reap the dead pid.
        sleep(Duration::from_millis(50)).await;

        let started = std::time::Instant::now();
        kill_process_group(pid, Duration::from_secs(5)).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "ESRCH path should short-circuit; took {elapsed:?}"
        );
    }
}
