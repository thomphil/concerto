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
        }
    }

    /// Set a custom startup timeout (how long `launch` will wait for a newly
    /// spawned backend to become healthy).
    pub fn with_startup_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
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
        let mut command = build_command(spec, gpu_id, port);
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
            // tokio::process::Child::kill sends SIGKILL on Unix; a cleaner
            // SIGTERM-first shutdown would need libc/nix. SIGKILL is
            // adequate for MVP — inference engines do no persistent work.
            child.kill().await?;
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
/// Checks `CONCERTO_PYTHON` first so operators can point at a virtualenv
/// (e.g. `/root/vllm-venv/bin/python`). Falls back to plain `"python"`.
fn python_binary() -> String {
    std::env::var("CONCERTO_PYTHON").unwrap_or_else(|_| "python".to_string())
}

/// Build (but do not spawn) the [`Command`] that would launch a backend for
/// `spec` on `gpu_id`, listening on `port`.
///
/// This is exposed publicly so tests can assert on the command shape without
/// spawning real processes. The `CUDA_VISIBLE_DEVICES` environment variable
/// is set to `gpu_id` so the child sees exactly one device.
pub fn build_command(spec: &ModelSpec, gpu_id: GpuId, port: u16) -> Command {
    let port_str = port.to_string();
    let weight = spec.weight_path.as_str();

    let mut command = match &spec.engine {
        EngineType::Vllm => {
            let mut c = Command::new(python_binary());
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
            let mut c = Command::new(python_binary());
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
