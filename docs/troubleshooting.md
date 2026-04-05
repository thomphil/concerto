# Troubleshooting

_Common failure modes when running Concerto and how to resolve them. This guide is maintained alongside the codebase; as real-hardware validation progresses, more entries will be added from field experience._

## Cold-start timeouts on 7B+ models

**Symptom:** First request to a model returns 504 Gateway Timeout with `{"kind":"load_timeout"}`. Concerto logs show `backend did not become healthy within the startup timeout`.

**Cause:** vLLM's startup time for a 7B model on a single A4000 is 60–90 seconds — frequently longer than the default `cold_start_timeout_secs = 120` once disk I/O and CUDA initialisation are factored in.

**Fix:** raise the timeout in `[routing]`:

```toml
[routing]
cold_start_timeout_secs = 240
```

For environments with heavier models, 300 is a safer ceiling. The corollary: if you're serving a model where 4 minutes to first token is unacceptable, pin it with `pin = true` so it stays warm once loaded.

## VRAM accounting drift over time

**Symptom:** `nvidia-smi` reports several GB more VRAM in use than Concerto's `/status` endpoint admits. Drift grows with uptime.

**Cause:** vLLM's KV cache grows with active sequence length and concurrency beyond the static `vram_required` value in config. Concerto's `ClusterState` tracks the config number, not the live NVML reading.

**Fix:** set `vram_required` with 20–30% headroom above the configured `--max-model-len`. For a 7B model at 4096 tokens on A4000, bumping `vram_required` from `14GB` to `17GB` in the model entry is a reasonable v0.1 workaround. NVML-driven reconciliation lands in Sprint 3.

## `CUDA_VISIBLE_DEVICES` not taking effect

**Symptom:** A backend spawned for GPU 1 shows up on GPU 0 in `nvidia-smi`. Placement decisions are correct but the child process ignores them.

**Cause:** `CUDA_VISIBLE_DEVICES` must be set in the child process environment _before_ CUDA is initialised. Concerto sets it via `Command::env` before `spawn`, which is correct — but certain engine wrappers (Python venv activation scripts, Docker entrypoint scripts that `source` a profile) can re-initialise CUDA with different visibility.

**Fix:** call the engine binary directly rather than through a wrapper. For vLLM, use `python -m vllm.entrypoints.openai.api_server` rather than a `start.sh` that activates a venv. To verify what the child actually saw, inspect its process environment: `cat /proc/<pid>/environ | tr '\0' '\n' | grep CUDA`.

## Orphan backend processes after SIGTERM

**Symptom:** `systemctl stop concerto` returns promptly but `pgrep -f vllm` still lists workers, or VRAM isn't reclaimed. Next concerto startup runs into port conflicts.

**Cause:** vLLM spawns its own worker subprocesses. A SIGTERM to the concerto parent doesn't propagate to grandchildren by default. `tokio::process::Child::kill` on Unix sends SIGKILL to the direct child only.

**Fix:** run concerto under systemd with `KillMode=control-group` (see [`deployment.md`](deployment.md)) so the entire cgroup is signalled. As a one-off: `pkill -f vllm.entrypoints` after stopping concerto, then verify with `pgrep -f vllm`. Proper process-group kill lands in Sprint 3 (tracked against ROADMAP §3 problem 6).

## Port conflicts with existing services

**Symptom:** `launch` fails repeatedly with `BackendError::LaunchFailed` mentioning "address already in use", or every backend slot shows the same failure at startup.

**Cause:** something else on the host is listening inside Concerto's port allocation range. Defaults are `port_range_start = 8100` and `port_range_end = 9000`; common culprits are Prometheus Node Exporter (9100), Python dev servers, and cached docker-compose stacks.

**Fix:** move Concerto's range to a clear window:

```toml
[routing]
port_range_start = 18100
port_range_end = 19000
```

Verify the new range is clean with `ss -tln | awk '{print $4}' | grep -oP ':\K\d+$' | sort -u`.

## All GPUs reported unhealthy

**Symptom:** `GET /ready` returns 503. `GET /status` shows every GPU with `"health": "Unhealthy"`. Chat requests return 503 with `{"kind":"backend_unavailable"}`.

**Cause:** one of (a) GPU temperature exceeds `max_degraded_temperature` (default 85°C), (b) uncorrected ECC errors detected via NVML, (c) NVML itself is unreachable (driver not loaded, `libnvidia-ml.so.1` not installed).

**Fix:** start with `nvidia-smi` to check temperature and ECC status directly. If those are clean, verify NVML reachability — on Debian/Ubuntu `apt install libnvidia-ml1` resolves the common missing-library case. On RHEL/Rocky, install the NVIDIA datacenter driver package. If your GPUs legitimately run hotter than defaults, tune `max_healthy_temperature` and `max_degraded_temperature` in `[routing]`.

## Graceful shutdown hanging

**Symptom:** After SIGTERM, concerto takes more than 30 seconds to exit, or blocks indefinitely. systemd eventually SIGKILLs it.

**Cause:** in-flight streaming requests don't complete within the drain window, or a backend has entered an unresponsive state that swallows SIGKILL (rare with mock backends; almost always a real inference engine bug).

**Fix:** inspect concerto's logs for `stopping backends` — if you see that but the process never exits, something is blocking on an in-flight stream. Sprint 3 will add a proper drain tracker with a bounded deadline. For now, `systemctl kill -s KILL concerto` is safe — Concerto has no persistent state between restarts.

## Streaming responses cut off mid-token

**Symptom:** Client receives partial SSE chunks, then the connection drops. Backend logs show no errors on the upstream side.

**Cause:** the router decided to evict the model mid-stream to make room for a higher-priority request. Concerto v0.1 applies a best-effort grace period (`eviction_grace_period_secs`, default 30) before killing the evicted backend, but fully-hardened streaming-during-eviction ships in Sprint 3 (tracked against ROADMAP §3 problem 5).

**Fix:** pin the model with `pin = true` in its `[[models]]` entry if it sits on a hot streaming path. Increase `eviction_grace_period_secs` if normal stream duration regularly exceeds 30 seconds. As a diagnostic, the `/status` endpoint will show `stopped_count` climbing under eviction pressure.

---

**Found an issue not covered here?** Open an issue at
<https://github.com/thomphil/concerto/issues>. Please include the relevant log
lines, your `concerto.toml` (with sensitive paths redacted), and `nvidia-smi`
output.

**Note:** this guide is drafted from design-time expected-issue analysis.
Entries will be refined and expanded as real-hardware validation produces
field experience. Updates are welcome via PR.
