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

## vLLM eats all my GPU memory and won't share with a second model

**Symptom:** A second model fails to load on a GPU that, by `vram_required` arithmetic, should have plenty of room. `nvidia-smi` shows the first model's vLLM process consuming 90%+ of total VRAM. Concerto refuses to schedule the second model and rejects the request with `backend_unavailable`.

**Cause:** vLLM defaults to `--gpu-memory-utilization=0.90`, which reserves 90% of the GPU's total VRAM for that single instance regardless of the model's actual weight footprint. The reservation includes a generously-sized KV cache that vLLM grows aggressively. Concerto's `vram_required` accounting is correct for the weights, but vLLM is not honouring it.

**Fix:** override the default per model in `concerto.toml` so each instance reserves only its share of the GPU:

```toml
[[models]]
id = "qwen2.5-0.5b"
name = "Qwen 2.5 0.5B"
weight_path = "/srv/models/qwen2.5-0.5b"
vram_required = "2GB"
engine = "VLLM"
engine_args = ["--gpu-memory-utilization", "0.5"]
```

Pick a fraction that divides the GPU between the models you intend to co-locate (`0.5` for two equal-sized models, `0.3` for three, etc.). Sprint 3 §A.6 (optional) may promote this to a first-class `max_vram_fraction` field so Concerto can compute the flag itself.

## I SIGKILL'd Concerto and now there's a python process holding GPU memory

**Symptom:** `concerto` exited via SIGKILL (OOM killer, `kill -9`, panic, hard reboot interrupted). On restart, Concerto either fails to bind a backend port or schedules a model and immediately runs out of VRAM. `nvidia-smi` lists a `python` or `vllm` process you didn't start.

**Cause:** SIGKILL doesn't run any cleanup. Backend processes spawned by Concerto were left running with their VRAM reservations intact. Sprint 3 §A.1 (process-group kill) prevents this on graceful shutdown, but cannot help when the parent dies without a chance to signal anyone — the orphans are now reparented to PID 1.

**Fix:** Sprint 3 §A.3 startup reconcile catches this on next start and reaps anything bound inside Concerto's port range before scheduling new backends. In the meantime, manual recovery:

```sh
nvidia-smi                 # find the orphan PID(s) under "Processes"
pkill -f vllm              # or kill -9 <pid> for non-vLLM engines
nvidia-smi                 # verify VRAM has been released
systemctl start concerto   # or however you launch it
```

If `pkill -f vllm` is too broad for your host (e.g. you also run vLLM standalone), match on the `--port` flag Concerto allocated from `port_range_start..port_range_end`.

## First request after a model goes idle takes 30–90s

**Symptom:** A request to a model that hasn't been used recently hangs for tens of seconds before tokens start arriving. Subsequent requests to the same model are fast.

**Cause:** Concerto unloads idle models to free VRAM for other tenants. The first request after that triggers a full inference-engine cold start: process spawn, weight load from disk, CUDA kernel warmup, KV cache allocation. See [`benchmarks.md`](benchmarks.md#cold-start-latencies) for measured numbers on a 2× RTX A4000 (27–34s for 0.5B–7B models; bigger models take proportionally longer).

This is expected. ROADMAP §11 R3 captures the three-layer mitigation:

1. **User self-selection.** The README's [Good fit / bad fit](../README.md#good-fit--bad-fit) section is explicit that Concerto is for steady-state multi-tenant routing on a single node, not zero-cold-start serverless.
2. **Pin hot models** with `pin = true` per model in `concerto.toml` so they're never evicted. Pinned models stay resident across the eviction policy's lifetime.
3. **Warm pool** in v0.2 — keeping a configurable set of models pre-loaded even when idle, paying the VRAM cost in exchange for cold-start avoidance.

**Fix:** decide which models warrant `pin = true`:

```toml
[[models]]
id = "primary-chat"
name = "Primary chat model"
weight_path = "/srv/models/qwen2.5-7b"
vram_required = "17GB"
engine = "VLLM"
pin = true
```

If pinning isn't an option (e.g. you need more total models than fit at once), accept the cold-start cost or wait for v0.2.

## Streaming completion stopped mid-tokens during shutdown

**Symptom:** A streaming chat completion was in flight when Concerto received SIGTERM, and the client received a truncated SSE stream — partial JSON in the last `data:` chunk, no `data: [DONE]` terminator, connection closed.

**Cause:** Pre-Sprint-3 versions stopped backends without coordinating with the proxy, so any in-flight stream was killed at the TCP layer. Sprint 3 §A.2 ([PR #16](https://github.com/thomphil/concerto/pull/16)) introduced a shutdown-aware proxy that, on receiving the shutdown signal, stops forwarding upstream chunks, emits a final `data: [DONE]` event, and closes the response body cleanly so OpenAI-compatible clients see a well-formed end-of-stream.

**Fix:** upgrade to a Concerto build that includes PR #16 (post-Sprint-3 §A.2). If you observe this on a build that already has §A.2, capture the truncated response, the corresponding server log lines, and your `concerto --version`, and open an issue at <https://github.com/thomphil/concerto/issues>.

## `/metrics` endpoint returns nothing for `concerto_active_backends` before any request lands

**Symptom:** Immediately after launch, scraping `/metrics` shows scheduling-related gauges (`concerto_active_backends`, etc.) absent or with no samples. Prometheus alerts on "no data" rather than reporting zero.

**Cause:** Expected. Concerto's metrics facade follows the standard Prometheus client convention: gauges are not emitted until they are first set. `concerto_active_backends` is set whenever the orchestrator updates the active-backend tally, which doesn't happen until a request triggers scheduling. Counters and histograms behave the same way — they only appear in scrape output once they have been incremented or observed at least once.

**Fix:** none required. Issue any chat completion request to populate the gauges. If you need a "no data = zero" semantic for alerting, configure your Prometheus alerting rules to use `absent()` and treat absence as zero, or set up a synthetic warm-up request in your deployment script.

## `request_timeout_secs` doesn't apply to streaming

**Symptom:** A streaming chat completion runs for many minutes despite `request_timeout_secs = 60` being set in `[routing]`. The middleware appears to be off.

**Cause:** Sprint 3 §A.4 design decision ([PR #15](https://github.com/thomphil/concerto/pull/15)). The timeout middleware bounds the *response future* — the time until the handler returns a response — not the lifetime of the response body. Streaming handlers construct an SSE body quickly and hand it back to axum; from that point the body is streamed by the framework with the middleware already out of the picture. The bound effectively becomes a time-to-first-byte limit for streamed responses.

This is intentional: bounding body lifetime would require killing live streams from the middleware layer, which conflicts with the streaming-during-shutdown contract documented above and with cooperative eviction.

**Fix:** if you need a hard wall-clock cap on streamed responses, terminate from the client side — most OpenAI-compatible client libraries accept a `timeout` or use an HTTP client with a configurable read deadline. For non-streaming completions, `request_timeout_secs` works as you'd expect.

## Backend process orphans after Concerto crash

**Symptom:** Concerto died unexpectedly (panic, SIGSEGV, OOM kill). After the parent exited, `pgrep -f vllm` (or your engine's name) still lists processes, and `nvidia-smi` still shows VRAM consumed.

**Cause:** Pre-Sprint-3 versions spawned each backend in the same process group as the parent and relied on per-`Child` kill signalling. A parent that dies before sending those signals leaves the children orphaned and reparented to PID 1.

**Fix:** Sprint 3 §A.1 (process-group kill — branch `sprint-3/a1-process-group-kill`) makes each backend its own process-group leader via `setsid`-style spawning, so Concerto signals the entire group with one syscall. Combined with `KillMode=control-group` under systemd (see [`deployment.md`](deployment.md)), a SIGKILL on the Concerto unit reaps the whole tree atomically. After §A.3 startup reconcile lands, even an unclean death is recovered automatically on the next start.

If you observe orphans on a build that has §A.1 and §A.3, the orphan is almost certainly a backend started by something other than Concerto — a leftover `vllm serve ...` from manual testing, a competing supervisor, or a stale Docker container. Cross-reference the orphan PID's `--port` flag against `port_range_start..port_range_end` to confirm before reaching for the issue tracker. See also the [Orchestrator State Machine](architecture.md#orchestrator-state-machine) section for the broader invariants Concerto enforces around backend lifecycle.

---

**Found an issue not covered here?** Open an issue at
<https://github.com/thomphil/concerto/issues>. Please include the relevant log
lines, your `concerto.toml` (with sensitive paths redacted), and `nvidia-smi`
output.

**Note:** this guide is drafted from design-time expected-issue analysis.
Entries will be refined and expanded as real-hardware validation produces
field experience. Updates are welcome via PR.
