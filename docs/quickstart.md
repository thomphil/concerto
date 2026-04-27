# Quickstart

Zero to two models served on one Linux GPU box, in about fifteen minutes.

This guide assumes you are evaluating Concerto on a development host. It is
deliberately the shortest path that exercises a real cold-start, a real
eviction, and a real graceful shutdown. For production deployment (systemd,
Docker Compose, Prometheus) read [`deployment.md`](deployment.md) afterwards.

## 1. Prerequisites

You need:

- **Linux host** (amd64 or aarch64). Ubuntu 22.04 and Debian 12 are the
  reference platforms.
- **NVIDIA driver 535+** with `libnvidia-ml.so.1` installed. Verify both at
  once:

  ```bash
  nvidia-smi
  ```

  This must succeed and list at least one GPU.
- **At least one GPU with 16 GB of VRAM or more.** Both walkthrough models fit
  on a single 16 GB device; one of the two must be unloaded to load the other,
  which is the entire point of the eviction demo in step 6.
- **Rust 1.75+** if you are building from source (`rustup install stable`).
  Skip if you are using the Docker image.
- **An inference engine on `$PATH`** for the engines you reference in your
  config. The example below uses vLLM:

  ```bash
  pip install vllm
  which vllm   # must resolve
  ```

  llama.cpp (`llama-server`) and SGLang work the same way. Concerto does not
  bundle any engine; it spawns whatever you point it at.
- **Two pre-downloaded HuggingFace model directories** on the local
  filesystem. Concerto does not download models. For example:

  ```bash
  mkdir -p ~/models
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
      --local-dir ~/models/qwen2.5-7b
  huggingface-cli download microsoft/Phi-3-mini-4k-instruct \
      --local-dir ~/models/phi-3-mini
  ```

  Any local directory the engine can load is fine; the paths just have to
  match `weight_path` in the config below.

If you do not have a GPU at hand, every step except cold-start latency works
under `--mock-gpus 2`, which spawns a bundled mock backend instead of vLLM.
That mode is documented at the end of this guide.

## 2. Install

Pick one path. Both produce a `concerto` binary.

### Option A — `cargo install`

```bash
cargo install --git https://github.com/thomphil/concerto \
    --bin concerto \
    --features nvml
```

The `nvml` feature pulls in real NVIDIA Management Library bindings. Without
it the binary requires `--mock-gpus N` at runtime.

When v0.1.0 is published to crates.io, the equivalent will be:

```bash
cargo install concerto-cli --features nvml
```

### Option B — Docker

```bash
docker pull ghcr.io/thomphil/concerto:latest
```

The container ships with the `nvml` feature enabled. To run it you need
`nvidia-container-toolkit` installed on the host; verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

Note that the published image lands as part of the v0.1.0 release. If your
`docker pull` fails with "manifest unknown", fall back to Option A.

## 3. Minimal `concerto.toml`

Save this as `concerto.toml` in your working directory. Two models, one GPU,
LRU eviction. Every field is documented inline so you can see what each line
controls.

```toml
[server]
host = "127.0.0.1"               # bind to loopback for local eval; use 0.0.0.0 in deployment
port = 8000                      # HTTP port for the OpenAI-compatible API and /metrics

[routing]
eviction_policy = "lru"          # "lru" | "lfu" | "size_weighted_lru" — least-recently-used here
cold_start_timeout_secs = 240    # vLLM 7B startup on consumer GPUs is 60-90s; give it headroom

[[models]]
id = "qwen2.5-7b"                # the string clients pass as the OpenAI "model" field
name = "Qwen 2.5 7B"             # human-readable label shown in /status
weight_path = "/home/you/models/qwen2.5-7b"   # absolute path; replace with yours
vram_required = "14GB"           # accounted against GPU capacity at scheduling time
engine = "vllm"                  # "vllm" | "llamacpp" | "sglang" | { custom = ... }
engine_args = ["--dtype", "float16", "--max-model-len", "4096"]

[[models]]
id = "phi-3-mini"
name = "Phi-3 Mini 3.8B"
weight_path = "/home/you/models/phi-3-mini"   # replace with yours
vram_required = "8GB"            # together with qwen2.5-7b this exceeds 16GB on purpose
engine = "vllm"
engine_args = ["--dtype", "float16", "--max-model-len", "4096"]

[[gpus]]
id = 0                           # 0-based GPU index reported by nvidia-smi
```

`14GB + 8GB = 22GB` does not fit on a 16 GB GPU, which is what makes the
eviction demo in step 6 fire. If your card has more VRAM, raise
`vram_required` on either model so they cannot both be resident at once.

A fuller example with multiple GPUs, custom engines, and the pin flag lives
at `concerto.example.toml` in the repository root.

## 4. Start

```bash
concerto --config concerto.toml
```

You should see, within a second or two:

```text
INFO concerto_cli::setup: concerto ready to serve model_count=2 gpu_count=1 addr=127.0.0.1:8000
INFO concerto_api::server: concerto listening addr=127.0.0.1:8000
```

If those two lines do not both appear, something is wrong — see
[`troubleshooting.md`](troubleshooting.md). Common causes: a port already in
use, NVML not visible to the user, or `nvidia-smi` returning zero devices.

No models are loaded yet. Concerto starts cold; the first request to a model
triggers its load.

Useful flags while you are evaluating:

- `--log-level debug` — verbose orchestrator decisions, including routing and
  eviction reasoning.
- `--log-format json` — structured logs, suitable for `jq` and log shippers.
- `--port-override 8001` — override `[server].port` without editing the
  config.
- `--mock-gpus 2` — skip NVML and the real engine entirely; useful on
  non-Linux dev machines.

## 5. Verify

Open a second terminal. The server stays in the foreground in the first.

### Liveness

```bash
curl -s http://127.0.0.1:8000/health
```

Returns `{"status":"ok"}`. This endpoint reports only that the HTTP server is
running; readiness with attached GPUs is at `/ready`.

### Cluster status

```bash
curl -s http://127.0.0.1:8000/status | jq
```

Shows the per-GPU view: total and used VRAM, temperature, health
classification (`Healthy` / `Degraded` / `Unhealthy`), and the list of
currently loaded models (empty on a fresh start).

### A chat completion

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model": "qwen2.5-7b",
          "messages": [{"role": "user", "content": "Reply with one short sentence."}]
        }' | jq
```

The first request to `qwen2.5-7b` blocks while Concerto launches a vLLM
process and waits for it to report ready. Expect 60–120 seconds on consumer
hardware; see [`benchmarks.md`](benchmarks.md) for measured cold-start
numbers on RTX A4000. Subsequent requests hit the warm backend and respond
in tens of milliseconds plus inference time.

The response is OpenAI-compatible:

```text
{
  "id": "...",
  "object": "chat.completion",
  "model": "qwen2.5-7b",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, ...}],
  "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
}
```

Streaming works too — add `"stream": true` and you get an SSE stream.

## 6. Try eviction

You have one GPU and two models that cannot both fit. Send a request to the
second model:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model": "phi-3-mini",
          "messages": [{"role": "user", "content": "Reply with one short sentence."}]
        }' | jq
```

Concerto will:

1. Compute that `phi-3-mini` does not fit alongside `qwen2.5-7b`.
2. Pick `qwen2.5-7b` as the LRU eviction candidate.
3. Stop the qwen backend and reclaim its VRAM.
4. Spawn a vLLM process for `phi-3-mini` and wait for it to become ready.
5. Forward the request.

Observe the change live:

```bash
curl -s http://127.0.0.1:8000/status | jq '.gpus[0].loaded_models'
```

Before the second request: `["qwen2.5-7b"]`. After: `["phi-3-mini"]`.

The Prometheus endpoint records what just happened:

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E 'concerto_(eviction|backend_launches|requests)_total'
```

You should see `concerto_eviction_total` at 1, `concerto_backend_launches_total`
at 2 (one per cold-start), and `concerto_requests_total` broken down by
decision label.

Send a request back to `qwen2.5-7b` and you will pay another cold-start —
this is the explicit tradeoff documented in the README's "Good fit / bad fit"
section. Concerto trades cold-start latency for the ability to serve more
models than fit in resident VRAM.

## 7. Stop

In the first terminal, send SIGTERM (Ctrl-C, or `kill -TERM <pid>` from
elsewhere). Concerto drains gracefully:

```text
INFO concerto_cli::signals: received SIGINT
INFO concerto_api::shutdown: graceful shutdown: notifying background tasks
INFO concerto_api::shutdown: stopping backends count=1
INFO concerto_api::shutdown: graceful shutdown complete
```

The `count=1` reflects whatever was loaded at the moment of shutdown. Each
backend is sent SIGTERM; in-flight streams get a grace window
(`routing.eviction_grace_period_secs`, default 30s) before forcible stop.

If you ran Concerto under systemd or Docker, the same logs land in
`journalctl -u concerto` or `docker logs concerto`. See
[`deployment.md`](deployment.md) for the production lifecycle and the
`KillMode=control-group` setting that prevents orphan vLLM workers.

## Where to next

- **Production deployment.** [`deployment.md`](deployment.md) covers systemd,
  Docker Compose, Prometheus scraping, security posture, and upgrades.
- **When something breaks.** [`troubleshooting.md`](troubleshooting.md) has
  the known failure modes: cold-start timeouts, port conflicts, orphan
  backend processes, VRAM accounting drift, `CUDA_VISIBLE_DEVICES` gotchas.
- **Whether Concerto is the right tool.** The README's
  [Good fit / bad fit](../README.md#good-fit--bad-fit) section is the honest
  short answer. If you need sub-second first-request latency on every model,
  Concerto's cold-start tradeoff is wrong for your workload.
- **Architecture and internals.** [`architecture.md`](architecture.md) is the
  contributor-oriented reference covering the per-crate split, routing rules,
  eviction policies, GPU health classification, and the orchestrator state
  machine.

## Appendix: running without a GPU

You can exercise the full HTTP surface, eviction logic, and shutdown path on
any machine — including macOS — by using the bundled mock backend:

```bash
cargo run -p concerto-cli -- --config concerto.toml --mock-gpus 2
```

This rewrites every model in the config to point at the
`mock-inference-backend` binary and substitutes a deterministic GPU monitor
for NVML. The mock backend produces OpenAI-shaped responses without doing
any inference, so cold-starts are sub-second instead of minute-scale, but
the routing, eviction, metrics, and shutdown paths are the same code that
runs in production.
