# Deployment

How to run Concerto as a managed service on a single Linux host with 1–8 GPUs.

> **Note:** This guide references features landing across Sprints 1–3. The `concerto` binary and `/metrics` endpoint are available as of Sprint 2. The `ghcr.io/thomphil/concerto` container image lands in Sprint 3. Examples are provided so operators can prepare ahead of time; verify every command against the version you are running with `concerto --version`.

For an architectural overview of what's running inside the process, see [`architecture.md`](architecture.md). For diagnosing problems once it's deployed, see [`troubleshooting.md`](troubleshooting.md). Configuration field reference lives alongside the schema in [`../README.md`](../README.md) and the annotated `concerto.example.toml` in the repo root.

## Prerequisites

- Linux host (amd64 or aarch64; Ubuntu 22.04 and Debian 12 are the reference platforms).
- NVIDIA driver 535+ and `libnvidia-ml.so.1` available for real GPU monitoring. For development, pass `--mock-gpus N` to bypass NVML entirely.
- `nvidia-container-toolkit` if you plan to use Docker.
- At least one inference engine installed and callable from the `concerto` user's `$PATH`: vLLM (`pip install vllm`), llama.cpp (`llama-server`), or SGLang. Concerto invokes engines via `Command::spawn` — there is no language-runtime assumption beyond what the engine itself needs.
- Pre-downloaded model weights on the local filesystem. Concerto does not download models.
- A `concerto.toml` configuration file (start from `concerto.example.toml` in the repo root).

## Option 1: systemd (recommended for production)

Save the following as `/etc/systemd/system/concerto.service`:

```ini
[Unit]
Description=Concerto inference multiplexer
Documentation=https://github.com/thomphil/concerto
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
# Type=simple is correct for v0.1: Concerto does not call sd_notify, so
# Type=notify would block startup forever. systemd considers the unit
# "started" as soon as the binary is execve'd; the API server starts
# accepting traffic a moment later once tokio + axum are up.
Type=simple
User=concerto
Group=concerto
WorkingDirectory=/var/lib/concerto
ExecStart=/usr/local/bin/concerto --config /etc/concerto/concerto.toml --log-format json
Restart=on-failure
RestartSec=10s
# Cold-start can take 30–90s for the first model on a 7B-class workload
# (see docs/troubleshooting.md). Give the binary plenty of headroom to
# load config, run NVML probing, and bind the listener before systemd
# considers the unit failed.
TimeoutStartSec=120s
StandardOutput=journal
StandardError=journal
# Graceful shutdown — give Concerto time to drain in-flight requests
# before SIGKILL. 60s is sized to cover the default 30s
# `routing.eviction_grace_period_secs` plus slack for the reverse-proxy
# connection drain. A dedicated `routing.shutdown_drain_secs` knob lands
# later in v0.1; raise this further if you tune that knob upwards.
KillSignal=SIGTERM
TimeoutStopSec=60s
# Kill the entire process group so vLLM worker subprocesses are cleaned up too
KillMode=control-group
# Resource limits
LimitNOFILE=65536
# Hardening — adjust ReadWritePaths if your model cache lives elsewhere.
# /var/lib/concerto is the state dir (port-allocator scratch, future
# reconcile snapshots); /var/log/concerto is only needed if you redirect
# stderr there instead of journald.
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/concerto /var/log/concerto
PrivateTmp=yes
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
```

Then:

```sh
sudo useradd -r -s /bin/false -d /var/lib/concerto concerto
sudo mkdir -p /etc/concerto /var/lib/concerto /var/log/concerto
sudo cp concerto.example.toml /etc/concerto/concerto.toml
# Edit /etc/concerto/concerto.toml for your models and GPU layout
sudo cp target/release/concerto /usr/local/bin/concerto
sudo systemctl daemon-reload
sudo systemctl enable --now concerto
sudo systemctl status concerto
journalctl -u concerto -f
```

**Notes:**

- `KillMode=control-group` is the important line. Without it, vLLM worker subprocesses outlive SIGTERM to the concerto parent, you get VRAM leaks, and the next startup hits port conflicts.
- The `concerto` user needs read access to the model weight paths you configured. `setfacl -R -m u:concerto:r /path/to/models` or group permissions are both fine.
- For real NVML monitoring (not `--mock-gpus`), the `concerto` user needs access to `/dev/nvidia*`. On Ubuntu this typically means adding it to the `video` or `render` group depending on your driver version.

## Option 2: Docker Compose

```yaml
# docker-compose.yml
services:
  concerto:
    image: ghcr.io/thomphil/concerto:latest   # published from Sprint 3 onwards
    container_name: concerto
    restart: unless-stopped
    ports:
      - "8000:8000"   # OpenAI-compatible API + /metrics + /status
    volumes:
      - ./concerto.toml:/etc/concerto/concerto.toml:ro
      - /opt/models:/models:ro             # your pre-downloaded model weights
      - concerto-data:/var/lib/concerto
    environment:
      - RUST_LOG=info
    command:
      - concerto
      - --config
      - /etc/concerto/concerto.toml
      - --log-format
      - json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, utility, compute]
    stop_grace_period: 60s

volumes:
  concerto-data:
```

**Notes:**

- Requires `nvidia-container-toolkit` installed on the host. Verify with `docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi` before starting Concerto.
- The `image:` tag does not exist yet; Sprint 3 will publish `ghcr.io/thomphil/concerto:v0.1.0` and `:latest`.
- `stop_grace_period: 60s` matches the systemd `TimeoutStopSec` — it gives Concerto time to drain in-flight streams before `docker compose down` escalates to SIGKILL. Raise it if you tune `routing.eviction_grace_period_secs` above 30s.
- Mount your weights read-only. Concerto never writes to weight paths.
- The `concerto-data` named volume backs the container's `/var/lib/concerto` state dir, mirroring the systemd setup. Keeping it on a Docker volume means upgrades that swap the image preserve any state Concerto's startup reconcile expects to find.
- The `deploy.resources.reservations.devices` block is the modern Compose Spec way to expose GPUs and is what we recommend. The older `runtime: nvidia` short-form still works on hosts where you have configured `"default-runtime": "nvidia"` in `/etc/docker/daemon.json`; if you prefer it, drop the `deploy:` block and add `runtime: nvidia` at the top level of the `concerto` service. Do not use both at once.

## Option 3: Bare metal

For development and evaluation, the simplest path is:

```sh
./target/release/concerto --config concerto.toml
```

Run it under `tmux` or `screen` so it survives SSH disconnects. Not recommended for production — use systemd or Docker so crashes are restarted and logs are captured.

## Prometheus scraping

Concerto exposes a `/metrics` endpoint on the same HTTP port as the OpenAI-compatible API (no separate listener). A minimal Prometheus scrape config:

```yaml
# /etc/prometheus/prometheus.yml (excerpt)
scrape_configs:
  - job_name: concerto
    scrape_interval: 15s
    static_configs:
      - targets: ['concerto-host:8000']
    metrics_path: /metrics
```

**Authentication:** the endpoint is unauthenticated in v0.1, matching the rest of the HTTP surface. If you don't want scrape access exposed alongside your inference API, bind to a private interface, put Concerto behind a reverse proxy, or use an `iptables` / firewall rule to restrict `/metrics` to your Prometheus host. Auth as an axum middleware layer lands in v0.2.

**Cardinality discipline:** Concerto deliberately does *not* emit per-model labels on its counters or histograms. With multi-tenant fine-tunes the model-id space is unbounded, and per-model time series would explode Prometheus' on-disk index. `concerto_requests_total` is labelled by routing `decision` (a closed set of five values, listed below), the GPU gauges are labelled by `gpu` id (1–8), and that's the entire label budget. If you need per-model breakdown, derive it from the structured logs (see [Log shipping](#log-shipping) below) — that's the right tool for high-cardinality dimensions. Do not patch in per-model labels locally; the on-disk index will be the first thing that breaks under sustained load.

Metrics currently emitted (see ROADMAP §8):

- `concerto_requests_total` (counter, labelled by `decision`: `loaded` / `loaded_after_load` / `rejected_backend_unavailable` / `rejected_all_unhealthy` / `error`)
- `concerto_backend_launches_total` (counter, per successful cold-start)
- `concerto_eviction_total` (counter, per eviction performed while satisfying a cold-start)
- `concerto_backend_health_check_failures_total` (counter, per backend dropped by the background health loop)
- `concerto_routing_decision_seconds` (histogram of the pure-logic routing phase)
- `concerto_model_load_duration_seconds` (histogram of cold-start + launch wall time; launcher path only, subscribers on the dedup channel do not double-count)
- `concerto_active_backends` (gauge: number of live backends the orchestrator is tracking)
- `concerto_gpu_memory_used_bytes` (gauge, labelled by `gpu` id; reflects Concerto's cluster bookkeeping, not the driver's ground truth — compare against `nvidia-smi` or a node exporter sample to detect drift)
- `concerto_gpu_memory_total_bytes` (gauge, labelled by `gpu` id)

## Reverse proxy / TLS termination

Concerto v0.1 has no built-in TLS. The HTTP listener speaks plain HTTP/1.1 on the port from `[server]` in `concerto.toml`. For any deployment that touches a network you don't fully trust, terminate TLS in front of Concerto with nginx, Caddy, or Cloudflare Tunnel and proxy plain HTTP over loopback or a private interface. nginx's `proxy_pass` with `proxy_buffering off` (so SSE chunks aren't held back) and Caddy's default `reverse_proxy` directive both work without further tuning; refer to the upstream nginx and Caddy documentation for current best-practice TLS configuration. Auth lives in the same seam — the Sprint 1 axum router has the middleware extension point ready, but bearer-token / OIDC auth as a first-class feature lands in v0.2.

## Log shipping

Concerto already supports structured JSON logging via `--log-format json` (Sprint 1). One JSON object per line, written to stdout, with `tracing` span context preserved as nested fields. The recommended pipeline on a host running under systemd is `journalctl -u concerto -o json` into [`vector`](https://vector.dev) or [`fluent-bit`](https://fluentbit.io); both have native systemd-journal sources and parse Concerto's payloads as-is. Under Docker Compose, the same shippers will tail the container log driver. Concerto does not include a native log exporter — there's no Loki, Elasticsearch, or Datadog plumbing in the binary, by design — and there are no plans to add one in v0.1; the JSON-on-stdout contract is the integration surface, and `vector` / `fluent-bit` are the supported routers from there.

## Upgrading

1. Stop the service: `systemctl stop concerto` or `docker compose down`.
2. Swap the binary or image: `sudo cp target/release/concerto /usr/local/bin/concerto` (or pull the new image).
3. Start again: `systemctl start concerto` / `docker compose up -d`.
4. Verify: `curl localhost:8000/health` and `curl localhost:8000/v1/models`.

Concerto has no persistent state between restarts in v0.1 — every loaded model's VRAM is reclaimed at shutdown and reloaded on demand. If you suspect orphan processes after an upgrade, run `pgrep -f vllm` before and after to confirm a clean slate.

## Security considerations

- **v0.1 has no authentication.** Anyone who can reach `POST /v1/chat/completions` can use your GPUs. Do not expose the port to the public internet directly.
- Deploy behind a reverse proxy (nginx, Caddy, Cloudflare Tunnel) with auth, or bind only to a private interface or VPN. Auth as a middleware layer lands in v0.2 — the extension point is already in the Sprint 1 axum router.
- Model weight paths are read by the `concerto` service user. Store weights on a filesystem with access controls that match your threat model.
- Container deployments: use a read-only root filesystem where possible, and mount models read-only.
- Logs at `info` level include model IDs and backend PIDs. At `debug` they also include request-level metadata. Treat logs as sensitive if your model IDs or prompts are.
