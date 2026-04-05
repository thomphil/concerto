# Deployment

How to run Concerto as a managed service on a single Linux host with 1–8 GPUs.

> **Note:** This guide references features landing across Sprints 1–3. The `concerto` binary comes from Sprint 1. The `ghcr.io/thomphil/concerto` container image and `/metrics` endpoint land in Sprint 3. Examples are provided so operators can prepare ahead of time; verify every command against the version you are running with `concerto --version`.

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
Type=simple
User=concerto
Group=concerto
WorkingDirectory=/var/lib/concerto
ExecStart=/usr/local/bin/concerto --config /etc/concerto/concerto.toml --log-format json
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
# Graceful shutdown — give Concerto time to drain in-flight requests before SIGKILL
KillSignal=SIGTERM
TimeoutStopSec=60s
# Kill the entire process group so vLLM worker subprocesses are cleaned up too
KillMode=control-group
# Resource limits
LimitNOFILE=65536
# Hardening — adjust ReadWritePaths if your model cache lives elsewhere
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
      - "8000:8000"
      # - "9090:9090"   # Prometheus metrics — lands in Sprint 3
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
- `stop_grace_period: 60s` matches the systemd `TimeoutStopSec` — it gives Concerto time to drain in-flight streams before `docker compose down` escalates to SIGKILL.
- Mount your weights read-only. Concerto never writes to weight paths.

## Option 3: Bare metal

For development and evaluation, the simplest path is:

```sh
./target/release/concerto --config concerto.toml
```

Run it under `tmux` or `screen` so it survives SSH disconnects. Not recommended for production — use systemd or Docker so crashes are restarted and logs are captured.

## Prometheus scraping

The `/metrics` endpoint lands in Sprint 3. Once it's available, a minimal Prometheus scrape config looks like:

```yaml
# /etc/prometheus/prometheus.yml (excerpt)
scrape_configs:
  - job_name: concerto
    scrape_interval: 15s
    static_configs:
      - targets: ['concerto-host:9090']
    metrics_path: /metrics
```

Core metrics planned for Sprint 3 (see ROADMAP §8):

- `concerto_requests_total` (counter, by decision: `loaded` / `loaded_after_load` / `rejected`)
- `concerto_active_backends` (gauge, by engine)
- `concerto_model_load_duration_seconds` (histogram)
- `concerto_eviction_total` (counter)
- `concerto_gpu_memory_used_bytes` (gauge, by gpu_id)
- `concerto_gpu_memory_total_bytes` (gauge, by gpu_id)
- `concerto_routing_decision_seconds` (histogram)
- `concerto_backend_health_check_failures_total` (counter)

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
