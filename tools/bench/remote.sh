#!/usr/bin/env bash
# tools/bench/remote.sh — one-command Vast.ai bench automation for Concerto
#
# Manages the full remote benchmark lifecycle from a local Mac:
# provisioning, bootstrap, code sync, scenario execution (with live
# streaming), artifact retrieval, and instance lifecycle.
#
# Usage: remote.sh <command> [options]
# Run `remote.sh help` for full usage.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
INSTANCE_FILE="$REPO_ROOT/.vast-instance"
ARTIFACTS_DIR="$REPO_ROOT/bench-artifacts"

# Load .env for credentials (VAST_API_KEY, etc.)
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
LABEL_PREFIX="concerto-bench"
DEFAULT_IMAGE="nvidia/cuda:12.2.0-devel-ubuntu22.04"
DEFAULT_DISK_GB=120

REMOTE_CONCERTO="/root/concerto"
REMOTE_MODELS="/root/models"
REMOTE_VLLM_VENV="/root/vllm-venv"
REMOTE_BIN="$REMOTE_CONCERTO/target/release/concerto"
DEFAULT_SCENARIO="tools/bench/scenarios/sprint-2-validation.yaml"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o LogLevel=ERROR)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ok()    { printf '\033[1;32m OK \033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

preflight_check() {
    local missing=()
    for cmd in vastai jq python3 rsync ssh scp; do
        command -v "$cmd" >/dev/null || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        die "Missing required tools: ${missing[*]}. Install them first."
    fi
}

# ---------------------------------------------------------------------------
# Instance ID resolution
# ---------------------------------------------------------------------------

resolve_instance_id() {
    if [[ -n "${OPT_INSTANCE_ID:-}" ]]; then
        INSTANCE_ID="$OPT_INSTANCE_ID"
    elif [[ -n "${VAST_INSTANCE_ID:-}" ]]; then
        INSTANCE_ID="$VAST_INSTANCE_ID"
    elif [[ -f "$INSTANCE_FILE" ]]; then
        INSTANCE_ID="$(cat "$INSTANCE_FILE")"
    else
        die "No instance ID. Use --instance-id, set VAST_INSTANCE_ID, or run 'bootstrap --new'."
    fi
}

save_instance_id() {
    echo "$INSTANCE_ID" > "$INSTANCE_FILE"
    info "Instance ID $INSTANCE_ID saved to .vast-instance"
}

clear_instance_id() {
    rm -f "$INSTANCE_FILE"
}

# ---------------------------------------------------------------------------
# SSH resolution
# ---------------------------------------------------------------------------

resolve_ssh() {
    local raw
    raw="$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)" || die "Cannot resolve SSH for instance $INSTANCE_ID. Is it running?"
    # vastai ssh-url returns e.g. "ssh://root@ssh3.vast.ai:24528/"
    # Strip protocol prefix and trailing slash
    raw="${raw#ssh://}"
    raw="${raw%/}"
    SSH_HOST="${raw#root@}"
    SSH_PORT="${SSH_HOST##*:}"
    SSH_HOST="${SSH_HOST%:*}"
    SSH_USER="root"
}

remote_exec() {
    ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "$@"
}

remote_exec_tty() {
    ssh "${SSH_OPTS[@]}" -t -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "$@"
}

# ---------------------------------------------------------------------------
# Vast.ai helpers
# ---------------------------------------------------------------------------

get_instance_status() {
    vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | jq -r '.actual_status // "unknown"'
}

get_instance_json() {
    vastai show instance "$INSTANCE_ID" --raw 2>/dev/null
}

wait_for_running() {
    local timeout="${1:-180}"
    local elapsed=0
    local interval=5
    while [[ "$elapsed" -lt "$timeout" ]]; do
        local status
        status="$(get_instance_status)"
        if [[ "$status" == "running" ]]; then
            return 0
        fi
        printf "  waiting for running... (status=%s, %ds)\n" "$status" "$elapsed"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    die "Instance did not reach 'running' within ${timeout}s"
}

# ---------------------------------------------------------------------------
# YAML step filter
# ---------------------------------------------------------------------------

filter_scenario_to_step() {
    local scenario_path="$1"
    local step_selector="$2"

    python3 << PYEOF
import yaml, sys

with open("$scenario_path") as f:
    scenario = yaml.safe_load(f)

steps = scenario["steps"]
selector = """$step_selector"""

# Try numeric index (1-based)
try:
    idx = int(selector) - 1
    if 0 <= idx < len(steps):
        matched = [steps[idx]]
    else:
        print("Step index %s out of range (1-%d)" % (selector, len(steps)), file=sys.stderr)
        sys.exit(1)
except ValueError:
    matched = [s for s in steps if s["name"] == selector]
    if not matched:
        names = [s["name"] for s in steps]
        print("No step named %r. Available: %s" % (selector, names), file=sys.stderr)
        sys.exit(1)

scenario["steps"] = matched
scenario["name"] = scenario["name"] + "-step-" + selector
yaml.dump(scenario, sys.stdout, default_flow_style=False, sort_keys=False)
PYEOF
}

# ---------------------------------------------------------------------------
# cmd_help
# ---------------------------------------------------------------------------

cmd_help() {
    cat <<'USAGE'
Usage: remote.sh <command> [options]

Instance lifecycle:
  bootstrap [--new] [--sha SHA]  Bootstrap instance (deps, build, models)
  start                          Start a stopped instance
  stop                           Stop instance (data preserved, billing paused)
  destroy                        Destroy instance permanently
  reap [--max-age-hours N]       Destroy old labelled instances

Development:
  sync [--rebuild]               Push local code to remote; --rebuild runs cargo
  ssh                            Open interactive SSH session

Benchmarking:
  run [options]                  Run scenario (streams output live)
    --scenario FILE              Scenario YAML (default: sprint-2-validation)
    --step NAME|NUMBER           Run a single step only
    --output NAME                Output directory name
    --detach                     Run in background via nohup
  status                         GPU state, processes, models, disk usage
  fetch [PATH]                   Download artifact tarball
  logs [OUTPUT_DIR]              Pull logs from a (possibly failed) run

Credentials (.env file in repo root, gitignored):
  VAST_API_KEY       Vast.ai API key (optional, vastai CLI also reads ~/.vast_api_key)

Instance resolution (priority order):
  1. --instance-id ID flag
  2. VAST_INSTANCE_ID env var
  3. .vast-instance file (written by bootstrap, gitignored)

Examples:
  remote.sh bootstrap --new                  # Provision + bootstrap fresh instance
  remote.sh bootstrap --instance-id 34264529 # Re-bootstrap existing instance
  remote.sh sync --rebuild                   # Push code + rebuild Rust binary
  remote.sh run                              # Full sprint-2-validation, live output
  remote.sh run --step backend-crash         # Re-run just step 6
  remote.sh run --step 1                     # Re-run just step 1
  remote.sh status                           # What's happening on the box?
  remote.sh logs                             # Pull logs from most recent run
  remote.sh fetch                            # Download latest artifact tarball
  remote.sh stop                             # Pause billing
  remote.sh start                            # Resume
USAGE
}

# ---------------------------------------------------------------------------
# cmd_bootstrap
# ---------------------------------------------------------------------------

cmd_bootstrap() {
    local new_instance=false
    local sha=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --new)       new_instance=true; shift ;;
            --sha)       sha="$2"; shift 2 ;;
            *)           die "Unknown bootstrap option: $1" ;;
        esac
    done

    # Default SHA to current local HEAD
    sha="${sha:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"

    if $new_instance; then
        info "Searching for 2x RTX A4000 offers..."
        vastai search offers \
            'num_gpus=2 gpu_name=RTX_A4000 disk_space>=120 reliability>0.98 inet_up>100' \
            -o 'dph+' | head -20

        echo ""
        read -rp "Enter offer ID to rent (or Ctrl+C to abort): " offer_id
        [[ -z "$offer_id" ]] && die "No offer selected."

        info "Creating instance from offer $offer_id..."
        local label="${LABEL_PREFIX}-$(date +%Y%m%dT%H%M)"
        local result
        result="$(vastai create instance "$offer_id" \
            --image "$DEFAULT_IMAGE" \
            --disk "$DEFAULT_DISK_GB" \
            --ssh \
            --label "$label" \
            --raw)"
        INSTANCE_ID="$(echo "$result" | jq -r '.new_contract')"
        [[ "$INSTANCE_ID" == "null" || -z "$INSTANCE_ID" ]] && die "Failed to create instance: $result"
        save_instance_id

        info "Waiting for instance $INSTANCE_ID to start..."
        wait_for_running 180
    else
        resolve_instance_id
        local status
        status="$(get_instance_status)"
        if [[ "$status" != "running" ]]; then
            info "Instance $INSTANCE_ID is $status, starting..."
            vastai start instance "$INSTANCE_ID"
            wait_for_running 180
        fi
        save_instance_id
    fi

    resolve_ssh
    info "Bootstrapping instance $INSTANCE_ID ($SSH_HOST:$SSH_PORT)..."
    info "  SHA: $sha"

    # Give SSH a moment to become ready after instance start
    sleep 3

    remote_exec "bash -s" <<BOOTSTRAP
set -euo pipefail

echo "==> Installing system packages..."
if ! command -v wrk >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        git curl build-essential pkg-config libssl-dev \
        python3 python3-pip python3-venv python3-dev \
        tmux htop jq wrk rsync ca-certificates
else
    echo "    (already installed)"
fi

echo "==> Installing Rust..."
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
. "\$HOME/.cargo/env"
rustc --version

echo "==> Cloning/updating concerto..."
if [ ! -d /root/concerto ]; then
    git clone https://github.com/thomphil/concerto.git /root/concerto
fi
cd /root/concerto
git fetch origin
git checkout $sha

echo "==> Building concerto (release + nvml)..."
BUILT_SHA=""
[ -f target/release/.concerto-sha ] && BUILT_SHA=\$(cat target/release/.concerto-sha)
CURRENT_SHA=\$(git rev-parse HEAD)
if [ "\$BUILT_SHA" != "\$CURRENT_SHA" ]; then
    cargo build --release -p concerto-cli --features nvml
    echo "\$CURRENT_SHA" > target/release/.concerto-sha
    echo "    built fresh"
else
    echo "    binary up to date"
fi
./target/release/concerto --version

echo "==> Installing bench rig..."
python3 -m pip install --user -e tools/bench 2>&1 | tail -3
export PATH="\$HOME/.local/bin:\$PATH"

echo "==> Setting up vLLM venv..."
if [ ! -d /root/vllm-venv ]; then
    python3 -m venv /root/vllm-venv
    /root/vllm-venv/bin/pip install --upgrade pip
    /root/vllm-venv/bin/pip install vllm huggingface_hub
else
    echo "    (already exists)"
fi

echo "==> Downloading model weights..."
mkdir -p /root/models
for spec in \
    "Qwen/Qwen2.5-0.5B-Instruct:qwen2.5-0.5b" \
    "microsoft/Phi-3-mini-4k-instruct:phi-3-mini" \
    "Qwen/Qwen2.5-7B-Instruct:qwen2.5-7b"; do
    hf_repo="\${spec%%:*}"
    local_dir="\${spec##*:}"
    if [ ! -d "/root/models/\$local_dir" ]; then
        echo "    downloading \$hf_repo -> \$local_dir"
        /root/vllm-venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('\$hf_repo', local_dir='/root/models/\$local_dir')"
    else
        echo "    \$local_dir already present"
    fi
done

echo "==> nvidia-smi:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "==> Bootstrap complete."
BOOTSTRAP

    ok "Bootstrap complete for instance $INSTANCE_ID"
}

# ---------------------------------------------------------------------------
# cmd_sync
# ---------------------------------------------------------------------------

cmd_sync() {
    local rebuild=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --rebuild)  rebuild=true; shift ;;
            *)          die "Unknown sync option: $1" ;;
        esac
    done

    resolve_instance_id
    resolve_ssh
    info "Syncing local code to instance $INSTANCE_ID..."

    rsync -azP --delete \
        --exclude='target/' \
        --exclude='.git/' \
        --exclude='bench-artifacts/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='*.egg-info/' \
        --exclude='.pytest_cache/' \
        --exclude='.mypy_cache/' \
        --exclude='.ruff_cache/' \
        --exclude='.venv/' \
        --exclude='.vast-instance' \
        -e "ssh ${SSH_OPTS[*]} -p $SSH_PORT" \
        "$REPO_ROOT/" \
        "${SSH_USER}@${SSH_HOST}:${REMOTE_CONCERTO}/"

    info "Reinstalling bench rig..."
    remote_exec "cd $REMOTE_CONCERTO && python3 -m pip install --user -e tools/bench 2>&1 | tail -2"

    if $rebuild; then
        info "Rebuilding concerto (--rebuild)..."
        remote_exec "cd $REMOTE_CONCERTO && . \$HOME/.cargo/env && cargo build --release -p concerto-cli --features nvml && git rev-parse HEAD > target/release/.concerto-sha"
        ok "Build complete"
    else
        warn "Rust binary NOT rebuilt. Add --rebuild if you changed Rust code."
    fi

    ok "Sync complete."
}

# ---------------------------------------------------------------------------
# cmd_run
# ---------------------------------------------------------------------------

cmd_run() {
    local scenario="$DEFAULT_SCENARIO"
    local step=""
    local output_name=""
    local detach=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --scenario)  scenario="$2"; shift 2 ;;
            --step)      step="$2"; shift 2 ;;
            --output)    output_name="$2"; shift 2 ;;
            --detach)    detach=true; shift ;;
            *)           die "Unknown run option: $1" ;;
        esac
    done

    resolve_instance_id
    resolve_ssh

    output_name="${output_name:-run-$(date -u +%Y%m%dT%H%M%SZ)}"
    local remote_output="/root/$output_name"
    local remote_scenario="$REMOTE_CONCERTO/$scenario"

    # Step filtering
    if [[ -n "$step" ]]; then
        info "Filtering scenario to step: $step"
        local filtered
        filtered="$(filter_scenario_to_step "$REPO_ROOT/$scenario" "$step")" || exit 1
        local tmpfile
        tmpfile="$(mktemp)"
        echo "$filtered" > "$tmpfile"
        scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$tmpfile" "${SSH_USER}@${SSH_HOST}:/tmp/filtered-scenario.yaml"
        rm "$tmpfile"
        remote_scenario="/tmp/filtered-scenario.yaml"
        warn "Running single step. Ensure prerequisite state (earlier steps) is established."
    fi

    local run_cmd="cd $REMOTE_CONCERTO && \
PATH=$REMOTE_VLLM_VENV/bin:\$PATH \
CONCERTO_PYTHON=$REMOTE_VLLM_VENV/bin/python \
python3 -m concerto_bench run \
    --scenario $remote_scenario \
    --concerto-bin $REMOTE_BIN \
    --models-dir $REMOTE_MODELS \
    --output $remote_output \
    --http-timeout 240 \
    --startup-timeout 60 \
    --log-level info"

    info "Running bench on instance $INSTANCE_ID"
    info "  Scenario: $scenario"
    [[ -n "$step" ]] && info "  Step: $step"
    info "  Output: $remote_output"
    echo ""

    local exit_code=0
    if $detach; then
        info "Launching in background (--detach)..."
        remote_exec "nohup bash -c '$run_cmd' > /root/${output_name}.log 2>&1 &"
        ok "Running in background. Check progress:"
        echo "  remote.sh ssh  (then: tail -f /root/${output_name}.log)"
        echo "  remote.sh logs $remote_output"
        return 0
    fi

    # Stream output live to the local terminal
    remote_exec_tty "bash -c '$run_cmd'" || exit_code=$?

    echo ""
    if [[ "$exit_code" -eq 0 ]]; then
        ok "Run completed successfully."
    else
        warn "Run exited with code $exit_code"
    fi

    # Try to fetch the artifact
    local remote_tarball="${remote_output}.tar.gz"
    if remote_exec "test -f $remote_tarball" 2>/dev/null; then
        info "Fetching artifact..."
        cmd_fetch "$remote_tarball"
    else
        warn "No tarball found at $remote_tarball"
        if [[ "$exit_code" -ne 0 ]]; then
            echo ""
            echo "  Pull logs:     remote.sh logs $remote_output"
            echo "  Interactive:   remote.sh ssh"
        fi
    fi

    return "$exit_code"
}

# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------

cmd_status() {
    resolve_instance_id

    local inst_json
    inst_json="$(get_instance_json)"
    local status
    status="$(echo "$inst_json" | jq -r '.actual_status // "unknown"')"

    info "Instance $INSTANCE_ID"
    echo "  Status:   $status"
    echo "  GPU:      $(echo "$inst_json" | jq -r '(.num_gpus | tostring) + "x " + .gpu_name')"
    echo "  Cost:     \$$(echo "$inst_json" | jq -r '.dph_total // "?"')/hr"
    echo "  Label:    $(echo "$inst_json" | jq -r '.label // "none"')"
    echo ""

    if [[ "$status" != "running" ]]; then
        warn "Instance is not running. Use 'remote.sh start' to resume."
        return 0
    fi

    resolve_ssh

    info "GPU state:"
    remote_exec "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader" 2>/dev/null || echo "  (nvidia-smi unavailable)"
    echo ""

    info "Processes:"
    remote_exec "ps aux | grep -E 'concerto|vllm|mock-inference' | grep -v grep || echo '  (no concerto/vllm processes)'"
    echo ""

    info "Concerto /status:"
    remote_exec "curl -sf http://127.0.0.1:8000/status 2>/dev/null | python3 -m json.tool 2>/dev/null || echo '  (concerto not responding)'"
    echo ""

    info "Tmux sessions:"
    remote_exec "tmux list-sessions 2>/dev/null || echo '  (none)'"
    echo ""

    info "Recent run artifacts:"
    remote_exec "ls -lht /root/run-*.tar.gz /root/sprint-*.tar.gz 2>/dev/null | head -5 || echo '  (none)'"
    echo ""

    info "Disk usage:"
    remote_exec "du -sh $REMOTE_MODELS $REMOTE_CONCERTO/target $REMOTE_VLLM_VENV 2>/dev/null || true"
}

# ---------------------------------------------------------------------------
# cmd_logs
# ---------------------------------------------------------------------------

cmd_logs() {
    local remote_dir="${1:-}"

    resolve_instance_id
    resolve_ssh

    # Find the most recent run directory if not specified
    if [[ -z "$remote_dir" ]]; then
        remote_dir="$(remote_exec "ls -dt /root/run-* /root/sprint-* 2>/dev/null | head -1" 2>/dev/null)" || true
        [[ -z "$remote_dir" ]] && die "No run directories found on remote. Specify a path."
        info "Most recent run: $remote_dir"
    fi

    local run_name
    run_name="$(basename "$remote_dir")"
    local local_dir="$ARTIFACTS_DIR/$run_name"
    mkdir -p "$local_dir"

    info "Pulling logs from $remote_dir..."

    # Download key files
    for file in summary.json manifest.json concerto-stdout.log concerto-stderr.log; do
        scp "${SSH_OPTS[@]}" -P "$SSH_PORT" \
            "${SSH_USER}@${SSH_HOST}:${remote_dir}/${file}" \
            "$local_dir/" 2>/dev/null || true
    done

    # Download step results
    mkdir -p "$local_dir/steps"
    remote_exec "find $remote_dir/steps -name 'result.json' 2>/dev/null" | while read -r rpath; do
        local step_dir
        step_dir="$(basename "$(dirname "$rpath")")"
        mkdir -p "$local_dir/steps/$step_dir"
        scp "${SSH_OPTS[@]}" -P "$SSH_PORT" \
            "${SSH_USER}@${SSH_HOST}:${rpath}" \
            "$local_dir/steps/$step_dir/" 2>/dev/null || true
    done

    ok "Logs downloaded to $local_dir"
    echo ""

    # Print summary
    if [[ -f "$local_dir/summary.json" ]]; then
        info "Summary:"
        python3 -c "
import json
s = json.load(open('$local_dir/summary.json'))
print('  Exit status:   %s' % s.get('exit_status', '?'))
print('  Steps passed:  %s/%s' % (s.get('steps_passed', '?'), s.get('step_count', '?')))
failed = s.get('failed_step_names', [])
if failed:
    print('  Failed steps:  %s' % ', '.join(failed))
"
        echo ""
    fi

    # Print step results
    info "Step results:"
    python3 << PYEOF
import json, glob, os
for path in sorted(glob.glob("$local_dir/steps/*/result.json")):
    r = json.load(open(path))
    dur = r.get("duration_ms", 0) / 1000
    name = r.get("step_name", "?")
    passed = r.get("passed", False)
    mark = "PASS" if passed else "FAIL"
    print("  %-4s %-30s  %.1fs" % (mark, name, dur))
    if not passed:
        for f in r.get("failures", []):
            reason = f if isinstance(f, str) else f.get("failure_reason", "?")
            print("       -> %s" % reason)
PYEOF
    echo ""

    # Tail stderr
    if [[ -f "$local_dir/concerto-stderr.log" ]]; then
        info "Last 50 lines of concerto stderr:"
        tail -50 "$local_dir/concerto-stderr.log"
    fi
}

# ---------------------------------------------------------------------------
# cmd_fetch
# ---------------------------------------------------------------------------

cmd_fetch() {
    local remote_path="${1:-}"

    resolve_instance_id
    resolve_ssh
    mkdir -p "$ARTIFACTS_DIR"

    # Find the latest tarball if not specified
    if [[ -z "$remote_path" ]]; then
        remote_path="$(remote_exec "ls -t /root/*.tar.gz 2>/dev/null | head -1")" || true
        [[ -z "$remote_path" ]] && die "No tarballs found on remote. Specify a path."
        info "Latest tarball: $remote_path"
    fi

    local basename
    basename="$(basename "$remote_path")"
    local local_path="$ARTIFACTS_DIR/$basename"

    info "Fetching $remote_path -> $local_path"
    scp "${SSH_OPTS[@]}" -P "$SSH_PORT" \
        "${SSH_USER}@${SSH_HOST}:${remote_path}" \
        "$local_path"

    # Fetch checksum if available
    local sha_remote="${remote_path}.sha256"
    if remote_exec "test -f $sha_remote" 2>/dev/null; then
        scp "${SSH_OPTS[@]}" -P "$SSH_PORT" \
            "${SSH_USER}@${SSH_HOST}:${sha_remote}" \
            "$ARTIFACTS_DIR/"

        # Verify
        local expected
        expected="$(awk '{print $1}' "$ARTIFACTS_DIR/${basename}.sha256")"
        local actual
        actual="$(shasum -a 256 "$local_path" | awk '{print $1}')"
        if [[ "$expected" == "$actual" ]]; then
            ok "Checksum verified."
        else
            warn "Checksum mismatch! Expected: $expected  Got: $actual"
        fi
    fi

    ok "Artifact: $local_path ($(du -h "$local_path" | awk '{print $1}'))"
    echo "  Summarize: concerto-bench summarize $local_path"
}

# ---------------------------------------------------------------------------
# cmd_start
# ---------------------------------------------------------------------------

cmd_start() {
    resolve_instance_id
    local status
    status="$(get_instance_status)"

    if [[ "$status" == "running" ]]; then
        ok "Instance $INSTANCE_ID already running."
        resolve_ssh
        echo "  SSH: ssh -p $SSH_PORT ${SSH_USER}@${SSH_HOST}"
        return 0
    fi

    info "Starting instance $INSTANCE_ID (was: $status)..."
    vastai start instance "$INSTANCE_ID"
    wait_for_running 180
    resolve_ssh
    ok "Instance $INSTANCE_ID running at $SSH_HOST:$SSH_PORT"
}

# ---------------------------------------------------------------------------
# cmd_stop
# ---------------------------------------------------------------------------

cmd_stop() {
    resolve_instance_id
    info "Stopping instance $INSTANCE_ID..."
    vastai stop instance "$INSTANCE_ID"
    ok "Instance stopped. Data preserved. Billing paused."
    echo "  Resume: remote.sh start"
}

# ---------------------------------------------------------------------------
# cmd_destroy
# ---------------------------------------------------------------------------

cmd_destroy() {
    resolve_instance_id
    echo ""
    warn "This will PERMANENTLY destroy instance $INSTANCE_ID and all its data."
    read -rp "Type 'yes' to confirm: " confirm
    if [[ "$confirm" != "yes" ]]; then
        die "Aborted."
    fi

    vastai destroy instance "$INSTANCE_ID"
    clear_instance_id
    ok "Instance $INSTANCE_ID destroyed."
}

# ---------------------------------------------------------------------------
# cmd_ssh
# ---------------------------------------------------------------------------

cmd_ssh() {
    resolve_instance_id
    resolve_ssh
    info "Connecting to instance $INSTANCE_ID ($SSH_HOST:$SSH_PORT)..."
    exec ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}"
}

# ---------------------------------------------------------------------------
# cmd_reap
# ---------------------------------------------------------------------------

cmd_reap() {
    local max_age_hours=24
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --max-age-hours)  max_age_hours="$2"; shift 2 ;;
            *)                die "Unknown reap option: $1" ;;
        esac
    done

    info "Looking for $LABEL_PREFIX-* instances older than ${max_age_hours}h..."

    local instances
    instances="$(vastai show instances --raw 2>/dev/null)" || die "Cannot list instances"

    local count=0
    local reaped=0

    echo "$instances" | jq -c ".[] | select(.label != null) | select(.label | startswith(\"$LABEL_PREFIX\"))" | while read -r inst; do
        local id label status age_hours
        id="$(echo "$inst" | jq -r '.id')"
        label="$(echo "$inst" | jq -r '.label')"
        status="$(echo "$inst" | jq -r '.actual_status')"
        # duration is in seconds
        age_hours="$(echo "$inst" | jq -r '(.duration // 0) / 3600 | floor')"

        if [[ "$age_hours" -ge "$max_age_hours" ]]; then
            warn "Reaping $id ($label, status=$status, age=${age_hours}h)"
            vastai destroy instance "$id"
        else
            info "  Keeping $id ($label, status=$status, age=${age_hours}h)"
        fi
    done

    ok "Reap complete."
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

main() {
    preflight_check

    # Global flags
    OPT_INSTANCE_ID=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --instance-id)  OPT_INSTANCE_ID="$2"; shift 2 ;;
            *)              break ;;
        esac
    done

    local command="${1:-help}"
    shift || true

    case "$command" in
        help|--help|-h)   cmd_help ;;
        bootstrap)        cmd_bootstrap "$@" ;;
        sync)             cmd_sync "$@" ;;
        run)              cmd_run "$@" ;;
        status)           cmd_status "$@" ;;
        logs)             cmd_logs "$@" ;;
        fetch)            cmd_fetch "$@" ;;
        start)            cmd_start "$@" ;;
        stop)             cmd_stop "$@" ;;
        destroy)          cmd_destroy "$@" ;;
        ssh)              cmd_ssh "$@" ;;
        reap)             cmd_reap "$@" ;;
        *)                die "Unknown command: $command. Run 'remote.sh help' for usage." ;;
    esac
}

main "$@"
