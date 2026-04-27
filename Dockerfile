# syntax=docker/dockerfile:1.7
#
# Concerto — production multi-stage Dockerfile.
#
# Builder: rust:1-bookworm. Builds the `concerto` binary
# (concerto-cli) with the `nvml` feature for GPU monitoring.
#
# Runtime: debian:bookworm-slim. Ships only the runtime
# dependencies (libssl3, ca-certificates, libnvidia-ml1) and
# the stripped binary, under a non-root `concerto` user.
#
# Size budget: <50 MB final image (asserted by
# .github/workflows/docker.yml). Builder cache mounts keep
# rebuilds fast on CI; the `strip` step keeps the runtime
# layer small.
#
# Base image tags are pinned to bookworm. Pin by SHA digest
# in C.3 (release workflow) once the digest is stable.

# ---------- builder ----------
FROM rust:1-bookworm AS builder

# Static-ish build target. We rely on the host's libssl/libnvml
# at runtime, but pinning the target keeps the binary
# reproducible across builders.
ARG TARGET=x86_64-unknown-linux-gnu

WORKDIR /work

# Workspace sources. Copy everything in one shot; cache mounts
# below take care of incremental builds.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY tests ./tests

# BuildKit cache mounts for the cargo registry, git index,
# and target directory. Each mount is shared across builds
# of this Dockerfile, dramatically speeding up CI on warm
# caches while staying invisible to the final image.
#
# After the build completes we copy the binary out of the
# (cache-mounted) target dir to a stable path so the runtime
# stage can `COPY --from=builder` deterministically.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/work/target \
    set -eux; \
    rustup target add "${TARGET}"; \
    cargo build \
        --release \
        --locked \
        --features nvml \
        -p concerto-cli \
        --target "${TARGET}"; \
    install -m 0755 \
        "/work/target/${TARGET}/release/concerto" \
        /usr/local/bin/concerto; \
    strip /usr/local/bin/concerto

# ---------- runtime ----------
FROM debian:bookworm-slim AS runtime

# Runtime dependencies:
# - libssl3:         reqwest's default native-tls backend
# - ca-certificates: TLS root store for outbound HTTPS
#
# We deliberately do NOT install libnvidia-ml1 from Debian's
# non-free component. Two reasons:
#
# 1. The Debian package pulls in `nvidia-alternative` →
#    `glx-alternative-nvidia` → graphics-stack transitive
#    deps that aren't in bookworm-slim and would blow the
#    50 MB image budget many times over.
# 2. In production every Concerto image runs with
#    `--gpus all` (or the docker-compose equivalent), which
#    activates the NVIDIA Container Toolkit. The toolkit
#    bind-mounts the *host* driver's libnvidia-ml.so.1 into
#    /usr/lib/x86_64-linux-gnu/ inside the container. That
#    file is what `nvml-wrapper` dlopens, and using the host
#    lib guarantees the version matches the host kernel
#    driver — which a stub package from Debian non-free
#    cannot.
#
# Without `--gpus all`, the binary still runs perfectly
# under `--mock-gpus N` (dev / CI smoke tests). NVML
# initialisation only fires when the operator opts in via
# the absence of `--mock-gpus`, at which point the host
# driver libs MUST be present anyway.
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libssl3 \
        ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Non-root user with a stable UID. 65532 mirrors the
# distroless `nonroot` convention so volume-mount perms are
# predictable.
RUN set -eux; \
    groupadd --system --gid 65532 concerto; \
    useradd --system --uid 65532 --gid 65532 \
        --home-dir /var/lib/concerto \
        --shell /usr/sbin/nologin \
        concerto; \
    install -d -o concerto -g concerto -m 0755 /var/lib/concerto

# Stripped binary from the builder.
COPY --from=builder --chown=concerto:concerto \
    /usr/local/bin/concerto /usr/local/bin/concerto
RUN chmod +x /usr/local/bin/concerto

# State directory — A.3 startup-reconcile lands here. Mount a
# named volume on top in production (see docs/deployment.md).
WORKDIR /var/lib/concerto

# OpenAI-compatible API + /metrics + /status. Concerto's
# default bind port is 8000 (concerto-config server.rs).
EXPOSE 8000

USER concerto

ENTRYPOINT ["/usr/local/bin/concerto"]
CMD ["--config", "/etc/concerto.toml"]
