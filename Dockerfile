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
# - libnvidia-ml1:   NVML stub, dlopen'd by nvml-wrapper when
#                    the `nvml` feature is enabled at runtime.
#                    Lives in Debian's `non-free` component, so
#                    we enable it just for this install and
#                    drop the source list afterwards. In
#                    production the NVIDIA Container Toolkit
#                    typically bind-mounts a matching libnvidia-ml
#                    from the host driver over this stub; the
#                    stub keeps the binary loadable without GPU
#                    in CI / smoke tests.
#
# Clean apt lists and the temporary non-free source in the
# same layer to keep the image small.
RUN set -eux; \
    echo "deb http://deb.debian.org/debian bookworm non-free" \
        > /etc/apt/sources.list.d/non-free.list; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libssl3 \
        ca-certificates \
        libnvidia-ml1; \
    rm -f /etc/apt/sources.list.d/non-free.list; \
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
