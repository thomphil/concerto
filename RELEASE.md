# Release runbook

This document is the single source of truth for shipping a Concerto release.
The flow is: run the pre-release checklist, push an annotated tag (which fires
the automated binary-build and `ghcr.io` image-push workflows), then perform
the manual `crates.io` publish in dependency order, and finally flip the
GitHub release from draft to published. The `crates.io` step is intentionally
manual for v0.1.0; Actions automation is queued for v0.1.x cleanup.

The same runbook applies to v0.1.x patch releases — substitute the version
number where it appears.

---

## 1. Pre-release checklist

Verify all of the following before tagging. Any unchecked item is a release
blocker.

- [ ] CI is green on `main` at the commit you intend to tag (`gh run list --branch main --limit 5`).
- [ ] `cargo test --all` is green locally.
- [ ] `cargo fmt --check` is clean.
- [ ] `cargo clippy --all-targets -- -D warnings` is clean.
- [ ] Bench rig hand-validation on Vast.ai 2× RTX A4000 — Sprint 2 Run 12 numbers within noise (per `SPRINT-3-PLAN.md` §10).
- [ ] `Cargo.toml` `[workspace.package].version` matches the tag you are about to push (e.g. `0.1.0` for `v0.1.0`).
- [ ] `docs/benchmarks.md` reflects the run you just validated.
- [ ] `ROADMAP.md` §1 status block bumped to "v0.1.0 released".
- [ ] No open Sprint-3 PRs against the tag commit.
- [ ] `LICENSE-MIT` and `LICENSE-APACHE` are present at the repo root (the release workflow copies both into each archive).

---

## 2. Tag and push

Tags must be annotated (`-a`) so they carry an author and message; the
release workflow keys off `refs/tags/v*`.

```bash
git tag -a v0.1.0 -m "v0.1.0: first shippable release"
git push origin v0.1.0
```

This triggers `.github/workflows/release.yml`, which:

- Builds `x86_64-unknown-linux-gnu` (with `--features nvml`) and
  `aarch64-apple-darwin` (no NVML — `nvml-wrapper` only links on Linux)
  binaries for `concerto-cli`.
- Packages each as `concerto-cli-<tag>-<triple>.tar.gz` alongside a
  `.sha256` checksum, with `LICENSE-MIT`, `LICENSE-APACHE`, and `README.md`
  bundled in.
- Drafts a GitHub release with auto-generated notes and both archives
  attached.

It also triggers the `ghcr.io` publish workflow from §C.3 of the sprint
plan, which pushes the image to `ghcr.io/thomphil/concerto:v0.1.0` and
updates the `:latest` tag.

Watch both workflows finish:

```bash
gh run list --workflow=release.yml --limit 1
gh run watch
```

When both are green, the GitHub release exists in **draft** state. Do not
publish it yet — the crates.io step still has to succeed first.

---

## 3. crates.io publish (manual)

Six crates publish, in this dependency order. Run `cargo publish --dry-run`
first per crate; if it succeeds, run `cargo publish`.

The dependency graph (verified against each crate's `Cargo.toml`):

- `concerto-core` — no workspace deps.
- `concerto-gpu`, `concerto-backend`, `concerto-config` — each depends only
  on `concerto-core`. Safe to publish in any order once `concerto-core` is
  on the index; effectively parallelisable, but the snippet below runs them
  serially for simplicity.
- `concerto-api` — depends on `concerto-core`, `concerto-config`,
  `concerto-gpu`, `concerto-backend`.
- `concerto-cli` — depends on all five above. Publishes the `concerto`
  binary.

`tests/mock_backend` and `tests/scenarios` are workspace members but are
**not** published (they are dev/test-only).

Make sure you are logged in: `cargo login <crates.io-api-token>`.

```bash
cd crates/concerto-core      && cargo publish --dry-run && cargo publish
cd ../concerto-gpu           && cargo publish --dry-run && cargo publish
cd ../concerto-backend       && cargo publish --dry-run && cargo publish
cd ../concerto-config        && cargo publish --dry-run && cargo publish
# crates.io takes a few seconds to reindex; if api/cli publish errors with
# "no matching package", wait 30s and retry. This is normal.
cd ../concerto-api           && cargo publish --dry-run && cargo publish
cd ../concerto-cli           && cargo publish --dry-run && cargo publish
```

A failed `--dry-run` is recoverable — fix and retry. A failed `cargo publish`
mid-sequence is **not** recoverable under the same version (see §6).

---

## 4. Publish the GitHub release

Once `crates.io` is happy, navigate to the release page that the tag push
created:

```bash
gh release view v0.1.0 --web
```

Edit the release, confirm:

- Both `concerto-cli-v0.1.0-x86_64-unknown-linux-gnu.tar.gz` and
  `concerto-cli-v0.1.0-aarch64-apple-darwin.tar.gz` are attached, each with
  a sibling `.sha256` file.
- Auto-generated notes mention the binaries and the docker image
  (`ghcr.io/thomphil/concerto:v0.1.0`).

Then click **Publish release** (it was draft).

### Smoke checks

On a fresh Linux box (the bench rig is fine):

```bash
cargo install concerto-cli --features nvml
concerto --version
```

On macOS (no NVML — the `nvml` feature is a no-op off Linux but the build
is cleaner without it):

```bash
cargo install concerto-cli --no-default-features
concerto --version
```

And the container image:

```bash
docker pull ghcr.io/thomphil/concerto:v0.1.0
docker run --rm ghcr.io/thomphil/concerto:v0.1.0 --help
```

---

## 5. Post-release

- Bump `Cargo.toml` `[workspace.package].version` to the next dev version
  (e.g. `0.2.0-dev`) and open a PR.
- Update `ROADMAP.md` §1 status block to reflect the post-release state.
- Open issues for any "v0.1.x" follow-ups noted during the release (e.g.
  the `crates.io` Actions automation deferred from §C.4).
- Optional: announce on the channels listed in `ROADMAP.md` Sprint 4.

---

## 6. If something goes wrong

### A binary build fails on tag push

Both the workflow and the GitHub release are tied to the tag. Delete the
local and remote tag, fix the underlying issue on `main`, and re-tag:

```bash
git tag -d v0.1.0
git push --delete origin v0.1.0
# fix, merge, then re-tag the new HEAD
git tag -a v0.1.0 -m "v0.1.0: first shippable release"
git push origin v0.1.0
```

If the draft GitHub release was already created, delete it first:

```bash
gh release delete v0.1.0 --yes
```

### A `cargo publish` fails partway through

`crates.io` publishes are immutable — a published version cannot be
re-issued. If `concerto-api` publishes successfully but `concerto-cli`
fails, you cannot fix `concerto-api` and re-publish under `0.1.0`.

The remediation is to bump the patch version (`0.1.1`), update
`Cargo.toml`, run the full pre-release checklist again, tag `v0.1.1`, and
republish all six crates. The leftover `0.1.0` versions on `crates.io`
become permanent dead tags. Yank them:

```bash
cargo yank --version 0.1.0 concerto-core
# … repeat for each crate that did publish …
```

This is why §3 emphasises `cargo publish --dry-run` before the real
publish: dry-runs catch dependency-graph errors locally.

### Docker push fails

The `:latest` tag will lag the actual release. Re-run the `publish-image`
job from the GitHub Actions UI — it is idempotent and will overwrite
`:latest` and `:v0.1.0` correctly. If `ghcr.io` rejects the push entirely,
verify the workflow's `permissions: packages: write` is set (per §C.3)
and that the repo's package settings allow workflow pushes.

### `cargo install concerto-cli` fails on a fresh box

If the failure is a missing `libnvidia-ml.so`, the operator is on a host
without the NVIDIA driver and should install with
`--no-default-features` (see §4 smoke checks). If the failure is a
compile error, the most likely cause is a transitive dependency drift
between dry-run and publish; bump the patch version and republish.