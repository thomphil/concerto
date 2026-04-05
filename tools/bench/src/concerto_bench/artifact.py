"""Artifact builder: materialises the on-disk tree and packages a tarball.

This module owns the second half of the rig's output pipeline. The runner
(step 7, later) produces pydantic records as it executes a scenario;
``artifact.py`` is responsible for serialising those records into the
exact on-disk shape specified in ``SPRINT-2-PLAN.md`` §4 B.1 / B.3 / B.5
and packaging the result as a ``.tar.gz`` bundle for harvest from a
Vast.ai box.

The on-disk tree looks like::

    <root_dir>/
      manifest.json
      summary.json
      environment.json
      concerto-stdout.log
      concerto-stderr.log
      steps/
        01-<step-slug>/
          pre-state.json
          post-state.json
          result.json
          request-<capture_as>.json    # one per captured RequestRecord
        02-<step-slug>/
          ...
      telemetry/
        nvidia-smi.jsonl
        concerto-status.jsonl
        concerto-metrics.jsonl
        pgrep-count.jsonl
        proc-stats.jsonl

and is packaged at ``<root_dir>.tar.gz`` with a sibling
``<root_dir>.tar.gz.sha256`` checksum file in ``sha256sum(1)`` format.

Design goals
------------

1. **Strict typing at the boundary.** The builder only accepts pydantic
   model instances (``ManifestV1``, ``SummaryV1``, ``HostInfo``,
   ``StepResult``, ``StateSnapshot``, ``RequestRecord``). Raw dicts are
   rejected with ``ArtifactError``. This is what makes "schema v1" a
   load-bearing contract rather than a polite suggestion.
2. **Reproducibility.** Tarball member ordering is sorted-by-path and
   every member's mtime is set to ``manifest.started_at``, so two
   invocations with identical inputs produce byte-identical bundles.
   This is a prerequisite for the regression-diff tooling landing in
   step 11.
3. **Samplers own their JSONL files.** The builder exposes a
   ``telemetry_dir()`` so samplers can open their own append-streams
   and write rows independently. The builder never opens, locks, or
   buffers those files — it only tars them at ``finalize()`` time.
4. **Single finalisation.** ``finalize()`` is not idempotent: calling
   it twice raises ``ArtifactError`` rather than silently clobbering.
5. **Verifiability.** ``verify_artifact_tree()`` walks a materialised
   tree and returns a list of structural / schema-level problems. This
   is the primitive the CI dry-run test (step 13) and the analyze /
   diff tools (step 11) build on.

The module is stdlib-only apart from the pydantic schema types it
serialises. ``tarfile``, ``hashlib``, ``shutil``, ``pathlib``, and
``re`` are enough.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import re
import shutil
import tarfile
import unicodedata
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, ValidationError

from concerto_bench.schema import (
    ARTIFACT_TREE_VERSION,
    SCHEMA_VERSION,
    ActionRecord,
    HostInfo,
    ManifestV1,
    RequestRecord,
    StateSnapshot,
    StepResult,
    SummaryV1,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArtifactError(RuntimeError):
    """Raised on any IO, type, or structural failure while building the artifact.

    The message always includes the offending filesystem path (or the
    name of the misbehaving input) so a failed run surfaces a concrete
    breadcrumb rather than a generic traceback.
    """


# ---------------------------------------------------------------------------
# Finalised-artifact descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalizedArtifact:
    """Immutable description of a sealed, tarred artifact bundle.

    Returned by :meth:`ArtifactBuilder.finalize`. Callers (the runner,
    the CLI, downstream harvesters) read these fields to report the
    result of a run and to feed ``sha256sum -c`` / ``tar -xzf`` pipelines.
    """

    tarball_path: Path
    sha256_path: Path
    sha256_hex: str
    size_bytes: int
    file_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SLUG_STRIP_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_SLUG_COLLAPSE_RE = re.compile(r"-{2,}")


def _slugify(value: str) -> str:
    """Reduce an arbitrary string to a filesystem-safe kebab-case slug.

    The rig allows scenario authors to pick human-readable step names
    like ``"Single-model smoke!"`` or ``"01 Crash Recovery / Smoke"``.
    Those cannot be used as directory names as-is: slashes become path
    separators, whitespace is a nuisance on the command line, and shell
    metacharacters trip up ``tar`` consumers.

    The slugifier is deliberately conservative — ASCII letters, digits,
    ``.``, ``_``, ``-`` — and collapses any run of other characters to
    a single ``-``. If the result is empty (e.g. the input was purely
    symbolic), a placeholder ``step`` is returned so the caller always
    gets a valid directory component.
    """
    normalised = unicodedata.normalize("NFKD", value)
    # Drop combining marks introduced by NFKD on accented characters.
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    cleaned = _SLUG_STRIP_RE.sub("-", ascii_only).strip("-_.")
    cleaned = _SLUG_COLLAPSE_RE.sub("-", cleaned)
    if not cleaned:
        return "step"
    return cleaned.lower()


def _step_dir_name(step: StepResult) -> str:
    """Return ``NN-<slug>`` for a given :class:`StepResult`.

    The ``NN`` component is zero-padded to two digits because the
    canonical Sprint 2 scenario has up to 8 steps today; three-digit
    scenarios are possible in future but the artifact-tree invariant
    is "lexical sort equals execution order", which ``%02d`` preserves
    for any reasonable scenario length.
    """
    return f"{step.step_number:02d}-{_slugify(step.step_name)}"


def _require_model(obj: object, expected: type[BaseModel], arg_name: str) -> None:
    """Raise :class:`ArtifactError` if ``obj`` is not an instance of ``expected``.

    Enforces the builder's strict contract: callers must hand in fully
    validated pydantic model instances, not raw dicts. The type check
    is deliberately cheap and runs before any IO so a caller mistake
    fails fast rather than producing a half-written tree.
    """
    if not isinstance(obj, expected):
        raise ArtifactError(
            f"{arg_name}: expected {expected.__name__} instance, "
            f"got {type(obj).__name__}"
        )


def _write_model_json(model: BaseModel, target: Path) -> None:
    """Serialise ``model`` to ``target`` as pretty-printed UTF-8 JSON.

    Uses ``model.model_dump_json(indent=2)`` directly so pydantic's
    field order (which mirrors the class definition) is preserved and
    datetimes pass through the schema's ISO-8601 encoder. IO errors
    are repackaged as :class:`ArtifactError` with the offending path.
    """
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = model.model_dump_json(indent=2)
        target.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"failed to write {target}: {exc}") from exc


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


class ArtifactBuilder:
    """Materialises a Concerto bench rig artifact on disk and tars it.

    Lifecycle::

        builder = ArtifactBuilder(root_dir=Path("/tmp/run-123"))
        builder.write_manifest(manifest)
        builder.write_summary(summary)
        builder.write_host_info(host_info)
        builder.write_concerto_logs(stdout_bytes, stderr_bytes)
        for step in step_results:
            builder.write_step(
                step,
                pre_state=pre_snapshot,
                post_state=post_snapshot,
                request_records={"greet": record},
            )
        builder.register_telemetry_file("nvidia-smi", jsonl_path)
        finalized = builder.finalize()

    The builder is **single-use**. Once :meth:`finalize` returns, the
    instance raises on every subsequent call. Construct a new builder
    for a new run.
    """

    # Canonical artifact-tree filenames. Kept as class constants so
    # tests and the verifier can reference the exact strings instead
    # of hard-coding them.
    MANIFEST_NAME = "manifest.json"
    SUMMARY_NAME = "summary.json"
    ENVIRONMENT_NAME = "environment.json"
    STDOUT_LOG_NAME = "concerto-stdout.log"
    STDERR_LOG_NAME = "concerto-stderr.log"
    STEPS_DIR_NAME = "steps"
    TELEMETRY_DIR_NAME = "telemetry"

    def __init__(self, root_dir: Path) -> None:
        """Create the artifact directory layout.

        ``root_dir`` is created if missing. If it exists already it
        must be empty (or only contain subdirectories the builder is
        about to write into); this protects against accidentally tarring
        leftover files from a previous run.
        """
        self._root = Path(root_dir)
        self._finalized = False
        self._manifest: Optional[ManifestV1] = None
        self._summary_written = False
        self._host_info_written = False
        self._step_numbers_seen: set[int] = set()
        # Externally-registered telemetry files, keyed by sampler name.
        self._telemetry_registrations: dict[str, Path] = {}

        try:
            self._root.mkdir(parents=True, exist_ok=True)
            (self._root / self.STEPS_DIR_NAME).mkdir(exist_ok=True)
            (self._root / self.TELEMETRY_DIR_NAME).mkdir(exist_ok=True)
        except OSError as exc:
            raise ArtifactError(
                f"failed to initialise artifact root {self._root}: {exc}"
            ) from exc

        logger.debug("artifact builder initialised at %s", self._root)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def root_dir(self) -> Path:
        """Absolute path to the artifact root directory."""
        return self._root

    def telemetry_dir(self) -> Path:
        """Return the ``telemetry/`` subdirectory path for samplers to use.

        Samplers built in step 6 open their own JSONL files inside this
        directory. The builder never touches those files until
        :meth:`finalize` packages them; it does not open, lock, or
        buffer the streams.
        """
        return self._root / self.TELEMETRY_DIR_NAME

    def step_dir(self, step: StepResult) -> Path:
        """Compute (and materialise) the ``steps/NN-<slug>/`` directory for a step.

        Safe to call before :meth:`write_step`: the directory is created
        if missing but no files are written. Useful for primitives that
        want to stream artifacts into a step directory (e.g. a future
        ``wrk_load`` primitive dumping a raw log) before the step's
        ``result.json`` is committed.
        """
        self._require_not_finalized()
        _require_model(step, StepResult, "step")
        path = self._root / self.STEPS_DIR_NAME / _step_dir_name(step)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactError(f"failed to create step dir {path}: {exc}") from exc
        return path

    # ------------------------------------------------------------------
    # Top-level record writers
    # ------------------------------------------------------------------

    def write_manifest(self, manifest: ManifestV1) -> Path:
        """Write ``manifest.json`` at the artifact root.

        Must be called before :meth:`finalize` because the manifest's
        ``started_at`` is used as the deterministic mtime for every
        tarball member. The builder caches the model instance so the
        reproducible-tarball invariant can be honoured without re-reading
        the file.
        """
        self._require_not_finalized()
        _require_model(manifest, ManifestV1, "manifest")
        target = self._root / self.MANIFEST_NAME
        _write_model_json(manifest, target)
        self._manifest = manifest
        return target

    def write_summary(self, summary: SummaryV1) -> Path:
        """Write ``summary.json`` at the artifact root."""
        self._require_not_finalized()
        _require_model(summary, SummaryV1, "summary")
        target = self._root / self.SUMMARY_NAME
        _write_model_json(summary, target)
        self._summary_written = True
        return target

    def write_host_info(self, host_info: HostInfo) -> Path:
        """Write ``environment.json`` (the :class:`HostInfo` blob)."""
        self._require_not_finalized()
        _require_model(host_info, HostInfo, "host_info")
        target = self._root / self.ENVIRONMENT_NAME
        _write_model_json(host_info, target)
        self._host_info_written = True
        return target

    # ------------------------------------------------------------------
    # Concerto log ingestion (two flavours: bytes in, or files on disk)
    # ------------------------------------------------------------------

    def write_concerto_logs(
        self,
        stdout_bytes: Optional[bytes] = None,
        stderr_bytes: Optional[bytes] = None,
    ) -> tuple[Path, Path]:
        """Write raw stdout/stderr bytes into the artifact root.

        Intended for tests and for callers that already hold the log
        contents in memory. Bytes are written verbatim — no decoding,
        no normalisation — because log files should be auditable exactly
        as concerto emitted them. Either argument may be ``None``, in
        which case a zero-byte file is still created so the artifact
        tree always has the two canonical log paths.
        """
        self._require_not_finalized()
        stdout_target = self._root / self.STDOUT_LOG_NAME
        stderr_target = self._root / self.STDERR_LOG_NAME
        try:
            stdout_target.write_bytes(stdout_bytes or b"")
            stderr_target.write_bytes(stderr_bytes or b"")
        except OSError as exc:
            raise ArtifactError(
                f"failed to write concerto logs under {self._root}: {exc}"
            ) from exc
        return stdout_target, stderr_target

    def copy_concerto_logs(
        self,
        stdout_path: Optional[Path] = None,
        stderr_path: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """Copy existing stdout/stderr log files into the artifact root.

        The ``concerto_proc`` module (step 2) tees the child's output
        into log files under its own ``log_dir``. This method copies
        those files into the canonical artifact layout. A missing
        source path is treated as an empty log — a zero-byte target is
        created so the tree shape is preserved either way.
        """
        self._require_not_finalized()
        stdout_target = self._root / self.STDOUT_LOG_NAME
        stderr_target = self._root / self.STDERR_LOG_NAME
        for src, dst in ((stdout_path, stdout_target), (stderr_path, stderr_target)):
            try:
                if src is not None and src.exists():
                    shutil.copyfile(src, dst)
                else:
                    dst.write_bytes(b"")
            except OSError as exc:
                raise ArtifactError(
                    f"failed to copy concerto log {src} -> {dst}: {exc}"
                ) from exc
        return stdout_target, stderr_target

    # ------------------------------------------------------------------
    # Per-step writes
    # ------------------------------------------------------------------

    def write_step(
        self,
        step: StepResult,
        pre_state: StateSnapshot,
        post_state: StateSnapshot,
        request_records: Optional[dict[str, RequestRecord]] = None,
    ) -> Path:
        """Write everything belonging to a single scenario step.

        Creates ``pre-state.json``, ``post-state.json``, ``result.json``,
        and a ``request-<capture_as>.json`` for every entry in
        ``request_records``. The step's directory is returned so callers
        can write additional primitive-specific artifacts (e.g. raw wrk
        logs) alongside the canonical files.

        Duplicate ``step_number`` across two calls raises
        :class:`ArtifactError`: the v1 tree shape assumes a 1-to-1
        mapping between step numbers and directories, and silently
        overwriting would break analyzer assumptions.
        """
        self._require_not_finalized()
        _require_model(step, StepResult, "step")
        _require_model(pre_state, StateSnapshot, "pre_state")
        _require_model(post_state, StateSnapshot, "post_state")
        if step.step_number in self._step_numbers_seen:
            raise ArtifactError(
                f"duplicate step_number={step.step_number} written to artifact tree"
            )
        self._step_numbers_seen.add(step.step_number)

        step_path = self.step_dir(step)
        _write_model_json(pre_state, step_path / "pre-state.json")
        _write_model_json(post_state, step_path / "post-state.json")
        _write_model_json(step, step_path / "result.json")

        if request_records:
            for capture_as, record in request_records.items():
                _require_model(
                    record, RequestRecord, f"request_records[{capture_as!r}]"
                )
                if not isinstance(capture_as, str) or not capture_as:
                    raise ArtifactError(
                        f"request_records keys must be non-empty strings, got {capture_as!r}"
                    )
                slug = _slugify(capture_as)
                _write_model_json(record, step_path / f"request-{slug}.json")

        return step_path

    # ------------------------------------------------------------------
    # Telemetry file registration
    # ------------------------------------------------------------------

    def register_telemetry_file(self, name: str, path: Path) -> Path:
        """Register a pre-existing JSONL file as ``telemetry/<name>.jsonl``.

        Two use cases:

        1. Tests that fabricate sampler data outside the runner and
           want it tarred under the standard telemetry layout.
        2. Samplers (step 6) that stream directly into
           ``telemetry_dir()`` and only need the builder to know about
           the file so it ends up in the tarball and the verifier.

        If ``path`` is already under ``telemetry_dir()`` it is recorded
        as-is. Otherwise the file is copied to
        ``<root>/telemetry/<name>.jsonl``. The original is not deleted.
        """
        self._require_not_finalized()
        if not name or not isinstance(name, str):
            raise ArtifactError(f"telemetry name must be a non-empty string, got {name!r}")
        slug = _slugify(name)
        target = self.telemetry_dir() / f"{slug}.jsonl"
        try:
            source = Path(path)
            if source.resolve() != target.resolve():
                if not source.exists():
                    raise ArtifactError(f"telemetry source file does not exist: {source}")
                shutil.copyfile(source, target)
        except OSError as exc:
            raise ArtifactError(
                f"failed to register telemetry file {path} as {target}: {exc}"
            ) from exc
        self._telemetry_registrations[slug] = target
        return target

    # ------------------------------------------------------------------
    # Finalisation: seal + tar + sha256
    # ------------------------------------------------------------------

    def finalize(self, *, include_empty_telemetry: bool = True) -> FinalizedArtifact:
        """Seal the on-disk tree and produce ``<root_dir>.tar.gz``.

        The tarball and its sibling ``.sha256`` file are placed **next
        to** ``root_dir``, never inside it, so the on-disk tree can be
        wiped independently of the shipped bundle.

        Parameters
        ----------
        include_empty_telemetry:
            If ``False``, zero-byte JSONL files in ``telemetry/`` are
            excluded from the tarball. Useful in tests that want to
            assert on a minimal bundle; the default ``True`` preserves
            the "empty samplers are still visible" debugging posture.

        Raises
        ------
        ArtifactError
            If the builder has already been finalised, if the manifest
            was never written (it drives reproducible mtimes), or if
            any IO step fails.
        """
        if self._finalized:
            raise ArtifactError(
                f"ArtifactBuilder at {self._root} has already been finalised"
            )
        if self._manifest is None:
            raise ArtifactError(
                "cannot finalise artifact: write_manifest() has not been called"
            )

        # Ensure the log files always exist, even if the caller never
        # called one of the write_*_logs helpers. Zero-byte placeholders
        # keep the tree shape stable for the verifier.
        for name in (self.STDOUT_LOG_NAME, self.STDERR_LOG_NAME):
            log_path = self._root / name
            if not log_path.exists():
                try:
                    log_path.write_bytes(b"")
                except OSError as exc:
                    raise ArtifactError(
                        f"failed to create placeholder log {log_path}: {exc}"
                    ) from exc

        tarball_path = self._root.with_name(self._root.name + ".tar.gz")
        sha256_path = tarball_path.with_name(tarball_path.name + ".sha256")

        # Reproducibility: use the manifest's started_at as the mtime
        # on every tar member. POSIX tar stores epoch seconds, so we
        # drop microseconds explicitly.
        mtime = int(self._manifest.started_at.astimezone(timezone.utc).timestamp())

        members = self._collect_tar_members(
            include_empty_telemetry=include_empty_telemetry
        )
        file_count = len(members)

        try:
            # Reproducibility requires two independent knobs on the
            # gzip layer:
            #
            #   1. ``mtime=0`` in the gzip header — otherwise the header
            #      timestamp bakes wall-clock time into byte 4.
            #   2. ``filename=""`` — tarfile.open("foo.tar.gz") embeds
            #      the source filename in the gzip header by default,
            #      which makes the bundle path-dependent.
            #
            # Neither knob is reachable from ``tarfile.open(name, "w:gz")``,
            # so we build the GzipFile ourselves and pass it as fileobj.
            # ``compresslevel=6`` is the Python default and is pinned
            # explicitly so future version bumps can't silently change
            # the compressed bytes.
            raw_fh = open(tarball_path, "wb")
            gz_fh = gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_fh,
                mtime=0,
            )
            with tarfile.open(
                fileobj=gz_fh,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as tar:
                for arcname, source in members:
                    info = tar.gettarinfo(str(source), arcname=arcname)
                    if info is None:
                        raise ArtifactError(
                            f"tarfile.gettarinfo returned None for {source}"
                        )
                    # Force stable metadata so the tarball is
                    # bit-for-bit reproducible across machines.
                    info.mtime = mtime
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    if info.isfile():
                        with open(source, "rb") as fh:
                            tar.addfile(info, fh)
                    else:
                        tar.addfile(info)
            gz_fh.close()
            raw_fh.close()
        except (OSError, tarfile.TarError) as exc:
            # Best-effort cleanup of the half-written file descriptors.
            try:
                gz_fh.close()  # type: ignore[possibly-unbound]
            except Exception:
                pass
            try:
                raw_fh.close()  # type: ignore[possibly-unbound]
            except Exception:
                pass
            raise ArtifactError(
                f"failed to write tarball {tarball_path}: {exc}"
            ) from exc

        try:
            sha256_hex = _sha256_file(tarball_path)
            size_bytes = tarball_path.stat().st_size
            sha256_path.write_text(
                f"{sha256_hex}  {tarball_path.name}\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ArtifactError(
                f"failed to write sha256 sidecar {sha256_path}: {exc}"
            ) from exc

        self._finalized = True
        logger.info(
            "artifact finalised: tarball=%s sha256=%s size=%d bytes files=%d",
            tarball_path,
            sha256_hex,
            size_bytes,
            file_count,
        )
        return FinalizedArtifact(
            tarball_path=tarball_path,
            sha256_path=sha256_path,
            sha256_hex=sha256_hex,
            size_bytes=size_bytes,
            file_count=file_count,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_not_finalized(self) -> None:
        """Guard against writes after :meth:`finalize`."""
        if self._finalized:
            raise ArtifactError(
                f"ArtifactBuilder at {self._root} is finalised; no further writes allowed"
            )

    def _collect_tar_members(
        self, *, include_empty_telemetry: bool
    ) -> list[tuple[str, Path]]:
        """Walk ``root_dir`` and produce a sorted list of ``(arcname, source)``.

        ``arcname`` is relative to ``root_dir`` and always uses ``/``
        as its separator so Windows-hosted tarballs (if anyone ever
        runs the rig there) extract cleanly on Linux. Sorting by
        ``arcname`` is what gives the tarball its deterministic order.

        Zero-byte telemetry files are skipped when ``include_empty_telemetry``
        is ``False``; all other files are included regardless of size.
        Empty intermediate directories (notably ``telemetry/`` on a
        non-GPU host) are preserved in the tarball as explicit directory
        entries so the extracted tree still matches the canonical
        layout the verifier expects.
        """
        telemetry_prefix = f"{self.TELEMETRY_DIR_NAME}/"
        file_entries: list[tuple[str, Path]] = []
        dir_entries: list[tuple[str, Path]] = []
        seen_dirs: set[str] = set()

        for path in sorted(self._root.rglob("*")):
            rel = path.relative_to(self._root).as_posix()
            if path.is_dir():
                # Only emit directory entries for dirs that would
                # otherwise be empty; non-empty dirs are implied by
                # their contained file entries.
                if not any(path.iterdir()):
                    dir_entries.append((rel, path))
                    seen_dirs.add(rel)
                continue
            if not path.is_file():
                continue
            if (
                not include_empty_telemetry
                and rel.startswith(telemetry_prefix)
                and path.stat().st_size == 0
            ):
                continue
            file_entries.append((rel, path))

        collected = file_entries + dir_entries
        collected.sort(key=lambda item: item[0])
        return collected


# ---------------------------------------------------------------------------
# Streaming SHA-256
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the hex SHA-256 digest of ``path`` computed in streaming mode.

    Avoids ``path.read_bytes()`` so multi-gigabyte tarballs don't blow
    the interpreter's memory. A 1 MiB chunk size is a sensible default
    on both SSD and network-attached storage.
    """
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"failed to hash {path}: {exc}") from exc
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


_STEP_DIR_RE = re.compile(r"^(\d{2,})-[a-z0-9._-]+$")


def verify_artifact_tree(root_dir: Path) -> list[str]:
    """Walk a materialised artifact tree and return a list of problems.

    Intended as the canonical validation primitive for the CI dry-run
    end-to-end test (step 13) and for the ``analyze`` and ``diff``
    subcommands (step 11). Called on an on-disk tree (either pre- or
    post-finalisation) and returns a list of human-readable error
    strings. An empty list means the tree is structurally valid and
    every top-level JSON file parses under its schema.

    Checks performed
    ----------------

    1. ``manifest.json``, ``summary.json``, ``environment.json`` exist
       and validate against :class:`ManifestV1` / :class:`SummaryV1` /
       :class:`HostInfo`.
    2. Every entry in ``steps/`` is a directory named ``NN-<slug>``
       with ``pre-state.json``, ``post-state.json``, and ``result.json``
       — all of which parse against the relevant schema model.
    3. ``len(steps/*) == manifest.step_count``.
    4. Every ``ActionRecord`` of type ``request`` that names a
       ``capture_as`` has a matching ``request-<slug>.json`` file, and
       every ``request-*.json`` in the step dir has a backing action.
    5. Every record's ``schema_version`` equals
       :data:`concerto_bench.schema.SCHEMA_VERSION`.
    6. ``telemetry/`` exists. Zero-byte JSONL files are tolerated (they
       are common on non-GPU hosts) and are reported as info-level
       warnings in the logger but **not** returned as errors.
    """
    errors: list[str] = []
    root = Path(root_dir)

    if not root.is_dir():
        return [f"artifact root does not exist or is not a directory: {root}"]

    # --- Manifest -----------------------------------------------------
    manifest: Optional[ManifestV1] = None
    manifest_path = root / ArtifactBuilder.MANIFEST_NAME
    try:
        manifest = ManifestV1.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.schema_version != SCHEMA_VERSION:
            errors.append(
                f"{manifest_path}: schema_version={manifest.schema_version} "
                f"does not match expected {SCHEMA_VERSION}"
            )
        if manifest.artifact_tree_version != ARTIFACT_TREE_VERSION:
            errors.append(
                f"{manifest_path}: artifact_tree_version="
                f"{manifest.artifact_tree_version} does not match expected "
                f"{ARTIFACT_TREE_VERSION}"
            )
    except FileNotFoundError:
        errors.append(f"missing required file: {manifest_path}")
    except (ValidationError, ValueError) as exc:
        errors.append(f"{manifest_path}: failed to validate ManifestV1: {exc}")

    # --- Summary ------------------------------------------------------
    summary_path = root / ArtifactBuilder.SUMMARY_NAME
    try:
        summary = SummaryV1.model_validate_json(summary_path.read_text(encoding="utf-8"))
        if summary.schema_version != SCHEMA_VERSION:
            errors.append(
                f"{summary_path}: schema_version={summary.schema_version} "
                f"does not match expected {SCHEMA_VERSION}"
            )
    except FileNotFoundError:
        errors.append(f"missing required file: {summary_path}")
    except (ValidationError, ValueError) as exc:
        errors.append(f"{summary_path}: failed to validate SummaryV1: {exc}")

    # --- Environment --------------------------------------------------
    env_path = root / ArtifactBuilder.ENVIRONMENT_NAME
    try:
        host_info = HostInfo.model_validate_json(env_path.read_text(encoding="utf-8"))
        if host_info.schema_version != SCHEMA_VERSION:
            errors.append(
                f"{env_path}: schema_version={host_info.schema_version} "
                f"does not match expected {SCHEMA_VERSION}"
            )
    except FileNotFoundError:
        errors.append(f"missing required file: {env_path}")
    except (ValidationError, ValueError) as exc:
        errors.append(f"{env_path}: failed to validate HostInfo: {exc}")

    # --- Concerto logs ------------------------------------------------
    for name in (ArtifactBuilder.STDOUT_LOG_NAME, ArtifactBuilder.STDERR_LOG_NAME):
        log_path = root / name
        if not log_path.exists():
            errors.append(f"missing required file: {log_path}")

    # --- Steps --------------------------------------------------------
    steps_root = root / ArtifactBuilder.STEPS_DIR_NAME
    if not steps_root.is_dir():
        errors.append(f"missing required directory: {steps_root}")
    else:
        step_dirs = sorted(p for p in steps_root.iterdir() if p.is_dir())
        if manifest is not None and len(step_dirs) != manifest.step_count:
            errors.append(
                f"{steps_root}: found {len(step_dirs)} step dirs but "
                f"manifest.step_count={manifest.step_count}"
            )
        for step_dir in step_dirs:
            if not _STEP_DIR_RE.match(step_dir.name):
                errors.append(
                    f"{step_dir}: directory name does not match NN-<slug> pattern"
                )
            errors.extend(_verify_step_dir(step_dir))

    # --- Telemetry ----------------------------------------------------
    telemetry_root = root / ArtifactBuilder.TELEMETRY_DIR_NAME
    if not telemetry_root.is_dir():
        errors.append(f"missing required directory: {telemetry_root}")
    else:
        for jsonl in sorted(telemetry_root.glob("*.jsonl")):
            if jsonl.stat().st_size == 0:
                logger.info(
                    "telemetry file %s is empty; tolerated on non-GPU hosts", jsonl
                )

    return errors


def _verify_step_dir(step_dir: Path) -> list[str]:
    """Validate the three canonical files in a ``steps/NN-*`` directory.

    Parses ``result.json`` into a :class:`StepResult` and cross-checks
    that every ``request-<slug>.json`` file has a matching action with a
    ``capture_as`` entry (and vice versa). Returns one error string per
    problem found. Internal helper for :func:`verify_artifact_tree`.
    """
    errors: list[str] = []

    pre_path = step_dir / "pre-state.json"
    post_path = step_dir / "post-state.json"
    result_path = step_dir / "result.json"

    for path in (pre_path, post_path, result_path):
        if not path.exists():
            errors.append(f"missing required file: {path}")

    for path, model in (
        (pre_path, StateSnapshot),
        (post_path, StateSnapshot),
    ):
        if not path.exists():
            continue
        try:
            instance = model.model_validate_json(path.read_text(encoding="utf-8"))
            if instance.schema_version != SCHEMA_VERSION:
                errors.append(
                    f"{path}: schema_version={instance.schema_version} "
                    f"does not match expected {SCHEMA_VERSION}"
                )
        except (ValidationError, ValueError) as exc:
            errors.append(f"{path}: failed to validate {model.__name__}: {exc}")

    step_result: Optional[StepResult] = None
    if result_path.exists():
        try:
            step_result = StepResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
            if step_result.schema_version != SCHEMA_VERSION:
                errors.append(
                    f"{result_path}: schema_version={step_result.schema_version} "
                    f"does not match expected {SCHEMA_VERSION}"
                )
        except (ValidationError, ValueError) as exc:
            errors.append(f"{result_path}: failed to validate StepResult: {exc}")

    # --- Request-capture cross-check ---------------------------------
    capture_files = {
        p.stem[len("request-") :]: p
        for p in step_dir.glob("request-*.json")
        if p.is_file()
    }

    captures_from_actions: set[str] = set()
    if step_result is not None:
        captures_from_actions = _capture_names_from_actions(step_result.actions)
        for capture in captures_from_actions:
            slug = _slugify(capture)
            if slug not in capture_files:
                errors.append(
                    f"{step_dir}: action referenced capture_as={capture!r} but "
                    f"no request-{slug}.json file exists"
                )

    action_slugs = {_slugify(name) for name in captures_from_actions}
    for file_slug, path in capture_files.items():
        if file_slug not in action_slugs:
            errors.append(
                f"{path}: capture file has no corresponding ActionRecord in result.json"
            )
        try:
            RequestRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError) as exc:
            errors.append(f"{path}: failed to validate RequestRecord: {exc}")

    return errors


def _capture_names_from_actions(actions: Iterable[ActionRecord]) -> set[str]:
    """Extract the set of ``capture_as`` names declared in an action list.

    Only ``request`` actions carry a ``capture_as`` arg today; other
    primitives may adopt the convention later. Non-string values and
    empty strings are ignored so a malformed action doesn't explode
    the verifier — its job is to *report* problems, not crash on them.
    """
    captures: set[str] = set()
    for action in actions:
        capture_as = action.args.get("capture_as") if action.args else None
        if isinstance(capture_as, str) and capture_as:
            captures.add(capture_as)
    return captures


__all__ = [
    "ArtifactBuilder",
    "ArtifactError",
    "FinalizedArtifact",
    "verify_artifact_tree",
]
