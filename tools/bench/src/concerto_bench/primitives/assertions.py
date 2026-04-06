"""``assertions`` primitive: validate conditions against ``/status``.

While ``wait_for`` polls until a condition becomes true, the
``assertions`` primitive checks the current state exactly once and
returns a pass/fail verdict. It is the scenario author's tool for
encoding correctness expectations: "at this point in the scenario,
the following must hold."

Shape at a glance
-----------------

* :class:`AssertAction` — frozen pydantic model, ``extra="forbid"``.
  Specifies the assertion type, expected value, and optional field
  path for structural comparisons.
* :class:`AssertError` — raised only on irrecoverable failures
  (e.g. unusable client, malformed URL).
* :class:`AssertPrimitive` — stateless executor. Fetches ``/status``
  once, evaluates the assertion, and returns the verdict.

Assertion types
---------------

``status_code``
    Check that ``GET /status`` returns HTTP 200. No other fields needed.
``model_loaded`` / ``model_not_loaded``
    Check model presence in the backends list. Requires ``model`` field.
``backend_count``
    Check ``len(backends) == expected``. Requires ``expected`` field.
``field_equals`` / ``field_gte`` / ``field_lte``
    Navigate a dot-separated ``field_path`` into the ``/status`` JSON
    and compare the resolved value against ``expected``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AssertError(RuntimeError):
    """Raised when the ``assertions`` primitive encounters an irrecoverable failure.

    Reserved for failures that prevent the assertion from being
    evaluated at all — a malformed base URL, an unusable HTTP client,
    or a structurally invalid assertion type. A *failed* assertion
    (condition not met) is **not** an error; it returns
    ``{"passed": False, ...}``.
    """


# ---------------------------------------------------------------------------
# Action argument model
# ---------------------------------------------------------------------------


_VALID_ASSERT_TYPES = frozenset({
    "status_code",
    "model_loaded",
    "model_not_loaded",
    "backend_count",
    "field_equals",
    "field_gte",
    "field_lte",
})


class AssertAction(BaseModel):
    """Scenario YAML arguments for a single ``assert`` action.

    Frozen so the runner can stash and reuse it across retries.

    Fields
    ------

    ``assert_type``
        One of: ``status_code``, ``model_loaded``, ``model_not_loaded``,
        ``backend_count``, ``field_equals``, ``field_gte``, ``field_lte``.
    ``model``
        Model ID for ``model_loaded``/``model_not_loaded`` assertions.
    ``expected``
        Expected value for comparison assertions.
    ``field_path``
        Dot-separated path into the ``/status`` JSON for ``field_*``
        assertions (e.g. ``"backends.0.status"``).
    ``message``
        Custom failure message. When provided, replaces the
        auto-generated message in the result dict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assert_type: str = Field(
        ...,
        description="Assertion type to evaluate.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model ID for model_loaded/model_not_loaded assertions.",
    )
    expected: Optional[Any] = Field(
        default=None,
        description="Expected value for comparison assertions.",
    )
    field_path: Optional[str] = Field(
        default=None,
        description="Dot-separated path into the /status JSON for field_* assertions.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Custom failure message.",
    )

    @field_validator("assert_type")
    @classmethod
    def _validate_assert_type(cls, value: str) -> str:
        if value not in _VALID_ASSERT_TYPES:
            raise ValueError(
                f"assert_type must be one of {sorted(_VALID_ASSERT_TYPES)}, got {value!r}"
            )
        return value


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


class AssertPrimitive:
    """Stateless executor for :class:`AssertAction`.

    Fetches ``GET {base_url}/status`` once, evaluates the assertion
    against the response, and returns a pass/fail verdict.

    Failure policy
    --------------

    * A *failed* assertion (condition not met) returns
      ``{"passed": False, ...}`` — it does NOT raise.
    * A transport error fetching ``/status`` raises
      :class:`AssertError`.
    """

    async def execute(
        self,
        action: AssertAction,
        *,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """Fetch ``/status`` and evaluate the assertion.

        Parameters
        ----------
        action:
            Frozen arguments for this invocation.
        base_url:
            Concerto's HTTP base URL, e.g. ``http://127.0.0.1:8000``.
        client:
            Optional pre-built :class:`httpx.AsyncClient`. When ``None``
            the primitive creates a short-lived internal client.

        Returns
        -------
        dict
            ``{"passed": bool, "actual": Any, "expected": Any,
            "message": str}``

        Raises
        ------
        :class:`AssertError`
            If ``/status`` cannot be fetched at all.
        """
        owned_client = client is None
        active_client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0))

        url = f"{base_url.rstrip('/')}/status"
        try:
            try:
                response = await active_client.get(url)
            except Exception as exc:
                raise AssertError(
                    f"GET {url} failed: {type(exc).__name__}: {exc}"
                ) from exc
        finally:
            if owned_client:
                await active_client.aclose()

        status_code = response.status_code
        try:
            status_data = response.json()
            if not isinstance(status_data, dict):
                status_data = {}
        except ValueError:
            status_data = {}

        return self._evaluate(action, status_code, status_data)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        action: AssertAction,
        status_code: int,
        status_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate the assertion against fetched /status data."""
        assert_type = action.assert_type

        if assert_type == "status_code":
            return self._eval_status_code(action, status_code)
        if assert_type == "model_loaded":
            return self._eval_model_loaded(action, status_data)
        if assert_type == "model_not_loaded":
            return self._eval_model_not_loaded(action, status_data)
        if assert_type == "backend_count":
            return self._eval_backend_count(action, status_data)
        if assert_type in ("field_equals", "field_gte", "field_lte"):
            return self._eval_field_comparison(action, status_data)

        # Unreachable thanks to the validator.
        return self._fail(
            actual=None,
            expected=None,
            message=action.message or f"unknown assert_type: {assert_type!r}",
        )

    def _eval_status_code(
        self,
        action: AssertAction,
        status_code: int,
    ) -> dict[str, Any]:
        passed = status_code == 200
        return {
            "passed": passed,
            "actual": status_code,
            "expected": 200,
            "message": action.message or ("" if passed else f"expected status 200, got {status_code}"),
        }

    def _eval_model_loaded(
        self,
        action: AssertAction,
        status_data: dict[str, Any],
    ) -> dict[str, Any]:
        loaded = self._is_model_loaded(status_data, action.model)
        return {
            "passed": loaded,
            "actual": loaded,
            "expected": True,
            "message": action.message or ("" if loaded else f"model {action.model!r} not loaded"),
        }

    def _eval_model_not_loaded(
        self,
        action: AssertAction,
        status_data: dict[str, Any],
    ) -> dict[str, Any]:
        loaded = self._is_model_loaded(status_data, action.model)
        passed = not loaded
        return {
            "passed": passed,
            "actual": loaded,
            "expected": False,
            "message": action.message or ("" if passed else f"model {action.model!r} is still loaded"),
        }

    def _eval_backend_count(
        self,
        action: AssertAction,
        status_data: dict[str, Any],
    ) -> dict[str, Any]:
        backends = self._extract_backends(status_data)
        actual = len(backends)
        expected = action.expected
        passed = actual == expected
        return {
            "passed": passed,
            "actual": actual,
            "expected": expected,
            "message": action.message or (
                "" if passed else f"expected {expected} backends, got {actual}"
            ),
        }

    def _eval_field_comparison(
        self,
        action: AssertAction,
        status_data: dict[str, Any],
    ) -> dict[str, Any]:
        if action.field_path is None:
            return self._fail(
                actual=None,
                expected=action.expected,
                message=action.message or "field_path is required for field_* assertions",
            )

        actual = self._resolve_field_path(status_data, action.field_path)
        if actual is _MISSING:
            return self._fail(
                actual=None,
                expected=action.expected,
                message=action.message or f"field_path {action.field_path!r} not found in /status",
            )

        expected = action.expected
        if action.assert_type == "field_equals":
            passed = actual == expected
            default_msg = f"field {action.field_path!r}: expected {expected!r}, got {actual!r}"
        elif action.assert_type == "field_gte":
            try:
                passed = actual >= expected
            except TypeError:
                passed = False
            default_msg = f"field {action.field_path!r}: expected >= {expected!r}, got {actual!r}"
        elif action.assert_type == "field_lte":
            try:
                passed = actual <= expected
            except TypeError:
                passed = False
            default_msg = f"field {action.field_path!r}: expected <= {expected!r}, got {actual!r}"
        else:
            passed = False
            default_msg = f"unknown field comparison: {action.assert_type!r}"

        return {
            "passed": passed,
            "actual": actual,
            "expected": expected,
            "message": action.message or ("" if passed else default_msg),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fail(*, actual: Any, expected: Any, message: str) -> dict[str, Any]:
        return {"passed": False, "actual": actual, "expected": expected, "message": message}

    @staticmethod
    def _is_model_loaded(status: dict[str, Any], model: Optional[str]) -> bool:
        """Check if a model appears in the backends list with a loaded/running status."""
        if model is None:
            return False
        # Check top-level ``backends`` list first (future-proof).
        backends = status.get("backends")
        if isinstance(backends, list) and backends:
            for backend in backends:
                if not isinstance(backend, dict):
                    continue
                backend_model = backend.get("model", backend.get("model_id", ""))
                backend_status = backend.get("status", "").lower()
                if backend_model == model and backend_status in ("loaded", "running", "ready"):
                    return True
            return False
        # Fall back to gpus[].loaded_models[] — concerto's actual
        # /status shape nests models here as objects with ``model_id``.
        gpus = status.get("gpus", [])
        if isinstance(gpus, list):
            for gpu in gpus:
                if isinstance(gpu, dict):
                    loaded = gpu.get("loaded_models", [])
                    if isinstance(loaded, list):
                        for entry in loaded:
                            if isinstance(entry, dict) and entry.get("model_id") == model:
                                return True
                            if isinstance(entry, str) and entry == model:
                                return True
        return False

    @staticmethod
    def _extract_backends(status: dict[str, Any]) -> list[dict[str, Any]]:
        # Check for a top-level ``backends`` key first (future-proof).
        backends = status.get("backends", [])
        if isinstance(backends, list) and backends:
            return [b for b in backends if isinstance(b, dict)]
        # Fall back to extracting from gpus[].loaded_models[].
        result: list[dict[str, Any]] = []
        gpus = status.get("gpus", [])
        if isinstance(gpus, list):
            for gpu in gpus:
                if isinstance(gpu, dict):
                    loaded = gpu.get("loaded_models", [])
                    if isinstance(loaded, list):
                        for entry in loaded:
                            if isinstance(entry, dict):
                                result.append(entry)
        return result

    @staticmethod
    def _resolve_field_path(data: Any, path: str) -> Any:
        """Navigate a dot-separated path into a JSON structure.

        Supports integer path components as list indices (e.g.
        ``"backends.0.status"``). Returns :data:`_MISSING` if any
        segment cannot be resolved.
        """
        current = data
        for segment in path.split("."):
            if isinstance(current, dict):
                if segment in current:
                    current = current[segment]
                else:
                    return _MISSING
            elif isinstance(current, list):
                try:
                    idx = int(segment)
                except ValueError:
                    return _MISSING
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return _MISSING
            else:
                return _MISSING
        return current


# Sentinel for missing field paths.
_MISSING = object()
