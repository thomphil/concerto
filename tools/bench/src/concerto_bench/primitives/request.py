"""``request`` primitive: POST ``/v1/chat/completions`` with full timing capture.

The bench rig's single most-used primitive. Every scenario that exercises
concerto's routing path issues at least one request via this primitive,
and the concurrent-load scenario issues thousands. Its job is to produce
a :class:`~concerto_bench.schema.RequestRecord` — a fully populated,
schema-versioned record — for every attempted request, *even on failure*.
That is the "partial success is better than black-box crash" principle
restated at the primitive layer.

Shape at a glance
-----------------

* :class:`RequestAction` — frozen pydantic model, ``extra="forbid"``.
  The as-written scenario YAML arguments. Validated once at scenario
  parse time; reused immutably for each invocation.
* :class:`RequestError` — raised only when the primitive cannot even
  construct a :class:`~concerto_bench.schema.RequestRecord` (malformed
  URL, catastrophic client state). Transport failures do *not* raise;
  they populate ``RequestRecord.error`` and the scenario-level
  assertions decide fatality.
* :class:`RequestPrimitive` — the stateless executor. A single instance
  can be reused across every request in a run; state lives in the
  optional injected ``httpx.AsyncClient`` (so HTTP/1.1 keepalive and
  connection pooling work).

Timing capture
--------------

``httpx`` does not expose a separate "connect" vs "TTFB" accounting out
of the box. The primitive works around this using the
``httpx.AsyncClient`` ``event_hooks`` interface:

* A ``request`` hook fires immediately before the request is dispatched.
  Wall-clock at this point becomes ``request_start``.
* A ``response`` hook fires as soon as the upstream has returned the
  status line and headers (before the body is consumed). Wall-clock at
  this point becomes ``response_start``.
* ``elapsed_connect_ms`` is approximated as "time between ``execute()``
  entry and the ``request`` hook firing". On a warm connection this is
  near-zero; on a cold connection it captures DNS + TCP + TLS. This is
  a deliberate approximation — getting a true split requires digging
  into httpcore internals and is not worth the maintenance cost for
  v0.1.
* ``elapsed_ttfb_ms`` is ``response_start - request_start``.
* ``elapsed_total_ms`` is wall-clock from ``execute()`` entry to the
  final :class:`RequestRecord` construction. It therefore includes body
  read time for non-streaming requests and full stream drain time for
  streaming requests.

Streaming vs non-streaming
--------------------------

* Non-streaming: ``response_body`` is populated from ``response.json()``;
  ``response_chunks`` is ``None``.
* Streaming: ``response_chunks`` is populated with the **raw** SSE lines
  (one per ``data: ...`` frame, ``[DONE]`` sentinel included if the
  upstream sent one); ``response_body`` is ``None``. The primitive does
  not attempt to reassemble the delta content into a semantic JSON
  object — downstream analysis can do that if it wants.

``expect_status`` contract
--------------------------

``expect_status`` is advisory. If the upstream returns a different
status, the returned :class:`~concerto_bench.schema.RequestRecord` still
has the actual status and body populated; the ``error`` field is set to
a descriptive string. The primitive does *not* raise. Rationale: the
scenario runner's assertion layer (a later Phase B.2 step) is where
fatality is decided, not here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from concerto_bench.schema import RequestRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RequestError(RuntimeError):
    """Raised when the ``request`` primitive cannot produce a RequestRecord.

    Reserved for the narrow set of failures where constructing a
    :class:`~concerto_bench.schema.RequestRecord` is impossible — a
    malformed ``base_url``, a client handed to us in a broken state,
    or an internal invariant violation. Transport-level failures
    (``httpx.TimeoutException``, ``httpx.ConnectError``, ``httpx.TransportError``)
    are **not** surfaced through this exception; those populate the
    record's ``error`` field and return normally so the artifact stays
    complete.

    Carries ``elapsed_ms`` so a raised error still reports how long the
    primitive spent before giving up, and ``response_status`` on the
    off chance the failure happened after headers were received.
    """

    def __init__(
        self,
        message: str,
        *,
        elapsed_ms: float = 0.0,
        response_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.elapsed_ms = elapsed_ms
        self.response_status = response_status

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        base = super().__str__()
        parts = [base, f"(elapsed_ms={self.elapsed_ms:.2f})"]
        if self.response_status is not None:
            parts.append(f"(response_status={self.response_status})")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Action argument model
# ---------------------------------------------------------------------------


class RequestAction(BaseModel):
    """Scenario YAML arguments for a single ``request`` action.

    Mirrors the :class:`~concerto_bench.schema.RequestRecord` envelope
    without duplicating its captured fields: the action holds the
    *inputs*, the record holds the *outputs*. Frozen so the runner can
    stash an action and reuse it across retries without worrying about
    mutation in flight.

    Fields
    ------

    ``model``
        Model ID as registered in ``concerto.toml`` — forwarded to
        concerto verbatim in the request body.
    ``content``
        User message text. Used as the single ``user`` role message
        content; must be non-empty.
    ``stream``
        If ``True``, ``stream: true`` is set on the outgoing body and the
        primitive consumes the SSE stream line-by-line.
    ``expect_status``
        Advisory expected HTTP status. See the module docstring's
        "``expect_status`` contract" section.
    ``timeout_secs``
        Per-request timeout in seconds (wall clock). Must be strictly
        positive.
    ``capture_as``
        Optional key under which the runner stores the resulting
        :class:`~concerto_bench.schema.RequestRecord` in the per-step
        context; also becomes the stem of the serialised file name
        (``steps/NN-.../request-<capture_as>.json``).
    ``max_tokens``
        Optional OpenAI ``max_tokens`` override; omitted from the body
        when ``None`` so upstream defaults apply.
    ``temperature``
        Optional OpenAI ``temperature`` override; omitted when ``None``.
    ``system``
        Optional system prompt; when set, the outgoing ``messages``
        array is prepended with a ``system`` role entry.
    ``extra_headers``
        Additional HTTP headers merged into the outgoing request. Keys
        are lowercased by ``httpx``; callers should not rely on case.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(
        ...,
        min_length=1,
        description="Model ID as registered in concerto.toml.",
    )
    content: str = Field(
        ...,
        description="User message content — must be non-empty.",
    )
    stream: bool = Field(
        default=False,
        description="If True, forward stream=true and consume SSE chunks.",
    )
    expect_status: int = Field(
        default=200,
        description="Advisory expected HTTP status; mismatches set error, do not raise.",
    )
    timeout_secs: float = Field(
        default=30.0,
        description="Per-request wall-clock timeout in seconds.",
    )
    capture_as: Optional[str] = Field(
        default=None,
        description="Key the runner uses to file the resulting RequestRecord.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="OpenAI max_tokens override; omitted from the body when None.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="OpenAI temperature override; omitted from the body when None.",
    )
    system: Optional[str] = Field(
        default=None,
        description="Optional system prompt prepended to the messages array.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers merged into the outgoing request.",
    )

    @field_validator("content")
    @classmethod
    def _validate_content_nonempty(cls, value: str) -> str:
        # Guard against whitespace-only content slipping through
        # ``min_length=1`` — an empty user message wedges some upstream
        # backends, so reject it loudly at scenario-parse time rather
        # than eating a 400 at run time.
        if not value or not value.strip():
            raise ValueError("content must be a non-empty, non-whitespace string")
        return value

    @field_validator("expect_status")
    @classmethod
    def _validate_expect_status_range(cls, value: int) -> int:
        if not 100 <= value <= 599:
            raise ValueError(
                f"expect_status must be in the HTTP 100..599 range, got {value}"
            )
        return value

    @field_validator("timeout_secs")
    @classmethod
    def _validate_timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"timeout_secs must be > 0, got {value}")
        return value


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


class RequestPrimitive:
    """Stateless executor for :class:`RequestAction`.

    Designed for reuse across an entire run. The primitive owns no
    mutable state of its own; connection pooling and keepalive come
    from an optionally-injected :class:`httpx.AsyncClient`. When no
    client is passed, :meth:`execute` constructs and tears down an
    internal one for the lifetime of the call.

    Contract
    --------

    * An injected client is **never** closed by the primitive. The
      caller owns its lifetime. This matches the two-call test where
      a runner wraps many actions in a single client context.
    * On transport failure (timeout, connect error, TLS failure, read
      failure) the primitive swallows the exception, logs it at
      ``WARNING``, and returns a :class:`~concerto_bench.schema.RequestRecord`
      with ``error`` set and the best-effort fields populated. Status
      is set to ``0`` when no status was ever received.
    * :class:`RequestError` is raised only when ``base_url`` is
      structurally invalid or when the primitive cannot even start
      constructing a request — e.g. the injected client has already
      been closed and every attempt to issue a request fails at the
      transport layer before any hook fires. In practice this is rare.
    """

    _CHAT_PATH = "/v1/chat/completions"

    async def execute(
        self,
        action: RequestAction,
        *,
        base_url: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> RequestRecord:
        """Execute one chat-completion request and return a populated record.

        Parameters
        ----------
        action:
            Frozen arguments for this invocation.
        base_url:
            Concerto's HTTP base URL, e.g. ``http://127.0.0.1:8000``.
            Must be a valid absolute URL; a malformed value raises
            :class:`RequestError` before any network IO happens.
        client:
            Optional pre-built :class:`httpx.AsyncClient`. When supplied
            the primitive uses it verbatim and does **not** close it on
            exit — the caller retains ownership. When ``None`` the
            primitive creates a short-lived internal client.

        Returns
        -------
        :class:`~concerto_bench.schema.RequestRecord`
            A fully populated record. On success every field is set; on
            transport failure ``status`` is ``0``, ``response_body`` /
            ``response_chunks`` are ``None``, and ``error`` is a
            human-readable diagnostic. On ``expect_status`` mismatch
            everything is populated as normal and ``error`` describes
            the mismatch.

        Raises
        ------
        :class:`RequestError`
            Only if the primitive cannot construct a
            :class:`~concerto_bench.schema.RequestRecord` at all — a
            malformed ``base_url`` being the canonical case. Transport
            failures do *not* raise; see the class docstring.
        """
        # Pre-flight URL validation. httpx.URL is lenient about relative
        # paths; we explicitly require an absolute http(s) URL so a
        # typo in scenario wiring fails fast with a clear error rather
        # than producing a mysterious 404 against localhost.
        try:
            parsed = httpx.URL(base_url)
        except httpx.InvalidURL as exc:
            raise RequestError(
                f"invalid base_url {base_url!r}: {exc}",
            ) from exc
        if not parsed.scheme or not parsed.host:
            raise RequestError(
                f"invalid base_url {base_url!r}: missing scheme or host",
            )
        if parsed.scheme not in ("http", "https"):
            raise RequestError(
                f"invalid base_url {base_url!r}: scheme must be http or https",
            )

        request_body = self._build_request_body(action)
        url = f"{str(parsed).rstrip('/')}{self._CHAT_PATH}"

        # Timing state. All three timestamps are captured as
        # ``time.perf_counter()`` values for monotonic reliability; the
        # absolute wall-clock is not interesting here, only the deltas.
        call_start = time.perf_counter()
        state: dict[str, float] = {}

        async def _request_hook(_request: httpx.Request) -> None:
            state["request_start"] = time.perf_counter()

        async def _response_hook(_response: httpx.Response) -> None:
            state["response_start"] = time.perf_counter()

        owned_client = client is None
        active_client: Optional[httpx.AsyncClient] = client
        try:
            if active_client is None:
                active_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(action.timeout_secs),
                )
            # Event hooks must be layered onto the client regardless of
            # ownership. We save the caller's existing hook lists and
            # restore them on exit so an injected client is not mutated
            # in a way the caller can observe after ``execute`` returns.
            saved_request_hooks = list(active_client.event_hooks.get("request", []))
            saved_response_hooks = list(active_client.event_hooks.get("response", []))
            active_client.event_hooks["request"] = saved_request_hooks + [_request_hook]
            active_client.event_hooks["response"] = saved_response_hooks + [_response_hook]

            try:
                record = await self._issue(
                    action=action,
                    url=url,
                    request_body=request_body,
                    client=active_client,
                    state=state,
                    call_start=call_start,
                )
            finally:
                # Restore the caller's hook lists. If we created the
                # client ourselves this is purely defensive.
                active_client.event_hooks["request"] = saved_request_hooks
                active_client.event_hooks["response"] = saved_response_hooks
        finally:
            if owned_client and active_client is not None:
                await active_client.aclose()

        return record

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_request_body(self, action: RequestAction) -> dict[str, Any]:
        """Assemble the JSON body sent to concerto.

        Kept as a pure function so unit tests can inspect the body
        construction without exercising the transport layer.
        """
        messages: list[dict[str, Any]] = []
        if action.system is not None:
            messages.append({"role": "system", "content": action.system})
        messages.append({"role": "user", "content": action.content})

        body: dict[str, Any] = {
            "model": action.model,
            "messages": messages,
            "stream": action.stream,
        }
        if action.max_tokens is not None:
            body["max_tokens"] = action.max_tokens
        if action.temperature is not None:
            body["temperature"] = action.temperature
        return body

    async def _issue(
        self,
        *,
        action: RequestAction,
        url: str,
        request_body: dict[str, Any],
        client: httpx.AsyncClient,
        state: dict[str, float],
        call_start: float,
    ) -> RequestRecord:
        """Inner request flow shared by streaming and non-streaming paths."""
        status: int = 0
        response_body: Optional[dict[str, Any]] = None
        response_chunks: Optional[list[str]] = None
        error: Optional[str] = None

        headers = dict(action.extra_headers)
        headers.setdefault("content-type", "application/json")
        # Bound the whole operation with a wall-clock timeout in
        # addition to httpx's per-IO timeout so an unresponsive upstream
        # cannot tie up the scenario indefinitely.
        try:
            if action.stream:
                status, response_chunks = await asyncio.wait_for(
                    self._issue_streaming(
                        url=url,
                        body=request_body,
                        headers=headers,
                        client=client,
                        timeout=action.timeout_secs,
                    ),
                    timeout=action.timeout_secs,
                )
            else:
                status, response_body = await asyncio.wait_for(
                    self._issue_nonstreaming(
                        url=url,
                        body=request_body,
                        headers=headers,
                        client=client,
                        timeout=action.timeout_secs,
                    ),
                    timeout=action.timeout_secs,
                )
        except asyncio.TimeoutError as exc:
            error = f"timeout: {exc}" if str(exc) else "timeout: asyncio.wait_for expired"
            logger.warning("request primitive timed out against %s: %s", url, error)
        except httpx.TimeoutException as exc:
            error = f"timeout: {type(exc).__name__}: {exc}"
            logger.warning("request primitive httpx timeout against %s: %s", url, error)
        except httpx.ConnectError as exc:
            error = f"connect_error: {exc}"
            logger.warning("request primitive connect error against %s: %s", url, error)
        except httpx.TransportError as exc:
            # Catches ReadError, RemoteProtocolError, ProxyError, ...
            error = f"transport_error: {type(exc).__name__}: {exc}"
            logger.warning("request primitive transport error against %s: %s", url, error)
        except httpx.InvalidURL as exc:
            # Should have been caught pre-flight; belt and braces.
            raise RequestError(
                f"invalid URL assembled for request: {exc}",
                elapsed_ms=(time.perf_counter() - call_start) * 1000.0,
            ) from exc

        # expect_status mismatch populates error without raising.
        if error is None and status != action.expect_status:
            error = (
                f"unexpected status {status} (expected {action.expect_status})"
            )

        elapsed_total_ms = (time.perf_counter() - call_start) * 1000.0

        request_start = state.get("request_start")
        response_start = state.get("response_start")
        elapsed_connect_ms: Optional[float] = None
        elapsed_ttfb_ms: Optional[float] = None
        if request_start is not None:
            elapsed_connect_ms = max((request_start - call_start) * 1000.0, 0.0)
        if request_start is not None and response_start is not None:
            elapsed_ttfb_ms = max((response_start - request_start) * 1000.0, 0.0)

        return RequestRecord(
            status=status,
            elapsed_total_ms=max(elapsed_total_ms, 0.0),
            elapsed_ttfb_ms=elapsed_ttfb_ms,
            elapsed_connect_ms=elapsed_connect_ms,
            request_body=request_body,
            response_body=response_body,
            response_chunks=response_chunks,
            error=error,
        )

    async def _issue_nonstreaming(
        self,
        *,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[int, Optional[dict[str, Any]]]:
        """Non-streaming request path. Returns ``(status, json_body_or_none)``."""
        response = await client.post(
            url,
            json=body,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )
        parsed: Optional[dict[str, Any]]
        try:
            data = response.json()
        except ValueError:
            parsed = None
        else:
            # Concerto always returns a JSON object for chat completions;
            # if something upstream hands back a list or a primitive we
            # still want the status captured, but response_body stays
            # None so the schema's ``Optional[dict]`` type is honoured.
            parsed = data if isinstance(data, dict) else None
        return response.status_code, parsed

    async def _issue_streaming(
        self,
        *,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        client: httpx.AsyncClient,
        timeout: float,
    ) -> tuple[int, list[str]]:
        """Streaming request path. Returns ``(status, raw_chunk_lines)``.

        The chunk list preserves every non-empty line the upstream
        produced, in order, **unparsed**. Each element is one SSE event
        line — typically ``data: {"choices": [...]}`` or
        ``data: [DONE]``. Empty separator lines between SSE events are
        skipped; everything else is retained verbatim.
        """
        chunks: list[str] = []
        async with client.stream(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        ) as response:
            async for line in response.aiter_lines():
                if line == "":
                    continue
                chunks.append(line)
            return response.status_code, chunks
