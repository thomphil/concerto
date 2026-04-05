"""Sampler: periodic ``GET /status`` against the running concerto API.

Captures concerto's own view of the cluster at 1 Hz into
``telemetry/concerto-status.jsonl``. Each row's ``values`` payload is
the parsed JSON body of the response (see
``crates/concerto-api/src/routes/status.rs`` for the Rust-side shape —
typically ``{"gpus": [...], "registry_size": N}``).

Transport failures and non-2xx responses are raised from
:meth:`sample_once` and therefore counted as tick failures by the base
loop, which continues running. That is the right behaviour during the
early lifetime of a scenario when concerto may still be starting up.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import httpx
from pydantic import Field

from concerto_bench.samplers.base import Sampler, SamplerConfig

logger = logging.getLogger(__name__)

# Short per-request timeout so a wedged ``/status`` endpoint cannot
# starve the 1 Hz tick. Consistent with the snapshot primitive's
# timeout budget.
_DEFAULT_TIMEOUT_SECS = 0.5

ClientFactory = Callable[[], httpx.AsyncClient]


class ConcertoStatusSamplerConfig(SamplerConfig):
    """Config for :class:`ConcertoStatusSampler`.

    Adds a ``base_url`` pointing at the running concerto API (e.g.
    ``http://127.0.0.1:8000``). Everything else is inherited from
    :class:`SamplerConfig`.
    """

    base_url: str = Field(
        ...,
        min_length=1,
        description="Base URL of the concerto API, e.g. ``http://127.0.0.1:8000``.",
    )
    timeout_secs: float = Field(
        default=_DEFAULT_TIMEOUT_SECS,
        gt=0.0,
        description="Per-request timeout in seconds.",
    )


class ConcertoStatusSampler(Sampler):
    """1 Hz sampler that records ``GET /status`` responses.

    Uses a long-lived :class:`httpx.AsyncClient` opened in
    :meth:`setup` so connection pooling works across ticks. Tests
    inject a mock transport via the optional ``client_factory``
    constructor argument, which sidesteps monkey-patching global
    state in :mod:`httpx`.
    """

    config: ConcertoStatusSamplerConfig

    def __init__(
        self,
        config: ConcertoStatusSamplerConfig,
        *,
        client_factory: Optional[ClientFactory] = None,
    ) -> None:
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._client_factory = client_factory

    async def setup(self) -> None:
        if self._client_factory is not None:
            self._client = self._client_factory()
        else:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_secs)

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def sample_once(self) -> dict[str, Any]:
        assert self._client is not None, "ConcertoStatusSampler.setup was not called"
        url = f"{self.config.base_url.rstrip('/')}/status"
        response = await self._client.get(url, timeout=self.config.timeout_secs)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"/status returned non-object JSON: {type(body).__name__}")
        return body
