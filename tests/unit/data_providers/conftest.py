"""Shared httpx mocks for the data_providers test suite.

Every test in this directory mocks the HTTP layer so no live Polygon traffic
is ever produced. The fixtures here build a minimal in-memory ``httpx.Client``
stand-in that records the URL each call hits and replays a queued sequence
of response objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest


@dataclass
class FakeResponse:
    """Minimal stand-in for :class:`httpx.Response` used by ``_request``."""

    status_code: int
    payload: dict[str, Any] | None = None
    text: str = ""
    raise_json: bool = False

    def json(self) -> dict[str, Any]:
        if self.raise_json:
            raise ValueError("not JSON")
        return self.payload or {}


@dataclass
class FakeClient:
    """In-memory replacement for :class:`httpx.Client`.

    Pops responses from ``responses`` in FIFO order. If ``raise_on_call`` is
    set, raises that exception once before continuing with the queued
    responses — handy for simulating transient network failures.
    """

    responses: list[FakeResponse | Exception] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def request(self, method: str, url: str, params: dict[str, Any] | None = None):
        self.calls.append({"method": method, "url": url, "params": params or {}})
        if not self.responses:
            raise AssertionError(f"no queued response for {method} {url}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        return None


@pytest.fixture
def fake_client_factory() -> Callable[..., FakeClient]:
    """Build :class:`FakeClient` instances with a given response queue."""

    def _make(*responses: FakeResponse | Exception) -> FakeClient:
        return FakeClient(responses=list(responses))

    return _make


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch out ``time.sleep`` in the polygon module so retry backoff is instant."""
    import markowitz.data_providers.polygon as polygon_mod

    monkeypatch.setattr(polygon_mod.time, "sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def _no_polygon_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure POLYGON_API_KEY does not leak in from the developer's shell."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)


@pytest.fixture
def httpx_error_factory() -> Callable[[str], httpx.HTTPError]:
    """Build a generic :class:`httpx.HTTPError` so tests can simulate a network fault."""

    def _make(message: str = "boom") -> httpx.HTTPError:
        return httpx.ConnectError(message)

    return _make
