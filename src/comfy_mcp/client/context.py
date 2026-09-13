"""Request-safe client injection and process-default composition."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

from .protocols import ComfyCliClient

ClientFactory = Callable[[], ComfyCliClient]

_bound_client: ContextVar[ComfyCliClient | None] = ContextVar(
    "comfy_mcp_client", default=None
)
_default_factory: ClientFactory | None = None
_default_client: ComfyCliClient | None = None
_default_lock = Lock()


def configure_default_factory(factory: ClientFactory) -> None:
    """Configure the composition-root factory without constructing at import."""

    global _default_factory, _default_client
    _default_factory = factory
    _default_client = None


def get_client() -> ComfyCliClient:
    """Resolve a request-bound client, falling back to the process default."""

    bound = _bound_client.get()
    if bound is not None:
        return bound
    global _default_client
    if _default_client is None:
        with _default_lock:
            if _default_client is None:
                if _default_factory is None:
                    raise RuntimeError(
                        "the comfy-cli client factory has not been configured"
                    )
                _default_client = _default_factory()
    return _default_client


@contextmanager
def bind_client(client: ComfyCliClient | None) -> Iterator[None]:
    """Bind an injected client to one request/task, including across awaits."""

    if client is None:
        yield
        return
    token = _bound_client.set(client)
    try:
        yield
    finally:
        _bound_client.reset(token)
