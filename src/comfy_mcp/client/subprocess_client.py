"""Concrete comfy-cli client composed from the subprocess runner functions.

The runner callables are injected instead of imported from ``server._internal`` so this
leaf package never points back at an MCP adapter.  It also preserves the
existing runner test seams while application code gains one explicit outbound
client dependency.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .protocols import RawComfyResult

RawRunner = Callable[..., RawComfyResult]
Runner = Callable[..., Any]
AsyncRunner = Callable[..., Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class SubprocessComfyCliClient:
    """Delegate the outbound port to the four guarded subprocess runners."""

    raw_runner: RawRunner
    runner: Runner
    async_runner: AsyncRunner
    streaming_runner: AsyncRunner

    def run_raw(self, *args: str, timeout: float | None = None) -> RawComfyResult:
        return self.raw_runner(*args, timeout=timeout)

    def run(
        self,
        *args: str,
        timeout: float | None = None,
        plain_ok: bool = False,
    ) -> Any:
        if plain_ok:
            return self.runner(*args, timeout=timeout, plain_ok=True)
        return self.runner(*args, timeout=timeout)

    async def run_async(
        self,
        *args: str,
        timeout: float | None = None,
        plain_ok: bool = False,
        stdout_cap: int | None = None,
    ) -> Any:
        if plain_ok and stdout_cap is not None:
            return await self.async_runner(
                *args,
                timeout=timeout,
                plain_ok=True,
                stdout_cap=stdout_cap,
            )
        if plain_ok:
            return await self.async_runner(*args, timeout=timeout, plain_ok=True)
        if stdout_cap is not None:
            return await self.async_runner(
                *args, timeout=timeout, stdout_cap=stdout_cap
            )
        return await self.async_runner(*args, timeout=timeout)

    async def run_streaming(
        self,
        *args: str,
        ctx: object | None = None,
        timeout: float | None = None,
        raise_on_timeout: bool = True,
    ) -> Any:
        kwargs: dict[str, object] = {"timeout": timeout}
        if ctx is not None:
            kwargs["ctx"] = ctx
        if not raise_on_timeout:
            kwargs["raise_on_timeout"] = False
        return await self.streaming_runner(*args, **kwargs)
