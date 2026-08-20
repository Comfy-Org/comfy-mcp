"""Typed outbound port implemented by the local comfy-cli adapter."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

RawComfyResult: TypeAlias = tuple[dict | None, str, tuple[str, ...], int, str]


class ComfyCliClient(Protocol):
    """The only interface application/tool code uses to reach comfy-cli.

    The result body is intentionally ``Any``: comfy-cli owns the versioned
    envelope data schema and individual commands return different objects.
    Both transports expose that same tool contract; transport adapters never
    reinterpret the result body.
    """

    def run_raw(self, *args: str, timeout: float | None = None) -> RawComfyResult: ...

    def run(
        self,
        *args: str,
        timeout: float | None = None,
        plain_ok: bool = False,
    ) -> Any: ...

    async def run_async(
        self,
        *args: str,
        timeout: float | None = None,
        plain_ok: bool = False,
        stdout_cap: int | None = None,
    ) -> Any: ...

    async def run_streaming(
        self,
        *args: str,
        ctx: object | None = None,
        timeout: float | None = None,
        raise_on_timeout: bool = True,
    ) -> Any: ...
