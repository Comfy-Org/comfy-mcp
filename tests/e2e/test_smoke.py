"""End-to-end smoke test: a real no-model round-trip against a LIVE local ComfyUI.

This is the manual validation ritual turned into a command. It drives the actual
tools (no mocks): ``server_info`` -> ``run_workflow`` on a checkpoint-free
``EmptyImage`` -> ``SaveImage`` graph (both ComfyUI core nodes) -> ``fetch_outputs``,
then asserts a real PNG landed in a temp out_dir.

**Gated.** It requires BOTH a live local ComfyUI answering on ``COMFYUI_URL`` (or
``http://127.0.0.1:8188``) AND the ``comfy`` binary on ``PATH`` (or ``COMFY_BIN``).
CI runners have neither, so it SKIPS cleanly there — CI stays green via skip, not
failure. Run it on a machine that has both with::

    python -m pytest tests/e2e -m e2e      # or: scripts/smoke.sh
"""

from __future__ import annotations

import asyncio
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from comfy_local_mcp import server

# PNG signature — the 8 magic bytes every PNG starts with. Enough to prove a real
# image landed without pulling in an image library (Pillow etc.).
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_WORKFLOW = Path(__file__).parent / "workflow_smoke.json"


def _comfyui_url() -> str:
    """Base URL of the local ComfyUI to probe (env override, else the default)."""
    return os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


def _server_responds() -> bool:
    """True iff a ComfyUI HTTP server answers 200 on ``/system_stats``."""
    url = _comfyui_url() + "/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _skip_reason() -> str | None:
    """Return why the smoke test can't run here, or None if the prereqs are met."""
    if shutil.which(server.COMFY_BIN) is None:
        return f"`{server.COMFY_BIN}` not on PATH — install comfy-cli or set COMFY_BIN"
    if not _server_responds():
        return f"no live ComfyUI at {_comfyui_url()} — launch one or set COMFYUI_URL"
    return None


_SKIP_REASON = _skip_reason()

# Marked `e2e` (run in isolation via `-m e2e`) and skipped unless the live
# prerequisites are present, so a plain `pytest` in CI collects it and skips.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        _SKIP_REASON is not None, reason=_SKIP_REASON or "prereqs present"
    ),
]


def _extract_prompt_id(result: object) -> str | None:
    """Best-effort pull of a ``prompt_id`` out of ``run_workflow``'s return.

    The exact envelope shape from ``comfy run --wait`` isn't pinned down, so look
    for the common spellings at the top level and one nesting level down.
    """
    if isinstance(result, str):
        return result or None
    if isinstance(result, dict):
        for key in ("prompt_id", "promptId", "id"):
            value = result.get(key)
            if value:
                return str(value)
        for container in ("data", "job", "result"):
            nested = _extract_prompt_id(result.get(container))
            if nested:
                return nested
    return None


def test_no_model_round_trip(tmp_path):
    """server_info -> run_workflow -> fetch_outputs leaves a valid PNG on disk."""
    # 1. Confirm the wrapper can talk to the local environment at all.
    info = server.server_info()
    assert info, f"server_info() returned nothing usable: {info!r}"

    # 2. Run the checkpoint-free EmptyImage -> SaveImage workflow to completion.
    # run_workflow went async in #11 (MCP progress streaming); drive it to
    # completion from this sync test the way a plain client would.
    result = asyncio.run(
        server.run_workflow(str(_WORKFLOW), wait=True, timeout_seconds=180.0)
    )
    prompt_id = _extract_prompt_id(result)
    assert prompt_id, f"no prompt_id in run_workflow result: {result!r}"

    # 3. Collect the outputs into a temp dir and prove a real PNG landed.
    out_dir = tmp_path / "smoke_out"
    collected = server.fetch_outputs(prompt_id, str(out_dir))
    saved = collected.get("saved") or []
    assert saved, f"fetch_outputs wrote no files: {collected!r}"

    pngs = [p for p in saved if Path(p).read_bytes()[:8] == _PNG_MAGIC]
    assert pngs, f"no valid PNG among collected outputs: {saved}"
