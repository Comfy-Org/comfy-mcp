"""End-to-end smoke tests: real round-trips against a LIVE local ComfyUI.

This is the manual validation ritual turned into a command. It drives the actual
tools (no mocks):

1. ``server_info`` -> ``run_workflow`` on a checkpoint-free ``EmptyImage`` ->
   ``SaveImage`` graph (both ComfyUI core nodes) -> ``fetch_outputs``, then
   asserts a real PNG landed in a temp out_dir.
2. ``generate_image(prompt=…, wait=True)`` — the text-prompt on-ramp — end to
   end, likewise down to a real PNG. This one is a REGRESSION GUARD: the tool
   used to wrap ``comfy generate``, a partner/cloud-only verb with no local mode
   and no ``--prompt`` flag, so every call died in comfy-cli's argument parser
   ("comfy-cli returned no JSON (exit 1)") and nothing was ever enqueued. No unit
   test could catch that — mocks happily accept an argv the real CLI rejects —
   so the only real defense is running it. Unlike case 1 it needs the default
   template's SD1.5 checkpoint (``v1-5-pruned-emaonly-fp16.safetensors``)
   installed locally; without it the run fails with comfy-cli's own missing-model
   error rather than skipping.

**Gated.** They require BOTH a live local ComfyUI answering on ``COMFYUI_URL`` (or
``http://127.0.0.1:8188``) AND the ``comfy`` binary on ``PATH`` (or ``COMFY_BIN``).
CI runners have neither, so they SKIP cleanly there — CI stays green via skip, not
failure. Run them on a machine that has both with::

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


_DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"


def _comfyui_url() -> str:
    """Base URL of the local ComfyUI to probe (env override, else the default)."""
    return os.environ.get("COMFYUI_URL", _DEFAULT_COMFYUI_URL).rstrip("/")


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

    # 3. Download the outputs into a temp dir and prove a real PNG landed there.
    #    fetch_outputs now wraps `comfy download --where local -o <dir>`, so we
    #    assert on the files it writes into out_dir rather than on its return shape.
    out_dir = tmp_path / "smoke_out"
    server.fetch_outputs(prompt_id, str(out_dir))

    pngs = [
        p
        for p in out_dir.rglob("*")
        if p.is_file() and p.read_bytes()[:8] == _PNG_MAGIC
    ]
    assert pngs, f"no valid PNG downloaded into {out_dir}"


def test_system_stats_reports_devices():
    """system_stats returns a real, non-empty devices list from the live server.

    The unit tests pin the argv and pass a canned envelope through; only a live
    run proves the pinned comfy-cli actually HAS the `system-stats` verb and that
    ComfyUI answers it with the shape the tool's docstring promises. That gap is
    exactly what this tool's version-skew hint exists for, so it is worth one
    real call.

    Skipped when `COMFYUI_URL` points somewhere other than the default loopback:
    the module gate probes THAT url, but `system_stats` is explicitly not
    diverted by it (`comfy system-stats` takes no `--host`/`--port`), so it would
    query whatever comfy-cli resolves locally — possibly nothing. Without this
    guard a remote-ComfyUI setup fails the test spuriously instead of skipping.
    """
    if _comfyui_url() != _DEFAULT_COMFYUI_URL:
        pytest.skip(
            f"COMFYUI_URL={_comfyui_url()} is remote, but `comfy system-stats` "
            "always targets comfy-cli's own local resolution"
        )

    stats = server.system_stats()

    assert isinstance(stats, dict), f"system-stats returned {stats!r}"
    devices = stats.get("devices")
    assert isinstance(devices, list) and devices, f"no devices in {stats!r}"
    # `vram_free` is the number the VRAM-coordination recipe reads; prove it is
    # a usable number rather than a string or a missing key.
    assert isinstance(devices[0].get("vram_free"), (int, float))


def test_generate_image_round_trip(tmp_path):
    """generate_image(prompt, wait=True) enqueues a real job and yields a PNG.

    The regression guard for the ``comfy generate`` era: a green run proves the
    pinned comfy-cli actually PARSES the argv this tool emits and runs the
    default text-to-image template to completion. Needs the template's SD1.5
    checkpoint installed (see the module docstring).
    """
    # `generate_image` raises ComfyCliError on an error envelope, so simply
    # returning is already the "non-error envelope" half of the assertion.
    result = asyncio.run(
        server.generate_image("a cat", wait=True, timeout_seconds=600.0)
    )
    prompt_id = _extract_prompt_id(result)
    assert prompt_id, f"no prompt_id in generate_image result: {result!r}"

    out_dir = tmp_path / "generate_out"
    server.fetch_outputs(prompt_id, str(out_dir))

    pngs = [
        p
        for p in out_dir.rglob("*")
        if p.is_file() and p.read_bytes()[:8] == _PNG_MAGIC
    ]
    assert pngs, f"generate_image produced no valid PNG in {out_dir}"
