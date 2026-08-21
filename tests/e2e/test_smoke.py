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
3. ``nodes(action="path", from_type="MODEL", to_type="IMAGE")`` against the live
   catalog. Also a REGRESSION GUARD, and for the same reason as case 2: the tool
   used to relay whatever comfy-cli's default traversal mode returned, and on a
   pre-1.16.0 engine that was nodes which never accept a MODEL socket, wrapped in
   a perfectly valid-looking payload. No mocked test can catch that — a canned
   envelope asserts whatever it was handed — so the only real defense is asking a
   real ``object_info``. Whether the answer arrives in the engine's own default
   mode or in the ``--loose`` re-ask is invisible here BY DESIGN: this asserts the
   semantics a caller is owed, which both modes must satisfy on a healthy install.
   Needs no checkpoint, only the core node set.

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

from comfy_mcp import server

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


# Marked `e2e` and DESELECTED by default (pyproject addopts). The live-prereq
# probe runs lazily, inside the fixture, never at import: collection of a
# plain `pytest` run must not open sockets or stall on a wedged port — module
# import happens during collection even for deselected tests.
pytestmark = [pytest.mark.e2e]


@pytest.fixture(scope="module", autouse=True)
def _live_prereqs():
    """Skip the module unless comfy-cli and a live ComfyUI are reachable."""
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)


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


def test_nodes_path_model_to_image_is_semantically_sound():
    """`nodes(action="path")` MODEL -> IMAGE routes through real socket links.

    The one assertion in this repo that a mocked test structurally cannot make.
    `nodes path`'s failure mode was SILENT: the payload kept its shape while the
    routes stopped meaning anything — widget-only loaders returned as if they
    accepted MODEL, every step's `from_type` blank, every path one hop long, the
    canonical route missing. A canned envelope proves the argv and the relay; only
    a live catalog proves the answer, so this pins the semantics against whatever
    `object_info` the machine actually has — on whichever comfy-cli is installed,
    which is exactly the axis the fix is about.

    Deliberately install-agnostic. It asserts the PROPERTIES the ticket's
    acceptance is written in — every step names the socket type it traversed, and
    `KSampler -> VAEDecode` is reachable — not a fixed node list, because a custom
    node pack legitimately adds routes and a hardcoded expectation would go red on
    a perfectly healthy install. `KSampler` and `VAEDecode` are ComfyUI core, like
    the `EmptyImage`/`SaveImage` pair the round-trip above leans on, so their
    absence is a broken install rather than a taste difference.

    `max_paths` is raised well above the tool's default of 10 ON PURPOSE, and it
    is not padding a flaky assertion. The engine's traversal is breadth-first, so
    it emits ALL one-hop routes before any two-hop one and stops the moment the
    budget is full; `KSampler -> VAEDecode` is two hops, so an install carrying
    ten or more node classes that take a MODEL link and emit IMAGE in a single
    hop would push it past the default budget. Every route returned in that case
    is still a genuine socket-linked path — that is ORDERING, the thing this
    ticket put out of scope, not the silent wrongness it is about — but this test
    is checking reachability, so it must ask for enough of the set to see it.
    """
    data = server.nodes(action="path", from_type="MODEL", to_type="IMAGE", max_paths=50)

    assert isinstance(data, dict), f"nodes path returned {data!r}"
    paths = data.get("paths")
    assert isinstance(paths, list) and paths, f"no MODEL -> IMAGE paths in {data!r}"

    for path in paths:
        steps = path.get("steps") or []
        assert steps, f"path with no steps: {path!r}"
        # Blank `from_type` was symptom #1: it means the step consumed nothing,
        # i.e. the node is in the answer without accepting the type traversed.
        for step in steps:
            assert step.get("from_type"), f"step does not name its input type: {step!r}"
            assert step.get("to_type"), f"step does not name its output type: {step!r}"
        # Each hop must hand its output to the next hop's input; a chain that
        # does not chain is the same silent wrongness in a different dress.
        for earlier, later in zip(steps, steps[1:]):
            assert earlier["to_type"] == later["from_type"], (
                f"broken link {earlier!r} -> {later!r}"
            )
        assert steps[0]["from_type"] == "MODEL", (
            f"path does not start at MODEL: {path!r}"
        )
        assert steps[-1]["to_type"] == "IMAGE", f"path does not end at IMAGE: {path!r}"

    # The canonical route — the ticket's headline symptom was its absence.
    chains = [tuple(s["node"] for s in (p.get("steps") or [])) for p in paths]
    assert ("KSampler", "VAEDecode") in chains, (
        f"canonical KSampler -> VAEDecode route missing from {chains!r}"
    )
