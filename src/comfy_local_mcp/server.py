"""comfy-local-mcp — a thin MCP wrapper over comfy-cli.

Every tool shells out to the ``comfy`` command (comfy-cli), pinned to the LOCAL
target, asks for JSON, parses comfy-cli's versioned ``envelope/1`` result, and
returns its ``data``. There is deliberately no HTTP client and no code shared
with the Comfy Cloud MCP — comfy-cli is the engine.

First cut: the run -> get-output core loop (2 tools). Next tools to add, each a
one-line passthrough: ``job_status`` (``comfy jobs status``), ``discover``
(``comfy discover`` / ``comfy which``), ``launch``/``stop``
(``comfy launch --background`` / ``comfy stop``).

NOTE: the exact ``comfy`` invocation + envelope shape still need a smoke test
against a real comfy-cli install and a running local ComfyUI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("comfy-local-mcp")

# Allow overriding the binary (e.g. a venv path) without touching code.
COMFY_BIN = os.environ.get("COMFY_BIN", "comfy")


class ComfyCliError(RuntimeError):
    """comfy-cli was missing, timed out, or returned an error envelope."""


def _run_comfy(*args: str, timeout: float | None = None) -> Any:
    """Run ``comfy <args> --where local --json`` and return the envelope's ``data``.

    comfy-cli emits a versioned ``envelope/1`` object on stdout (a single line
    for ``--json``, or an NDJSON stream whose final line is the envelope). We
    keep the last JSON object and unwrap ``ok`` / ``data`` / ``error``.
    """
    if shutil.which(COMFY_BIN) is None:
        raise ComfyCliError(
            f"`{COMFY_BIN}` not found on PATH. Install comfy-cli "
            "(`pip install comfy-cli`) or set the COMFY_BIN env var."
        )
    # Global flags (--json, --where) MUST precede the subcommand in comfy-cli;
    # a trailing --json errors with "No such option". (Verified against comfy-cli.)
    cmd = [COMFY_BIN, "--json", "--where", "local", *args]
    # Belt-and-suspenders: pin the target via env too, so we never touch cloud.
    env = {**os.environ, "COMFY_WHERE": "local"}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ComfyCliError(
            f"comfy-cli timed out after {timeout}s: {' '.join(cmd)}"
        ) from exc

    envelope = _last_json_object(proc.stdout)
    if envelope is None:
        raise ComfyCliError(
            f"comfy-cli returned no JSON (exit {proc.returncode}). "
            f"stderr: {proc.stderr.strip()[:500]}"
        )
    if not envelope.get("ok", False):
        err = envelope.get("error") or {}
        raise ComfyCliError(
            f"comfy {' '.join(args)} failed "
            f"[{err.get('code', 'unknown')}]: "
            f"{err.get('message') or proc.stderr.strip()[:500]}"
        )
    return envelope.get("data")


def _last_json_object(stdout: str) -> dict | None:
    """Return the last JSON object on stdout, preferring a ``type==envelope`` one."""
    best: dict | None = None
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "envelope":
            best = obj  # an explicit envelope always wins; keep the latest
        elif best is None or best.get("type") != "envelope":
            best = obj  # fallback to any JSON object until an envelope appears
    return best


@mcp.tool()
def run_workflow(workflow_path: str, timeout_seconds: float = 600.0) -> Any:
    """Run a ComfyUI workflow JSON on the LOCAL ComfyUI and wait for it to finish.

    Accepts an API-format or UI-export workflow file. Wraps
    ``comfy run --workflow <path> --wait``. Returns the run result (prompt_id,
    status, and output references). For minutes-long generations, raise
    ``timeout_seconds`` — this call blocks until the run completes.
    """
    return _run_comfy(
        "run", "--workflow", workflow_path, "--wait", timeout=timeout_seconds
    )


@mcp.tool()
def fetch_outputs(prompt_id: str, out_dir: str) -> Any:
    """Download a completed job's output files to a local directory.

    Wraps ``comfy download <prompt_id> --out-dir <dir>``. Returns the saved
    files' absolute local paths.
    """
    return _run_comfy("download", prompt_id, "--out-dir", out_dir)


def main() -> None:
    """Entry point: serve the MCP over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
