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
import urllib.request
from typing import Any
from urllib.parse import parse_qs, urlparse

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
def server_info() -> Any:
    """Report the local ComfyUI / comfy-cli environment.

    Wraps ``comfy env``. Returns whether a local ComfyUI server is running and
    its URL, plus the selected workspace and Python info. Call this first to
    confirm a local ComfyUI is up before running a workflow.
    """
    return _run_comfy("env", timeout=60.0)


@mcp.tool()
def run_workflow(
    workflow_path: str, wait: bool = True, timeout_seconds: float = 600.0
) -> Any:
    """Run a ComfyUI workflow JSON on the LOCAL ComfyUI.

    Accepts an API-format or UI-export workflow file. Wraps
    ``comfy run --workflow <path>``. With ``wait=True`` (default) this blocks
    until the run finishes and returns the full result; with ``wait=False`` it
    submits and returns immediately with a ``prompt_id`` to poll via
    ``job_status`` — use that for minutes-long generations so the call does not
    block.
    """
    args = ["run", "--workflow", workflow_path]
    if wait:
        args.append("--wait")
    return _run_comfy(*args, timeout=timeout_seconds if wait else 60.0)


@mcp.tool()
def job_status(prompt_id: str) -> Any:
    """Check a submitted job's status (queued / running / completed / error).

    Wraps ``comfy jobs status <prompt_id>``. Returns the job status and, when
    finished, its output references. Poll this after ``run_workflow(wait=False)``.
    """
    return _run_comfy("jobs", "status", prompt_id, timeout=60.0)


@mcp.tool()
def fetch_outputs(prompt_id: str, out_dir: str) -> Any:
    """Collect a completed job's output files into ``out_dir``; returns the paths.

    A LOCAL ComfyUI writes outputs straight to disk (and also serves them at a
    ``/view`` URL), so there is no remote download step — this resolves the job's
    outputs via ``comfy jobs status`` and, for each, copies the on-disk file or
    fetches the ``/view`` URL into ``out_dir``. (``comfy download`` is a cloud
    verb and refuses local file paths.)
    """
    status = _run_comfy("jobs", "status", prompt_id, timeout=60.0)
    outputs = status.get("outputs") or [] if isinstance(status, dict) else []
    os.makedirs(out_dir, exist_ok=True)
    saved: list[str] = []
    for ref in outputs:
        parsed = urlparse(ref)
        if parsed.scheme in ("http", "https"):
            name = parse_qs(parsed.query).get("filename", ["output"])[0]
            dst = os.path.join(out_dir, os.path.basename(name) or "output")
            with urllib.request.urlopen(ref, timeout=30) as resp, open(dst, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            saved.append(dst)
        elif os.path.exists(ref):
            dst = os.path.join(out_dir, os.path.basename(ref))
            shutil.copy2(ref, dst)
            saved.append(dst)
    return {"saved": saved, "source_outputs": outputs}


@mcp.tool()
def upload_file(paths: list[str], overwrite: bool = False) -> Any:
    """Upload local files into the LOCAL ComfyUI ``input`` directory.

    Wraps ``comfy upload <files...> [--overwrite]``. Use this to stage source
    images/masks a workflow references by filename before running it — it is
    what unlocks img2img / inpaint workflows on a local ComfyUI. Pass
    ``overwrite=True`` to replace files that already exist in the input dir
    (otherwise comfy-cli skips or errors on collisions).
    """
    args = ["upload", *paths]
    if overwrite:
        args.append("--overwrite")
    return _run_comfy(*args, timeout=300.0)


@mcp.tool()
def validate_workflow(workflow_path: str) -> Any:
    """Pre-flight a workflow against the live local ComfyUI before running it.

    Wraps ``comfy validate --workflow <path>``. Checks the workflow's
    class_types, input shapes, enum values and wiring against the running
    ComfyUI's ``object_info`` and returns the validation result — cheap
    insurance before a slow ``run_workflow``. On an invalid workflow this
    raises :class:`ComfyCliError` carrying comfy-cli's structured error code
    (e.g. ``workflow_unknown_nodes``) and message, so a missing-node or
    missing-model problem stays actionable instead of failing deep inside a run.
    """
    return _run_comfy("validate", "--workflow", workflow_path, timeout=60.0)


def main() -> None:
    """Entry point: serve the MCP over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
