"""Tests for ``partner_generate`` — the thin passthrough to ``comfy generate <model>``.

These lock in the three things this wrapper is responsible for, given that all
the partner logic (model catalog, schema validation, the spend gate itself)
lives in comfy-cli:

1. The passthrough argv: global flags before the subcommand, the model as the
   first positional, per-model params as ``--name=value``.
2. The SPEND CONSENT posture — ``--yes`` is sent only when the caller explicitly
   sets ``confirm_spend=True``, and there is no second, covert way to send it.
   The engine still owns the interlock; this only proves what we forward.
3. The no-envelope result contract: ``comfy generate`` prints human text and
   exits 0 without an ``envelope/1``, so success is synthesized (``plain_ok``)
   while a non-zero exit — the engine's fail-closed consent refusal — raises.

comfy-cli is mocked throughout: no real ComfyUI, and no real credit spend.
"""

from __future__ import annotations

import subprocess

import pytest

from comfy_local_mcp import server


def _fake_run_plain(returncode: int, stdout: str = "", stderr: str = ""):
    """``subprocess.run`` stand-in emitting HUMAN text (no envelope) + a returncode.

    Mirrors the real ``comfy generate``, which prints its result through its own
    printer and never emits a renderer envelope on the generate path.
    """
    calls: list[dict] = []

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
        calls.append({"cmd": cmd, "env": env, "timeout": timeout})
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=stdout, stderr=stderr
        )

    return fake, calls


@pytest.fixture
def patched_plain_run(monkeypatch):
    """``setup(returncode, stdout, stderr) -> calls`` for the no-envelope path."""

    def setup(returncode: int = 0, stdout: str = "", stderr: str = "") -> list[dict]:
        fake, calls = _fake_run_plain(returncode, stdout, stderr)
        monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
        monkeypatch.setattr(server.subprocess, "run", fake)
        return calls

    return setup


# --- passthrough argv -------------------------------------------------------


def test_partner_generate_argv_is_a_local_json_passthrough(patched_plain_run):
    """`comfy --json --where local generate <model> --param=value` (flags first)."""
    calls = patched_plain_run(0, stdout="Saved image to /tmp/out.png")

    server.partner_generate("flux-pro", params={"prompt": "a red fox"})

    cmd = calls[0]["cmd"]
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["generate", "flux-pro", "--prompt=a red fox"]
    assert calls[0]["env"]["COMFY_WHERE"] == "local"  # belt-and-suspenders pin


def test_partner_generate_marshals_param_types(patched_plain_run):
    """Params render in the spellings comfy-cli's schema parser accepts."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate(
        "flux-pro",
        params={
            "prompt": "a cat",
            "steps": 30,  # int -> plain str
            "guidance": 3.5,  # float -> plain str
            "raw": True,  # bool -> lowercase true/false, never Python's "True"
            "tiled": False,
            "sizes": [512, 768],  # list -> JSON, the form its array parser takes
            "seed": None,  # dropped entirely, NOT sent as the string "None"
        },
    )

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--prompt=a cat",
        "--steps=30",
        "--guidance=3.5",
        "--raw=true",
        "--tiled=false",
        "--sizes=[512, 768]",
    ]


def test_partner_generate_param_value_with_leading_dash_stays_a_value(
    patched_plain_run,
):
    """`--name=value` keeps a dash-leading value from being read as the next flag."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro", params={"prompt": "--not-a-flag, just text"})

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--prompt=--not-a-flag, just text",
    ]


def test_partner_generate_forwards_download_path(patched_plain_run):
    """`download` forwards comfy-cli's `--download`, in the `=value` form."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro", download="/tmp/out.png")

    assert calls[0]["cmd"][4:] == ["generate", "flux-pro", "--download=/tmp/out.png"]


def test_partner_generate_omits_params_when_none(patched_plain_run):
    """No params -> a bare `generate <model>`, not an empty flag token."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro")

    assert calls[0]["cmd"][4:] == ["generate", "flux-pro"]


# --- spend consent ----------------------------------------------------------


def test_partner_generate_withholds_consent_by_default(patched_plain_run):
    """The default sends NO `--yes`: comfy-cli's gate is left to fail closed."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro", params={"prompt": "a cat"})

    assert "--yes" not in calls[0]["cmd"]


def test_partner_generate_forwards_consent_when_confirmed(patched_plain_run):
    """`confirm_spend=True` forwards `--yes`, comfy-cli's non-interactive consent."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro", params={"prompt": "a cat"}, confirm_spend=True)

    assert calls[0]["cmd"][4:] == ["generate", "flux-pro", "--prompt=a cat", "--yes"]


def test_partner_generate_consent_refusal_raises_and_surfaces_the_reason(
    patched_plain_run,
):
    """The engine's fail-closed refusal (non-zero exit) raises — no false success.

    This is the interlock working: comfy-cli exits 1 without spending when no
    consent was given, so `plain_ok` must NOT synthesize a success for it, and
    the CLI's own remediation text has to reach the caller.
    """
    patched_plain_run(
        1,
        stderr=(
            "`comfy generate flux-pro` spends Comfy credits and no consent was "
            "given. Re-run with --yes, or persist consent with "
            "`comfy generate consent always`."
        ),
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.partner_generate("flux-pro", params={"prompt": "a cat"})

    assert "spends Comfy credits" in str(excinfo.value)  # the reason survives
    assert "--yes" in str(excinfo.value)  # and so does the way forward


@pytest.mark.parametrize("name", ["yes", "async", "json", "download", "api_key"])
def test_partner_generate_refuses_run_level_flags_as_params(patched_plain_run, name):
    """A run-level comfy-cli flag can't be smuggled in through `params`.

    Most important is `yes`: without this guard, `params={"yes": True}` would
    render `--yes` and grant spend consent behind `confirm_spend`'s back — a
    second, undocumented consent path. `json`/`async` would instead break the
    result contract this tool documents. Underscore spellings are caught too.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="run-level"):
        server.partner_generate("flux-pro", params={name: True})

    assert calls == []  # refused before comfy-cli was ever invoked


# --- input guards -----------------------------------------------------------


@pytest.mark.parametrize("model", ["", "--help", "-x"])
def test_partner_generate_rejects_option_like_model(patched_plain_run, model):
    """An empty/dash-leading model would be parsed by comfy-cli as an option."""
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="invalid model"):
        server.partner_generate(model)

    assert calls == []  # never reached comfy-cli


@pytest.mark.parametrize(
    "target", ["list", "schema", "refresh", "upload", "resume", "consent"]
)
def test_partner_generate_rejects_reserved_subactions(patched_plain_run, target):
    """comfy-cli's own `generate` sub-actions are not partner models.

    `consent` matters most — it is the spend gate's configuration surface, and
    this tool must not become a back door to it.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="sub-action"):
        server.partner_generate(target)

    assert calls == []


def test_partner_generate_rejects_option_like_param_name(patched_plain_run):
    """A dash-leading/empty param name would emit a malformed `---flag` token."""
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="invalid parameter name"):
        server.partner_generate("flux-pro", params={"--prompt": "a cat"})

    assert calls == []


def test_partner_generate_clamps_timeout(patched_plain_run):
    """An absurd timeout is clamped, so no `comfy generate` child runs forever."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro", timeout_seconds=float("inf"))

    assert calls[0]["timeout"] == server._MAX_GENERATE_TIMEOUT


# --- result contract (no envelope on the generate path) ---------------------


def test_partner_generate_synthesizes_success_without_an_envelope(patched_plain_run):
    """`comfy generate` exits 0 printing text -> synthesized success, not an error.

    Raising here would be a false negative on a call that already SPENT the
    user's credits, inviting a retry that spends them twice.
    """
    patched_plain_run(0, stdout="Generated 1 image -> /tmp/out.png")

    result = server.partner_generate(
        "flux-pro", params={"prompt": "a cat"}, confirm_spend=True
    )

    assert result["ok"] is True
    assert result["action"] == "generate flux-pro"  # stops before the flags
    assert "/tmp/out.png" in result["message"]


def test_partner_generate_prefers_a_real_envelope_when_comfy_cli_emits_one(
    monkeypatch,
):
    """If comfy-cli ever DOES emit an envelope for generate, its data wins.

    `plain_ok` is a stopgap, not an override: a real `envelope/1` is unwrapped
    normally, so this tool needs no change when the CLI starts emitting one.
    """

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"schema": "envelope/1", "type": "envelope", "ok": true, '
            '"data": {"images": ["/tmp/out.png"]}}',
            stderr="",
        )

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    result = server.partner_generate("flux-pro", confirm_spend=True)

    assert result == {"images": ["/tmp/out.png"]}


def test_partner_generate_error_envelope_raises_with_code(monkeypatch):
    """A real error envelope still surfaces comfy-cli's structured code."""

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout='{"schema": "envelope/1", "type": "envelope", "ok": false, '
            '"error": {"code": "unknown_model", "message": "no such model"}}',
            stderr="",
        )

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match="unknown_model"):
        server.partner_generate("nope-not-a-model", confirm_spend=True)


def test_instructions_warn_that_partner_generate_spends_credits():
    """The handshake must teach the spend posture, not just the tool's name."""
    instructions = server.mcp.instructions

    assert "partner_generate" in instructions
    assert "confirm_spend" in instructions
    assert "credits" in instructions
