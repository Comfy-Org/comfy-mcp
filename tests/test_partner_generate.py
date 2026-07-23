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
    assert cmd[4:] == ["generate", "flux-pro", "--prompt=a red fox", "--timeout=600.0"]
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
        "--timeout=600.0",
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
        "--timeout=600.0",
    ]


def test_partner_generate_forwards_download_path(patched_plain_run):
    """`download` forwards comfy-cli's `--download`, in the `=value` form."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro", download="/tmp/out.png")

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--download=/tmp/out.png",
        "--timeout=600.0",
    ]


def test_partner_generate_omits_params_when_none(patched_plain_run):
    """No params -> a bare `generate <model>`, not an empty flag token."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro")

    assert calls[0]["cmd"][4:] == ["generate", "flux-pro", "--timeout=600.0"]


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

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--prompt=a cat",
        "--timeout=600.0",
        "--yes",
    ]


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


def test_partner_generate_rejects_param_names_that_smuggle_a_value(
    patched_plain_run,
):
    """A `=` inside the KEY would land as a run-level flag comfy-cli honors.

    comfy-cli splits `--<body>` at the FIRST `=`, so `{"output-prefix=/tmp/x": v}`
    renders `--output-prefix=/tmp/x=v` and sets the blocked run-level
    `output-prefix` — the meta-flag check above never matches because it only
    sees the whole key.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="cannot contain"):
        server.partner_generate("flux-pro", params={"output-prefix=/tmp/x": "v"})

    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "flux\0pro"}, "invalid model"),
        ({"model": "flux-pro", "params": {"pro\0mpt": "a cat"}}, "parameter name"),
        ({"model": "flux-pro", "params": {"prompt": "a\0cat"}}, "value for parameter"),
        ({"model": "flux-pro", "download": "/tmp/\0.png"}, "invalid download"),
    ],
)
def test_partner_generate_rejects_embedded_nul(patched_plain_run, kwargs, match):
    """A NUL is legal in a JSON string but `subprocess` raises a bare ValueError.

    Uncaught, that surfaces as an internal error instead of a ComfyCliError, so
    catch the shape here — before any child is spawned.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match=match):
        server.partner_generate(**kwargs)

    assert calls == []


def test_partner_generate_rejects_an_empty_download_path(patched_plain_run):
    """`download=""` is a caller mistake, not "use the default location"."""
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="invalid download"):
        server.partner_generate("flux-pro", download="")

    assert calls == []


def test_partner_generate_clamps_timeout(patched_plain_run):
    """An absurd timeout is clamped, so no `comfy generate` child runs forever."""
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro", timeout_seconds=float("inf"))

    assert f"--timeout={server._MAX_GENERATE_TIMEOUT}" in calls[0]["cmd"]
    assert (
        calls[0]["timeout"]
        == server._MAX_GENERATE_TIMEOUT + server._GENERATE_TIMEOUT_GRACE
    )


def test_partner_generate_lets_the_engine_own_the_deadline(patched_plain_run):
    """`timeout_seconds` becomes comfy-cli's own `--timeout`; we only backstop it.

    Without the forwarded flag comfy-cli falls back to its 300s default, so a
    caller asking for longer silently got five minutes — and a parent kill at
    the same moment would destroy the resumable handle for a job the partner had
    already accepted and CHARGED for, inviting a double-spending retry.
    """
    calls = patched_plain_run(0, stdout="done")

    server.partner_generate("flux-pro", timeout_seconds=900.0)

    assert "--timeout=900.0" in calls[0]["cmd"]
    assert calls[0]["timeout"] == 900.0 + server._GENERATE_TIMEOUT_GRACE
    assert calls[0]["timeout"] > 900.0  # the engine gives up first


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -5.0])
def test_partner_generate_rejects_a_non_positive_or_nan_timeout(patched_plain_run, bad):
    """NaN survives `min(max(...))` — and reaches `subprocess.run` as a ValueError.

    Every NaN comparison is False, so the old clamp returned it unchanged,
    defeating the ceiling with the one value it most needed to stop. A
    non-positive timeout is refused rather than floored to an instant timeout.
    """
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        server.partner_generate("flux-pro", timeout_seconds=bad)

    assert calls == []


# --- the engine's spend gate must actually be installed ---------------------


def test_partner_generate_refuses_when_comfy_cli_has_no_spend_gate(monkeypatch):
    """No `comfy generate consent` -> no interlock -> refuse BEFORE spending.

    The fail-closed guarantee is comfy-cli's, and it landed after the >= 1.12.0
    floor this server enforces, so the version check cannot prove it is there.
    Against an older CLI a default `confirm_spend=False` call would charge the
    user with nothing to stop it.
    """
    monkeypatch.setattr(server, "_spend_gate_probed", False)
    calls: list[list[str]] = []

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
        calls.append(cmd)
        # An older comfy-cli reads `consent` as a model name and exits non-zero.
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such model")

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    with pytest.raises(server.ComfyCliError, match="no `comfy generate` spend gate"):
        server.partner_generate("flux-pro", params={"prompt": "a cat"})

    assert len(calls) == 1  # only the probe ran
    assert calls[0][4:] == ["generate", "consent", "show"]  # never the generation
    assert server._spend_gate_probed is False  # a failed probe is not latched


def test_partner_generate_probes_the_spend_gate_once_then_generates(monkeypatch):
    """A CLI that HAS the gate is probed once, then the generation goes through."""
    monkeypatch.setattr(server, "_spend_gate_probed", False)
    calls: list[list[str]] = []

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")

    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    server.partner_generate("flux-pro", confirm_spend=True)
    server.partner_generate("flux-pro", confirm_spend=True)

    assert [c[4:6] for c in calls] == [
        ["generate", "consent"],  # probed once...
        ["generate", "flux-pro"],
        ["generate", "flux-pro"],  # ...not again on the second call
    ]


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
