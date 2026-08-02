"""Tests for driving a configurable (remote/tailnet) ComfyUI host.

The run/queue tools normally target the implicit local 127.0.0.1:8188. Setting
``COMFYUI_URL`` — or the ``COMFYUI_HOST`` / ``COMFYUI_PORT`` pair — points them
at a ComfyUI running elsewhere by forwarding ``--host`` / ``--port`` to the
comfy-cli verbs that accept them (``comfy run``, ``comfy run-template``, and
every ``comfy jobs`` subcommand). These lock in:

1. ``_comfy_target`` env parsing (URL, host/port, defaults, malformed values).
2. ``--host`` / ``--port`` forwarded into the SUBCOMMAND (not the global prefix)
   for ``run`` / ``run-template`` / ``jobs``, and NOT for verbs that don't accept
   them (``env`` / ``download`` / ``upload`` / …).
3. Byte-identical local behavior when nothing is configured.
4. ``server_info`` surfacing the configured ``comfy_target``.
5. Submit and poll agreeing on ONE server: a ``generate_image`` / ``run_template``
   submission and the ``wait_for_job`` that follows it must carry the same
   ``--host`` / ``--port``, or the ``prompt_id`` from one is meaningless to the
   other.
6. The tools that CANNOT be diverted saying so themselves: ``download_model``
   refusing outright, and ``system_stats`` / ``free_memory`` annotating their
   payload with a ``comfy_target_note``.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import _OK_STREAM, envelope

from comfy_mcp import server

# --- _comfy_target env parsing ---------------------------------------------


def test_target_none_when_unset():
    """Nothing configured -> None -> local default (byte-identical to today)."""
    assert server._comfy_target() is None


def test_target_from_host_defaults_port(monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    assert server._comfy_target() == (
        "gpu.example",
        server.DEFAULT_COMFYUI_PORT,
        "COMFYUI_HOST",
    )


def test_target_from_host_and_port(monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    assert server._comfy_target() == ("gpu.example", 9001, "COMFYUI_HOST")


def test_target_from_url(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    assert server._comfy_target() == ("gpu.example", 9001, "COMFYUI_URL")


def test_target_from_url_defaults_port(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example")
    assert server._comfy_target() == (
        "gpu.example",
        server.DEFAULT_COMFYUI_PORT,
        "COMFYUI_URL",
    )


def test_target_url_without_scheme(monkeypatch):
    """A scheme-less ``host:port`` still parses (prefixed with ``//`` internally)."""
    monkeypatch.setenv("COMFYUI_URL", "gpu.example:9001")
    assert server._comfy_target() == ("gpu.example", 9001, "COMFYUI_URL")


def test_target_url_ipv6(monkeypatch):
    """An IPv6 URL yields the bare host (comfy-cli re-brackets it for its URLs)."""
    monkeypatch.setenv("COMFYUI_URL", "http://[2001:db8::1]:8188")
    assert server._comfy_target() == ("2001:db8::1", 8188, "COMFYUI_URL")


def test_target_url_wins_over_host(monkeypatch):
    """COMFYUI_URL takes precedence over the COMFYUI_HOST/PORT pair."""
    monkeypatch.setenv("COMFYUI_URL", "http://from-url.example:1234")
    monkeypatch.setenv("COMFYUI_HOST", "from-host.example")
    monkeypatch.setenv("COMFYUI_PORT", "5678")
    assert server._comfy_target() == ("from-url.example", 1234, "COMFYUI_URL")


def test_target_url_no_host_raises(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://:8188")
    with pytest.raises(server.ComfyCliError, match="names no host"):
        server._comfy_target()


def test_target_bad_port_raises(monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "not-a-number")
    with pytest.raises(server.ComfyCliError, match="COMFYUI_PORT must be an integer"):
        server._comfy_target()


def test_target_port_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "70000")
    with pytest.raises(server.ComfyCliError, match="out of range"):
        server._comfy_target()


def test_target_url_https_scheme_rejected(monkeypatch):
    """https:// can't be forwarded (comfy-cli speaks http) -> reject, don't downgrade."""
    monkeypatch.setenv("COMFYUI_URL", "https://gpu.example:8188")
    with pytest.raises(server.ComfyCliError, match="scheme"):
        server._comfy_target()


def test_target_url_with_path_rejected(monkeypatch):
    """A reverse-proxy base path can't be forwarded -> reject, don't silently drop."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:8188/comfyui")
    with pytest.raises(server.ComfyCliError, match="path"):
        server._comfy_target()


def test_target_url_root_path_allowed(monkeypatch):
    """A bare trailing slash is not a real base path -> accepted."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001/")
    assert server._comfy_target() == ("gpu.example", 9001, "COMFYUI_URL")


def test_target_url_port_zero_rejected(monkeypatch):
    """`:0` must not silently collapse to 8188 (the COMFYUI_PORT path rejects 0)."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:0")
    with pytest.raises(server.ComfyCliError, match="out of range"):
        server._comfy_target()


def test_target_url_unbalanced_ipv6_raises_comfy_error(monkeypatch):
    """A malformed URL (`http://[::1`) surfaces as ComfyCliError, not raw ValueError."""
    monkeypatch.setenv("COMFYUI_URL", "http://[::1")
    with pytest.raises(server.ComfyCliError, match="malformed"):
        server._comfy_target()


def test_target_url_redacts_userinfo_in_error(monkeypatch):
    """A credential embedded in COMFYUI_URL is not echoed raw in the error message.

    The userinfo is written with angle-bracket placeholders (see AGENTS.md) so a
    secret scanner does not read the fixture itself as a leaked credential; the
    masking under test is scheme- and content-blind, so the assertions below are
    the same ones a bare ``user:sekret`` fixture would make.
    """
    monkeypatch.setenv("COMFYUI_URL", "https://<user>:<sekret>@gpu.example:8188")
    with pytest.raises(server.ComfyCliError) as excinfo:
        server._comfy_target()
    assert "sekret" not in str(excinfo.value)
    assert "***@gpu.example" in str(excinfo.value)


def test_target_host_strips_ipv6_brackets(monkeypatch):
    """A bracketed COMFYUI_HOST is normalized bare, matching the URL path's hostname."""
    monkeypatch.setenv("COMFYUI_HOST", "[::1]")
    assert server._comfy_target() == (
        "::1",
        server.DEFAULT_COMFYUI_PORT,
        "COMFYUI_HOST",
    )


def test_target_port_without_host_raises(monkeypatch):
    """COMFYUI_PORT alone must not silently fall back to the local default."""
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    with pytest.raises(server.ComfyCliError, match="COMFYUI_HOST is not"):
        server._comfy_target()


def test_local_only_verb_survives_malformed_config(patched_run, monkeypatch):
    """A malformed COMFYUI_URL must not brick local-only verbs (env/download/...)."""
    monkeypatch.setenv(
        "COMFYUI_URL", "https://gpu.example"
    )  # scheme rejected by _comfy_target
    calls = patched_run(envelope())

    # `env` never touches the remote, so it must run despite the bad config.
    server._run_comfy("env")

    assert calls[0]["cmd"][4:] == ["env"]


# --- forwarding into _run_comfy (plain --json path) ------------------------


def test_run_forwards_host_port_into_subcommand(patched_run, monkeypatch):
    """`comfy run` gets --host/--port appended AFTER the verb, not in the global prefix."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    calls = patched_run(envelope(data={"prompt_id": "p1"}))

    server._run_comfy("run", "--workflow", "wf.json")

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global prefix unchanged
    assert cmd[4:] == [
        "run",
        "--workflow",
        "wf.json",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_jobs_forwards_host_port_into_subcommand(patched_run, monkeypatch):
    """Every `comfy jobs` subcommand accepts --host/--port; they are forwarded."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    calls = patched_run(envelope(data={"status": "running"}))

    server._run_comfy("jobs", "status", "abc")

    assert calls[0]["cmd"][4:] == [
        "jobs",
        "status",
        "abc",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_run_template_forwards_host_port_into_subcommand(patched_run, monkeypatch):
    """`comfy run-template` accepts --host/--port, so a submit follows the remote.

    Driven through the raw wrapper rather than a tool so this asserts the
    allowlist entry itself; the two tool-level tests below cover the argv the
    tools actually build.
    """
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    calls = patched_run(envelope(data={"prompt_id": "p1"}))

    server._run_comfy("run-template", "default", "--async")

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global prefix unchanged
    assert cmd[4:] == [
        "run-template",
        "default",
        "--async",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_generate_image_submit_forwards_host_port(patched_run, monkeypatch):
    """`generate_image(wait=False)` submits to the remote, not to this machine.

    The whole ticket in one assertion: this is the easiest text-to-image
    on-ramp, it goes through `run-template`, and while that verb was off the
    allowlist the run happened HERE while `wait_for_job` polled THERE.
    """
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    calls = patched_run(envelope(data={"prompt_id": "p1"}))

    result = asyncio.run(server.generate_image("a red fox in snow", wait=False))

    assert result == {"prompt_id": "p1"}
    assert calls[0]["cmd"][4:] == [
        "run-template",
        "default",
        '--param=6.text="a red fox in snow"',
        "--timeout=60",
        "--async",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_run_template_tool_submit_forwards_host_port(patched_run, monkeypatch):
    """`run_template(wait=False)` submits to the remote too — same verb, same fix."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    calls = patched_run(envelope(data={"prompt_id": "p1"}))

    asyncio.run(
        server.run_template("image_flux2", params={"prompt": "a cat"}, wait=False)
    )

    assert calls[0]["cmd"][4:] == [
        "run-template",
        "image_flux2",
        '--param=prompt="a cat"',
        "--timeout=60",
        "--async",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_env_is_not_forwarded_host_port(patched_run, monkeypatch):
    """`comfy env` takes no --host/--port; forwarding would error 'No such option'."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    calls = patched_run(envelope())

    server._run_comfy("env")

    assert calls[0]["cmd"][4:] == ["env"]  # untouched even with a remote configured


def test_download_and_upload_not_forwarded(patched_run, monkeypatch):
    """download / upload verbs don't accept --host/--port -> stay local-only."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    calls = patched_run(envelope())

    server._run_comfy("download", "abc", "-o", "/tmp/out")
    server._run_comfy("upload", "a.png")

    assert calls[0]["cmd"][4:] == ["download", "abc", "-o", "/tmp/out"]
    assert calls[1]["cmd"][4:] == ["upload", "a.png"]


def test_fetch_outputs_docs_do_not_deny_remote_retrieval():
    """Not-forwarded must not be documented as not-working, for this one verb.

    ``download`` takes no ``--host`` / ``--port`` (asserted above) and the
    obvious reading — that ``fetch_outputs`` therefore cannot collect a job that
    ran on the configured remote — is wrong, which is why this is a tripwire
    rather than a comment. comfy-cli's ``download`` resolves the ``prompt_id``
    from the state file the SUBMITTING run wrote on THIS machine, and for a
    non-loopback target that file records each output as an absolute
    ``http://<remote>:<port>/view?…`` URL it then streams from the remote (the
    on-disk shortcut in ``run.execution.format_image_path`` is gated on a
    loopback host, so it cannot fire for a remote job). Only an id this machine
    never submitted has no such file.

    Two review passes read the non-forwarding as a broken chain and asked for
    the denial to be written into the docstrings; documenting it would have
    talked users out of a path that works. Keyed on the load-bearing MECHANISM
    (the state file) plus the absence of a denial, so the prose can be rewritten
    without churn.
    """
    doc = server.fetch_outputs.__doc__ or ""
    assert "state file" in doc, (
        "fetch_outputs no longer explains WHY it reaches a remote job's outputs "
        "— the state file is the whole mechanism"
    )
    assert "_TARGET_AWARE_SUBCOMMANDS" in doc, (
        "the docstring no longer ties its behavior to the forwarding allowlist"
    )
    # The denial this guards against, in the spellings a rewrite would reach for.
    lowered = " ".join(doc.split()).lower()
    for denial in (
        "cannot collect",
        "can't collect",
        "cannot fetch a remote",
        "can't fetch a remote",
        "unreachable for these jobs",
    ):
        assert denial not in lowered, (
            f"fetch_outputs' docstring now denies remote retrieval ({denial!r}); "
            "comfy-cli resolves the prompt_id from the local state file, so it works"
        )


def test_local_default_is_byte_identical(patched_run):
    """With nothing configured the argv is exactly today's — no --host appended."""
    calls = patched_run(envelope())

    server._run_comfy("run", "--workflow", "wf.json")

    assert calls[0]["cmd"][4:] == ["run", "--workflow", "wf.json"]  # no --host/--port


def test_run_template_local_default_is_byte_identical(patched_run):
    """Adding `run-template` to the allowlist changes NOTHING with no remote set.

    The mirror of the assertion above for the newly-forwarded verb: forwarding is
    conditional on a resolved target, so an unconfigured install must produce the
    same argv it produced before this verb was ever allowlisted.
    """
    calls = patched_run(envelope(data={"prompt_id": "p1"}))

    asyncio.run(server.generate_image("a cat", wait=False))

    assert calls[0]["cmd"][4:] == [
        "run-template",
        "default",
        '--param=6.text="a cat"',
        "--timeout=60",
        "--async",
    ]


def test_run_template_malformed_config_fails_like_run_does(patched_run, monkeypatch):
    """A malformed COMFYUI_URL breaks `run-template` no worse than it breaks `run`.

    `_with_target` checks the VERB before it resolves the target, so a bad config
    can only ever reach a verb that would have used it. For a target-aware verb
    that means a named `ComfyCliError` and NO spawn — the same failure `run` has
    always had, not a silent fall back to running on this machine, which is the
    outcome the ticket is about. (Local-only verbs are unaffected; that is
    `test_local_only_verb_survives_malformed_config` above.)
    """
    monkeypatch.setenv("COMFYUI_URL", "https://gpu.example")  # scheme rejected
    calls = patched_run(envelope(data={"prompt_id": "p1"}))

    with pytest.raises(server.ComfyCliError, match="scheme"):
        server._run_comfy("run", "--workflow", "wf.json")
    with pytest.raises(server.ComfyCliError, match="scheme"):
        asyncio.run(server.generate_image("a cat", wait=False))

    assert calls == []  # neither verb ever spawned comfy-cli


# --- forwarding into the streaming (--json-stream) path --------------------


def test_run_workflow_stream_forwards_host_port(patched_stream, monkeypatch):
    """run_workflow(wait=True) streams `comfy run --wait` with --host/--port forwarded."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    procs = patched_stream(_OK_STREAM)

    result = asyncio.run(server.run_workflow("wf.json", wait=True))

    assert result == {"outputs": ["/x.png"]}
    cmd = procs[0].cmd
    assert cmd[1:4] == ["--json-stream", "--where", "local"]  # global prefix unchanged
    assert cmd[4:] == [
        "run",
        "--workflow",
        "wf.json",
        "--wait",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_watch_job_stream_forwards_host_port(patched_stream, monkeypatch):
    """watch_job tails `comfy jobs watch <id>` with --host/--port forwarded."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.watch_job("pid"))

    assert procs[0].cmd[4:] == [
        "jobs",
        "watch",
        "pid",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_generate_image_stream_forwards_host_port(patched_stream, monkeypatch):
    """`generate_image(wait=True)` streams from the remote, with the flags forwarded."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    monkeypatch.setenv("COMFYUI_PORT", "9001")
    procs = patched_stream(_OK_STREAM)

    result = asyncio.run(server.generate_image("a cat"))

    assert result == {"outputs": ["/x.png"]}
    cmd = procs[0].cmd
    assert cmd[1:4] == ["--json-stream", "--where", "local"]  # global prefix unchanged
    assert cmd[4:] == [
        "run-template",
        "default",
        '--param=6.text="a cat"',
        "--timeout=120",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


def test_run_template_stream_forwards_host_port(patched_stream, monkeypatch):
    """`run_template(wait=True)` streams from the remote as well."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    procs = patched_stream(_OK_STREAM)

    asyncio.run(server.run_template("image_flux2"))

    assert procs[0].cmd[4:] == [
        "run-template",
        "image_flux2",
        "--timeout=120",
        "--host",
        "gpu.example",
        "--port",
        "9001",
    ]


# --- submit and poll must agree on ONE server ------------------------------


def _target_flags(cmd: list[str]) -> list[str]:
    """The `--host`/`--port` pair a spawned argv carries, or `[]` for none."""
    if "--host" not in cmd:
        return []
    start = cmd.index("--host")
    return cmd[start : start + 4]


@pytest.mark.parametrize("submit_argv", ["generate_image", "run_template"])
def test_submit_then_wait_for_job_hit_the_same_server(
    patched_run, monkeypatch, submit_argv
):
    """The end-to-end break: submit and poll must land on the SAME ComfyUI.

    This is the reported failure, not merely "the run happened on the wrong
    machine". `wait_for_job` goes through the `jobs` verb, which was already
    forwarded, while the submit went through `run-template`, which was not — so a
    client got a `prompt_id` from a LOCAL run and then asked the REMOTE queue
    about it, where it had never been submitted, and got `prompt_not_found`. The
    invariant that has to hold is agreement, so assert the two argvs carry the
    SAME target rather than re-asserting one tool's flags.
    """
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    calls = patched_run(envelope(data={"prompt_id": "p1", "status": "completed"}))

    if submit_argv == "generate_image":
        submitted = asyncio.run(server.generate_image("a cat", wait=False))
    else:
        submitted = asyncio.run(server.run_template("image_flux2", wait=False))
    polled = server.wait_for_job(submitted["prompt_id"])

    assert polled["status"] == "completed"  # no `prompt_not_found`
    assert calls[0]["cmd"][4] == "run-template"
    assert calls[1]["cmd"][4:7] == ["jobs", "status", "p1"]
    assert _target_flags(calls[0]["cmd"]) == ["--host", "gpu.example", "--port", "9001"]
    assert _target_flags(calls[0]["cmd"]) == _target_flags(calls[1]["cmd"])


# --- server_info surfaces the configured target ----------------------------


def _patch_env_for_server_info(monkeypatch, patched_run):
    """Stub `comfy env` + the version detection so `server_info()` runs offline."""
    patched_run(envelope(data={"running": False}))
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.13.0")


def test_server_info_reports_remote_target(monkeypatch, patched_run):
    _patch_env_for_server_info(monkeypatch, patched_run)
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")

    result = server.server_info()

    assert result["comfy_target"]["host"] == "gpu.example"
    assert result["comfy_target"]["port"] == 9001
    assert result["comfy_target"]["source"] == "COMFYUI_URL"
    assert result["running"] is False  # local `comfy env` data still preserved
    assert result["compatibility"]["envelope_schema"] == "envelope/1"


def test_server_info_omits_target_when_local(monkeypatch, patched_run):
    _patch_env_for_server_info(monkeypatch, patched_run)

    result = server.server_info()

    assert "comfy_target" not in result  # no remote configured -> no block


def test_server_info_reports_malformed_target_as_data(monkeypatch, patched_run):
    """A malformed remote config surfaces as a diagnostic field, not a hard failure."""
    _patch_env_for_server_info(monkeypatch, patched_run)
    monkeypatch.setenv("COMFYUI_URL", "https://gpu.example")

    result = server.server_info()

    assert "error" in result["comfy_target"]
    assert result["running"] is False  # local `comfy env` data still returned


# --- download_model refuses a configured remote ----------------------------
#
# `comfy model download` is not target-aware and never can be from here: it has
# no `--host` / `--port`, it resolves its destination from the workspace of the
# machine running THIS server, and ComfyUI's HTTP surface has no endpoint that
# writes into a models directory. So with a remote configured the download would
# "succeed" onto a disk the remote cannot read, and the run that needed the model
# would fail later on a missing file. It is refused up front instead.


def _download(**kwargs):
    """Drive the async ``download_model`` from these synchronous tests."""
    return asyncio.run(server.download_model("https://hf.co/x.safetensors", **kwargs))


@pytest.mark.parametrize("wait", [True, False])
def test_download_model_refuses_configured_url_target(patched_run, monkeypatch, wait):
    """A COMFYUI_URL remote fails the call, and spawns NOTHING — either `wait`."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    calls = patched_run(envelope(data={"download_id": "a1b2c3d4e5f6"}))

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download(wait=wait)

    message = str(excinfo.value)
    assert "gpu.example:9001" in message  # names the remote that would miss it
    assert "COMFYUI_URL" in message  # ... and which knob selected it
    assert server.REMOTE_SHARED_MODELS_ENV in message  # ... and the way out
    # The guard is only worth anything if it lands BEFORE the submit: a started
    # transfer writes to the wrong disk whatever this call then returns.
    assert calls == []


def test_download_model_refuses_configured_host_target(patched_run, monkeypatch):
    """The COMFYUI_HOST spelling is guarded too, with its default port named."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    calls = patched_run(envelope(data={"download_id": "a1b2c3d4e5f6"}))

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download(wait=False)

    message = str(excinfo.value)
    assert f"gpu.example:{server.DEFAULT_COMFYUI_PORT}" in message
    assert "COMFYUI_HOST" in message
    assert server.REMOTE_SHARED_MODELS_ENV in message
    assert calls == []


def test_download_model_raises_on_malformed_url_target(patched_run, monkeypatch):
    """A malformed remote config fails LOUDLY here, unlike the local-only verbs.

    ``_with_target`` deliberately ignores a malformed ``COMFYUI_URL`` for verbs
    that never touch the remote, so a bad value cannot brick them. This tool is
    the opposite case: the caller asked for a remote, and the whole question the
    guard answers is *which* one — so an unparseable answer is an error, not
    something to shrug off and download locally.
    """
    monkeypatch.setenv("COMFYUI_URL", "https://gpu.example")  # scheme rejected
    calls = patched_run(envelope(data={"download_id": "a1b2c3d4e5f6"}))

    with pytest.raises(server.ComfyCliError, match="scheme"):
        _download(wait=False)

    assert calls == []


def test_download_model_shared_models_optin_downloads_unchanged(
    patched_run, monkeypatch
):
    """The shared-storage opt-in restores TODAY's argv exactly — no target flags.

    `model` is still not target-aware, so the escape hatch must not start
    forwarding `--host` / `--port` to it; it only says "this workspace IS the
    remote's models volume, proceed."
    """
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    monkeypatch.setenv(server.REMOTE_SHARED_MODELS_ENV, "1")
    submit = {"download_id": "a1b2c3d4e5f6", "dest": "/models/x.safetensors"}
    calls = patched_run(envelope(data=submit))

    assert _download(wait=False) == submit

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/x.safetensors",
        "--background",
    ]


def test_download_model_shared_models_optin_ignores_malformed_target(
    patched_run, monkeypatch
):
    """Opted in, the target is never resolved — so a malformed one cannot raise.

    The opt-in means "behave exactly as an unconfigured install does", and an
    unconfigured `download_model` ignores `COMFYUI_URL` entirely.
    """
    monkeypatch.setenv("COMFYUI_URL", "https://gpu.example")  # scheme rejected
    monkeypatch.setenv(server.REMOTE_SHARED_MODELS_ENV, "1")
    calls = patched_run(envelope(data={"download_id": "a1b2c3d4e5f6"}))

    _download(wait=False)

    assert calls[0]["cmd"][4] == "model"


@pytest.mark.parametrize("value", ["", "0", "false", "no", "maybe", " "])
def test_download_model_shared_models_optin_fails_closed(
    patched_run, monkeypatch, value
):
    """Anything but a recognized truthy value leaves the guard ARMED.

    A typo'd opt-in must not silently land a multi-GB checkpoint on the wrong
    machine — the failure names the value to set.
    """
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    monkeypatch.setenv(server.REMOTE_SHARED_MODELS_ENV, value)
    calls = patched_run(envelope(data={"download_id": "a1b2c3d4e5f6"}))

    with pytest.raises(server.ComfyCliError, match="LOCAL-ONLY"):
        _download(wait=False)

    assert calls == []


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_download_model_shared_models_optin_spellings(patched_run, monkeypatch, value):
    """The documented `1` plus the obvious synonyms, case- and space-insensitive."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    monkeypatch.setenv(server.REMOTE_SHARED_MODELS_ENV, value)
    calls = patched_run(envelope(data={"download_id": "a1b2c3d4e5f6"}))

    _download(wait=False)

    assert calls[0]["cmd"][4] == "model"


def test_download_model_unconfigured_is_unchanged(patched_run):
    """No remote configured -> byte-identical to today (the guard is a no-op)."""
    submit = {"download_id": "a1b2c3d4e5f6", "dest": "/models/x.safetensors"}
    calls = patched_run(envelope(data=submit))

    assert _download(wait=False) == submit

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/x.safetensors",
        "--background",
    ]


def test_download_lifecycle_tools_are_not_guarded(patched_run, monkeypatch):
    """status / wait / cancel manage an ALREADY-submitted local download.

    Those ids only exist because a download was submitted on this machine while
    it was allowed to be, so guarding them would strand a transfer mid-flight —
    unpollable and uncancellable — the moment a remote was configured.
    """
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    calls = patched_run(envelope(data={"status": "completed"}))

    server.download_status("a1b2c3d4e5f6")
    server.wait_for_download("a1b2c3d4e5f6", timeout_seconds=1.0)
    server.cancel_download("a1b2c3d4e5f6")

    assert [call["cmd"][4:6] for call in calls] == [
        ["model", "download-status"],
        ["model", "download-status"],
        ["model", "download-cancel"],
    ]


# --- system_stats / free_memory annotate a configured remote ---------------
#
# `comfy system-stats` and `comfy free` take no `--host` / `--port` (only
# `--where`), so they read and free whichever ComfyUI comfy-cli itself targets
# while `run_workflow` submits to the configured remote. That divergence is a
# WARNING, not an error — freeing local VRAM while running remote is legitimate
# (the local-LLM coexistence recipe) — so the payload gains a `comfy_target_note`
# rather than the call failing.

_STATS = {
    "devices": [{"name": "cuda:0", "vram_free": 11_000_000_000}],
    "system": {"ram_free": 30_000_000_000, "comfyui_version": "0.3.0"},
}


def test_system_stats_annotates_configured_remote(patched_run, monkeypatch):
    """The note names the remote the RUN tools use; the stats themselves are intact."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    patched_run(envelope(data=_STATS))

    result = server.system_stats()

    note = result["comfy_target_note"]
    assert note["host"] == "gpu.example"
    assert note["port"] == server.DEFAULT_COMFYUI_PORT
    assert note["source"] == "COMFYUI_HOST"
    assert "NOT the remote" in note["note"]
    # Annotation only — comfy-cli's own payload is passed through untouched.
    assert result["devices"] == _STATS["devices"]
    assert result["system"] == _STATS["system"]


def test_free_memory_annotates_configured_remote(patched_run, monkeypatch):
    """`free` gets the same note alongside comfy-cli's `requested` acknowledgement."""
    monkeypatch.setenv("COMFYUI_URL", "http://gpu.example:9001")
    ack = {"requested": {"unload_models": True}, "note": "queued"}
    patched_run(envelope(data=ack))

    result = server.free_memory()

    assert result["comfy_target_note"]["host"] == "gpu.example"
    assert result["comfy_target_note"]["port"] == 9001
    assert result["comfy_target_note"]["source"] == "COMFYUI_URL"
    assert result["requested"] == ack["requested"]
    assert result["note"] == "queued"  # comfy-cli's own `note` is not overwritten


def test_resource_tools_unconfigured_are_byte_identical(patched_run):
    """No remote configured -> no key, and the payload is exactly what comfy-cli sent."""
    patched_run(envelope(data=_STATS))
    assert server.system_stats() == _STATS

    ack = {"requested": {"unload_models": True}}
    patched_run(envelope(data=ack))
    assert server.free_memory() == ack


def test_resource_tools_survive_malformed_target(patched_run, monkeypatch):
    """A malformed remote config must not break these local-only tools.

    Same contract `_with_target` honors for the local-only verbs: the caller
    never asked these two to reach the remote, so an unparseable `COMFYUI_URL`
    costs the annotation, not the call.
    """
    monkeypatch.setenv("COMFYUI_URL", "https://gpu.example")  # scheme rejected

    patched_run(envelope(data=_STATS))
    assert server.system_stats() == _STATS

    ack = {"requested": {"unload_models": True}}
    patched_run(envelope(data=ack))
    assert server.free_memory() == ack


def test_resource_tools_pass_through_foreign_payload_shapes(patched_run, monkeypatch):
    """A payload that isn't a dict is returned untouched, not reshaped.

    Mirrors `_drop_cloud_jobs`: a comfy-cli that answers with some other shape is
    handed to the caller as-is rather than wrapped to make room for the note.
    """
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")

    patched_run(envelope(data=["not", "a", "dict"]))
    assert server.system_stats() == ["not", "a", "dict"]

    patched_run(envelope(data="ok"))
    assert server.free_memory() == "ok"


def test_resource_tools_do_not_clobber_a_colliding_key(patched_run, monkeypatch):
    """If comfy-cli ever emitted the key itself, ITS value wins — never overwritten."""
    monkeypatch.setenv("COMFYUI_HOST", "gpu.example")
    payload = {"devices": [], "comfy_target_note": "comfy-cli's own"}
    patched_run(envelope(data=payload))

    assert server.system_stats() == payload
