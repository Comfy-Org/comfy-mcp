"""Tests for ``emit_partner_workflow`` — ``comfy generate <model> --emit-workflow``.

This tool is the local counterpart to ``partner_generate``: instead of posting
to the partner proxy it asks comfy-cli to WRITE a runnable graph containing the
partner's API node, which ``run_workflow`` then executes on the user's own
ComfyUI. What the wrapper is responsible for, and what these lock in:

1. The passthrough argv — the model as the first positional, ``params`` as
   ``--name=value``, and ``--emit-workflow=<out_path>`` reaching the CLI.
2. That its input validation is literally ``partner_generate``'s, so the two
   tools cannot drift on what they accept.
3. That it does NOT run the spend interlock: emit-workflow calls no partner API
   and spends nothing, so probing the gate or prompting for consent would be
   asking the user to approve a cost that does not exist.
4. The result contract — unlike the proxy path this shares a verb with,
   ``--emit-workflow`` DOES emit an ``envelope/1``, so ``data`` comes back
   structured and a failure surfaces comfy-cli's ``emit_workflow_failed``
   message intact, supported-model list and all.
5. The capability gate — this tool's "spends nothing" claim only holds on a
   comfy-cli that actually recognises ``--emit-workflow`` as a flag, so it must
   REFUSE (not silently run a spending proxy call) against one that does not.

comfy-cli is mocked throughout: nothing here runs a real CLI or writes a graph.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import envelope

from comfy_mcp import argv, params
from comfy_mcp.server import _internal as server


def _emit(*args, **kwargs):
    """Drive the async ``emit_partner_workflow`` tool from a sync test."""
    return asyncio.run(server.emit_partner_workflow(*args, **kwargs))


# The envelope comfy-cli 1.13.0 emits for a successful emit-workflow run.
_EMIT_DATA = {"out": "/tmp/flux.json", "model": "flux-pro", "nodes": 2}

# comfy-cli's own refusal for a model it has no node mapping for. The whole
# point of surfacing this verbatim is that it names the supported set, so an
# agent can retarget without a round trip.
_UNSUPPORTED_ERROR = {
    "code": "emit_workflow_failed",
    "message": (
        "--emit-workflow does not support model 'flux-ultra'.\n"
        "Supported models: flux-2, flux-pro, kling-i2v, nano-banana, seedance.\n"
        "These map to ComfyUI API nodes; other proxy models have no node "
        "mapping yet."
    ),
    "hint": "check the model name and that all required inputs are provided",
}


# --- passthrough argv -------------------------------------------------------


def test_emit_flag_reaches_the_cli(patched_run):
    """`comfy --json --where local generate <model> --param=… --emit-workflow=<path>`."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    _emit("flux-pro", "/tmp/flux.json", params={"prompt": "a red fox"})

    cmd = calls[0]["cmd"]
    assert cmd[0] == server.COMFY_BIN
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == [
        "generate",
        "flux-pro",
        "--prompt=a red fox",
        "--emit-workflow=/tmp/flux.json",
    ]
    assert calls[0]["env"]["COMFY_WHERE"] == "local"


def test_emit_path_uses_the_equals_form_so_a_dash_path_stays_a_value(patched_run):
    """`--emit-workflow=<path>` (not two tokens), so `./-out.json` is the value."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    _emit("flux-pro", "./-out.json")

    assert calls[0]["cmd"][-1] == "--emit-workflow=./-out.json"


def test_emit_works_without_params(patched_run):
    """Params are optional here — the partner node carries its own defaults."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    _emit("flux-pro", "/tmp/flux.json")

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--emit-workflow=/tmp/flux.json",
    ]


def test_emit_marshals_param_types_like_partner_generate(patched_run):
    """Params render through the SAME marshaller, in the same spellings."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    _emit(
        "flux-pro",
        "/tmp/flux.json",
        params={
            "prompt": "a cat",
            "steps": 30,
            "raw": True,
            "sizes": [512, 768],
            "seed": None,  # dropped, not sent as the string "None"
        },
    )

    assert calls[0]["cmd"][4:] == [
        "generate",
        "flux-pro",
        "--prompt=a cat",
        "--steps=30",
        "--raw=true",
        "--sizes=[512, 768]",
        "--emit-workflow=/tmp/flux.json",
    ]


# --- validation shared with partner_generate --------------------------------


@pytest.mark.parametrize("name", ["yes", "async", "json", "download", "api_key"])
def test_emit_refuses_run_level_flags_as_params(patched_run, name):
    """`_GENERATE_META_FLAGS` binds here too — same validator, same refusals."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    with pytest.raises(server.ComfyCliError, match="run-level"):
        _emit("flux-pro", "/tmp/flux.json", params={name: True})

    assert calls == []  # refused before comfy-cli was ever invoked


def test_emit_refuses_emit_workflow_as_a_param(patched_run):
    """The flag this tool owns cannot ALSO be smuggled through `params`.

    Two `--emit-workflow` tokens would let a caller retarget the write behind
    `out_path`'s back — comfy-cli's meta-flag parser keeps the last one.
    """
    calls = patched_run(envelope(data=_EMIT_DATA))

    with pytest.raises(server.ComfyCliError, match="run-level"):
        _emit("flux-pro", "/tmp/flux.json", params={"emit_workflow": "/tmp/evil.json"})

    assert calls == []


@pytest.mark.parametrize(
    "target", ["list", "schema", "refresh", "upload", "resume", "consent"]
)
def test_emit_rejects_reserved_subactions(patched_run, target):
    """`_GENERATE_RESERVED_TARGETS` binds here too, `consent` above all."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    with pytest.raises(server.ComfyCliError, match="sub-action"):
        _emit(target, "/tmp/flux.json")

    assert calls == []


@pytest.mark.parametrize("model", ["", "--help", "-x"])
def test_emit_rejects_option_like_model(patched_run, model):
    """An empty/dash-leading model would be parsed by comfy-cli as an option."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    with pytest.raises(server.ComfyCliError, match="invalid model"):
        _emit(model, "/tmp/flux.json")

    assert calls == []


def test_emit_validation_is_literally_partner_generates(monkeypatch):
    """Not just equivalent — the SAME function, so the two cannot diverge.

    The parametrized cases above would both keep passing if someone forked the
    model guard and then changed one copy. This asserts the shared call site.
    """
    seen: list[str] = []
    monkeypatch.setattr(params, "_validate_generate_model", seen.append)

    with pytest.raises(server.ComfyCliError, match="invalid out_path"):
        _emit("flux-pro", "")

    assert seen == ["flux-pro"]


@pytest.mark.parametrize("bad", ["", None])
def test_emit_requires_an_out_path(patched_run, bad):
    """No default destination exists for `--emit-workflow`; the flag IS the path."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    with pytest.raises(server.ComfyCliError, match="invalid out_path"):
        _emit("flux-pro", bad or "")

    assert calls == []


def test_emit_rejects_an_oversized_out_path(no_spawn):
    """An oversized `out_path` is refused before it can reach argv.

    It rides `--emit-workflow=<path>`, so its size lands in a single argv
    string, where the OS rejects the exec with an `OSError` (`E2BIG`)
    `_run_comfy_raw` never converts — its `try` wraps only `communicate()`, not
    the `Popen(...)` that raises.
    """
    oversized = "/tmp/" + "e" * argv._MAX_PATH_ARG_LEN + ".json"

    with pytest.raises(server.ComfyCliError, match="exceeds") as excinfo:
        _emit("flux-pro", oversized)

    # Length-not-value: the size check runs ahead of the two value guards,
    # whose echoes would name the value instead of its size.
    assert oversized not in str(excinfo.value)


def test_emit_allows_an_out_path_at_the_ceiling(patched_run):
    """The boundary value itself rides through into `--emit-workflow=`."""
    calls = patched_run(envelope(data=_EMIT_DATA))
    at_ceiling = "/tmp/" + "e" * (argv._MAX_PATH_ARG_LEN - len("/tmp/"))
    assert len(at_ceiling) == argv._MAX_PATH_ARG_LEN

    _emit("flux-pro", at_ceiling)

    assert calls[0]["cmd"][-1] == f"--emit-workflow={at_ceiling}"


def test_emit_rejects_an_option_like_out_path(patched_run):
    """A dash-leading destination is a caller mistake worth naming."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    with pytest.raises(server.ComfyCliError, match="invalid out_path"):
        _emit("flux-pro", "--out.json")

    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "flux\0pro", "out_path": "/tmp/f.json"}, "invalid model"),
        ({"model": "flux-pro", "out_path": "/tmp/f\0.json"}, "invalid out_path"),
        (
            {"model": "flux-pro", "out_path": "/tmp/f.json", "params": {"p": "a\0b"}},
            "invalid value for parameter",
        ),
    ],
)
def test_emit_rejects_embedded_nul(patched_run, kwargs, match):
    """A NUL cannot ride in argv; `subprocess` would raise a bare ValueError."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    with pytest.raises(server.ComfyCliError, match=match):
        _emit(**kwargs)

    assert calls == []


# --- no spend interlock -----------------------------------------------------


def test_emit_never_probes_the_spend_gate_or_asks_for_consent(patched_run, monkeypatch):
    """emit-workflow spends nothing, so neither half of the interlock may run.

    `_require_spend_gate` shells out to `comfy generate consent show`, and
    `_resolve_spend_consent` raises the elicitation prompt. Running either here
    would ask the user to approve a cost that does not exist — and the gate
    probe would make a free, offline call fail on a comfy-cli that lacks a verb
    it does not need.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("the spend interlock must not run for emit-workflow")

    monkeypatch.setattr(server, "_require_spend_gate", _boom)
    monkeypatch.setattr(server, "_resolve_spend_consent", _boom)
    monkeypatch.setattr(server, "_engine_auto_confirms", _boom)
    calls = patched_run(envelope(data=_EMIT_DATA))

    _emit("flux-pro", "/tmp/flux.json", params={"prompt": "a cat"})

    assert "--yes" not in calls[0]["cmd"]  # and no consent is forwarded either


def test_emit_takes_no_confirm_spend_argument():
    """The tool's public schema must not offer a spend knob it does not honor."""
    tool = asyncio.run(server.mcp.get_tool("emit_partner_workflow"))

    assert "confirm_spend" not in tool.parameters["properties"]
    assert set(tool.parameters["required"]) == {"model", "out_path"}


# --- comfy-cli must actually support --emit-workflow -----------------------
#
# `emit_partner_workflow`'s whole safety claim is that `--emit-workflow`
# returns before any partner-proxy call, so it deliberately skips the spend
# gate above. That is true only on a comfy-cli that recognises the flag —
# `comfy generate` is registered with Click's `ignore_unknown_options` /
# `allow_extra_args`, so on an older install `--emit-workflow` is instead
# forwarded as a MODEL PARAMETER and a real, spending proxy call runs. These
# lock in the capability probe that stands between this tool and that outcome.


def test_emit_refuses_when_comfy_cli_lacks_the_emit_workflow_flag(
    monkeypatch, patched_run
):
    """No `--emit-workflow` in `comfy generate --help` -> refuse, never run.

    This is the regression the ticket is about: on an incapable comfy-cli, the
    argv this tool builds would run a real, spending partner generation with no
    consent interlock — from a tool whose contract says it spends nothing and
    so raises no prompt. Refusing is what stands in for the missing prompt.
    """
    monkeypatch.setattr(server, "_emit_workflow_capability_probed", False)
    # An older comfy-cli's `generate --help` text never mentions the flag.
    calls = patched_run(
        "comfy generate — call ComfyUI partner nodes\n\n"
        "Usage:\n"
        "  comfy generate <model> [--<param> value]... [--download PATH] [--yes]\n"
    )

    with pytest.raises(server.ComfyCliError, match="does not recognise"):
        _emit("flux-pro", "/tmp/flux.json", params={"prompt": "a cat"})

    assert len(calls) == 1  # only the probe ran — never the spending call
    assert calls[0]["cmd"][4:] == ["generate", "--help"]
    assert server._emit_workflow_capability_probed is False  # not latched


def test_emit_capability_probe_uses_generate_help_not_a_real_emit_call(
    monkeypatch, patched_run
):
    """The probe itself must not be the spending call it exists to prevent.

    Probing with a real `comfy generate <model> --emit-workflow <path>` would,
    on the exact incapable install this guards against, BE an unguarded
    spending call — reproducing the hazard instead of catching it. The probe
    argv must carry no model and no `--emit-workflow`.
    """
    monkeypatch.setattr(server, "_emit_workflow_capability_probed", False)
    calls = patched_run("--emit-workflow")

    server._require_emit_workflow_capability()

    assert len(calls) == 1
    probe_argv = calls[0]["cmd"][4:]
    assert probe_argv == ["generate", "--help"]
    assert "flux-pro" not in probe_argv


def test_emit_capability_probe_failure_is_not_latched(monkeypatch, patched_run):
    """A probe that cannot even run fails CLOSED, and does not wedge the process.

    Mirrors `_require_spend_gate`'s posture: a transient spawn failure must not
    be latched as "no capability" forever, so a later, healthy call can still
    succeed.
    """
    monkeypatch.setattr(server, "_emit_workflow_capability_probed", False)
    patched_run(raises=OSError("no such file or directory: comfy"))

    with pytest.raises(server.ComfyCliError, match="could not confirm"):
        _emit("flux-pro", "/tmp/flux.json")

    assert server._emit_workflow_capability_probed is False


def test_emit_probes_the_capability_once_then_emits(monkeypatch, patched_run):
    """A CLI that HAS the flag is probed once; every later emit call skips it."""
    monkeypatch.setattr(server, "_emit_workflow_capability_probed", False)
    # One canned stdout that answers BOTH calls the fake cannot tell apart: a
    # help line carrying the flag (what the probe greps for), followed by the
    # envelope the real emit call parses — `_last_json_object` keeps the last
    # JSON line on stdout and ignores the leading prose ahead of it.
    calls = patched_run(
        '  comfy generate flux-pro --prompt "a fox" --emit-workflow flux.json\n'
        + json.dumps(envelope(data=_EMIT_DATA))
    )

    _emit("flux-pro", "/tmp/flux.json")
    _emit("flux-pro", "/tmp/flux.json")

    assert [c["cmd"][4:6] for c in calls] == [
        ["generate", "--help"],  # probed once...
        ["generate", "flux-pro"],
        ["generate", "flux-pro"],  # ...not again on the second call
    ]
    assert server._emit_workflow_capability_probed is True


# --- result contract (emit-workflow DOES emit an envelope) ------------------


def test_emit_returns_the_envelope_data(patched_run):
    """`{"out", "model", "nodes"}` comes back structured — no text scraping."""
    patched_run(envelope(data=_EMIT_DATA))

    assert _emit("flux-pro", "/tmp/flux.json") == _EMIT_DATA


def test_emit_does_not_synthesize_success_from_a_bare_exit_0(patched_run):
    """No `plain_ok` here: this verb emits an envelope, so a missing one is a bug.

    Falling back to "clean exit == success" would report a workflow written when
    none was, and `run_workflow` would then fail on a file that is not there.
    """
    patched_run(stdout="Wrote workflow: /tmp/flux.json", returncode=0)

    with pytest.raises(server.ComfyCliError, match="no JSON"):
        _emit("flux-pro", "/tmp/flux.json")


def test_unsupported_model_error_passes_through_with_its_model_list(patched_run):
    """comfy-cli's `emit_workflow_failed` reaches the agent VERBATIM.

    That message is self-documenting — it names every alias emit-workflow does
    support — so flattening it to a generic failure would cost the caller the
    one piece of information that lets it retarget without another round trip.
    """
    patched_run(envelope(ok=False, error=_UNSUPPORTED_ERROR), returncode=1)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _emit("flux-ultra", "/tmp/flux.json")

    text = str(excinfo.value)
    assert "emit_workflow_failed" in text  # the structured code
    assert "does not support model 'flux-ultra'" in text
    for alias in ("flux-2", "flux-pro", "kling-i2v", "nano-banana", "seedance"):
        assert alias in text  # the whole supported set, not a summary
    assert "check the model name" in text  # and comfy-cli's own hint
    assert excinfo.value.code == "emit_workflow_failed"


def test_emit_bounds_the_child(patched_run):
    """A wedged CLI cannot hold the request open indefinitely."""
    calls = patched_run(envelope(data=_EMIT_DATA))

    _emit("flux-pro", "/tmp/flux.json")

    assert calls[0]["timeout"] == server._EMIT_WORKFLOW_TIMEOUT


# --- discoverability --------------------------------------------------------


def test_instructions_route_local_partner_runs_to_the_emit_chain():
    """The handshake has to teach WHICH tool puts the local install in the path.

    Without this an agent asked to "use my local ComfyUI with a partner node"
    reaches for `partner_generate`, which runs entirely on partner hardware.
    """
    instructions = server.mcp.instructions

    assert "emit_partner_workflow" in instructions
    assert "run_workflow" in instructions
    assert "fetch_outputs" in instructions


def test_docstring_states_the_coverage_limit_and_the_fallback():
    """Narrow coverage must not read as "partner nodes are unsupported".

    Only five aliases map to a node; every other model still generates today
    through `partner_generate`, so the docstring has to name both the supported
    set and where to send everything else.
    """
    doc = server.emit_partner_workflow.__doc__

    for alias in ("flux-2", "flux-pro", "kling-i2v", "nano-banana", "seedance"):
        assert alias in doc
    assert "partner_generate" in doc  # the fallback for everything else
    assert "run_workflow" in doc and "fetch_outputs" in doc  # the intended chain


# --- QA 0.8.0: the emitted file carried no cost provenance -------------------
# 521 bytes, 2 nodes, no price, credit estimate, billing key, confirm_spend,
# warning or Note node. The file's whole purpose is to be handed to
# run_workflow, where billing happens — quite possibly by someone who never saw
# the tool response, since that is where every scrap of provenance lived.

_EMITTED = {
    "1": {
        "class_type": "Flux2ProImageNode",
        "_meta": {"title": "Flux2ProImageNode (flux-2)"},
        "inputs": {"prompt": "a red fox"},
    },
    "2": {
        "class_type": "SaveImage",
        "_meta": {"title": "save generated image"},
        "inputs": {"images": ["1", 0]},
    },
}


def test_stamped_provenance_marks_the_billing_node(tmp_path):
    """The graph says, on disk, that running it spends money."""
    out = tmp_path / "flux.json"
    out.write_text(json.dumps(_EMITTED))

    block = server._stamp_partner_provenance(str(out), "flux-2")

    assert block["spends_credits"] is True
    assert block["partner_model"] == "flux-2"
    assert "confirm_spend=True" in block["requires"]

    written = json.loads(out.read_text())
    stamp = written["1"]["_meta"]["comfy_mcp"]
    assert stamp["spends_credits"] is True
    assert "BILLS Comfy credits" in stamp["warning"]
    # The SaveImage node is not a billing node and is left alone.
    assert "comfy_mcp" not in written["2"]["_meta"]


def test_stamping_preserves_the_runnable_graph(tmp_path):
    """The stamp must not change what the graph DOES.

    `_meta` is frontend passthrough — ComfyUI reads `_meta.title` and ignores the
    rest — so class_types, inputs and wiring must come back byte-identical.
    Verified against a real ComfyUI too: `comfy validate` reports valid: true
    with zero errors and zero warnings, stamped or not.
    """
    out = tmp_path / "flux.json"
    out.write_text(json.dumps(_EMITTED))

    server._stamp_partner_provenance(str(out), "flux-2")

    written = json.loads(out.read_text())
    for nid, node in _EMITTED.items():
        assert written[nid]["class_type"] == node["class_type"]
        assert written[nid]["inputs"] == node["inputs"]
        assert written[nid]["_meta"]["title"] == node["_meta"]["title"]


def test_stamping_is_best_effort(tmp_path):
    """A file that cannot be read must not turn a successful emit into an error."""
    assert (
        server._stamp_partner_provenance(str(tmp_path / "missing.json"), "flux-2")
        is None
    )


def test_stamping_never_destroys_the_emitted_workflow(tmp_path, monkeypatch):
    """A failed stamp must leave the original graph intact.

    Opening `out_path` with "w" truncates BEFORE the write completes, so a
    failure part-way (full disk, a serialization error) would replace a
    perfectly good workflow with a stump — and the emit that produced it has
    already returned, so there is nothing to retry. The write goes to a sibling
    temp file and is swapped in with os.replace.
    """
    out = tmp_path / "flux.json"
    original = json.dumps(_EMITTED)
    out.write_text(original)

    def exploding_dump(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(server.json, "dump", exploding_dump)

    assert server._stamp_partner_provenance(str(out), "flux-2") is None
    # The caller's file is byte-identical, not truncated.
    assert out.read_text() == original
    # And no temp file is left behind.
    assert list(tmp_path.glob("*.tmp")) == []
