"""Guards on the machine-aware routing policy carried in ``INSTRUCTIONS``.

The routing block is the only place an agent learns *whether* this machine
should run local diffusion at all — it rides the client handshake and nothing
else re-states it at call time. It is pure prose, so no functional test would
notice it disappearing in a future rewrite of the constant; these assertions are
that tripwire. They are deliberately keyed on the load-bearing facts (the signal
name, the VRAM threshold, the redirect, the LOCAL-only guarantee) rather than on
whole sentences, so wording can be edited without a test churn.
"""

from __future__ import annotations

from conftest import envelope

from comfy_local_mcp import server

# ``INSTRUCTIONS`` is hard-wrapped prose, so any phrase long enough to matter can
# straddle a newline. Match against a whitespace-collapsed copy: these tests are
# guarding that the POLICY is still stated, not where the line breaks fall.
FLAT = " ".join(server.INSTRUCTIONS.split())


def test_the_server_hands_these_instructions_to_the_client():
    """The constant is only policy if the handshake actually carries it.

    ``INSTRUCTIONS`` is wired into the server once, at construction, and every
    assertion below is worthless if that wiring is ever dropped — a client would
    connect to a server that advertises no guidance at all and none of the
    content tests would notice.
    """
    assert server.mcp.instructions == server.INSTRUCTIONS


def test_instructions_name_the_hardware_block_as_the_routing_signal():
    """``hardware`` is the ``server_info`` key the whole policy reads.

    Drop the name and the guidance becomes unactionable — the agent is told to
    check the machine with no field to check.
    """
    assert "hardware" in FLAT.lower()


def test_instructions_carry_the_24gb_local_default_threshold():
    """The >= 24 GB VRAM line is the "local generation is a good default" cutoff.

    Without a number the routing collapses into taste; this pins the one the
    policy was written around.
    """
    assert "24 GB" in FLAT


def test_instructions_redirect_rather_than_dead_end_on_a_weak_machine():
    """A machine that should not run local diffusion gets a PATH, not a refusal.

    The whole point of the steer is to hand the user somewhere to go — partner
    nodes or the Comfy Cloud MCP. An edit that keeps the "do NOT run local
    diffusion" half and loses the alternatives turns guidance into a dead end.
    """
    lowered = FLAT.lower()
    assert "partner nodes" in lowered
    assert "comfy cloud mcp" in lowered


def test_instructions_keep_video_reachable_on_a_mac_via_partner_infrastructure():
    """The no-local-video-on-a-Mac line must stay scoped to the Mac's own GPU.

    Read as a blanket "no video on a Mac" it would deny a capability this server
    really has: the gallery's ``API``-tagged video templates and
    ``emit_partner_workflow`` put the model on partner infrastructure and run
    fine on any machine.
    """
    assert 'search_templates(tag="Video")' in FLAT
    assert "emit_partner_workflow" in FLAT


def test_instructions_steer_model_choice_to_discovery_not_a_hardcoded_default():
    """No model registry lives in this repo — the agent searches for one.

    Naming a "classic default" in the constant would rot as the gallery moves;
    the evergreen instruction is to look it up.
    """
    assert "search_templates" in FLAT
    assert "search_models" in FLAT


def test_instructions_retain_the_local_only_closing_guarantee():
    """The routing block adds a cloud/partner steer; it must not read as cloud ACCESS.

    The closing guarantee is what keeps "point them at Comfy Cloud" from being
    mistaken for "this server can run it there".
    """
    assert (
        "Everything targets the LOCAL server only — there is no cloud access here."
        in FLAT
    )
    assert "this server cannot run cloud jobs itself" in FLAT


def test_instructions_offer_a_probe_fallback_for_a_comfy_cli_without_hardware():
    """``hardware`` is new in comfy-cli, so the absent case needs an answer.

    Older comfy-cli releases (1.13.0 included) emit no ``hardware`` key at all;
    without the fallback an agent would read the missing block as "unknown" and
    either stall or route blind.
    """
    assert "system_profiler" in FLAT
    assert "nvidia-smi" in FLAT


def test_server_info_docstring_points_at_the_routing_guidance():
    """The tool a routing agent calls first must mention the block and the policy.

    ``server_info`` is a bare ``comfy env`` passthrough, so its docstring is the
    only per-tool place the ``hardware`` key is documented.
    """
    doc = server.server_info.__doc__ or ""
    assert "hardware" in doc
    assert "routing" in doc


def test_server_info_adds_no_hardware_parsing(monkeypatch, patched_run):
    """``hardware`` passes through untouched — this repo derives nothing from it.

    The thin-wrapper guardrail (AGENTS.md): the routing policy is text, not code.
    A future edit that starts branching on VRAM here would breach it, and this
    is the test that fails when it does.
    """
    monkeypatch.setattr(server, "MIN_COMFY_CLI_VERSION", None)
    monkeypatch.setattr(server, "_detect_comfy_cli_version", lambda: "1.13.0")
    hardware = {
        "os": "Darwin",
        "arch": "arm64",
        "ram_bytes": 68719476736,
        "gpu": {
            "vendor": "Apple",
            "model": "Apple M4 Max",
            "vram_bytes": None,
            "unified_memory": True,
        },
    }
    patched_run(
        stdout=envelope(
            ok=True,
            data={"server": {"running": False}, "hardware": hardware},
        )
    )
    result = server.server_info()
    assert result["hardware"] == hardware
