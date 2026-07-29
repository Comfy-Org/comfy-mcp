"""Guards on the machine-aware routing policy carried in ``INSTRUCTIONS``.

The routing block is the only place an agent learns *whether* this machine
should run local diffusion at all — it rides the client handshake and nothing
else re-states it at call time. It is pure prose, so no functional test would
notice it disappearing in a future rewrite of the constant; these assertions are
that tripwire. They are deliberately keyed on the load-bearing facts (the signal
name, the units, the thresholds, the redirect, the LOCAL-only guarantee) rather
than on whole sentences, so wording can be edited without a test churn.

Almost every assertion runs against the ROUTING BLOCK sliced out of
``INSTRUCTIONS``, not against the whole constant. That scoping is what makes
them tripwires at all: names like ``search_templates``, ``search_models`` and
``emit_partner_workflow`` are also used by pre-existing bullets elsewhere in
``INSTRUCTIONS``, so a whole-constant ``in`` check would keep passing after the
routing block had been deleted outright. Slicing means a deleted block fails
``test_instructions_carry_a_routing_block`` first, and every content assertion
after it.
"""

from __future__ import annotations

import pytest
from conftest import envelope

from comfy_local_mcp import server

# ``INSTRUCTIONS`` is hard-wrapped prose, so any phrase long enough to matter can
# straddle a newline. Match against a whitespace-collapsed copy: these tests are
# guarding that the POLICY is still stated, not where the line breaks fall.
FLAT = " ".join(server.INSTRUCTIONS.split())

_ROUTING_HEADER = "Routing — check the machine before running local diffusion:"
_ROUTING_END = "Everything targets the LOCAL server only"


@pytest.fixture(scope="module")
def routing() -> str:
    """The routing block alone, whitespace-collapsed. See the module docstring."""
    start = FLAT.find(_ROUTING_HEADER)
    assert start != -1, f"routing block header is gone: {_ROUTING_HEADER!r}"
    end = FLAT.find(_ROUTING_END, start + 1)
    assert end > start, f"routing block terminator is gone: {_ROUTING_END!r}"
    return FLAT[start:end]


def test_the_server_hands_these_instructions_to_the_client():
    """The constant is only policy if the handshake actually carries it.

    ``INSTRUCTIONS`` is wired into the server once, at construction, and every
    assertion below is worthless if that wiring is ever dropped — a client would
    connect to a server that advertises no guidance at all and none of the
    content tests would notice.
    """
    assert server.mcp.instructions == server.INSTRUCTIONS


def test_instructions_carry_a_routing_block(routing):
    """The block every other assertion in this file is scoped to still exists.

    This is the tripwire the content tests hang off: delete the block and this
    fails outright, rather than the content checks quietly passing on phrases
    that also appear in unrelated bullets.
    """
    assert routing.startswith(_ROUTING_HEADER)


def test_routing_names_the_hardware_block_as_its_signal(routing):
    """``hardware`` is the ``server_info`` key the whole policy reads.

    Drop the name and the guidance becomes unactionable — the agent is told to
    check the machine with no field to check.
    """
    assert "`hardware` block" in routing


def test_routing_states_the_units_and_rounds_to_the_nominal_band(routing):
    """The block's thresholds are GB; the payload's fields are bytes.

    comfy-cli reports ``ram_bytes`` / ``gpu.vram_bytes`` as raw byte counts, so
    without the units stated a literal reading compares a ten-digit number
    against "8 GB" and routes wrong. The rounding half matters just as much: the
    divisor yields GiB and drivers report just under nominal capacity, so a
    nominal 24 GB card reads 23.99 and would silently drop into the band below —
    the exact misclassification the threshold rows exist to make unambiguous.

    Anchors are backticked (```ram_bytes```` is NOT a substring of
    ```gpu.vram_bytes````, where the name is preceded by ``v``) so each field
    name can fail independently.
    """
    assert "BYTES" in routing
    assert "`ram_bytes`" in routing and "`gpu.vram_bytes`" in routing
    assert "1073741824" in routing  # the bytes -> GiB divisor, stated explicitly
    assert "ROUND TO THE NEAREST WHOLE GB" in routing


def test_routing_scopes_the_unified_memory_substitution_to_apple(routing):
    """A null ``vram_bytes`` means "read ``ram_bytes``" only on Apple Silicon.

    Stated generically it would hand a 32 GB Intel/AMD integrated-GPU laptop the
    Apple ">= 32 GB, images OK" verdict, when the only row keyed on
    ``ram_bytes`` is the Apple one and such a machine belongs in partner/cloud.
    """
    assert "`gpu.unified_memory` is true" in routing
    assert "APPLE-ONLY" in routing


def test_routing_defers_to_comfy_target_instead_of_local_hardware(routing):
    """``hardware`` describes THIS host, which may not be where the job runs.

    ``COMFYUI_URL`` / ``COMFYUI_HOST`` point the run tools at another machine
    (surfaced as ``comfy_target``). Routing off local hardware would then reject
    a capable remote GPU, or pile work onto a weaker one.
    """
    assert "comfy_target" in routing


def test_routing_thresholds_do_not_overlap_at_24gb(routing):
    """A 24 GB card is a common capacity and must match exactly one row.

    An inclusive "8-24 GB" band alongside ">= 24 GB" gives that card two
    contradictory verdicts ("good default" vs "video slow or infeasible").
    """
    assert ">= 24 GB" in routing
    assert "8 GB to under 24 GB" in routing
    assert "8-24 GB" not in routing


def test_routing_covers_non_nvidia_discrete_gpus(routing):
    """An AMD/Intel card with real VRAM must not fall through every row.

    ``hardware.gpu.vendor`` reports these, so a block scoped only to NVIDIA
    leaves a machine that plainly has a usable GPU with no verdict at all.

    Anchored on ``ROCm/XPU`` rather than ``AMD/Intel``: the latter also appears
    in the ``nvidia-smi`` caveat further down, so it would keep passing after
    this row's non-NVIDIA coverage was deleted.
    """
    assert "ROCm/XPU" in routing


def test_routing_covers_apple_silicon_below_the_32gb_line(routing):
    """A Mac under 32 GB is not "no GPU" and needs its own stated verdict.

    README ships this row; the handshake has to carry the same guidance, or the
    documentation describes a policy the agent was never given.
    """
    assert ">= 32 GB" in routing
    assert "under 32 GB" in routing


def test_routing_redirects_rather_than_dead_ending_a_weak_machine(routing):
    """A machine that should not run local diffusion gets a PATH, not a refusal.

    The whole point of the steer is to hand the user somewhere to go — partner
    nodes or the Comfy Cloud MCP. An edit that keeps the "do NOT run local
    diffusion" half and loses the alternatives turns guidance into a dead end.
    """
    lowered = routing.lower()
    assert "partner nodes" in lowered
    assert "comfy cloud mcp" in lowered


def test_routing_keeps_video_reachable_on_a_mac_via_a_filter_that_works(routing):
    """The no-local-video-on-a-Mac line must stay scoped to the Mac's own GPU.

    Read as a blanket "no video on a Mac" it would deny a capability this server
    really has. The named escape hatch also has to be a filter that actually
    isolates partner-run graphs: ``tag`` is a single exact-match forward to
    ``--tag``, and the compact rows omit ``tags`` (see
    ``test_templates.py``), so ``tag="Video"`` alone returns LOCAL video
    templates the caller cannot tell apart. ``tag="API"`` plus ``type="video"``
    narrows on both axes.

    The rule is scoped to the APPLE GPU, not to Macs generally — an Intel Mac
    with a discrete card follows the discrete-GPU row, and "any Mac" handed that
    machine two contradictory verdicts.
    """
    assert 'search_templates(tag="API", type="video")' in routing
    assert 'tag="Video"' in routing  # the value the caveat is actually about
    assert "emit_partner_workflow" in routing
    assert "APPLE-GPU rule, not" in routing


def test_routing_steers_model_choice_to_discovery_not_a_hardcoded_default(routing):
    """No model registry lives in this repo — the agent searches for one.

    Naming a "classic default" in the constant would rot as the gallery moves;
    the evergreen instruction is to look it up.
    """
    assert "search_templates" in routing and "search_models" in routing
    assert "classic default" in routing  # phrase unique to this bullet


def test_routing_asks_rather_than_reading_an_unknown_gpu_as_no_gpu(routing):
    """Present-but-incomplete ``hardware`` must route like absent, not like "no GPU".

    A discrete card comfy-cli cannot size reports ``vram_bytes: null`` with
    ``unified_memory`` false or absent, and ``gpu`` itself can be null/missing.
    Scoping the escape hatch to ``hardware`` being *absent* left those machines
    falling through to "no GPU at all -> do NOT run local diffusion", stranding a
    machine that has a perfectly usable GPU.
    """
    assert "present but missing the figure" in routing
    assert 'never read an UNKNOWN as "no GPU"' in routing


def test_routing_prefers_asking_over_an_unreliable_shell_probe(routing):
    """``hardware`` is new in comfy-cli, so the absent case needs an answer.

    It has to be an answer that WORKS: no single probe yields both numbers —
    ``system_profiler SPDisplaysDataType`` names the GPU but not unified-memory
    size (that is ``sysctl -n hw.memsize``), and ``nvidia-smi`` gives VRAM but no
    system RAM and nothing at all for AMD/Intel. These also run outside the
    audited ``_run_comfy`` path, so the block leads with asking the user and
    bounds any probe it does mention.
    """
    assert "ASK the user" in routing
    assert "sysctl -n hw.memsize" in routing  # the RAM figure system_profiler lacks
    assert "system_profiler" in routing and "nvidia-smi" in routing
    assert "short-lived" in routing  # no unbounded shell-out


def test_instructions_retain_the_local_only_closing_guarantee():
    """The routing block adds a cloud/partner steer; it must not read as cloud ACCESS.

    The closing guarantee is what keeps "point them at Comfy Cloud" from being
    mistaken for "this server can run it there". It sits just past the routing
    block, so this one is scoped to the whole constant on purpose.
    """
    assert (
        "Everything targets the LOCAL server only — there is no cloud access here."
        in FLAT
    )
    assert "this server cannot run cloud jobs itself" in FLAT


def test_server_info_docstring_points_at_the_routing_guidance():
    """The tool a routing agent calls first must mention the block and the policy.

    ``server_info`` is a bare ``comfy env`` passthrough, so its docstring is the
    only per-tool place the ``hardware`` key is documented — and it must document
    the key as CONDITIONAL, since an older comfy-cli omits it and a caller who
    read "the result includes a hardware block" literally would hit a KeyError.
    """
    doc = " ".join((server.server_info.__doc__ or "").split())
    assert "hardware" in doc
    assert "routing" in doc
    assert "when the installed comfy-cli reports one" in doc


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
