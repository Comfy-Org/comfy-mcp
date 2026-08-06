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

from comfy_mcp import server

# ``INSTRUCTIONS`` is hard-wrapped prose, so any phrase long enough to matter can
# straddle a newline. Match against a whitespace-collapsed copy: these tests are
# guarding that the POLICY is still stated, not where the line breaks fall.
FLAT = " ".join(server.INSTRUCTIONS.split())

_ROUTING_HEADER = "Routing — check the machine before running local diffusion."
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

    ``startswith`` would be tautological — the fixture slices from the header's
    own index — so assert what the slice does NOT guarantee: that the block has
    real substance between its header and terminator, and still ends on the
    last bullet rather than having been hollowed out to a stub.
    """
    assert len(routing) > 500, f"routing block hollowed out to {len(routing)} chars"
    assert routing.count("- ") >= 6, "routing block lost bullets"
    assert routing.rstrip().endswith("current templates track current models.")


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
    divisor yields GiB and drivers report under the advertised size — a consumer
    24 GB card reads 23.99, and an ECC/reserving datacenter card (A10, L4) about
    22.3 — so any exact-rounding recipe still drops a band.

    Rounding up is bounded to that driver overhead, though. On a MIG/vGPU
    partition ``gpu.model`` still names the whole card while ``vram_bytes`` is
    the slice actually available, so mapping the figure up to the model's
    nominal size would route a 6 GB slice of an A100 into the ">= 24 GB" band
    and OOM the run. Both halves have to be stated or one failure mode replaces
    the other.

    The ~10% cutoff is asserted because the two halves are otherwise a
    contradiction rather than a rule: "small shortfall" and "far below nominal"
    have no boundary between them, and a 20 GiB vGPU slice of a 24 GB card
    (~17% short) reads as either one. A number puts the driver-overhead cases
    (23.99 and ~22.3 of 24, i.e. ~0% and ~7%) inside and that slice outside.

    Anchors are backticked (```ram_bytes```` is NOT a substring of
    ```gpu.vram_bytes````, where the name is preceded by ``v``) so each field
    name can fail independently.
    """
    assert "BYTES" in routing
    assert "`ram_bytes`" in routing and "`gpu.vram_bytes`" in routing
    assert "1073741824" in routing  # the bytes -> GiB divisor, stated explicitly
    assert "SMALL shortfall" in routing  # rounds up only within driver overhead
    assert "within ~10% of a nominal size" in routing  # ...and says how small
    assert "MIG/vGPU PARTITION" in routing  # ...but never on a partitioned card


def test_routing_reads_the_apple_shape_as_identification_not_a_figure(routing):
    """On Apple Silicon the null ``vram_bytes`` names the machine, not its size.

    Since the Apple verdict went unconditional there is no threshold left for a
    memory figure to be compared against, so the block must stop telling the
    agent to substitute one — an agent that still reads ``ram_bytes`` as a
    routing number has no row to route it on and will invent a band.

    The APPLE-ONLY scoping still has to survive that rewrite, in the other
    direction: a null ``vram_bytes`` on an Intel/AMD integrated GPU is UNKNOWN
    (step 3), and a block that generalised the short-circuit would hand those
    machines the Apple verdict without ever asking.
    """
    assert "`gpu.unified_memory` is true" in routing
    assert "APPLE-ONLY" in routing
    assert "IDENTIFIES" in routing  # ...the machine, rather than sizing it
    assert "changes no verdict" in routing


def test_routing_defers_to_comfy_target_instead_of_local_hardware(routing):
    """``hardware`` describes THIS host, which may not be where the job runs.

    ``COMFYUI_URL`` / ``COMFYUI_HOST`` point the run tools at another machine
    (surfaced as ``comfy_target``). Routing off local hardware would then reject
    a capable remote GPU, or pile work onto a weaker one.

    The exemption still has to be NARROW, though: a loopback host is this same
    machine, and a malformed config yields an error-shaped block that resolves no
    remote at all. Treating any ``comfy_target`` as "runs elsewhere" would switch
    routing off on both.

    Loopback is stated as a RANGE, not a pair of literals: ``_comfy_target``
    accepts a bracketed IPv6 host (``COMFYUI_URL=http://[::1]:8188``, see
    ``_strip_brackets``) and the rest of ``127.0.0.0/8`` is equally this
    machine. And ``comfy_target`` is not the only remoting signal —
    ``COMFY_LOCAL_URL`` repoints comfy-cli's resolved ``server`` URL without
    producing the block at all.
    """
    assert "comfy_target" in routing
    assert "ERROR-shaped" in routing
    assert "127.0.0.0/8" in routing and "::1" in routing
    assert "COMFY_LOCAL_URL" in routing


def test_routing_names_every_submitting_tool_as_diverted(routing):
    """All three job-SUBMITTING tools follow ``comfy_target`` — the block must say so.

    ``_TARGET_AWARE_SUBCOMMANDS`` covers ``run``, ``run-template``, ``jobs`` and
    ``upload`` (the last stages inputs rather than submitting a job, so it is not
    in this guard's list),
    so ``run_workflow``, ``generate_image`` and ``run_template`` all submit to a
    configured remote. The block used to name the last two as *still local*,
    which is the failure this guards against in both directions: an agent that
    believes ``generate_image`` runs here will apply this machine's VRAM
    thresholds to a job that lands elsewhere, and — worse — will read a
    ``prompt_not_found`` from ``wait_for_job`` as a broken queue rather than as
    the two calls having been pointed at different servers.

    Scoped to the diversion SENTENCE, not the whole block: every one of these
    names appears elsewhere in the routing prose, so a whole-block ``in`` check
    would keep passing after the sentence had been rewritten to exclude them.
    """
    assert "diverts" in routing, "the routing block no longer states a diversion"
    sentence = routing.split("diverts", 1)[1].split(". ", 1)[0]
    for tool in ("run_workflow", "generate_image", "run_template"):
        assert tool in sentence, f"{tool} is target-aware but not named as diverted"


def test_routing_asks_when_it_cannot_place_the_target_host(routing):
    """Whether a host IS this machine is not answerable from tool output alone.

    STEP 1 keys the local/remote split on the ``host`` being neither loopback
    nor "this host's own name or address", but nothing the server returns
    carries the local hostname or interface addresses — ``hardware`` reports
    only ``os`` / ``arch`` / memory — and STEP 3 forbids shelling out to find
    them. Left there, a ``host`` naming this same machine by hostname or LAN IP
    reads as a genuine remote and switches routing off; the converse also bites,
    since a loopback host can be a tunnel to a remote GPU. The block has to send
    the unresolvable case to the user, the same answer STEP 3 gives.
    """
    assert "ASK the user which machine it is" in routing


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

    Anchored on ``ROCm/XPU`` rather than ``AMD/Intel`` because that is the
    load-bearing half: such a card only routes on the VRAM bands when the
    install is a ROCm/XPU build, and a row degraded to naming the vendors
    without stating the build requirement would still satisfy an ``AMD/Intel``
    anchor.
    """
    assert "ROCm/XPU" in routing


def test_routing_sends_every_apple_silicon_mac_to_cloud(routing):
    """Apple Silicon routes away from local diffusion at ANY memory size.

    This supersedes the old ">= 32 GB, images OK / under 32 GB, treat as no-GPU"
    split (product decision, 2026-08-06): current image and video models do not
    run at a workable speed on the Apple GPU at any unified-memory size, and the
    older models that do fit are not worth steering anyone toward.

    The retired threshold is asserted GONE as well as the new rule asserted
    present, because a half-applied edit is the dangerous outcome — leaving the
    band in place alongside the unconditional row hands an M4 Max two
    contradictory verdicts, which is worse than either policy alone.
    """
    assert "Apple Silicon: do NOT run local diffusion" in routing
    assert "at ANY memory size" in routing
    assert ">= 32 GB" not in routing
    assert "under 32 GB" not in routing


def test_routing_can_act_on_the_answer_it_asks_for(routing):
    """STEP 3 must not ask a question STEP 4 has no row to answer with.

    STEP 3 sends non-Apple unified-memory machines (Jetson/Grace, Strix Halo) to
    "ask the user ... and route on their answer", but STEP 4's rows are a VRAM
    table for dedicated cards plus a unified-memory row declared APPLE-ONLY in
    STEP 2. Without a row for the answer the procedure interrogates the user and
    then has nowhere to go — the dead-end STEP 5 exists to prevent, arrived at
    from the opposite direction.
    """
    assert "A figure the USER gave you" in routing
    assert "have no row of their own" in routing


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
    isolates partner-run graphs, and the caveat has to justify BOTH axes without
    pinning the failure on either one: ``tag`` and ``type`` are separate
    exact-match forwards to ``--tag`` / ``--type``, so ``tag`` alone does not
    constrain the output type and ``type`` alone does not constrain where the
    model runs. Either way the compact rows omit ``tags`` (see
    ``test_templates.py``), so the caller cannot tell a local template from an
    ``API`` one in the results.

    The rule is scoped to the APPLE GPU, not to Macs generally — an Intel Mac
    with a discrete card follows the discrete-GPU row, and "any Mac" handed that
    machine two contradictory verdicts.
    """
    assert 'search_templates(tag="API", type="video")' in routing
    assert "neither alone isolates partner-run video" in routing
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

    The condition is keyed on non-Apple rather than on non-unified-memory: a
    non-Apple UNIFIED part (Jetson/Grace, a Strix Halo APU) also reports a null
    ``vram_bytes``, and the ``ram_bytes`` substitution above is Apple-only, so
    keying on unified-memory left that machine matching neither branch — with
    64-128 GB of usable memory and a "do NOT run local diffusion" verdict.
    The Apple path needs no mirror case: since its verdict went unconditional it
    depends on no figure at all, so a missing ``ram_bytes`` there is not a
    question to ask — which is why the block says Apple never reaches this step.

    Zero is asserted alongside null on BOTH paths. A CLI or driver that reports
    an unsizable card as ``vram_bytes: 0`` would otherwise skip this step
    entirely and land on step 4's "under 8 GB, do NOT run local diffusion" —
    stranding the very machine this step exists to rescue, and only on the
    non-Apple path, since the Apple path already said "missing or zero".

    Because UNKNOWN swallows every shape that used to read as "no GPU", the
    no-GPU verdict has to say what a CONFIRMED absence looks like, or step 4's
    branch becomes unreachable. It is asserted to be the USER's answer
    specifically: a data-shaped absence ("``gpu`` names no device") contradicts
    this step, which already classifies a null-or-absent ``gpu`` as UNKNOWN, and
    the block's header forbids step 4 overriding it — so that phrasing left the
    branch dead and every CPU-only machine getting interrogated about its GPU.
    """
    assert "is UNKNOWN, NOT" in routing
    assert "`vram_bytes` null or zero on ANY non-Apple GPU" in routing
    assert "Jetson/Grace" in routing and "Strix Halo" in routing
    assert "never reaches\n  this step for a memory figure" in server.INSTRUCTIONS
    # the confirmed absence is the user's answer, not a payload shape
    assert "CONFIRMED absence of a GPU is the USER telling you" in routing
    assert "no `hardware` payload states it" in routing


def test_routing_asks_the_user_and_names_no_shell_probe(routing):
    """The missing-figure answer is to ASK — the block must not hand out probes.

    An earlier draft named ``system_profiler`` / ``sysctl`` / ``nvidia-smi`` as a
    fallback. They could not actually supply what the thresholds key on (no one
    command yields both numbers, and ``nvidia-smi`` reports nothing for
    AMD/Intel), and they steer the client agent into shelling out on a path this
    server can neither bound nor audit — a ``PATH``-shadowed ``nvidia-smi`` runs
    whatever it likes with the agent's privileges, against advisory prose as the
    only mitigation. Asking the user is both more reliable and free of that
    surface, so the probe commands are gone and this test keeps them gone.
    """
    assert "Ask the user what GPU" in routing
    assert "do not shell out to probe" in routing
    for probe in ("system_profiler", "sysctl", "nvidia-smi"):
        assert probe not in routing, (
            f"routing block re-introduced a shell probe: {probe}"
        )


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

    ``patched_run`` replays one canned stdout for every invocation and
    ``server_info`` makes two (``comfy env``, then the ``outdated`` freshness
    probe), so equality alone cannot say WHICH call the block came from. Pin it
    by asserting the first invocation is the ``env`` one — that is the call the
    canned envelope is standing in for, and the only one this passthrough claim
    is about.
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
    calls = patched_run(
        stdout=envelope(
            ok=True,
            data={"server": {"running": False}, "hardware": hardware},
        )
    )
    result = server.server_info()
    assert result["hardware"] == hardware
    # `hardware` rides the FIRST call, `comfy env` — not a probe of our own.
    assert "env" in calls[0]["cmd"]
