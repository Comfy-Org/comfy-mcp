"""The test doubles for MCP ``Context``/``ServerSession`` must match the real SDK.

Every consent and progress test in this suite drives a hand-rolled fake rather
than a live MCP session, so those fakes ARE the contract those tests check
against. That makes them a blind spot precisely where it costs the most: the
production call sites pass by keyword (``ctx.elicit(message=..., schema=...)``)
inside broad ``except`` handlers, so an SDK signature change lands as a
``TypeError`` that surfaces as "could not confirm the credit spend with the
user" on every paid call — or, on the progress path, as silently dropped
notifications — while the fakes keep the suite green.

These tests close that loop by asserting each fake against the REAL signature
imported from the installed SDK. They are cheap, and they are the reason a
future major bump fails loudly here instead of quietly in production.

Each consent gate keeps its OWN copy of those fakes on purpose — the copies are
documented as deliberate in the files that hold them, so that a change to one
tool's prompt cannot silently retune another tool's tests. That decision stands;
what it costs is that the registries below have to name every copy, because a
copy nobody registered is a copy that can drift. :func:`_discover_fakes` is the
backstop for exactly that: it walks the test package and fails when a
module-level double defines one of these methods without being listed.
"""

import ast
import asyncio
import inspect
import pathlib

import conftest
import pytest
import test_install_node
import test_network_exposure
import test_partner_generate
import test_run_template
import test_run_workflow_spend
import test_switch_version
import test_untracked_kill
import test_update_consent
from mcp import types
from mcp.client import Client
from mcp.server.mcpserver import Context
from mcp.server.session import ServerSession

from comfy_mcp import server

_TESTS_DIR = pathlib.Path(__file__).parent


def _label(fake: type) -> str:
    """``module.QualName`` for a registered fake, so a failure names the file.

    DERIVED from the class rather than typed beside it: a hand-written label can
    be paired with the wrong object, and then the registry reads right while the
    signature test checks some other fake twice and the real one goes unverified
    — exactly the drift this module exists to catch.
    """
    return f"{fake.__module__}.{fake.__qualname__}"


def _params(func) -> list[tuple[str, bool]]:
    """``(name, has_default)`` per bindable parameter, ``self`` dropped.

    Annotations are deliberately ignored: what the call sites depend on is the
    NAME (they pass by keyword) and whether an argument may be omitted, not the
    type spelling — which the SDK is free to restate without breaking anyone.
    """
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return [
        (name, param.default is not inspect.Parameter.empty)
        for name, param in inspect.signature(func).parameters.items()
        if name != "self" and param.kind in kinds
    ]


# Every fake standing in for the real `Context.elicit`. One per consent gate: the
# three spend gates, the version switch, `update_comfyui`'s node-pack gate, the
# network-exposure gate on the launch pair, and `restart_comfyui`'s
# untracked-server kill. Each entry is the class itself —
# `_label` says where it lives.
_ELICIT_FAKES = [
    test_install_node._FakeCtx,
    test_network_exposure._FakeCtx,
    test_partner_generate._FakeCtx,
    test_run_template._FakeCtx,
    test_run_workflow_spend._FakeCtx,
    test_switch_version._FakeCtx,
    test_untracked_kill._FakeCtx,
    test_update_consent._FakeCtx,
]

# Only the doubles that stand in for a context which also carries progress: the
# two STREAMING run verbs hand the same object to the prompt and to the run. The
# `switch_version` / `update_consent` / `install_node` / `network_exposure` /
# `untracked_kill` fakes have no `report_progress` because none of those tools
# stream, so
# registering them here would assert a method their production path never calls.
_PROGRESS_FAKES = [
    conftest._RecordingCtx,
    test_run_template._FakeCtx,
    test_run_workflow_spend._FakeCtx,
]

_SESSION_FAKES = [
    test_install_node._FakeSession,
    test_network_exposure._FakeSession,
    test_partner_generate._FakeSession,
    test_run_template._FakeSession,
    test_run_workflow_spend._FakeSession,
    test_switch_version._FakeSession,
    test_untracked_kill._FakeSession,
    test_update_consent._FakeSession,
]

#: Which registry above owns each SDK method a double may stand in for. Drives
#: the completeness check below, so adding a registry means adding an entry here.
_REGISTRY_BY_METHOD = {
    "elicit": ("_ELICIT_FAKES", _ELICIT_FAKES),
    "report_progress": ("_PROGRESS_FAKES", _PROGRESS_FAKES),
    "check_client_capability": ("_SESSION_FAKES", _SESSION_FAKES),
}


def _defines(node: ast.ClassDef, method: str) -> bool:
    """Does this class body bind ``method`` ITSELF, rather than inherit it?

    Mirrors the old ``method in vars(cls)``: a subclass that merely inherits a
    registered fake's method is not a second copy and needs no second entry.
    An ASSIGNED binding (``elicit = _raiser``) counts too, for the same reason
    ``vars()`` counted it — it is a stand-in the call sites reach all the same.
    """
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name == method:
                return True
        elif isinstance(stmt, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == method for t in stmt.targets):
                return True
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == method:
                return True
    return False


def _discover_fakes(method: str) -> set[str]:
    """Labels of every module-level class under ``tests/`` that defines ``method``.

    Read from the SOURCE with :mod:`ast` rather than by importing each file. The
    walk has to be recursive — ``tests/e2e/`` already exists, and a double added
    there must not slip past — but a nested directory is only importable once
    pytest has collected something in it, so an import-based walk would depend
    on which files the run happened to select. Parsing sees every file the same
    way whether the suite runs whole or one file at a time, and imports nothing
    that pytest would not have imported anyway.

    Module-level classes only, and named by the class rather than by whatever
    binds it, so a module-level alias of a registered fake is not a second copy
    demanding a redundant entry. The one-off doubles defined INSIDE a test
    function stay out of scope: no registry could name them (another module
    cannot import a function-local class), and they are explode-on-call
    stand-ins whose own test asserts the failure they produce — so drift there
    surfaces as that test failing, not as a silently green one.
    """
    found: set[str] = set()
    for path in sorted(_TESTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _defines(node, method):
                # `path.stem` is how pytest imports these — `tests/` has no
                # `__init__.py`, so each file's `__module__` is its bare stem.
                found.add(f"{path.stem}.{node.name}")
    return found


@pytest.mark.parametrize("fake", _ELICIT_FAKES, ids=_label)
def test_fake_elicit_matches_the_real_context_signature(fake):
    """A renamed ``elicit`` keyword must fail HERE, not as a spend-path TypeError."""
    assert _params(fake.elicit) == _params(Context.elicit), _label(fake)
    assert inspect.iscoroutinefunction(fake.elicit) == inspect.iscoroutinefunction(
        Context.elicit
    ), _label(fake)


@pytest.mark.parametrize("fake", _PROGRESS_FAKES, ids=_label)
def test_fake_report_progress_matches_the_real_context_signature(fake):
    """The production call only debug-logs on failure, so drift is silent there."""
    assert _params(fake.report_progress) == _params(Context.report_progress), _label(
        fake
    )
    assert inspect.iscoroutinefunction(
        fake.report_progress
    ) == inspect.iscoroutinefunction(Context.report_progress), _label(fake)


@pytest.mark.parametrize("fake", _SESSION_FAKES, ids=_label)
def test_fake_check_client_capability_matches_the_real_session_signature(fake):
    """The capability probe decides whether a human is asked before money moves."""
    real = ServerSession.check_client_capability
    assert _params(fake.check_client_capability) == _params(real), _label(fake)
    assert inspect.iscoroutinefunction(
        fake.check_client_capability
    ) == inspect.iscoroutinefunction(real), _label(fake)


@pytest.mark.parametrize("method", sorted(_REGISTRY_BY_METHOD))
def test_every_module_level_fake_is_registered(method):
    """Every copy of a consent fake must be CHECKED, not merely written.

    The registries above are hand-written, and hand-written lists rot: most of
    the consent fakes that existed when this test was added had gone unlisted
    for as long as they existed, so they could have drifted from the SDK while
    their own tests stayed green. This asserts the lists name every module-level
    double, in both directions — an unregistered fake fails here, and so does a
    registry entry for a class that no longer defines the method.

    Deliberately phrased without a count. Every new consent gate adds a fake,
    and a docstring that said "a seventh copy" was already wrong by the time the
    seventh landed — the number is exactly the sort of hand-maintained detail
    this test exists to stop relying on.
    """
    registry_name, registry = _REGISTRY_BY_METHOD[method]
    labels = [_label(fake) for fake in registry]
    # Checked before the set compare, which would otherwise absorb a duplicate.
    assert len(labels) == len(set(labels)), registry_name
    assert _discover_fakes(method) == set(labels), registry_name


def test_the_spend_prompt_keywords_bind_to_the_real_elicit():
    """`_elicit_approval` passes `message=`/`schema=` — they must still bind.

    The fakes above are checked against the real signature, but this asserts the
    other direction directly on the SDK: the exact keywords server.py sends are
    accepted by `Context.elicit` as installed.
    """
    signature = inspect.signature(Context.elicit)
    signature.bind(object(), message="ask", schema=type("S", (), {}))


def test_the_progress_keywords_bind_to_the_real_report_progress():
    """The progress pump sends all three by keyword — `progress=` included."""
    signature = inspect.signature(Context.report_progress)
    signature.bind(object(), progress=0.5, total=1.0, message="half")


# --- the 2026-07-28 input-required call shape -------------------------------
#
# From 2026-07-28 there is no back-channel for `elicitation/create`, so a consent
# gate carries its question out as the tool's `InputRequiredResult` and reads the
# answer off `ctx.input_responses` when the client retries. These pin the exact
# SDK surface `_elicit_approval` reaches for on that path, the same way the elicit
# tests above pin the back-channel one — a renamed property or a changed
# `InputRequiredResult` shape must fail HERE, not as a paid call that never asks.


def test_context_exposes_the_input_required_properties():
    """`_elicit_approval` reads these by name; a rename must fail loudly here."""
    for name in ("protocol_version", "input_responses", "request_state"):
        assert isinstance(getattr(Context, name), property), name


def test_the_production_path_builds_a_valid_input_required_result():
    """`_consent_via_input_required`'s first round yields a wire-valid result.

    Driving the real helper (no context answers) proves the `ElicitRequest` it
    nests validates against the installed SDK and surfaces as
    `resultType: "input_required"` — the shape the gated tools return verbatim.
    """

    class _Approve(server.BaseModel):
        approve: bool = False

    # The helper reads only `input_responses`; `None` is the first (unanswered) round.
    ctx = type("Ctx", (), {"input_responses": None})()
    with pytest.raises(server._ElicitationInputRequired) as excinfo:
        server._consent_via_input_required(
            ctx, "spend?", _Approve, server._SPEND_APPROVAL_WORDING
        )
    result = excinfo.value.result
    # Re-validate through the SDK model, as the wire codec would.
    reparsed = types.InputRequiredResult.model_validate(
        result.model_dump(by_alias=True)
    )
    assert reparsed.result_type == "input_required"
    [request] = reparsed.input_requests.values()
    assert isinstance(request, types.ElicitRequest)
    assert request.params.message == "spend?"
    assert request.params.requested_schema["properties"]["approve"]["type"] == "boolean"


def test_co_firing_gates_answer_on_distinct_channels():
    """`restart_comfyui` raises the network gate then the kill gate in one call.

    A shared `input_responses` key would let the kill gate read the network gate's
    "yes" as its own and stop a server nobody approved — so the two keys must
    differ. This is the one pair that can fire within a single `tools/call`.
    """
    network = server._consent_channel_key(server._NETWORK_APPROVAL_WORDING)
    kill = server._consent_channel_key(server._KILL_UNTRACKED_APPROVAL_WORDING)
    assert network != kill


def test_in_process_handshake_records_both_protocol_versions():
    """Smoke test: drive `server/discover` + `initialize` in process, per revision.

    The high-level client's `auto` mode probes `server/discover` and the `legacy`
    mode forces the `initialize` handshake; both return a negotiated version, and
    the server identifies itself with the release `_server_version` reports. A
    version pin adopts 2026-07-28 directly. This exercises the real handshake
    against the real `MCPServer` — the surface the input-required branch keys on —
    without a transport.
    """

    async def _drive():
        recorded = {}
        for mode in ("auto", "legacy", server._INPUT_REQUIRED_PROTOCOL):
            async with Client(server.mcp, mode=mode) as client:
                tools = await client.list_tools()
                recorded[mode] = (client.protocol_version, len(tools.tools))
                if client.server_info is not None:
                    assert client.server_info.version == server._server_version()
        return recorded

    recorded = asyncio.run(_drive())
    assert recorded["legacy"][0] == "2025-11-25"
    assert recorded["auto"][0] == server._INPUT_REQUIRED_PROTOCOL
    assert (
        recorded[server._INPUT_REQUIRED_PROTOCOL][0] == server._INPUT_REQUIRED_PROTOCOL
    )
    # Tools are discoverable on every revision (the gated ones among them).
    assert all(count > 0 for _, count in recorded.values())
