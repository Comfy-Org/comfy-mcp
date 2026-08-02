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

import importlib
import inspect
import pathlib

import conftest
import pytest
import test_network_exposure
import test_partner_generate
import test_run_template
import test_run_workflow_spend
import test_switch_version
from mcp.server.mcpserver import Context
from mcp.server.session import ServerSession


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


# Every fake standing in for the real `Context.elicit`, named by where it lives
# so a failure says which module to fix. One per consent gate: the three spend
# gates, the version switch, and the network-exposure gate on the launch pair.
_ELICIT_FAKES = [
    ("test_network_exposure._FakeCtx", test_network_exposure._FakeCtx),
    ("test_partner_generate._FakeCtx", test_partner_generate._FakeCtx),
    ("test_run_template._FakeCtx", test_run_template._FakeCtx),
    ("test_run_workflow_spend._FakeCtx", test_run_workflow_spend._FakeCtx),
    ("test_switch_version._FakeCtx", test_switch_version._FakeCtx),
]

# Only the doubles that stand in for a context which also carries progress: the
# two STREAMING run verbs hand the same object to the prompt and to the run. The
# `switch_version` / `network_exposure` fakes have no `report_progress` because
# neither tool streams, so registering them here would assert a method their
# production path never calls.
_PROGRESS_FAKES = [
    ("conftest._RecordingCtx", conftest._RecordingCtx),
    ("test_run_template._FakeCtx", test_run_template._FakeCtx),
    ("test_run_workflow_spend._FakeCtx", test_run_workflow_spend._FakeCtx),
]

_SESSION_FAKES = [
    ("test_network_exposure._FakeSession", test_network_exposure._FakeSession),
    ("test_partner_generate._FakeSession", test_partner_generate._FakeSession),
    ("test_run_template._FakeSession", test_run_template._FakeSession),
    ("test_run_workflow_spend._FakeSession", test_run_workflow_spend._FakeSession),
    ("test_switch_version._FakeSession", test_switch_version._FakeSession),
]

#: Which registry above owns each SDK method a double may stand in for. Drives
#: the completeness check below, so adding a registry means adding an entry here.
_REGISTRY_BY_METHOD = {
    "elicit": ("_ELICIT_FAKES", _ELICIT_FAKES),
    "report_progress": ("_PROGRESS_FAKES", _PROGRESS_FAKES),
    "check_client_capability": ("_SESSION_FAKES", _SESSION_FAKES),
}


def _discover_fakes(method: str) -> set[str]:
    """Labels of every module-level class in ``tests/`` that defines ``method``.

    Own-``__dict__`` only, so a subclass that merely inherits the method is not
    double-counted, and same-module only, so an imported name is attributed
    where it is defined rather than everywhere it is used. The one-off doubles
    defined INSIDE a test function are deliberately out of reach: they are
    narrow overrides of a registered fake, scoped to a single assertion.
    """
    found: set[str] = set()
    for path in sorted(pathlib.Path(__file__).parent.glob("*.py")):
        module = importlib.import_module(path.stem)
        for attr, obj in vars(module).items():
            own = inspect.isclass(obj) and obj.__module__ == module.__name__
            if own and method in vars(obj):
                found.add(f"{module.__name__}.{attr}")
    return found


@pytest.mark.parametrize("label,fake", _ELICIT_FAKES, ids=[n for n, _ in _ELICIT_FAKES])
def test_fake_elicit_matches_the_real_context_signature(label, fake):
    """A renamed ``elicit`` keyword must fail HERE, not as a spend-path TypeError."""
    assert _params(fake.elicit) == _params(Context.elicit), label
    assert inspect.iscoroutinefunction(fake.elicit) == inspect.iscoroutinefunction(
        Context.elicit
    ), label


@pytest.mark.parametrize(
    "label,fake", _PROGRESS_FAKES, ids=[n for n, _ in _PROGRESS_FAKES]
)
def test_fake_report_progress_matches_the_real_context_signature(label, fake):
    """The production call only debug-logs on failure, so drift is silent there."""
    assert _params(fake.report_progress) == _params(Context.report_progress), label
    assert inspect.iscoroutinefunction(
        fake.report_progress
    ) == inspect.iscoroutinefunction(Context.report_progress), label


@pytest.mark.parametrize(
    "label,fake", _SESSION_FAKES, ids=[n for n, _ in _SESSION_FAKES]
)
def test_fake_check_client_capability_matches_the_real_session_signature(label, fake):
    """The capability probe decides whether a human is asked before money moves."""
    real = ServerSession.check_client_capability
    assert _params(fake.check_client_capability) == _params(real), label
    assert inspect.iscoroutinefunction(
        fake.check_client_capability
    ) == inspect.iscoroutinefunction(real), label


@pytest.mark.parametrize("method", sorted(_REGISTRY_BY_METHOD))
def test_every_module_level_fake_is_registered(method):
    """A sixth copy of a consent fake must be CHECKED, not merely written.

    The registries above are hand-written, and hand-written lists rot: three of
    the five consent fakes went unlisted for as long as they existed, so they
    could have drifted from the SDK while their own tests stayed green. This
    asserts the lists name every module-level double, in both directions — an
    unregistered fake fails here, and so does a registry entry for a class that
    no longer defines the method.
    """
    registry_name, registry = _REGISTRY_BY_METHOD[method]
    assert _discover_fakes(method) == {label for label, _ in registry}, registry_name


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
