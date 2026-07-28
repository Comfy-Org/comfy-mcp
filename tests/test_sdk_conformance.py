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
"""

import inspect

import conftest
import pytest
import test_partner_generate
import test_run_template
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
# so a failure says which module to fix.
_ELICIT_FAKES = [
    ("test_partner_generate._FakeCtx", test_partner_generate._FakeCtx),
    ("test_run_template._FakeCtx", test_run_template._FakeCtx),
]

_PROGRESS_FAKES = [
    ("conftest._RecordingCtx", conftest._RecordingCtx),
    ("test_run_template._FakeCtx", test_run_template._FakeCtx),
]

_SESSION_FAKES = [
    ("test_partner_generate._FakeSession", test_partner_generate._FakeSession),
    ("test_run_template._FakeSession", test_run_template._FakeSession),
]


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


def test_the_spend_prompt_keywords_bind_to_the_real_elicit():
    """`_elicit_spend_approval` passes `message=`/`schema=` — they must still bind.

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
