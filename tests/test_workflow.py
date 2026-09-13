"""Tests for the ``comfy workflow`` tools — list / set-slot / vary / notes.

These lock in the passthrough argv (global flags before the subcommand, same
rule the wrapper enforces) for the tools that let an agent parameterize a
fetched template — the ``fetch_template`` -> ``set_workflow_slot`` ->
``run_workflow`` loop — without hand-editing raw workflow JSON, plus
``list_workflow_notes``, the read-only reader for the authored documentation
that same template carries. The behaviors they own on top of the passthrough:
1. ``set_workflow_slot`` passes each override as a positional ``ADDR=VALUE`` and
   defaults to ``--stdout`` (non-destructive), togglable off.
2. ``vary_workflow`` repeats ``--slot`` per address and forwards ``--out-dir``
   only when given.
3. ``vary_workflow`` pre-checks each slot entry's value against the JSON-array
   contract comfy-cli enforces, so an unquoted comma-bearing prompt is named
   here instead of failing opaquely (or behind a server-connection error) later.
4. ``list_workflow_notes`` degrades to the ``unsupported`` shape on a comfy-cli
   that predates the ``workflow notes`` verb, instead of relaying Click's raw
   usage dump — the common case while the verb is newer than the version floor
   — and refuses to fire that degrade for a phrase Click merely echoed back out
   of the caller's own ``workflow_path``.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest
from conftest import envelope
from pydantic import ValidationError

from comfy_mcp import argv, errors, params
from comfy_mcp.server import _internal as server


def test_list_workflow_slots_argv(patched_run):
    """Passthrough: `comfy --json --where local workflow slots <path>`."""
    data = [{"addr": "6.text", "value": "a cat"}, {"addr": "3.seed", "value": 42}]
    calls = patched_run(envelope(data=data))

    assert server.list_workflow_slots("/tmp/flux.json") == data

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["workflow", "slots", "/tmp/flux.json"]  # subcommand after


def test_list_workflow_notes_argv(patched_run):
    """Passthrough: `comfy --json --where local workflow notes <path>`.

    The envelope `data` is a `{workflow, count, notes[]}` dict and comes back
    unchanged — this repo parses no workflow JSON of its own (AGENTS.md).
    """
    data = {
        "workflow": "/tmp/flux.json",
        "count": 2,
        "notes": [
            {
                "id": 5,
                "type": "MarkdownNote",
                "title": "Note",
                "text": "Trigger word: `ohwx person`",
                "pos": [-5870, 1890],
                "size": [230, 88],
                "subgraph": None,
            },
            {
                "id": 7,
                "type": "Note",
                "title": None,
                "text": "Download the LoRA into models/loras.",
                "pos": [10, 20],
                "size": [200, 60],
                "subgraph": {"id": "abc-123", "name": "sampler"},
            },
        ],
    }
    calls = patched_run(envelope(data=data))

    assert server.list_workflow_notes("/tmp/flux.json") == data

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["workflow", "notes", "/tmp/flux.json"]  # subcommand after


def test_list_workflow_notes_empty_is_not_an_error(patched_run):
    """A workflow with no notes is a normal `count: 0` payload, not a raise."""
    data = {"workflow": "/tmp/flux.json", "count": 0, "notes": []}
    patched_run(envelope(data=data))

    assert server.list_workflow_notes("/tmp/flux.json") == data


def test_list_workflow_notes_degrades_without_the_verb(patched_run):
    """A comfy-cli predating `workflow notes` reads as a version gap, not a break.

    The verb ships in releases AFTER the `_MIN_COMFY_CLI` floor, so an install
    that satisfies the version guard can still lack it — the common path today.
    Relaying Click's raw usage dump would read as a broken MCP server, so this
    degrades to the `unsupported` shape `_freshness_report` established, and
    points at the path that still works: the notes are in the frontend-format
    JSON `fetch_template` already wrote.
    """
    patched_run(
        "",
        returncode=2,
        stderr="Usage: comfy workflow [OPTIONS] COMMAND\nNo such command 'notes'.",
    )

    result = server.list_workflow_notes("/tmp/flux.json")

    assert result["unsupported"] is True
    assert "workflow notes unavailable" in result["error"]
    # Redirects to the reachable fallback rather than dead-ending.
    assert "fetch_template" in result["error"]
    assert "widgets_values[0]" in result["error"]
    assert "/tmp/flux.json" in result["error"]
    # None of the raw wrapper/CLI text leaks through.
    assert "No such command" not in result["error"]
    assert "Usage: comfy" not in result["error"]
    assert "returned no JSON" not in result["error"]


def test_list_workflow_notes_keeps_a_real_error_raw(patched_run):
    """A verb comfy-cli DID dispatch must never be waved through as a gap.

    An API-format export is the case that matters: comfy-cli rejects it with
    `workflow_not_frontend_format`, and the agent has to see that to know to
    re-fetch. Degrading it would instead assert nothing is wrong while the
    template's documentation silently never gets read.
    """
    patched_run(
        envelope(
            ok=False,
            error={
                "code": "workflow_not_frontend_format",
                "message": "`comfy workflow` requires the frontend-format workflow.",
            },
        )
    )

    with pytest.raises(server.ComfyCliError, match="workflow_not_frontend_format"):
        server.list_workflow_notes("/tmp/api.json")


def test_list_workflow_notes_relayed_phrase_is_not_unsupported(patched_run):
    """A failure that merely QUOTES the phrase, inside an envelope, stays raw.

    `_is_missing_verb_error` requires the no-envelope + usage-exit pair exactly
    so a nested error relaying "No such command 'notes'" from somewhere else
    cannot be mistaken for the verb itself being absent.
    """
    patched_run(
        envelope(
            ok=False,
            error={
                "code": "workflow_read_failed",
                "message": "a hook failed: No such command 'notes'.",
            },
        ),
        returncode=2,
    )

    with pytest.raises(server.ComfyCliError, match="workflow_read_failed"):
        server.list_workflow_notes("/tmp/flux.json")


def test_list_workflow_notes_echoed_phrase_is_not_unsupported(patched_run):
    """A caller cannot forge the version gap through its own `workflow_path`.

    The path is a bare positional, and Click echoes an offending value verbatim
    in a usage error — the same exit 2 with no envelope `_is_missing_verb_error`
    reads, which is the one route to a false `unsupported` its two conditions
    cannot close. `_phrase_is_only_the_caller_s` subtracts the caller's own text
    so a real failure stays a real failure.
    """
    path = "no such command 'notes'"
    patched_run(
        "",
        returncode=2,
        stderr=(
            "Usage: comfy workflow notes [OPTIONS] FILE\n"
            f"Error: Invalid value for 'FILE': {path!r} does not exist."
        ),
    )

    with pytest.raises(server.ComfyCliError):
        server.list_workflow_notes(workflow_path=path)


def test_list_workflow_notes_degrades_with_an_ordinary_path(patched_run):
    """The echoed-input check must not cost the genuine degrade.

    Discounting the caller's own text is subtraction, not a veto: an ordinary
    `workflow_path` shares no wording with Click's message, so the parser's own
    phrase survives and the version gap still reports as one.
    """
    patched_run(
        "",
        returncode=2,
        stderr="Usage: comfy workflow [OPTIONS] COMMAND\nNo such command 'notes'.",
    )

    assert server.list_workflow_notes("workflow.json")["unsupported"] is True


def test_set_workflow_slot_argv_default_stdout(patched_run):
    """Default: positional ADDR=VALUE overrides + trailing --stdout (non-destructive)."""
    calls = patched_run(envelope(data={"modified": True}))

    result = server.set_workflow_slot(
        "/tmp/flux.json", ["6.text=a red bicycle", "3.seed=42"]
    )
    assert result == {"modified": True}

    assert calls[-1]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "/tmp/flux.json",
        "6.text=a red bicycle",  # each override passed as a positional ADDR=VALUE
        "3.seed=42",
        "--stdout",  # default: return the workflow, don't mutate the file
    ]


def test_set_workflow_slot_stdout_false_writes_in_place(patched_run):
    """stdout=False drops --stdout so comfy-cli writes the change back to the file."""
    calls = patched_run(envelope(data=None))

    server.set_workflow_slot("/tmp/flux.json", ["3.seed=7"], stdout=False)

    cmd = calls[-1]["cmd"]
    assert cmd[4:] == ["workflow", "set-slot", "/tmp/flux.json", "3.seed=7"]
    assert "--stdout" not in cmd


def test_set_workflow_slot_rejects_option_like_override(no_spawn):
    """A leading-dash override is refused before any child spawns.

    Splatted in as a positional it would BE the flag — `"--stdout"` would flip
    the in-place-write behavior the ``stdout`` argument owns.
    """
    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.set_workflow_slot("/tmp/flux.json", ["--stdout"])

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.set_workflow_slot("/tmp/flux.json", ["6.text=x", "--stdout"])


def test_workflow_path_positional_rejects_option_like(no_spawn):
    """The sibling `workflow_path` positional is guarded too, not just the overrides.

    All four tools splat the path in bare, so a leading-dash path is read as a
    flag: for `set-slot` that shifts the first override into the path slot,
    which is the very injection the override guard exists to stop. The error
    names the escape hatch — a genuinely dash-leading filename works as `./-x`.
    """
    with pytest.raises(server.ComfyCliError, match=r"leading '-'.*\./"):
        server.set_workflow_slot("--stdout", ["6.text=x"])

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.list_workflow_slots("--stdout")

    with pytest.raises(server.ComfyCliError, match=r"leading '-'.*\./"):
        server.list_workflow_notes("--stdout")

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.vary_workflow("--out-dir", ["3.seed=[1,2]"])


def test_workflow_path_guard_message_tracks_format_requirement(no_spawn):
    """Frontend-only tools name the frontend format; dual-format tools do not."""
    with pytest.raises(server.ComfyCliError, match="frontend-format"):
        server.list_workflow_slots("--stdout")

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.validate_workflow("--stdout")
    assert "frontend-format" not in str(excinfo.value)


def test_workflow_path_guard_allows_dot_slash_dash_name(patched_run):
    """The documented escape hatch actually works: `./-flux.json` is not refused."""
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot("./-flux.json", ["6.text=x"])
    server.list_workflow_notes("./-flux.json")

    argvs = [c["cmd"][4:] for c in calls]
    assert ["workflow", "set-slot", "./-flux.json", "6.text=x", "--stdout"] in argvs
    assert ["workflow", "notes", "./-flux.json"] in argvs


# The six tools that take a `workflow_path`. One guard covers all of them, so
# every test of that guard runs against the whole set rather than a
# representative — `no_spawn` blocks both spawn paths, which is what makes
# `run_workflow` (which streams rather than going through `_run_comfy`) provable
# in the same parametrization.
_WORKFLOW_PATH_CALLS = [
    pytest.param(
        lambda path: asyncio.run(server.run_workflow(path)), id="run_workflow"
    ),
    pytest.param(server.validate_workflow, id="validate_workflow"),
    pytest.param(server.list_workflow_slots, id="list_workflow_slots"),
    pytest.param(server.list_workflow_notes, id="list_workflow_notes"),
    pytest.param(
        lambda path: server.set_workflow_slot(path, ["6.text=x"]),
        id="set_workflow_slot",
    ),
    pytest.param(
        lambda path: server.vary_workflow(path, ["3.seed=[1,2]"]),
        id="vary_workflow",
    ),
]


@pytest.mark.parametrize("call", _WORKFLOW_PATH_CALLS)
def test_workflow_path_guard_rejects_an_oversized_path(call, no_spawn):
    """A path far past PATH_MAX is refused before it can reach argv.

    An oversized argv is rejected by the OS with an `OSError` (`E2BIG`) no
    caller converts, rather than failing as a clean `ComfyCliError` like every
    other bad input. One check in `_guard_workflow_path` covers all six tools
    that take a `workflow_path`, so all six are exercised here — `no_spawn`
    blocks both spawn paths, which is what makes `run_workflow` (which streams
    rather than going through `_run_comfy`) provable in the same parametrization.
    """
    oversized = "w" * (argv._MAX_PATH_ARG_LEN + 1)

    with pytest.raises(server.ComfyCliError, match="exceeds") as excinfo:
        call(oversized)

    # Length-not-value: the size check runs first so the failure is named as a
    # size rather than as whatever shape complaint the value happens to also
    # trip, and so the refused string itself stays out of the tool response and
    # the failure log.
    assert oversized not in str(excinfo.value)


def test_workflow_path_guard_allows_a_path_at_the_ceiling():
    """The cap is generous enough that no real path is anywhere near it.

    It is an argv-safety guard, not an attempt to close the residual
    `_phrase_is_only_the_caller_s` forgery window — which would need a cap under
    500 characters. The boundary value itself passes.
    """
    at_ceiling = "w" * argv._MAX_PATH_ARG_LEN

    assert argv._guard_workflow_path(at_ceiling)
    assert argv._guard_workflow_path(at_ceiling, frontend=True)


def test_guard_arg_len_reads_the_module_constant_at_call_time(monkeypatch):
    """The default ceiling is resolved in the BODY, not bound at definition.

    `limit: int = _MAX_PATH_ARG_LEN` would copy the constant into the signature
    once, so the repo's patch-the-owning-module convention would move the name
    while every call kept reading the old 4096 — a guard that silently tests
    nothing. `_MAX_PATH_ARG_LEN` stays the single source of truth instead.
    """
    monkeypatch.setattr(argv, "_MAX_PATH_ARG_LEN", 8)

    with pytest.raises(server.ComfyCliError, match="exceeds"):
        argv._guard_arg_len("workflow_path", "w" * 9)
    assert argv._guard_arg_len("workflow_path", "w" * 8) == "w" * 8
    # An explicit `limit` still wins over the module default.
    assert argv._guard_arg_len("url", "u" * 9, 16) == "u" * 9


@pytest.mark.parametrize("call", _WORKFLOW_PATH_CALLS)
def test_workflow_path_guard_rejects_an_unencodable_path(call, no_spawn):
    """Length is not the only way a string fails to reach `execve`.

    A lone surrogate survives the MCP JSON wire intact — `json.loads` accepts
    `"\\ud800"` — and passes every value guard, but `subprocess` encodes POSIX
    argv with `os.fsencode`, which cannot render it. That raises
    `UnicodeEncodeError` from the `Popen(...)` OUTSIDE `_run_comfy_raw`'s `try`,
    so it escapes as an internal error rather than a `ComfyCliError`. Covered
    for the whole path-shaped family in `_guard_arg_len`, so all six
    `workflow_path` tools are exercised here.
    """
    with pytest.raises(server.ComfyCliError, match="cannot be encoded") as excinfo:
        call("/tmp/\ud800.json")

    # The message names the encoding rather than asserting a cause: under a
    # non-UTF-8 filesystem encoding an ordinary multibyte name fails the same way.
    assert sys.getfilesystemencoding() in str(excinfo.value)


def test_guard_arg_len_reports_size_before_encodability():
    """A value that is BOTH oversized and unencodable is named as a size.

    Same per-value ordering rule the rest of these guards follow: the size is
    the more actionable of the two, and it is what the caller can see.
    """
    oversized = "\ud800" + "w" * argv._MAX_PATH_ARG_LEN

    with pytest.raises(server.ComfyCliError, match="exceeds"):
        argv._guard_arg_len("workflow_path", oversized)


def test_option_like_rejection_bounds_the_value_it_echoes():
    """A value UNDER the cap is still bounded on its way into the error.

    The length guards above stop the megabyte case, but a dash-leading value at
    the 4096-character ceiling is legal input to `_reject_option_like`, whose
    echo has the widest reach in the module — most of its twenty-odd call sites
    guard values with no length cap at all. It renders through
    `_clip_for_error`, so the message honors `errors._MAX_ERROR_FIELD_CHARS`
    like every other field that quotes caller input.
    """
    at_ceiling = "-" + "w" * (argv._MAX_PATH_ARG_LEN - 1)

    with pytest.raises(server.ComfyCliError, match="leading '-'") as excinfo:
        argv._guard_workflow_path(at_ceiling)

    message = str(excinfo.value)
    assert at_ceiling not in message
    # A few words of prose around the bounded field, not 4 KB of `w`.
    assert len(message) < errors._MAX_ERROR_FIELD_CHARS + 200


def test_option_like_rejection_is_unchanged_for_an_ordinary_value():
    """Bounding the echo did not change the message any real caller sees.

    `_clip_for_error` quotes the fragment itself, so for anything whose rendered
    form already fits the bound it is byte-identical to the `{value!r}` this
    used to interpolate — the wording every other guard test asserts on.
    """
    with pytest.raises(server.ComfyCliError) as excinfo:
        argv._guard_workflow_path("-flux.json")

    assert repr("-flux.json") in str(excinfo.value)


def test_workflow_tools_reject_embedded_nul(no_spawn):
    """A NUL anywhere surfaces as ComfyCliError, not subprocess's bare ValueError.

    Orthogonal to the leading-dash guard: `subprocess` cannot carry a NUL in
    argv at all, so it is refused on option values (`--slot`, `--out-dir`) too,
    not just on the bare positionals.
    """
    for call in (
        lambda: server.list_workflow_slots("/tmp/f\0.json"),
        lambda: server.list_workflow_notes("/tmp/f\0.json"),
        lambda: server.set_workflow_slot("/tmp/f\0.json", ["6.text=x"]),
        lambda: server.set_workflow_slot("/tmp/f.json", ["6.text=\0"]),
        lambda: server.vary_workflow("/tmp/f\0.json", ["3.seed=[1,2]"]),
        lambda: server.vary_workflow("/tmp/f.json", ["3.seed=\0"]),
        lambda: server.vary_workflow("/tmp/f.json", ["3.seed=[1,2]"], out_dir="/o\0"),
    ):
        with pytest.raises(server.ComfyCliError, match="embedded NUL"):
            call()


def test_vary_workflow_option_value_guards_read_the_first_char_only(patched_run):
    """The `slots`/`out_dir` guards refuse a LEADING dash and nothing more.

    Those two are option VALUES, which Click takes verbatim (`--out-dir --slot`
    parses as `out_dir="--slot"`, not as a shift), so they were injection-safe
    unguarded. They are guarded anyway as input hygiene — the same call
    `search_templates` makes for its filters — which makes over-rejection the
    real risk here rather than injection: a dash INSIDE a slot value, and a
    relative path that merely contains one, must still ride through.
    """
    calls = patched_run(envelope(data={"variants": 2}))

    server.vary_workflow(
        "/tmp/flux.json", ['6.text=["a -b", "c"]'], out_dir="./out-dir"
    )

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        '6.text=["a -b", "c"]',
        "--out-dir",
        "./out-dir",
    ]


def test_set_workflow_slot_guard_leaves_valid_overrides_alone(patched_run):
    """The guard reads the override's FIRST character only: `-` inside a VALUE is fine."""
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot("/tmp/flux.json", ["6.text=x", "4.ckpt=sd-xl --turbo"])

    assert calls[-1]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "/tmp/flux.json",
        "6.text=x",
        "4.ckpt=sd-xl --turbo",
        "--stdout",
    ]


def test_vary_workflow_argv_repeats_slot_flag(patched_run):
    """Each address becomes its own `--slot "ADDR=[...]"`; no --out-dir when unset."""
    calls = patched_run(envelope(data={"variants": 3}))

    result = server.vary_workflow(
        "/tmp/flux.json", ["3.seed=[1,2,3]", '6.text=["cat","dog","fish"]']
    )
    assert result == {"variants": 3}

    cmd = calls[0]["cmd"]
    assert cmd[4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        "3.seed=[1,2,3]",
        "--slot",
        '6.text=["cat","dog","fish"]',
    ]
    assert "--out-dir" not in cmd  # stdout NDJSON mode when out_dir is unset


def test_vary_workflow_forwards_out_dir(patched_run, tmp_path):
    """out_dir appends `--out-dir <dir>` so variants are written to files."""
    calls = patched_run(envelope(data=None))

    out = tmp_path / "variants"
    server.vary_workflow("/tmp/flux.json", ["3.seed=[1,2]"], out_dir=str(out))

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        "3.seed=[1,2]",
        "--out-dir",
        str(out),
    ]


def test_vary_workflow_rejects_option_like_slot_and_out_dir(no_spawn):
    """`--slot` values and `--out-dir` are guarded as input hygiene.

    Click takes an option's value verbatim, so neither is an injection vector
    (unlike the sibling `workflow_path` positional, covered above). They are
    still refused so a dash-leading slot expression or output directory fails
    with a named error instead of a comfy-cli usage error or `--help` text that
    then fails envelope parsing — the same call `search_templates` makes for its
    filters.
    """
    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.vary_workflow("/tmp/flux.json", ["--help"])

    # The guard scans every slot, not just the first.
    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.vary_workflow("/tmp/flux.json", ["3.seed=[1,2]", "--help"])

    with pytest.raises(server.ComfyCliError, match=r"leading '-'.*\./"):
        server.vary_workflow("/tmp/flux.json", ["3.seed=[1,2]"], out_dir="--help")


def test_vary_workflow_rejects_an_oversized_out_dir(no_spawn):
    """An oversized `out_dir` is refused before it can reach argv.

    The sibling of the `workflow_path` cap above, on the OTHER caller-supplied
    path this tool forwards: an oversized argv string is rejected by the OS with
    an `OSError` (`E2BIG`) `_run_comfy_raw` never converts, because its `try`
    wraps only `communicate()` and not the `Popen(...)` that raises.
    """
    oversized = "/tmp/" + "v" * argv._MAX_PATH_ARG_LEN

    with pytest.raises(server.ComfyCliError, match="exceeds") as excinfo:
        server.vary_workflow("/tmp/flux.json", ["3.seed=[1,2]"], out_dir=oversized)

    # Length-not-value: the size check runs ahead of both value guards, whose
    # echoes would name the value instead of its size.
    assert oversized not in str(excinfo.value)


def test_vary_workflow_allows_an_out_dir_at_the_ceiling(patched_run):
    """The boundary value itself rides through to `--out-dir`."""
    calls = patched_run(envelope(data=None))
    at_ceiling = "/tmp/" + "v" * (argv._MAX_PATH_ARG_LEN - len("/tmp/"))
    assert len(at_ceiling) == argv._MAX_PATH_ARG_LEN

    server.vary_workflow("/tmp/flux.json", ["3.seed=[1,2]"], out_dir=at_ceiling)

    assert calls[0]["cmd"][-2:] == ["--out-dir", at_ceiling]


def test_vary_workflow_accepts_json_quoted_comma_bearing_prompt(patched_run):
    """The working form from the field: commas INSIDE a JSON-quoted prompt.

    This is the case the contract exists for — a prompt naturally contains
    commas, and only JSON quoting keeps them part of one value instead of
    splitting it. Two prompts, one seed each, zipped into two variants.
    """
    calls = patched_run(envelope(data={"count": 2}))

    slot = (
        '1.prompt=["a lighthouse at dawn, oil painting", '
        '"a lighthouse at noon, watercolor"]'
    )
    server.vary_workflow("/tmp/flux.json", [slot, "3.seed=[1,2]"])

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        slot,  # forwarded byte-for-byte; the guard parses, it never rewrites
        "--slot",
        "3.seed=[1,2]",
    ]


def test_vary_workflow_rejects_unquoted_comma_bearing_slot_value(no_spawn):
    """The failing form from the field, named before any child spawns.

    `[a lighthouse at dawn, oil painting]` is not valid JSON, so comfy-cli reads
    it as one bare string and dies with `value must be a JSON array (got str)`
    — after it has already loaded the workflow and fetched `object_info`, so
    with ComfyUI down the real mistake never surfaces at all. The pre-flight
    names WHICH entry is wrong and shows the quoted form that fixes it.
    """
    with pytest.raises(server.ComfyCliError) as excinfo:
        server.vary_workflow(
            "/tmp/flux.json",
            ["3.seed=[1,2]", "1.prompt=[a lighthouse at dawn, oil painting]"],
        )

    message = str(excinfo.value)
    assert "slots[1]" in message  # names the offending entry, not just "a value"
    assert "1.prompt" in message  # ...and its address
    assert "must be a JSON array" in message  # ...and the contract it broke
    assert "not valid JSON" in message
    assert '"a lighthouse at dawn, oil painting"' in message  # ...and the fix


def test_vary_workflow_rejects_non_array_json_slot_value(no_spawn):
    """Valid JSON that isn't an array is refused too, with the type named.

    comfy-cli parses `3.seed=42` fine — as an int — then rejects it, because
    `vary` zips LISTS. The one-element-array fix is spelled out.
    """
    with pytest.raises(server.ComfyCliError, match=r"slots\[0\].*must be a JSON array"):
        server.vary_workflow("/tmp/flux.json", ["3.seed=42"])

    with pytest.raises(server.ComfyCliError, match="got int"):
        server.vary_workflow("/tmp/flux.json", ["3.seed=42"])

    with pytest.raises(server.ComfyCliError, match=r"3\.seed=\[42\]"):
        server.vary_workflow("/tmp/flux.json", ["3.seed=42"])


def test_vary_workflow_rejects_slot_without_equals(no_spawn):
    """An entry with no `=` can't be split into ADDR and value — say so here.

    comfy-cli's own message for this (`Expected ADDR=VALUE`) never mentions the
    JSON-array half of the contract, so a caller who fixes only the `=` walks
    straight into the second error.
    """
    with pytest.raises(server.ComfyCliError, match=r"slots\[0\].*ADDR=\[v1,v2"):
        server.vary_workflow("/tmp/flux.json", ["3.seed"])


def test_vary_workflow_slot_error_is_bounded(no_spawn):
    """A huge malformed slot can't produce an unbounded error string."""
    with pytest.raises(server.ComfyCliError) as excinfo:
        server.vary_workflow("/tmp/flux.json", ["6.text=[" + "x" * 10_000 + "]"])

    assert "x" * 10_000 not in str(excinfo.value)
    assert len(str(excinfo.value)) < 2_000


@pytest.mark.parametrize(
    "value, marker",
    [("42", "got int"), ("[bad", "not valid JSON")],
    ids=["non-array", "invalid-json"],
)
def test_vary_workflow_slot_error_bounds_the_address_too(no_spawn, value, marker):
    """The ADDRESS half is caller-sized as well, and is clipped like the value.

    Everything before the first `=` is whatever the caller sent — nothing
    upstream caps it — so echoing it raw would blow the per-field bound from the
    other side and, with the opt-in failure log on, write a multi-KB line per
    attempt. Both error branches interpolate the address, so both are checked.
    """
    address = "A" * 10_000
    with pytest.raises(server.ComfyCliError) as excinfo:
        server.vary_workflow("/tmp/flux.json", [f"{address}={value}"])

    message = str(excinfo.value)
    assert marker in message  # still the branch we meant to exercise
    assert address not in message
    assert len(message) < 2_000


def test_vary_workflow_defers_unparseably_nested_slot_to_the_engine(
    patched_run, monkeypatch
):
    """Nesting too deep for THIS stack is not a verdict on comfy-cli's.

    `json.loads` raises `RecursionError` once the nesting outruns the remaining
    stack, and this pre-check runs several frames down (MCP handler -> tool ->
    loop -> helper) from where comfy-cli parses — a fresh subprocess. So a depth
    that fails here can still parse there, and refusing it would break the
    invariant the pre-check rests on: it may only refuse what comfy-cli would
    also refuse. Forward it and let the engine's own parse decide.

    The error is injected rather than provoked with real nesting: the depth that
    trips the C scanner moved between 3.10 and 3.14 (3.14 parses 20k-deep arrays
    that 3.10 refuses), so a literal deep value would silently stop exercising
    this branch on one of the two interpreters CI runs.
    """
    calls = patched_run(envelope(data={"count": 1}))

    nested = "[" * 200 + "]" * 200
    slot = f"6.text={nested}"
    real_loads = server.json.loads

    def loads(text, *args, **kwargs):
        if text == nested:
            raise RecursionError("maximum recursion depth exceeded")
        return real_loads(text, *args, **kwargs)

    monkeypatch.setattr(server.json, "loads", loads)

    server.vary_workflow("/tmp/flux.json", [slot])

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        slot,  # untouched: the guard abstained, it did not rewrite or refuse
    ]

    # No sticky state: the injected failure is scoped to `nested`, so a normal
    # entry right behind it still goes through the guard and out to the engine.
    server.vary_workflow("/tmp/flux.json", ["3.seed=[1,2]"])
    assert calls[1]["cmd"][4:][-2:] == ["--slot", "3.seed=[1,2]"]


def test_vary_workflow_survives_a_real_recursion_error(patched_run):
    """A genuine stack exhaustion, not an injected one, leaves the server usable.

    The companion test above injects `RecursionError` for deterministic branch
    coverage, which by construction cannot say anything about the interpreter's
    health afterwards. This one provokes the real thing: CPython's JSON C
    scanner raises through `Py_EnterRecursiveCall`, which reserves stack
    headroom rather than running the stack to the ground, so the frames unwind
    cleanly and the next call parses normally. That matters here because this is
    a long-lived single process — one hostile slot must not degrade every later
    one.

    Skipped rather than silently vacuous if the interpreter happens to swallow
    the depth: the threshold moved between 3.10 and 3.14 and may move again.
    """
    # The whole entry must stay UNDER the spawn-size gate, or the guard abstains
    # there and never reaches `json.loads` — the assertions below would still
    # pass while proving nothing about recursion. Derived from the gate and
    # asserted, so it cannot drift back over if either number changes.
    prefix = "6.text="
    depth = (params._MAX_PRECHECKED_SLOT_BYTES - len(prefix)) // 2
    nested = "[" * depth + "]" * depth
    slot = prefix + nested
    assert len(slot.encode("utf-8")) <= params._MAX_PRECHECKED_SLOT_BYTES

    try:
        json.loads(nested)
    except RecursionError:
        pass
    else:  # pragma: no cover - depends on the interpreter's limit
        pytest.skip(f"this interpreter parses {depth}-deep arrays without raising")

    calls = patched_run(envelope(data={"count": 1}))

    server.vary_workflow("/tmp/flux.json", [slot])
    server.vary_workflow("/tmp/flux.json", ["3.seed=[1,2]"])

    assert len(calls) == 2
    assert calls[1]["cmd"][4:][-2:] == ["--slot", "3.seed=[1,2]"]


def test_vary_workflow_defers_oversized_int_literal_to_the_engine(patched_run):
    """An interpreter limit is not a syntax error, and must not be reported as one.

    Since 3.11 `json.loads` refuses an integer literal longer than
    `sys.get_int_max_str_digits()` (4300 by default) with a PLAIN `ValueError`,
    not a `JSONDecodeError` — `[<5000 digits>]` is perfectly well-formed JSON
    that this interpreter simply declines to build. comfy-cli parses in its own
    subprocess, under its own interpreter and limit, so it may well accept it.
    Refusing here would both mislabel it 'not valid JSON' and reject a value the
    engine takes, so the guard abstains.
    """
    calls = patched_run(envelope(data={"count": 1}))

    slot = "3.seed=[" + "9" * 5_000 + "]"
    server.vary_workflow("/tmp/flux.json", [slot])

    assert calls[0]["cmd"][4:] == ["workflow", "vary", "/tmp/flux.json", "--slot", slot]


def test_vary_workflow_slot_error_bound_survives_repr_escaping(no_spawn):
    """The cap has to hold on the RENDERED field, not the pre-quoted content.

    `repr` turns one control character into a 4-character escape, so clipping
    the raw text to the cap and quoting afterwards would put ~4x the cap into
    the message — a bound that reads as if it holds while the field blows past
    it. The other bounded tests use printable characters and would not catch it.
    """
    # NUL is rejected earlier by `_reject_nul`; \x01 is not, and reprs as `\x01`.
    with pytest.raises(server.ComfyCliError) as excinfo:
        server.vary_workflow("/tmp/flux.json", ["6.text=[" + "\x01" * 10_000 + "]"])

    assert len(str(excinfo.value)) < 2_000


@pytest.mark.parametrize(
    "text",
    ["\x01" * 10_000, "x" * 10_000, "\U0001f600" * 10_000],
    ids=["control", "printable", "astral"],
)
def test_clip_for_error_never_exceeds_the_cap(text):
    """The returned field is bounded whatever the escaping does to it.

    Checked directly rather than through a message, because the cap is this
    helper's whole contract and the call sites add their own fixed prose on top.
    """
    assert len(argv._clip_for_error(text)) <= errors._MAX_ERROR_FIELD_CHARS


def test_clip_for_error_matches_an_unsliced_repr():
    """Slicing the source before `repr` must not change the rendered prefix.

    The pre-slice is an allocation guard, not a behavior change: every source
    character contributes at least one character to the repr, so the escaping of
    the retained prefix cannot depend on anything past the cap.

    Stated over quote-free input on purpose — the surrounding quote character is
    the one part that CAN differ, which the next test pins down.
    """
    for text in ("\x01" * 10_000, "y" * 10_000, "a, b" * 5_000):
        clipped = argv._clip_for_error(text)
        assert clipped.endswith("…")
        assert repr(text).startswith(clipped[:-1])


def test_clip_for_error_quote_style_follows_the_slice():
    """The one thing the pre-slice changes, pinned so it stays cosmetic.

    `repr` switches to double quotes for a string containing an apostrophe and
    no double quote. That decision is now made over the SLICED prefix, so an
    apostrophe past the cap flips it relative to an unsliced `repr`. Harmless in
    an already-truncated preview — but asserted rather than assumed, so it stays
    a quoting difference and does not quietly become an escaping one.
    """
    cap = errors._MAX_ERROR_FIELD_CHARS
    text = "a" * (cap + 100) + "'"

    clipped = argv._clip_for_error(text)

    assert len(clipped) <= cap
    assert not repr(text).startswith(clipped[:-1])  # the quote char differs...
    assert repr(text[:cap]).startswith(clipped[:-1])  # ...and nothing else does


def test_vary_workflow_defers_unspawnably_long_slot_to_the_engine(patched_run):
    """A value past the single-argv limit is forwarded, not parsed and not refused.

    Linux caps one argv entry at `MAX_ARG_STRLEN` (128 KiB), so a `--slot` value
    beyond it can never reach comfy-cli — the spawn fails first. Parsing it here
    would be real work in the long-lived parent for a value that cannot land, so
    the guard abstains. Crucially it abstains rather than refuses: the pre-check
    still never rejects anything the engine would have accepted.
    """
    calls = patched_run(envelope(data={"count": 1}))

    # Deliberately a value the contract check would REFUSE (a JSON string, not
    # an array): forwarding it is only possible if the gate abstained before the
    # parse. A well-formed array here would ride through either way and prove
    # nothing about which branch ran.
    oversized = '"' + "9" * (params._MAX_PRECHECKED_SLOT_BYTES + 1) + '"'
    slot = f"3.seed={oversized}"
    server.vary_workflow("/tmp/flux.json", [slot])

    assert calls[0]["cmd"][4:] == ["workflow", "vary", "/tmp/flux.json", "--slot", slot]


def test_vary_workflow_size_gate_counts_bytes_not_characters(patched_run):
    """`MAX_ARG_STRLEN` is a BYTE cap, so the gate has to encode before measuring.

    A value of astral characters is four bytes each: well under the limit by
    `len()` and well over it once `execve` sees it. Counting characters would
    let exactly the largest values — the ones the gate exists for — slip through
    and be parsed in the parent anyway.
    """
    calls = patched_run(envelope(data={"count": 1}))

    # A quarter of the byte budget in characters, four bytes each => over it.
    # A JSON string rather than an array, so reaching the parse would REFUSE it
    # and forwarding proves the gate measured bytes and abstained.
    chars = (params._MAX_PRECHECKED_SLOT_BYTES // 4) + 100
    value = '"' + "\U0001f600" * chars + '"'
    slot = f"6.text={value}"
    assert len(slot) < params._MAX_PRECHECKED_SLOT_BYTES  # under, counted wrong
    assert len(slot.encode("utf-8")) > params._MAX_PRECHECKED_SLOT_BYTES  # over

    server.vary_workflow("/tmp/flux.json", [slot])

    assert calls[0]["cmd"][4:] == ["workflow", "vary", "/tmp/flux.json", "--slot", slot]


def test_vary_workflow_still_checks_a_slot_just_under_the_spawn_limit(no_spawn):
    """The abstain threshold is a ceiling, not a hole — just under it still checks.

    Guards against the size gate silently swallowing the ordinary contract check
    for any value large enough to matter.
    """
    # Well-formed JSON, not an array, at a length the gate still inspects.
    value = '"' + "p" * (params._MAX_PRECHECKED_SLOT_BYTES - 100) + '"'
    with pytest.raises(server.ComfyCliError, match="got str"):
        server.vary_workflow("/tmp/flux.json", [f"6.text={value}"])


# --- structured {address, value} slot items -------------------------------
#
# The second accepted form for both tools' slot lists, added because the string
# form's value portion is JSON-parsed by comfy-cli with a literal-string
# fallback — so `"6.text=true"` sets a BOOLEAN and there is no way to spell the
# literal string "true" through it. The structured form is lossless precisely
# because `json.dumps` is the inverse of the `json.loads` comfy-cli runs, so
# these tests assert on the exact argv the encode produces.


def test_set_workflow_slot_structured_override_serializes_to_addr_json(patched_run):
    """A structured override reaches argv as `ADDR=<json>`, positionally as before."""
    calls = patched_run(envelope(data={"modified": True}))

    result = server.set_workflow_slot(
        "/tmp/flux.json", [{"address": "6.text", "value": "a cat"}]
    )
    assert result == {"modified": True}

    assert calls[-1]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "/tmp/flux.json",
        '6.text="a cat"',  # JSON-encoded, so comfy-cli's json.loads returns a str
        "--stdout",
    ]


@pytest.mark.parametrize(
    "value, encoded",
    [
        ("true", '"true"'),  # the footgun: a string that LOOKS like a boolean
        ("42", '"42"'),  # ...and one that looks like a number
        (True, "true"),  # a real boolean still arrives as one
        (42, "42"),
        (1.5, "1.5"),
        (None, "null"),
        ("a lighthouse at dawn, oil painting", '"a lighthouse at dawn, oil painting"'),
        ({"k": [1, 2]}, '{"k": [1, 2]}'),
    ],
    ids=["str-true", "str-42", "bool", "int", "float", "null", "comma-str", "object"],
)
def test_set_workflow_slot_structured_override_preserves_value_type(
    patched_run, value, encoded
):
    """Round-trip fidelity: `json.dumps` here, `json.loads` in comfy-cli.

    The string form cannot express the first two rows at all — `"6.text=true"`
    parses to the boolean and `"6.text=42"` to the int — which is the whole
    reason this form exists.
    """
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot("/tmp/flux.json", [{"address": "6.text", "value": value}])

    override = calls[-1]["cmd"][4:][3]
    assert override == f"6.text={encoded}"
    # ...and comfy-cli's own parse of that entry yields the value we were sent.
    assert json.loads(override.partition("=")[2]) == value


def test_set_workflow_slot_string_override_still_passes_through_verbatim(patched_run):
    """Regression: the string form is untouched, coercion and all.

    `"6.text=true"` still reaches argv byte-for-byte (and so still sets the
    BOOLEAN downstream) — the structured form is additive, not a rewrite of the
    existing one.
    """
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot("/tmp/flux.json", ["6.text=true", "3.seed=42"])

    assert calls[-1]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "/tmp/flux.json",
        "6.text=true",
        "3.seed=42",
        "--stdout",
    ]


def test_set_workflow_slot_mixes_string_and_structured_overrides(patched_run):
    """Each entry is rendered on its own, so a mixed list is fine — and ordered."""
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot(
        "/tmp/flux.json",
        [
            "3.seed=42",
            {"address": "6.text", "value": "true"},
            "4.ckpt=sd-xl",
        ],
        stdout=False,
    )

    assert calls[-1]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "/tmp/flux.json",
        "3.seed=42",
        '6.text="true"',
        "4.ckpt=sd-xl",
    ]


def test_vary_workflow_structured_slot_serializes_values_as_json_array(patched_run):
    """`values` is encoded as the JSON array comfy-cli's `--slot` contract wants.

    That also dissolves the string form's quoting gotcha: the comma inside the
    first prompt stays inside it with nothing for the caller to escape.
    """
    calls = patched_run(envelope(data={"count": 2}))

    result = server.vary_workflow(
        "/tmp/flux.json",
        [
            {"address": "3.seed", "values": [1, 2]},
            {
                "address": "1.prompt",
                "values": ["a lighthouse at dawn, oil painting", "a cabin"],
            },
        ],
    )
    assert result == {"count": 2}

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        "3.seed=[1, 2]",
        "--slot",
        '1.prompt=["a lighthouse at dawn, oil painting", "a cabin"]',
    ]


def test_vary_workflow_structured_slot_preserves_element_types(patched_run):
    """Same fidelity guarantee per element: `["true"]` is not `[true]`."""
    calls = patched_run(envelope(data={"count": 2}))

    server.vary_workflow(
        "/tmp/flux.json", [{"address": "6.text", "values": ["true", 1]}]
    )

    slot = calls[0]["cmd"][4:][-1]
    assert slot == '6.text=["true", 1]'
    assert json.loads(slot.partition("=")[2]) == ["true", 1]


def test_vary_workflow_mixes_string_and_structured_slots(patched_run):
    """A mixed list works, and each entry keeps its own `--slot` flag and order."""
    calls = patched_run(envelope(data={"count": 2}))

    server.vary_workflow(
        "/tmp/flux.json",
        ["3.seed=[1,2]", {"address": "6.text", "values": ["a cat", "a dog"]}],
    )

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        "3.seed=[1,2]",  # verbatim, straight through the existing pre-check
        "--slot",
        '6.text=["a cat", "a dog"]',
    ]


def test_vary_workflow_rejects_empty_structured_values(no_spawn):
    """An empty `values` produces zero variants, so name it instead of forwarding.

    comfy-cli ZIPS the lists: an empty one silently makes the whole sweep empty
    and the run "succeeds" having generated nothing.
    """
    with pytest.raises(server.ComfyCliError) as excinfo:
        server.vary_workflow(
            "/tmp/flux.json",
            [
                {"address": "3.seed", "values": [1, 2]},
                {"address": "6.text", "values": []},
            ],
        )

    message = str(excinfo.value)
    assert "slots[1]" in message  # names the offending entry...
    assert "6.text" in message  # ...and its address
    assert "empty" in message


@pytest.mark.parametrize(
    "address, marker",
    [
        ("--stdout", "leading '-'"),
        ("-6.text", "leading '-'"),
        ("6.text=x", "contains '='"),
        ("6.te\0xt", "embedded NUL"),
        ("", "empty"),
        ("   ", "empty"),
    ],
    ids=["option-like", "dash-leading", "equals", "nul", "empty", "whitespace-only"],
)
def test_structured_slot_address_is_guarded(no_spawn, address, marker):
    """The structured `address` carries every constraint the string entry did.

    It BECOMES the portion before the first `=` of an argv entry, so it gets the
    same `_reject_option_like` / `_reject_nul` pair the string path runs — named
    against the `address` field the caller actually sent — plus the two checks
    only this form can express: an empty address (a bare `=value` entry) and an
    embedded `=` (which would re-split, silently shifting the value).
    """
    for call in (
        lambda: server.set_workflow_slot(
            "/tmp/flux.json", [{"address": address, "value": "x"}]
        ),
        lambda: server.vary_workflow(
            "/tmp/flux.json", [{"address": address, "values": [1]}]
        ),
    ):
        with pytest.raises(server.ComfyCliError) as excinfo:
            call()
        message = str(excinfo.value)
        assert marker in message
        assert "address" in message  # the named parameter, not a generic entry


def test_structured_slot_address_is_stripped(patched_run):
    """Surrounding whitespace is trimmed rather than smuggled into the argv entry."""
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot("/tmp/flux.json", [{"address": "  6.text\n", "value": 1}])

    assert calls[-1]["cmd"][4:][3] == "6.text=1"


def test_structured_slot_value_must_be_json_data(no_spawn):
    """A Python object with no JSON form is named, not raised as a bare TypeError.

    Unreachable over MCP (everything there is JSON already); this covers the
    in-process caller and keeps the tool's error type uniform.
    """
    with pytest.raises(server.ComfyCliError, match="not JSON data"):
        server.set_workflow_slot(
            "/tmp/flux.json", [{"address": "6.text", "value": {1}}]
        )

    with pytest.raises(server.ComfyCliError, match="not JSON data"):
        server.vary_workflow("/tmp/flux.json", [{"address": "6.text", "values": [{1}]}])


def test_structured_slot_item_requires_both_fields(no_spawn):
    """A half-filled structured item is a validation error, not a silent default."""
    for call in (
        lambda: server.set_workflow_slot("/tmp/flux.json", [{"address": "6.text"}]),
        lambda: server.set_workflow_slot("/tmp/flux.json", [{"value": "a cat"}]),
        lambda: server.vary_workflow("/tmp/flux.json", [{"address": "6.text"}]),
        lambda: server.vary_workflow("/tmp/flux.json", [{"values": [1]}]),
    ):
        with pytest.raises(ValidationError):
            call()


def test_slot_tools_advertise_the_union_item_type():
    """The MCP schema offers BOTH forms per item, so a client can send either.

    FastMCP 4 dereferences the model into ``anyOf`` by default. The structured
    object shape is what makes that form discoverable to a client.
    """
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}

    for name, param, value_field in (
        ("set_workflow_slot", "overrides", "value"),
        ("vary_workflow", "slots", "values"),
    ):
        schema = tools[name].parameters
        item = schema["properties"][param]["items"]
        assert {"type": "string"} in item["anyOf"]
        structured = next(
            option for option in item["anyOf"] if option.get("type") == "object"
        )
        assert structured["properties"]["address"]["type"] == "string"
        assert set(structured["required"]) == {"address", value_field}

    # No existing parameter was renamed or dropped.
    assert set(tools["set_workflow_slot"].parameters["properties"]) == {
        "workflow_path",
        "overrides",
        "stdout",
    }
    assert set(tools["vary_workflow"].parameters["properties"]) == {
        "workflow_path",
        "slots",
        "out_dir",
    }


def test_structured_slot_items_deserialize_through_the_mcp_boundary(patched_run):
    """End-to-end through FastMCP: a JSON dict lands as the model, not a stray dict.

    The direct-call tests above go through the same coercion helper but not
    through FastMCP's own argument validation, which is what a real client
    actually exercises — and it is the union that has to hold there.
    """
    calls = patched_run(envelope(data={"modified": True}))

    asyncio.run(
        server.mcp.call_tool(
            "set_workflow_slot",
            {
                "workflow_path": "/tmp/flux.json",
                "overrides": ["3.seed=42", {"address": "6.text", "value": "true"}],
            },
        )
    )
    assert calls[-1]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "/tmp/flux.json",
        "3.seed=42",
        '6.text="true"',
        "--stdout",
    ]

    asyncio.run(
        server.mcp.call_tool(
            "vary_workflow",
            {
                "workflow_path": "/tmp/flux.json",
                "slots": [{"address": "6.text", "values": ["a cat", "a dog"]}],
            },
        )
    )
    # Indexed by verb, not position: set_workflow_slot above now probes
    # `workflow slots` before writing, so positional indices are not stable.
    assert [c["cmd"][4:] for c in calls if c["cmd"][5:6] == ["vary"]] == [
        [
            "workflow",
            "vary",
            "/tmp/flux.json",
            "--slot",
            '6.text=["a cat", "a dog"]',
        ]
    ]


def test_vary_workflow_string_form_still_allows_an_empty_array(patched_run):
    """The empty-`values` rejection is scoped to the STRUCTURED form only.

    `'6.text=[]'` is a valid JSON array, so it rides through the existing
    pre-check untouched and reaches comfy-cli exactly as it always did. Worth
    locking in: the structured form's extra rejection is a nudge on a shape that
    yields zero variants, NOT the removal of a way to send an empty list.
    """
    calls = patched_run(envelope(data={"count": 0}))

    server.vary_workflow("/tmp/flux.json", ["6.text=[]"])

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        "6.text=[]",
    ]


# --- QA 0.8.0: mis-paired slot names --------------------------------------
# `comfy workflow slots` zips a node's input NAMES (from object_info) positionally
# against its `widgets_values`. A node whose object_info under-reports its inputs
# shifts every later pairing. Verbatim from `comfy workflow slots` on
# api_minimax_h3_t2v (comfy-cli 1.15.0) — reproduced with no MCP involved, so the
# bug is comfy-cli's; what this server can do is refuse to relay it as fact.
_MISPAIRED = {
    "workflow": "/tmp/h3.json",
    "count": 4,
    "slots": [
        {
            "address": "8.filename_prefix",
            "name": "filename_prefix",
            "type": "STRING",
            "current_value": "video/MiniMax_H3_t2v",
        },
        {
            "address": "8.format",
            "name": "format",
            "type": "COMBO",
            "current_value": "auto",
        },
        {
            "address": "23.seed",
            "name": "seed",
            "type": "INT",
            "current_value": "MiniMax H3",
        },
        {
            "address": "23.watermark",
            "name": "watermark",
            "type": "BOOLEAN",
            "current_value": "768P",
        },
    ],
}


def test_list_workflow_slots_flags_mispaired_slots(patched_run):
    """The INT holding a title and the BOOLEAN holding a resolution are named."""
    patched_run(envelope(data=_MISPAIRED))

    result = server.list_workflow_slots("/tmp/h3.json")

    assert result["suspect_slots"] == ["23.seed", "23.watermark"]
    assert "do NOT belong together" in result["warning"]
    # Additive: the original slots are all still there, plus a per-slot flag.
    assert len(result["slots"]) == 4
    assert result["slots"][2]["pairing_suspect"] is True
    # And the genuinely fine ones are untouched.
    assert "pairing_suspect" not in result["slots"][0]


def test_list_workflow_slots_leaves_clean_payloads_alone(patched_run):
    """No false positives: a numeric string in an INT is normal, not corruption."""
    patched_run(
        envelope(
            data={
                "slots": [
                    {
                        "address": "3.seed",
                        "name": "seed",
                        "type": "INT",
                        "current_value": "42",
                    },
                    {
                        "address": "6.text",
                        "name": "text",
                        "type": "STRING",
                        "current_value": "a cat",
                    },
                    {
                        "address": "4.ckpt",
                        "name": "ckpt",
                        "type": "COMBO",
                        "current_value": "sd.safetensors",
                    },
                ]
            }
        )
    )

    result = server.list_workflow_slots("/tmp/ok.json")

    assert "suspect_slots" not in result
    assert "warning" not in result
    # And the payload is returned WHOLE: absence of the new keys would still
    # hold if the flagger had mutated the slots it was handed.
    assert result["slots"] == [
        {"address": "3.seed", "name": "seed", "type": "INT", "current_value": "42"},
        {
            "address": "6.text",
            "name": "text",
            "type": "STRING",
            "current_value": "a cat",
        },
        {
            "address": "4.ckpt",
            "name": "ckpt",
            "type": "COMBO",
            "current_value": "sd.safetensors",
        },
    ]


def test_set_workflow_slot_refuses_a_mispaired_target(patched_run):
    """The half that destroys data: the write would land in a DIFFERENT field.

    comfy-cli reports success, `applied` names the requested address, and
    validate_workflow still says valid: true — so nothing looks wrong afterwards.
    Failing closed here is the only point it can be caught.
    """
    patched_run(envelope(data=_MISPAIRED))

    with pytest.raises(server.ComfyCliError) as exc:
        server.set_workflow_slot("/tmp/h3.json", ["23.seed=42"])

    message = str(exc.value)
    assert "refusing to set 23.seed" in message
    assert "would land in a different field" in message


def test_set_workflow_slot_allows_an_unaffected_slot(patched_run):
    """Only the mis-paired addresses are blocked — the rest of the graph still works."""
    calls = patched_run(envelope(data=_MISPAIRED))

    server.set_workflow_slot("/tmp/h3.json", ["8.filename_prefix=out"])

    assert calls[-1]["cmd"][4:][:2] == ["workflow", "set-slot"]
