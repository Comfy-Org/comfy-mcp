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
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import envelope
from pydantic import ValidationError

from comfy_local_mcp import server


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


def test_set_workflow_slot_argv_default_stdout(patched_run):
    """Default: positional ADDR=VALUE overrides + trailing --stdout (non-destructive)."""
    calls = patched_run(envelope(data={"modified": True}))

    result = server.set_workflow_slot(
        "/tmp/flux.json", ["6.text=a red bicycle", "3.seed=42"]
    )
    assert result == {"modified": True}

    assert calls[0]["cmd"][4:] == [
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

    cmd = calls[0]["cmd"]
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


def test_workflow_path_guard_allows_dot_slash_dash_name(patched_run):
    """The documented escape hatch actually works: `./-flux.json` is not refused."""
    calls = patched_run(envelope(data={"modified": True}))

    server.set_workflow_slot("./-flux.json", ["6.text=x"])
    server.list_workflow_notes("./-flux.json")

    assert calls[0]["cmd"][4:] == [
        "workflow",
        "set-slot",
        "./-flux.json",
        "6.text=x",
        "--stdout",
    ]
    assert calls[1]["cmd"][4:] == ["workflow", "notes", "./-flux.json"]


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

    assert calls[0]["cmd"][4:] == [
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
    depth = (server._MAX_PRECHECKED_SLOT_BYTES - len(prefix)) // 2
    nested = "[" * depth + "]" * depth
    slot = prefix + nested
    assert len(slot.encode("utf-8")) <= server._MAX_PRECHECKED_SLOT_BYTES

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
    assert len(server._clip_for_error(text)) <= server._MAX_ERROR_FIELD_CHARS


def test_clip_for_error_matches_an_unsliced_repr():
    """Slicing the source before `repr` must not change the rendered prefix.

    The pre-slice is an allocation guard, not a behavior change: every source
    character contributes at least one character to the repr, so the escaping of
    the retained prefix cannot depend on anything past the cap.

    Stated over quote-free input on purpose — the surrounding quote character is
    the one part that CAN differ, which the next test pins down.
    """
    for text in ("\x01" * 10_000, "y" * 10_000, "a, b" * 5_000):
        clipped = server._clip_for_error(text)
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
    cap = server._MAX_ERROR_FIELD_CHARS
    text = "a" * (cap + 100) + "'"

    clipped = server._clip_for_error(text)

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
    oversized = '"' + "9" * (server._MAX_PRECHECKED_SLOT_BYTES + 1) + '"'
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
    chars = (server._MAX_PRECHECKED_SLOT_BYTES // 4) + 100
    value = '"' + "\U0001f600" * chars + '"'
    slot = f"6.text={value}"
    assert len(slot) < server._MAX_PRECHECKED_SLOT_BYTES  # under, counted wrong
    assert len(slot.encode("utf-8")) > server._MAX_PRECHECKED_SLOT_BYTES  # over

    server.vary_workflow("/tmp/flux.json", [slot])

    assert calls[0]["cmd"][4:] == ["workflow", "vary", "/tmp/flux.json", "--slot", slot]


def test_vary_workflow_still_checks_a_slot_just_under_the_spawn_limit(no_spawn):
    """The abstain threshold is a ceiling, not a hole — just under it still checks.

    Guards against the size gate silently swallowing the ordinary contract check
    for any value large enough to matter.
    """
    # Well-formed JSON, not an array, at a length the gate still inspects.
    value = '"' + "p" * (server._MAX_PRECHECKED_SLOT_BYTES - 100) + '"'
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

    assert calls[0]["cmd"][4:] == [
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

    override = calls[0]["cmd"][4:][3]
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

    assert calls[0]["cmd"][4:] == [
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

    assert calls[0]["cmd"][4:] == [
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

    assert calls[0]["cmd"][4:][3] == "6.text=1"


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

    `anyOf: [string, $ref]` is what makes the structured form discoverable at
    all — a client reads the schema, not the docstring, to decide what to send.
    """
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}

    for name, param, model, value_field in (
        ("set_workflow_slot", "overrides", "SlotOverride", "value"),
        ("vary_workflow", "slots", "SlotVariants", "values"),
    ):
        schema = tools[name].input_schema
        item = schema["properties"][param]["items"]
        assert {"type": "string"} in item["anyOf"]
        assert {"$ref": f"#/$defs/{model}"} in item["anyOf"]

        model_schema = schema["$defs"][model]
        assert model_schema["properties"]["address"]["type"] == "string"
        assert sorted(model_schema["required"]) == sorted(["address", value_field])

    # No existing parameter was renamed or dropped.
    assert set(tools["set_workflow_slot"].input_schema["properties"]) == {
        "workflow_path",
        "overrides",
        "stdout",
    }
    assert set(tools["vary_workflow"].input_schema["properties"]) == {
        "workflow_path",
        "slots",
        "out_dir",
    }


def test_structured_slot_items_deserialize_through_the_mcp_boundary(patched_run):
    """End-to-end through MCPServer: a JSON dict lands as the model, not a stray dict.

    The direct-call tests above go through the same coercion helper but not
    through MCPServer's own argument validation, which is what a real client
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
    assert calls[0]["cmd"][4:] == [
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
    assert calls[1]["cmd"][4:] == [
        "workflow",
        "vary",
        "/tmp/flux.json",
        "--slot",
        '6.text=["a cat", "a dog"]',
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
