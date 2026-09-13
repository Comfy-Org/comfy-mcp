"""Param/slot marshaling into comfy-cli argv — the last stop before ``comfy generate`` / ``run-template`` / ``workflow set-slot`` / ``workflow vary``.

Leaf module over :mod:`comfy_mcp.errors` and :mod:`comfy_mcp.argv`: it owns
turning a caller's ``params``/``overrides``/``slots`` dict-or-string input into
the ``--name=value`` / ``--param=KEY=VALUE`` / ``ADDR=VALUE`` /
``ADDR=[v1,v2,...]`` tokens those four comfy-cli surfaces take —
:func:`_generate_param_args` (``comfy generate``), :func:`_run_template_param_args`
(``comfy run-template``), :func:`_slot_override_arg` (``comfy workflow
set-slot``), :func:`_slot_variants_arg` (``comfy workflow vary``) — plus the
shared key/model-target validation (:func:`_validate_param_key`,
:func:`_validate_generate_model`) and the structured-input coercion
(:func:`_as_slot_model`) shared across them. Nothing here imports ``server``.

:class:`SlotOverride` and :class:`SlotVariants` are this module's two PUBLIC
names: ``server`` name-imports them (``from .params import SlotOverride,
SlotVariants``) rather than reaching them module-qualified, per the carve-out
AGENTS.md documents for public exception and model types — they ride directly
in ``@mcp.tool()`` signatures (``set_workflow_slot``'s ``overrides``,
``vary_workflow``'s ``slots``), and with ``from __future__ import
annotations`` those annotations are resolved lazily against the defining
module's globals, so a module-qualified ``params.SlotOverride`` in the
signature would change the exported MCP tool schema's ``$defs`` — the schema
gate this module's own extraction was reviewed under. Everything else here is
private and reached as ``params._name`` — there is no other public name in
this module.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from . import argv
from .errors import ComfyCliError

# comfy-cli reserves these words as `comfy generate` SUB-ACTIONS (its own
# list / schema / refresh / upload / resume / consent verbs) rather than model
# aliases. This tool's contract is "run this partner MODEL", so a reserved word
# is refused instead of silently dispatching a different verb — `consent` in
# particular is the spend gate's own configuration surface.
_GENERATE_RESERVED_TARGETS = frozenset(
    {"list", "schema", "refresh", "upload", "resume", "consent"}
)

# comfy-cli treats these `comfy generate` flags as RUN-level rather than model
# inputs (its `_separate_meta_flags`): they change how the call runs, not what
# is generated. They are refused inside `params` so a "model parameter" can
# never silently retarget the run — above all `yes`, which would otherwise be a
# second, undocumented way to grant spend consent behind `confirm_spend`'s back,
# and `json` / `async`, which would break this tool's result contract.
_GENERATE_META_FLAGS = frozenset(
    {
        "download",
        "async",
        "json",
        "timeout",
        "api-key",
        "emit-workflow",
        "output-prefix",
        "yes",
    }
)


def _validate_param_key(
    key: str, *, empty_msg: str, invalid_msg: str, nul_label: str
) -> None:
    """Shared key-shape gate for the two argv param marshalers.

    Order is load-bearing and identical in both callers: empty-or-leading-dash,
    then ``=``-or-whitespace, then NUL. Each caller passes its own fully rendered
    messages so its error text survives byte-for-byte, and keeps its own comment
    for WHY the ``=``/whitespace check matters for its argv shape. If the two
    gates ever need to diverge, split this back into the callers rather than
    growing parameters here.
    """
    if not key or key.startswith("-"):
        raise ComfyCliError(empty_msg)
    if "=" in key or any(ch.isspace() for ch in key):
        raise ComfyCliError(invalid_msg)
    argv._reject_nul(nul_label, key)


def _generate_param_args(params: dict[str, Any]) -> list[str]:
    """Marshal per-model ``params`` into ``comfy generate`` ``--name=value`` tokens.

    ``comfy generate`` takes a model's inputs as schema-driven flags whose names
    and types come from that model's OWN schema, so this wrapper neither knows
    nor validates them: each pair is forwarded verbatim for comfy-cli to accept
    or reject. The ``--name=value`` form (rather than two argv tokens) means a
    value that begins with ``-`` is read as the value instead of being
    mis-parsed as the next option.

    Conversions are spelling-only, so comfy-cli's parser sees the form it
    expects: ``None`` drops the flag entirely (rather than sending the string
    "None"), bools become ``true`` / ``false``, and list / dict values are
    JSON-encoded — what its array parser accepts. Everything else is
    ``str()``-rendered.
    """
    param_args: list[str] = []
    for name, value in params.items():
        # `=`/whitespace in a NAME is argv-integrity here: comfy-cli splits
        # `--<body>` at the FIRST `=`, so a key carrying its own `=` would land
        # as a run-level flag, smuggling past the meta-flag check below (which
        # only ever sees the whole key).
        _validate_param_key(
            name,
            empty_msg=(
                f"invalid parameter name: {name!r} — expected a model parameter "
                "name (e.g. 'prompt'), not an empty or option-like value."
            ),
            invalid_msg=(
                f"invalid parameter name: {name!r} — a parameter name cannot "
                "contain '=' or whitespace. Pass the value as the dict value, "
                "not inside the key."
            ),
            nul_label=f"parameter name {name!r}",
        )
        # Compare hyphen-normalized so `api_key` / `emit_workflow` are caught
        # too; agents naturally spell CLI flags with underscores. Case is NOT
        # normalized: comfy-cli matches its run-level flags case-sensitively in
        # lower case, so `Json` can never reach one, while a model's schema
        # flags come verbatim from its OpenAPI property names and may legitimately
        # be capitalized — folding case here would refuse a real parameter to
        # block an unreachable one.
        if name.replace("_", "-") in _GENERATE_META_FLAGS:
            raise ComfyCliError(
                f"`{name}` is a run-level `comfy generate` flag, not a model "
                "parameter. Use the tool argument that covers it "
                "(partner_generate: confirm_spend for --yes, out_path for "
                "--download, timeout_seconds for --timeout; "
                "emit_partner_workflow: out_path for --emit-workflow); the "
                "remaining run-level flags are not forwarded by these tools, so "
                "use comfy-cli directly for those."
            )
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        argv._reject_nul(f"value for parameter {name!r}", rendered)
        param_args.append(f"--{name}={rendered}")
    return param_args


def _validate_generate_model(model: str) -> None:
    """Refuse a ``comfy generate`` target that is not usable as a partner model.

    Shared by :func:`partner_generate` and :func:`emit_partner_workflow` so the
    two cannot drift on what they accept — they hand the SAME first positional to
    the SAME comfy-cli verb, and a guard that held on only one of them would be a
    guard the other could be used to walk around (``consent`` most of all: it is
    the spend gate's own configuration surface).
    """
    if not model:
        raise ComfyCliError(
            f"invalid model: {model!r} — expected a partner model alias "
            "(e.g. 'flux-pro'), not an empty value."
        )
    # A leading-dash target is read by comfy-cli as an option rather than a
    # model (the same guard job(action="watch") applies to prompt_id).
    argv._reject_option_like(
        "model", model, expected="a partner model alias (e.g. 'flux-pro')"
    )
    if model in _GENERATE_RESERVED_TARGETS:
        raise ComfyCliError(
            f"invalid model: {model!r} is a `comfy generate` sub-action, not a "
            "partner model. Use comfy-cli directly for those verbs."
        )
    argv._reject_nul("model", model)


def _run_template_param_args(params: dict[str, Any]) -> list[str]:
    """Marshal template ``params`` into ``comfy run-template`` ``--param=KEY=VALUE`` tokens.

    ``comfy run-template`` fills a template's parameterized slots: KEY is a slot
    address (``6.text``) or a unique slot name (``prompt``), and VALUE parses as
    JSON with a string fallback. Each value is JSON-encoded so its Python type
    round-trips exactly — ``42`` stays an int, the string ``"42"`` stays a
    string, ``True`` becomes ``true``, and lists/dicts become JSON arrays/objects
    — rather than leaning on the bare-string fallback, which would coerce a
    numeric-looking string to a number. ``None`` drops the pair entirely. The
    single ``--param=KEY=VALUE`` token (comfy-cli splits on the FIRST ``=``)
    keeps a value that contains ``=`` or begins with ``-`` intact.
    """
    param_args: list[str] = []
    for key, value in params.items():
        # `=` is the load-bearing check here: comfy-cli splits the `--param`
        # value on its FIRST `=` to separate slot key from value. Whitespace is
        # refused for a weaker reason — KEY rides inside the single
        # `--param=KEY=VALUE` token so it is never argv-ambiguous, but a clear
        # error beats the engine's "matches no slot".
        _validate_param_key(
            key,
            empty_msg=(
                f"invalid param key: {key!r} — expected a slot address (e.g. "
                "'6.text') or a slot name (e.g. 'prompt'), not an empty or "
                "option-like value."
            ),
            invalid_msg=(
                f"invalid param key: {key!r} — a slot key cannot contain '=' or "
                "whitespace. Pass the value as the dict value, not in the key."
            ),
            nul_label=f"param key {key!r}",
        )
        if value is None:
            continue
        # json.dumps escapes a NUL inside a string value (so it can't crash
        # subprocess the way a raw NUL in the KEY would), but a NUL slot value is
        # never intentional — refuse it explicitly, matching partner_generate.
        # Checked recursively: a NUL nested in a list/dict is the same mistake
        # and would otherwise land as a literal `\u0000` in the filled graph.
        argv._reject_nul_deep(f"value for param {key!r}", value)
        rendered = json.dumps(value)
        param_args.append(f"--param={key}={rendered}")
    return param_args


class SlotOverride(BaseModel):
    """One ``set_workflow_slot`` override, as structured data instead of a string.

    The reason this form exists: comfy-cli splits an override on its first ``=``
    and runs the value portion through ``json.loads``, falling back to the
    literal string when that fails. So the ``"ADDR=VALUE"`` string form
    COERCES — ``"6.text=true"`` sets the boolean ``true``, ``"6.text=123"`` sets
    the integer ``123``, and there is no way to spell the literal strings
    ``"true"`` / ``"123"`` through it. Sending the value as DATA is lossless in
    both directions: this server JSON-encodes it, and comfy-cli's ``json.loads``
    decodes exactly what was sent.
    """

    address: str = Field(
        description=(
            "The slot address to set, exactly as `list_workflow_slots` reports "
            "it in a slot's `address` field (e.g. '6.text')."
        )
    )
    value: Any = Field(
        description=(
            "The value to set, as JSON data. Its type is preserved exactly: a "
            "string stays a string (even 'true' or '42'), a number stays a "
            "number, a boolean stays a boolean."
        )
    )


class SlotVariants(BaseModel):
    """One ``vary_workflow`` slot — an address and the values to sweep over it.

    The structured counterpart to the ``"ADDR=[v1,v2,...]"`` string form, and
    the same lossless-vs-coercing trade-off as :class:`SlotOverride`: comfy-cli
    requires the value portion to parse to a JSON array, so ``values`` is sent
    as one and every element keeps its type. It also removes the quoting
    footgun the string form carries — a comma inside a prompt no longer has to
    be hand-quoted to stay part of its value.
    """

    address: str = Field(
        description=(
            "The slot address to vary, exactly as `list_workflow_slots` reports "
            "it in a slot's `address` field (e.g. '3.seed')."
        )
    )
    values: list[Any] = Field(
        description=(
            "The values to sweep over this address, as JSON data. comfy-cli "
            "ZIPS the lists across slots, so every slot's list must be the same "
            "length. Must be non-empty."
        )
    )


def _slot_address_arg(label: str, address: str) -> str:
    """Validate a structured slot item's ``address`` and return it normalized.

    A structured item's address becomes the portion before the first ``=`` of an
    argv entry, so it inherits every constraint the string form's entry already
    carries — hence the same ``argv._reject_option_like`` / ``argv._reject_nul`` pair the
    string path runs, applied to the address specifically so the error names the
    field the caller actually sent.

    Two checks are new here because only the structured form can express them.
    An empty (or all-whitespace) address would produce a bare ``"=value"`` entry
    that comfy-cli splits into an address of ``""``; and an address containing
    ``=`` would silently re-split, so ``{"address": "6.text=x", "value": "y"}``
    would reach the engine as address ``6.text`` with value ``x="y"`` rather
    than failing. Both are caller mistakes worth naming rather than forwarding.

    The dash-leading rejection is defense in depth: a node id is non-negative
    where ``list_workflow_slots`` surfaces it, so no reachable address starts
    with ``-`` (see :func:`argv._reject_option_like`'s note on slot ADDRs). It is
    guarded anyway for parity with the string path.
    """
    argv._reject_nul(f"{label} address", address)
    stripped = address.strip()
    expected = (
        "a '<node_id>.<input>' address as `list_workflow_slots` reports it "
        "(e.g. '6.text')"
    )
    if not stripped:
        raise ComfyCliError(f"invalid {label} address: empty — expected {expected}")
    if "=" in stripped:
        raise ComfyCliError(
            f"invalid {label} address: {argv._clip_for_error(stripped)} contains "
            f"'=' — the address is only the part BEFORE the first '=', and the "
            f"value belongs in its own field; expected {expected}"
        )
    return argv._reject_option_like(f"{label} address", stripped, expected=expected)


def _slot_value_json(label: str, value: Any) -> str:
    """JSON-encode a structured slot value, naming a non-encodable one.

    Everything arriving over MCP is JSON already, so this can only fire for a
    direct in-process caller that passed a Python object with no JSON form (a
    ``set``, a ``datetime``). Naming it beats letting ``TypeError`` escape a
    tool that reports every other input mistake as :class:`ComfyCliError`.

    Note what encoding does to the NUL guard, since the asymmetry is deliberate:
    the string form refuses a NUL because a raw one cannot ride in argv at all,
    while ``json.dumps`` escapes it to ``\\u0000`` — so a structured value may
    carry one, and comfy-cli's ``json.loads`` decodes it back on the far side.
    That is the encoding doing its job (argv stays clean), not a hole in the
    guard: the reason to refuse it never applies to an encoded value.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ComfyCliError(
            f"invalid {label}: a {type(value).__name__} is not JSON data — send "
            "a string, number, boolean, null, array, or object."
        ) from exc


_SlotModel = TypeVar("_SlotModel", bound=BaseModel)


def _as_slot_model(item: Any, model: type[_SlotModel]) -> _SlotModel:
    """Coerce one structured slot item to its model.

    Over MCP, FastMCP has already validated the item into ``model``. A plain
    mapping only reaches here from an in-process caller (this module's own
    tests, a script importing the tool), so it is validated through the same
    model rather than read key-by-key — one definition of the shape, one set of
    error messages.
    """
    return item if isinstance(item, model) else model.model_validate(item)


def _slot_override_arg(item: str | SlotOverride) -> str:
    """Render one ``set_workflow_slot`` override as the ``ADDR=VALUE`` argv entry.

    A string passes through byte-for-byte — that is the pre-existing form and
    its coercing-but-established behavior is unchanged. A structured item is
    serialized with ``json.dumps``, which is exactly what makes it lossless:
    comfy-cli's ``json.loads`` is the inverse, so ``"true"`` arrives as the
    string ``"true"`` (encoded ``"true"`` with its quotes) and ``42`` as the
    integer.
    """
    if isinstance(item, str):
        return item
    override = _as_slot_model(item, SlotOverride)
    address = _slot_address_arg("override", override.address)
    return f"{address}={_slot_value_json('override value', override.value)}"


# A slot entry's value portion is fed to `json.loads` by comfy-cli, so the two
# examples that make the contract concrete: the mechanical shape, and the one
# that trips callers up — a string value containing a comma, which is ONLY a
# single value when it is JSON-quoted.
_SLOT_VALUE_EXAMPLE = "3.seed=[1,2,3]"
_SLOT_QUOTED_EXAMPLE = '6.text=["a lighthouse at dawn, oil painting", "a cabin"]'

# Past this, a slot cannot reach comfy-cli at all: Linux caps a SINGLE argv
# entry at `MAX_ARG_STRLEN` (32 pages = 128 KiB) regardless of the roomier total
# `ARG_MAX`, so the spawn fails before comfy-cli parses anything. That makes this
# a bound the pre-check can honor without inventing a policy — it is the engine's
# own reachability limit, not a taste call about sweep size. The kernel counts
# BYTES, so this must be measured after encoding: a value of multibyte
# characters is several times its character count on the wire.
_MAX_PRECHECKED_SLOT_BYTES = 128 * 1024


def _reject_non_json_array_slot(index: int, slot: str) -> None:
    """Reject a ``vary_workflow`` slot whose value is not a JSON array.

    comfy-cli splits each ``--slot`` entry on its first ``=`` and runs the value
    portion through :func:`json.loads`, *falling back to the literal string* when
    that fails — then rejects anything that did not parse to a list. So the
    natural first attempt at a text sweep,
    ``6.text=[a lighthouse at dawn, oil painting]``, is not a two-element list at
    all: it is invalid JSON, comes back as one bare string, and dies as
    ``value must be a JSON array (got str)`` with nothing pointing at the missing
    quotes.

    Checking here rather than passing the failure through buys two things. The
    message can name WHICH entry was malformed and show the quoted form that
    fixes it (comfy-cli sees only the value it already failed to parse), and the
    check lands before the subprocess: ``comfy workflow vary`` loads the file and
    fetches ``object_info`` from the live ComfyUI *before* it parses ``--slot``,
    so with the server down a malformed slot surfaces as a connection failure
    that hides the real mistake entirely.

    This mirrors comfy-cli's own parse exactly — ``json.loads`` on the value
    portion, accept only a ``list`` — so it can only refuse input comfy-cli would
    also refuse — with one deliberate exception. A failure that is a property of
    the PARSING PROCESS rather than of the input (recursion depth, an interpreter
    limit like ``sys.get_int_max_str_digits``) says nothing about how comfy-cli's
    own fresh subprocess will fare, so those are handed to the engine untouched
    rather than guessed at. Only a genuine syntax error — a
    :class:`json.JSONDecodeError` — is refused here.

    A value too long to survive ``execve`` abstains the same way: see
    :data:`_MAX_PRECHECKED_SLOT_BYTES`. That keeps the parse — the one piece of
    real work this thin wrapper does in-process rather than in the disposable
    subprocess — bounded by what the engine could actually have received.
    """
    fix = (
        f"quote each value as JSON — e.g. '{_SLOT_VALUE_EXAMPLE}', or "
        f"'{_SLOT_QUOTED_EXAMPLE}' when a value contains a comma or spaces "
        "(an unquoted comma splits the value, and unquoted text is not JSON)"
    )
    if "=" not in slot:
        raise ComfyCliError(
            f"invalid slots[{index}] {argv._clip_for_error(slot)}: expected an "
            f"'ADDR=[v1,v2,...]' string whose value is a JSON array — {fix}"
        )
    # Measured over the whole entry, encoded: `slot` IS the argv string, and the
    # kernel's limit is in bytes. `surrogatepass` because a lone surrogate can
    # arrive over the wire and this guard must not be the thing that raises.
    if len(slot.encode("utf-8", "surrogatepass")) > _MAX_PRECHECKED_SLOT_BYTES:
        # Too long to survive `execve` (see `_MAX_PRECHECKED_SLOT_BYTES`), so
        # there is no verdict worth computing: the spawn fails before comfy-cli
        # reads it either way. Parsing it anyway would do real work in the
        # long-lived parent for a value that cannot land — allocating an object
        # graph several times its size, and on an interpreter without
        # `sys.get_int_max_str_digits` converting a multi-million-digit literal
        # in quadratic time. Abstaining costs nothing: this is the same
        # engine-decides path the other unparseable cases take, so it cannot
        # over-reject.
        return
    addr, _, raw = slot.partition("=")
    # Clipped like every other caller-supplied fragment here: the address is the
    # portion BEFORE the first `=` and is just as caller-sized as the value, so
    # echoing it raw would hand back a multi-KB message and defeat the bound.
    addr = argv._clip_for_error(addr.strip())
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ComfyCliError(
            f"invalid slots[{index}] for address {addr}: value must be a JSON "
            f"array, but {argv._clip_for_error(raw)} is not valid JSON — {fix}"
        ) from None
    except (ValueError, RecursionError):
        # NOT a syntax error — the input is well-formed JSON that THIS
        # interpreter declined to build: nesting deeper than the stack left us
        # (`RecursionError`), or an integer literal over
        # `sys.get_int_max_str_digits` (a plain `ValueError`, not a
        # `JSONDecodeError`, since 3.11). Both are limits of the process doing
        # the parsing, and this one parses several frames down from the MCP
        # handler with whatever limits this interpreter was started with, while
        # comfy-cli parses in a fresh subprocess of its own. Refusing here would
        # reject values the engine accepts and break the invariant above, so
        # abstain and let the engine's parse be the verdict.
        return
    if not isinstance(value, list):
        raise ComfyCliError(
            f"invalid slots[{index}] for address {addr}: value must be a JSON "
            f"array, got {type(value).__name__} — wrap a single value in a "
            f"one-element array ('3.seed=[42]'), and {fix}"
        )


def _slot_variants_arg(index: int, item: str | SlotVariants) -> str:
    """Render one ``vary_workflow`` slot as the ``ADDR=[v1,v2,...]`` argv entry.

    A string passes through byte-for-byte, into the existing
    :func:`_reject_non_json_array_slot` pre-check. A structured item is
    serialized with ``json.dumps(values)`` — which IS the wire form comfy-cli
    wants, since it requires the value portion to parse to a JSON array — so the
    quoting gotcha the string form carries cannot arise: a comma inside a prompt
    stays inside its value with nothing for the caller to escape.

    An empty ``values`` is refused here rather than forwarded. comfy-cli zips
    the lists, so an empty one yields zero variants: the run "succeeds" having
    done nothing, which reads as a broken sweep rather than as the input mistake
    it is.
    """
    if isinstance(item, str):
        return item
    variants = _as_slot_model(item, SlotVariants)
    address = _slot_address_arg("slot", variants.address)
    if not variants.values:
        raise ComfyCliError(
            f"invalid slots[{index}] for address {argv._clip_for_error(address)}: "
            "`values` is empty — comfy-cli zips the value lists, so an empty "
            "one produces zero variants. Give at least one value (and the same "
            "count as every other slot)."
        )
    return f"{address}={_slot_value_json(f'slots[{index}] values', variants.values)}"
