"""Ceiling on the LLM-facing payload: tool docstrings + handshake instructions.

Every MCP session ships every tool description plus INSTRUCTIONS before any
work happens. After a three-pass docstring diet, the payload is approximately
15.1k estimated tokens — a fixed tax on every conversation. This test freezes
a ceiling just above the measured size so growth is a deliberate decision (bump
the constant in the same PR, with justification), and the ceiling ratchets DOWN
as further trims land.

Token estimate: len(chars) / 4 — crude but stable, and only deltas matter.
Docstrings are collected via AST rather than SDK introspection so the test
does not depend on MCP SDK internals.
"""

from __future__ import annotations

import ast
from pathlib import Path

from comfy_mcp import instructions

_SERVER_SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_mcp" / "server.py"

# Ceiling, in estimated tokens (chars/4). Measured 2026-08-07 after the
# tool-consolidation series' third and final commit (`nodes(action=...)`,
# 47 -> 39 tools): ~13.84k (39 tool docstrings, 40248 doc chars + INSTRUCTIONS
# 15126 chars). Ratcheted 17,000 -> 15,000 accordingly — comfortably above the
# measurement so ordinary edits never trip it while real growth does. Ratchet
# DOWN as the docstring diet lands; never bump without stating why in the same
# PR.
#
# 15,000 -> 15,250 when `nodes` documented `is_api_node` / `exclude_api`: that
# paragraph is ~130 tokens and the tree was ALREADY at 14,984, so the slack the
# ratchet was set with had been spent by intervening docstring growth rather
# than by this change. Measured ~15,112 after it. Deliberately tight — the
# point of this ceiling is that the next growth is a decision too, not that
# there is room for one.
#
# 15,250 -> 15,400 for `generate_image`'s expired-wait contract (+~145 tokens
# across its docstring and the long-generation flow in INSTRUCTIONS). Growth a
# caller cannot do without: that a `wait=True` call is bounded by the CLIENT's
# invisible transport cap as much as by `timeout_seconds`, and that an expired
# wait returns the `prompt_id` of a job that is still running rather than
# failing, is not inferable from any other tool's docs — and an agent that does
# not know it re-runs a generation that is already on the GPU. Measured ~15,256
# after the edit, itself already trimmed twice against this ceiling.
#
# 15,400 -> 15,650 when `billing_status` was added on top of that. This one is
# a whole new TOOL, not docstring growth on an existing one: its description
# is ~1,110 chars, which is the mean length of the 39 that were already here,
# so there was no version of it that fit the slack left after the
# `generate_image` bump. Measured ~15,596 after it (40 tool docstrings), so the
# next growth is still a decision.
#
# 15,650 -> 15,750 for `launch_comfyui`'s remote-target refusal (+~110 tokens,
# already trimmed twice against this ceiling). Growth a caller cannot do
# without: that `launch_comfyui` is LOCAL-ONLY and, with a remote configured,
# spawns NOTHING and returns a structured refusal (``launched: False``) rather
# than starting a second local ComfyUI and reporting success, is not inferable
# from any other tool's docs — and an agent that does not know it acts on a
# launch that never touched the remote it meant to (re)start. Measured ~15,704
# after the edit, so the next growth is still a decision.
#
# `restart_comfyui`'s matching remote-target refusal (it RAISES rather than kill
# the local server) landed under the same ceiling, trimmed to one sentence and
# offset by dropping its now-redundant "skipped with a remote" note. Measured
# ~15,747 after it — the next growth is a decision.
#
# 15,750 -> 15,950 for the hand-written-workflow bullet in INSTRUCTIONS plus
# one sentence on `run_workflow` (+~170 tokens). The tree was already at
# ~15,747, so no useful version fit the 3 tokens of slack. Not inferable
# elsewhere: an API-format graph dragged into ComfyUI is auto-arranged and
# labelled by `_meta.title` (else a generic node-type name), so an agent that
# omits it hands the user a canvas that doesn't say what each step does; and
# its invented class_types need the `validate_workflow` pointer the existing
# missing-nodes bullet doesn't cover. Measured ~15,916 after the edit.
#
# 15,950 -> 16,050 for `search_models`'s target-awareness contract (+~60 tokens,
# the paragraph already trimmed twice against this ceiling). Growth a caller
# cannot do without, and the same shape as the `launch_comfyui` /
# `restart_comfyui` refusals above: that `search_models` now FOLLOWS
# `COMFYUI_URL` / `COMFYUI_HOST` like `run` / `jobs` — reading the REMOTE
# install's model catalog, and RAISING on a comfy-cli too old to accept the
# forwarded `--host` rather than silently listing THIS machine's — is not
# inferable from any other tool's docs, and an agent that does not know it
# trusts a local catalog as the remote's (the exact wrong-machine bug the
# forward fixes). The tree was already at ~15,916, so no useful version fit the
# slack. Measured ~15,986 after the edit, so the next growth is still a decision.
#
# 16,050 -> 16,350 when `check_template` was added. A whole new TOOL, not
# docstring growth on an existing one (same shape as the `billing_status` bump
# above): its description is ~1,050 chars — near the ~1,140-char mean of the 41
# already here — so there was no version of it that fit the slack left under the
# prior ceiling. The verdict vocabulary it documents (`runnable` /
# `missing-models` / `api-required` / `unknown`, and that `api-required` is the
# same paid-node gate `run_template` enforces) is the agent contract, not
# inferable from any other tool's docs. Measured ~16,294 after it (42 tool
# docstrings); the review round's caveats (`runnable` is weights only, a remote
# target gets a `comfy_target_note`) took it to ~16,342 within the same ceiling,
# so the next growth is still a decision.
#
# 16,350 -> 16,450 when `list_workflow_slots` gained `node`. Docstring growth on
# an existing tool rather than a new one, so the bar is the one the comment
# above sets: the three sentences are the agent contract for the parameter
# (what it selects, the comfy-cli 1.23.0 floor, that below the floor it is
# REFUSED rather than silently answered unfiltered, and that an unknown id is
# an error rather than an empty list). None of that is inferable from another
# tool's docs, and an agent that assumes the filter applied when it did not
# reads a whole graph's slots as one node's. The tree was already at ~16,337 of
# 16,350, so no version of it fit the slack. Measured ~16,409 after the edit,
# so the next growth is still a decision.
# 16,450 -> 16,475 to correct a FALSE claim rather than to add a feature. Both
# download docstrings said `completed` from `download(action="status")` was "the only
# proof a model is usable". It is not: nothing on comfy-cli's download path reads the
# response body or its content-type, so a URL answering `200 text/html` transfers
# successfully and leaves the web page at `dest` under the caller's filename, reporting
# `status: "completed"`, `percent: 100.0`, `error: null`. Measured 2026-11-23 against
# comfy-cli main: 577 bytes of HTML saved as `notamodel.safetensors`, and 276,289 bytes
# of model-card HTML as `flux1-dev.safetensors` from a bare `huggingface.co/<org>/<repo>`
# link. That sentence is the one an agent acts on when it tells a user a model is
# installed, and `search_models` cannot catch it either -- it lists filenames and never
# content -- so the docstrings now name the size check instead.
#
# The true statement is longer than the false one and the tree was at ~16,448 of 16,450,
# so no version of it fit the slack; it was trimmed to the agent contract first (the
# measurements and the reasoning are in a comment above `download`, and
# `download_model` gave up its duplicate copy of the mid-transfer note to pay for the
# rest).
#
# 16,475 -> 16,500 in review. The first cut of this correction added the size check to
# `download_model` only, while the CHANGELOG entry and the comment above `download` both
# claimed BOTH docstrings named it -- and `download` is the tool that actually reports
# `completed`, so it was the one that had to say it (2 of 6 panel reviewers). Adding the
# clause there measures 16,472, which fits 16,475 with three tokens to spare: too thin to
# leave, since the next unrelated word anywhere in the tree would then fail an innocent
# PR on this test. 25 tokens is the smallest step that is not a trap. Rewording the two
# proof clauses into one sentence paid 5 tokens back, so the measurement is 16,467 of
# 16,500 after the edit, and the next growth is still a decision. `download`'s own
# per-tool ceiling in `test_downloads.py` moved 245 -> 255 for the same clause.
#
# 16,500 -> 16,600 when `run_workflow`'s `confirm_spend` regained the
# spend-agreement requirement. On a client that cannot elicit,
# `_resolve_optin_spend_consent` returns True with no prompt, so the caller's flag
# is the only consent that exists. "Set True ONLY when the user has actually
# agreed - never merely to clear an error" is therefore the agent contract on
# exactly the path where nobody is asked, it is the line `run_template` and
# `partner_generate` both already carry, and comfy-mcp#300 is the record of an
# agent passing the flag prophylactically because the description had invited it.
# It costs ~29 tokens net of moving the `Gotchas` bullet about a gateless source
# build into a comment, which it belongs in anyway (no parameter selects that
# branch, so an agent cannot act on it).
#
# The step is 100 and not 25, and the two raises above are the evidence. This
# branch first raised 16,450 -> 16,600 because `main` had reached ~16,449 of
# 16,450 on its own. While it waited, `main` raised the ceiling twice more on
# the same reasoning - 16,475's author found three tokens of slack and called it
# "too thin to leave", then took the smallest step that is "not a trap", +25 -
# and that step was consumed inside two days: rebased on `main` at 5fccf21 this
# tree measures 16,496 of 16,500, four tokens clear. So a 25-token step is
# demonstrably a tripwire at this repo's current rate, and it is the NEXT
# author's innocent pull request that trips it. 100 leaves ~104 tokens, about
# one parameter description. Measured 16,496 before this raise; the next growth
# is still a decision, and the thing to check when it fires is whether `main`
# ate the slack rather than whether your own branch did.
_BUDGET_TOKENS = 16_600


def _is_tool_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        # matches @mcp.tool() and @mcp.tool
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "tool"
            and isinstance(target.value, ast.Name)
            and target.value.id == "mcp"
        ):
            return True
    return False


def test_llm_payload_within_budget():
    tree = ast.parse(_SERVER_SRC.read_text(encoding="utf-8"))
    doc_chars = 0
    tool_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_tool_decorated(node):
                tool_count += 1
                doc_chars += len(ast.get_docstring(node) or "")
    # Was `> 40`: the tool-consolidation series (six job tools into one grouped
    # `job(action=...)` here, `download`/`nodes` in later commits) walks the
    # live count from 53 down to 39. Relaxed rather than pinned exactly, so
    # this stays a discovery sanity check, not a second copy of
    # `test_the_readme_tool_count_matches_live_tool_set`.
    assert tool_count >= 35, f"tool discovery broke: found {tool_count} tools"

    total_chars = doc_chars + len(instructions.INSTRUCTIONS)
    est_tokens = total_chars // 4
    assert est_tokens <= _BUDGET_TOKENS, (
        f"LLM payload grew to ~{est_tokens} est. tokens "
        f"({tool_count} tool docstrings {doc_chars} chars + "
        f"INSTRUCTIONS {len(instructions.INSTRUCTIONS)} chars) — budget is "
        f"{_BUDGET_TOKENS}. Trim the docstring (agent contract only; "
        "maintainer rationale goes in comments) or, if growth is deliberate, "
        "bump _BUDGET_TOKENS in this same PR and say why."
    )
