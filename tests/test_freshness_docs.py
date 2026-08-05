"""Guards on the per-surface ``Freshness:`` policy in the catalog tool docstrings.

This server has four catalog-shaped surfaces with four genuinely different
freshness behaviors, and a docstring is the only place a calling agent can learn
which one it is holding:

* the node tools read the running ComfyUI's ``object_info`` — **LIVE**;
* the template tools read comfy-cli's gallery cache — **CACHED**, with a TTL that
  depends on the installed comfy-cli;
* ``search_models`` reads the install's disk — **LIVE**, filenames only;
* the partner-catalog tools read a curated allowlist plus a spec vendored into
  the comfy-cli wheel — **PINNED**.

Conflating them is not hypothetical: an agent reading the pinned partner catalog
told a user a model "does not exist" when it was merely un-allowlisted, and
another silently substituted a cheaper variant of a model family whose variants
live in a different tool's schema. These are prose assertions, so no functional
test would notice the caveats being dropped in a future docstring rewrite; this
file is that tripwire.

Assertions are keyed on the load-bearing FACTS (the classification word, the
source of truth, the escape hatch, the "absence is not non-existence" caveat)
rather than whole sentences, so wording can be edited without test churn. Every
docstring is whitespace-collapsed first — these are hard-wrapped, so any phrase
long enough to matter straddles a newline.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from comfy_mcp import server

# The node tools, all of which resolve against the live local `object_info`.
LIVE_NODE_TOOLS = (
    "search_nodes",
    "get_node",
    "list_nodes",
    "nodes_upstream",
    "nodes_downstream",
    "nodes_path",
    "nodes_types",
    "nodes_categories",
)

# The gallery-backed template tools.
CACHED_TEMPLATE_TOOLS = ("search_templates", "get_template", "fetch_template")

# The comfy-cli-pinned partner catalog. `generate_image` is deliberately NOT
# here: it runs one local gallery template with a local checkpoint and carries no
# model catalog of its own, so the pinned-catalog caveat would be false on it.
PINNED_PARTNER_TOOLS = ("list_partner_models", "partner_model_schema")

ALL_CATALOG_TOOLS = (
    *LIVE_NODE_TOOLS,
    *CACHED_TEMPLATE_TOOLS,
    "search_models",
    *PINNED_PARTNER_TOOLS,
)


def doc(tool_name: str) -> str:
    """One tool's docstring, whitespace-collapsed. See the module docstring."""
    return " ".join(raw_doc(tool_name).split())


def raw_doc(tool_name: str) -> str:
    """One tool's docstring VERBATIM — for the few assertions about layout.

    The example snippets are code an agent copies, so their indentation is part
    of the contract (a gate the run sits outside of does not gate); those checks
    need the lines the way they are written, not collapsed.
    """
    return getattr(server, tool_name).__doc__ or ""


def on_ramp_snippet() -> str:
    """``fetch_template``'s template on-ramp example, dedented and parseable.

    Pulled out as real code rather than grepped as prose because its SHAPE is
    what an agent copies: whether the run sits inside the gate's branch is a
    structural fact, and asserting it on an AST cannot be fooled by a prose
    sentence that happens to mention the same identifiers.
    """
    lines = raw_doc("fetch_template").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.rstrip().endswith("on-ramp::"))
    # The docstring reaches us already dedented (the tool decorator cleans it),
    # so the block's own indent is whatever its first code line carries — read it
    # rather than assuming a literal column.
    body = lines[start + 1 :]
    first = next(ln for ln in body if ln.strip())
    indent = len(first) - len(first.lstrip())
    assert indent, "the on-ramp example is no longer an indented literal block"
    block: list[str] = []
    for ln in body:
        # Blank lines ride along; the first non-blank line back at prose level
        # ends the block.
        if ln.strip() and not ln.startswith(" " * indent):
            break
        block.append(ln)
    return textwrap.dedent("\n".join(block)).strip("\n")


@pytest.mark.parametrize("tool_name", ALL_CATALOG_TOOLS)
def test_every_catalog_tool_states_a_freshness_policy(tool_name):
    """The whole point: an agent can read the policy off the tool it is calling.

    A catalog tool without this line is one an agent can neither trust nor
    caveat, so the marker itself is the contract — the per-class assertions below
    only check that the right one was stated.
    """
    assert "Freshness:" in doc(tool_name), (
        f"{tool_name} has no `Freshness:` line — a catalog tool must state which "
        "of the four freshness behaviors it has."
    )


@pytest.mark.parametrize("tool_name", LIVE_NODE_TOOLS)
def test_node_tools_are_documented_as_live(tool_name):
    """LIVE, and explicitly inclusive of the install's own staleness.

    The staleness clause is the non-obvious half: "live" alone reads as
    "authoritative", when what it actually means is "authoritative about THIS
    install" — an outdated ComfyUI truthfully reports an outdated node set.
    """
    text = doc(tool_name)
    assert "Freshness: LIVE" in text
    assert "object_info" in text
    assert "outdated install lists outdated nodes" in text


@pytest.mark.parametrize("tool_name", CACHED_TEMPLATE_TOOLS)
def test_template_tools_are_documented_as_cached_with_the_ttl(tool_name):
    """CACHED — and the 24h TTL is now unconditional in the supported range.

    The TTL landed in Comfy-Org/comfy-cli#559 and shipped in v1.14.0, which IS
    this server's hard floor (``_MIN_COMFY_CLI``), so every install that
    satisfies the floor expires the cache. That is why these docstrings state the
    TTL flatly and name v1.14.0 rather than hedging "NEWER than <floor>", which
    would claim the TTL needs something newer than the release carrying it.
    ``comfy templates refresh`` still has to appear: it is the manual escape
    hatch, and the only behavior left on a build that slipped past the fail-OPEN
    version guard.
    """
    text = doc(tool_name)
    assert "Freshness: CACHED" in text
    assert "24h TTL" in text
    assert "v1.14.0" in text
    assert "comfy templates refresh" in text
    # The install-independence caveat — the reason a cached hit can be unrunnable.
    assert "NOT read from the local install" in text


def test_the_cached_ttl_release_is_covered_by_this_servers_version_floor():
    """The TTL prose may only be unconditional while the floor carries the TTL.

    This is the guard that fired when the floor moved from 1.13.0 to 1.14.0: back
    then the TTL release was ABOVE the floor, so the docstrings had to document
    both behaviors, and the test pinned the floor so the prose could not silently
    outlive it. The floor now includes comfy-cli#559, so the assertion is
    inverted — it holds the docstrings' unconditional claim to the condition that
    makes it true. If the TTL release ever ends up above the floor again (a
    revert, a floor lowered), the template tools must go back to documenting the
    split rather than promising a TTL the floor does not guarantee.
    """
    ttl_release = (1, 14, 0)  # Comfy-Org/comfy-cli#559 shipped here
    assert server._MIN_COMFY_CLI >= ttl_release, (
        "the comfy-cli floor no longer covers comfy-cli#559 — the template "
        "tools' `Freshness:` caveat states the 24h TTL unconditionally, which is "
        "only true while the floor carries it. Restore the version split."
    )


def test_search_models_is_documented_as_live_disk_filenames():
    """LIVE, but a *narrower* kind of live than the node tools' — and it is the
    one surface where absence is most easily misread as non-existence."""
    text = doc("search_models")
    assert "Freshness: LIVE" in text
    assert "no registry metadata" in text
    # The joined phrase, not two loose substrings: asserting `"never"` and
    # `"no such model"` separately stays green if a rewrite drops the caveat
    # while some unrelated sentence still happens to contain "never".
    assert 'never means "no such model"' in text


def test_search_models_absence_names_the_out_of_scope_reading_first():
    """Absence has TWO readings here, and the wrong one costs a multi-GB download.

    Each mode searches something narrower than "the install" — no-argument lists
    folder names, ``folder`` reads one folder, and below the v1.14.0 floor
    ``query`` reads ``checkpoints`` only. A LoRA already on disk is absent from those
    results, so "not downloaded here" cannot be the only documented reading.
    """
    text = doc("search_models")
    assert "present but outside what this call searched" in text
    assert 'folder="loras"' in text
    assert "redundant multi-GB download" in text


@pytest.mark.parametrize("tool_name", PINNED_PARTNER_TOOLS)
def test_partner_catalog_tools_are_documented_as_pinned(tool_name):
    """PINNED — with the refresh escape hatch, since a stale wheel under-reports.

    The catalog is a curated allowlist in comfy-cli's own code resolved against a
    spec vendored into the installed wheel, so it can lag upstream by a release.
    Naming ``comfy generate refresh`` is what turns "I don't see it" into an
    actionable next step for the user.
    """
    text = doc(tool_name)
    assert "Freshness: PINNED" in text
    assert "comfy generate refresh" in text
    assert "INSTALLED comfy-cli" in text


def test_partner_list_refuses_to_read_as_proof_of_non_existence():
    """The "there is no such model" regression, stated as a docstring contract.

    An agent asserted a model did not exist off this list; the model was real and
    merely absent from comfy-cli's allowlist. The docstring has to say that
    absence here is not evidence, and point at the schema for the variant level
    this list collapses.
    """
    text = doc("list_partner_models")
    assert "NOT evidence it does not exist" in text
    assert "do not tell the user the model does not exist" in text
    assert "partner_model_schema" in text


def test_partner_list_names_the_upgrade_not_the_refresh_as_the_remedy():
    """The two escape hatches are not interchangeable on THIS tool.

    These rows are ``_ENDPOINT_ALLOWLIST`` (comfy-cli source) intersected with the
    active spec's paths, so a model that is simply not allowlisted cannot be
    surfaced by any spec re-pull — only a comfy-cli upgrade reaches it. Offering
    ``comfy generate refresh`` as an equal remedy sends the user through a step
    that cannot work for the cause the same paragraph names as likeliest.
    ``partner_model_schema`` is the tool where refresh genuinely is the fix: its
    variant enums are read from the spec.
    """
    text = doc("list_partner_models")
    assert "comfy-cli UPGRADE, not" in text
    assert "the allowlist is code" in text


def test_partner_schema_refuses_the_silent_variant_substitution():
    """The Pro-to-Lite regression: a downgrade the user never agreed to.

    The variant enum is the finest-grained thing this server can see, and it is
    pinned — so the correct move on a miss is to report the gap, never to swap in
    a neighbouring variant.
    """
    text = doc("partner_model_schema")
    assert "do NOT quietly substitute" in text
    assert "``lite``" in text and "``pro``" in text


def test_fetch_template_presents_validation_as_mandatory():
    """Step 4 of the on-ramp is a gate, not a suggestion.

    The failure this guards is an agent going straight from ``fetch_template`` to
    ``run_workflow`` because the fetch succeeded — gallery content has never been
    compared to the install, so "it downloaded" is not "it runs". The two skip
    paths (``check_local=False`` and a ``checked: false`` verdict) must both be
    documented as MOVING the gate, not removing it.
    """
    text = doc("fetch_template")
    assert "Step 4 is not optional" in text
    assert "validate_workflow" in text
    # `checked: false` is not a pass — it is the gate left undone.
    assert "leaves step 4 UNDONE" in text
    assert "moves the gate onto you, it does not remove it" in text


def test_fetch_template_example_gates_the_run_and_survives_an_unchecked_block():
    """The SNIPPET is the contract — an LLM pattern-matches on its shape.

    Two ways the example itself could teach the bug it warns about: putting
    ``run_workflow`` after the ``if`` at the same indentation (so the run happens
    either way), and indexing ``["runnable"]`` on a block that has no such key
    (``_unchecked`` returns ``{"checked", "reason", "summary"}``, which is what
    ``check_local=False`` and a not-running ComfyUI both produce). Assert the run
    is reached only through the cleared branch.
    """
    tree = ast.parse(on_ramp_snippet())

    gates = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
    assert len(gates) == 1, "the example should gate the run on exactly one `if`"
    gate = gates[0]
    # `.get("runnable")`, never `["runnable"]` — an `_unchecked` block has no
    # such key, so a subscript raises KeyError on the documented-normal paths.
    assert isinstance(gate.test, ast.Call), ast.dump(gate.test)
    assert isinstance(gate.test.func, ast.Attribute) and gate.test.func.attr == "get"
    assert [c.value for c in gate.test.args] == ["runnable"]

    runs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "run_workflow"
    ]
    assert len(runs) == 1, "the example should call `run_workflow` exactly once"
    # And it must be reachable ONLY through the cleared branch: a `run_workflow`
    # that is a sibling of the `if` rather than inside its body runs either way,
    # which is the fetch-then-run-anyway shape the surrounding prose forbids.
    cleared = [n for stmt in gate.body for n in ast.walk(stmt)]
    assert runs[0] in cleared, (
        "`run_workflow` sits outside the gate's cleared branch, so the example "
        "runs the workflow whether or not the check passed."
    )


def test_fetch_template_does_not_offer_a_world_writable_example_path():
    """The mandated gate validates a FILE, so a predictable shared path breaks it.

    ``/tmp/flux.json`` is pre-creatable as a symlink by any other local user and
    rewritable between the validate and the run — a TOCTOU where the validated
    bytes are not the executed bytes. The server's own ``_check_template_by_name``
    uses ``tempfile.mkdtemp`` for the same reason.
    """
    snippet = on_ramp_snippet()
    assert "fetch_template(" in snippet, "the on-ramp example lost its fetch call"
    # Only the SNIPPET is constrained — the prose below it names `/tmp/flux.json`
    # on purpose, as the anti-pattern.
    assert "/tmp/" not in snippet, snippet
    assert "only the user can write" in doc("fetch_template")


def test_fetch_template_does_not_claim_validate_workflow_is_an_equal_gate():
    """``validate_workflow`` alone is WEAKER than ``local_check``, and says so.

    Gallery templates are UI-format exports; a comfy-cli too old to lower one to
    API format checks zero nodes and reports ``valid: true``
    (``validate_workflow``'s blind spot 3). ``_local_template_check`` catches that
    and downgrades it to ``workflow_not_converted``, so presenting the raw call as
    "the same gate" would let a vacuous pass clear a mandatory check.
    """
    text = doc("fetch_template")
    assert "WEAKER" in text
    assert "non_node_key" in text
    assert "blind spot 3" in text


def test_get_template_documents_local_check_as_conditional():
    """``local_check`` is attached to comfy-cli's payload, not guaranteed by it.

    On a drifted (non-dict) payload ``get_template`` hands the payload back
    untouched, so there is no ``local_check`` key at all and a caller indexing it
    gets a ``KeyError`` — document it the way ``server_info``'s ``hardware`` block
    already is.
    """
    text = doc("get_template")
    assert "CONDITIONAL" in text
    assert "no ``local_check`` key at all" in text


def test_server_instructions_agree_with_the_per_tool_freshness_policy():
    """The tripwire's blind spot: ``INSTRUCTIONS`` is sent to every client too.

    The docstring assertions above cannot see it, so a policy stated on the tools
    and contradicted in the preamble would ship green — which is exactly what
    happened while ``INSTRUCTIONS`` still called the gallery "served fresh" and
    walked an agent from ``fetch_template`` straight to ``run_workflow``.
    """
    text = " ".join(server.INSTRUCTIONS.split())
    assert "served fresh" not in text, (
        "INSTRUCTIONS still calls the gallery live; it is comfy-cli's CACHED "
        "gallery (see the template tools' `Freshness:` blocks)."
    )
    assert "CACHED" in text
    assert "MANDATORY, not advisory" in text
    assert '.get("runnable")' in text


def test_search_templates_on_ramp_flags_validation_as_mandatory():
    """The on-ramp's step numbering starts here, so the gate has to be named here
    too — an agent that only reads step 1 must still learn step 4 exists."""
    text = doc("search_templates")
    assert "MANDATORY" in text
    assert "local_check" in text


def test_generate_image_carries_no_pinned_catalog_caveat():
    """``generate_image`` is not a catalog surface, and must not claim to be one.

    It runs ONE local gallery template with a locally-installed checkpoint, so it
    exposes no partner model list. Pasting the pinned-catalog caveat onto it (the
    obvious over-application of this policy) would document a catalog it does not
    have and point users at a ``comfy generate`` refresh that has no bearing on
    what it runs.
    """
    text = doc("generate_image")
    assert "Freshness: PINNED" not in text
    assert "comfy generate refresh" not in text
