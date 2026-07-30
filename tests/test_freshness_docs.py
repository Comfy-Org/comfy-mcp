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

import pytest

from comfy_local_mcp import server

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
    tool = getattr(server, tool_name)
    return " ".join((tool.__doc__ or "").split())


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
def test_template_tools_are_documented_as_cached_with_the_version_split(tool_name):
    """CACHED — and the TTL is comfy-cli-version-dependent, which must be stated.

    The 24h TTL landed in Comfy-Org/comfy-cli#559, *after* v1.13.0 was cut, and
    v1.13.0 is this server's hard floor (``_MIN_COMFY_CLI``). So both behaviors
    are live in the supported range: documenting the TTL unconditionally would be
    a false claim for every user on the floor version, where the first fetch is
    kept until ``comfy templates refresh``. Same shape as the comfy-cli version
    split already documented in ``search_models``.
    """
    text = doc(tool_name)
    assert "Freshness: CACHED" in text
    assert "24h TTL" in text
    assert "v1.13.0" in text
    assert "comfy templates refresh" in text
    # The install-independence caveat — the reason a cached hit can be unrunnable.
    assert "NOT read from the local install" in text


def test_the_cached_ttl_floor_claim_still_matches_this_servers_version_floor():
    """The version in the TTL caveat is the floor, not a number frozen in prose.

    The caveat's "NEWER than v1.13.0" is only meaningful while v1.13.0 is what
    ``_MIN_COMFY_CLI`` requires. If the floor moves past the release that carries
    the TTL, the split disappears and these docstrings should stop hedging — this
    is the test that says so instead of leaving stale prose behind.
    """
    assert server._MIN_COMFY_CLI_STR == "1.13.0", (
        "the comfy-cli floor moved — re-check the template tools' `Freshness:` "
        "caveat: once the floor includes comfy-cli#559 the 24h TTL is "
        "unconditional and the v1.13.0 hedge should be dropped."
    )


def test_search_models_is_documented_as_live_disk_filenames():
    """LIVE, but a *narrower* kind of live than the node tools' — and it is the
    one surface where absence is most easily misread as non-existence."""
    text = doc("search_models")
    assert "Freshness: LIVE" in text
    assert "no registry metadata" in text
    assert "never" in text and "no such model" in text


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
