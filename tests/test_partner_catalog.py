"""Tests for the partner-catalog discovery tools — ``list_partner_models`` and
``partner_model_schema``.

These are the discovery half of the partner surface: ``partner_generate`` runs a
model, and these two answer "which models are there?" and "what does this one
take?" without the caller leaving the MCP. What they lock in:

1. The passthrough argv — ``comfy generate list`` / ``comfy generate schema
   <model>`` with the global flags first, and the ``--style`` / ``--partner`` /
   ``--query`` filters FORWARDED to comfy-cli rather than reimplemented here.
2. ``envelope/1`` parsing only. ``list_partner_models`` reads comfy-cli's
   ``{models, count, filters}`` payload; neither tool ever parses a rendered
   table, and against a comfy-cli whose ``generate list`` still only renders one
   the failure names that gap instead of relaying a wall of box-drawing
   characters.
3. The response-size guard — ``limit``/``offset`` paging with a hard ceiling, the
   same shape ``search_templates`` uses on the 558-row template gallery.
4. That adding these tools did NOT weaken ``_GENERATE_RESERVED_TARGETS``:
   ``partner_generate``'s contract is still "run a MODEL", so ``list`` /
   ``schema`` / ``consent`` remain refused as model targets. The new verbs are
   reached by their own tools, not through a model name.

comfy-cli is mocked throughout: nothing here runs a real CLI.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import envelope

from comfy_local_mcp import server

# A representative slice of the real `comfy generate list` payload
# (comfy_cli/schemas/generate_list.json): one record per model, `count`, and the
# filters echoed back.
MODELS = [
    {
        "alias": "flux-pro",
        "id": "bfl/flux-pro-1.1/generate",
        "partner": "bfl",
        "category": "text-to-image",
        "mode": "async",
        "summary": "Flux Pro 1.1 text-to-image generation.",
    },
    {
        "alias": "flux-fill",
        "id": "bfl/flux-pro-1.0-fill/generate",
        "partner": "bfl",
        "category": "inpaint",
        "mode": "async",
        "summary": "Flux inpainting and outpainting.",
    },
    {
        "alias": "ideogram-edit",
        "id": "ideogram/v3/edit",
        "partner": "ideogram",
        "category": "image-edit",
        "mode": "sync",
        "summary": "Edit an image with Ideogram v3.",
    },
    {
        "alias": "seedance",
        "id": "bytedance/seedance/text-to-video",
        "partner": "bytedance",
        "category": "video",
        "mode": "async",
        "summary": "Seedance text-to-video generation.",
    },
]


def _list_payload(models=None, **filters) -> dict:
    models = MODELS if models is None else models
    return {
        "models": models,
        "count": len(models),
        "filters": {
            "partner": filters.get("partner"),
            "category": filters.get("category"),
            "query": filters.get("query"),
        },
    }


# A representative slice of `comfy generate schema flux-pro`
# (comfy_cli/schemas/generate_schema.json), required parameters first.
SCHEMA_DATA = {
    "model": "flux-pro",
    "id": "bfl/flux-pro-1.1/generate",
    "partner": "bfl",
    "category": "text-to-image",
    "summary": "Flux Pro 1.1 text-to-image generation.",
    "mode": "async",
    "polling": "bfl",
    "content_type": "application/json",
    "params": [
        {
            "name": "prompt",
            "type": "string",
            "kind": "string",
            "required": True,
            "default": None,
            "enum": [],
            "description": "Text prompt for image generation.",
            "item_type": None,
            "upload_mode": None,
        },
        {
            "name": "output_format",
            "type": "enum",
            "kind": "enum",
            "required": False,
            "default": "jpeg",
            "enum": ["jpeg", "png"],
            "description": "Output image format.",
            "item_type": None,
            "upload_mode": None,
        },
    ],
    "example": 'comfy generate flux-pro --prompt "a red fox"',
}


# --- list_partner_models: passthrough argv ----------------------------------


def test_list_partner_models_argv(patched_run):
    """Global flags first, then the bare `generate list` sub-action."""
    calls = patched_run(envelope(data=_list_payload()))

    result = server.list_partner_models()

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["generate", "list"]
    assert result["total"] == 4
    assert result["shown"] == 4
    assert result["offset"] == 0
    assert result["models"] == MODELS


def test_list_partner_models_returns_structured_records(patched_run):
    """Records, not rendered text — every field the catalog schema declares."""
    patched_run(envelope(data=_list_payload()))

    row = server.list_partner_models()["models"][0]

    assert row == {
        "alias": "flux-pro",
        "id": "bfl/flux-pro-1.1/generate",
        "partner": "bfl",
        "category": "text-to-image",
        "mode": "async",
        # The FULL summary, not the `…`-clipped form the human table renders.
        "summary": "Flux Pro 1.1 text-to-image generation.",
    }


def test_list_partner_models_echoes_comfy_cli_filters(patched_run):
    """comfy-cli's own filter echo rides back, so a zero-row answer is legible."""
    patched_run(envelope(data=_list_payload(models=[], category="text-to-image")))

    result = server.list_partner_models(style="text-to-image")

    assert result["total"] == 0
    assert result["models"] == []
    assert result["filters"] == {
        "partner": None,
        "category": "text-to-image",
        "query": None,
    }


# --- list_partner_models: filters are FORWARDED, not reimplemented ----------


def test_list_partner_models_forwards_style_filter(patched_run):
    """`style` is comfy-cli's `--style`; the CLI does the filtering."""
    t2i = [m for m in MODELS if m["category"] == "text-to-image"]
    calls = patched_run(envelope(data=_list_payload(t2i, category="text-to-image")))

    result = server.list_partner_models(style="text-to-image")

    assert calls[0]["cmd"][-3:] == ["list", "--style", "text-to-image"]
    assert [m["alias"] for m in result["models"]] == ["flux-pro"]


def test_list_partner_models_forwards_partner_and_query_filters(patched_run):
    """All three filters ride together, each as its own `--flag value` pair."""
    calls = patched_run(envelope(data=_list_payload([])))

    server.list_partner_models(style="video", partner="bytedance", query="seedance")

    assert calls[0]["cmd"][4:] == [
        "generate",
        "list",
        "--style",
        "video",
        "--partner",
        "bytedance",
        "--query",
        "seedance",
    ]


def test_list_partner_models_omits_empty_filters(patched_run):
    """An unset filter adds no flag at all — no empty-string value reaches argv."""
    calls = patched_run(envelope(data=_list_payload()))

    server.list_partner_models(style="", partner="", query="")

    assert calls[0]["cmd"][4:] == ["generate", "list"]


# --- list_partner_models: response-size guard -------------------------------


def test_list_partner_models_pages_with_limit_and_offset(patched_run):
    """`total` is the pre-paging match count; `models` is the current window."""
    patched_run(envelope(data=_list_payload()))
    first = server.list_partner_models(limit=2)
    assert first == {
        "total": 4,
        "shown": 2,
        "offset": 0,
        "filters": first["filters"],
        "models": MODELS[:2],
    }

    patched_run(envelope(data=_list_payload()))
    second = server.list_partner_models(limit=2, offset=2)
    assert second["total"] == 4
    assert second["shown"] == 2
    assert second["offset"] == 2
    assert second["models"] == MODELS[2:]


def test_list_partner_models_negative_offset_clamps_to_zero(patched_run):
    patched_run(envelope(data=_list_payload()))

    result = server.list_partner_models(offset=-5)

    assert result["offset"] == 0
    assert result["models"] == MODELS


def test_list_partner_models_limit_capped(monkeypatch, patched_run):
    """An oversized `limit` can't build a response past the tool-output cap."""
    monkeypatch.setattr(server, "_PARTNER_MODEL_MAX_LIMIT", 2)
    patched_run(envelope(data=_list_payload()))

    result = server.list_partner_models(limit=10_000)

    assert result["shown"] == 2
    assert result["total"] == 4


def test_list_partner_models_negative_limit_raises(patched_run):
    calls = patched_run(envelope(data=_list_payload()))

    with pytest.raises(server.ComfyCliError, match="invalid limit"):
        server.list_partner_models(limit=-1)

    assert calls == []  # never reached comfy-cli


def test_list_partner_models_zero_limit_still_reports_the_total(patched_run):
    """`limit=0` is a count probe, not an empty catalog."""
    patched_run(envelope(data=_list_payload()))

    result = server.list_partner_models(limit=0)

    assert result["total"] == 4
    assert result["shown"] == 0
    assert result["models"] == []


# --- list_partner_models: input hygiene -------------------------------------


@pytest.mark.parametrize("bad", ["-x", "--style", "-"])
def test_list_partner_models_rejects_option_like_filter_values(patched_run, bad):
    """A dash-leading value collides with `comfy generate`'s own flag split."""
    for kwargs in ({"style": bad}, {"partner": bad}, {"query": bad}):
        calls = patched_run(envelope(data=_list_payload()))
        with pytest.raises(server.ComfyCliError, match="value"):
            server.list_partner_models(**kwargs)
        assert calls == []


def test_list_partner_models_rejects_embedded_nul_filter_values(patched_run):
    for kwargs in ({"style": "a\0b"}, {"partner": "a\0b"}, {"query": "a\0b"}):
        calls = patched_run(envelope(data=_list_payload()))
        with pytest.raises(server.ComfyCliError, match="NUL"):
            server.list_partner_models(**kwargs)
        assert calls == []


# --- list_partner_models: shape drift ---------------------------------------


def test_list_partner_models_bad_shape_raises(patched_run):
    """A payload with no `models` list is drift — raise, never dump it raw."""
    patched_run(envelope(data={"count": 0}))

    with pytest.raises(server.ComfyCliError, match="expected a dict with a `models`"):
        server.list_partner_models()


def test_list_partner_models_non_dict_records_raise(patched_run):
    """Dropping non-object rows would silently undercount `total`."""
    patched_run(envelope(data={"models": [MODELS[0], "flux-pro"], "count": 2}))

    with pytest.raises(server.ComfyCliError, match="1 of 2 models are not objects"):
        server.list_partner_models()


# --- list_partner_models: comfy-cli that predates JSON output ---------------

# What a comfy-cli whose `generate list` only renders a table leaves on stdout:
# no envelope at all, just the Rich box-drawing the ticket exists to stop agents
# scraping.
_RENDERED_TABLE = (
    "┏━━━━━━━━━━━━┳━━━━━━━━━┓\n"
    "┃ Model      ┃ Partner ┃\n"
    "┡━━━━━━━━━━━━╇━━━━━━━━━┩\n"
    "│ flux-pro   │ bfl     │\n"
    "└────────────┴─────────┘\n"
)


def test_list_partner_models_names_a_comfy_cli_that_emits_no_json(patched_run):
    """The failure explains the version gap instead of relaying the table."""
    patched_run(_RENDERED_TABLE, returncode=0)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.list_partner_models()

    message = str(excinfo.value)
    assert "emitted no JSON" in message
    assert "pip install --upgrade comfy-cli" in message
    # The generic wrapper error is kept as the underlying cause, not replaced.
    assert "returned no JSON" in message


def test_partner_model_schema_names_a_comfy_cli_that_emits_no_json(patched_run):
    patched_run("Model: flux-pro\n  --prompt  Text prompt\n", returncode=0)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.partner_model_schema("flux-pro")

    assert "generate schema` emitted no JSON" in str(excinfo.value)


def test_a_nonzero_exit_without_an_envelope_is_not_reported_as_a_version_gap(
    patched_run,
):
    """Only a CLEAN exit with no envelope is the version gap.

    Everything else that reaches "no JSON" — a crash, a macOS TCC denial, an
    unreadable spec cache, a usage error — exits non-zero, and calling one of
    those "upgrade comfy-cli" would send the caller after the wrong thing.
    """
    patched_run(
        "",
        returncode=1,
        stderr="Traceback (most recent call last):\nyaml.scanner.ScannerError",
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.list_partner_models()

    message = str(excinfo.value)
    assert "pip install --upgrade comfy-cli" not in message
    assert "ScannerError" in message  # comfy-cli's own diagnosis, relayed


def test_a_real_error_envelope_is_not_reported_as_a_version_gap(patched_run):
    """A current comfy-cli's own error must reach the caller untouched."""
    patched_run(
        envelope(
            ok=False,
            error={
                "code": "generate_model_unknown",
                "message": "Unknown model 'flux-turbo'.",
                "hint": "run `comfy generate list` to see the available aliases",
            },
        ),
        returncode=1,
    )

    with pytest.raises(server.ComfyCliError) as excinfo:
        server.partner_model_schema("flux-turbo")

    assert excinfo.value.code == "generate_model_unknown"
    assert "Unknown model" in str(excinfo.value)
    assert "pip install --upgrade" not in str(excinfo.value)


# --- partner_model_schema ---------------------------------------------------


def test_partner_model_schema_argv_and_payload(patched_run):
    """The model is the second positional, and `data` comes back verbatim."""
    calls = patched_run(envelope(data=SCHEMA_DATA))

    result = server.partner_model_schema("flux-pro")

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["generate", "schema", "flux-pro"]
    assert result == SCHEMA_DATA


def test_partner_model_schema_marks_which_params_are_required(patched_run):
    """The AC the tool exists for: `required` is data, not prose to parse."""
    patched_run(envelope(data=SCHEMA_DATA))

    params = server.partner_model_schema("flux-pro")["params"]

    assert [p["name"] for p in params if p["required"]] == ["prompt"]
    assert [p["name"] for p in params if not p["required"]] == ["output_format"]
    assert params[1]["enum"] == ["jpeg", "png"]
    assert params[1]["default"] == "jpeg"


def test_partner_model_schema_accepts_an_endpoint_id(patched_run):
    """comfy-cli takes the canonical id anywhere it takes an alias."""
    calls = patched_run(envelope(data=SCHEMA_DATA))

    server.partner_model_schema("bfl/flux-pro-1.1/generate")

    assert calls[0]["cmd"][-1] == "bfl/flux-pro-1.1/generate"


@pytest.mark.parametrize("model", ["", "--help", "-x"])
def test_partner_model_schema_rejects_option_like_model(patched_run, model):
    """An empty/dash-leading model would be parsed by comfy-cli as an option."""
    calls = patched_run(envelope(data=SCHEMA_DATA))

    with pytest.raises(server.ComfyCliError, match="invalid model"):
        server.partner_model_schema(model)

    assert calls == []


def test_partner_model_schema_rejects_embedded_nul_model(patched_run):
    calls = patched_run(envelope(data=SCHEMA_DATA))

    with pytest.raises(server.ComfyCliError, match="NUL"):
        server.partner_model_schema("flux\0pro")

    assert calls == []


def test_partner_model_schema_reads_the_catalog_no_spend_gate(patched_run):
    """A spec read spends nothing, so it must not probe or prompt for consent.

    Asserted by argv: one call, and it is the schema read — a spend-gate probe
    (`generate consent show`) would show up here as an extra invocation.
    """
    calls = patched_run(envelope(data=SCHEMA_DATA))

    server.partner_model_schema("flux-pro")

    assert len(calls) == 1
    assert "consent" not in calls[0]["cmd"]
    assert "--yes" not in calls[0]["cmd"]


def test_list_partner_models_reads_the_catalog_no_spend_gate(patched_run):
    calls = patched_run(envelope(data=_list_payload()))

    server.list_partner_models()

    assert len(calls) == 1
    assert "--yes" not in calls[0]["cmd"]


# --- the reserved-target block is UNCHANGED ---------------------------------


def test_generate_reserved_targets_are_unchanged():
    """Adding these tools must not relax `partner_generate`'s target block.

    The two new verbs are reached by their OWN tools, calling the sub-actions
    directly — never by passing `list` / `schema` as a model name. `consent`
    above all stays blocked: it is the spend gate's configuration surface, and a
    tool whose contract is "run a MODEL" must not be a way in to it.
    """
    assert server._GENERATE_RESERVED_TARGETS == frozenset(
        {"list", "schema", "refresh", "upload", "resume", "consent"}
    )


@pytest.mark.parametrize("target", ["list", "schema", "consent"])
def test_partner_generate_still_refuses_reserved_targets(patched_plain_run, target):
    """`partner_generate("list")` is still a refusal, not a catalog listing."""
    calls = patched_plain_run(0, stdout="done")

    with pytest.raises(server.ComfyCliError, match="sub-action"):
        asyncio.run(server.partner_generate(target))

    assert calls == []


@pytest.mark.parametrize("target", ["list", "schema", "consent"])
def test_emit_partner_workflow_still_refuses_reserved_targets(patched_run, target):
    """The same block on the other tool that hands comfy-cli a model positional."""
    calls = patched_run(envelope(data={"out": "/tmp/x.json"}))

    with pytest.raises(server.ComfyCliError, match="sub-action"):
        asyncio.run(server.emit_partner_workflow(target, "/tmp/x.json"))

    assert calls == []


# --- the catalog is genuinely reachable end to end --------------------------


def test_discovery_chain_list_then_schema(patched_run):
    """An alias from `list_partner_models` is what `partner_model_schema` takes.

    The whole point of the pair: no shelling out, no table scraping, and no
    hand-copied alias between the two calls.
    """
    patched_run(envelope(data=_list_payload()))
    alias = server.list_partner_models(style="text-to-image")["models"][0]["alias"]

    calls = patched_run(envelope(data=SCHEMA_DATA))
    schema = server.partner_model_schema(alias)

    assert calls[0]["cmd"][-1] == "flux-pro"
    assert schema["model"] == alias
    assert json.dumps(schema)  # the payload is JSON-serializable as returned
