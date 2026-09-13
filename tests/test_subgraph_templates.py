"""Regression tests: subgraph templates flow through the MCP layer untouched.

A gallery template built with the frontend's "subgraph" feature carries
UUID-typed instance nodes plus a ``definitions.subgraphs`` block in its
workflow JSON. The whole stack supports that shape — comfy-cli expands
``definitions.subgraphs`` client-side, and ``comfy workflow slots`` addresses
subgraph-interior inputs as ``A/B.name`` — so this server must stay a pure
passthrough: no tool may inspect the JSON and reject or block on subgraphs.
(The reported failure mode was informational, not code: an agent preemptively
refused a subgraph template as "can't unpack", so the INSTRUCTIONS now say the
shape is supported, and these tests pin the passthrough behavior itself.)
"""

from __future__ import annotations

import asyncio
import json

from conftest import envelope

from comfy_mcp.server import _internal as server
from comfy_mcp.server import instructions

# The UUID doubles as the instance node's `type` — the discriminator that makes
# a workflow "subgraphed" in the frontend format.
SUBGRAPH_ID = "8e9c5f5a-4a4f-4bde-9d7a-2f27c1c4a2b1"

# A minimal frontend-format workflow using a subgraph: one top-level UUID-typed
# instance node (115) and one interior node (75) inside the definition.
SUBGRAPH_WORKFLOW = {
    "nodes": [
        {
            "id": 115,
            "type": SUBGRAPH_ID,
            "inputs": [],
            "widgets_values": ["a photo of a fox"],
        },
    ],
    "links": [],
    "definitions": {
        "subgraphs": [
            {
                "id": SUBGRAPH_ID,
                "name": "Style pack",
                "nodes": [
                    {
                        "id": 75,
                        "type": "LoraLoaderModelOnly",
                        "widgets_values": ["style.safetensors", 0.7],
                    },
                ],
                "links": [],
            },
        ],
    },
}


def test_instructions_teach_that_subgraph_templates_are_supported():
    """The word "subgraph" must appear in INSTRUCTIONS — the guidance that stops
    an agent from preemptively refusing a subgraph template. A rewrite that
    drops it silently reintroduces the refusal failure mode."""
    assert "subgraph" in instructions.INSTRUCTIONS.lower()


def test_fetch_template_writes_subgraph_json_untouched(monkeypatch, tmp_path):
    """The subgraphed JSON comfy-cli writes reaches disk byte-identical.

    ``fetch_template`` composes ``templates fetch`` + ``validate`` and must not
    parse, rewrite, or veto the file in between — ``definitions.subgraphs``
    included.
    """
    out = tmp_path / "subgraph_template.json"
    raw = json.dumps(SUBGRAPH_WORKFLOW, indent=2)
    calls: list[tuple] = []

    def fake(*args, **kwargs):
        calls.append(args)
        if args[0] == "templates":
            # comfy-cli materializes the template at --out; the wrapper only
            # forwards the path.
            out.write_text(raw, encoding="utf-8")
            return None
        assert args[0] == "validate"
        return {
            "valid": True,
            "error_count": 0,
            "warning_count": 0,
            "errors": [],
            "warnings": [],
            "converted_from_ui": True,
        }

    monkeypatch.setattr(server, "_run_comfy", fake)

    result = server.fetch_template("subgraph_style", str(out))

    assert calls[0] == ("templates", "fetch", "subgraph_style", "--out", str(out))
    assert result["path"] == str(out)
    assert result["local_check"]["runnable"] is True
    # Byte-identical: the MCP layer never opened, reserialized, or edited it.
    assert out.read_text(encoding="utf-8") == raw
    assert json.loads(out.read_text(encoding="utf-8")) == SUBGRAPH_WORKFLOW


def test_run_workflow_submits_subgraph_workflow_without_inspection(
    patched_run, tmp_path
):
    """`comfy run --workflow <path>` is invoked verbatim — no pre-inspection
    rejection of the on-disk ``definitions.subgraphs`` content."""
    path = tmp_path / "subgraph_template.json"
    path.write_text(json.dumps(SUBGRAPH_WORKFLOW), encoding="utf-8")
    calls = patched_run(envelope(data={"prompt_id": "abc123"}))

    result = asyncio.run(server.run_workflow(str(path), wait=False))

    assert result == {"prompt_id": "abc123"}
    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]
    assert cmd[4:] == ["run", "--workflow", str(path)]


def test_list_workflow_slots_passes_subgraph_workflow_verbatim(patched_run, tmp_path):
    """The path goes to `comfy workflow slots` as-is, and subgraph-interior
    slot addresses (``115/75.strength``) come back unfiltered."""
    path = tmp_path / "subgraph_template.json"
    path.write_text(json.dumps(SUBGRAPH_WORKFLOW), encoding="utf-8")
    slots = [
        # Interior input: instance node 115 -> interior node 75.
        {"address": "115/75.strength", "value": 0.7},
        # Proxy widget promoted onto the instance node itself.
        {"address": "115.text", "value": "a photo of a fox"},
    ]
    calls = patched_run(envelope(data=slots))

    assert server.list_workflow_slots(str(path)) == slots

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]
    assert cmd[4:] == ["workflow", "slots", str(path)]
