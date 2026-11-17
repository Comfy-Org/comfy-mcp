"""Validation diagnostics must survive the CLI envelope's error rendering."""

import asyncio
import json

import pytest
from conftest import envelope

from comfy_mcp import errors, server


@pytest.mark.parametrize(
    ("key", "detail", "expected"),
    [
        (
            "errors",
            [
                {
                    "node_id": "10",
                    "field": "model_name",
                    "code": "unknown_enum_value",
                    "suggestions": ["available-model.safetensors"],
                }
            ],
            ["10", "model_name", "unknown_enum_value", "available-model.safetensors"],
        ),
        (
            "node_errors",
            {
                "46": {
                    "class_type": "SaveImage",
                    "errors": [
                        {
                            "type": "required_input_missing",
                            "message": "Required input is missing",
                            "details": "images",
                        }
                    ],
                }
            },
            ["46", "SaveImage", "required_input_missing", "images"],
        ),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_validation_details_survive_cli_failure(
    patched_run, patched_stream, key, detail, expected, streaming
):
    failed = envelope(
        ok=False,
        error={
            "code": "prompt_rejected",
            "message": "Workflow validation failed",
            "hint": f"Inspect details.{key} and fix the workflow",
            "details": {key: detail},
        },
    )
    if streaming:
        patched_stream(json.dumps(failed) + "\n")
    else:
        patched_run(failed, returncode=1)

    with pytest.raises(server.ComfyCliError) as caught:
        if streaming:
            asyncio.run(server._run_comfy_streaming("run", "--workflow", "wf.json"))
        else:
            server._run_comfy("run", "--workflow", "wf.json")

    assert caught.value.code == "prompt_rejected"
    message = str(caught.value)
    assert f"{key}:" in message
    for value in expected:
        assert value in message


@pytest.mark.parametrize("key", ["errors", "node_errors"])
def test_validation_details_are_scrubbed_before_truncation(key):
    url = "https://<user>:<pass>@example.com/model?token=SECRET_QUERY#SECRET_FRAGMENT"
    detail = {"node_id": "10", "message": f"Cannot load {url}", "details": "x" * 2000}
    value = [detail] if key == "errors" else {"10": {"errors": [detail]}}

    rendered = errors._render_error_details({key: value, "unrelated": "omit me"})

    assert rendered is not None
    assert "https://" in rendered and "example.com/model" in rendered
    assert "<user>" not in rendered and "<pass>" not in rendered
    assert "SECRET_QUERY" not in rendered and "SECRET_FRAGMENT" not in rendered
    assert "omit me" not in rendered
    assert len(rendered) <= len(key) + 2 + errors._MAX_ERROR_FIELD_CHARS


def test_validation_details_do_not_change_partner_node_rendering():
    assert (
        errors._render_error_details(
            {
                "partner_nodes": ["ExampleNode", "OtherNode"],
                "errors": [],
                "node_errors": {},
            }
        )
        == "partner_nodes: ExampleNode, OtherNode"
    )
    assert errors._render_error_details({"unrelated": "omit me"}) is None
