"""The handshake's creative-assistant policy stays stated, without a playbook engine.

The policy is prose in ``instructions.INSTRUCTIONS``. These checks key on the
load-bearing ideas inside the "Working with the user" section, not on every
sentence, so wording can move. They also keep that section from being copied
into every generation docstring, and from growing a recommendation endpoint.
"""

from __future__ import annotations

import ast
from pathlib import Path

from comfy_mcp import instructions

_SERVER_SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_mcp" / "server.py"

FLAT = " ".join(instructions.INSTRUCTIONS.split())
_GUIDANCE_HEADER = "Working with the user — handle technical complexity"
_ROUTING_HEADER = "Routing — check the machine before running local diffusion."


def _guidance() -> str:
    start = FLAT.find(_GUIDANCE_HEADER)
    assert start != -1, "user-guidance section is gone"
    end = FLAT.find(_ROUTING_HEADER, start + 1)
    assert end > start, "user-guidance section no longer ends at the routing block"
    return FLAT[start:end]


def test_guidance_section_is_present_once_and_substantial():
    assert FLAT.count(_GUIDANCE_HEADER) == 1
    guidance = _guidance()
    assert len(guidance) > 500


def test_guidance_handles_technical_choices_and_surfaces_meaningful_ones():
    guidance = _guidance()
    assert "handle technical complexity, surface meaningful choices" in guidance
    assert "creative intent" in guidance
    assert "monetary cost" in guidance
    assert "which checkpoint is installed" in guidance


def test_guidance_drafts_before_production_quality():
    guidance = _guidance()
    assert "Draft before quality" in guidance
    assert "lowest PRACTICAL draft" in guidance
    assert "final or high-quality output" in guidance
    for aspect in (
        "composition",
        "subject consistency",
        "movement",
        "camera",
        "timing",
        "prompt adherence",
    ):
        assert aspect in guidance


def test_guidance_preserves_seed_and_comparability():
    guidance = _guidance()
    assert "Preserve comparability" in guidance
    assert "especially the seed" in guidance
    assert "Do not promise memory beyond what this session's context" in guidance


def test_guidance_prepares_oversized_keyframes_and_asks_about_crops():
    guidance = _guidance()
    assert "keyframe or reference image" in guidance
    assert "multi-megapixel" in guidance
    assert "meaningful crop" in guidance
    assert "Do not crop meaningful content away silently" in guidance
    assert "Do not upscale a source" in guidance


def test_guidance_recommends_one_available_workflow_when_none_is_named():
    guidance = _guidance()
    assert "ONE primary recommendation" in guidance
    assert "what THIS install can actually run" in guidance
    assert "no recommendation tool and no fixed model ranking" in guidance
    assert "text-to-video" in guidance and "native audio" in guidance


def test_guidance_inspects_before_a_final_quality_escalation():
    guidance = _guidance()
    assert "inspect or retrieve the result" in guidance
    assert "not a reason to raise resolution" in guidance
    assert "substantial escalation" in guidance


def test_existing_spend_guidance_remains_outside_the_new_section():
    """The creative policy must not replace the deterministic spend gate."""
    before = FLAT.split(_GUIDANCE_HEADER, 1)[0]
    assert "Every spending call confirms with the USER first" in before
    assert "comfy generate consent always" in before
    assert "confirm_spend" in before
    guidance = _guidance()
    assert "does not relax or replace that gate" in guidance


def test_machine_snapshot_routing_remains_in_the_handshake():
    assert "Machine snapshot" in FLAT
    assert _ROUTING_HEADER in FLAT
    assert FLAT.count(_ROUTING_HEADER) == 1


def test_generation_docstrings_point_at_the_policy_without_copying_it():
    """Global rules stay in the handshake. Tool text only points at them."""
    tree = ast.parse(_SERVER_SRC.read_text(encoding="utf-8"))
    docs: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docs[node.name] = " ".join((ast.get_docstring(node) or "").split())
    for name in ("run_workflow", "generate_image", "run_template", "partner_generate"):
        assert "seed" in docs[name]
        assert "lowest PRACTICAL draft" not in docs[name]
    assert "oversized reference" in docs["init_upload"]
    assert "generation policy lives in the handshake" in docs["init_upload"]
    assert "lowest PRACTICAL draft" not in docs["init_upload"]


def test_no_recommendation_endpoint_was_added():
    tree = ast.parse(_SERVER_SRC.read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "recommend_model" not in names
    assert "recommend_workflow" not in names
    assert "recommend_model" not in FLAT
    assert "recommend_workflow" not in FLAT
