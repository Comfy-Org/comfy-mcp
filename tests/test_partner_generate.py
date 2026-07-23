# Tests for the ``partner_generate`` tool — the thin passthrough to
# ``comfy generate <model>`` (partner / hosted-API proxy, spends credits).
# comfy-cli is mocked: no real proxy call, no real spend. The tool uses the
# plain ``_run_comfy`` path (no streaming), so these just assert the command
# mapping and the spend-safety invariants.
from __future__ import annotations

import pytest

from comfy_local_mcp import server


def _patch_run_comfy(monkeypatch):
    """Record the args/timeout of the (mocked) ``_run_comfy`` call."""
    seen: dict = {}

    def fake_run_comfy(*args, timeout=None):
        seen["args"] = args
        seen["timeout"] = timeout
        return {"prompt_id": "p1", "outputs": ["/x.png"]}

    monkeypatch.setattr(server, "_run_comfy", fake_run_comfy)
    return seen


def test_partner_generate_wait_true_maps_command(monkeypatch):
    """wait=True → `comfy generate <model>` with the long timeout, no --async."""
    seen = _patch_run_comfy(monkeypatch)

    result = server.partner_generate("flux-pro")

    assert result == {"prompt_id": "p1", "outputs": ["/x.png"]}
    assert seen["args"] == ("generate", "flux-pro")  # no --async, no params
    assert seen["timeout"] == 600.0


def test_partner_generate_forwards_params_as_equals(monkeypatch):
    """params → `--key=value`, in insertion order, after the model."""
    seen = _patch_run_comfy(monkeypatch)

    server.partner_generate(
        "ideogram-edit", params={"prompt": "a cat", "aspect_ratio": "16:9"}
    )

    assert seen["args"] == (
        "generate",
        "ideogram-edit",
        "--prompt=a cat",
        "--aspect_ratio=16:9",
    )


def test_partner_generate_wait_false_uses_async(monkeypatch):
    """wait=False submits with --async and the short fire-and-return timeout."""
    seen = _patch_run_comfy(monkeypatch)

    server.partner_generate("dalle", params={"prompt": "x"}, wait=False)

    assert seen["args"] == ("generate", "dalle", "--prompt=x", "--async")
    assert seen["timeout"] == 60.0


def test_partner_generate_param_value_leading_dash_kept(monkeypatch):
    """A value starting with `-` survives via the `--key=value` form."""
    seen = _patch_run_comfy(monkeypatch)

    server.partner_generate("flux-pro", params={"seed": "-1"})

    assert seen["args"] == ("generate", "flux-pro", "--seed=-1")


def test_partner_generate_never_sends_yes(monkeypatch):
    """Spend-safety invariant: this tool never sends --yes/-y — consent is the
    engine's job, never inferred from tool permission."""
    seen = _patch_run_comfy(monkeypatch)

    server.partner_generate("flux-pro", params={"prompt": "x"}, wait=False)

    assert "--yes" not in seen["args"]
    assert "-y" not in seen["args"]


@pytest.mark.parametrize("bad_model", ["", "-flux", "--flux"])
def test_partner_generate_rejects_flag_like_or_empty_model(monkeypatch, bad_model):
    """An empty or flag-like model is rejected before any shell-out."""
    called = False

    def must_not_run(*args, timeout=None):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(server, "_run_comfy", must_not_run)

    with pytest.raises(server.ComfyCliError):
        server.partner_generate(bad_model)
    assert called is False  # no command was run for rejected input
