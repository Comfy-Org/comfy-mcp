"""Tests for the template tools — the search -> fetch -> run_workflow on-ramp.

These lock in the passthrough argv (global flags before the subcommand, same
rule the wrapper enforces) and the behaviors ``search_templates`` owns on top of
comfy-cli's real ``templates ls`` payload:

1. comfy-cli emits ``{total_in_gallery, matched, shown, filters, rows: [...]}`` —
   the row list is keyed ``rows`` (verified against comfy-cli
   ``comfy_cli/command/templates.py`` ``ls_cmd``). ``search_templates`` filters
   ``rows`` client-side (free-text ``query`` the CLI has no flag for), forwards
   the CLI's own ``--tag/--type/--model/--provider`` gallery filters, drops
   API-tagged rows on ``exclude_api``, and pages via ``limit``/``offset``.
2. A payload with no ``rows`` list is a shape drift -> raise, never a raw dump.
3. ``fetch_template`` returns the ABSOLUTE output path for ``run_workflow``.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from comfy_local_mcp import server


def _fake_run(envelope: dict):
    """Return a subprocess.run stand-in that captures the call and emits an envelope."""
    calls: list[dict] = []

    def fake(cmd, capture_output, text, encoding, timeout, env, check):  # noqa: ARG001
        calls.append({"cmd": cmd, "env": env})
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(envelope), stderr=""
        )

    return fake, calls


# A representative slice of the real comfy-cli `templates ls` payload shape.
ROWS = [
    {
        "name": "image_to_image",
        "title": "Image to Image",
        "output_type": "image",
        "category_title": "Image",
        "tags": [],
        "models": ["SD1.5"],
        "providers": [],
        "description": "Basic image to image starter.",
    },
    {
        "name": "flux_i2i",
        "title": "Flux I2I",
        "output_type": "image",
        "category_title": "Image",
        "tags": ["API"],
        "models": ["Flux"],
        "providers": ["Black Forest Labs"],
        "description": "image to image with the Flux API.",
    },
    {
        "name": "seedream_5",
        "title": "Seedream 5",
        "output_type": "image",
        "category_title": "Image",
        "tags": ["API"],
        "models": ["Seedream"],
        "providers": ["ByteDance"],
        "description": "Text to image via Seedream 5.",
    },
    {
        "name": "basic_txt2img",
        "title": "Basic Text to Image",
        "output_type": "image",
        "category_title": "Image",
        "tags": [],
        "models": ["SD1.5"],
        "providers": [],
        "description": "Simple prompt to picture.",
    },
    {
        "name": "clip_maker",
        "title": "Clip Maker",
        "output_type": "video",
        "category_title": "Video",
        "tags": [],
        "models": ["Wan"],
        "providers": [],
        "description": "Make short clips from a prompt.",
    },
]


def _payload(rows=None) -> dict:
    rows = ROWS if rows is None else rows
    return {
        "total_in_gallery": 558,
        "matched": len(rows),
        "shown": len(rows),
        "filters": {
            "type": None,
            "category": None,
            "tag": None,
            "model": None,
            "provider": None,
            "name": None,
        },
        "rows": rows,
    }


def _patch_ls(monkeypatch, rows=None):
    """Make `_run_comfy` return the real-shape payload directly (filter-logic tests)."""
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: _payload(rows))


def _rows(result) -> list[dict]:
    return result["rows"]


def _names(result) -> list[str]:
    return [r["name"] for r in _rows(result)]


def test_search_templates_argv_and_empty_query_pages(monkeypatch):
    """Passthrough `comfy --json --where local templates ls`; empty query = first page + total."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": _payload()})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    result = server.search_templates()

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["templates", "ls"]  # no client-only flags forwarded
    # Empty query never returns the whole catalog: paged + a total count.
    assert result["total"] == len(ROWS)
    assert result["offset"] == 0
    assert result["shown"] == len(ROWS)  # fixture is smaller than the default limit


def test_search_templates_compact_projection(monkeypatch):
    """Listing rows carry only name/title/description/output_type — not the full detail."""
    _patch_ls(monkeypatch)
    row = _rows(server.search_templates())[0]
    assert set(row) == {"name", "title", "description", "output_type"}
    # The heavy fields live in get_template(name), not the listing.
    assert "tags" not in row and "models" not in row and "category_title" not in row


def test_search_templates_query_narrows_reported_cases(monkeypatch):
    """The two dogfooding queries return only their matches, not the whole catalog."""
    _patch_ls(monkeypatch)

    i2i = server.search_templates("image to image")
    assert set(_names(i2i)) == {"image_to_image", "flux_i2i"}
    assert i2i["total"] == 2

    seedream = server.search_templates("seedream 5")
    assert _names(seedream) == ["seedream_5"]

    # Case-insensitive, and a miss returns an empty page (not the fallback dump).
    assert _names(server.search_templates("IMAGE TO IMAGE")) == [
        "image_to_image",
        "flux_i2i",
    ]
    assert server.search_templates("no-such-template")["total"] == 0


def test_search_templates_query_hits_tags_and_models_not_output_type(monkeypatch):
    """Query matches tags/models list items, but never the output_type field."""
    _patch_ls(monkeypatch)

    # "wan" only appears in clip_maker's models list.
    assert _names(server.search_templates("wan")) == ["clip_maker"]
    # "seedream" appears in seedream_5's models (and name/title).
    assert "seedream_5" in _names(server.search_templates("seedream"))
    # "video" is ONLY clip_maker's output_type -> deliberately NOT matched,
    # otherwise query="image" would hit output_type on hundreds of rows.
    assert server.search_templates("video")["total"] == 0


def test_search_templates_pagination(monkeypatch):
    """limit/offset page the filtered rows deterministically; total is stable."""
    _patch_ls(monkeypatch)

    first = server.search_templates(limit=2, offset=0)
    assert _names(first) == ["image_to_image", "flux_i2i"]
    assert first["total"] == len(ROWS) and first["offset"] == 0 and first["shown"] == 2

    second = server.search_templates(limit=2, offset=2)
    assert _names(second) == ["seedream_5", "basic_txt2img"]
    assert second["offset"] == 2

    tail = server.search_templates(limit=2, offset=4)
    assert _names(tail) == ["clip_maker"]  # last partial page
    assert server.search_templates(limit=2, offset=100)["rows"] == []  # past the end


def test_search_templates_forwards_gallery_filters(monkeypatch):
    """tag/type/model/provider become the corresponding `templates ls` flags."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": _payload()})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    server.search_templates(
        tag="API", type="image", model="Flux", provider="Black Forest Labs"
    )

    assert calls[0]["cmd"][4:] == [
        "templates",
        "ls",
        "--tag",
        "API",
        "--type",
        "image",
        "--model",
        "Flux",
        "--provider",
        "Black Forest Labs",
    ]


def test_search_templates_exclude_api(monkeypatch):
    """exclude_api drops every row whose tags include `API` (case-insensitive)."""
    _patch_ls(monkeypatch)

    kept = server.search_templates(exclude_api=True)
    names = _names(kept)
    assert "flux_i2i" not in names and "seedream_5" not in names
    assert set(names) == {"image_to_image", "basic_txt2img", "clip_maker"}
    assert kept["total"] == 3


def test_search_templates_bad_shape_raises(monkeypatch):
    """A payload with no `rows` list is a shape drift -> raise, never a raw dump."""
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: {"count": 2})
    with pytest.raises(server.ComfyCliError, match="rows"):
        server.search_templates("flux")

    # A bare list (an older/other shape) is likewise rejected, not returned.
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: ["flux_dev"])
    with pytest.raises(server.ComfyCliError, match="rows"):
        server.search_templates()


def test_search_templates_non_dict_rows_raise(monkeypatch):
    """Non-dict rows are shape drift -> raise loudly, never silently dropped (BE-3342).

    Silently filtering them would undercount `total` and vanish templates, which
    contradicts the loud-fail guard the rest of the function is built around.
    """
    monkeypatch.setattr(
        server, "_run_comfy", lambda *a, **k: _payload(rows=ROWS[:1] + ["bare-string"])
    )
    with pytest.raises(server.ComfyCliError, match="not objects"):
        server.search_templates()


def test_search_templates_rejects_leading_dash_filter_values(monkeypatch):
    """A filter value starting with '-' is rejected before it reaches comfy-cli argv."""
    _patch_ls(monkeypatch)
    for kwargs in (
        {"tag": "--foo"},
        {"type": "-x"},
        {"model": "--bar"},
        {"provider": "-p"},
    ):
        with pytest.raises(server.ComfyCliError, match="leading '-'"):
            server.search_templates(**kwargs)


def test_search_templates_negative_limit_raises(monkeypatch):
    """A negative limit is a caller typo -> raise, not a silently-empty page."""
    _patch_ls(monkeypatch)
    with pytest.raises(server.ComfyCliError, match="limit"):
        server.search_templates(limit=-1)


def test_search_templates_limit_capped(monkeypatch):
    """An oversized limit is clamped so the response can't blow the tool-output cap."""
    big = [
        dict(ROWS[0], name=f"t{i}") for i in range(server._TEMPLATE_LIST_MAX_LIMIT + 50)
    ]
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: _payload(rows=big))

    result = server.search_templates(limit=10_000)
    assert result["total"] == len(big)  # total still reports the full match count
    assert result["shown"] == server._TEMPLATE_LIST_MAX_LIMIT  # page is capped


def test_get_template_argv(monkeypatch):
    """Passthrough: `comfy --json --where local templates show <name>`."""
    fake, calls = _fake_run(
        {"type": "envelope", "ok": True, "data": {"name": "flux_dev", "nodes": 12}}
    )
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    assert server.get_template("flux_dev") == {"name": "flux_dev", "nodes": 12}
    assert calls[0]["cmd"][4:] == ["templates", "show", "flux_dev"]


def test_fetch_template_argv_and_returns_abspath(monkeypatch, tmp_path):
    """Passthrough argv is `templates fetch <name> --out <path>`; returns the abs path."""
    fake, calls = _fake_run({"type": "envelope", "ok": True, "data": None})
    monkeypatch.setattr(server.shutil, "which", lambda _: "/fake/comfy")
    monkeypatch.setattr(server.subprocess, "run", fake)

    out = tmp_path / "flux.json"
    result = server.fetch_template("flux_dev", str(out))

    assert calls[0]["cmd"][4:] == ["templates", "fetch", "flux_dev", "--out", str(out)]
    assert result == str(out)  # tmp_path is already absolute
    assert os.path.isabs(result)


def test_fetch_template_resolves_relative_path(monkeypatch):
    """A relative out_path is returned as an absolute path (ready for run_workflow)."""
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: None)
    result = server.fetch_template("flux_dev", "flux.json")
    assert result == os.path.abspath("flux.json")
    assert os.path.isabs(result)
