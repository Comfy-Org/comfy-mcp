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
3. ``fetch_template`` returns ``{"path": <ABSOLUTE path>, "local_check": {...}}``
   — the path for ``run_workflow``, and the cross-check of the template against
   the live local ``object_info`` (``get_template`` reports the same block).
"""

from __future__ import annotations

import os

import pytest
from conftest import envelope

from comfy_mcp import server

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


def test_search_templates_argv_and_empty_query_pages(patched_run):
    """Passthrough `comfy --json --where local templates ls`; empty query = first page + total."""
    calls = patched_run(envelope(data=_payload()))

    result = server.search_templates()

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    assert cmd[4:] == ["templates", "ls"]  # no client-only flags forwarded
    # Empty query never returns the whole catalog: paged + a total count.
    assert result["total"] == len(ROWS)
    assert result["offset"] == 0
    assert result["shown"] == len(ROWS)  # fixture is smaller than the default limit


def test_search_templates_compact_projection(monkeypatch):
    """Rows carry name/title/description/output_type + the derived `api` flag."""
    _patch_ls(monkeypatch)
    row = _rows(server.search_templates())[0]
    assert set(row) == {"name", "title", "description", "output_type", "api"}
    # The heavy fields still live in get_template(name), not the listing — `api`
    # is one derived boolean, not the `tags` list coming back by the side door.
    assert "tags" not in row and "models" not in row and "category_title" not in row


def test_search_templates_rows_flag_api_templates(monkeypatch):
    """`api` is true exactly for the API-tagged rows, matched case-insensitively.

    Without it the projection stripped `tags` and two templates with identical
    titles (`api_minimax_h3_t2v` / `video_minimax_h3_t2v` in the real gallery)
    were indistinguishable in results — so an agent could recommend the paid one
    while reporting no free version exists.
    """
    rows = [
        {"name": "local_one", "title": "T", "output_type": "image", "tags": []},
        {"name": "paid_upper", "title": "T", "output_type": "image", "tags": ["API"]},
        # comfy-cli passes gallery tags through verbatim, so the case is the
        # gallery's to choose — the same predicate `exclude_api` uses folds it.
        {"name": "paid_lower", "title": "T", "output_type": "image", "tags": ["api"]},
        {"name": "paid_mixed", "title": "T", "output_type": "image", "tags": ["Api"]},
        # A row with no `tags` key at all is local, not a KeyError.
        {"name": "no_tags", "title": "T", "output_type": "image"},
        # `tags` present but null, and a non-string item, are both tolerated.
        {"name": "null_tags", "title": "T", "output_type": "image", "tags": None},
        {"name": "odd_tags", "title": "T", "output_type": "image", "tags": [None, 7]},
    ]
    _patch_ls(monkeypatch, rows)

    flags = {r["name"]: r["api"] for r in _rows(server.search_templates())}
    assert flags == {
        "local_one": False,
        "paid_upper": True,
        "paid_lower": True,
        "paid_mixed": True,
        "no_tags": False,
        "null_tags": False,
        "odd_tags": False,
    }
    assert all(isinstance(v, bool) for v in flags.values())


def test_search_templates_exclude_api_page_is_all_api_false(monkeypatch):
    """The flag and the filter read the same tag, so they can never disagree.

    `exclude_api=True` drops the API rows; whatever survives must therefore
    report `api: false`. A second copy of the tag test is what would let these
    two drift apart.
    """
    _patch_ls(monkeypatch)
    kept = _rows(server.search_templates(exclude_api=True))
    assert kept and all(r["api"] is False for r in kept)


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


def test_search_templates_forwards_gallery_filters(patched_run):
    """tag/type/model/provider become the corresponding `templates ls` flags."""
    calls = patched_run(envelope(data=_payload()))

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

    # The fixture tags are upper-case; a lower-case one drops just the same.
    lowered = [dict(r, tags=[t.lower() for t in r["tags"]]) for r in ROWS]
    _patch_ls(monkeypatch, lowered)
    assert set(_names(server.search_templates(exclude_api=True))) == set(names)


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


def test_search_templates_rejects_embedded_nul_filter_values(monkeypatch):
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    _patch_ls(monkeypatch)
    for kwargs in (
        {"tag": "a\0"},
        {"type": "a\0"},
        {"model": "a\0"},
        {"provider": "a\0"},
    ):
        with pytest.raises(server.ComfyCliError, match="embedded NUL"):
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


def test_get_template_argv(patched_run):
    """Passthrough: `comfy --json --where local templates show <name>`."""
    calls = patched_run(envelope(data={"name": "flux_dev", "nodes": 12}))

    result = server.get_template("flux_dev", check_local=False)

    assert calls[0]["cmd"][4:] == ["templates", "show", "flux_dev"]
    # The metadata rides through untouched; only `local_check` is added.
    assert {k: v for k, v in result.items() if k != "local_check"} == {
        "name": "flux_dev",
        "nodes": 12,
    }
    assert result["local_check"] == {
        "checked": False,
        "reason": "not_requested",
        "summary": "not checked against your ComfyUI install (check_local=False).",
    }
    assert len(calls) == 1  # `check_local=False` costs exactly one call


def test_get_template_returns_drifted_payload_untouched(monkeypatch):
    """A non-dict `templates show` payload has nowhere to attach a check."""
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: ["not", "a", "dict"])
    assert server.get_template("flux_dev") == ["not", "a", "dict"]


# `get_template` / `fetch_template` leading-dash + NUL rejection is covered by
# `test_get_template_rejects_option_like_name`,
# `test_fetch_template_rejects_option_like_name_and_out_path` and
# `test_template_tools_reject_embedded_nul` below (landed on main in #86).


def test_fetch_template_argv_and_returns_abspath(patched_run, tmp_path):
    """Passthrough argv is `templates fetch <name> --out <path>`; returns the abs path."""
    calls = patched_run(envelope(data=None))

    out = tmp_path / "flux.json"
    result = server.fetch_template("flux_dev", str(out), check_local=False)

    assert calls[0]["cmd"][4:] == ["templates", "fetch", "flux_dev", "--out", str(out)]
    assert result["path"] == str(out)  # tmp_path is already absolute
    assert os.path.isabs(result["path"])
    assert result["local_check"]["reason"] == "not_requested"
    assert len(calls) == 1  # no validate call when the check is skipped


def test_fetch_template_resolves_relative_path(monkeypatch):
    """A relative out_path is returned as an absolute path (ready for run_workflow)."""
    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: None)
    result = server.fetch_template("flux_dev", "flux.json", check_local=False)
    assert result["path"] == os.path.abspath("flux.json")
    assert os.path.isabs(result["path"])


def test_get_template_rejects_option_like_name(monkeypatch):
    """A leading-dash name is refused before any child spawns.

    ``name`` is a bare positional on ``templates show``, so comfy-cli reads a
    dash-leading value as an option rather than the template to show.
    """

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.get_template("--help")


def test_fetch_template_rejects_option_like_name_and_out_path(monkeypatch):
    """Both the ``name`` positional and the ``--out`` value are guarded."""

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    with pytest.raises(server.ComfyCliError, match="leading '-'"):
        server.fetch_template("--help", "/tmp/flux.json")

    # The escape hatch is named in the error, so a genuinely dash-leading
    # filename stays reachable as `./-flux.json`.
    with pytest.raises(server.ComfyCliError, match=r"leading '-'.*\./"):
        server.fetch_template("flux_dev", "--help")


def test_fetch_template_rejects_an_oversized_out_path(no_spawn):
    """An oversized `out_path` is refused before it can reach argv.

    `--out`'s value rides the same argv as everything else, where the OS rejects
    an oversized exec with an `OSError` (`E2BIG`) that `_run_comfy_raw` does not
    convert — its `try` wraps only `communicate()`, not the `Popen(...)` that
    raises. The cap is what turns that into a clean `ComfyCliError`.
    """
    oversized = "/tmp/" + "o" * server._MAX_PATH_ARG_LEN + ".json"

    with pytest.raises(server.ComfyCliError, match="exceeds") as excinfo:
        server.fetch_template("flux_dev", oversized)

    # Length-not-value: the size check runs ahead of `_reject_option_like` /
    # `_reject_nul`, whose echoes would name the value instead of its size.
    assert oversized not in str(excinfo.value)


def test_fetch_template_reports_a_bad_name_ahead_of_an_oversized_out_path(no_spawn):
    """Size-before-value is a PER-VALUE rule, not a whole-function one.

    `out_path`'s cap sits ahead of `out_path`'s own guards so an oversized path
    is named as a size rather than as a shape. Hoisting it above `name`'s check
    too would make a call with both mistakes report the wrong argument.
    """
    oversized = "/tmp/" + "o" * server._MAX_PATH_ARG_LEN + ".json"

    with pytest.raises(server.ComfyCliError, match="invalid name") as excinfo:
        server.fetch_template("--help", oversized)

    assert "exceeds" not in str(excinfo.value)


def test_fetch_template_allows_an_out_path_at_the_ceiling(patched_run):
    """The boundary value itself rides through to argv — the cap is generous."""
    calls = patched_run(envelope(data=None))
    at_ceiling = "/tmp/" + "o" * (server._MAX_PATH_ARG_LEN - len("/tmp/"))
    assert len(at_ceiling) == server._MAX_PATH_ARG_LEN

    server.fetch_template("flux_dev", at_ceiling, check_local=False)

    assert calls[0]["cmd"][4:] == [
        "templates",
        "fetch",
        "flux_dev",
        "--out",
        at_ceiling,
    ]


def test_template_tools_reject_embedded_nul(monkeypatch):
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""

    def boom(*a, **k):
        raise AssertionError("no comfy-cli child may be spawned")

    monkeypatch.setattr(server, "_run_comfy", boom)

    for call in (
        lambda: server.get_template("flux\0dev"),
        lambda: server.fetch_template("flux\0dev", "/tmp/flux.json"),
        lambda: server.fetch_template("flux_dev", "/tmp/f\0.json"),
    ):
        with pytest.raises(server.ComfyCliError, match="embedded NUL"):
            call()


# --- local_check: does the user's install actually support this template? ----
#
# The gallery is served fresh while the install is whatever the user has, so a
# template can reference a node class — or a model option inside one — that this
# ComfyUI does not expose yet. The tools cross-check via `comfy validate` against
# the LIVE object_info and report it; the checks below pin the three outcomes
# that matter, especially the two that must NOT read as "your install can't run
# this": a check that could not run, and a vacuous pass.


def _fake_comfy(monkeypatch, handler):
    """Dispatch `_run_comfy` per subcommand; returns the recorded arg tuples."""
    calls: list[tuple] = []

    def fake(*args, **kwargs):
        calls.append(args)
        return handler(args)

    monkeypatch.setattr(server, "_run_comfy", fake)
    return calls


def _report(*, valid: bool, errors=(), warnings=(), **extra) -> dict:
    """A `comfy validate` payload, shaped like cmdline.py's `validate` emits."""
    return {
        "valid": valid,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": list(errors),
        "warnings": list(warnings),
        **extra,
    }


def test_fetch_template_reports_a_clean_local_check(monkeypatch, tmp_path):
    """A valid workflow: `validate` runs on the written file, verdict is runnable."""
    out = tmp_path / "flux.json"
    calls = _fake_comfy(
        monkeypatch,
        lambda args: (
            _report(valid=True, converted_from_ui=True)
            if args[0] == "validate"
            else None
        ),
    )

    result = server.fetch_template("flux_dev", str(out))

    assert calls[0] == ("templates", "fetch", "flux_dev", "--out", str(out))
    # Checked against the file just written, by absolute path.
    assert calls[1] == ("validate", "--workflow", str(out))
    assert result["path"] == str(out)
    assert result["local_check"]["checked"] is True
    assert result["local_check"]["runnable"] is True
    assert result["local_check"]["error_count"] == 0


def test_fetch_template_warns_when_the_install_lacks_a_model_option(
    monkeypatch, tmp_path
):
    """The reported case: the template names a model key this install has never seen.

    comfy-cli reports an invalid workflow as an envelope whose `ok` mirrors
    `valid` and whose `data` is the full report, so it reaches us as a raised
    `ComfyCliError` carrying that report — the verdict, not a failure to check.
    """
    out = tmp_path / "seedream.json"
    error = {
        "node_id": "12",
        "code": "invalid_enum_value",
        "message": "'seedream-5.0-pro' is not a valid value for input 'model'",
        "suggestions": ["seedream-5.0-lite", "seedream-4.0"],
    }

    def handler(args):
        if args[0] != "validate":
            return None
        raise server.ComfyCliError(
            "comfy validate --workflow x failed [unknown]: ",
            data=_report(valid=False, errors=[error], converted_from_ui=True),
        )

    _fake_comfy(monkeypatch, handler)
    result = server.fetch_template("seedream_5_pro", str(out))

    check = result["local_check"]
    assert check["checked"] is True
    assert check["runnable"] is False
    assert check["error_count"] == 1
    assert check["errors"] == [
        "node 12: 'seedream-5.0-pro' is not a valid value for input 'model' "
        "(this install has: seedream-5.0-lite, seedream-4.0)"
    ]
    assert "update comfyui" in check["summary"].lower()
    # Advisory, never a refusal: the file is still written and still handed back.
    assert result["path"] == str(out)


def test_fetch_template_masks_credentials_in_a_local_check_finding(
    monkeypatch, tmp_path
):
    """A finding quoting a credential-bearing input is masked in `local_check`.

    Same findings, same client, same mask as `validate_workflow`'s own relay:
    the validator quotes the offending widget value, and a workflow input can
    carry userinfo in a URL.
    """
    out = tmp_path / "wf.json"
    error = {
        "node_id": "4",
        "code": "invalid_value",
        "message": "'https://<user>:<pass>@example.invalid/a.safetensors' is not valid",
    }

    def handler(args):
        if args[0] != "validate":
            return None
        raise server.ComfyCliError(
            "comfy validate --workflow x failed [unknown]: ",
            data=_report(valid=False, errors=[error], converted_from_ui=True),
        )

    _fake_comfy(monkeypatch, handler)
    line = server.fetch_template("flux_dev", str(out))["local_check"]["errors"][0]

    assert "<pass>" not in line
    # The mask removes the credential, not the diagnostic.
    assert "example.invalid" in line
    assert line.startswith("node 4: ")


def test_fetch_template_does_not_deny_when_the_check_cannot_run(monkeypatch, tmp_path):
    """No live object_info (ComfyUI down) is `checked: false`, NOT `runnable: false`.

    An unreachable node catalog says nothing about the template. Reporting it as
    an incompatibility would send the user chasing an install problem that may
    not exist.
    """
    out = tmp_path / "flux.json"

    def handler(args):
        if args[0] != "validate":
            return None
        raise server.ComfyCliError(
            "comfy validate --workflow x failed [cql_no_graph]: no object_info",
            code="cql_no_graph",
        )

    _fake_comfy(monkeypatch, handler)
    check = server.fetch_template("flux_dev", str(out))["local_check"]

    assert check == {
        "checked": False,
        "reason": "check_unavailable",
        "summary": check["summary"],
    }
    assert "runnable" not in check
    assert "launch_comfyui" in check["summary"]
    assert "cql_no_graph" in check["summary"]  # the cause survives for the user


def test_fetch_template_does_not_report_a_vacuous_pass(monkeypatch, tmp_path):
    """An un-converted UI-format graph validates zero nodes — not a clean bill.

    Gallery templates are UI exports; a comfy-cli too old to lower one to API
    format emits `non_node_key` warnings, checks nothing, and calls it valid.
    """
    out = tmp_path / "flux.json"
    _fake_comfy(
        monkeypatch,
        lambda args: (
            _report(
                valid=True,
                warnings=[{"code": "non_node_key", "message": "ignored key 'links'"}],
            )
            if args[0] == "validate"
            else None
        ),
    )

    check = server.fetch_template("flux_dev", str(out))["local_check"]

    assert check["checked"] is False
    assert check["reason"] == "workflow_not_converted"


def test_fetch_template_survives_a_drifted_validate_payload(monkeypatch, tmp_path):
    """A payload that is not a validate report is `checked: false`, never a verdict."""
    out = tmp_path / "flux.json"
    _fake_comfy(
        monkeypatch,
        lambda args: {"something": "else"} if args[0] == "validate" else None,
    )

    check = server.fetch_template("flux_dev", str(out))["local_check"]

    assert check["checked"] is False
    assert check["reason"] == "unexpected_payload"


def test_get_template_checks_via_a_scratch_copy_and_cleans_up(monkeypatch):
    """`templates show` has no graph, so the check fetches one — to scratch space."""
    fetched: list[str] = []

    def handler(args):
        if args[0] == "templates" and args[1] == "show":
            return {"template": {"name": "flux_dev"}}
        if args[0] == "templates" and args[1] == "fetch":
            fetched.append(args[4])
            return None
        return _report(valid=True, converted_from_ui=True)

    calls = _fake_comfy(monkeypatch, handler)
    result = server.get_template("flux_dev")

    assert result["template"] == {"name": "flux_dev"}
    assert result["local_check"]["runnable"] is True
    # Fetched to a scratch path (never the caller's cwd) and validated there...
    assert calls[1][:4] == ("templates", "fetch", "flux_dev", "--out")
    assert calls[2][:2] == ("validate", "--workflow")
    assert calls[2][2] == fetched[0]
    # ...and the scratch directory is gone afterwards.
    assert not os.path.exists(os.path.dirname(fetched[0]))


def test_get_template_reports_an_unfetchable_template_as_unchecked(monkeypatch):
    """A failed scratch fetch is a check that did not happen, not a bad template."""

    def handler(args):
        if args[0] == "templates" and args[1] == "show":
            return {"template": {"name": "flux_dev"}}
        raise server.ComfyCliError("template fetch failed [template_fetch_failed]")

    _fake_comfy(monkeypatch, handler)
    check = server.get_template("flux_dev")["local_check"]

    assert check["checked"] is False
    assert check["reason"] == "template_fetch_failed"
