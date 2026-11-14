"""Unit tests for the comfy-cli envelope parser."""

import json

import pytest

from comfy_mcp.server import _last_json_object


def test_prefers_envelope_over_plain_json():
    out = '{"foo": 1}\n{"type": "envelope", "ok": true, "data": {"x": 1}}\n'
    assert _last_json_object(out) == {
        "type": "envelope",
        "ok": True,
        "data": {"x": 1},
    }


def test_ignores_non_json_noise():
    out = 'loading...\nprogress 50%\n{"type": "envelope", "ok": true, "data": null}\n'
    assert _last_json_object(out) == {"type": "envelope", "ok": True, "data": None}


def test_falls_back_to_last_json_when_no_envelope():
    assert _last_json_object('{"a": 1}\n{"b": 2}\n') == {"b": 2}


def test_returns_none_when_no_json():
    assert _last_json_object("just text\nmore text\n") is None


# `str.splitlines()` treats these as line breaks, and JSON allows them
# unescaped inside a string. U+2028 is the one the MiniMax H3 image-to-video
# template ships in its default prompt.
@pytest.mark.parametrize(
    "separator",
    ["\u2028", "\u2029", "\u0085"],
    ids=["U+2028", "U+2029", "U+0085"],
)
def test_unicode_line_separator_stays_inside_a_json_string(separator):
    out = (
        '{"type": "envelope", "ok": true, "data": {"prompt": "fade'
        + separator
        + 'out"}}\n'
    )
    assert separator in out
    assert json.loads(out)["data"]["prompt"] == f"fade{separator}out"
    assert _last_json_object(out) == {
        "type": "envelope",
        "ok": True,
        "data": {"prompt": f"fade{separator}out"},
    }


def test_separator_in_an_earlier_line_does_not_drop_the_envelope():
    earlier = '{"note": "fade\u2028out"}\n'
    envelope = '{"type": "envelope", "ok": true, "data": null}\n'
    assert _last_json_object(earlier + envelope) == {
        "type": "envelope",
        "ok": True,
        "data": None,
    }


def test_crlf_delimited_envelopes_still_parse():
    out = '{"foo": 1}\r\n{"type": "envelope", "ok": true, "data": {"x": 1}}\r\n'
    assert _last_json_object(out) == {
        "type": "envelope",
        "ok": True,
        "data": {"x": 1},
    }


def test_bare_cr_before_an_envelope_still_splits():
    """A progress redraw's bare ``\\r`` is a line break; U+2028 inside is not.

    ``_run_comfy_async`` decodes raw bytes, so a foreground download's
    ``100%\\r{envelope}`` reaches this parser without universal-newline
    translation. A raw ``\\r`` cannot occur unescaped inside a JSON string.
    """
    out = (
        'download 100%\r{"type": "envelope", "ok": true, "data": '
        '{"prompt": "fade\u2028out"}}\n'
    )
    assert _last_json_object(out) == {
        "type": "envelope",
        "ok": True,
        "data": {"prompt": "fade\u2028out"},
    }
