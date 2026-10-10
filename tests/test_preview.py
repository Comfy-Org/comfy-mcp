"""Tests for the ``preview_media`` thin wrapper over ``comfy preview``."""

from __future__ import annotations

import base64

import pytest
from conftest import envelope

from comfy_mcp import argv, server

_FAKE_PNG = b"\x89PNG\r\n\x1a\npreview-pixels"


def test_preview_media_wraps_cli_and_inlines_the_generated_png(patched_run, tmp_path):
    media = tmp_path / "clip.mp4"
    out = tmp_path / "clip-contact-sheet.png"
    metadata = {
        "input": str(media),
        "kind": "video",
        "preview": str(out),
        "duration": 12.5,
        "fps": 24.0,
        "has_audio": True,
    }

    def render_preview(_cmd):
        out.write_bytes(_FAKE_PNG)

    calls = patched_run(envelope(data=metadata), on_spawn=render_preview)

    result = server.preview_media(str(media), str(out))

    assert calls[0]["cmd"][4:] == [
        "preview",
        str(media),
        "--out",
        str(out),
        "--grid",
        "4x3",
        "--width",
        "480",
    ]
    assert calls[0]["timeout"] == pytest.approx(180.0)
    assert result[0] == metadata
    assert isinstance(result[1], server.Image)
    content = result[1].to_image_content()
    assert content.mime_type == "image/png"
    assert base64.b64decode(content.data) == _FAKE_PNG


def test_preview_media_forwards_grid_width_and_omits_optional_out_path(patched_run):
    metadata = {"kind": "audio", "preview": "track.preview.png"}
    calls = patched_run(envelope(data=metadata))

    assert (
        server.preview_media("track.wav", grid="3x2", width=720, inline_image=False)
        == metadata
    )
    assert calls[0]["cmd"][4:] == [
        "preview",
        "track.wav",
        "--grid",
        "3x2",
        "--width",
        "720",
    ]


def test_preview_media_resolves_relative_output_from_comfy_project(
    patched_run, tmp_path, monkeypatch
):
    preview = tmp_path / "clip.preview.png"
    preview.write_bytes(_FAKE_PNG)
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path))
    calls = patched_run(envelope(data={"kind": "video", "preview": "clip.preview.png"}))

    result = server.preview_media("clip.mp4")

    assert calls[0]["cwd"] == str(tmp_path)
    assert isinstance(result[1], server.Image)
    assert base64.b64decode(result[1].to_image_content().data) == _FAKE_PNG


def test_preview_media_keeps_metadata_when_inline_file_is_missing(
    patched_run, tmp_path
):
    metadata = {"kind": "video", "preview": str(tmp_path / "missing.preview.png")}
    patched_run(envelope(data=metadata))

    assert server.preview_media("clip.mp4") == [metadata]


def test_preview_media_does_not_inline_a_non_png_engine_path(patched_run, tmp_path):
    rendered = tmp_path / "clip.webp"
    rendered.write_bytes(b"webp")
    metadata = {"kind": "video", "preview": str(rendered)}
    patched_run(envelope(data=metadata))

    assert server.preview_media("clip.mp4") == [metadata]


def test_preview_media_does_not_inline_an_oversized_preview(
    patched_run, tmp_path, monkeypatch
):
    rendered = tmp_path / "clip.preview.png"
    rendered.write_bytes(b"x" * 11)
    metadata = {"kind": "video", "preview": str(rendered)}
    monkeypatch.setattr(server, "_INLINE_IMAGE_MAX_BYTES", 10)
    patched_run(envelope(data=metadata))

    assert server.preview_media("clip.mp4") == [metadata]


@pytest.mark.parametrize(
    "media_path",
    ["--help", "clip\x00.mp4", "x" * (argv._MAX_PATH_ARG_LEN + 1)],
)
def test_preview_media_rejects_invalid_input_path_before_spawn(no_spawn, media_path):
    with pytest.raises(server.ComfyCliError):
        server.preview_media(media_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"out_path": "preview\x00.png"},
        {"out_path": "x" * (argv._MAX_PATH_ARG_LEN + 1)},
        {"grid": "4x3\x00"},
    ],
)
def test_preview_media_rejects_invalid_options_before_spawn(no_spawn, kwargs):
    with pytest.raises(server.ComfyCliError):
        server.preview_media("clip.mp4", **kwargs)


def test_preview_media_inline_false_does_not_read_the_reported_path(
    patched_run, tmp_path
):
    reported = tmp_path / "missing.preview.png"
    metadata = {"kind": "image", "preview": str(reported)}
    patched_run(envelope(data=metadata))

    assert server.preview_media("image.png", inline_image=False) == metadata


def test_preview_media_is_taught_in_the_handshake():
    instructions = server.mcp.instructions

    assert "preview_media" in instructions
    assert "video/audio" in instructions
    assert "fetch_outputs" in instructions
