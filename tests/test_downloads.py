"""Regression tests for the background model-download family.

Split out of ``test_wrapper.py``, which had grown to own several whole tool
groups alongside the genuine wrapper core. This file is the
``download_model`` / ``download_status`` / ``wait_for_download`` /
``cancel_download`` group: the argument guards, the submit-then-poll
background flow, the companions, and the legacy synchronous fallback for a
comfy-cli that predates ``model download --background``.

``NO_SUCH_OPTION_STDERR`` is shared with ``test_wrapper.py`` — the
``_is_missing_option_error`` tests there use it too — so it lives in
``conftest.py`` with the other shared helpers rather than being duplicated or
cross-imported between two test modules.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import PurePosixPath, PureWindowsPath

import pytest
from conftest import NO_SUCH_OPTION_STDERR, envelope

from comfy_mcp import failure_log, server


def _download_model(*args, **kwargs):
    """Drive the async ``download_model`` from these synchronous tests.

    ``download_model`` went async when it started submitting to a background
    worker and polling for it (a blocking poll loop on the event loop would
    stall every other concurrent MCP request). Its contract is otherwise
    unchanged, and an inline ``asyncio.run`` at each of the ~30 call sites below
    — several of them multi-line — would bury what each test asserts.
    """
    return asyncio.run(server.download_model(*args, **kwargs))


def _missing_background_error() -> server.ComfyCliError:
    """The `ComfyCliError` an old comfy-cli's `--background` rejection produces."""
    return server.ComfyCliError(
        f"comfy-cli returned no JSON (exit 2). stderr: {NO_SUCH_OPTION_STDERR} "
        "| stdout: <empty>",
        no_envelope=True,
        returncode=2,
    )


def _reject_background(*args, **kwargs):
    """Stand in for ``_run_comfy`` on a CLI too old to know ``--background``.

    The whole-of-``_run_comfy`` counterpart to :func:`legacy_comfy_cli`, for the
    tests that only care about what the FALLBACK is handed and stub
    ``_run_comfy_async`` themselves — there is no second call to pass through, so
    a per-argv dispatch would be ceremony.
    """
    raise _missing_background_error()


@pytest.fixture
def legacy_comfy_cli(monkeypatch):
    """Simulate a comfy-cli too old to know ``model download --background``.

    Answers only the ``--background`` submit with the Click usage error such a
    CLI produces and passes every other call through to the REAL ``_run_comfy``.
    The fallback itself runs on ``_run_comfy_async``, which this leaves entirely
    alone, so it still exercises the genuine plain-exit machinery underneath
    (``patched_async_run`` + ``_synthesize_plain_result``) rather than a stub of
    it. This is a per-ARGV reply rather than a second spawn shape, which is why
    it lives here instead of in ``conftest``.
    """
    real = server._run_comfy

    def dispatch(*args, **kwargs):
        if "--background" in args:
            raise _missing_background_error()
        return real(*args, **kwargs)

    monkeypatch.setattr(server, "_run_comfy", dispatch)


def test_download_model_url_only(patched_run):
    """download_model submits `model download --url … --background`, returns its data."""
    submit = {"download_id": "a1b2c3d4e5f6", "dest": "/models/x.safetensors"}
    calls = patched_run({"type": "envelope", "ok": True, "data": submit})

    assert _download_model("https://hf.co/x.safetensors", wait=False) == submit

    cmd = calls[0]["cmd"]
    assert cmd[1:4] == ["--json", "--where", "local"]  # global flags first
    # SINGULAR `model` verb group (download engine), not the plural catalog.
    assert cmd[4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/x.safetensors",
        "--background",
    ]
    # The submit is metadata-only, so it gets its own budget — never the
    # transfer's 1800s, which is what used to hold the MCP request open.
    assert calls[0]["timeout"] == server._DOWNLOAD_SUBMIT_TIMEOUT


def test_download_model_threads_relative_path(patched_run):
    """--relative-path is appended only when provided."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://hf.co/l.safetensors", relative_path="models/loras", wait=False
    )

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/l.safetensors",
        "--relative-path",
        "models/loras",
        "--background",
    ]


def test_download_model_threads_filename(patched_run):
    """--filename is appended only when provided."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://hf.co/c.safetensors", filename="renamed.safetensors", wait=False
    )

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://hf.co/c.safetensors",
        "--filename",
        "renamed.safetensors",
        "--background",
    ]


def test_download_model_threads_all_optionals(patched_run):
    """Both optional args thread through together, in order, only when set."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://civitai.com/api/download/models/42",
        relative_path="models/checkpoints",
        filename="sd.safetensors",
        wait=False,
    )

    assert calls[0]["cmd"][4:] == [
        "model",
        "download",
        "--url",
        "https://civitai.com/api/download/models/42",
        "--relative-path",
        "models/checkpoints",
        "--filename",
        "sd.safetensors",
        "--background",
    ]


def test_download_model_omits_absent_optionals(patched_run):
    """Neither optional flag is emitted when the argument is left unset."""
    calls = patched_run(envelope(data=_submit()))

    _download_model("https://hf.co/x.safetensors", wait=False)

    cmd = calls[0]["cmd"]
    assert "--relative-path" not in cmd
    assert "--filename" not in cmd


def test_download_model_rejects_option_like_url(patched_run):
    """A leading-dash url is refused so comfy-cli can't parse it as a flag."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid url"):
        _download_model("--config")

    assert calls == []


def test_download_model_rejects_option_like_relative_path(patched_run):
    """A leading-dash relative_path is refused (argument injection guard)."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path="-rf")

    assert calls == []


def test_download_model_rejects_option_like_filename(patched_run):
    """A leading-dash filename is refused (argument injection guard)."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        _download_model("https://hf.co/x.safetensors", filename="--evil")

    assert calls == []


@pytest.mark.parametrize(
    "bad_url", ["file:///etc/passwd", "ftp://host/x", "/etc/passwd"]
)
def test_download_model_rejects_non_http_scheme(bad_url, patched_run):
    """Only http(s) URLs are allowed; file://, ftp:// and bare paths are refused."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid url"):
        _download_model(bad_url)

    assert calls == []


@pytest.mark.parametrize(
    "bad_path",
    ["../../etc", "models/../../etc", "/abs/models", "..\\..\\etc", "C:evil"],
)
def test_download_model_rejects_traversal_relative_path(bad_path, patched_run):
    """relative_path must stay within the models dir: no `..`, absolute paths,
    or a drive prefix (``C:evil`` has no separator but is drive-relative on
    Windows)."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


# --- Windows-shaped escapes are refused on EVERY host, including Linux CI ----
#
# The guard runs wherever the MCP server runs; the write happens wherever
# comfy-cli runs. Judging these from the string's own shape plus
# `ntpath.splitdrive` — never the host's `os.path` — is what makes them fail on a
# POSIX box too, so these cases are the regression pins for that: to `posixpath`
# on Linux CI every string below is an ordinary relative name.


@pytest.mark.parametrize(
    "bad_path",
    [
        "C:evil",  # drive-RELATIVE: no separator, resolves against C:'s cwd
        "C:/Windows",  # drive-absolute with a forward slash
        "C:\\Windows",  # drive-absolute with a backslash
        "\\\\server\\share",  # UNC root — no `:` at all, so `:`-checks miss it
        "\\\\server\\share\\evil",  # UNC with a trailing path
        "\\evil",  # root-relative on the current drive
    ],
)
def test_download_model_rejects_windows_drive_relative_path(bad_path, patched_run):
    """A Windows drive/UNC/root-relative ``relative_path`` is refused before any
    child spawns — on Linux CI as much as on Windows."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


# --- A colon ANYWHERE, not just a leading drive ------------------------------
#
# `ntpath.splitdrive` only ever reads a LEADING drive, so it sees nothing in
# `models/C:evil` — every colon case pinned above (`C:evil`, `C:/Windows`,
# `C:\Windows`) happens to carry that leading drive, and would still be refused
# with the `":" in relative_path` term deleted. These are the shapes that term
# alone catches, so it cannot be dropped as redundant with the drive check.


@pytest.mark.parametrize(
    "bad_path",
    [
        "models/C:evil",  # drive-relative in a LATER segment: splitdrive is blind
        "models/loras:ads",  # NTFS alternate-data-stream syntax (`name:stream`)
        "models/x:y",
        "models/evil:",  # trailing colon — still an ADS name on NTFS
    ],
)
def test_download_model_rejects_mid_path_colon_relative_path(bad_path, patched_run):
    """A colon in a NON-leading segment is refused before any child spawns.

    Regression pin for the `":" in relative_path` term specifically:
    `ntpath.splitdrive` reads only a leading drive, so it reports no drive for
    any value here and these shapes are refused by that term ALONE. Without this
    test the whole suite stays green with the term deleted as "redundant with
    splitdrive", and `models/C:evil` / `models/loras:ads` start being accepted.
    The colon term lives in the FIRST raise, so the diagnosis is traversal.
    """
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match=r"invalid relative_path.*traversal"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


def test_download_model_rejects_mid_path_colon_resetting_the_anchor(patched_run):
    """Named regression pin for what the mid-path colon rejection prevents.

    A drive-carrying component joined on Windows does not extend the path — it
    RESETS the anchor to that drive's own working directory, discarding the
    workspace entirely. `<workspace>/models/C:evil` is not a folder under the
    models dir; it is whatever `C:evil` resolves to on drive C:, outside the
    workspace the guard claims to confine the write to.
    """
    calls = patched_run(envelope(data=_submit()))

    # Pin the mechanism, so this test fails loudly if pathlib ever changes: the
    # workspace anchor is discarded by the drive-carrying component.
    assert PureWindowsPath("D:/ws") / "C:evil" == PureWindowsPath("C:evil")

    with pytest.raises(server.ComfyCliError, match=r"invalid relative_path.*traversal"):
        _download_model(
            "https://attacker.example/payload", relative_path="models/C:evil"
        )

    assert calls == []


# --- A `..` wearing trailing spaces/periods is still a `..` ------------------
#
# Windows strips trailing spaces and periods from every path component at
# syscall time, so `".. "` and `"..."` reach the filesystem as `..` while an
# equality check against `".."` says they are something else. `normpath` does
# not collapse them either, which is why the guard has to match the dot RUN
# rather than the exact string. Like the drive cases above, these have to be
# refused from a POSIX host too — Linux keeps `".. "` as a literal directory
# name, so nothing here fires locally on CI unless the string check does it.


@pytest.mark.parametrize(
    "bad_path",
    [
        "models/.. /evil",  # trailing space: normalizes back to `..` on Windows
        "models/.../evil",  # dot run: strips to `..`
        "models/... /evil",  # both
        ".. ",  # the whole value is a disguised `..`
        "models/. /evil",  # a disguised `.` — not an escape, but not a real name
    ],
)
def test_download_model_rejects_dot_run_relative_path(bad_path, patched_run):
    """A `..` disguised by trailing spaces/periods is refused before any child
    spawns — on Linux CI as much as on Windows."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


@pytest.mark.parametrize(
    "good_path",
    [
        "models/.hidden",  # a leading dot is an ordinary name, not a dot run
        "models/v1.5",  # interior periods survive `strip(" .")`
        "models/loras/",  # trailing slash: an EMPTY segment, still accepted
        "models//loras",  # doubled slash, likewise
    ],
)
def test_download_model_accepts_dotted_but_ordinary_relative_path(
    good_path, patched_run
):
    """The dot-run check must not widen into ordinary names or empty segments —
    only a component that is *nothing but* dots and spaces is a disguised `..`."""
    calls = patched_run(envelope(data=_submit()))

    _download_model("https://hf.co/x.safetensors", relative_path=good_path, wait=False)

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--relative-path") + 1] == good_path


# --- relative_path must land in the MODELS tree, not just inside the workspace -
#
# comfy-cli joins `--relative-path` to the WORKSPACE ROOT (`local_filepath =
# get_workspace() / relative_path / local_filename`, defaulting to
# `DEFAULT_COMFY_MODEL_PATH = "models"`) — which is why the documented shape is
# `models/loras` rather than a bare `loras`. Every value below is traversal-clean,
# so the checks above pass it happily; only the first-segment check keeps the
# write out of the workspace's other top-level directories.


@pytest.mark.parametrize(
    "bad_path",
    [
        "custom_nodes/pwn",  # ComfyUI's import path — see the RCE pin below
        "custom_nodes/x/y",  # deeper, same tree
        "user",  # sibling top-level workspace dirs
        "input",
        "output",
        "web",
        "loras",  # a BARE folder name is rejected, not assumed to mean models/
        "modelsx",  # a prefix of `models` is a different directory
        "models2/loras",
        "Models/loras",  # matched exactly: no case-folding a security guard
        "~",  # comfy-cli expanduser()s this, escaping the workspace entirely
        "~/evil",
    ],
)
def test_download_model_rejects_non_models_relative_path(bad_path, patched_run):
    """A traversal-clean ``relative_path`` that does not start at ``models`` is
    refused before any child spawns."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


def test_download_model_rejects_custom_nodes_init_py_rce(patched_run):
    """Named regression pin for the RCE shape this check exists for.

    ComfyUI imports `custom_nodes/*/__init__.py` on startup, so a write there is
    attacker-controlled code on the import path — code execution on the next
    ComfyUI restart, on every platform. Both arguments are individually legal
    (`custom_nodes/pwn` is traversal-clean; `__init__.py` is a bare filename);
    it is the models-tree confinement that refuses the combination.
    """
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model(
            "https://attacker.example/payload",
            relative_path="custom_nodes/pwn",
            filename="__init__.py",
        )

    assert calls == []


@pytest.mark.parametrize(
    "bad_path",
    ["../../etc", "models/../../etc", "/abs/models", "C:evil", "models/.. /evil"],
)
def test_download_model_traversal_still_reports_as_traversal(bad_path, patched_run):
    """Ordering pin: the models-tree check runs AFTER the traversal checks, so a
    traversal string keeps its own (more specific) diagnosis rather than being
    relabelled as a wrong-folder error."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match=r"invalid relative_path.*traversal"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


@pytest.mark.parametrize(
    "good_path",
    [
        "models",  # the models dir itself
        "models/loras",
        "models/checkpoints",
        "models/loras/",  # trailing slash: an empty trailing segment
        "models//loras",  # doubled slash
    ],
)
def test_download_model_forwards_models_relative_path_unchanged(good_path, patched_run):
    """Accepted values are forwarded to comfy-cli VERBATIM — the guard rejects,
    it never rewrites the argument."""
    calls = patched_run(envelope(data=_submit()))

    _download_model("https://hf.co/x.safetensors", relative_path=good_path, wait=False)

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--relative-path") + 1] == good_path


@pytest.mark.parametrize(
    "bad_path",
    [
        "models\\loras",  # the case below, as a plain parametrization
        "models\\checkpoints\\sdxl",
        "models\\",  # trailing separator, still a `\`
        "models/loras\\ckpt",  # mixed spelling
    ],
)
def test_download_model_rejects_backslash_relative_path(bad_path, patched_run):
    """A `\\` separator is refused even when every segment is otherwise legal.

    The guard splits on `\\` to decide, but comfy-cli receives the value verbatim
    — so on a POSIX host the two disagree and the write misses the models tree
    (see the named pin below). `/` reaches the same directory on Windows, so this
    costs a spelling, not a destination.
    """
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match=r"invalid relative_path.*separator"):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


def test_download_model_rejects_backslash_escaping_models_tree(patched_run):
    """Named regression pin for what the `\\` rejection actually prevents.

    `models\\loras` splits to segments `models` + `loras`, so the first-segment
    check passes it — but pathlib on a POSIX host reads the backslash as an
    ordinary character, making `<workspace>/models\\loras` a top-level directory
    SIBLING of the models dir rather than a folder inside it. The write would land
    outside the tree the guard claims to confine it to.
    """
    calls = patched_run(envelope(data=_submit()))

    # Pin the mechanism, so this test fails loudly if pathlib ever changes: the
    # backslash is one literal path component, not a separator, off POSIX.
    assert (PurePosixPath("/ws") / "models\\loras").parts == (
        "/",
        "ws",
        "models\\loras",
    )

    with pytest.raises(server.ComfyCliError, match="invalid relative_path"):
        _download_model(
            "https://attacker.example/payload", relative_path="models\\loras"
        )

    assert calls == []


@pytest.mark.parametrize(
    ("bad_path", "expected"),
    [
        # Ordering pin: the `\` check runs LAST, so the security-meaningful
        # diagnoses keep their own (more specific) message.
        ("models\\..\\evil", "traversal"),  # `\`-split makes `..` a real segment
        ("\\evil", "traversal"),  # root-relative: an empty leading segment
        ("custom_nodes\\pwn", "must be the models dir"),  # wrong tree first
        ("loras\\x", "must be the models dir"),
    ],
)
def test_download_model_backslash_check_is_ordered_last(
    bad_path, expected, patched_run
):
    """A `\\` value that is ALSO a traversal or a wrong-tree value reports as that,
    not as a separator-spelling complaint."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match=expected):
        _download_model("https://hf.co/x.safetensors", relative_path=bad_path)

    assert calls == []


@pytest.mark.parametrize("bad_name", [".. ", "...", ". ", "... ", " "])
def test_download_model_rejects_dot_run_filename(bad_name, patched_run):
    """Same disguise on the bare-name side: `".. "` is a `..`, not a filename."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        _download_model("https://hf.co/x.safetensors", filename=bad_name)

    assert calls == []


@pytest.mark.parametrize("good_name", [".gitkeep", "v1.5.safetensors", "model."])
def test_download_model_accepts_dotted_but_ordinary_filename(good_name, patched_run):
    """Regression pin for the dot-run filename check: a name that merely
    *contains* dots still has something left after `strip(" .")`."""
    calls = patched_run(envelope(data=_submit()))

    _download_model("https://hf.co/x.safetensors", filename=good_name, wait=False)

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--filename") + 1] == good_name


@pytest.mark.parametrize(
    "bad_name", ["../evil", "sub/dir.safetensors", "..", "a\\b", "C:evil.dll"]
)
def test_download_model_rejects_pathy_filename(bad_name, patched_run):
    """filename must be a bare name: no separators, `..`, or a drive prefix to
    escape the dir (``C:evil.dll`` has no separator but is drive-relative on
    Windows)."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        _download_model("https://hf.co/x.safetensors", filename=bad_name)

    assert calls == []


@pytest.mark.parametrize(
    "bad_name",
    [
        "C:evil.exe",  # drive-RELATIVE: a bare-name check alone lets this pass
        "C:/evil.exe",
        "\\\\server\\share\\evil.exe",  # UNC
        "\\evil.exe",  # root-relative on the current drive
    ],
)
def test_download_model_rejects_windows_drive_relative_filename(bad_name, patched_run):
    """A Windows drive/UNC/root-relative ``filename`` is refused before any child
    spawns — on Linux CI as much as on Windows."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        _download_model("https://hf.co/x.safetensors", filename=bad_name)

    assert calls == []


@pytest.mark.parametrize("bad_name", ["evil:ads", "model:stream.safetensors"])
def test_download_model_rejects_non_drive_colon_filename(bad_name, patched_run):
    """The bare-name colon check reaches past a drive prefix.

    The drive shapes (`C:evil.dll`, `C:evil.exe`) are pinned above, but every one
    of them is also what `ntpath.splitdrive` would report — so they leave the
    colon term free to be narrowed to a drive check. These are `name:stream`
    NTFS alternate-data-stream values with no drive at all: refused by the `":"
    in filename` term alone.
    """
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="invalid filename"):
        _download_model("https://hf.co/x.safetensors", filename=bad_name)

    assert calls == []


def test_download_model_still_accepts_ordinary_path_and_name(patched_run):
    """Regression pin: the Windows-shaped rejections above must not catch the
    ordinary values these tools are actually called with."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://hf.co/x.safetensors",
        relative_path="models/loras",
        filename="x.safetensors",
        wait=False,
    )

    cmd = calls[0]["cmd"]
    assert "--relative-path" in cmd and cmd[cmd.index("--relative-path") + 1] == (
        "models/loras"
    )
    assert "--filename" in cmd and cmd[cmd.index("--filename") + 1] == "x.safetensors"


def test_download_model_rejects_embedded_nul_url(patched_run):
    """A NUL surfaces as ComfyCliError, not subprocess's bare ValueError."""
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        _download_model("https://hf.co/x\0.safetensors")

    assert calls == []


def test_download_model_rejects_embedded_nul_relative_path(patched_run):
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        _download_model("https://hf.co/x.safetensors", relative_path="models/\0")

    assert calls == []


def test_download_model_rejects_embedded_nul_filename(patched_run):
    calls = patched_run(envelope(data=_submit()))

    with pytest.raises(server.ComfyCliError, match="embedded NUL"):
        _download_model("https://hf.co/x.safetensors", filename="x\0.safetensors")

    assert calls == []


def test_download_model_omits_empty_string_optionals(patched_run):
    """Explicit empty-string optionals are treated as unset, not forwarded as ``""``."""
    calls = patched_run(envelope(data=_submit()))

    _download_model(
        "https://hf.co/x.safetensors", relative_path="", filename="", wait=False
    )

    cmd = calls[0]["cmd"]
    assert "--relative-path" not in cmd
    assert "--filename" not in cmd


# --- the background download family: submit, poll, cancel ------------------
#
# `download_model` used to be one blocking `comfy model download` that held the
# MCP request open for the whole multi-GB transfer, so the client's deadline
# fired while the download quietly succeeded — a false failure with no handle to
# verify it by. It now submits to comfy-cli's background worker and polls the
# resulting `download_id`, the shape `run_workflow(wait=False)` + `wait_for_job`
# already use for long generations.


def _submit(download_id: str = "a1b2c3d4e5f6", **extra) -> dict:
    """A `model download --background` submit payload."""
    return {
        "download_id": download_id,
        "dest": "/models/x.safetensors",
        "total_bytes": 4_000_000_000,
        "status": "starting",
        **extra,
    }


def _status(status: str, **extra) -> dict:
    """A `model download-status` payload."""
    return {
        "id": "a1b2c3d4e5f6",
        "status": status,
        "completed_bytes": 1_000_000,
        "total_bytes": 4_000_000_000,
        "dest": "/models/x.safetensors",
        "error": None,
        **extra,
    }


class _FakeClock:
    """A ``time``-shaped stand-in that only ``server`` sees.

    The download family is a deadline machine — every branch below turns on how
    much of `timeout_seconds` is left — so these tests have to drive the clock.
    Patching ``server.time.monotonic`` would do it by mutating the STDLIB module
    (``server.time`` *is* ``time``), freezing it process-wide including for the
    asyncio loop these tests run under, whose ``BaseEventLoop.time()`` is
    ``time.monotonic()``. Replacing ``server``'s own ``time`` BINDING keeps the
    fake where it belongs, and lets the tests stop racing wall time.

    ``sleep`` is a no-op rather than a clock advance: only an explicit
    `advance()` moves time, so a poll loop's own pacing never eats the budget
    the test is measuring.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start
        self._per_read = 0.0

    def monotonic(self) -> float:
        now = self._now
        self._now += self._per_read
        return now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def reset(self, start: float) -> None:
        self._now = start
        self._per_read = 0.0

    def tick_per_read(self, step: float) -> None:
        """Advance `step` on every READ of the clock.

        The bounded poll loop consults the clock rather than sleeping through
        the budget, so a per-read tick is how a test drains a deadline without
        waiting for one.
        """
        self._per_read = step

    def sleep(self, _seconds: float) -> None:
        pass

    def __getattr__(self, name):  # anything else still comes from the real module
        return getattr(time, name)


def _install_clock(monkeypatch, start: float = 1_000.0) -> _FakeClock:
    """Give `server` a controllable clock for the duration of one test.

    Idempotent: `_sequenced` installs one for every test that uses it, so a
    second call RESETS that clock rather than orphaning it behind a new one —
    otherwise a test could end up advancing a clock `server` no longer reads.
    """
    clock = getattr(server, "time", None)
    if isinstance(clock, _FakeClock):
        clock.reset(start)
        return clock
    clock = _FakeClock(start)
    monkeypatch.setattr(server, "time", clock)
    return clock


def _sequenced(monkeypatch, replies: list, *, seconds_per_call: float = 0.0) -> list:
    """Answer successive `_run_comfy` calls from `replies`; record their argv.

    The conftest fakes hand back ONE canned reply per fixture, and a submit
    followed by N polls needs a different answer each time — the multi-call case
    AGENTS.md leaves to a local stub. An exhausted iterator fails loudly rather
    than looping forever.

    Installs the fake clock too: the poll loop between replies is `time.sleep`,
    which must not really wait. `seconds_per_call` charges each spawn that much
    simulated time, for the tests about a deadline draining as the loop runs.
    """
    calls: list[tuple] = []
    pending = iter(replies)
    clock = _install_clock(monkeypatch)

    def fake_run(*args, **kwargs):
        calls.append(args)
        clock.advance(seconds_per_call)
        try:
            return next(pending)
        except StopIteration:
            raise AssertionError(f"unexpected extra comfy-cli call: {args}") from None

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    return calls


def test_download_model_submits_then_polls_to_completion(monkeypatch):
    """wait=True submits in the background and polls until the transfer lands."""
    calls = _sequenced(
        monkeypatch,
        [_submit(), _status("downloading"), _status("completed")],
    )

    result = _download_model("https://hf.co/x.safetensors")

    assert result == _status("completed")
    assert calls[0][:4] == ("model", "download", "--url", "https://hf.co/x.safetensors")
    assert calls[0][-1] == "--background"
    # Polls the id the submit handed back, and keeps polling past `downloading`.
    assert calls[1] == ("model", "download-status", "a1b2c3d4e5f6")
    assert len(calls) == 3


def test_download_model_returns_a_timed_out_envelope_rather_than_raising(monkeypatch):
    """A transfer still running at the bound is PROGRESS — the id comes back.

    This inversion is the entire bug: the old blocking call let the client's
    deadline fire on a download that was succeeding, leaving no handle to verify
    it with. A bound that expires must never look like a failure.
    """
    # Each spawn burns a simulated second, so the 1e-9 bound is long gone by the
    # time the first poll returns — deterministically, not by racing the wall.
    calls = _sequenced(
        monkeypatch, [_submit(), _status("downloading")], seconds_per_call=1.0
    )

    result = _download_model("https://hf.co/x.safetensors", timeout_seconds=1e-9)

    assert result == {
        "timed_out": True,
        "download_id": "a1b2c3d4e5f6",
        "status": _status("downloading"),
    }
    # A bound that expires before the first poll still reports a real status.
    assert len(calls) == 2


def test_download_model_tiny_bound_fails_without_starting_a_transfer(monkeypatch):
    """A bound too small to resolve the download must not leave one running.

    The docstring promises this outright, and the promise has real teeth: the
    submit is capped to the caller's bound, so a tiny bound kills it — and that
    timeout must NOT be mistaken for the missing-`--background` degrade, which
    would retry the whole thing as an 1800s blocking transfer nobody asked for.
    The existing end-to-end-budget test pins the cap; this pins what happens
    when the cap actually bites.
    """
    calls: list[float] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise server.ComfyCliError(
            "comfy-cli timed out after 0.001s", timed_out=True, returncode=None
        )

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="timed out"):
        _download_model("https://hf.co/x.safetensors", timeout_seconds=0.001)

    # The submit took the caller's bound, not its own fixed budget...
    assert calls == [0.001]
    # ...and exactly one spawn happened: no legacy fallback, so nothing detached
    # and no second copy of a multi-GB file was fetched.


def test_download_model_raises_on_a_failed_download(monkeypatch):
    """A terminal `failed` carries comfy-cli's own error text into the raise."""
    _sequenced(
        monkeypatch,
        [_submit(), _status("failed", error="HTTP 404: model not found")],
    )

    with pytest.raises(server.ComfyCliError, match="HTTP 404: model not found"):
        _download_model("https://hf.co/missing.safetensors")


def test_download_model_raises_on_a_cancelled_download(monkeypatch):
    """A download cancelled out from under the wait is a failure, not a result."""
    _sequenced(monkeypatch, [_submit(), _status("cancelled")])

    with pytest.raises(server.ComfyCliError, match="cancelled"):
        _download_model("https://hf.co/x.safetensors")


def test_download_model_wait_false_returns_the_submit_payload(monkeypatch):
    """wait=False hands back the submit envelope and polls nothing."""
    calls = _sequenced(monkeypatch, [_submit()])

    assert _download_model("https://hf.co/x.safetensors", wait=False) == _submit()
    assert len(calls) == 1  # no `download-status` call at all


def test_download_model_rejects_a_submit_envelope_with_no_download_id(monkeypatch):
    """No id means nothing to poll — that is a broken contract, not a result.

    Returning the payload anyway would hand back a `status: starting` blob that
    reads like a finished download and leaves the caller no handle at all.
    """
    _sequenced(monkeypatch, [{"dest": "/models/x.safetensors", "status": "starting"}])

    with pytest.raises(server.ComfyCliError, match="no usable `download_id`"):
        _download_model("https://hf.co/x.safetensors")


def test_download_model_wait_false_also_rejects_an_envelope_with_no_id(monkeypatch):
    """The id is validated on the wait=False path too, not just the waiting one.

    Handing back a malformed envelope unchecked would leave a transfer running
    detached behind a payload carrying nothing to poll or cancel it by — the
    same broken contract, just without a poll loop to notice it.
    """
    _sequenced(monkeypatch, [{"dest": "/models/x.safetensors", "status": "starting"}])

    with pytest.raises(server.ComfyCliError, match="no usable `download_id`"):
        _download_model("https://hf.co/x.safetensors", wait=False)


def test_download_model_spends_one_end_to_end_budget(monkeypatch):
    """`timeout_seconds` covers submit AND poll, not each of them separately.

    Two independent budgets add up: a submit that used its full
    `_DOWNLOAD_SUBMIT_TIMEOUT` plus a full-length poll ran ~230s, while the 110s
    default exists precisely to come in under a typical client's ~120s request
    budget. Overshooting it means the client aborts and never receives the
    `download_id` the submit already obtained — the exact failure this tool's
    async shape was built to prevent.
    """
    seen: dict[str, float] = {}
    elapsed = 4.0
    clock = _install_clock(monkeypatch)

    def fake_run(*args, **kwargs):
        seen["submit_timeout"] = kwargs["timeout"]
        # Simulate a submit that spent `elapsed` seconds resolving metadata.
        clock.advance(elapsed)
        return _submit()

    def fake_poll(download_id, timeout_seconds):
        seen["poll_bound"] = timeout_seconds
        return _status("completed")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server, "_poll_download", fake_poll)

    _download_model("https://hf.co/x.safetensors", timeout_seconds=30.0)

    # The submit is capped to the caller's bound as well as its own...
    assert seen["submit_timeout"] == 30.0
    # ...and the poll gets only what the submit left of it. The clock is a fake,
    # so this is exact: a tolerance wide enough to swallow `elapsed` would pass
    # just as happily on a poll handed the caller's WHOLE budget.
    assert seen["poll_bound"] == pytest.approx(30.0 - elapsed)


def test_download_model_wait_false_keeps_the_full_submit_budget(monkeypatch):
    """wait=False is exempt from the end-to-end cap — it never waits.

    A submit cut short may leave no transfer running at all, so the fire-and-
    return path keeps the fixed submit budget however impatient the caller is.
    """
    seen: dict[str, float] = {}

    def fake_run(*args, **kwargs):
        seen["submit_timeout"] = kwargs["timeout"]
        return _submit()

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    _download_model("https://hf.co/x.safetensors", wait=False, timeout_seconds=1.0)

    assert seen["submit_timeout"] == server._DOWNLOAD_SUBMIT_TIMEOUT


def test_download_model_attaches_the_id_when_the_poll_raises(monkeypatch):
    """A poll that errors must not orphan the transfer it was polling.

    The id was minted INSIDE this call, so letting the exception through
    untouched leaves a multi-GB download running detached with no handle to
    enumerate or cancel it. `wait_for_download` needs no such wrapping — its
    caller passed the id in and still holds it.
    """

    def fake_poll(download_id, timeout_seconds):
        raise server.ComfyCliError(
            "comfy-cli timed out after 1.0s", timed_out=True, returncode=None
        )

    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: _submit())
    monkeypatch.setattr(server, "_poll_download", fake_poll)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model("https://hf.co/x.safetensors")

    message = str(excinfo.value)
    assert "a1b2c3d4e5f6" in message  # the handle survives the failure
    assert "cancel_download" in message
    assert "comfy-cli timed out after 1.0s" in message  # original diagnosis kept
    # Structured attributes survive too, so a caller branching on them still can.
    assert excinfo.value.timed_out is True
    assert isinstance(excinfo.value.__cause__, server.ComfyCliError)


def test_download_model_falls_back_to_the_foreground_call_on_an_old_cli(
    patched_async_run, legacy_comfy_cli
):
    """A comfy-cli without `--background` still downloads, the old blocking way."""
    procs = patched_async_run(
        stderr="Done in 55.8s. Saved to /models/x.safetensors",
    )

    result = _download_model("https://hf.co/x.safetensors")

    assert result["ok"] is True
    assert "Done in 55.8s" in result["message"]
    # Exactly one spawn reached comfy-cli — the rejected `--background` submit
    # never ran anything — and the retry drops the flag it choked on.
    assert len(procs) == 1
    assert "--background" not in procs[0].cmd
    # Spawned through the ASYNC runner, with the same guardrails as every other
    # async spawn: own process group (so a kill reaps the tree) and no inherited
    # stdin (that fd is this server's JSON-RPC channel).
    assert procs[0].start_new_session is True
    assert procs[0].stdin_arg == asyncio.subprocess.DEVNULL
    # It ran on the legacy path, and now says so even though `wait=True` was
    # honored as well as it can be here — there is still no `download_id`.
    assert result["background_unsupported"] is True


def test_download_model_fallback_flags_a_wait_false_it_could_not_honor(
    patched_async_run, legacy_comfy_cli
):
    """The legacy path blocks even on wait=False, and says so in the payload.

    There is no `--background` to detach and no `download_id` to hand back, so
    the whole transfer runs inside the call. Marking that lets a caller who
    asked NOT to block see that it blocked — and that the download family has no
    id to poll — instead of inferring both from a missing key.

    Deliberately still a success: refusing would remove the only way to download
    a model on a comfy-cli that predates `--background`, and the file did land.
    """
    patched_async_run(stderr="Done in 55.8s. Saved to /models/x.safetensors")

    result = _download_model("https://hf.co/x.safetensors", wait=False)

    assert result["ok"] is True
    assert result["background_unsupported"] is True
    assert "download_id" not in result


def test_download_model_legacy_wait_true_spends_what_the_submit_left(monkeypatch):
    """The waited fallback is bounded by `timeout_seconds`, not a silent 30 min.

    The rejected `--background` submit already spent part of the caller's
    deadline, so the foreground transfer gets the remainder — the same two-phase
    spend the modern path does with its poll — and `_DOWNLOAD_SYNC_TIMEOUT` is
    only the cap on it.
    """
    seen: dict[str, float] = {}
    elapsed = 4.0
    clock = _install_clock(monkeypatch)

    def fake_run(*args, **kwargs):
        # The submit burns part of the deadline before failing. Charged to the
        # fake clock, not slept on the wall: a real sleep both slows the suite
        # and makes the assertion a race with the runner's scheduling.
        clock.advance(elapsed)
        raise _missing_background_error()

    async def fake_async(*args, timeout=None, **kwargs):
        seen["bound"] = timeout
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    _download_model("https://hf.co/x.safetensors", timeout_seconds=30.0)

    assert seen["bound"] == pytest.approx(30.0 - elapsed)


def test_download_model_legacy_wait_true_is_capped_by_the_sync_ceiling(monkeypatch):
    """A caller who budgeted more than the cap is still held only to the cap."""
    seen: dict[str, float] = {}

    async def fake_async(*args, timeout=None, **kwargs):
        seen["bound"] = timeout
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", _reject_background)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    _download_model(
        "https://hf.co/x.safetensors", timeout_seconds=server._MAX_DOWNLOAD_WAIT_TIMEOUT
    )

    assert seen["bound"] == server._DOWNLOAD_SYNC_TIMEOUT


def test_download_model_legacy_refuses_an_exhausted_remainder(monkeypatch):
    """An all-but-spent deadline REFUSES the transfer instead of starting one.

    Starting one anyway would be worse than useless: `comfy model download` writes
    straight to the FINAL path, so a child killed moments after it opened the
    destination truncates whatever complete model was already there — bought with a
    bound that could never have finished the transfer. The `--background` path
    takes the same position on a bound too small to resolve the download at all.
    """
    spawned: list[float | None] = []
    clock = _install_clock(monkeypatch)

    def fake_run(*args, **kwargs):
        clock.advance(1.2)  # overspends the caller's whole 1s budget
        raise _missing_background_error()

    async def fake_async(*args, timeout=None, **kwargs):
        spawned.append(timeout)
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model("https://hf.co/x.safetensors", timeout_seconds=1.0)

    assert spawned == []  # nothing spawned, so nothing on disk was touched
    message = str(excinfo.value)
    # The caller's deadline expiring, just detected before the spend instead of
    # after it — so it reads as a timeout to anyone keying on that.
    assert excinfo.value.timed_out is True
    assert "NOTHING was downloaded" in message
    assert "predates `model download --background`" in message
    # And the same way out the timeout path offers, so the refusal is actionable.
    assert "timeout_seconds" in message and "1800" in message
    assert "upgrade comfy-cli" in message


def test_download_model_legacy_spends_a_remainder_above_the_minimum(monkeypatch):
    """A remainder that COULD carry a transfer is spent, not refused.

    The refusal above is scoped to a budget too small to start under; anything
    above `_MIN_LEGACY_DOWNLOAD_TIMEOUT` still gets its foreground transfer, which
    is the only way to download a model on a CLI this old.
    """
    seen: dict[str, float] = {}

    async def fake_async(*args, timeout=None, **kwargs):
        seen["bound"] = timeout
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", _reject_background)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    _download_model(
        "https://hf.co/x.safetensors",
        timeout_seconds=server._MIN_LEGACY_DOWNLOAD_TIMEOUT + 5.0,
    )

    assert seen["bound"] >= server._MIN_LEGACY_DOWNLOAD_TIMEOUT


def test_download_model_legacy_wait_false_keeps_the_full_cap(monkeypatch):
    """wait=False stays on the cap: it explicitly decoupled from waiting.

    It never reads `timeout_seconds` (the docstring says so), so tightening its
    bound would only truncate a transfer nobody asked to be quick. What it gains
    from the async runner is the reaping on cancellation, not a shorter deadline.
    """
    seen: dict[str, float] = {}

    async def fake_async(*args, timeout=None, **kwargs):
        seen["bound"] = timeout
        return {"ok": True}

    monkeypatch.setattr(server, "_run_comfy", _reject_background)
    monkeypatch.setattr(server, "_run_comfy_async", fake_async)

    _download_model("https://hf.co/x.safetensors", wait=False, timeout_seconds=5.0)

    assert seen["bound"] == server._DOWNLOAD_SYNC_TIMEOUT


def test_download_model_does_not_fall_back_on_a_real_submit_failure(monkeypatch):
    """Only an unknown `--background` degrades — anything else must propagate.

    A submit that failed for some other reason may already have started a
    transfer, and re-running it here would fetch the same multi-GB file twice.
    """
    calls: list[tuple] = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        raise server.ComfyCliError(
            "comfy-cli returned no JSON (exit 1). stderr: connection reset",
            no_envelope=True,
            returncode=1,
        )

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="connection reset"):
        _download_model("https://hf.co/x.safetensors")

    assert len(calls) == 1  # never retried


def test_download_model_clamps_an_oversized_timeout(monkeypatch):
    """An `inf` bound is clamped before the poll loop can run on it forever."""
    seen: dict[str, float] = {}

    def fake_poll(download_id, timeout_seconds):
        seen["bound"] = timeout_seconds
        return _status("completed")

    monkeypatch.setattr(server, "_run_comfy", lambda *a, **k: _submit())
    monkeypatch.setattr(server, "_poll_download", fake_poll)

    _download_model("https://hf.co/x.safetensors", timeout_seconds=float("inf"))

    # The poll gets the ceiling MINUS whatever the submit just spent, so this is
    # bounded-just-under rather than equal — the deduction is the point (see
    # `test_download_model_spends_one_end_to_end_budget`), and the submit here is
    # a stubbed call that returns in microseconds.
    assert 0 < seen["bound"] <= server._MAX_DOWNLOAD_WAIT_TIMEOUT
    assert seen["bound"] == pytest.approx(server._MAX_DOWNLOAD_WAIT_TIMEOUT, abs=1.0)


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_download_model_rejects_a_bad_timeout_before_submitting(monkeypatch, bad):
    """NaN/0/negative are refused BEFORE anything detaches.

    Validating after the submit would leave a background worker running that
    nobody is waiting on — and with NaN every comparison is False, so the poll
    loop would never exit on its own.
    """
    calls = _sequenced(monkeypatch, [_submit()])

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        _download_model("https://hf.co/x.safetensors", timeout_seconds=bad)

    assert calls == []


def test_download_model_wait_false_ignores_the_timeout_argument(monkeypatch):
    """wait=False never reads `timeout_seconds`, so it must not validate it.

    The submit runs on its own fixed budget; rejecting a value it ignores would
    newly refuse a call that works fine today (same rule as `run_workflow`).
    """
    _sequenced(monkeypatch, [_submit()])

    assert (
        _download_model(
            "https://hf.co/x.safetensors", wait=False, timeout_seconds=float("nan")
        )
        == _submit()
    )


def test_download_status_maps_command_and_returns_data(patched_run):
    """download_status wraps `comfy model download-status <id>`."""
    payload = _status("downloading", percent=25.0)
    calls = patched_run(envelope(data=payload))

    assert server.download_status("a1b2c3d4e5f6") == payload
    assert calls[0]["cmd"][4:] == ["model", "download-status", "a1b2c3d4e5f6"]
    assert calls[0]["timeout"] == 60.0


def test_cancel_download_maps_command_and_returns_data(patched_run):
    """cancel_download wraps `comfy model download-cancel <id>`."""
    calls = patched_run(envelope(data=_status("cancelled")))

    assert server.cancel_download("a1b2c3d4e5f6")["status"] == "cancelled"
    assert calls[0]["cmd"][4:] == ["model", "download-cancel", "a1b2c3d4e5f6"]
    assert calls[0]["timeout"] == 60.0


def test_cancel_download_unknown_id_raises_error_envelope(patched_run):
    """Cancelling an unknown download surfaces comfy-cli's error envelope."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "download_not_found", "message": "no such download"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="download_not_found"):
        server.cancel_download("nope")


# Click's usage error for a verb the installed comfy-cli does not have: exit 2,
# no envelope. The background-download verb group ships only in releases after
# 1.13.0, while this repo's floor is 1.12.0, so this is the COMMON path today.
_NO_DOWNLOAD_STATUS = (
    2,
    "",
    "Usage: comfy [OPTIONS] COMMAND\nNo such command 'download-status'.",
)
_NO_DOWNLOAD_CANCEL = (
    2,
    "",
    "Usage: comfy [OPTIONS] COMMAND\nNo such command 'download-cancel'.",
)


@pytest.mark.parametrize(
    "tool, reply, verb",
    [
        ("download_status", _NO_DOWNLOAD_STATUS, "download-status"),
        ("wait_for_download", _NO_DOWNLOAD_STATUS, "download-status"),
        ("cancel_download", _NO_DOWNLOAD_CANCEL, "download-cancel"),
    ],
)
def test_download_companions_degrade_on_a_comfy_cli_without_the_verb(
    patched_run, tool, reply, verb
):
    """A missing verb reads as a capability gap, not as a broken MCP.

    `download_model` already degrades for the option-shaped half of this same
    version gap (`--background`); these three are the verb-shaped half. The
    group is all-or-nothing, so a CLI missing them also rejects `--background`
    and can never have minted an id — nothing is actually lost, and the message
    points at the inline `download_model` that DOES work there.
    """
    returncode, stdout, stderr = reply
    patched_run(stdout, returncode=returncode, stderr=stderr)

    result = getattr(server, tool)("a1b2c3d4e5f6")

    assert result["unsupported"] is True
    assert f"model {verb} unavailable" in result["error"]
    # Points at the path that still works rather than dead-ending.
    assert "download_model" in result["error"]
    # None of the raw wrapper/CLI text leaks through.
    assert "No such command" not in result["error"]
    assert "Usage: comfy" not in result["error"]
    assert "returned no JSON" not in result["error"]


@pytest.mark.parametrize("tool", ["download_status", "wait_for_download"])
def test_download_companions_keep_a_real_error_raw(patched_run, tool):
    """A verb comfy-cli DID dispatch must never be waved through as a gap.

    An unknown id is a real answer from a working CLI. Degrading it would tell
    the agent nothing is broken while its download silently isn't there.
    """
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "download_not_found", "message": "no such download"},
        }
    )

    with pytest.raises(server.ComfyCliError, match="download_not_found"):
        getattr(server, tool)("a1b2c3d4e5f6")


@pytest.mark.parametrize(
    "tool, verb",
    [
        ("download_status", "download-status"),
        ("wait_for_download", "download-status"),
        ("cancel_download", "download-cancel"),
    ],
)
def test_download_companions_echoed_phrase_is_not_unsupported(patched_run, tool, verb):
    """A caller cannot forge the version gap through its own `download_id`.

    The id is a bare positional and `_guard_download_id` deliberately permits
    any characters, so Click echoing an offending value verbatim lands on the
    same exit 2 with no envelope `_is_missing_verb_error` reads — the one route
    to a false `unsupported` its two conditions cannot close. Subtracting the
    caller's own text keeps a real failure a real failure.
    """
    download_id = f"no such command {verb!r}"
    patched_run(
        "",
        returncode=2,
        stderr=(
            f"Usage: comfy model {verb} [OPTIONS] DOWNLOAD_ID\n"
            f"Error: Invalid value for 'DOWNLOAD_ID': {download_id!r} does not exist."
        ),
    )

    with pytest.raises(server.ComfyCliError):
        getattr(server, tool)(download_id)


@pytest.mark.parametrize(
    "tool, verb",
    [
        ("download_status", "download-status"),
        ("wait_for_download", "download-status"),
        ("cancel_download", "download-cancel"),
    ],
)
def test_download_companions_degrade_with_a_colliding_download_id(
    patched_run, tool, verb
):
    """A short id that COLLIDES with Click's phrase must not cost the degrade.

    The subtraction is a global `str.replace`, so discounting a value that is
    merely a substring of comfy-cli's own message would shred the parser's
    phrase and suppress the genuine version gap. `"a"` is an id
    `_guard_download_id` accepts and comfy-cli's `[A-Za-z0-9_-]{1,64}` store
    resolves, and it appears in both `download-status` and `download-cancel`.
    `_phrase_is_only_the_caller_s` only subtracts a value that matches the
    pattern ITSELF — one that cannot carry the phrase cannot have forged it.
    """
    patched_run(
        "",
        returncode=2,
        stderr=f"Usage: comfy model [OPTIONS] COMMAND\nNo such command '{verb}'.",
    )

    assert getattr(server, tool)("a")["unsupported"] is True


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_download_companions_reject_a_blank_download_id(blank):
    """A whitespace-only id is the same caller mistake as an empty one.

    `_guard_download_id`'s docstring says an empty id can only be a caller
    mistake, but `" "` is truthy — so it took a `strip()` to refuse it rather
    than forward `comfy model download-status " "`.
    """
    with pytest.raises(server.ComfyCliError, match="invalid download_id"):
        server.download_status(blank)


@pytest.mark.parametrize("dash_led", [" -x", "\t--help"])
def test_download_companions_reject_a_padded_dash_led_id(dash_led):
    """Both shape tests read the STRIPPED value, so they cannot disagree.

    The guard's docstring says a dash-led id stays out of argv. Testing that on
    the raw string while testing emptiness on the stripped one let `" -x"`
    through — defused today only by accident, because Click keys option
    detection on the first character.
    """
    with pytest.raises(server.ComfyCliError, match="invalid download_id"):
        server.download_status(dash_led)


def test_guard_download_id_reports_a_size_not_an_oversized_value():
    """The length check runs BEFORE the shape checks, which echo the value.

    The shape branch interpolates `{download_id!r}` in full, so an oversized
    blank would otherwise be mirrored back verbatim through the tool response
    and the failure log — defeating the "report the length, not the value" rule
    the cap exists to enforce.
    """
    oversized_blank = " " * (server._MAX_DOWNLOAD_ID_LEN + 50)

    with pytest.raises(server.ComfyCliError) as excinfo:
        server._guard_download_id(oversized_blank)

    assert "exceeds the" in str(excinfo.value)
    assert oversized_blank not in str(excinfo.value)


@pytest.mark.parametrize("bad", [None, 12, ["a1b2c3"]])
def test_guard_download_id_rejects_a_non_string(bad):
    """The guard's contract is total: a non-string raises ComfyCliError.

    Every check in the guard is a `str` method, so testing shape before type
    would leave a bare `AttributeError` for an in-process caller instead of the
    error every other bad input produces.
    """
    with pytest.raises(server.ComfyCliError, match="expected a string"):
        server._guard_download_id(bad)


def test_wait_for_download_returns_the_terminal_payload(monkeypatch):
    """wait_for_download polls until terminal and returns that final payload."""
    calls = _sequenced(monkeypatch, [_status("downloading"), _status("completed")])

    assert server.wait_for_download("a1b2c3d4e5f6") == _status("completed")
    assert calls[0] == ("model", "download-status", "a1b2c3d4e5f6")
    assert len(calls) == 2


def test_wait_for_download_returns_a_failure_rather_than_raising(monkeypatch):
    """Like `wait_for_job`, a terminal failure comes back as a payload to read.

    Only `download_model` turns that into a raise — this tool is the polling
    primitive, and an agent chaining it wants the `status` / `error` fields.
    """
    _sequenced(monkeypatch, [_status("failed", error="checksum mismatch")])

    result = server.wait_for_download("a1b2c3d4e5f6")

    assert result["status"] == "failed"
    assert result["error"] == "checksum mismatch"


def test_wait_for_download_times_out_cleanly(monkeypatch):
    """An unfinished transfer returns the timed-out envelope, carrying the id."""
    _sequenced(monkeypatch, [_status("downloading")] * 10)

    # A monotonic clock that jumps 10s per read, so the 25s bound expires.
    _install_clock(monkeypatch, start=0.0).tick_per_read(10.0)

    assert server.wait_for_download("a1b2c3d4e5f6") == {
        "timed_out": True,
        "download_id": "a1b2c3d4e5f6",
        "status": _status("downloading"),
    }


def test_wait_for_download_caps_each_poll_to_the_remaining_bound(monkeypatch):
    """A single poll never gets a longer subprocess budget than the wait itself."""
    seen: list[float] = []

    def fake_run(*args, timeout=None, **kwargs):
        seen.append(timeout)
        return _status("downloading")

    monkeypatch.setattr(server, "_run_comfy", fake_run)
    clock = _install_clock(monkeypatch, start=0.0)
    clock.tick_per_read(1.0)  # the bound drains as the loop consults the clock

    result = server.wait_for_download("a1b2c3d4e5f6", timeout_seconds=5.0)

    assert result["timed_out"] is True
    assert seen == [4.0, 2.0]  # each poll gets only what is left of the 5s bound


@pytest.mark.parametrize("oversized", [float("inf"), 86_400.0])
def test_wait_for_download_clamps_an_oversized_timeout(monkeypatch, oversized):
    """An oversized bound is clamped to the ceiling, so the poll loop terminates."""
    _sequenced(monkeypatch, [_status("downloading")] * 4)

    reads = iter([0.0, 1.0, server._MAX_DOWNLOAD_WAIT_TIMEOUT + 1.0])

    def fake_monotonic():
        try:
            return next(reads)
        except StopIteration:
            pytest.fail("wait_for_download kept polling past the clamped ceiling")

    # On the fake clock object, not on the stdlib module `server` imported.
    monkeypatch.setattr(_install_clock(monkeypatch), "monotonic", fake_monotonic)

    assert server.wait_for_download("a1b2c3d4e5f6", timeout_seconds=oversized)[
        "timed_out"
    ]


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_wait_for_download_rejects_a_non_positive_or_nan_timeout(monkeypatch, bad):
    """NaN/0/negative are refused before the first poll."""
    calls = _sequenced(monkeypatch, [])

    with pytest.raises(server.ComfyCliError, match="timeout_seconds"):
        server.wait_for_download("a1b2c3d4e5f6", timeout_seconds=bad)

    assert calls == []


# The download family takes its id as a bare positional exactly as the jobs
# family takes `prompt_id`, so it carries the same guard: a leading dash would
# reach comfy-cli as an option, a NUL makes `subprocess.Popen` raise a bare
# ValueError, an empty id can only be a caller mistake, and an oversized one
# fails in the exec as an OSError nobody converts.
@pytest.mark.parametrize(
    "tool",
    ["download_status", "cancel_download", "wait_for_download"],
)
@pytest.mark.parametrize("bad_id", ["--help", "-o", "", "d\x001", "d" * 129])
def test_download_tools_reject_an_unusable_download_id(monkeypatch, tool, bad_id):
    """A dash-led / empty / NUL-bearing / oversized id is refused before any spawn."""

    def fake_run(*args, **kwargs):
        raise AssertionError(f"{tool} spawned comfy-cli with {bad_id!r}")

    monkeypatch.setattr(server, "_run_comfy", fake_run)

    with pytest.raises(server.ComfyCliError, match="download_id"):
        getattr(server, tool)(bad_id)


def test_download_model_guards_the_id_comfy_cli_hands_back(monkeypatch):
    """The polled id is guarded too — it becomes argv on every poll."""
    _sequenced(monkeypatch, [_submit(download_id="--evil")])

    with pytest.raises(server.ComfyCliError, match="download_id"):
        _download_model("https://hf.co/x.safetensors")


# --- download_model: the LEGACY synchronous fallback ------------------------
#
# A comfy-cli that predates `model download --background` rejects the flag at
# parse time, and `download_model` falls back to the old one-shot synchronous
# call. That path keeps the old contract in full, including the BE-3345
# no-envelope handling below — which is now scoped to it alone, since a
# `--background` submit returns a real envelope. `legacy_comfy_cli` is what
# makes the installed CLI look that old.


def test_download_model_echoed_phrase_does_not_reach_the_fallback(
    patched_run, monkeypatch
):
    """A caller cannot forge the version gap through its own `url`.

    The stake here is the highest of the echoed-argv group: the degrade is not a
    label but a re-run, so a false `--background` gap would silently repeat a
    multi-GB transfer the failed submit may already have started. `url` /
    `relative_path` / `filename` are the caller's, and Click echoing one back in
    a usage error lands on the same exit 2 with no envelope
    `_is_missing_option_error` reads — so the subtraction has to run before the
    fallback does.
    """

    async def _never(*args, **kwargs):
        raise AssertionError("the legacy foreground fallback must not run")

    monkeypatch.setattr(server, "_run_comfy_async", _never)
    url = "https://hf.co/x.safetensors?q=No such option: --background"
    patched_run(
        "",
        returncode=2,
        stderr=(
            "Usage: comfy model download [OPTIONS]\n"
            f"Error: Invalid value for '--url': {url!r} is not reachable."
        ),
    )

    with pytest.raises(server.ComfyCliError):
        _download_model(url)


def test_download_model_falls_back_with_a_colliding_filename(
    patched_async_run, legacy_comfy_cli
):
    """A benign `filename` that COLLIDES with Click's phrase keeps the fallback.

    The regression this pins is the sharpest consequence of an unbounded
    subtraction: `filename="background"` is an ordinary bare name, but it is
    also a substring of `No such option: --background`, so discounting it
    unconditionally would erase the flag out of comfy-cli's OWN message, make
    `_phrase_is_only_the_caller_s` report a forgery, and `raise` past
    `_legacy_foreground_download` — turning a download that works on a
    pre-`--background` comfy-cli into an outright failure. Unlike the verb
    sites, where an over-subtraction costs only the friendly message, here the
    suppressed degrade is the only path that performs the transfer.
    """
    patched_async_run(stderr="Done in 9.1s. Saved to /models/background")

    result = _download_model("https://hf.co/background", filename="background")

    assert result["ok"] is True
    assert result["background_unsupported"] is True


def test_download_model_synthesizes_success_on_plain_exit(
    patched_async_run, legacy_comfy_cli
):
    """`comfy model download` streams text + exits 0, no envelope -> success.

    The download landed on disk (exit 0), so instead of raising the
    "returned no JSON" false negative — which would invite a bandwidth-expensive
    retry of a multi-GB fetch — a success payload is synthesized carrying the
    CLI's printed tail (where "Done in …" and the saved path live).
    """
    patched_async_run(
        stderr="Downloading x.safetensors...\nDone in 55.8s. Saved to /models/x.safetensors",
    )

    result = _download_model("https://hf.co/x.safetensors")

    assert result["ok"] is True
    assert result["action"] == "model download"
    assert "Done in 55.8s" in result["message"]


def test_download_model_nonzero_exit_still_raises(patched_async_run, legacy_comfy_cli):
    """A real download failure (non-zero exit, no envelope) must still raise."""
    patched_async_run(returncode=1, stderr="HTTP 404: model not found")

    with pytest.raises(server.ComfyCliError, match="returned no JSON"):
        _download_model("https://hf.co/missing.safetensors")


def test_download_model_synthesizes_despite_stray_non_envelope_json(
    patched_async_run, legacy_comfy_cli
):
    """A stray non-envelope JSON line on a clean download exit is still success.

    `_last_json_object` returns any JSON object (not just `type==envelope`), so a
    diagnostic line that happens to parse must NOT be mistaken for a result
    envelope and unwrapped into a spurious failure (BE-3345 edge case).
    """
    patched_async_run(
        '{"level": "info", "msg": "connection reused"}\n',
        stderr="Done in 12.3s. Saved to /models/x.safetensors",
    )

    result = _download_model("https://hf.co/x.safetensors")

    assert result["ok"] is True
    assert result["action"] == "model download"
    assert "Done in 12.3s" in result["message"]


def test_download_model_envelope_then_diagnostic_keeps_envelope_data(
    patched_async_run, legacy_comfy_cli
):
    """A real envelope FOLLOWED BY a diagnostic JSON line still wins (BE-3345).

    `_last_json_object` prefers a `type==envelope` object over a later plain JSON
    line, so a trailing diagnostic must NOT null out `real_envelope` and demote a
    genuine success into the synthesized fast-path — the envelope's `data` (the
    real saved-path metadata) must be returned, not the printed-text stopgap.
    """
    patched_async_run(
        '{"type": "envelope", "ok": true, "data": {"saved": "/models/x"}}\n'
        '{"level": "info", "msg": "cleanup done"}\n'
    )

    result = _download_model("https://hf.co/x.safetensors")

    # The legacy marker rides along on the envelope's own data, which is
    # otherwise returned verbatim.
    assert result == {"saved": "/models/x", "background_unsupported": True}


def test_download_model_error_envelope_then_diagnostic_still_raises(
    patched_async_run, legacy_comfy_cli
):
    """An error envelope followed by a diagnostic line still raises, not synthesized.

    Even on exit 0, a trailing diagnostic JSON line must not mask an earlier
    error envelope as a synthesized success — the error envelope is preferred and
    propagates its code (BE-3345).
    """
    patched_async_run(
        '{"type": "envelope", "ok": false, '
        '"error": {"code": "download_failed", "message": "checksum mismatch"}}\n'
        '{"level": "info", "msg": "cleanup done"}\n'
    )

    with pytest.raises(server.ComfyCliError, match=r"download_failed"):
        _download_model("https://hf.co/x.safetensors")


def test_download_model_still_honors_a_real_error_envelope(patched_run):
    """A real error envelope on download still raises with its code (not synthesized)."""
    patched_run(
        {
            "type": "envelope",
            "ok": False,
            "error": {"code": "download_failed", "message": "checksum mismatch"},
        }
    )

    with pytest.raises(server.ComfyCliError, match=r"download_failed"):
        _download_model("https://hf.co/x.safetensors")


def test_download_model_keeps_saved_path_tail_on_verbose_output(
    patched_async_run, legacy_comfy_cli
):
    """A verbose multi-GB fetch caps to the TAIL so the saved-path survives.

    `comfy model download` streams progress noise ahead of the `Done in …` /
    saved-path tail that the synthesized payload exists to surface. A front-slice
    cap would drop that tail as noise, so the message must keep the last chars.
    """
    tail = "Done in 903.4s. Saved to /models/checkpoints/big.safetensors"
    noise = "\n".join(f"Downloading... {i}% ({i * 40} MiB)" for i in range(100))
    patched_async_run(stderr=f"{noise}\n{tail}")

    result = _download_model("https://hf.co/big.safetensors")

    assert result["ok"] is True
    assert len(result["message"]) <= 1000
    assert "Saved to /models/checkpoints/big.safetensors" in result["message"]


def test_download_model_fallback_omits_url_when_no_output(
    patched_async_run, legacy_comfy_cli
):
    """The no-output fallback message must not echo the (possibly signed) URL.

    A `model download` URL can carry a token / userinfo in its query string; the
    synthesized fallback lands in the tool response and host logs, so it reports
    only the flag-free `model download` action, never the raw args.
    """
    patched_async_run()  # exit 0 with no stdout/stderr -> fallback message

    result = _download_model("https://hf.co/x.safetensors?sig=SECRETTOKEN")

    assert result["ok"] is True
    assert "SECRETTOKEN" not in result["message"]
    assert "hf.co" not in result["message"]
    assert "model download" in result["message"]


def test_download_model_legacy_timeout_kills_the_transfer_and_guides_the_caller(
    patched_async_run, legacy_comfy_cli, monkeypatch
):
    """A waited fallback that overruns its bound kills the transfer and says so.

    This is the BE-5167 shape: the bytes were moving inside THIS call, so unlike
    the `--background` path there is no detached worker left to poll — the
    transfer is dead and a partial file may be on disk with no `download_id` to
    check it with. The error has to carry all of that, plus the way out.
    """
    # Shrink the MINIMUM, not the cap, so a fraction-of-a-second budget is still
    # big enough to spawn under: `_DOWNLOAD_SYNC_TIMEOUT` has to stay real so the
    # retry guidance quotes the number a caller would actually pass.
    monkeypatch.setattr(server, "_MIN_LEGACY_DOWNLOAD_TIMEOUT", 0.05)
    # `server`'s deadline arithmetic runs on the fake clock, so the rejected
    # submit hands the transfer the WHOLE sub-second budget however long the
    # thread-pool hop really took — a slow runner can no longer divert this into
    # the exhausted-remainder refusal. The bound itself is still real wall time:
    # `_run_comfy_async` spends it in `asyncio.wait_for`, on the loop's own clock.
    _install_clock(monkeypatch)
    logged: list[dict] = []
    monkeypatch.setattr(
        failure_log,
        "_log_failure",
        lambda kind, args, **kwargs: logged.append({"kind": kind, **kwargs}),
    )
    # A child that outlives its deadline, having printed some progress first: the
    # runner's drain reads that into its sinks before the bound cancels it, which
    # is the whole reason those sinks are owned by the caller rather than by the
    # reader coroutine.
    procs = patched_async_run(stderr="Downloading big.safetensors... 12%", hang=True)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model(
            "https://hf.co/big.safetensors",
            relative_path="models/checkpoints",
            filename="big.safetensors",
            # Enough budget to start the transfer, nowhere near enough to finish.
            timeout_seconds=0.3,
        )

    message = str(excinfo.value)
    assert excinfo.value.timed_out is True
    # The original diagnosis is APPENDED to, not replaced — including the stderr
    # tail the CANCELLED reader had already collected.
    assert "comfy-cli timed out after" in message
    assert "Downloading big.safetensors... 12%" in message
    assert "predates `model download --background`" in message
    # Name the exact partial file: `filename` was supplied, so it is knowable.
    assert "models/checkpoints/big.safetensors" in message
    assert "timeout_seconds" in message and "1800" in message  # how to retry
    assert "upgrade comfy-cli" in message
    # The process TREE was killed rather than left to keep writing.
    assert procs[0].killed is True
    assert [record["kind"] for record in logged] == ["timeout"]


def test_download_model_legacy_timeout_names_the_folder_without_a_filename(
    patched_async_run, legacy_comfy_cli, monkeypatch
):
    """Without `filename` the basename is comfy-cli's to derive, so name the dir.

    Guessing a name from the URL would send the caller looking for the wrong
    file — worse than reporting the folder the partial is somewhere inside.
    """
    monkeypatch.setattr(server, "_MIN_LEGACY_DOWNLOAD_TIMEOUT", 0.05)
    # `server`'s deadline arithmetic runs on the fake clock, so the rejected
    # submit hands the transfer the WHOLE sub-second budget however long the
    # thread-pool hop really took — a slow runner can no longer divert this into
    # the exhausted-remainder refusal. The bound itself is still real wall time:
    # `_run_comfy_async` spends it in `asyncio.wait_for`, on the loop's own clock.
    _install_clock(monkeypatch)
    patched_async_run(hang=True)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model("https://hf.co/big.safetensors", timeout_seconds=0.3)

    message = str(excinfo.value)
    assert "models/ (relative to the workspace root" in message  # the default dir
    assert "name from the URL" in message


@pytest.mark.parametrize("wait", [False, True])
def test_download_model_legacy_timeout_at_the_cap_does_not_suggest_a_bigger_bound(
    patched_async_run, legacy_comfy_cli, monkeypatch, wait
):
    """A bound that WAS the cap makes "raise timeout_seconds" dead advice.

    Keyed off the effective bound, not off `wait`, because both callers here were
    already capped: `wait=False` never reads `timeout_seconds` on this path, and a
    waiting caller who passed a value at or above the cap was clamped to it. Either
    way, pointing them at the parameter would send them round a loop that changes
    nothing — the upgrade route is the real one.
    """
    monkeypatch.setattr(server, "_DOWNLOAD_SYNC_TIMEOUT", 0.05)
    patched_async_run(hang=True)

    with pytest.raises(server.ComfyCliError) as excinfo:
        # `timeout_seconds` far above the (shrunken) cap, so the waiting call is
        # clamped to it exactly as `wait=False` always is.
        _download_model(
            "https://hf.co/big.safetensors", wait=wait, timeout_seconds=30.0
        )

    message = str(excinfo.value)
    assert "retry with a larger `timeout_seconds`" not in message
    assert "cannot widen it" in message
    assert "upgrade comfy-cli" in message


def test_download_model_legacy_timeout_message_redacts_the_signed_url(
    patched_async_run, legacy_comfy_cli, monkeypatch
):
    """The argv echoed in a timeout must not carry the URL's credential.

    A CivitAI / HuggingFace model URL keeps its token in a `?token=…` query (or in
    `<user>:<pass>@` userinfo), and this message lands in the tool response the MCP
    client renders and in the host's logs — the same exposure that makes
    `_synthesize_plain_result` omit raw args altogether.
    """
    monkeypatch.setattr(server, "_DOWNLOAD_SYNC_TIMEOUT", 0.05)
    patched_async_run(hang=True)

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model(
            "https://<user>:<pw>@hf.co/big.safetensors?token=SECRETTOKEN", wait=False
        )

    message = str(excinfo.value)
    assert "SECRETTOKEN" not in message
    assert "<user>:<pw>" not in message
    # Masked, not dropped: which command wedged is still legible.
    assert "model download --url https://***@hf.co/big.safetensors" in message


@pytest.mark.parametrize(
    ("relative_path", "filename", "expected"),
    [
        # No `relative_path` -> comfy-cli's own DEFAULT_COMFY_MODEL_PATH.
        (None, None, "models/ (relative"),
        (None, "x.safetensors", "models/x.safetensors (relative"),
        ("models/loras", None, "models/loras/ (relative"),
        ("models/loras", "x.safetensors", "models/loras/x.safetensors (relative"),
    ],
)
def test_legacy_download_partial_names_what_it_can(relative_path, filename, expected):
    """The exact path is only knowable when the caller supplied `filename`.

    Without it comfy-cli derives the basename from the URL, so naming a guessed
    file would send the caller looking for the wrong one — report the directory.
    """
    described = server._legacy_download_partial(relative_path, filename)

    assert described.startswith(expected)
    if not filename:
        assert "name from the URL" in described


def test_download_model_legacy_non_timeout_failure_propagates_unwrapped(
    patched_async_run, legacy_comfy_cli
):
    """Only a TIMEOUT gets the partial-file guidance; a real failure passes through.

    A 404 or a checksum mismatch is comfy-cli's own verdict — dressing it up as
    "the transfer was killed at its bound, a partial may remain" would be a lie.
    """
    patched_async_run(returncode=1, stderr="HTTP 404: model not found")

    with pytest.raises(server.ComfyCliError) as excinfo:
        _download_model("https://hf.co/missing.safetensors")

    message = str(excinfo.value)
    assert "HTTP 404" in message
    assert excinfo.value.timed_out is False
    assert "INCOMPLETE file may remain" not in message


def test_download_model_legacy_cancellation_reaps_the_transfer(
    patched_async_run, legacy_comfy_cli, monkeypatch
):
    """Cancelling the tool call must kill the transfer, not orphan it (BE-5167).

    This is what `asyncio.to_thread(_run_comfy, …)` could not do: its cancellation
    never reached the thread, so an MCP cancel notification (or a stdio shutdown
    tearing the task group down) left `comfy model download` and its multi-GB
    partial running, killable only by pid.
    """
    procs = patched_async_run(hang=True)

    async def drive():
        # Wrap the fixture's fake so the cancel fires at a DETERMINISTIC point —
        # once the fallback child exists. Cancelling on a fixed number of loop
        # turns would race the `to_thread` hop the rejected submit makes first.
        spawned = asyncio.Event()
        fake_exec = server.asyncio.create_subprocess_exec

        async def notifying_exec(*args, **kwargs):
            proc = await fake_exec(*args, **kwargs)
            spawned.set()
            return proc

        monkeypatch.setattr(server.asyncio, "create_subprocess_exec", notifying_exec)
        task = asyncio.ensure_future(
            server.download_model("https://hf.co/big.safetensors", wait=False)
        )
        await spawned.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())

    assert len(procs) == 1
    assert procs[0].killed is True  # the `finally` fired
