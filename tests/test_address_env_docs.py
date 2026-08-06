"""Guards on the documented split between the two address environment variables.

``COMFYUI_URL`` and ``COMFY_LOCAL_URL`` read like two spellings of one knob and
are not: only the first is **ours**. ``_comfy_target`` reads ``COMFYUI_URL`` /
``COMFYUI_HOST`` / ``COMFYUI_PORT`` and forwards ``--host`` / ``--port``;
``COMFY_LOCAL_URL`` is comfy-cli's own variable, resolved inside comfy-cli off
the environment ``_comfy_env`` forwards, and this server never reads it at all.

That asymmetry is the reason neither name is being renamed, and it lives only in
README prose — no functional test would notice the section going stale, and a
future "strip the word local" pass would happily rename a variable this repo does
not own. These assertions are that tripwire: they key on the SPLIT (who reads
what) rather than on wording, so the section can be rewritten freely.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from comfy_mcp import server

_SECTION = "## Which address variable do I want?"
# Anchored at both ends so the slice cannot start at a DEMOTED (`### …`) copy of
# the heading and cannot swallow the next H2's prose.
_SECTION_START = f"\n{_SECTION}\n"
_SECTION_END = "\n## "

# The package's own sources — every module `server` reaches, since a read of
# comfy-cli's variable would breach the ownership split from any of them.
_PACKAGE_SOURCES = sorted(pathlib.Path(server.__file__).resolve().parent.glob("*.py"))


def _reads_env(source: str, name: str) -> bool:
    """Does ``source`` READ the environment variable ``name``?

    Matches the read SPELLINGS rather than one literal call, so
    ``os.getenv(...)``, subscripting, and either quote style all count — the
    bare name alone would not work, because these variables are named all over
    this package's prose. An indirect read through a constant would still slip
    past; that is the known floor of a source-level tripwire.
    """
    return (
        re.search(
            rf"""(?:environ\.get|getenv)\(\s*["']{name}["']"""
            rf"""|environ\[\s*["']{name}["']\s*\]""",
            source,
        )
        is not None
    )


@pytest.fixture(scope="module")
def readme() -> str:
    return (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text(
        encoding="utf-8"
    )


@pytest.fixture(scope="module")
def section(readme: str) -> str:
    """The comparison section alone. A deleted section fails here first."""
    assert _SECTION_START in readme, (
        f"the address-variable section is gone (or is no longer an H2): {_SECTION!r}"
    )
    body = readme.split(_SECTION_START, 1)[1]
    # Should this ever become the last H2, the slice runs to EOF — still the
    # section, just with nothing after it to cut at.
    return body.split(_SECTION_END, 1)[0]


def test_server_reads_the_comfyui_vars_and_not_comfy_local_url():
    """The ownership claim the README makes, asserted against the source itself.

    If a future change made this server read ``COMFY_LOCAL_URL`` directly, the
    documented "comfy-cli's, not ours" split would become a lie — and the whole
    argument for leaving the name alone would go with it.
    """
    sources = {path.name: path.read_text(encoding="utf-8") for path in _PACKAGE_SOURCES}
    assert "server.py" in sources, "the package layout moved out from under this test"

    for ours in ("COMFYUI_URL", "COMFYUI_HOST", "COMFYUI_PORT"):
        assert _reads_env(sources["server.py"], ours), f"{ours} is no longer read"

    for name, source in sources.items():
        assert not _reads_env(source, "COMFY_LOCAL_URL"), (
            f"{name} now reads COMFY_LOCAL_URL directly — it is comfy-cli's "
            "variable, and the README's ownership split (and the decision not to "
            "rename it) assumes the passthrough, not a direct read"
        )


def test_section_names_the_owner_of_each_variable(section: str):
    """A reader has to be able to tell which program reads which variable."""
    assert "COMFYUI_URL" in section and "COMFY_LOCAL_URL" in section
    assert "comfy-cli" in section, "the section no longer names COMFY_LOCAL_URL's owner"
    assert "never reads it" in section, "the 'not read by this server' claim is gone"


def test_section_states_that_neither_variable_is_deprecated(section: str):
    """Users have these in MCP client configs; the doc must say they still work.

    The ticket that produced this section asked for backwards compatibility with
    a deprecation notice. The resolution was that nothing gets renamed, so the
    compatibility statement IS the deliverable — drop it and a reader is left
    assuming a rename is pending.
    """
    assert "deprecated" in section
    assert "keeps working unchanged" in section


def test_section_warns_against_setting_both(section: str):
    """Set together they split the tool surface across two ComfyUIs, silently."""
    assert "not both" in section


def test_target_aware_tools_do_not_claim_to_be_local_only():
    """The run/job tools FOLLOW ``COMFYUI_URL``; their summaries must not deny it.

    ``_TARGET_AWARE_SUBCOMMANDS`` is
    ``{"run", "run-template", "jobs", "upload"}``, so exactly these tools are
    diverted to a configured remote. Their one-line
    summaries used to open with "LOCAL", which is the one place the local/remote
    distinction is load-bearing and was simply wrong. The tools that genuinely
    stay on this machine (lifecycle, ``fetch_outputs``, discovery, …) keep saying
    so, and are deliberately not covered here.

    ``generate_image`` and ``run_template`` are the ``run-template`` pair: they
    were the tools whose summaries said LOCAL *while* their submissions really
    did stay local, and both halves moved together — forwarding the flags without
    correcting the summary would leave the one sentence a caller reads before
    deciding which machine a job lands on saying the opposite of what happens.
    ``upload_file`` is the same pairing for ``upload``: its summary said LOCAL
    while it staged a remote run's inputs onto the wrong disk.
    """
    for tool in (
        server.run_workflow,
        server.generate_image,
        server.run_template,
        server.job_status,
        server.wait_for_job,
        server.watch_job,
        server.cancel_job,
        server.get_queue,
        server.upload_file,
    ):
        lines = (tool.__doc__ or "").strip().splitlines()
        # Report a missing docstring as itself: under `python -OO` (or if one is
        # deleted) indexing [0] would raise IndexError instead.
        assert lines, f"{tool.__name__} has no docstring for this guard to read"
        summary = lines[0]
        assert "LOCAL" not in summary, (
            f"{tool.__name__} is target-aware but its summary claims LOCAL: {summary!r}"
        )
