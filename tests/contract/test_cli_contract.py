"""Live contract checks against a real installed comfy-cli binary.

The unit suite mocks every spawn, so the argv/envelope contract it protects
is otherwise never exercised against the real, unpinned comfy-cli. These
tests need the binary only — no running ComfyUI — and run in the scheduled
cli-contract workflow (and locally via `pytest -m cli_contract` when comfy
is on PATH). A failure here means comfy-cli moved, not that this repo broke.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from comfy_mcp import server

pytestmark = pytest.mark.cli_contract


@pytest.fixture(scope="module", autouse=True)
def _needs_binary():
    if shutil.which(server.COMFY_BIN) is None:
        pytest.skip(f"`{server.COMFY_BIN}` not on PATH")


def test_env_returns_a_current_major_envelope_with_a_dict_payload():
    data = server._run_comfy("env", timeout=60.0)
    assert isinstance(data, dict), f"comfy env payload drifted: {type(data)}"
    assert "server" in data or "workspace" in data, (
        f"comfy env payload keys drifted: {sorted(data)[:10]}"
    )


def test_templates_ls_payload_has_the_rows_contract():
    data = server._run_comfy("templates", "ls", timeout=60.0)
    assert isinstance(data, dict) and isinstance(data.get("rows"), list), (
        "templates ls payload drifted — search_templates parses "
        f"{{rows: [...]}}, got: {type(data)}"
    )
    if data["rows"]:
        row = data["rows"][0]
        missing = {"name", "title", "output_type", "tags"} - set(row)
        assert not missing, f"template row lost fields the projection relays: {missing}"


def test_project_init_and_status_round_trip(tmp_path, monkeypatch):
    """COMFY_PROJECT anchoring against the REAL binary: init then status agree
    on `root`, and status's `project/1` payload has the shape `project` parses.
    """
    monkeypatch.setenv("COMFY_PROJECT", str(tmp_path))

    init_data = server.project(action="init")
    assert init_data["root"] == str(tmp_path), (
        f"comfy project init did not anchor to COMFY_PROJECT: {init_data.get('root')!r}"
    )

    status_data = server.project(action="status")
    assert status_data["root"] == str(tmp_path)
    assert status_data["schema"] == "project/1", (
        f"project status schema drifted: {status_data.get('schema')!r}"
    )
    assert isinstance(status_data.get("recent_runs"), list), (
        f"recent_runs is not a list: {type(status_data.get('recent_runs'))}"
    )


def test_cloud_status_verb_exists_on_the_installed_comfy_cli():
    """`billing_status`' degrade branch is about a version gap, so prove the gap.

    `comfy-mcp` 0.11.0 shipped documentation, a comment and a test all saying
    no tagged comfy-cli release had `comfy cloud status` yet. It had shipped 86
    days earlier, in 1.17.0, and nothing in CI disagreed because nothing asked
    the binary. This does: a `--help` on the leaf exits 0 while Click exits 2
    with a usage dump for a verb it cannot find, which is the same signal
    `clitext._is_missing_verb_error` reads at runtime.

    Failing here means the verb was REMOVED or renamed upstream, which would
    make the degrade the common path again — exactly the claim this repo got
    wrong by asserting it in prose.
    """
    proc = subprocess.run(  # fixed argv, no shell, no caller input
        [server.COMFY_BIN, "cloud", "status", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        "`comfy cloud status --help` did not exit 0 — the verb billing_status "
        f"wraps may be gone (floor {server._MIN_BILLING_COMFY_CLI_STR}). "
        f"rc={proc.returncode} stderr={proc.stderr[-400:]!r}"
    )


def test_templates_ls_advertises_the_runnable_filter():
    """`search_templates(runnable_only=True)` names a floor, so prove the floor.

    This points the OPPOSITE way to the `cloud status` check above. That one
    guards against a verb being removed; this one guards against a floor claim
    that was never true. The `--runnable` degrade used to say the filter "landed
    after comfy-cli 1.26.0" — correct while nothing carried it, and a riddle
    once 1.27.0 did. Naming 1.27.0 instead is only worth doing if something
    checks it, and nothing mocked can: the unit suite asserts the message a
    SYNTHETIC `NoSuchOption` produces, which is true whatever version is named.

    The workflow installs the newest released comfy-cli, so a failure here means
    either the flag never shipped where this repo says it did, or upstream took
    it away — and both make the degrade the common path, which is exactly the
    claim the repo got wrong by asserting it in prose.
    """
    proc = subprocess.run(  # fixed argv, no shell, no caller input
        [server.COMFY_BIN, "templates", "ls", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"`comfy templates ls --help` did not exit 0: rc={proc.returncode} "
        f"stderr={proc.stderr[-400:]!r}"
    )
    assert "--runnable" in proc.stdout, (
        "`comfy templates ls` does not advertise `--runnable`, which this repo "
        f"says comfy-cli {server._MIN_RUNNABLE_COMFY_CLI_STR} and newer carry "
        "(search_templates(runnable_only=True) wraps it). "
        f"--help stdout={proc.stdout[-400:]!r}"
    )


def test_models_search_advertises_the_host_forward():
    """`search_models`' remote mode names a floor too, and the stakes are higher.

    The version-skew retry exists because `comfy models` once rejected the
    forwarded `--host`. Its refusal message now names comfy-cli
    `_MIN_MODELS_HOST_COMFY_CLI_STR` as the release that accepts it. If that is
    wrong, the failure is not cosmetic: the whole point of forwarding the flag
    is that an unforwarded `comfy models` answers "what models are installed?"
    about THIS machine while `run_workflow` submits to the remote, so an agent
    gates a run — or a redundant multi-GB `download_model` — on the wrong disk.
    """
    proc = subprocess.run(  # fixed argv, no shell, no caller input
        [server.COMFY_BIN, "models", "search", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"`comfy models search --help` did not exit 0: rc={proc.returncode} "
        f"stderr={proc.stderr[-400:]!r}"
    )
    assert "--host" in proc.stdout, (
        "`comfy models search` does not advertise `--host`, which this repo "
        f"says comfy-cli {server._MIN_MODELS_HOST_COMFY_CLI_STR} and newer "
        "carry — without it search_models answers about the wrong machine. "
        f"--help stdout={proc.stdout[-400:]!r}"
    )
