"""Catalog routing shares Desktop discovery without making it a remote target."""

from __future__ import annotations

import json
import socket

import pytest
from conftest import envelope

from comfy_mcp import server, target


@pytest.fixture
def desktop_locks(monkeypatch, tmp_path):
    locks = tmp_path / "port-locks"
    locks.mkdir()
    monkeypatch.setattr(target, "_comfy_desktop_port_locks_dir", lambda: locks)
    # Exercise real lock parsing, never inspect an unrelated process.
    monkeypatch.setattr(target, "_pid_is_alive", lambda pid: pid == 42)
    return locks


def _write_lock(locks, port=8003, pid=42):
    (locks / f"port-{port}.json").write_text(json.dumps({"pid": pid}))


def _catalog_call(tool, saved):
    kwargs = {"object_info_path": "catalog.json"} if saved else {}
    if tool == "nodes":
        result = server.nodes(action="types", **kwargs)
        args = ["nodes", "types"]
    else:
        result = server.validate_workflow("workflow.json", **kwargs)
        args = ["validate", "--workflow", "workflow.json"]
    if saved:
        args += ["--input", "catalog.json"]
    return result, args


@pytest.mark.parametrize("tool", ["nodes", "validate"])
@pytest.mark.parametrize("saved", [False, True])
def test_catalog_uses_single_live_desktop(desktop_locks, patched_run, tool, saved):
    _write_lock(desktop_locks)
    data = {"valid": True, "errors": []}
    calls = patched_run(envelope(data=data))

    result, args = _catalog_call(tool, saved)

    assert result == data
    assert calls[0]["cmd"] == [
        server.COMFY_BIN,
        "--json",
        "--where",
        "local",
        *args,
        "--host",
        "127.0.0.1",
        "--port",
        "8003",
    ]
    assert target._comfy_target() is None


@pytest.mark.parametrize("tool", ["nodes", "validate"])
@pytest.mark.parametrize("state", ["none", "stale", "malformed", "ambiguous"])
def test_catalog_does_not_guess_desktop(desktop_locks, patched_run, tool, state):
    if state == "stale":
        _write_lock(desktop_locks, pid=43)
    elif state == "malformed":
        (desktop_locks / "port-8003.json").write_text("{")
    elif state == "ambiguous":
        _write_lock(desktop_locks)
        _write_lock(desktop_locks, port=8004)
    calls = patched_run(envelope(data={}))

    _, args = _catalog_call(tool, False)

    assert calls[0]["cmd"][4:] == args


def _unexpected_discovery():
    pytest.fail("An explicit address must not fall through to Desktop discovery")


@pytest.mark.parametrize("tool", ["nodes", "validate"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize(
    "config, flags",
    [
        ({"COMFY_LOCAL_URL": "http://127.0.0.1:9002"}, []),
        ({"COMFYUI_URL": "http://gpu.example:9001"}, []),
        ({"COMFYUI_HOST": "gpu.example", "COMFYUI_PORT": "9001"}, []),
        (
            {
                "COMFYUI_URL": "http://[::1]:9001",
                "COMFY_LOCAL_URL": "http://127.0.0.1:9002",
            },
            ["--host", "::1", "--port", "9001"],
        ),
    ],
)
def test_catalog_explicit_address_prevents_discovery(
    monkeypatch, patched_run, tool, saved, config, flags
):
    for key, value in config.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(target, "_discover_comfy_desktop_port", _unexpected_discovery)
    calls = patched_run(envelope(data={}))

    _, args = _catalog_call(tool, saved)

    assert calls[0]["cmd"][4:] == [*args, *flags]


@pytest.mark.parametrize("tool", ["nodes", "validate"])
def test_catalog_bad_config_does_not_discover(monkeypatch, patched_run, tool):
    monkeypatch.setenv("COMFYUI_URL", "https://gpu.example")
    monkeypatch.setattr(target, "_discover_comfy_desktop_port", _unexpected_discovery)
    calls = patched_run()

    with pytest.raises(server.ComfyCliError, match="scheme"):
        _catalog_call(tool, False)
    assert calls == []


@pytest.mark.parametrize(
    "host",
    ["gpu.example", "localhost.example", "::1%lo0", "::ffff:127.0.0.1"],
)
def test_catalog_filter_never_uses_dns(monkeypatch, patched_run, host):
    monkeypatch.setenv("COMFYUI_HOST", host)
    monkeypatch.setattr(target, "_discover_comfy_desktop_port", _unexpected_discovery)
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **kw: pytest.fail("DNS lookup")
    )
    calls = patched_run(envelope(data={}))

    server.nodes(action="types")

    assert calls[0]["cmd"][4:] == ["nodes", "types"]


@pytest.mark.parametrize("verb", ["run", "run-template", "jobs", "upload", "models"])
def test_existing_verbs_share_desktop_target(desktop_locks, verb):
    _write_lock(desktop_locks)
    assert target._with_target((verb,)) == (
        verb,
        "--host",
        "127.0.0.1",
        "--port",
        "8003",
    )


def test_desktop_is_not_an_explicit_remote(desktop_locks):
    _write_lock(desktop_locks)
    assert target._catalog_target() == (
        "127.0.0.1",
        8003,
        target._COMFY_DESKTOP_PORT_LOCK_SOURCE,
    )
    assert target._comfy_target() is None
    target._reject_remote_model_download()
    target._reject_remote_restart()
    assert server._remote_target_configured() is False
    payload = {"devices": []}
    assert target._annotate_comfy_target(payload) is payload


def test_template_check_uses_desktop_catalog(desktop_locks, patched_run, tmp_path):
    _write_lock(desktop_locks)
    calls = patched_run(envelope(data={"valid": True, "errors": []}))
    out = tmp_path / "workflow.json"

    check = server.fetch_template("flux_dev", str(out))["local_check"]

    assert calls[1]["cmd"][4:] == [
        "validate",
        "--workflow",
        str(out),
        "--host",
        "127.0.0.1",
        "--port",
        "8003",
    ]
    assert check["runnable"] is True
    assert "non-loopback" not in check["summary"]


@pytest.mark.parametrize("tool", ["fetch", "get"])
@pytest.mark.parametrize("outcome", ["unavailable", "missing_nodes"])
def test_desktop_template_repair_stays_on_desktop(
    desktop_locks, patched_run, tmp_path, tool, outcome
):
    _write_lock(desktop_locks)
    response = (
        envelope(ok=False, error={"code": "cql_no_graph", "message": "no object_info"})
        if outcome == "unavailable"
        else envelope(
            ok=False,
            data={
                "valid": False,
                "error_count": 1,
                "errors": [
                    {
                        "node_id": "12",
                        "code": "missing_node_type",
                        "message": "missing Foo",
                    }
                ],
            },
        )
    )
    validation_batches = []

    def after_fetch(cmd):
        if cmd[4:6] == ["templates", "fetch"]:
            validation_batches.append(patched_run(response, returncode=1))

    calls = patched_run(envelope(data={}), on_spawn=after_fetch)
    if tool == "fetch":
        result = server.fetch_template("flux_dev", str(tmp_path / "workflow.json"))
    else:
        result = server.get_template("flux_dev")

    assert all(call["cmd"][4] == "templates" for call in calls)
    assert len(validation_batches) == 1
    assert len(validation_batches[0]) == 1
    assert validation_batches[0][0]["cmd"][7:] == [
        "--host",
        "127.0.0.1",
        "--port",
        "8003",
    ]
    check = result["local_check"]
    summary = check["summary"]
    assert "non-loopback" not in summary
    assert "launch_comfyui" not in summary
    assert "update_comfyui" not in summary
    if outcome == "unavailable":
        assert check["checked"] is False
        assert check["reason"] == "check_unavailable"
        assert "Start ComfyUI on the configured target" in summary
    else:
        assert check["checked"] is True
        assert check["runnable"] is False
        assert check["errors"] == ["node 12: missing Foo"]
        assert "Update ComfyUI and its custom nodes on the configured target" in summary
