# comfy-local-mcp

A local [MCP](https://modelcontextprotocol.io) server for **ComfyUI** — a thin wrapper over
[`comfy-cli`](https://github.com/Comfy-Org/comfy-cli) that lets AI agents (Claude, Cursor, …)
drive your **local** ComfyUI.

It is a small, standalone codebase. Each tool shells out to the `comfy` command with
`--where local --json`, parses comfy-cli's `envelope/1` output, and returns it. There is no HTTP
client and **no code shared with the Comfy Cloud MCP** — comfy-cli is the engine.

> **Status:** early POC. Four tools; the comfy-cli plumbing is smoke-tested, a full generation round-trip is pending a running ComfyUI.

## Prerequisites

- Python ≥ 3.10
- [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli) on your `PATH` (`pip install comfy-cli`),
  pointed at a local ComfyUI workspace. (Override the binary with the `COMFY_BIN` env var.)
- A local ComfyUI you can run (`comfy launch`).

## Tools (first cut)

| Tool | Wraps | Purpose |
|---|---|---|
| `server_info()` | `comfy env` | Is a local ComfyUI running, where, and which workspace. Call first. |
| `run_workflow(workflow_path, wait=True, timeout_seconds=600)` | `comfy run --workflow <path> [--wait]` | Run a workflow JSON; `wait=False` submits async and returns a `prompt_id`. |
| `job_status(prompt_id)` | `comfy jobs status <prompt_id>` | Poll a submitted job's status + outputs. |
| `fetch_outputs(prompt_id, out_dir)` | `comfy download <prompt_id> --out-dir <dir>` | Download a finished job's outputs to disk. |

Planned next, each a one-line passthrough: `discover` (`comfy discover`),
`launch`/`stop` (`comfy launch --background` / `comfy stop`) — plus a real
generation round-trip test.

## Run

```bash
pip install -e .
comfy-local-mcp   # serves over stdio
```

Point your MCP client at the `comfy-local-mcp` command.

## License

GPL-3.0-only (matching comfy-cli / ComfyUI).
