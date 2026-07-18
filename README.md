# comfy-local-mcp

A local [MCP](https://modelcontextprotocol.io) server for **ComfyUI** — a thin wrapper over
[`comfy-cli`](https://github.com/Comfy-Org/comfy-cli) that lets AI agents (Claude Code, Claude
Desktop, Cursor, …) drive your **local** ComfyUI: run workflows, wait on jobs, collect the
resulting images, and inspect the nodes/models/templates your install actually has.

It is a small, standalone codebase. Each tool shells out to the `comfy` command with
`--where local --json`, parses comfy-cli's `envelope/1` output, and returns it. There is no HTTP
client and **no code shared with the Comfy Cloud MCP** — comfy-cli is the engine.

> **Status:** early POC. Twenty-two tools, core loop validated end-to-end against a live local
> ComfyUI (`server_info → run_workflow → fetch_outputs` → PNG on disk). CI runs pytest + ruff on
> Python 3.10 and 3.14.

## Prerequisites

- **Python ≥ 3.10.**
- **comfy-cli** on your `PATH`: `pip install comfy-cli`. This is the engine every tool wraps.
- **A ComfyUI workspace.** If you don't have one, `comfy-cli` can create it: `comfy install`
  sets up a ComfyUI workspace it will point at. (An existing ComfyUI checkout works too — see
  `comfy set-default <path>`.)
- **A running ComfyUI.** ComfyUI must be **started before you use the tools** — launch it with
  `comfy launch` (or, from an agent, the `launch_comfyui` tool), and confirm it is up with
  `server_info`. Nothing here starts ComfyUI implicitly.
- **`COMFY_BIN` override (optional).** By default the server calls `comfy` from `PATH`. MCP
  clients launch the server with their own environment, which often does **not** include your
  shell's `PATH` — so if `comfy` lives in a virtualenv or a non-standard location, set
  `COMFY_BIN` to its absolute path (e.g. `/path/to/venv/bin/comfy`). Every client example below
  shows where it goes.

## Install

From a checkout of this repo:

```bash
pip install .          # or `pip install -e .` for a working copy
comfy-local-mcp        # serves the MCP over stdio
```

`pip install` puts a `comfy-local-mcp` console script on your `PATH`; that command is what you
point your AI client at below. (Installing into a dedicated venv is fine — just remember MCP
clients may not see that venv's `PATH`, which is exactly what `COMFY_BIN` is for.)

## Configure your AI client

All three clients speak the same MCP stdio contract: run the `comfy-local-mcp` command as a
server. Pick your client.

> The `COMFY_BIN` env entry is shown in every example. Drop it if `comfy` is already on the
> environment your client launches the server with; keep it (pointing at the absolute path) if
> it isn't.

### Claude Code

One command registers the server:

```bash
claude mcp add comfy-local -e COMFY_BIN=/path/to/venv/bin/comfy -- comfy-local-mcp
```

Or, to check it into a project, add a `.mcp.json` at the repo root:

```json
{
  "mcpServers": {
    "comfy-local": {
      "command": "comfy-local-mcp",
      "env": { "COMFY_BIN": "/path/to/venv/bin/comfy" }
    }
  }
}
```

### Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config; on macOS it lives at
`~/Library/Application Support/Claude/claude_desktop_config.json`) and add the server, then
restart Claude Desktop:

```json
{
  "mcpServers": {
    "comfy-local": {
      "command": "comfy-local-mcp",
      "env": { "COMFY_BIN": "/path/to/venv/bin/comfy" }
    }
  }
}
```

### Cursor

Add the server to `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` in a project:

```json
{
  "mcpServers": {
    "comfy-local": {
      "command": "comfy-local-mcp",
      "env": { "COMFY_BIN": "/path/to/venv/bin/comfy" }
    }
  }
}
```

## Quickstart

Zero to a generated image:

1. **Install the pieces.**

   ```bash
   pip install comfy-cli          # the engine
   comfy install                  # create a ComfyUI workspace (skip if you have one)
   pip install .                  # this MCP server → the `comfy-local-mcp` command
   ```

2. **Launch ComfyUI** and leave it running:

   ```bash
   comfy launch
   ```

3. **Add the server to your client** using the snippet for your client above, then restart /
   reload it so the tools appear.

4. **Ask your agent to run a workflow.** For example:

   > "Confirm my local ComfyUI is running, then run the workflow at
   > `~/workflows/txt2img.json` and show me the image."

   Under the hood the agent calls `server_info` to confirm ComfyUI is up, `run_workflow` to
   execute your workflow JSON (API-format or a UI export), and `fetch_outputs` to collect the
   result. No hand-authored workflow? Ask it to start from a template instead — it can
   `search_templates`, `fetch_template` to write a runnable JSON, and run that.

**Where the images land.** ComfyUI writes generated files into your ComfyUI **workspace's
`output/` directory** (part of the workspace `comfy install` created). On top of that,
`fetch_outputs(prompt_id, out_dir)` **copies** a finished job's outputs into any directory you
name — so telling the agent "save them to `./outputs`" puts a copy right where you asked while
the originals stay in the ComfyUI workspace.

## Tools

Twenty-two tools. Every tool runs `comfy` with the global `--json --where local` flags, unwraps
comfy-cli's `envelope/1`, and returns its `data`.

| Tool | Wraps | Purpose |
|---|---|---|
| `server_info()` | `comfy env` | Is a local ComfyUI running, where, and which workspace. **Call first.** |
| `auth_status()` | `comfy cloud whoami` | Comfy Cloud credential status for partner-API nodes (read-only, never returns secrets). Adds a local `registration_env_key_present` bool for the `COMFY_API_KEY` registration-env slot whoami can't see. |
| `run_workflow(workflow_path, wait=True, timeout_seconds=600.0)` | `comfy run --workflow <path> [--wait]` | Run a workflow JSON; `wait=False` submits async and returns a `prompt_id`. |
| `job_status(prompt_id)` | `comfy jobs status <prompt_id>` | Poll a submitted job's status + outputs. |
| `wait_for_job(prompt_id, timeout_seconds=25.0)` | `comfy jobs status <prompt_id>` (polled) | Bounded wait until a job reaches a terminal status; returns a `{"timed_out": True, …}` payload on expiry. Chain several. |
| `watch_job(prompt_id, timeout_seconds=600.0)` | `comfy jobs watch <prompt_id>` (streamed) | Tail an async-submitted job's live execution, streaming progress notifications; bounded, returns a `{"timed_out": True, …}` payload on expiry. Streaming counterpart to `wait_for_job`. |
| `cancel_job(prompt_id)` | `comfy jobs cancel <prompt_id>` | Cancel a queued or running job. |
| `get_queue()` | `comfy jobs ls` | List known jobs with status (pending/running/completed). |
| `fetch_outputs(prompt_id, out_dir, url_only=False)` | `comfy download <prompt_id> --where local -o <out_dir> [--url-only]` | Wraps `comfy download --where local` to write a finished local job's outputs into `out_dir`; `url_only=True` emits the output URLs without copying bytes. |
| `launch_comfyui(extra_args=None)` | `comfy launch --background [-- <extras>]` | Start the local ComfyUI detached; forwards `extra_args` to ComfyUI. |
| `stop_comfyui()` | `comfy stop` | Stop the ComfyUI that comfy-cli launched (only its own recorded pid). |
| `discover()` | `comfy discover` | comfy-cli's self-describing surface (commands, arg schemas, error codes) — learn the CLI's own contract at runtime. |
| `which()` | `comfy which` | Which ComfyUI install/workspace comfy-cli currently targets (a lighter answer than `server_info`). |
| `search_templates(query="")` | `comfy templates ls` (filtered client-side) | Find a built-in workflow template by name/description; empty `query` lists all. |
| `get_template(name)` | `comfy templates show <name>` | Show one template's details/schema before fetching it. |
| `fetch_template(name, out_path)` | `comfy templates fetch <name> --out <path>` | Write a template's runnable workflow JSON to `out_path`; returns the absolute path for `run_workflow`. |
| `search_nodes(query)` | `comfy nodes search <query>` | Find node classes in the **live local** `object_info` (includes installed custom nodes). |
| `get_node(name)` | `comfy nodes show <ClassName>` | Full input/output schema for one node class — what you need to author/repair a graph. |
| `list_nodes(produces="", accepts="", category="", pack="", label="")` | `comfy nodes ls [--produces/--accepts/--category/--pack/--label …]` | List node classes, filtered by output/input type, category, pack, or label; bare call lists all. Reads the **live install**. |
| `nodes_upstream(name, limit=None)` | `comfy nodes upstream <name> [--limit N]` | Nodes whose outputs can feed `<name>`'s inputs ("what wires INTO this?"). Reads the **live install**. |
| `nodes_downstream(name, limit=None)` | `comfy nodes downstream <name> [--limit N]` | Nodes that accept `<name>`'s output types ("what does this wire INTO?"). Reads the **live install**. |
| `nodes_path(from_type, to_type, max_depth=6, max_paths=10)` | `comfy nodes path <FROM> <TO> --max-depth N --max-paths N` | Node chains routing a value between two connection types (e.g. `MODEL` → `IMAGE`). Reads the **live install**. |
| `nodes_types()` | `comfy nodes types` | All connection types (`MODEL`, `IMAGE`, …) ranked by connectivity. Reads the **live install**. |
| `nodes_categories()` | `comfy nodes categories` | The node category tree. Reads the **live install**. |
| `search_models(query="", folder="")` | `comfy models search` / `models list-folder <folder>` / `models list-folders` | List/search model files on disk. **Local:** filenames only, no cloud enrichment. |
| `upload_file(paths, overwrite=False)` | `comfy upload <files...> [--overwrite]` | Stage source images/masks into the local `input` dir (unlocks img2img / inpaint). |
| `validate_workflow(workflow_path)` | `comfy validate --workflow <path>` | Pre-flight a workflow against the live `object_info` before a slow run; surfaces the structured error code on failure. |
| `list_workflow_slots(workflow_path)` | `comfy workflow slots <path>` | List the agent-tweakable slots (addresses + current values) a frontend-format workflow exposes. |
| `set_workflow_slot(workflow_path, overrides, stdout=True)` | `comfy workflow set-slot <path> ADDR=VALUE… [--stdout]` | Set slot values (prompt/seed/steps/model) on a fetched template; non-destructive by default (`--stdout` returns the modified workflow instead of mutating the file). |
| `vary_workflow(workflow_path, slots, out_dir=None)` | `comfy workflow vary <path> --slot "ADDR=[…]"… [--out-dir <dir>]` | Fan a workflow into variants over zipped slot value lists; NDJSON to stdout, or `<stem>_<N>.json` files when `out_dir` is set. |

Node introspection (`search_nodes` / `get_node` / `list_nodes` / `nodes_upstream` /
`nodes_downstream` / `nodes_path` / `nodes_types` / `nodes_categories`) and `search_models`
read the **user's live install** (custom nodes included), not a static catalog — that's the
local differentiator from the cloud MCP's equivalents. The graph-wiring verbs (`upstream` /
`downstream` / `path`) are what an agent authoring a workflow uses to find compatible nodes.

## Smoke test

Turn the manual validation ritual into one command. The e2e smoke test drives the
real tools end-to-end (no mocks): `server_info` → `run_workflow` on a checkpoint-free
`EmptyImage` → `SaveImage` graph → `fetch_outputs`, and asserts a valid PNG lands in
a temp out_dir.

```bash
./scripts/smoke.sh            # or: python -m pytest tests/e2e -m e2e
```

It needs a running local ComfyUI (`COMFYUI_URL`, default `http://127.0.0.1:8188`)
**and** the `comfy` binary on `PATH` (or `COMFY_BIN`). Without both it **skips**
rather than fails, so it's safe to run anywhere — and the plain `pytest` gate stays
green on CI runners that have neither.

## License

GPL-3.0-only (matching comfy-cli / ComfyUI).
