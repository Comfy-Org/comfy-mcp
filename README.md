<div align="center">

<img src="assets/logo.svg" alt="Comfy" width="160"/>

<h1>Comfy Local MCP</h1>

**Drive your local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) from Claude Code, Claude Desktop, Cursor, or any MCP-speaking AI agent — a local [MCP](https://modelcontextprotocol.io) server built on [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli).**

<p>
  <a href="https://github.com/comfyanonymous/ComfyUI"><img src="https://img.shields.io/badge/ComfyUI-local-blue?style=for-the-badge" alt="ComfyUI local"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-Compatible-green?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTUtMTAtNXoiLz48cGF0aCBkPSJNMiAxN2wxMCA1IDEwLTUiLz48cGF0aCBkPSJNMiAxMmwxMCA1IDEwLTUiLz48L3N2Zz4=" alt="MCP"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-%E2%89%A5%203.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
</p>

<p>
  <a href="https://github.com/Comfy-Org/comfy-local-mcp/actions/workflows/ci.yml"><img src="https://github.com/Comfy-Org/comfy-local-mcp/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Comfy-Org/comfy-local-mcp/releases/latest"><img src="https://img.shields.io/github/v/release/Comfy-Org/comfy-local-mcp" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

<p>
  <a href="#install"><strong>Install</strong></a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#tools">Tools</a> ·
  <a href="#configure-your-ai-client">Configure your client</a> ·
  <a href="#contributing">Contributing</a>
</p>

</div>

**What it does:**

- 🖼️ **Generate** — run a workflow JSON (API-format or UI export), or go text-prompt → image in one call.
- ⏱️ **Monitor jobs** — submit async, then wait / watch / cancel, read the failure verdict, and collect the output PNGs.
- 🔍 **Introspect your live install** — search the nodes, models, and templates your ComfyUI *actually* has (custom nodes included), not a static catalog.
- 🧩 **Build workflows** — validate a graph, edit a template's slots, and fan one workflow into variants.
- ♻️ **Manage ComfyUI** — launch / stop / restart the server, tail its logs, and stage input assets.

Each tool shells out to the `comfy` command with `--where local --json`, parses comfy-cli's
`envelope/1` output, and returns it. There is no HTTP client and **no code shared with the Comfy
Cloud MCP** — comfy-cli is the engine.

> **Status:** beta. 35 tools; core loop validated end-to-end against a live local ComfyUI
> (`server_info → run_workflow → fetch_outputs` → PNG on disk). CI runs pytest + ruff on
> Python 3.10 and 3.14.

## Table of contents

- [Prerequisites](#prerequisites)
- [Partner-API nodes](#partner-api-nodes)
- [Install](#install)
- [Configure your AI client](#configure-your-ai-client)
- [Quickstart](#quickstart)
- [Tools](#tools)
- [Smoke test](#smoke-test)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- **Python ≥ 3.10.**
- **comfy-cli ≥ 1.12.0** on your `PATH`: `pip install 'comfy-cli>=1.12.0'`. This is the engine
  every tool wraps; the server refuses to run against an older comfy-cli with an upgrade message.
- **A ComfyUI workspace.** If you don't have one, `comfy-cli` can create it: `comfy install`
  sets up a ComfyUI workspace it will point at. (An existing ComfyUI checkout works too — see
  `comfy set-default <path>`.)
- **A running ComfyUI.** ComfyUI must be **started before you use the tools** — launch it with
  `comfy launch` (or, from an agent, the `launch_comfyui` tool), and confirm it is up with
  `server_info`. Nothing here starts ComfyUI implicitly.

<details>
<summary><strong>Optional environment variables</strong> (<code>COMFY_BIN</code>, <code>COMFY_API_KEY</code>)</summary>

<br>

- **`COMFY_BIN` override (optional).** By default the server calls `comfy` from `PATH`. MCP
  clients launch the server with their own environment, which often does **not** include your
  shell's `PATH` — so if `comfy` lives in a virtualenv or a non-standard location, set
  `COMFY_BIN` to its absolute path (e.g. `/path/to/venv/bin/comfy`). Every client example below
  shows where it goes.
- **`COMFY_API_KEY` (optional — needed only for partner-API nodes).** Workflows that use
  partner-API nodes (Seedream / Seedance / Nano Banana / Gemini / Veo / Kling / …) need a Comfy
  credential, and — exactly like `COMFY_BIN` — an MCP client launches the server with its own
  minimal environment, so a key from your shell won't reach it. Set `COMFY_API_KEY` in the
  client registration `env` block. See **[Partner-API nodes](#partner-api-nodes)** below for the
  full precedence chain; every client example shows where it goes.

</details>

## Partner-API nodes

Some ComfyUI nodes call out to Comfy's partner APIs (Seedream / Seedance / Nano Banana / Gemini /
Veo / Kling / …). Running one **locally** still needs a Comfy credential, and comfy-cli resolves
it in this order (first match wins):

1. a per-call flag (not exposed by this server);
2. a live Comfy Cloud OAuth session (`comfy cloud login`);
3. the **`COMFY_API_KEY`** environment variable;
4. a stored key set with `comfy auth set comfy-cloud-api-key --key <KEY>`.

Because an MCP client spawns the server with its own minimal environment (the same reason
`COMFY_BIN` exists), a `COMFY_API_KEY` from your interactive shell is **not** inherited — put it
in the client registration `env` block (shown in every example below). If a run fails with
`partner_node_requires_credential`, the error now carries comfy-cli's hint verbatim, including
the `comfy auth set comfy-cloud-api-key --key …` fallback and the list of offending nodes; the
server also retries a transient credential failure briefly before surfacing it.

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
> it isn't. `COMFY_API_KEY` is also shown, commented as optional — keep it only if you use
> [partner-API nodes](#partner-api-nodes) (Seedream / Veo / Kling / Gemini / …); drop it
> otherwise.

### Claude Code

One command registers the server:

```bash
# COMFY_API_KEY is optional — add it only if you use partner-API nodes (see above).
claude mcp add comfy-local \
  -e COMFY_BIN=/path/to/venv/bin/comfy \
  -e COMFY_API_KEY=<your-comfy-api-key> \
  -- comfy-local-mcp
```

Or, to check it into a project, add a `.mcp.json` at the repo root:

```json
{
  "mcpServers": {
    "comfy-local": {
      "command": "comfy-local-mcp",
      "env": {
        "COMFY_BIN": "/path/to/venv/bin/comfy",
        "COMFY_API_KEY": "<your-comfy-api-key>"
      }
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
      "env": {
        "COMFY_BIN": "/path/to/venv/bin/comfy",
        "COMFY_API_KEY": "<your-comfy-api-key>"
      }
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
      "env": {
        "COMFY_BIN": "/path/to/venv/bin/comfy",
        "COMFY_API_KEY": "<your-comfy-api-key>"
      }
    }
  }
}
```

## Quickstart

Zero to a generated image:

1. **Install the pieces.**

   ```bash
   pip install 'comfy-cli>=1.12.0'  # the engine (>= 1.12.0 required)
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

35 tools, grouped below by what they do. Every tool runs `comfy` with the global
`--json --where local` flags, unwraps comfy-cli's `envelope/1`, and returns its `data`.

### Run and monitor

| Tool | Wraps | What it does |
|---|---|---|
| `run_workflow(workflow_path, wait=True, timeout_seconds=600.0)` | `comfy run --workflow <path> [--wait]` | Run a workflow JSON; `wait=False` submits async and returns a `prompt_id`. |
| `generate_image(prompt, checkpoint=None, wait=True, timeout_seconds=600.0)` | `comfy generate --prompt <prompt> [--checkpoint <ckpt>]` | Text prompt → image in one call — comfy-cli owns the graph/checkpoint injection, so no hand-assembled workflow needed. Same envelope shape as `run_workflow` (`prompt_id` + outputs); the fast on-ramp. |
| `job_status(prompt_id)` | `comfy jobs status <prompt_id>` | Poll a submitted job's status + outputs. |
| `wait_for_job(prompt_id, timeout_seconds=25.0)` | `comfy jobs status <prompt_id>` (polled) | Bounded wait until a job reaches a terminal status; returns a `{"timed_out": True, …}` payload on expiry. Chain several. |
| `watch_job(prompt_id, timeout_seconds=600.0)` | `comfy jobs watch <prompt_id>` (streamed) | Tail an async-submitted job's live execution, streaming progress notifications; bounded, returns a `{"timed_out": True, …}` payload on expiry. Streaming counterpart to `wait_for_job`. |
| `get_execution_error(prompt_id)` | `comfy jobs status <prompt_id>` | Compact failure verdict for a failed run — the failing node, `exception_type`/`exception_message`, and a bounded traceback tail — so an agent can self-repair; returns `error: None` on a healthy prompt. |
| `cancel_job(prompt_id)` | `comfy jobs cancel <prompt_id>` | Cancel a queued or running job. |
| `get_queue()` | `comfy jobs ls` | List known jobs with status (pending/running/completed). |
| `fetch_outputs(prompt_id, out_dir, url_only=False, inline_images=False)` | `comfy download <prompt_id> --where local -o <out_dir> [--url-only]` | Write a finished local job's outputs into `out_dir`; `url_only=True` emits the output URLs without copying bytes; `inline_images=True` also returns the copied images as inline MCP image content so the agent can see them without a second read. |

### Diagnostics

| Tool | Wraps | What it does |
|---|---|---|
| `server_info()` | `comfy env` | Is a local ComfyUI running, where, and which workspace. **Call first.** |
| `auth_status()` | `comfy cloud whoami` | Comfy Cloud credential status for partner-API nodes (read-only, never returns secrets). Adds a local `registration_env_key_present` bool for the `COMFY_API_KEY` registration-env slot whoami can't see. |
| `which()` | `comfy which` | Which ComfyUI install/workspace comfy-cli currently targets (a lighter answer than `server_info`). |
| `get_logs(tail=200)` | `comfy logs --tail <tail>` | Tail the background ComfyUI's captured log (`<workspace>/user/comfyui_<port>.log`) — closes the debugging loop after a detached `launch_comfyui`. Returns `{lines, path, truncated}`; a missing log file returns `{"error": "no_log_file", …}` rather than raising. |
| `discover()` | `comfy discover` | comfy-cli's self-describing surface (commands, arg schemas, error codes) — learn the CLI's own contract at runtime. |

### Workflow building

| Tool | Wraps | What it does |
|---|---|---|
| `validate_workflow(workflow_path)` | `comfy validate --workflow <path>` | Pre-flight a workflow against the live `object_info` before a slow run; surfaces the structured error code on failure. |
| `list_workflow_slots(workflow_path)` | `comfy workflow slots <path>` | List the agent-tweakable slots (addresses + current values) a frontend-format workflow exposes. |
| `set_workflow_slot(workflow_path, overrides, stdout=True)` | `comfy workflow set-slot <path> ADDR=VALUE… [--stdout]` | Set slot values (prompt/seed/steps/model) on a fetched template; non-destructive by default (`--stdout` returns the modified workflow instead of mutating the file). |
| `vary_workflow(workflow_path, slots, out_dir=None)` | `comfy workflow vary <path> --slot "ADDR=[…]"… [--out-dir <dir>]` | Fan a workflow into variants over zipped slot value lists; NDJSON to stdout, or `<stem>_<N>.json` files when `out_dir` is set. |

### Discovery and templates

| Tool | Wraps | What it does |
|---|---|---|
| `search_templates(query="", limit=25, offset=0, tag="", type="", model="", provider="", exclude_api=False)` | `comfy templates ls [--tag/--type/--model/--provider …]` | Find a built-in workflow template: free-text `query` (client-side over name/title/description/tags/models), paged via `limit`/`offset`, narrowed by the `tag`/`type`/`model`/`provider` gallery filters or `exclude_api=True`. Returns `{total, shown, offset, rows:[{name,title,description,output_type}]}`. |
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

### Lifecycle and assets

| Tool | Wraps | What it does |
|---|---|---|
| `launch_comfyui(extra_args=None)` | `comfy launch --background [-- <extras>]` | Start the local ComfyUI detached; forwards `extra_args` to ComfyUI. |
| `stop_comfyui()` | `comfy stop` | Stop the ComfyUI that comfy-cli launched (only its own recorded pid). |
| `restart_comfyui(extra_args=None)` | `comfy stop` then `comfy launch --background [-- <extras>]` | Stop-then-launch the local ComfyUI (best-effort stop); forwards `extra_args` to the fresh server. Handy for relaunching with different flags. |
| `upload_file(paths, overwrite=False)` | `comfy upload <files...> [--overwrite]` | Stage source images/masks into the local `input` dir (unlocks img2img / inpaint). |
| `download_model(url, relative_path=None, filename=None)` | `comfy model download --url <url> [--relative-path <path>] [--filename <name>]` | Download a model file by direct URL (HuggingFace / CivitAI) into the local models dir; download-by-URL only, not a hub search. |

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

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev
setup (`pip install -e '.[dev]'`, `pytest`, `ruff`) and the thin-wrapper
architecture rule, and [`AGENTS.md`](AGENTS.md) for the full guidelines. This
project follows a [Code of Conduct](CODE_OF_CONDUCT.md). To report a
vulnerability, see [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). It wraps the
GPL-3.0 [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli) by shelling out to
the `comfy` binary as a separate process — no GPL code is imported or linked.
`comfy-cli` remains GPL-3.0-licensed and is distributed separately; how its
copyleft applies depends on how the programs interact.
