<!--
mcp-name: io.github.Comfy-Org/comfy-mcp

The line above is not decoration and not a comment for readers: it is the
ownership token the official MCP Registry checks. Publishing the listing in
`server.json` makes registry.modelcontextprotocol.io fetch this package's
metadata from PyPI and look for `mcp-name: <the server.json name>` in the
description — which for this project is this file, verbatim, as
`[project] readme` in pyproject.toml. Finding it is how the registry proves
that whoever owns the GitHub namespace also owns the PyPI package.

So the string must stay byte-identical to `server.json`'s `name` (including
its capitalisation), must stay in README.md rather than move to a file PyPI
never sees, and must keep a space or a newline after it — a token glued to
the following character is read as a prefix of some longer name and rejected.
tests/test_packaging.py pins all three. Removing it does not break anything
here; it breaks the next release's registry publish.
-->

<div align="center">

<img src="assets/logo.svg" alt="Comfy" width="160"/>

<h1>Comfy MCP</h1>

**Drive your own [ComfyUI](https://github.com/comfyanonymous/ComfyUI) from Claude Code, Claude Desktop, Cursor, or any MCP-speaking AI agent — an [MCP](https://modelcontextprotocol.io) server built on [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli).**

<p>
  <a href="https://github.com/comfyanonymous/ComfyUI"><img src="https://img.shields.io/badge/ComfyUI-self--hosted-blue?style=for-the-badge" alt="ComfyUI self-hosted"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-Compatible-green?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTUtMTAtNXoiLz48cGF0aCBkPSJNMiAxN2wxMCA1IDEwLTUiLz48cGF0aCBkPSJNMiAxMmwxMCA1IDEwLTUiLz48L3N2Zz4=" alt="MCP"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-%E2%89%A5%203.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
</p>

<p>
  <a href="https://github.com/Comfy-Org/comfy-mcp/actions/workflows/ci.yml"><img src="https://github.com/Comfy-Org/comfy-mcp/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Comfy-Org/comfy-mcp/releases/latest"><img src="https://img.shields.io/github/v/release/Comfy-Org/comfy-mcp" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0--or--later%20OR%20Commercial-blue.svg" alt="License: AGPL-3.0-or-later OR Commercial"></a>
</p>

<p>
  <a href="#quickstart"><strong>Quickstart</strong></a> ·
  <a href="#configure-your-ai-client">Configure your client</a> ·
  <a href="#comfy-cloud-mcp">Comfy Cloud MCP</a> ·
  <a href="#tools">Tools</a> ·
  <a href="#contributing">Contributing</a>
</p>

</div>

> Looking for the cloud-hosted version? [Comfy Cloud MCP](#comfy-cloud-mcp) is set up below — install
> it instead of this server, or alongside it.

**What it does:**

- 🖼️ **Generate** — run a workflow JSON (API-format or UI export), or go text-prompt → image in one call.
- ⏱️ **Monitor jobs** — submit async, then wait / watch / cancel, read the failure verdict, and collect the output PNGs.
- 🔍 **Introspect your live install** — search the nodes, models, and templates your ComfyUI *actually* has (custom nodes included), not a static catalog.
- 🧩 **Build workflows** — validate a graph, edit a template's slots, and fan one workflow into variants.
- ♻️ **Manage ComfyUI** — launch / stop / restart the server, tail its logs, and stage input assets.

Each tool shells out to the `comfy` command with `--where local --json`, parses comfy-cli's
`envelope/1` output, and returns it — comfy-cli is the engine, and by default everything targets
the ComfyUI on **your** machine (`127.0.0.1:8188`).

**Scope — local-first, not local-only.** A few flows already reach beyond your machine:
[`partner_generate`](#spending-credits-on-partner-models) runs hosted partner models
(Flux / Ideogram / Kling / …) entirely on partner infrastructure — no local ComfyUI in the
execution path — and [partner-API nodes](#partner-api-nodes) let a locally-executed workflow call
those same hosted models, while [`COMFYUI_URL`](#driving-a-remote-comfyui) points the run/job tools
at a ComfyUI on another machine you control.

**This server vs. [Comfy Cloud MCP](#comfy-cloud-mcp).** Two different servers, and running both is
normal. This one runs on **a machine you control**, either as a client-launched stdio subprocess or
as your own [long-running Remote HTTP server](#remote-streamable-http); it drives that machine's
comfy-cli and ComfyUI install (or a separately configured ComfyUI you control). Comfy Cloud MCP is
a managed HTTP server at `https://cloud.comfy.org/mcp` that executes workflows on **Comfy Cloud
GPUs** — no local GPU or ComfyUI install. Both
authenticate: this one signs in to Comfy through comfy-cli ([`auth_login`](#partner-api-nodes) or
`COMFY_API_KEY`), the cloud one through OAuth in your browser or a Comfy Cloud API key. Both can
spend credits on **partner** models, so partner generation is not the dividing line — what this
server has no path to is Comfy Cloud itself: no cloud-hosted execution, no cloud queue, no
cross-session cloud batches. Every tool here shells out to `comfy --where local`. Pick by where you
want the work to run, or [install the cloud server too](#comfy-cloud-mcp).

> **Status:** beta. stdio and Streamable HTTP serve the same 39-tool FastMCP application. The core loop is validated against a live local ComfyUI
> (`server_info → run_workflow → fetch_outputs` → PNG on disk); automated transport tests also cover command-based HTTP upload and signed output delivery. CI runs pytest + ruff on
> Python 3.10 and 3.14.

## Quickstart

Four steps take you from a fresh install to your first generated image.

1. **Install the pieces.**

   ```bash
   pip install comfy-mcp "comfy-cli>=1.14.0"  # this server + the engine it wraps
   comfy install                              # create a ComfyUI workspace (skip if you have one)
   ```

   **Both names, one command — on purpose.** Installing `comfy-mcp` does **not** install
   comfy-cli: it is not a declared dependency of this package, because the server runs whichever
   `comfy` binary your `PATH` (or [`COMFY_BIN`](#prerequisites)) resolves to, which is often not
   the environment you installed the server into. Declaring it would put a second copy in this
   venv that may not be the one your tools actually drive; the version that matters is checked at
   **runtime** instead, against the binary really being called. Skip it and the server still
   starts and completes the MCP handshake — every tool call then fails with "`comfy` not found on
   PATH", which is a missing engine, not a broken install.

   `pip install comfy-mcp` puts a `comfy-mcp` console script on your `PATH`; that command is what
   you point your AI client at in step 3. (A dedicated venv is fine — MCP clients may not see that
   venv's `PATH`, which is exactly what `COMFY_BIN` is for; see [Prerequisites](#prerequisites).)
   `comfy-mcp --version` confirms it landed — but a bare `comfy-mcp` is the **stdio** mode, so in a
   terminal it waits for MCP on stdin and exits without printing anything. Use `comfy-mcp serve`
   for a persistent HTTP listener; `comfy-mcp --help` shows both modes.

   To install from a checkout of this repo instead, run `pip install .` there (`pip install -e .`
   for a working copy) — the comfy-cli half is the same either way.

   > Installed this server back when it was called `comfy-local-mcp`? Do
   > [Upgrading from `comfy-local-mcp`](#upgrading-from-comfy-local-mcp) first — installing
   > `comfy-mcp` alone will **not** clean up after the old name.

2. **Launch ComfyUI** and leave it running:

   ```bash
   comfy launch
   ```

3. **Add the server to your client** using the snippet for your client in
   [Configure your AI client](#configure-your-ai-client), then restart / reload it so the tools
   appear.

4. **Ask your agent to run a workflow.** For example:

   > "Confirm my local ComfyUI is running, then run the workflow at
   > `~/workflows/txt2img.json` and show me the image."

   Under the hood the agent calls `server_info` to confirm ComfyUI is up, `run_workflow` to
   execute your workflow JSON (API-format or a UI export), and `fetch_outputs` to collect the
   result. No hand-authored workflow? Ask it to start from a template instead — it can
   `search_templates`, `fetch_template` to write a runnable JSON, and run that — and
   `fetch_template` tells it up front if [your install can't run that
   template](#templates-your-install-cant-run) yet.

**Where the images land.** ComfyUI writes generated files into your ComfyUI **workspace's
`output/` directory** (part of the workspace `comfy install` created). Over stdio,
`fetch_outputs(prompt_id, out_dir)` **copies** a finished job's outputs into any directory visible
to that same machine. Over Streamable HTTP, the MCP server cannot write into the remote client's
filesystem: it returns a temporary signed download URL and ready `curl` / `curl.exe` commands for
each output instead. The originals stay in the ComfyUI workspace in both cases.

## Upgrading from `comfy-local-mcp`

This server used to be called **`comfy-local-mcp`**. It was never published to PyPI under that
name, so this only affects you if you installed it from a source checkout — but for those installs
the rename is **not** something `pip install .` finishes on its own, because `comfy-mcp` is a
*different distribution*, not a new version of the old one. Four things moved:

| Was | Is now |
| --- | --- |
| distribution / import package `comfy-local-mcp` / `comfy_local_mcp` | `comfy-mcp` / `comfy_mcp` |
| console script `comfy-local-mcp` (the `"command"` in your client config) | `comfy-mcp` |
| env var `COMFY_LOCAL_MCP_DEBUG_LOG` | `COMFY_MCP_DEBUG_LOG` |
| failure-log directory leaf `comfy-local-mcp/` | `comfy-mcp/` |

1. **Uninstall the old distribution first.** Installing the new one leaves the old one in place,
   and its `comfy-local-mcp` script stays on your `PATH` pointing at a package that no longer
   exists — so an "upgraded" environment either keeps running the old code or fails with
   `ModuleNotFoundError`:

   ```bash
   pip uninstall comfy-local-mcp   # then: pip install comfy-mcp   (or `pip install -e .`)
   ```

2. **Change `"command"` to `comfy-mcp`** in every MCP client config that starts this server
   (`.mcp.json`, `claude_desktop_config.json`, `~/.cursor/mcp.json` — see
   [Configure your AI client](#configure-your-ai-client)), then restart the client. The old
   command name is gone; nothing aliases it.

3. **Rename the failure-log env var if you set it.** `COMFY_LOCAL_MCP_DEBUG_LOG` is no longer
   read, and an env block that still sets it logs **nothing** — a disabled log and a stale
   variable look identical from the outside. Use `COMFY_MCP_DEBUG_LOG`; see
   [Failure log (opt-in)](#failure-log-opt-in).

4. **Move an existing failure log if you're mid-investigation.** The default path's directory leaf
   changed with the package, so a fresh run starts an empty `failures.jsonl` rather than appending
   to the trail you were collecting. Nothing reads the old directory any more — copy it across, or
   delete it:

   ```bash
   # macOS; ~/AppData/Local on Windows, ~/.config on Linux
   cd ~/Library/Application\ Support
   mkdir -p comfy-mcp
   mv comfy-local-mcp/failures.jsonl* comfy-mcp/ && rmdir comfy-local-mcp
   ```

   The glob carries the two rotations (`failures.jsonl.1`, `failures.jsonl.2`) along with the
   live file, and `mkdir -p` first means this is also safe once the new directory exists.

## Remote Streamable HTTP

Run a long-lived MCP server independently of any client's stdin lifecycle:

```bash
COMFY_BIN=/path/to/venv/bin/comfy \
COMFY_API_KEY=YOUR_COMFY_API_KEY \
COMFY_LOCAL_URL=http://127.0.0.1:8188 \
comfy-mcp serve \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 9000 \
  --log-level INFO
```

The endpoint is `http://127.0.0.1:9000/mcp`. The application is pinned to
FastMCP 4.0.0b3 and MCP SDK 2.0.0. One minimal builder owns its name, version,
and base instructions and creates **one FastMCP application**. Every tool is
registered on that instance once. The process then selects one adapter:
`mcp.run(transport="stdio")` or `mcp.http_app(...)` handed to uvicorn.

Consequently both transports expose the same 39 tools, parameters, results,
confirmations, machine snapshot, and `ComfyCliClient` path. HTTP is not a
second product API and has no DTO/VO translation layer. Filesystem arguments
still resolve on the machine running `comfy-mcp`, except for the explicit
command-based upload and signed-output boundary below.

### Uploads and downloads over HTTP

`upload_file` uses the same command-oriented schema as ComfyCloud. Pass one absolute path on the
**MCP client** and the client's operating system:

```json
{
  "file_path": "/Users/me/task_04.webp",
  "client_os": "darwin"
}
```

Over stdio the server can read that path and immediately runs `comfy upload`. Over HTTP it does not
pretend the path exists on the server and does not embed base64 into JSON-RPC. It returns a
credential-free `curl`/`curl.exe` PUT command with a single-use URL on the same MCP listener. Run
the command verbatim; success prints `{"name":"...","subfolder":"","type":"input"}`. The route
accepts JPG/JPEG/PNG/WebP/GIF up to 50 MiB, expires after five minutes, is consumed by the first
attempt, writes owner-only scratch, calls the same `ComfyCliClient → comfy upload` path, and removes
scratch on every exit.

For `fetch_outputs`, HTTP mode treats `out_dir` as the desired path on the **client**. The server
first runs `comfy download` into an owner-only temporary directory, then returns each file with a
**temporary signed download URL (valid for five minutes)** on the same MCP listener port and one ready
command selected by `client_os`. The URL is a bearer capability: do not share or log it. It expires
automatically, cannot be edited, and the temporary bytes are removed with it. A reverse proxy must
forward `/mcp`, `/api/uploads/`, and `/downloads/`, plus the client-facing scheme, Host, and port via
`Forwarded` or `X-Forwarded-Proto` / `X-Forwarded-Host` / `X-Forwarded-Port`. Transfer commands and
signed links use that upstream origin, including a non-default external port; no second file server
or port is started. If the proxy does not preserve those fields, set the complete origin explicitly,
for example `COMFY_MCP_PUBLIC_URL=https://mcp.example.test:8443` (no path or credentials). This follows the
[Comfy MCP upload/download contract](https://docs.comfy.org/agent-tools/mcp#uploads-and-downloads):
the server returns a download link because it cannot write directly into a remote client's
filesystem.

The Python module boundary follows the same rule. `comfy_mcp.server` is a
small public package that exports only `mcp`, `main`, and the 39 registered tool
callables. Guarded runners, locks, parsers, consent state, and startup helpers
remain private in `comfy_mcp.server._internal`; importing the public package
does not make those implementation details part of the supported API.

FastMCP owns JSON-RPC, modern/legacy protocol negotiation, validation,
Streamable HTTP, and the ASGI lifespan. Stateless JSON responses prevent a
stale session ID from selecting half-initialized server state. This project
does not implement parallel JSON-RPC, legacy SSE, reconnect logic, or a private
session patch. Application/server logs go to stderr; child stdout is parsed
only as comfy-cli `envelope/1` data.

A minimal FastMCP 4 client (auto-negotiates the modern protocol and falls back
to a legacy handshake when required):

```python
import asyncio

from fastmcp import Client


async def main() -> None:
    async with Client("http://127.0.0.1:9000/mcp") as client:
        tools = await client.list_tools()
        print([tool.name for tool in tools])

        info = await client.call_tool("server_info")
        print(info.data)


asyncio.run(main())
```

The listener address is separate from the ComfyUI target:

| Concern | CLI / environment | Example |
| --- | --- | --- |
| MCP listener | `--host` / `COMFY_MCP_HOST`, `--port` / `COMFY_MCP_PORT`, `--path` / `COMFY_MCP_PATH` | `127.0.0.1:9000/mcp` |
| Same-machine ComfyUI | `COMFY_LOCAL_URL` (optional at the default address) | `http://127.0.0.1:8188` |
| Different-machine ComfyUI | `COMFYUI_URL`, or `COMFYUI_HOST` + `COMFYUI_PORT` | `http://gpu-box:8188` |

`COMFY_MCP_TRANSPORT`, `COMFY_MCP_LOG_LEVEL`, and comma-separated
`COMFY_MCP_ALLOWED_HOSTS` provide the other `serve` defaults. CLI flags override
their matching environment values.

The default loopback bind is intentional. To bind a non-loopback interface you
must explicitly allow the Host header patterns accepted by the SDK's
DNS-rebinding protection, for example:

```bash
comfy-mcp serve \
  --host 0.0.0.0 \
  --port 9000 \
  --allowed-host 'mcp.example.test:*'
```

That check is not authentication. This release does not add an account system,
OAuth implementation, or TLS termination; for access beyond a trusted private
network, put the listener behind an authenticated TLS reverse proxy and allow
only that proxy's public Host value. HTTP exposes the complete tool surface,
including lifecycle, installation, and update tools; the same MCP elicitation,
comfy-cli consent, argument guards, and local-state protections apply in both
transports, but they are not a substitute for authenticating an untrusted
network deployment.

## Configure your AI client

Choose one connection mode, then use the matching client configuration below:

| Mode | Who starts `comfy-mcp` | Client points to | Tool surface |
| --- | --- | --- | --- |
| stdio | The AI client, as a subprocess | the `comfy-mcp` command | all 39 tools |
| Streamable HTTP | You or a service manager, before the client connects | `http://127.0.0.1:9000/mcp` by default | the same 39 tools |

For HTTP, first start the independently managed listener as described in
[Remote Streamable HTTP](#remote-streamable-http), then register its URL with the client. Do not
register both examples under different names unless you intentionally want two separately running
server processes that expose the same application contract.

HTTP mode still needs the same comfy-cli environment. It creates a **second listener**, not a
replacement for ComfyUI: ComfyUI continues on its own port (normally `8188`), while MCP listens on
another port (here `9000`). Start ComfyUI first, then start the MCP listener:

```bash
# COMFY_API_KEY is optional unless you use partner-API nodes.
# COMFY_LOCAL_URL is optional when ComfyUI already uses 127.0.0.1:8188.
COMFY_BIN=/path/to/venv/bin/comfy \
COMFY_API_KEY=YOUR_COMFY_API_KEY \
COMFY_LOCAL_URL=http://127.0.0.1:8188 \
comfy-mcp serve --host 127.0.0.1 --port 9000
```

The AI client connects to `http://127.0.0.1:9000/mcp`; `comfy-mcp` reaches ComfyUI through
comfy-cli at `http://127.0.0.1:8188`. When a service manager starts `comfy-mcp serve`, put these
environment variables in that service definition.

> The **server key** (`comfy-mcp` in every snippet below) is just the label your client files
> these tools under — it is yours to choose, and the `"command"` (`comfy-mcp`) is the only part
> that has to match the installed console script. Earlier versions of this README used
> `comfy-local`, so **if your config already has a `comfy-local` entry, edit it rather than
> pasting a second one** — two keys pointing at the same command register the server twice and
> your client shows every tool twice. Keeping the old key is equally fine; nothing reads it.

> `COMFY_BIN` is needed whenever `comfy` is not already on the server process's `PATH`: in stdio
> that process is launched by the AI client, while in HTTP it is the independently started
> `comfy-mcp serve`. `COMFY_API_KEY` is optional — keep it only if you use [partner-API
> nodes](#partner-api-nodes) (Seedream / Veo / Kling / Gemini / …). In HTTP mode these variables
> belong to the `comfy-mcp serve` process, not to the client's URL-only configuration.

> **On macOS, keep ComfyUI out of `~/Documents`, `~/Desktop` and `~/Downloads`** — or grant your
> client Full Disk Access. macOS blocks apps (and everything they launch) from reading those
> folders, so an install there fails with `Operation not permitted` before anything runs. See
> [Troubleshooting](#troubleshooting).

### Claude Code

Claude Code supports both transports. For HTTP, start `comfy-mcp serve` first, then register its
URL:

```bash
claude mcp add --transport http comfy-mcp http://127.0.0.1:9000/mcp
```

To check that HTTP registration into the project, add `--scope project` before the server name:

```bash
claude mcp add --transport http --scope project \
  comfy-mcp http://127.0.0.1:9000/mcp
```

The resulting `.mcp.json` may also be written directly. Environment expansion lets each developer
override the listener URL without editing the checked-in file:

```json
{
  "mcpServers": {
    "comfy-mcp": {
      "type": "http",
      "url": "${COMFY_MCP_URL:-http://127.0.0.1:9000/mcp}"
    }
  }
}
```

For client-launched local stdio, use:

```bash
# COMFY_API_KEY is optional — add it only if you use partner-API nodes
# (see the Partner-API nodes section).
claude mcp add --transport stdio comfy-mcp \
  --env COMFY_BIN=/path/to/venv/bin/comfy \
  --env COMFY_API_KEY=YOUR_COMFY_API_KEY \
  -- comfy-mcp
```

Keep the server name before the first `--env`: Claude Code's `--env <env...>` option accepts
multiple values, so putting it first can consume the name and produce a misleading
`missing required argument 'commandOrUrl'` error.

Or check the stdio form into a project with this `.mcp.json`:

```json
{
  "mcpServers": {
    "comfy-mcp": {
      "command": "comfy-mcp",
      "env": {
        "COMFY_BIN": "/path/to/venv/bin/comfy",
        "COMFY_API_KEY": "<your-comfy-api-key>"
      }
    }
  }
}
```

### Claude Desktop

For a remotely reachable HTTP deployment, open **Settings → Connectors → Add custom connector**
and enter its HTTPS `/mcp` URL. Claude Desktop does not load remote servers from
`claude_desktop_config.json`; that file is only for locally spawned integrations. A loopback
`http://127.0.0.1:9000/mcp` listener is not a remotely reachable connector, so use stdio for a
same-Mac ComfyUI unless you have deliberately published the HTTP service behind an authenticated
TLS reverse proxy.

For local stdio, edit `claude_desktop_config.json` (Settings → Developer → Edit Config; on macOS
it lives at `~/Library/Application Support/Claude/claude_desktop_config.json`) and add the server,
then restart Claude Desktop:

```json
{
  "mcpServers": {
    "comfy-mcp": {
      "command": "comfy-mcp",
      "env": {
        "COMFY_BIN": "/path/to/venv/bin/comfy",
        "COMFY_API_KEY": "<your-comfy-api-key>"
      }
    }
  }
}
```

### Cursor

Cursor accepts a Streamable HTTP URL directly. After starting `comfy-mcp serve`, add this to
`~/.cursor/mcp.json` (global) or `.cursor/mcp.json` in a project:

```json
{
  "mcpServers": {
    "comfy-mcp": {
      "url": "http://127.0.0.1:9000/mcp"
    }
  }
}
```

For client-launched local stdio, use the command form instead:

```json
{
  "mcpServers": {
    "comfy-mcp": {
      "command": "comfy-mcp",
      "env": {
        "COMFY_BIN": "/path/to/venv/bin/comfy",
        "COMFY_API_KEY": "<your-comfy-api-key>"
      }
    }
  }
}
```

## Comfy Cloud MCP

Everything above sets up **this** server, which runs on your machine. Comfy also runs a **hosted**
MCP server — **Comfy Cloud MCP** — and it is a good fit when the machine can't carry local
diffusion, or when you'd rather not install ComfyUI at all. It lives at:

```
https://cloud.comfy.org/mcp
```

Your client connects to that URL over **remote HTTP** (no subprocess, nothing to `pip install`) and
workflows execute on **Comfy Cloud GPUs**. You need a [Comfy Cloud](https://cloud.comfy.org)
account — sign up first if you don't have one, since the sign-in below uses it.

**Two ways to authenticate.** **OAuth** is the default: your client opens a browser, you pick a
workspace, and tokens refresh themselves. For clients that don't speak MCP OAuth (Cursor today) and
for headless/CI use, create a **Comfy Cloud API key** at
[platform.comfy.org/profile/api-keys](https://platform.comfy.org/profile/api-keys) — it starts with
`comfyui-` — and pass it as an `X-API-Key` header. Prefer your client's env interpolation
(`${env:COMFY_API_KEY}`) over pasting a key into a file you might commit.

Note that this is a **separate** credential path from the `COMFY_API_KEY` this server's own examples
show: that one is read by comfy-cli on **this** machine for [partner-API
nodes](#partner-api-nodes). The same key works for both, but each server is configured on its own.

### Claude Code (cloud)

Install the `comfy-cloud` plugin — it registers the MCP connection and adds `/comfy-cloud:*` slash
commands in one step:

```
/plugin marketplace add Comfy-Org/comfy-skills
/plugin install comfy-cloud@comfy-skills
```

Then run `/mcp`, select **comfy-cloud** → **Authenticate**, and finish the sign-in in your browser.

Prefer just the connection, without the plugin? Add the server directly (`-s user` makes it
available in every project):

```bash
claude mcp add --transport http comfy-cloud https://cloud.comfy.org/mcp
```

and authenticate the same way, via `/mcp`.

### Claude Desktop (cloud)

Claude Desktop adds it as a **custom connector** through its UI:

1. Sidebar → **Customize** → **Connectors**.
2. Click **+** in the Connectors header → **Add custom connector**.
3. **Name** it (e.g. `Comfy Cloud MCP`), set **Remote MCP server URL** to
   `https://cloud.comfy.org/mcp`, and click **Add**.
4. A browser window opens: choose your workspace and click **Continue** to authorize.

### Cursor (cloud)

Cursor connects to remote MCP servers over HTTP but does **not** support MCP OAuth today, so use an
API key. Add this to `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project), with
`COMFY_API_KEY` set in your shell or system environment:

```json
{
  "mcpServers": {
    "comfy-cloud": {
      "url": "https://cloud.comfy.org/mcp",
      "headers": {
        "X-API-Key": "${env:COMFY_API_KEY}"
      }
    }
  }
}
```

### Other clients (cloud)

Any client with a **remote HTTP** MCP transport can connect to the same URL. Most use a JSON config
with a `url` field (Windsurf uses `serverUrl` instead):

```json
{
  "mcpServers": {
    "comfy-cloud": {
      "url": "https://cloud.comfy.org/mcp"
    }
  }
}
```

Sign in through the browser if the client supports MCP OAuth; otherwise add the `X-API-Key` header
shown above. Restart the client and you should see the cloud tools (`search_templates`,
`submit_workflow`, `get_output`, …) registered under the **comfy-cloud** server.

**Codex** and **OpenClaw** have first-class setup steps — including `codex mcp add comfy-cloud --url
https://cloud.comfy.org/mcp` and `openclaw mcp set` / `openclaw mcp login` — in the [Comfy Cloud MCP
docs](https://docs.comfy.org/agent-tools/mcp), which is also where the screenshot walkthroughs, the
full cloud tool list, and the slash-command/prompt tables live.

## Table of contents

- [Quickstart](#quickstart)
- [Upgrading from `comfy-local-mcp`](#upgrading-from-comfy-local-mcp)
- [Remote Streamable HTTP](#remote-streamable-http)
- [Configure your AI client](#configure-your-ai-client)
- [Comfy Cloud MCP](#comfy-cloud-mcp)
- [Prerequisites](#prerequisites)
- [When to use this server](#when-to-use-this-server)
- [Using with local LLMs (VRAM coordination)](#using-with-local-llms-vram-coordination)
- [Partner-API nodes](#partner-api-nodes)
- [Spending credits on partner models](#spending-credits-on-partner-models)
- [Templates your install can't run](#templates-your-install-cant-run)
- [Driving a remote ComfyUI](#driving-a-remote-comfyui)
- [Targeting a non-default ComfyUI address](#targeting-a-non-default-comfyui-address)
- [Which address variable do I want?](#which-address-variable-do-i-want)
- [Project anchoring](#project-anchoring)
- [Tools](#tools)
- [Troubleshooting](#troubleshooting)
- [Failure log (opt-in)](#failure-log-opt-in)
- [Smoke test](#smoke-test)
- [Contributing](#contributing)
- [License](#license)
- [Trademarks](#trademarks)

## Prerequisites

- **Python ≥ 3.10.**
- **comfy-cli ≥ 1.14.0** on your `PATH`: `pip install "comfy-cli>=1.14.0"`. This is the engine
  every tool wraps; the server refuses to run against an older comfy-cli with an upgrade message.
  1.14.0 is the first release carrying everything this server needs. 1.13.0 brought the basics —
  the `comfy logs` verb, the `envelope/1` contract, the `comfy outdated` verb behind
  `server_info`'s `freshness` block, and the machine-readable `login_url` event `auth_login`
  waits for — but a large slice of the tool surface calls verbs that only exist from 1.14.0 on:
  `comfy node deps` (`node_dependencies`), `system-stats` / `free`, `workflow notes`
  (`list_workflow_notes`), `logs --port`, the background download group (`--background` plus
  `download-status` / `downloads` / `download-cancel`), `models search`'s cross-folder walk, the
  templates gallery cache TTL, and `comfy run`'s `--allow-spend` interlock. On 1.13.0 enough of
  the surface is inert that the server reads as broken rather than as out-of-date, which is why
  the floor moved rather than each tool degrading. **Installing this server does not install
  comfy-cli** — it is deliberately *not* a declared dependency of the `comfy-mcp` package, because
  the binary that matters is whichever one `PATH` / `COMFY_BIN` resolves to rather than one pinned
  into this server's environment, so the floor above is enforced at runtime against that binary
  (the rationale lives in `pyproject.toml`). Install the two together:
  `pip install comfy-mcp "comfy-cli>=1.14.0"`.
- **Every capability degrade is still in place**, because the floor and a degrade guard different
  failures: the floor catches a *wrong comfy-cli version*, a degrade catches a *correct version in
  a broken environment*. The version guard fails **open** on a `--version` it can't parse (a
  source build, a fork), so such a build can still reach the tools from below the floor — and a
  dependency outside comfy-cli (a ComfyUI-Manager too old to know a flag comfy-cli forwarded to
  it) can fail on an otherwise-compliant install. In those cases you get a named capability gap —
  `{"error": "…", "unsupported": true}`, e.g.
  `freshness: {"error": "freshness unavailable: …", "unsupported": true}` — rather than a raw
  Click usage dump, and the rest of the tool keeps working.
- **A ComfyUI workspace.** If you don't have one, `comfy-cli` can create it: `comfy install`
  sets up a ComfyUI workspace it will point at. (An existing ComfyUI checkout works too — see
  `comfy set-default <path>`.)
- **A running ComfyUI.** ComfyUI must be **started before you use the tools** — launch it with
  `comfy launch` (or, from an agent, the `launch_comfyui` tool), and confirm it is up with
  `server_info`. Nothing here starts ComfyUI implicitly.

<details>
<summary><strong>Optional environment variables</strong> (<code>COMFY_BIN</code>, <code>COMFY_API_KEY</code>, <code>COMFYUI_URL</code>, <code>COMFY_MCP_REMOTE_SHARED_MODELS</code>, <code>COMFY_LOCAL_URL</code>, <code>COMFY_T2I_TEMPLATE</code>; plus <code>COMFY_USER_AGENT</code>, which the server sets itself)</summary>

<br>

- **`COMFY_BIN` override (optional).** By default the server calls `comfy` from `PATH`. MCP
  clients launch the server with their own environment, which often does **not** include your
  shell's `PATH` — so if `comfy` lives in a virtualenv or a non-standard location, set
  `COMFY_BIN` to its absolute path (e.g. `/path/to/venv/bin/comfy`). Every example in
  [Configure your AI client](#configure-your-ai-client) shows where it goes. Setting it is
  sufficient on its own — you do **not** also have to put
  that directory on the client's `PATH`. The server prepends the resolved binary's directory to
  the `PATH` it hands comfy-cli, because some comfy-cli commands (notably the background
  `launch`) re-invoke `comfy` by name and have to be able to find themselves.
- **`COMFY_API_KEY` (optional — needed only for partner-API nodes).** Workflows that use
  partner-API nodes (Seedream / Seedance / Nano Banana / Gemini / Veo / Kling / …) need a Comfy
  credential, and — exactly like `COMFY_BIN` — an MCP client launches the server with its own
  minimal environment, so a key from your shell won't reach it. Set `COMFY_API_KEY` in the
  client registration `env` block. See **[Partner-API nodes](#partner-api-nodes)** below for the
  full precedence chain; every example in
  [Configure your AI client](#configure-your-ai-client) shows where it goes.
- **`COMFYUI_URL` / `COMFYUI_HOST` / `COMFYUI_PORT` (optional — drive a ComfyUI on *another
  machine*).** Read by **this server**. By default every tool targets `127.0.0.1:8188`. Set
  `COMFYUI_URL` (e.g. `http://gpu-box:8188`) — or the `COMFYUI_HOST` (+ optional `COMFYUI_PORT`,
  default `8188`) pair — to point the **submit / job** tools at a ComfyUI running elsewhere, e.g. a
  GPU box reachable over a private network (Tailscale). See **[Driving a remote
  ComfyUI](#driving-a-remote-comfyui)** for what is and isn't remoted. Unset ⇒ nothing changes.
- **`COMFY_MCP_REMOTE_SHARED_MODELS` (optional — only meaningful alongside the variables above).**
  `download_model` writes to *this* machine's models dir and has no remote mode, so with a remote
  configured it refuses rather than downloading onto the wrong disk. Set this to `1` when this
  machine's workspace models dir **is** the remote's — shared storage, e.g. an NFS / tailnet mount —
  to skip that guard and download as usual. Nothing else reads it, and it does nothing at all when
  no remote is configured.
- **`COMFY_LOCAL_URL` (optional — a ComfyUI on *this machine*, on a non-default port).** Read by
  **comfy-cli**, never by this server — it rides the environment passthrough, so setting it in the
  client `env` block re-points every tool. For a ComfyUI on this machine that isn't on
  `127.0.0.1:8188` (e.g. `:8189` because Docker Desktop's ComfyUI holds `:8188`). See
  **[Targeting a non-default ComfyUI address](#targeting-a-non-default-comfyui-address)**, and
  **[Which address variable do I want?](#which-address-variable-do-i-want)** for the difference
  between the two.
- **`COMFY_USER_AGENT` (set by the server — not yours to configure).** Every comfy-cli call this
  server makes is labelled `comfy-mcp`, which is how comfy-cli tells work that came from this MCP
  apart from a human typing the same command — most usefully on the partner-API calls that spend
  credits. A value you set is **overridden**, on purpose: it would otherwise file this server's
  calls under someone else's name. The label is a caller identity, not content — nothing about
  your prompts, workflows, or outputs travels with it, and comfy-cli's own telemetry stays subject
  to comfy-cli's consent settings (`comfy tracking disable`, or the `DO_NOT_TRACK` /
  `COMFY_NO_TELEMETRY` environment variables, both of which this server passes straight through).
- **`COMFY_T2I_TEMPLATE` / `COMFY_T2I_PROMPT_SLOT` / `COMFY_T2I_CHECKPOINT_SLOT` (optional — retarget
  `generate_image`).** `generate_image(prompt)` runs the gallery's `default` template (ComfyUI's own
  basic SD1.5 text-to-image graph), filling its positive-prompt slot `6.text` and, when you pass
  `checkpoint`, its `ckpt_name` slot. To point that on-ramp at a different local text-to-image graph,
  set all three **together** — the slot keys describe one specific template, so changing the template
  alone leaves the prompt address matching no slot. List a replacement's slots with
  `comfy templates fetch <name> -o wf.json && comfy workflow slots wf.json`. For a one-off run of some
  other template, prefer the `run_template` tool over these.

</details>

## When to use this server

Local diffusion is only a good default on a machine that can actually carry it, so the server's client instructions tell your agent to read `server_info`'s `hardware` block (`os`, `arch`, `ram_bytes`, and a `gpu` object with `vendor` / `model` / `vram_bytes` / `unified_memory`) **before** the first generation and route on it. The agent does not even have to make that call: at startup the server probes `comfy env` once and appends a **`Machine snapshot`** section — the same `hardware` block verbatim, plus the configured remote target if any — to the instructions the MCP handshake carries, so the routing figures are in the agent's context from the first message. The probe is best-effort: if it fails (no `comfy` on PATH yet, a timeout), the section is simply absent and the instructions still say to call `server_info` first; a healthy probe on an older comfy-cli that reports no `hardware` states the figure is unknown, which routes to step 3 below (ask). The snapshot never carries *live* state — whether a server is running stays `server_info`'s job. The thresholds:

| Machine | Guidance |
|---|---|
| Discrete GPU, **≥ 24 GB** VRAM | Local generation is a good default. |
| Discrete GPU, **8 GB to under 24 GB** VRAM | Images are fine (prefer current, smaller models); video will be slow or infeasible. |
| **< 8 GB** VRAM, or the user confirming there is no GPU | Don't run local diffusion. Use partner nodes (plain web calls, fine on any machine) or the [Comfy Cloud MCP](#comfy-cloud-mcp) if your client has it connected. |
| **Apple Silicon**, **≥ 32 GB** unified memory | Images are OK. Video on the Apple GPU is not recommended — time estimates are unreliable and thermals suffer. |
| **Apple Silicon**, under 32 GB unified memory | Same as the no-GPU row above — go partner/cloud rather than local. |

The discrete-GPU rows are written for NVIDIA but apply to an AMD or Intel card on a ROCm/XPU build too — the VRAM number is what matters. The no-local-video rule is an *Apple GPU* rule rather than a Mac rule: an Intel Mac with a discrete card follows the discrete-GPU rows.

The instructions walk these as an ordered procedure, because several of the checks only make sense in sequence:

1. **Is the work even local?** `hardware` describes the machine *this server* runs on, and that is where most tools execute. A `comfy_target` block ([Driving a remote ComfyUI](#driving-a-remote-comfyui)) diverts every tool that submits a job — `run_workflow`, `generate_image`, `run_template` — along with the queue/`jobs` tools, while discovery, templates, downloads, outputs and the lifecycle tools stay here; so against a genuine remote the thresholds below describe the wrong machine. It counts as another machine only when its `host` is neither loopback (anything in `127.0.0.0/8`, `localhost`, IPv6 `::1`) nor this host's own address, and a malformed config produces an error-shaped `{error, note}` block that resolves no remote at all. Nothing the server returns carries the local hostname or interface addresses, so a `host` the agent can't place is a question for you rather than a guess — a hostname or LAN IP can be this same machine, and a loopback host can be a tunnel to a remote GPU. `COMFY_LOCAL_URL` is a second signal worth checking: it repoints comfy-cli without producing a `comfy_target` block.
2. **Get a memory figure.** The sizes are bytes (`ram_bytes`, `gpu.vram_bytes`) and the divisor gives GiB, while drivers report under the advertised size — a consumer 24 GB card reads 23.99, an ECC/reserving datacenter card (A10, L4) about 22.3 — so a *small* shortfall, within ~10% of a nominal size, reads as that nominal capacity. A gap wider than that is not driver overhead and is taken at face value instead: on a MIG/vGPU partition the model string names the whole card while `vram_bytes` is the slice you actually get, and rounding a 6 GB A100 slice up into the ≥ 24 GB band would OOM the run. On Apple Silicon `gpu.vram_bytes` is `null` (with `gpu.unified_memory` true) and the figure is `ram_bytes` — an Apple-only substitution.
3. **If the figure is missing, ask.** A `null` **or zero** `vram_bytes` on any **non-Apple** GPU (a discrete card comfy-cli can't size, but also a non-Apple unified part like a Jetson/Grace board or a Strix Halo APU), a missing `gpu` object, or a missing/zero `ram_bytes` on the Apple path all mean **unknown**, not "no GPU" — the agent asks rather than stranding a machine that has one. The "no GPU" verdict is reserved for a *confirmed* absence, and the only thing that confirms one is your own answer: no `hardware` payload encodes it, because a null or missing `gpu` is unknown by this same step. Nothing in this repo probes hardware, and the instructions tell the agent not to shell out either: a probe runs on a path this server can neither bound nor audit.
4. **Route on the figure**, then **redirect rather than dead-end** when the answer is "not on this machine". A figure that came from your answer rather than the payload routes on whichever row fits the machine — the unified-memory row on an Apple Silicon Mac, the VRAM rows otherwise, which is what covers the non-Apple unified-memory boards that have no row of their own.

"No local video on a Mac" is about the Apple GPU, not about video as such: `API`-tagged video templates (`search_templates(tag="API", type="video")` — both filters, since neither alone isolates partner-run video; each row's `api` boolean then confirms which side of the line it fell on) and `emit_partner_workflow` run the model on partner infrastructure, so they work on any machine. See **[Partner-API nodes](#partner-api-nodes)**.

The `hardware` block comes straight through from `comfy env`, and a comfy-cli that predates it simply omits the key. There is no HTTP client and no cloud code here — the cloud/partner steer is guidance text only.

**Which model to use is deliberately not encoded here.** The instructions tell the agent to pick via `search_templates` / `search_models` rather than assume a classic default (e.g. SDXL), because the gallery tracks current models and a hardcoded name would rot. Current-model guidance lives in **[Comfy-Org/comfy-skills](https://github.com/Comfy-Org/comfy-skills)**, which is its canonical home.

## Using with local LLMs (VRAM coordination)

Running a local LLM (Ollama, LM Studio, llama.cpp) and ComfyUI on the same GPU means the two compete for the same VRAM, and the LLM is usually the one holding it when the image job needs it. This server gives the agent both halves of the read/free loop, but the *coordination* is the client's — see why below.

The recipe, in order:

1. **Read the headroom.** `system_stats()` returns per-device `vram_free` / `vram_total` straight from the live ComfyUI. Compare `vram_free` against what the workflow's checkpoint needs.
2. **If it is tight, the client unloads its own LLM** using its runtime's own mechanism — this server has no way to do it (step 5 below):
   - **Ollama** — send `keep_alive: 0` on the next `/api/generate` (or `/api/chat`) call, which unloads the model as soon as that call returns, or run `ollama stop <model>`.
   - **LM Studio** — let the model's TTL / JIT auto-evict expire, or unload explicitly with `lms unload <model>` (`lms unload --all` for everything).
   - **llama.cpp (`llama-server`)** — in [router mode](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (started with no `-m`, or with `--models-dir`) `POST /models/unload` with `{"model": "<name>"}` unloads one model; `GET /models` lists what is currently loaded. Independently of router mode, `--sleep-idle-seconds N` makes the server unload the model and its KV cache after N idle seconds and reload it automatically on the next request — which handles both step 2 and step 5 with no orchestration at all. Only a classic single-model server started without either (`llama-server -m model.gguf`) has nothing to call: there, stopping and restarting the process is the reclaim.
3. **Free ComfyUI's own models too** with `free_memory()`. ComfyUI applies it when its queue worker next iterates — immediate if idle, after the current job if busy — and it never interrupts a running job. Re-read `system_stats()` to confirm the VRAM actually came back before committing to a big run.
4. **Run the job** — `run_workflow(...)` / `run_template(...)` / `generate_image(...)` — then collect with `fetch_outputs(...)`.
   (This whole recipe is about **this machine's** VRAM: `system_stats` and `free_memory` are never
   remoted, so with a `COMFYUI_URL` configured steps 1–3 measure and free the wrong box while step 4
   submits to the remote. See [Driving a remote ComfyUI](#driving-a-remote-comfyui).)
5. **The client reloads its LLM** afterwards, again through its own runtime. Ollama, LM Studio and a sleep-idle `llama-server` all reload on demand, so for those "reload" is just the next request; a single-model `llama-server` stopped in step 2 has to be started again.

**Why steps 2 and 5 cannot live in this MCP server.** This server is a **stdio subprocess of your MCP client** — it holds no handle on whatever LLM runtime that client is using, is not told which one it is, and has no business reaching into a process it does not own. Reaching one anyway would also breach the [thin-wrapper rule](AGENTS.md): every tool here is a `comfy` passthrough, and there is no `comfy` subcommand for "unload someone else's model". The deeper reason is step 5: the *model* that was unloaded cannot ask for itself back, so something still running has to sequence unload → run → reload. Where the LLM's own runtime can do that (Ollama's on-demand load, LM Studio's JIT, `llama-server --sleep-idle-seconds`) it should — that is the least-coordination option and it needs nothing from this server. Otherwise the client, or the orchestrator driving it, is the only participant present throughout. Either way the split is structural rather than a missing feature: this server owns reading and freeing **ComfyUI's** memory, and the client owns its **own** model's lifecycle.

A note on scope: `free_memory()` asks ComfyUI to release *its* models. It does nothing about VRAM held by an LLM runtime, a browser, or another process — if `system_stats()` still shows little free VRAM after a `free_memory()` call, the memory is probably someone else's and step 2 is what reclaims it.

**Read that signal against the lag, not instantly.** The free applies on the queue worker's next iteration, so on a *busy* server an immediate re-read legitimately shows no change while the VRAM is still ComfyUI's — the request simply has not been serviced yet. Before concluding the memory belongs to another process, either wait for the current job to finish (`job(action="queue")` shows whether one is running) or re-poll `system_stats()` a few times over a few seconds. Only a number that stays flat on an *idle* server means the holder is someone else.

A second caveat: `system_stats()` and `free_memory()` are **not** redirected by `COMFYUI_URL` / `COMFYUI_HOST` — they always describe and act on whichever ComfyUI comfy-cli itself targets, because `comfy system-stats` and `comfy free` take no `--host` / `--port`. With a remote ComfyUI configured, `run_workflow` / `generate_image` / `run_template` submit *there* while these two read and free the *local* install, so this recipe applies to a local-ComfyUI setup. Don't gate a remote run on it. Both payloads say so themselves when a remote is configured: a top-level `comfy_target_note` (`host` / `port` / `source` / `note`) names the target the run tools submit to and states that these numbers describe whichever ComfyUI comfy-cli itself targets, which need not be the same box — the note reports that divergence rather than adjudicating it, since a configured host can perfectly well resolve to this machine (and a loopback one can be a tunnel to a remote GPU). A *malformed* `COMFYUI_URL` / `COMFYUI_HOST` gives an error-shaped note (`error` / `note`) instead of breaking these local-only calls, so an absent key means exactly one thing: nothing is configured, and the payloads are unchanged.

## Partner-API nodes

Some ComfyUI nodes call out to Comfy's partner APIs (Seedream / Seedance / Nano Banana / Gemini /
Veo / Kling / …). Running one **locally** still needs a Comfy credential, and comfy-cli resolves
it in this order (first match wins):

1. a per-call flag (not exposed by this server);
2. a live Comfy Cloud OAuth session (`comfy cloud login`);
3. the **`COMFY_API_KEY`** environment variable;
4. a stored key set with `comfy auth set comfy-cloud-api-key --key <KEY>`.

Option 2 does not have to be typed into a terminal: the agent can call **`auth_login`**, which starts `comfy cloud login` in the background and hands back the OAuth URL for you to open. Complete the sign-in in your browser, then have the agent confirm it with `auth_status`. The sign-in itself is comfy-cli's — this server never sees your tokens, and the browser callback is handled by the CLI's own loopback listener on this machine (so `auth_login` is for a *local* MCP; on a remote/containerised one, sign in where comfy-cli actually runs).

Because an MCP client spawns the server with its own minimal environment (the same reason
`COMFY_BIN` exists), a `COMFY_API_KEY` from your interactive shell is **not** inherited — put it
in the client registration `env` block (shown in every
[client example](#configure-your-ai-client)). If a run fails with
`partner_node_requires_credential`, the error now carries comfy-cli's hint verbatim, including
the `comfy auth set comfy-cloud-api-key --key …` fallback and the list of offending nodes; the
server also retries a transient credential failure briefly before surfacing it.

## Confirmation prompts on clients that can't show them

Several tools ask you to confirm before they act: `install_node`,
`update_comfyui(target="all")`, `switch_comfyui_version`, `restart_comfyui` when it must stop a
server it did not start, and `launch_comfyui` with `--listen` (which exposes an unauthenticated
ComfyUI to your network). Each raises an MCP *elicitation* and **fails closed** if it is not
approved.

Some MCP clients answer that request without ever showing you a prompt. When that happens the
tool refuses and tells you so — nothing is changed, and the error names the equivalent terminal
command you can run instead.

If you would rather pre-authorize specific gates, set `COMFY_MCP_ASSUME_CONSENT` in the server's
environment — the `env` block of your client registration, alongside `COMFY_BIN`:

```jsonc
"env": { "COMFY_MCP_ASSUME_CONSENT": "install_node,update_all" }
```

Accepted names: `install_node`, `update_all`, `version_switch`, `kill_untracked`,
`network_exposure` — or `all` for every one of them. List only what you want; authorizing node
installs should not silently also authorize binding ComfyUI to every network interface.

This is a setting **you** write, in a file the model cannot edit. That is the point: an agent
cannot grant itself permission by passing an argument, which is why no tool parameter does this.

**Spending credits is deliberately excluded.** No value — including `all` — pre-authorizes
`partner_generate`, `run_template` or `run_workflow`. Money keeps a single owner: comfy-cli's own
durable consent (`comfy generate consent always`). See
[Spending credits on partner models](#spending-credits-on-partner-models).

## Spending credits on partner models

`partner_generate` is the one tool whose *whole purpose* is to spend: it wraps `comfy generate
<model>`, which calls a hosted partner API and **spends your Comfy credits**. So every call is
confirmed with you first.

The other tools execute on **your** machine, and on their own they cost nothing — but that is a
property of the tool, not a guarantee about the workflow you hand it. A workflow run through
`run_workflow` can itself contain the partner-API nodes described just above (Seedream, Veo,
Kling, …), or any other node that bills a hosted service, and **those still spend your credits** —
they bill through the workflow, below this server. `run_workflow` therefore carries the same opt-in
`confirm_spend` gate `run_template` does ([below](#workflows-that-spend--run_workflow)).
That gate covers the partner-API nodes comfy-cli recognizes, which is not the same as every node
that can bill something: an arbitrary custom node can still call a paid service of its own, with
nothing to gate it. **Check what a workflow contains before running one you did not build.**
`generate_image` needs no gate because it runs a free OSS template — though note it is
[retargetable via `COMFY_T2I_TEMPLATE`](#prerequisites), and pointed at an `API`-tagged template it
would spend with no prompt.

`emit_partner_workflow` sits on the free side of that line for the same reason: it only *writes* a
graph containing a partner API node, never calls the partner, and so has no confirmation prompt.
The graph it writes is exactly one of the workflows the paragraph above is warning about — running
it with `run_workflow` bills the partner node, so that step needs `confirm_spend=True`.

**On a client that supports [MCP elicitation](https://modelcontextprotocol.io/specification/server/elicitation)**
(Claude Code and Claude Desktop do), the call raises a confirmation prompt naming the model and
saying that it spends credits:

- **Approve** → the server forwards comfy-cli's `--yes` and the generation runs.
- **Decline** (or dismiss it) → the tool returns an error, and comfy-cli is never started. No
  credits are spent.
- **Leave it unanswered** → after five minutes the prompt lapses into a refusal, so a forgotten
  call never sits pending forever. Nothing is spent; call the tool again to get a fresh prompt.

**Don't want to be asked every time?** Persist it in comfy-cli, not here:

```bash
comfy generate consent always   # spend without prompting
comfy generate consent show     # what is it set to?
comfy generate consent ask      # back to confirming each call
```

The server reads that setting per call and skips its own prompt when it is on — the durable
"always proceed" lives in comfy-cli's config, and this server keeps no spend state of its own.

**On a client that cannot elicit**, there is no prompt to raise, so consent has to be explicit in
the call: `confirm_spend=True` forwards `--yes`. Without it comfy-cli's gate fails closed (an MCP
server has no terminal to prompt at) and the call errors having spent nothing. On a client that
*can* elicit you are asked anyway — `confirm_spend=True` is not a way around the prompt.

Two things the server deliberately will **not** do:

- **Treat tool permission as spend consent.** Your agent host's "always allow this tool" toggle
  authorizes *calling* `partner_generate`; it never authorizes spending your money, and is never
  read as consent. Only the prompt you answered, or the comfy-cli setting you persisted, is.
- **Run against a comfy-cli with no spend gate.** The fail-closed guarantee is the engine's, so if
  `comfy generate consent` is missing the tool refuses up front rather than spending on the
  assumption something would have stopped it. `pip install -U comfy-cli` to fix.

### Templates that spend — `run_template`

`run_template` is the other tool that *can* spend, and it is confirmed the same way, with the
differences the verb forces. Most gallery templates are free OSS graphs that run on your machine;
some embed partner-API nodes and bill through them.

- **`confirm_spend=False` (the default) never prompts.** Nothing is forwarded, so comfy-cli's gate
  fails closed on a paid template — there is nothing to consent to. A free template just runs. This
  is deliberate: prompting on every template run would train you to click through the one prompt
  that matters.
- **`confirm_spend=True` asks you first**, naming the template, on any client that can elicit.
  Approve → `--allow-spend` is forwarded. Decline → the tool errors and comfy-cli is never started.
  As with `partner_generate`, an agent setting the argument for itself is *not* your consent; on a
  client that cannot elicit it stands alone as the fallback.
- **`comfy generate consent always` does not apply here.** That setting is scoped to
  `comfy generate` — `comfy run-template` never reads it — so it grants nothing for templates and
  the prompt is raised regardless.

Unlike `partner_generate`, there is no up-front gate probe: `run-template` carries its spend gate
inside the verb itself, so a comfy-cli that has the verb has the gate.

### Workflows that spend — `run_workflow`

`run_workflow` takes the **same** `confirm_spend` argument, with the same three rules as
`run_template` above — default never prompts and forwards nothing, `confirm_spend=True` asks you
per call on a client that can elicit, and `comfy generate consent always` grants nothing here
either. Most workflows are ordinary local graphs and are unaffected; the ones this matters for are
the graph `emit_partner_workflow` writes and an `API`-tagged gallery template you fetched with
`fetch_template`. When consent is withheld the engine refuses with `spend_consent_required` and
names the offending `partner_nodes`, so you learn *which* nodes cost money without a second call.
Consent is resolved once per call, so the server's brief credential retry never re-asks you.

One caveat specific to this verb, and the reason it is called out rather than folded into the
section above: `comfy run` **long predates** its spend gate, so unlike `run-template` the verb's
presence proves nothing. The gate shipped in comfy-cli 1.14.0, which is the floor this server
enforces, so on every *published* comfy-cli it accepts the interlock is there and
`confirm_spend=False` is a guarantee rather than a default. The probe stays because the floor
can't prove it: the version guard fails **open**, so a source build or fork whose `--version`
can't be read reaches the tool without the flag. The server probes `comfy run --help` on the
calls you approved and simply omits `--allow-spend` when that comfy-cli has no such flag, so an
approved run still runs instead of dying on a usage error — but on such a build what authorizes
the spend is your answer to the prompt, not an engine gate, and a paid workflow runs and spends
whether or not you pass `confirm_spend`. `pip install -U comfy-cli` closes that residual case.

## Templates your install can't run

The template gallery is served fresh from `Comfy-Org/workflow_templates`, while your ComfyUI is
whatever version you installed. So the catalog can legitimately offer a template your install
cannot run yet — it references a node class you don't have, or a *model option inside* a node you
do have (a partner model key added in a later release is the common one). Discovery succeeds, the
run fails, and you get to work out why.

`get_template` and `fetch_template` cross-check the template against your install and report it
as a `local_check` block. Under the hood it is `comfy validate` — the same engine
[`validate_workflow`](#workflow-building) uses, reading the **live `object_info`** of your running
ComfyUI, so it sees your custom nodes and your model options, not a bundled catalog.

| `local_check` | Means |
|---|---|
| `{"checked": true, "runnable": true, …}` | Every node class and input option the template uses exists in your install. Necessary, not sufficient — `validate_workflow`'s documented blind spots still apply. |
| `{"checked": true, "runnable": false, "errors": [...], …}` | Running it will fail as-is: the `errors` name what is missing (and, where comfy-cli can, what your install offers instead). Update ComfyUI and its custom nodes, or pick another template. |
| `{"checked": false, "reason": …, …}` | The comparison could **not** be made — almost always because ComfyUI isn't running, so there is no live catalog to compare against. This is not a verdict about the template. |

The check is advisory and fails open: the workflow file is written either way, `path` always comes
back, and nothing is ever refused on its account. Pass `check_local=False` to skip it.

## Driving a remote ComfyUI

By default the server drives ComfyUI on the local `127.0.0.1:8188`. Point it at a ComfyUI running
**elsewhere** — e.g. a GPU box reachable over a private network (Tailscale) — by setting one of:

- **`COMFYUI_URL`** — a full URL, e.g. `http://gpu-box:8188` (host-only is fine; port defaults to
  `8188`). Takes precedence over the pair below. Only the **host and port** are forwarded to
  comfy-cli, so the URL must be plain `http://` with **no base path** and **no query or fragment**:
  an `https://` scheme, a reverse-proxy path (`http://gpu-box:8188/comfyui`), or a query / fragment /
  `;params` (`http://gpu-box:8188/?token=…`) is rejected rather than silently downgraded to http /
  dropped. That last one is the shape an auth-proxied ComfyUI is usually written as, and comfy-cli
  has nowhere to put it — a dropped token would submit every run unauthenticated and fail later as a
  401/403 naming nothing — so it is refused up front. Front a TLS/base-path/auth proxy locally if you
  need one, and point `COMFYUI_URL` at that.
- **`COMFYUI_HOST`** (+ optional **`COMFYUI_PORT`**, default `8188`) — e.g. `COMFYUI_HOST=gpu-box`.
  A port without a host (setting only `COMFYUI_PORT`) is rejected — set the host too.

Set it in the client registration `env` block (same place as `COMFY_BIN`). With nothing set,
behavior is unchanged (`127.0.0.1:8188` on this machine). If what you actually have is a ComfyUI on
*this* machine on a different port, you want `COMFY_LOCAL_URL` instead — see [Which address variable
do I want?](#which-address-variable-do-i-want).

When configured, the server forwards `--host` / `--port` to comfy-cli for exactly the verbs that
accept them — `comfy run`, `comfy run-template`, `comfy jobs …` and `comfy upload` — so every tool
that **submits a job, reads one back, or stages the files a job will read** targets the remote:
`run_workflow`, `generate_image`, `run_template`, `job` (every action), `upload_file`. `server_info`
reports the configured target under a `comfy_target` block.

That set is deliberately closed under submit-then-poll: a `prompt_id` only means something to the
server that issued it, so a tool that submits and a tool that polls must never resolve to different
machines. `upload_file` is in it for the same reason one step earlier: an input file is only useful
on the machine that runs the workflow reading it, so staging it here while submitting there would
fail the run on a filename the remote cannot see. `upload_file(file_path, client_os)` names one
absolute client path. stdio passes it straight through; HTTP returns a single-use PUT command, then
stages the received bytes briefly on the MCP server and follows the identical `comfy upload` path.
Remote upload needs comfy-cli **≥ 1.14.0**
(this server's floor); an older one rejects the forwarded `--host` and `upload_file` raises with the
upgrade step rather than silently staging into the local `input` dir.

**Not remoted (this repo is a thin wrapper and never opens its own socket):**

- **Lifecycle** (`launch_comfyui`, `stop_comfyui`, `restart_comfyui`, `update_comfyui`,
  `switch_comfyui_version`, `install_node`, `get_logs`) — these manage a **local** ComfyUI
  process/install and stay local-only; they cannot start/stop, update, version-switch, install node
  packs into, or read logs from a remote box. Start and update ComfyUI, and install its node packs,
  on the remote host yourself. `install_node` in particular writes into *this* machine's ComfyUI
  workspace and venv, so with `COMFYUI_URL` set the pack lands where the run isn't.
- **Catalog / partner verbs** — `search_templates` / `search_models` / `download_model` /
  `partner_generate` — this server forwards **no** `--host`/`--port` to these verbs (they accept
  none at all), so they run against comfy-cli's local default. A model must be installed on the
  machine that actually runs the job, and `download_model` cannot do that for you — so rather than
  writing the checkpoint to the wrong disk and letting the run fail later on a missing model, it
  **refuses** while `COMFYUI_URL`/`COMFYUI_HOST` is set, naming the remote it would have missed.
  Install the model on the remote host itself (its own comfy-cli or MCP server). The exception is
  **shared storage** — an NFS / tailnet mount where this machine's workspace models dir *is* the
  remote's — which no environment check can distinguish, and which this server may not probe the
  remote to confirm: assert it with **`COMFY_MCP_REMOTE_SHARED_MODELS=1`** and the download runs as
  it always did. `download` (every action) is never guarded —
  they manage downloads already submitted here.
- **Output download** (`fetch_outputs` → `comfy download`) takes no `--host`/`--port` either, but it
  still retrieves a **remote** job's files, because it never asks a server which job that is: the
  same comfy-cli run that submitted the job wrote a state file **on this machine** keyed by
  `prompt_id`, and for a non-loopback target that file records each output as an absolute
  `http://<remote>:<port>/view?…` URL, which `comfy download` then streams from the remote. It falls
  back to querying the local default server only when no such state file exists (an id this machine
  never submitted). Over stdio it copies directly into `out_dir`; over HTTP it downloads into a
  temporary server directory and returns five-minute signed URLs on the MCP listener, because
  `out_dir` belongs to the client machine. `run_workflow(wait=True)` / `job(action="status")`
  return the engine's source URLs, but `fetch_outputs` is the client-delivery step.
- **Discovery / validation** (`nodes`, `validate_workflow`, and the
  `local_check` block on `fetch_template` / `get_template`) — their comfy-cli verbs *do* accept
  `--host`/`--port`, but this version forwards only to the submit/poll tools, so they still
  describe the **local** install. Remoting them is a planned follow-up; until then a workflow or
  template can pass a local check and still fail on a remote whose node set differs, so
  author/validate against a local ComfyUI matching the remote's.
- The remote ComfyUI must be reachable and **unauthenticated** on that network (the private network
  is the boundary); the server does not authenticate to it. `server_info` does not live-probe the
  remote — reachability surfaces on the first run/job call.

## Targeting a non-default ComfyUI address

The section above drives a ComfyUI on **another machine**. This one is for a ComfyUI on **this**
machine that simply isn't on the default `127.0.0.1:8188` — most often a port clash, e.g. Docker
Desktop's ComfyUI already holds `:8188` so yours came up on `:8189`.

That address is resolved by **comfy-cli**, not by this server. Every tool shells out to `comfy`
with the server's full environment, so a `COMFY_LOCAL_URL` set in your MCP client's `env` block
reaches comfy-cli and re-points *every* local-targeting verb. Nothing to change here — set it
alongside `COMFY_BIN` in the client registration:

```json
{
  "mcpServers": {
    "comfy-mcp": {
      "command": "comfy-mcp",
      "env": {
        "COMFY_BIN": "/path/to/venv/bin/comfy",
        "COMFY_LOCAL_URL": "http://127.0.0.1:8189"
      }
    }
  }
}
```

**Accepted values.** `http://host:port`, `host:port`, or `http://host` (port defaults to `8188`;
the scheme is optional and, if present, must be `http`). IPv6 literals are bracketed:
`http://[::1]:8189`. A malformed value is ignored with a one-line stderr warning rather than
breaking the call.

**Verify it took effect — call `server_info` first.** `server_info` wraps `comfy env`, which
resolves the local address by the same rules, so the server URL it reports *is* the resolved
address. Seeing `:8189` there (and the server reported running) confirms the override is live.

**Requires comfy-cli ≥ 1.14.0 (this server's floor).** `COMFY_LOCAL_URL` itself landed after the
1.12.0 release and first shipped in 1.13.0 — below the enforced floor, so every *published* comfy-cli
this server accepts honors the variable. The floor is not a guarantee, though: the version guard
fails OPEN on a `--version` it can't parse, that errors, or that times out, so a source build or
fork older than 1.13.0 can still slip past it and silently ignore the variable. On any comfy-cli
without the support the variable is simply ignored (no error) and every tool keeps targeting
`127.0.0.1:8188` — which is why the `server_info` check above is the way to confirm it took
effect, rather than the version alone.

**Still reporting `:8188`?** Three causes, all silent, in the order worth checking:

1. **The value never reached comfy-cli** — it's in the wrong `env` block, or the client wasn't
   restarted after the edit. The workspace/Python fields `server_info` reports confirm which
   comfy-cli install you're actually talking to.
2. **The value is malformed** — comfy-cli ignores it and falls back to `127.0.0.1:8188`, emitting
   only a one-line stderr warning that this server's success path discards, so a typo
   (`https://…` — only `http` is accepted; a non-numeric port; a port outside 1–65535) looks
   exactly like the other two causes from the MCP side. Confirm by running
   `COMFY_LOCAL_URL=<your value> comfy env` in a terminal and reading stderr; see
   **Accepted values** above.
3. **comfy-cli is too old** — it predates the variable and ignored it. `server_info`'s
   `compatibility.comfy_cli_version` reports the detected version.

**Precedence** (comfy-cli resolves this, first match wins): an explicit `--host`/`--port` flag →
`COMFY_LOCAL_URL` → a comfy-cli-launched background server → `127.0.0.1:8188`.

## Which address variable do I want?

Two variables point ComfyUI work at an address, their names are similar, and they are **not**
alternative spellings of each other — they belong to **different programs** and are read at
different layers. Pick by which one you need; the table is the whole answer.

| | `COMFYUI_URL` (+ `COMFYUI_HOST` / `COMFYUI_PORT`) | `COMFY_LOCAL_URL` |
| --- | --- | --- |
| **Read by** | **this MCP server** (`_comfy_target`) | **comfy-cli** (`comfy_cli/local_address.py`); this server never reads it |
| **Means** | "a ComfyUI on **another machine** I control" | "the ComfyUI on **this machine** is not on `127.0.0.1:8188`" |
| **How it acts** | this server forwards `--host` / `--port` to the verbs that accept them | comfy-cli resolves its own target from the environment it inherits |
| **What it moves** | the **submit / job** tools plus `upload_file` (`run_workflow`, `generate_image`, `run_template`, the `jobs` family, and input staging) — see [what is and isn't remoted](#driving-a-remote-comfyui) | **every** verb, including the ones that take no `--host` / `--port` (`comfy env`, templates, models, download) |
| **Reported as** | a `comfy_target` block on `server_info` | the resolved `server` URL on `server_info` — **no** `comfy_target` block |
| **Use it for** | a GPU box over Tailscale / a private network | a port clash, a second instance, a container publishing a different port |

**Set one, not both.** They resolve independently, so together they *split* your tools rather than
conflicting loudly: comfy-cli ranks an explicit `--host` / `--port` flag above `COMFY_LOCAL_URL`, so
the submit/job tools would follow `COMFYUI_URL` while every other verb followed `COMFY_LOCAL_URL` — two
different ComfyUIs, no error. For a non-default address on **this** machine prefer `COMFY_LOCAL_URL`
alone, since it also reaches the verbs `COMFYUI_URL` cannot.

**Neither name is changing, and neither is deprecated.** They look like a rename waiting to happen —
they are not, because only one of them is ours. `COMFY_LOCAL_URL` is comfy-cli's own published
variable: renaming it here would document a name nothing reads, and its "local" is a factual
address-scope word (comfy-cli's *local* target, as opposed to its cloud one), not this project's
branding. `COMFYUI_URL` is this server's, and already carries no "local" to strip. So there is no
old spelling to accept and no deprecation period to sit through — if you have either variable in an
MCP client config today, it keeps working unchanged.

## Project anchoring

comfy-cli 1.15.0 ships a `project/1` convention (`comfy project init` / `comfy project status`,
this server's `project` tool) — a `comfy.yaml` plus `assets/` / `fragments/` / `blueprints/` /
`outputs/` / `.comfy/` under a root directory, with `status` reporting `recent_runs` and other
project-scoped state. comfy-cli resolves **which** project governs a call by walking **up** from
its own process's working directory only — there is no `--project` flag and no env var it reads
itself. That assumes a persistent shell session sitting inside the project tree; this server's own
working directory is whatever the MCP client happened to launch it from, arbitrary and unrelated to
any project the user has in mind — so out of the box, this server cannot participate in projects at
all.

Set **`COMFY_PROJECT`** to an absolute path to fix that: every comfy-cli spawn this server makes
then runs with that directory as its `cwd`, so comfy-cli's own cwd-walk resolves it exactly as if a
shell had `cd`'d there first. Read from the environment **once per process** (a value changed
mid-session is not picked up until restart) and validated on every spawn: the directory does **not**
need to contain `comfy.yaml` yet — call `project(action="init")` for that — but it does need to
**exist**, and it must be **absolute**. A relative value is rejected outright, never silently
resolved against this server's own (client-assigned, arbitrary) working directory — that resolution
would be exactly as non-deterministic as leaving `COMFY_PROJECT` unset while looking configured. A
set-but-relative or set-but-missing (or non-directory) value **fails closed**: the next comfy-cli
spawn raises rather than silently falling back to the unanchored default, because a silent fallback
would reintroduce exactly the non-determinism this feature exists to remove. Fix it by setting an
absolute path, unsetting `COMFY_PROJECT`, or creating the directory.

**This also moves where relative tool arguments land.** Relative path arguments (`workflow_path`,
`out_path`, `out_dir`, …) resolve against whatever directory comfy-cli's `cwd` is — the project root
once `COMFY_PROJECT` is set, not this server's original launch directory. Pass absolute paths when
you mean somewhere else.

Calling `project(action="init")` on a root that is **already** governed by a project (its own or an
ancestor's `comfy.yaml`) is not a no-op: comfy-cli raises `project_already_exists` rather than
re-initializing it. Call `project(action="status")` first when unsure whether a root is already
governed.

**Unset (the default): behavior is unchanged.** No `cwd` is passed to any spawn, exactly as before
this feature existed — every tool keeps acting on this server's own process directory, an unanchored
`comfy project status` returning comfy-cli's own `project_not_found`.

Set it in the client registration `env` block, same as `COMFY_BIN`:

```json
{
  "mcpServers": {
    "comfy-mcp": {
      "command": "comfy-mcp",
      "env": {
        "COMFY_BIN": "/path/to/venv/bin/comfy",
        "COMFY_PROJECT": "/Users/you/comfy-projects/my-project"
      }
    }
  }
}
```

## Tools

39 tools, grouped below by what they do. Every tool runs `comfy` with the global
`--json --where local` flags, unwraps comfy-cli's `envelope/1`, and returns its `data`.

**Argument naming** is uniform, so an agent never has to guess it (the server's handshake
instructions say the same thing): an input workflow file is always `workflow_path`, an output
file is `out_path`, an output directory is `out_dir`, a registry lookup key is `name`, and a job
handle is `prompt_id`.

### Run and monitor

| Tool | Wraps | What it does |
|---|---|---|
| `run_workflow(workflow_path, wait=True, timeout_seconds=110.0, confirm_spend=False)` | `comfy run --workflow <path> [--wait] [--allow-spend]` | Run a workflow JSON; `wait=False` submits async and returns a `prompt_id`. Most workflows are free local graphs, but one embedding partner (paid) nodes spends credits: `confirm_spend=True` unlocks that, and on an elicitation-capable client asks **you** per call, same posture as `run_template`. On a comfy-cli whose `comfy run` carries the gate the default fails closed (`spend_consent_required`, naming the `partner_nodes`); that is every release from 1.14.0 on, which the floor requires, so a paid graph only slips through on a build that got past the fail-open version guard. See [Workflows that spend](#workflows-that-spend--run_workflow). |
| `generate_image(prompt, checkpoint=None, wait=True, timeout_seconds=600.0)` | `comfy run-template default --param=6.text=<prompt> [--param=ckpt_name=<ckpt>]` | Text prompt → image in one call, with no hand-assembled workflow needed: it runs ComfyUI's own default SD1.5 text-to-image gallery template through the same verb (and the same run path) as `run_template`. Free — nothing here spends credits. Runs on whichever ComfyUI the server targets, so it follows `COMFYUI_URL`/`COMFYUI_HOST` like `run_workflow` does ([Driving a remote ComfyUI](#driving-a-remote-comfyui)); the checkpoint has to be installed on *that* machine. Retarget the template with [`COMFY_T2I_TEMPLATE` and its slot-key companions](#prerequisites). Same envelope shape as `run_workflow` (`prompt_id` + outputs); the fast on-ramp. |
| `partner_generate(model, params=None, confirm_spend=False, out_path=None, timeout_seconds=600.0)` | `comfy generate <model> [--param=value…] [--download=<path>] [--timeout=<s>] [--yes]` | Run a hosted **partner** model (Flux / Ideogram / DALL·E / Recraft / …). **Spends Comfy credits** on every call, where the local `run_workflow` / `generate_image` paths spend only when the graph itself carries partner nodes. Every call confirms the spend with you first — see [Spending credits](#spending-credits-on-partner-models) below. Runs **entirely on the partner's infrastructure** — your local ComfyUI is never in the execution path; use `emit_partner_workflow` below for the path where it is. `params` are the model's own schema-driven inputs, forwarded verbatim — `list_partner_models()` gives you the `model` aliases and `partner_model_schema(model)` the parameter list, so neither needs a terminal. `out_path` becomes comfy-cli's `--download` and is a save-path *template*: a plain path names the file, `{request_id}` / `{index}` / `{ext}` are substituted per output, and a trailing slash means "a default filename in this directory". `timeout_seconds` becomes comfy-cli's own `--timeout` so the engine — not a parent kill — owns the deadline on a job the partner may already have charged for. The result carries comfy-cli's printed text as `message` and, when it named the files it wrote, the resolved paths as `saved_paths` — so a caller reads the destination as data instead of scraping prose that rich may have wrapped mid-filename. |
| `emit_partner_workflow(model, out_path, params=None)` | `comfy generate <model> [--param=value…] --emit-workflow=<path>` | Write a runnable workflow JSON that drives the partner model's **API node** instead of calling the proxy, so **your own ComfyUI executes the partner model** (the other way there is an existing `API`-tagged gallery template via `search_templates` / `run_template`; this is the path from a model *alias*). Chain it: `emit_partner_workflow` → `run_workflow` → `fetch_outputs` (the three stay separate so the graph can be inspected, edited with `set_workflow_slot`, re-run, or embedded in a bigger pipeline). Calls no partner API, needs no API key, and **spends nothing**, so unlike `partner_generate` it has no `confirm_spend` argument and raises no confirmation prompt — *running* the emitted graph is what bills the partner node, so that `run_workflow` step is the one that needs `confirm_spend=True`. **Coverage is narrow:** comfy-cli maps only `flux-2`, `flux-pro`, `kling-i2v`, `nano-banana` and `seedance` to a node class, a small subset of `list_partner_models()`; every other model reaches its partner through the proxy only, so send those to `partner_generate`. An unsupported model raises with comfy-cli's own `emit_workflow_failed` message, which names the supported set for the comfy-cli you actually have installed. Returns comfy-cli's envelope data — `{"out", "model", "nodes"}`. |
| `run_template(name, params=None, confirm_spend=False, wait=True, timeout_seconds=600.0, ctx=None)` | `comfy run-template <name> [--param=KEY=VALUE…] [--timeout=<s>] [--allow-spend] [--async]` | One-command template run — fetch the gallery template, fill its parameterized slots, and run it on whichever ComfyUI the server targets, so it follows `COMFYUI_URL`/`COMFYUI_HOST` like `run_workflow` does ([Driving a remote ComfyUI](#driving-a-remote-comfyui)) (the one-shot alternative to `fetch_template` → `run_workflow`). `params` are `{slot: value}` (slot address `6.text` or name `prompt`), JSON-encoded so types round-trip. Most templates are free OSS graphs; one embedding partner (paid) nodes spends credits and fails closed unless `confirm_spend=True` unlocks it — and on an elicitation-capable client that asks **you** per call before anything runs (same posture as `partner_generate`; a default, free run is never prompted, and `comfy generate consent always` does not apply to this verb). No capability probe is needed here (unlike `partner_generate`): this verb's gate ships inside the verb itself, so a comfy-cli that has `run-template` has the gate. `wait=True` (the default) streams the run's live progress as MCP progress notifications, the same way `run_workflow` / `job(action="watch")` do, so a long template run is not a silent block; `wait=False` submits `--async` and returns a `prompt_id`. comfy-cli's `--timeout` for this verb is *per-event*, not a whole-run deadline, so `timeout_seconds` is forwarded only to tighten it below the engine's 120s default — prefer `wait=False` over a large `timeout_seconds` for long runs. |
| `job(action="status", prompt_id="", timeout_seconds=None)` | `comfy jobs status/watch/cancel/ls <prompt_id>` | One grouped tool over the six former `job_status`/`wait_for_job`/`watch_job`/`get_execution_error`/`cancel_job`/`get_queue` tools — pick a behavior with `action`. `"status"` (default) polls status + outputs. `"error"` returns a compact failure verdict — the failing node, `exception_type`/`exception_message`, and a bounded traceback tail — so an agent can self-repair; `error: None` on a healthy prompt. Failures comfy-cli diagnosed itself rather than ComfyUI (a `server_died` crash mid-run) carry no node-level fields, so the verdict also reports `error_code` — comfy-cli's own code, `None` on an ordinary node failure — with its message backfilling `exception_message`. `"wait"` polls (bounded, default 25.0s) until a job reaches a terminal status, returning a `{"timed_out": True, …}` payload on expiry — chain several rather than one long call. `"watch"` streams live progress (bounded, default 600.0s) as MCP progress notifications, same `timed_out` shape except `status` is a live `{progress, total, nodes_done}` snapshot. `"cancel"` stops a queued/running job. `"queue"` lists known jobs (Comfy Cloud-tracked rows filtered out, since this server never drives them; follows a configured remote like the other job actions). `prompt_id` is required for every action but `"queue"`; `timeout_seconds` only for `"wait"`/`"watch"` — passing either where the action does not use it is rejected rather than silently ignored. |
| `fetch_outputs(prompt_id, out_dir, url_only=False, inline_images=False, client_os="darwin")` | `comfy download <prompt_id> --where local -o <out_dir> [--url-only]` | Deliver a finished job's outputs — including a job that ran on a configured remote, which comfy-cli resolves from the local `prompt_id` state file (see [Driving a remote ComfyUI](#driving-a-remote-comfyui)). stdio writes directly into `out_dir`; Streamable HTTP cannot write on the client, so it returns temporary signed URLs on the MCP listener (five-minute lifetime) plus one ready command selected by `client_os`. In HTTP mode `url_only` and `inline_images` do not bypass that transfer boundary. In stdio, `url_only=True` emits the engine URLs without copying bytes and `inline_images=True` also returns bounded MCP image content. |

### Resource management

| Tool | Wraps | What it does |
|---|---|---|
| `system_stats()` | `comfy system-stats` | Read the live local ComfyUI's VRAM per device and system RAM. ComfyUI's whole `/system_stats` payload is forwarded **unmodified except for a `comfy_target_note` key** added when a remote target is configured (see below), so treat it as a passthrough, not a fixed schema: a `devices` list plus a `system` dict, whose keys are whatever that ComfyUI reports. The ones this server's guidance reads are per-device `vram_free` / `vram_total` (byte counts, alongside e.g. `name`, `type`, `index`, `torch_vram_free`) and `system.ram_free` / `ram_total` / `comfyui_version` — examples, not an exhaustive list. Nothing is filtered, so the `system` block also carries ComfyUI's `python_version` and `argv` (its full launch command line), which reach the model's context verbatim. Call it before a heavy `run_workflow` / `run_template` to decide whether to free memory first, and again afterwards to confirm the headroom landed. Read-only. Needs a running ComfyUI (the numbers come from the server), and unlike the run/job tools it is **not** diverted by `COMFYUI_URL`/`COMFYUI_HOST` — `comfy system-stats` takes no `--host`/`--port`. When one of those is set, a top-level `comfy_target_note` (`host` / `port` / `source` / `note`) is added naming that target and saying these numbers describe whichever ComfyUI comfy-cli itself targets — settle whether that host is *this* machine before gating a run on them. A malformed value gives an error-shaped note (`error` / `note`) rather than breaking the call; with nothing configured the key is absent. |
| `free_memory(unload_models=True, free_memory=None)` | `comfy free [--unload-models\|--no-unload-models] [--free-memory]` | Ask ComfyUI to unload models from VRAM and reset its executor cache (`POST /free`). `free_memory=None` means **follow `unload_models`**, so the default call requests both — maximum headroom, and a deliberate divergence from comfy-cli's `--free-memory`, which defaults to off; pass `free_memory=False` for the CLI's lighter unload that keeps cached executor state. **The cache reset can't be had without the unload:** ComfyUI's worker resolves the pair as `flags.get("unload_models", free_memory)` and its `/free` handler only records `unload_models` when true, so `unload_models=False, free_memory=True` would unload everything — that pair is rejected rather than sent. `unload_models=False` therefore asks ComfyUI to do nothing; it's a deliberate no-op kept for symmetry with the CLI. **Not immediate and never destructive:** ComfyUI applies the request when its queue worker next iterates — immediate if idle, after the current job if busy — and it does **not** interrupt a running job, so it cannot be used to stop one (`job(action="cancel")` does that). Returns comfy-cli's acknowledgement of what was *requested*, not a measurement; read `system_stats` afterwards to confirm. Also **not** diverted by `COMFYUI_URL`/`COMFYUI_HOST`, and it carries the same `comfy_target_note` key when one of those is set — naming the target this call may not have freed. See [Using with local LLMs](#using-with-local-llms-vram-coordination). |

### Diagnostics

| Tool | Wraps | What it does |
|---|---|---|
| `server_info()` | `comfy env` + `comfy outdated` | Is a local ComfyUI running, where, and which workspace. **Call first.** Passes through comfy-cli's `hardware` block (GPU vendor/model, VRAM or unified memory, total RAM) when the installed comfy-cli reports one — the signal behind [When to use this server](#when-to-use-this-server), which the startup `Machine snapshot` also carries in the handshake instructions so routing never waits on this call. Also attaches a `freshness` block (`comfy outdated`): installed-vs-latest for ComfyUI core and each custom node pack, so a stale install is flagged before it masquerades as a missing model/node. On a comfy-cli without the `outdated` verb the block degrades to `freshness: {"error": "freshness unavailable: …", "unsupported": true}` (a benign capability gap — skip staleness advice, nothing is broken); on any other probe failure such as a network error it degrades to `freshness: {"error": …}` carrying the real reason. Either way the tool itself still succeeds. Reports the configured remote under a `comfy_target` block when `COMFYUI_URL`/`COMFYUI_HOST` is set (see [Driving a remote ComfyUI](#driving-a-remote-comfyui)). |
| `auth_status()` | `comfy cloud whoami` | Comfy Cloud credential status for partner-API nodes (read-only, never returns secrets). Adds a local `registration_env_key_present` bool for the `COMFY_API_KEY` registration-env slot whoami can't see. |
| `auth_login()` | `comfy cloud login --no-browser --timeout 600` | Start Comfy Cloud sign-in and return `{"status": "awaiting_browser", "login_url": …, "expires_in_s": …}` — the URL for the **user** to open, so an agent can get them signed in instead of telling them to run the CLI by hand. Returns as soon as comfy-cli emits the URL; the sign-in keeps running in the background (comfy-cli owns the OAuth flow and the loopback callback, so no OAuth logic lives here). Confirm the result with `auth_status`. Only one sign-in at a time: calling it again while one is pending re-reports the same URL without spawning a second flow, and calling it after the flow ended reports `completed` / `failed` once and then clears. Never returns tokens. |
| `which()` | `comfy which` | Which ComfyUI install/workspace comfy-cli currently targets (a lighter answer than `server_info`). |
| `project(action="status")` | `comfy project status` / `comfy project init` | Report or create the operator-anchored `project/1` (`action="status"` / `"init"`). See [Project anchoring](#project-anchoring). |
| `get_logs(tail=200, port=None)` | `comfy logs --tail <tail> [--port <port>]` | Tail the background ComfyUI's captured log (`<workspace>/user/comfyui_<port>.log`) — closes the debugging loop after a detached `launch_comfyui`. Returns `{lines, path, truncated}`; a missing log file returns `{"error": "no_log_file", …}` rather than raising, and on a newer comfy-cli that message lists every candidate path checked. Pass `port` when several instances/ports have run, or after a crash, to force `user/comfyui_<port>.log` resolution. A newer comfy-cli also returns `source` / `port_mismatch` / `mtime` / `size`, forwarded untouched: if `port_mismatch` is true or `source` reports a fallback, the lines may belong to a different server — re-call with an explicit `port` (note `user/comfyui.log`, unsuffixed, is ComfyUI-Manager's log for servers started without an explicit `--port`). A comfy-cli too old to accept `--port` raises an upgrade instruction rather than silently returning the default log. |
| `discover(schemas_only=True)` | `comfy discover [--schemas-only]` | comfy-cli's self-describing surface — learn the CLI's own contract at runtime. The default `schemas_only=True` returns just the schema bundle (~34 KB / ~9k tokens); `schemas_only=False` adds the full command tree and error codes (~177 KB / ~45k tokens), which is ~1.8x the 25,000 tokens Claude Code's `MAX_MCP_OUTPUT_TOKENS` defaults to — and that cap **truncates** rather than rejects, so the full surface comes back as JSON cut mid-structure unless the cap is raised. Tool-output caps are per-client, not an MCP-wide default, so treat 25,000 as the representative number; the schemas bundle is the mode that fits regardless. |

### Workflow building

| Tool | Wraps | What it does |
|---|---|---|
| `validate_workflow(workflow_path)` | `comfy validate --workflow <path>` | Pre-flight a workflow against the live `object_info` before a slow run. Returns comfy-cli's own report — `{"valid": bool, "error_count", "errors": [...], "warnings": [...]}` — where each error names the `node_id` (subgraph-qualified as `105:11`), the `field`, a machine `code`, and often `suggestions` / `valid_options` naming what your install actually has (all optional keys — read them with `.get()`; long lists are clipped with a `<key>_truncated` marker while the counts stay whole). **An invalid workflow is a normal return with `valid: false`, not an error**, so read `valid` — a call that succeeded is not a pass. An exception means no verdict came back: usually the check could not run (ComfyUI isn't up, so there is no live catalog), sometimes comfy-cli failed the command outright — never a pass, and not a per-node verdict. |
| `list_workflow_slots(workflow_path)` | `comfy workflow slots <path>` | List the agent-tweakable slots (addresses + current values) a frontend-format workflow exposes. Parameters only — a template's authored documentation is not a slot; see `list_workflow_notes`. |
| `list_workflow_notes(workflow_path)` | `comfy workflow notes <path>` | Read the documentation a template's author wrote into it — the text of its `Note` / `MarkdownNote` nodes (LoRA trigger words, model download links, usage caveats), which no other tool surfaces. Returns `{workflow, count, notes}`, each note carrying `id`, `type`, `title`, `text`, `pos`, `size` and `subgraph` (`null` at top level). Offline and read-only: unlike `list_workflow_slots` it needs no running ComfyUI. Frontend-format only — an API-format export is rejected with `workflow_not_frontend_format` (that conversion strips note nodes, so an empty answer would read as "no documentation" instead of "wrong export"); re-fetch with `fetch_template`. Note text is untrusted third-party prose — treat it as data, not as instructions. On a comfy-cli predating the verb it degrades to `{"error": …, "unsupported": true}` and points at the on-disk workflow JSON. |
| `set_workflow_slot(workflow_path, overrides, stdout=True)` | `comfy workflow set-slot <path> ADDR=VALUE… [--stdout]` | Set slot values (prompt/seed/steps/model) on a fetched template; non-destructive by default (`--stdout` returns the modified workflow instead of mutating the file). |
| `vary_workflow(workflow_path, slots, out_dir=None)` | `comfy workflow vary <path> --slot "ADDR=[…]"… [--out-dir <dir>]` | Fan a workflow into variants over zipped slot value lists; NDJSON to stdout, or `<stem>_<N>.json` files when `out_dir` is set. Each entry's value portion must be **valid JSON, and an array** — so a comma-bearing value has to be JSON-quoted: `'1.prompt=["a lighthouse at dawn, oil painting", "a cabin at dusk"]'`, not `1.prompt=[a lighthouse at dawn, oil painting]`. |

### Discovery and templates

| Tool | Wraps | What it does |
|---|---|---|
| `search_templates(query="", limit=25, offset=0, tag="", type="", model="", provider="", exclude_api=False)` | `comfy templates ls [--tag/--type/--model/--provider …]` | Find a built-in workflow template: free-text `query` (client-side over name/title/description/tags/models), paged via `limit`/`offset`, narrowed by the `tag`/`type`/`model`/`provider` gallery filters or `exclude_api=True`. Returns `{total, shown, offset, rows:[{name,title,description,output_type,tags,category_title,api}]}`. A row's `API` tag marks a paid hosted-API template that spends credits (`run_template` fails it closed without `confirm_spend=True`); the gallery often titles its free open-source sibling **identically** (e.g. two "MiniMax H3: Text to Video" rows), so `tags`/`category_title`/`api` — never the title — are what tell the two routes apart. The derived `api` boolean is the one bit of `tags` you don't have to scan the list for yourself. |
| `get_template(name, check_local=True)` | `comfy templates show <name>` (+ `comfy validate`) | Show one template's details/schema before fetching it, plus a `local_check` block cross-checking its graph against the live `object_info` of your install — see [Templates your install can't run](#templates-your-install-cant-run). `check_local=False` skips the check (metadata only, one call). |
| `fetch_template(name, out_path, check_local=True)` | `comfy templates fetch <name> --out <path>` (+ `comfy validate`) | Write a template's runnable workflow JSON to `out_path`; returns `{path, local_check}` — `path` is the absolute path for `run_workflow`, `local_check` is the same cross-check run on the file just written. The file is written either way. |
| `nodes(action="search", query="", name="", produces="", accepts="", category="", pack="", label="", limit=None, from_type="", to_type="", max_depth=None, max_paths=None, exclude_api=False)` | `comfy nodes search/show/ls/upstream/downstream/path/types/categories` | One grouped tool over the eight former `search_nodes`/`get_node`/`list_nodes`/`nodes_upstream`/`nodes_downstream`/`nodes_path`/`nodes_types`/`nodes_categories` tools — pick a behavior with `action`. `"search"` (default) finds a class name by keyword; `"get"` returns one class's full input/output schema; `"list"` filters by `produces`/`accepts`/`category`/`pack`/`label` (bare call lists all); `"upstream"`/`"downstream"` list classes that can feed `name`'s inputs / accept its outputs (`limit` caps the count); `"path"` finds node chains routing a value from `from_type` to `to_type` (`max_depth`/`max_paths` default to 6/10 when omitted); `"types"` lists every connection type by connectivity; `"categories"` returns the menu-category tree. Each param is scoped to the actions that consume it — `query` only `"search"`, `name` only `"get"`/`"upstream"`/`"downstream"`, the five list filters only `"list"`, `limit` only `"upstream"`/`"downstream"`, `from_type`/`to_type`/`max_depth`/`max_paths` only `"path"`, `exclude_api` only `"search"` — passing one where the action does not use it is rejected rather than silently ignored. `"search"` and `"list"` rows carry comfy-cli's own `is_api_node` flag once the installed engine emits it: `true` marks a **paid partner-API node** that bills the signed-in account's credits when executed (`run_workflow`/`run_template` fail it closed without `confirm_spend=True`), `false` a free node running on local weights — two classes can share a display name and differ only in this flag (the MiniMax H3 pair), so it, not the title, is what tells the paid and free routes apart. `exclude_api=True` (search only — `"list"` is already filter-rich) drops the `true` rows client-side and reports the page as `count` plus a new `excluded_api`, leaving `total` exactly as comfy-cli reported it: `total` counts every match the engine found *before* its own 20-row page cap, so recomputing it post-filter would report an all-paid page as `total: 0` and read as "no such node". Rows that do **not** carry the field at all are KEPT — unknown is not paid, so an engine predating the field filters nothing rather than hiding free nodes. All reads are against the **live local** `object_info` (includes installed custom nodes). |
| `workflow_deps(workflow_path)` | `comfy node deps-in-workflow --workflow <path> --output <tmp>` | Which node **packs** a workflow needs, resolved from the node classes it references — the **diagnosis** half of the missing-node story, and the only tool here that can go from a class name to a pack id. `nodes` (`"get"`/`"search"`) reads the running install's live `object_info`, so by construction it only ever finds classes you already have; this reads ComfyUI-Manager's node→pack map, which covers packs that are not installed. Returns Manager's manifest verbatim: `{"custom_nodes": {"<pack-id-or-repo-url>": {"state": "installed" | "not-installed" | "disabled" | "invalid-installation", …}}, "unknown_nodes": ["SomeClass", …]}` — the `not-installed` keys are the `install_node` list, and `unknown_nodes` are classes Manager could attribute to no pack at all (a hand-written node, or a typo in the graph). So the full loop is `validate_workflow` → `workflow_deps` → `install_node` → `restart_comfyui`. Read-only: it installs and downloads nothing. Accepts the same `.json` (API or UI export) file `run_workflow` takes, plus a `.png` with an embedded workflow. **Requires ComfyUI-Manager** — the verb resolves through Manager's `cm-cli`, so an install without it gets `{"error": …, "unsupported": true}` rather than a raw usage dump. comfy-cli emits no envelope for this verb and writes its answer to a `--output` file, so this server supplies a temporary path of its own, reads the manifest back, and removes it. |
| `node_dependencies(pack="", registry_id="")` | `comfy node deps [<pack>] [--registry <id>]` | A custom node **pack**'s declared Python requirements (its `requirements.txt` / `pyproject.toml`) against the versions actually installed in the workspace venv — each requirement carrying a `satisfied` / `mismatch` / `missing` / `unparseable` / `unknown` status, plus per-pack counts. This is what tells you whether a pack's imports are failing because a dependency is absent, or whether installing one pack would conflict with another. `pack` empty reports every installed pack; `registry_id` pre-checks a **not-yet-installed** registry pack against the same venv before you install it (its latest published version — the registry exposes no per-version endpoint). The two are additive, so naming the same id both ways yields an installed row and a registry row to compare. Read-only: nothing is installed or changed. Pack-level filesystem + venv introspection, so it is deliberately separate from `nodes`, which introspects node **classes** over the live `object_info`. The verb ships in comfy-cli 1.14.0, this server's floor; on a build without it (one that got past the fail-open version guard) it returns `{"error": …, "unsupported": true}` rather than a raw usage dump. |
| `search_models(query="", folder="")` | `comfy models search` / `models list-folder <folder>` / `models list-folders` | List/search model files on disk. **Local:** filenames only, no cloud enrichment. |
| `list_partner_models(style="", partner="", query="", limit=100, offset=0)` | `comfy generate list [--style S] [--partner P] [--query Q]` | The catalog of hosted **partner** models `partner_generate` can run — the only place that list exists (nothing in `discover` / `nodes` / `search_templates` carries the partner aliases). One record per model: `alias` (what you pass as `model`), `id`, `partner`, `category` (the model's style, and the axis `style` filters on — `text-to-image`, `image-edit`, `image-to-image`, `text-to-video`, `image-to-video`, `video-extend`, `controlnet`, `inpaint`, `outpaint`, `upscale`, `background`, `lipsync`, `vectorize` as this is written; comfy-cli owns that set, so read it off an unfiltered call), `mode` (`sync`/`async`, the partner's protocol — `partner_generate` waits either way) and the model's full, untruncated `summary`. Filters are forwarded to comfy-cli: `style` is exact and **case-sensitive**, `partner` exact and case-insensitive, `query` a substring over `id` + `summary`. `limit` (default 100, capped at 200) / `offset` page the result (`{total, shown, offset, filters, models}`) so a growing catalog can't trip the client's tool-output cap; 52 models as this is written, so the default returns all of them — check `shown` against `total` rather than assuming that stays true. |
| `partner_model_schema(model)` | `comfy generate schema <model>` | One partner model's callable parameters — what to put in `partner_generate`'s `params`. Returns `{model, id, partner, category, summary, mode, polling, content_type, params, example}`, where each `params` record carries `name`, `type` (`string`/`integer`/`number`/`boolean`/`enum`/`object`/`array`/`binary` — `binary` is a local file path comfy-cli uploads or inlines for you), `required`, `default`, `enum` and the spec's own `description`. Reads the spec only: no partner call, no key, no spend. |

### Lifecycle and assets

| Tool | Wraps | What it does |
|---|---|---|
| `launch_comfyui(extra_args=None, confirm_network_exposure=False)` | `comfy launch --background [-- <extras>]` | Start the local ComfyUI detached; forwards `extra_args` to ComfyUI. **Network-exposing flags ask the USER first.** ComfyUI has no authentication, so `--listen` on a non-loopback address — including a **bare** `--listen`, which ComfyUI expands to every interface — or `--enable-cors-header` would publish its full API (arbitrary workflow execution, plus file reads/writes under whichever directories it was started with — the same `extra_args` can move those roots with `--base-directory`) to anything that can reach this machine. Those raise an MCP elicitation naming exactly that **and echoing the whole argument list**, so the user approves the command line that actually runs; a decline starts nothing; on a client that cannot show prompts the call errors unless `confirm_network_exposure=True`, which an agent may pass **only** when the user has actually agreed. That prompt is raised even when `confirm_network_exposure=True` is passed, so a host's "always allow this tool" toggle is not standing authority to publish the machine. `--listen 127.0.0.1` / `::1` / `localhost` is the default bind spelled explicitly and needs no confirmation, and every other flag (`--port`, `--cpu`, …) passes straight through unprompted. `extra_args` is bounded (64 entries, 4096 characters each) so an oversized argv is a named error rather than an `OSError` from the spawn. Serialized with `stop_comfyui` / `restart_comfyui`: they share comfy-cli's one recorded server and the ComfyUI port, so a call made while another lifecycle call is in flight is refused immediately rather than racing it. |
| `stop_comfyui()` | `comfy stop` | Stop the ComfyUI that comfy-cli launched (only its own recorded pid). Shares the launch/restart one-at-a-time lock, so it cannot land between a restart's stop and its launch. |
| `restart_comfyui(extra_args=None, confirm_network_exposure=False, confirm_kill_untracked=False)` | `comfy stop` then `comfy launch --background [-- <extras>]` | Stop-then-launch the local ComfyUI (best-effort stop); forwards `extra_args` to the fresh server. Handy for relaunching with different flags — which is why it carries `launch_comfyui`'s **network-exposure confirmation** unchanged (`--listen` on a non-loopback address, or `--enable-cors-header`, asks the USER first). The gate runs **before** the stop, so a declined restart leaves the running server alone rather than killing it and then refusing to bring it back. **When the stop finds nothing recorded and the launch then loses the port**, a ComfyUI is running that comfy-cli did not start — so this asks comfy-cli what it is (`comfy stop --port <p> --dry-run`, which reports the process it *would* stop without stopping it) and, if the engine positively identifies a ComfyUI, offers to recycle it: the USER is shown that process's **pid, command line and port** and asked, and only on a yes is it stopped (`comfy stop --port <p>`) and the launch retried once. Declining, an engine that will not vouch for the listener, and a comfy-cli too old to have `comfy stop --port` all land on the same port error as before, enriched with whatever identity the dry run did establish. That confirmation necessarily comes *mid*-sequence, which is safe because the stop half was a no-op — nothing was recorded, so declining leaves the running server untouched. On a client that cannot show prompts the kill needs `confirm_kill_untracked=True`, which an agent may pass **only** once the user has actually agreed (the equivalent outside this server is `comfy stop --port <p>` in a terminal); like every other confirm flag it grants nothing on a client that *can* be prompted. Sessions pointed at a remote ComfyUI (`COMFYUI_URL` / `COMFYUI_HOST`) never reach that path — the lifecycle verbs are local-only, so which machine's port is in question stops being obvious. Both halves run inside one lifecycle slot, so a concurrent `launch_comfyui` / `stop_comfyui` is refused rather than slipping into the gap between them (bounded by the timeouts, ~4 minutes worst case, or ~10 if it is waiting on that confirmation — the prompt is raised without dropping the slot). |
| `update_comfyui(target="comfy", confirm_update_all=False)` | `comfy update <all\|comfy\|cli>` | Update the local install: `"comfy"` = ComfyUI core, `"all"` = the installed custom node packs, `"cli"` = comfy-cli itself. This is what `server_info`'s `freshness` block points at when it reports a stale install. Slow (a core update re-installs requirements; 30-minute timeout) and the updated code only takes effect after a `restart_comfyui`. **`target="all"` asks the USER first — and only that target.** It `git pull`s and `pip install`s **every** third-party custom node pack into ComfyUI's Python environment, so it runs code those packs' authors have published since you installed them, and it can move a pack (or a shared dependency) to a version other packs and your saved workflows don't work with. comfy-cli does not gate that, so on a client that supports MCP elicitation a prompt naming exactly that is raised and a decline runs nothing; on a client that cannot show prompts the call errors unless `confirm_update_all=True`, which an agent may pass **only** when the user has actually agreed. That prompt is raised even when `confirm_update_all=True` is passed, so a host's "always allow this tool" toggle is not standing authority to run third-party code. `target="comfy"` and `target="cli"` update first-party code from known repositories and are never prompted. Any other `target` is rejected before comfy-cli is invoked (and before anyone is asked), and a second update requested while one is still running is refused rather than run in parallel (concurrent `git`/`pip` against one workspace can leave it half-installed) — that refusal comes before the prompt too, so nobody approves a call that was never going to run. |
| `switch_comfyui_version(version, confirm_switch=False)` | `comfy update comfy --version <version>` | Move the local ComfyUI install to a **specific** version — `"nightly"`, `"latest"`, or a release like `"0.24.0"` / `"v0.24.0"` — so you can roll **back** to reproduce or rule out a regression (`update_comfyui` only ever moves forward to the latest). **Destructive:** the engine stashes any uncommitted changes in the ComfyUI checkout, moves it to that version, and reinstalls that version's Python dependencies (minutes, not seconds; 15-minute timeout). **The USER is asked to confirm every call** — on a client that supports MCP elicitation a prompt naming exactly that is raised, and a decline cancels with nothing changed; on a client that cannot show prompts the call errors unless `confirm_switch=True`, which an agent may pass **only** when the user has actually agreed. That prompt is raised even when `confirm_switch=True` is passed, so a host's "always allow this tool" toggle is not standing authority over the install. It **refuses while a local ComfyUI is running** (reinstalling under a live process can leave it serving half-replaced code) — checked both before the prompt and again immediately before the switch, since the prompt may sit unanswered for minutes, and fail-closed, so a `comfy env` this server cannot read is refused rather than read as "stopped" — and it does **not** restart anything — the flow is `stop_comfyui` → `switch_comfyui_version` → `launch_comfyui` → `server_info` to confirm what came up. Returns `{switched_to, result, restart_required: true}`. A malformed version is rejected before comfy-cli is invoked; a comfy-cli whose `comfy update` predates `--version` surfaces as an "upgrade comfy-cli" error rather than a raw usage dump; and it shares `update_comfyui`'s one-at-a-time lock. |
| `install_node(names, confirm_install=False)` | `comfy node install <name...> --exit-on-fail` | Install custom node packs into the local ComfyUI — the acquisition half of the missing-node story, after `validate_workflow` / `run_workflow` names a node class this install lacks and `node_dependencies(registry_id=…)` pre-checks the pack's requirements. `names` are **registry pack ids** (slugs like `"comfyui-impact-pack"`), not node class names: a git URL, a filesystem path, or `"all"` is refused before comfy-cli is invoked — the URL case deliberately, because the confirmation prompt promises the user a *named pack from the registry*, so nothing else may ride through it. (To update the packs you already have, use `update_comfyui(target="all")`; to install from a URL, run `comfy node install` in a terminal.) **Installing a pack runs third-party code** — a `pip install` of its dependencies into the ComfyUI environment plus the pack's own install script — so **the USER is asked to confirm every call**, and that prompt is raised even when `confirm_install=True`, since a host's "always allow this tool" toggle is not standing authority to execute third-party code and the pack names are frequently a model's guess. On a client that cannot show prompts the call errors unless `confirm_install=True`, which an agent may pass **only** once the user has actually agreed. It does **not** restart anything — new nodes are invisible until ComfyUI restarts, so the flow is `install_node` → `restart_comfyui` → `nodes(action="search")` — and it shares `update_comfyui`'s one-at-a-time lock (same venv, same `pip`). `--exit-on-fail` is always forwarded, because without it comfy-cli reports a failed install as success — but it is not sufficient on its own: ComfyUI-Manager prints a pack's failure *before* consulting the flag, so `comfy node install` can report a pack as failed and still exit 0. The verdict is therefore read out of the engine's own output rather than off the exit status. 30-minute timeout. Returns `{installed, result, restart_required}` — **`installed` lists only the packs the engine did not report as failed, not an echo of `names`** — plus `{failed, error}` when any pack failed, where each `failed` entry carries the engine's own message and a `code` of `pack_not_found` (the id is not in this install's registry channel, so retrying it will not help) or `install_failed`. `restart_required` is `false` when nothing was installed, because there is then nothing for a restart to pick up. |
| `upload_file(file_path, client_os)` | `comfy upload <file> --no-overwrite` | Stage one JPG/JPEG/PNG/WebP/GIF into the target ComfyUI's `input` dir (unlocks img2img / inpaint), using the same public schema and remote command report as ComfyCloud. `file_path` is absolute on the MCP client and `client_os` is `darwin`, `linux`, or `windows`. stdio runs the command directly. HTTP returns a credential-free, five-minute, single-use PUT command on the MCP listener; the route accepts at most 50 MiB, responds with `{"name":"...","subfolder":"","type":"input"}`, and passes owner-only scratch through the same `ComfyCliClient`. Goes to whichever ComfyUI the server targets — local by default or the remote configured by `COMFYUI_URL`/`COMFYUI_HOST`; remote upload needs comfy-cli ≥ 1.14.0; base64/data-inline upload is unsupported. |
| `download_model(url, relative_path=None, filename=None, wait=True, timeout_seconds=110.0)` | `comfy model download --url <url> [--relative-path <path>] [--filename <name>] --background` | Download a model file by direct URL (HuggingFace / CivitAI) into the local models dir; download-by-URL only, not a hub search. Local-only and **enforced**: `comfy model download` has no `--host`/`--port`, so with a remote configured (`COMFYUI_URL`/`COMFYUI_HOST`) this refuses instead of writing the checkpoint to a disk the remote cannot see — install the model on the remote host itself, or set `COMFY_MCP_REMOTE_SHARED_MODELS=1` if this machine's models dir *is* the remote's (shared NFS / tailnet mount). See [Driving a remote ComfyUI](#driving-a-remote-comfyui). The transfer is **submitted** to comfy-cli's background worker and returns a `download_id`, so a multi-GB checkpoint no longer holds the MCP request open past the client's deadline: `wait=True` (default) polls that id for you within a bounded budget and returns `{"timed_out": True, "download_id": …}` — not an error — if the transfer is still running, while a `failed` / `cancelled` download raises with comfy-cli's own error. On that path `timeout_seconds` is the **end-to-end** budget for the whole call, submit included, so the submit and the poll cannot add up past the client deadline the 110s default is chosen to sit under. `wait=False` returns the submit payload immediately and keeps the submit's own fixed budget. Every payload from that background path keys the handle `download_id`, matching the argument name every download tool takes, so an id read out of one result goes straight back into the next call — the legacy foreground fallback below is the exception, since no id is ever minted on it. The file is written straight to its final path as it transfers, so a filesystem / `search_models` check mid-flight sees a present-but-incomplete file — `download(action="status")` is the source of truth. `relative_path` resolves from the workspace root and must be the models dir or a subfolder of it — `models`, `models/loras` (a bare `loras` is rejected, not assumed); sibling dirs like `custom_nodes/…`, `input`, `output` are refused. Use `/` as the separator on every host, Windows included. Against a comfy-cli too old to know `--background` (anything below 1.14.0, which only reaches here past the fail-open version guard) it falls back to the previous foreground download — which has no id to detach or poll, so it blocks even on `wait=False`, and every payload it returns is marked `background_unsupported: true` to say so. On that fallback `wait=True` is bounded by what is left of your `timeout_seconds` (capped at 1800s) rather than by a silent half hour: when the bound expires the transfer is killed and the error names where an incomplete file may remain, since there is no `download_id` to check it with. Cancelling the tool call kills the transfer the same way instead of orphaning it. |
| `download(action="status", download_id="", timeout_seconds=None)` | `comfy model download-status/download-cancel <download_id>` | One grouped tool over the three former `download_status`/`wait_for_download`/`cancel_download` tools — pick a behavior with `action`. Does **not** start a transfer — that's `download_model`, whose `download_id` this tool consumes. `"status"` (default) returns `status`, `completed_bytes` / `total_bytes` / `percent`, `elapsed_seconds`, `dest`, and `error` — the only proof a model is complete and loadable. `"wait"` polls (bounded, default 25.0s, ceiling 3600s) until a download reaches a terminal state (completed / failed / cancelled), returning a `{"timed_out": True, …}` payload on expiry — chain several rather than one long call, the `job(action="wait")` shape, for transfers. `"cancel"` stops a running download and removes its partial file. `download_id` is required for every action; `timeout_seconds` only for `"wait"` — passing it elsewhere is rejected rather than silently ignored. Every payload keys the handle `download_id` on the way back out too, including the status nested inside a `"wait"` timeout, so a handle read out of one result passes straight into the next call without renaming; comfy-cli spells the same field `id`, and that spelling is kept alongside rather than replaced. On a comfy-cli without the verb, returns `{"error": …, "unsupported": true}` instead of a raw usage dump — that shape carries no handle at all, since a CLI that old can never have minted one. |

Node introspection (`nodes`, all eight actions) and `search_models`
read the **user's live install** (custom nodes included), not a static catalog — that's the
local differentiator from the cloud MCP's equivalents. The graph-wiring actions (`"upstream"` /
`"downstream"` / `"path"`) are what an agent authoring a workflow uses to find compatible nodes.
`workflow_deps` is the one node tool that does NOT read the live install: it resolves a workflow's classes against ComfyUI-Manager's node→pack map, which is what lets it name a pack that is not installed — precisely the question the live-catalog tool cannot answer.
`node_dependencies` reads that same live install from the other side — the **packs** on disk and
the venv they installed into, rather than the node classes ComfyUI loaded from them — which is how
an agent tells "this pack's nodes are missing from `object_info`" apart from "this pack's Python
dependencies never installed".

## Troubleshooting

### macOS: `PermissionError: [Errno 1] Operation not permitted` / `Fatal Python error`

**Symptom.** Setup fails with a raw Python startup crash naming a file under `~/Documents`,
`~/Desktop` or `~/Downloads` — most often the ComfyUI venv's `pyvenv.cfg`:

```text
Fatal Python error: init_import_site: Failed to import the site module
PermissionError: [Errno 1] Operation not permitted: '/Users/you/Documents/ComfyUI/venv/pyvenv.cfg'
```

**Cause.** macOS protects those three folders with TCC (Transparency, Consent & Control). An app
without **Full Disk Access** cannot read them — and neither can the processes it spawns. So when
your ComfyUI install (and its `venv`) lives under one of them, the `comfy` binary your MCP client
launches dies before it executes a single line. Nothing is wrong with ComfyUI, comfy-cli, or this
server: it is a macOS privacy setting.

**Fix — either one works:**

1. **Grant your MCP client Full Disk Access.** System Settings → Privacy & Security → Full Disk
   Access → add the app (Claude Desktop, Cursor, or the terminal you launch the client from), then
   **quit and reopen it** so the new permission takes effect.
2. **Or move the ComfyUI folder somewhere unprotected** — e.g. `~/ComfyUI` — and re-point comfy-cli
   at it with `comfy set-default <path>`. Update `COMFY_BIN` in your client config too if it names
   a path inside the old location.

Where it can, the server says this for you: a tool call blocked this way returns the guidance above
instead of the raw traceback. The one case it cannot catch is **its own** interpreter startup (this
server installed under a protected folder) — Python dies before any of its code runs, so that one
surfaces as the raw traceback in your client's MCP logs. Same fix.

## Failure log (opt-in)

When you're diagnosing a flaky setup, an MCP client's transcript is a poor record: it scrolls, it
truncates, and the interesting failures (a missing `comfy` binary, a crash before any JSON, a
timeout) are exactly the ones that leave the least behind. Set **`COMFY_MCP_DEBUG_LOG`** and
the server appends one JSON object per comfy-cli **failure** to a local file you can `jq`, grep, or
zip up and attach to a bug report.

| Value | Behavior |
| --- | --- |
| unset, empty, or `0` | **Off (the default).** Nothing is created and no log file is opened. |
| `1` | On, at the default path for your OS (below). |
| anything else | On, and the value is used as the log file path (parent directories are created). |

Default paths — the same per-OS local-state convention comfy-cli itself uses:

| OS | Path |
| --- | --- |
| macOS | `~/Library/Application Support/comfy-mcp/failures.jsonl` |
| Windows | `~/AppData/Local/comfy-mcp/failures.jsonl` |
| Linux / other | `~/.config/comfy-mcp/failures.jsonl` |

Each line records the failure `kind` (`error_envelope`, `no_json`, `timeout`, `binary_missing`,
`spawn_failed`, `schema_mismatch`), a UTC `ts`, the comfy-cli `args`, its `exit_code` and the envelope's
`error_code`, the message you saw in your client, and up to 4,000 characters of `stdout_tail` /
`stderr_tail` — deliberately more output than an error message can carry:

```console
$ jq -r 'select(.kind == "timeout") | .ts + "  " + (.args | join(" "))' \
    ~/Library/Application\ Support/comfy-mcp/failures.jsonl
```

The setting is read when the **MCP server process starts**:

- **stdio:** put `COMFY_MCP_DEBUG_LOG=1` in the same MCP client `env` block as
  `COMFY_BIN` / `COMFY_API_KEY`, then fully restart that client-launched server.
- **Streamable HTTP:** set it on the machine and service that run
  `comfy-mcp serve`, not in the remote client's URL-only configuration, then
  restart that HTTP process. For example:

  ```bash
  COMFY_MCP_DEBUG_LOG=1 \
  COMFY_BIN=/path/to/venv/bin/comfy \
  COMFY_API_KEY=YOUR_COMFY_API_KEY \
  COMFY_LOCAL_URL=http://127.0.0.1:8188 \
  comfy-mcp serve --host 127.0.0.1 --port 9000
  ```

In both modes the runner publishes the same immutable failure event. A single
JSONL writer observes those events and returns immediately while the setting is
off; when enabled it writes on the **MCP server host**. This deliberately small
observer design keeps subprocess code independent of storage without adding a
general event bus. Both adapters therefore produce the same failure kinds,
redaction, rotation, and owner-only permissions.

The file rotates itself: 1 MiB per file with two older generations kept (`failures.jsonl.1`,
`failures.jsonl.2`), so it stops growing at roughly 3 MiB no matter how long you leave it on.
Successful calls are never recorded, and nothing is ever transmitted anywhere — the log is local,
full stop.

HTTP upload request bytes are never passed as comfy-cli arguments and never enter the failure event
or JSONL file. A failed capability upload can record its generated temporary path and comfy-cli's
bounded output, but the temporary file itself is removed before the route returns.

> **Privacy — review before sharing.** The log contains local file paths and comfy-cli's own
> command output, which can include the workflow or prompt text comfy-cli echoed back. Credentials
> in a URL are masked (`user:pass@` userinfo, and the whole query string, are stripped) wherever
> the URL appears — in `args`, in `message`, and in the `stdout_tail` / `stderr_tail` captures —
> but read a file over before you attach it to an issue. The log directory is created `0700` and
> its files `0600`, so on a shared machine they are readable only by you.

## Smoke test

Validation has two complementary stages. Run both when changing the MCP
transport, shared client boundary, runner, or workflow/job behavior.

**1. Automated MCP transport and business-flow smoke (no live ComfyUI).** This
starts a real stdio child and a real loopback Streamable HTTP/ASGI server, uses
FastMCP clients for negotiation and discovery, and exercises the shared
comfy-cli path with a deterministic fake engine:

```bash
pytest -q \
  tests/test_stdio_business_flow.py \
  tests/test_remote_transport.py \
  tests/test_remote_http.py \
  tests/test_fastmcp_app.py \
  tests/test_file_transfer.py \
  tests/test_failure_log.py
```

Both transports cover input staging → `server_info` → workflow submission → job status
→ `fetch_outputs` against the same 39-tool application. The HTTP flow downloads the returned
temporary signed URL from the same listener, rejects a modified signature, and proves the upload URL is single-use;
the stdio flow proves that direct local paths and output writes stay compatible. The tests use a deterministic
fake engine (including a real temporary executable on the stdio path). HTTP coverage also checks
concurrent clients, legacy and modern protocol negotiation, native tool-error
behavior, opt-in failure observation (including upload-byte non-disclosure), scratch cleanup, and clean
ASGI/uvicorn shutdown.

**2. Live ComfyUI smoke.** This drives the actual tools through the actual
`comfy` binary into a running local ComfyUI:

```bash
./scripts/smoke.sh            # or: python -m pytest tests/e2e -m e2e
```

It needs a running same-machine ComfyUI (`COMFY_LOCAL_URL`, default
`http://127.0.0.1:8188`) and the `comfy` binary on `PATH` (or `COMFY_BIN`). Keep
`COMFY_API_KEY` set when that local ComfyUI requires it. If the binary or
ComfyUI is absent, the tests **skip** rather than fail. The checkpoint-free
`EmptyImage` → `SaveImage` round trip and `system_stats` need no model; the
additional `generate_image` regression test requires the documented SD1.5
checkpoint and fails with comfy-cli's own missing-model error when it is not
installed. These e2e tests are deselected from plain `pytest` and CI by default.

**3. Live remote Streamable HTTP MCP smoke.** When the MCP server and ComfyUI
run on another machine, point the opt-in client test at a reachable endpoint
(an SSH local-forward is fine) and name the checkpoint-free workflow on the
**server** filesystem:

```bash
COMFY_MCP_TEST_URL=http://127.0.0.1:9000/mcp \
COMFY_MCP_TEST_WORKFLOW=/absolute/server/path/tests/e2e/workflow_smoke.json \
  pytest -q tests/e2e/test_remote_mcp.py -m e2e
```

This test never imports the server application locally. It negotiates both
modern and legacy protocols, discovers the 39-tool shared application, uploads
a generated client-local PNG through the single-use PUT URL, runs the no-model
workflow, polls it, and downloads the result through the five-minute signed
URL. It consumes no cloud credits, but it deliberately leaves one tiny input
image and one smoke output in the configured ComfyUI.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev
setup, the thin-wrapper/shared-client architecture, FastMCP 4 transport
guardrails, failure-log requirements, and the mandatory targeted + full test
gates. [`AGENTS.md`](AGENTS.md) contains the complete implementation rules.
This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). To report a
vulnerability, see [`SECURITY.md`](SECURITY.md).

## License

Comfy MCP is dual-licensed (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)):

- **[GNU Affero General Public License v3.0 or later](LICENSE)** — free for use
  under the AGPL's terms, including the network-use source-disclosure
  obligation in section 13.
- **Commercial license** — for use in proprietary products or hosted services
  without AGPL obligations. Contact
  **[licensing@comfy.org](mailto:licensing@comfy.org)**.

It wraps the GPL-3.0 [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli) by
shelling out to the `comfy` binary as a separate process — no GPL code is
imported or linked. `comfy-cli` remains GPL-3.0-licensed and is distributed
separately; how its copyleft applies depends on how the programs interact.

© Comfy Org.

## Trademarks

"Comfy," "ComfyUI," and the Comfy Org name and logos — including the mark in
[`assets/logo.svg`](assets/logo.svg) — are trademarks of Comfy Org. The AGPL is
a copyright license and grants **no** rights to use those names or logos; the
commercial license grants none either unless it says so in writing. Forks and
derivative works are welcome under the license, but must not be named or branded
in a way that suggests they are official Comfy Org software or carry Comfy Org's
endorsement.

Accurate, descriptive references — tutorials, reviews, integrations — are
welcome. See the [brand guidelines](https://www.comfy.org/brand) for the full
rules and how to request permission beyond them.
