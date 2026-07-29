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

> Looking for the cloud-hosted version? See the [Comfy Cloud MCP docs](https://docs.comfy.org/agent-tools/mcp).

**What it does:**

- 🖼️ **Generate** — run a workflow JSON (API-format or UI export), or go text-prompt → image in one call.
- ⏱️ **Monitor jobs** — submit async, then wait / watch / cancel, read the failure verdict, and collect the output PNGs.
- 🔍 **Introspect your live install** — search the nodes, models, and templates your ComfyUI *actually* has (custom nodes included), not a static catalog.
- 🧩 **Build workflows** — validate a graph, edit a template's slots, and fan one workflow into variants.
- ♻️ **Manage ComfyUI** — launch / stop / restart the server, tail its logs, and stage input assets.

Each tool shells out to the `comfy` command with `--where local --json`, parses comfy-cli's
`envelope/1` output, and returns it. There is no HTTP client and **no code shared with the Comfy
Cloud MCP** — comfy-cli is the engine.

> **Status:** beta. 49 tools; core loop validated end-to-end against a live local ComfyUI
> (`server_info → run_workflow → fetch_outputs` → PNG on disk). CI runs pytest + ruff on
> Python 3.10 and 3.14.

## Table of contents

- [Prerequisites](#prerequisites)
- [When to use this server](#when-to-use-this-server)
- [Using with local LLMs (VRAM coordination)](#using-with-local-llms-vram-coordination)
- [Partner-API nodes](#partner-api-nodes)
- [Spending credits on partner models](#spending-credits-on-partner-models)
- [Templates your install can't run](#templates-your-install-cant-run)
- [Driving a remote ComfyUI](#driving-a-remote-comfyui)
- [Targeting a non-default ComfyUI address](#targeting-a-non-default-comfyui-address)
- [Install](#install)
- [Configure your AI client](#configure-your-ai-client)
- [Quickstart](#quickstart)
- [Tools](#tools)
- [Troubleshooting](#troubleshooting)
- [Failure log (opt-in)](#failure-log-opt-in)
- [Smoke test](#smoke-test)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- **Python ≥ 3.10.**
- **comfy-cli ≥ 1.13.0** on your `PATH`: `pip install 'comfy-cli>=1.13.0'`. This is the engine
  every tool wraps; the server refuses to run against an older comfy-cli with an upgrade message.
  1.13.0 is the first release carrying everything this server needs — the `comfy logs` verb, the
  `envelope/1` contract, the `comfy outdated` verb behind `server_info`'s `freshness` block, and
  the machine-readable `login_url` event `auth_login` waits for. On a comfy-cli that slips past
  the version guard (it fails open on a `--version` it can't parse — a source build, a fork) and
  lacks `outdated`, the `freshness` block degrades to
  `freshness: {"error": "freshness unavailable: …", "unsupported": true}` — update checks are
  skipped, nothing is broken, and everything else works unchanged.
  The background model download (`download_model`'s `--background` submit and the
  `download_status` / `wait_for_download` / `cancel_download` verbs) likewise ships in a release
  **after** 1.13.0; against an older comfy-cli `download_model` falls back to the previous
  blocking synchronous download, and the three polling tools report the gap the same way
  `freshness` does — `{"error": "model download-status unavailable: …", "unsupported": true}`
  rather than a raw usage dump. Nothing is lost there: such a comfy-cli never mints a
  `download_id` in the first place, and `download_model` still downloads, inline.
- **A ComfyUI workspace.** If you don't have one, `comfy-cli` can create it: `comfy install`
  sets up a ComfyUI workspace it will point at. (An existing ComfyUI checkout works too — see
  `comfy set-default <path>`.)
- **A running ComfyUI.** ComfyUI must be **started before you use the tools** — launch it with
  `comfy launch` (or, from an agent, the `launch_comfyui` tool), and confirm it is up with
  `server_info`. Nothing here starts ComfyUI implicitly.

<details>
<summary><strong>Optional environment variables</strong> (<code>COMFY_BIN</code>, <code>COMFY_API_KEY</code>, <code>COMFYUI_URL</code>, <code>COMFY_LOCAL_URL</code>, <code>COMFY_T2I_TEMPLATE</code>)</summary>

<br>

- **`COMFY_BIN` override (optional).** By default the server calls `comfy` from `PATH`. MCP
  clients launch the server with their own environment, which often does **not** include your
  shell's `PATH` — so if `comfy` lives in a virtualenv or a non-standard location, set
  `COMFY_BIN` to its absolute path (e.g. `/path/to/venv/bin/comfy`). Every client example below
  shows where it goes. Setting it is sufficient on its own — you do **not** also have to put
  that directory on the client's `PATH`. The server prepends the resolved binary's directory to
  the `PATH` it hands comfy-cli, because some comfy-cli commands (notably the background
  `launch`) re-invoke `comfy` by name and have to be able to find themselves.
- **`COMFY_API_KEY` (optional — needed only for partner-API nodes).** Workflows that use
  partner-API nodes (Seedream / Seedance / Nano Banana / Gemini / Veo / Kling / …) need a Comfy
  credential, and — exactly like `COMFY_BIN` — an MCP client launches the server with its own
  minimal environment, so a key from your shell won't reach it. Set `COMFY_API_KEY` in the
  client registration `env` block. See **[Partner-API nodes](#partner-api-nodes)** below for the
  full precedence chain; every client example shows where it goes.
- **`COMFYUI_URL` / `COMFYUI_HOST` / `COMFYUI_PORT` (optional — drive a *remote* ComfyUI).** By
  default every tool targets the local `127.0.0.1:8188`. Set `COMFYUI_URL`
  (e.g. `http://gpu-box:8188`) — or the `COMFYUI_HOST` (+ optional `COMFYUI_PORT`, default `8188`)
  pair — to point the **run / job** tools at a ComfyUI running elsewhere, e.g. a GPU box reachable
  over a private network (Tailscale). See **[Driving a remote ComfyUI](#driving-a-remote-comfyui)**
  for what is and isn't remoted. Unset ⇒ local behavior is unchanged.
- **`COMFY_LOCAL_URL` (optional — a local ComfyUI on a non-default port).** For a ComfyUI on
  *this* machine that isn't on `127.0.0.1:8188` (e.g. `:8189` because Docker Desktop's ComfyUI
  holds `:8188`). Read by comfy-cli, not by this server — it rides the environment passthrough,
  so setting it in the client `env` block re-points every tool. See
  **[Targeting a non-default ComfyUI address](#targeting-a-non-default-comfyui-address)**.
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

Local diffusion is only a good default on a machine that can actually carry it, so the server's client instructions tell your agent to read `server_info`'s `hardware` block (`os`, `arch`, `ram_bytes`, and a `gpu` object with `vendor` / `model` / `vram_bytes` / `unified_memory`) **before** the first generation and route on it. The thresholds:

| Machine | Guidance |
|---|---|
| Discrete GPU, **≥ 24 GB** VRAM | Local generation is a good default. |
| Discrete GPU, **8 GB to under 24 GB** VRAM | Images are fine (prefer current, smaller models); video will be slow or infeasible. |
| **< 8 GB** VRAM, or the user confirming there is no GPU | Don't run local diffusion. Use partner nodes (plain web calls, fine on any machine) or the Comfy Cloud MCP if your client has it connected. |
| **Apple Silicon**, **≥ 32 GB** unified memory | Images are OK. Video on the Apple GPU is not recommended — time estimates are unreliable and thermals suffer. |
| **Apple Silicon**, under 32 GB unified memory | Same as the no-GPU row above — go partner/cloud rather than local. |

The discrete-GPU rows are written for NVIDIA but apply to an AMD or Intel card on a ROCm/XPU build too — the VRAM number is what matters. The no-local-video rule is an *Apple GPU* rule rather than a Mac rule: an Intel Mac with a discrete card follows the discrete-GPU rows.

The instructions walk these as an ordered procedure, because several of the checks only make sense in sequence:

1. **Is the work even local?** `hardware` describes the machine *this server* runs on, and that is where most tools execute. A `comfy_target` block ([Driving a remote ComfyUI](#driving-a-remote-comfyui)) diverts only `run_workflow` and the queue/`jobs` tools — `generate_image`, `run_template` and the rest stay local, so the thresholds still govern them. It counts as another machine only when its `host` is neither loopback (anything in `127.0.0.0/8`, `localhost`, IPv6 `::1`) nor this host's own address, and a malformed config produces an error-shaped `{error, note}` block that resolves no remote at all. Nothing the server returns carries the local hostname or interface addresses, so a `host` the agent can't place is a question for you rather than a guess — a hostname or LAN IP can be this same machine, and a loopback host can be a tunnel to a remote GPU. `COMFY_LOCAL_URL` is a second signal worth checking: it repoints comfy-cli without producing a `comfy_target` block.
2. **Get a memory figure.** The sizes are bytes (`ram_bytes`, `gpu.vram_bytes`) and the divisor gives GiB, while drivers report under the advertised size — a consumer 24 GB card reads 23.99, an ECC/reserving datacenter card (A10, L4) about 22.3 — so a *small* shortfall, within ~10% of a nominal size, reads as that nominal capacity. A gap wider than that is not driver overhead and is taken at face value instead: on a MIG/vGPU partition the model string names the whole card while `vram_bytes` is the slice you actually get, and rounding a 6 GB A100 slice up into the ≥ 24 GB band would OOM the run. On Apple Silicon `gpu.vram_bytes` is `null` (with `gpu.unified_memory` true) and the figure is `ram_bytes` — an Apple-only substitution.
3. **If the figure is missing, ask.** A `null` **or zero** `vram_bytes` on any **non-Apple** GPU (a discrete card comfy-cli can't size, but also a non-Apple unified part like a Jetson/Grace board or a Strix Halo APU), a missing `gpu` object, or a missing/zero `ram_bytes` on the Apple path all mean **unknown**, not "no GPU" — the agent asks rather than stranding a machine that has one. The "no GPU" verdict is reserved for a *confirmed* absence, and the only thing that confirms one is your own answer: no `hardware` payload encodes it, because a null or missing `gpu` is unknown by this same step. Nothing in this repo probes hardware, and the instructions tell the agent not to shell out either: a probe runs on a path this server can neither bound nor audit.
4. **Route on the figure**, then **redirect rather than dead-end** when the answer is "not on this machine". A figure that came from your answer rather than the payload routes on whichever row fits the machine — the unified-memory row on an Apple Silicon Mac, the VRAM rows otherwise, which is what covers the non-Apple unified-memory boards that have no row of their own.

"No local video on a Mac" is about the Apple GPU, not about video as such: `API`-tagged video templates (`search_templates(tag="API", type="video")` — both filters, since neither alone isolates partner-run video, and the compact rows omit `tags`) and `emit_partner_workflow` run the model on partner infrastructure, so they work on any machine. See **[Partner-API nodes](#partner-api-nodes)**.

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
5. **The client reloads its LLM** afterwards, again through its own runtime. Ollama, LM Studio and a sleep-idle `llama-server` all reload on demand, so for those "reload" is just the next request; a single-model `llama-server` stopped in step 2 has to be started again.

**Why steps 2 and 5 cannot live in this MCP server.** This server is a **stdio subprocess of your MCP client** — it holds no handle on whatever LLM runtime that client is using, is not told which one it is, and has no business reaching into a process it does not own. Reaching one anyway would also breach the [thin-wrapper rule](AGENTS.md): every tool here is a `comfy` passthrough, and there is no `comfy` subcommand for "unload someone else's model". The deeper reason is step 5: the *model* that was unloaded cannot ask for itself back, so something still running has to sequence unload → run → reload. Where the LLM's own runtime can do that (Ollama's on-demand load, LM Studio's JIT, `llama-server --sleep-idle-seconds`) it should — that is the least-coordination option and it needs nothing from this server. Otherwise the client, or the orchestrator driving it, is the only participant present throughout. Either way the split is structural rather than a missing feature: this server owns reading and freeing **ComfyUI's** memory, and the client owns its **own** model's lifecycle.

A note on scope: `free_memory()` asks ComfyUI to release *its* models. It does nothing about VRAM held by an LLM runtime, a browser, or another process — if `system_stats()` still shows little free VRAM after a `free_memory()` call, the memory is probably someone else's and step 2 is what reclaims it.

**Read that signal against the lag, not instantly.** The free applies on the queue worker's next iteration, so on a *busy* server an immediate re-read legitimately shows no change while the VRAM is still ComfyUI's — the request simply has not been serviced yet. Before concluding the memory belongs to another process, either wait for the current job to finish (`get_queue()` shows whether one is running) or re-poll `system_stats()` a few times over a few seconds. Only a number that stays flat on an *idle* server means the holder is someone else.

A second caveat: `system_stats()` and `free_memory()` are **not** redirected by `COMFYUI_URL` / `COMFYUI_HOST` — they always describe and act on whichever ComfyUI comfy-cli itself targets, because `comfy system-stats` and `comfy free` take no `--host` / `--port`. With a remote ComfyUI configured, `run_workflow` submits *there* while these two read and free the *local* install, so this recipe applies to a local-ComfyUI setup. Don't gate a remote run on it.

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
in the client registration `env` block (shown in every example below). If a run fails with
`partner_node_requires_credential`, the error now carries comfy-cli's hint verbatim, including
the `comfy auth set comfy-cloud-api-key --key …` fallback and the list of offending nodes; the
server also retries a transient credential failure briefly before surfacing it.

## Spending credits on partner models

`partner_generate` is the one tool whose *whole purpose* is to spend: it wraps `comfy generate
<model>`, which calls a hosted partner API and **spends your Comfy credits**. So every call is
confirmed with you first.

The other tools execute on **your** machine, and on their own they cost nothing — but that is a
property of the tool, not a guarantee about the workflow you hand it. A workflow run through
`run_workflow` / `generate_image` can itself contain the partner-API nodes described just above
(Seedream, Veo, Kling, …), or any other node that bills a hosted service, and **those still spend
your credits** — they bill through the workflow, below this server, with no confirmation prompt.
Check what a workflow contains before running one you did not build.

`emit_partner_workflow` sits on the free side of that line for the same reason: it only *writes* a
graph containing a partner API node, never calls the partner, and so has no confirmation prompt.
The graph it writes is exactly one of the workflows the paragraph above is warning about — running
it with `run_workflow` bills the partner node.

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
  comfy-cli, so the URL must be plain `http://` with **no base path**: an `https://` scheme or a
  reverse-proxy path (`http://gpu-box:8188/comfyui`) is rejected rather than silently downgraded to
  http / dropped. Front a TLS/base-path proxy locally if you need one.
- **`COMFYUI_HOST`** (+ optional **`COMFYUI_PORT`**, default `8188`) — e.g. `COMFYUI_HOST=gpu-box`.
  A port without a host (setting only `COMFYUI_PORT`) is rejected — set the host too.

Set it in the client registration `env` block (same place as `COMFY_BIN`). With nothing set,
behavior is unchanged (local `127.0.0.1:8188`).

When configured, the server forwards `--host` / `--port` to comfy-cli for exactly the verbs that
accept them — `comfy run` and `comfy jobs …` — so the **run and job tools** target the remote:
`run_workflow`, `job_status`, `wait_for_job`, `watch_job`, `cancel_job`, `get_queue`. `server_info`
reports the configured target under a `comfy_target` block.

**Not remoted (this repo is a thin wrapper and never opens its own socket):**

- **Lifecycle** (`launch_comfyui`, `stop_comfyui`, `restart_comfyui`, `update_comfyui`,
  `switch_comfyui_version`, `get_logs`) — these manage a **local** ComfyUI process/install and stay
  local-only; they cannot start/stop, update, version-switch, or read logs from a remote box. Start
  and update ComfyUI on the remote host yourself.
- **Output download** (`fetch_outputs` → `comfy download`) and `search_templates` / `search_models`
  / `generate_image` / `run_template` / `partner_generate` — this server forwards **no**
  `--host`/`--port` to these verbs (most of them accept none at all), so they run
  against comfy-cli's local default. Against a remote target, prefer `run_workflow(wait=True)` /
  `job_status` (which return the remote job's `/view` output URLs) to retrieve results.
- **Discovery / validation** (`search_nodes`, `get_node`, `validate_workflow`) — their comfy-cli
  verbs *do* accept `--host`/`--port`, but this version forwards only to the run/job tools (the
  ticket's scope), so they still target local. Remoting them is a planned follow-up; until then,
  author/validate against a local ComfyUI matching the remote's node set.
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
    "comfy-local": {
      "command": "comfy-local-mcp",
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

**Requires comfy-cli ≥ 1.13.0.** `COMFY_LOCAL_URL` landed after the 1.12.0 release and first
shipped in 1.13.0 — which is also this server's enforced floor, so every *published* comfy-cli
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

> **Use this *or* `COMFYUI_URL`, not both.** [`COMFYUI_URL`/`COMFYUI_HOST`](#driving-a-remote-comfyui)
> makes this server forward explicit `--host`/`--port` flags for `comfy run` / `comfy jobs`, and an
> explicit flag **outranks** `COMFY_LOCAL_URL` — so with both set the run/job tools would follow
> `COMFYUI_URL` while every other verb followed `COMFY_LOCAL_URL`. For a non-default address on
> *this* machine prefer `COMFY_LOCAL_URL` alone: it also covers the verbs that accept no
> `--host`/`--port` (`comfy env`, templates, models, download), which `COMFYUI_URL` cannot reach.

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

> **On macOS, keep ComfyUI out of `~/Documents`, `~/Desktop` and `~/Downloads`** — or grant your
> client Full Disk Access. macOS blocks apps (and everything they launch) from reading those
> folders, so an install there fails with `Operation not permitted` before anything runs. See
> [Troubleshooting](#troubleshooting).

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
   pip install 'comfy-cli>=1.13.0'  # the engine (>= 1.13.0 required)
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
   `search_templates`, `fetch_template` to write a runnable JSON, and run that — and
   `fetch_template` tells it up front if [your install can't run that
   template](#templates-your-install-cant-run) yet.

**Where the images land.** ComfyUI writes generated files into your ComfyUI **workspace's
`output/` directory** (part of the workspace `comfy install` created). On top of that,
`fetch_outputs(prompt_id, out_dir)` **copies** a finished job's outputs into any directory you
name — so telling the agent "save them to `./outputs`" puts a copy right where you asked while
the originals stay in the ComfyUI workspace.

## Tools

49 tools, grouped below by what they do. Every tool runs `comfy` with the global
`--json --where local` flags, unwraps comfy-cli's `envelope/1`, and returns its `data`.

**Argument naming** is uniform, so an agent never has to guess it (the server's handshake
instructions say the same thing): an input workflow file is always `workflow_path`, an output
file is `out_path`, an output directory is `out_dir`, a registry lookup key is `name`, and a job
handle is `prompt_id`.

### Run and monitor

| Tool | Wraps | What it does |
|---|---|---|
| `run_workflow(workflow_path, wait=True, timeout_seconds=600.0)` | `comfy run --workflow <path> [--wait]` | Run a workflow JSON; `wait=False` submits async and returns a `prompt_id`. |
| `generate_image(prompt, checkpoint=None, wait=True, timeout_seconds=600.0)` | `comfy run-template default --param=6.text=<prompt> [--param=ckpt_name=<ckpt>]` | Text prompt → image in one call, with no hand-assembled workflow needed: it runs ComfyUI's own default SD1.5 text-to-image gallery template through the same verb (and the same local run path) as `run_template`. Free and fully local — nothing here spends credits. Retarget it with [`COMFY_T2I_TEMPLATE` and its slot-key companions](#prerequisites). Same envelope shape as `run_workflow` (`prompt_id` + outputs); the fast on-ramp. |
| `partner_generate(model, params=None, confirm_spend=False, out_path=None, timeout_seconds=600.0)` | `comfy generate <model> [--param=value…] [--download=<path>] [--timeout=<s>] [--yes]` | Run a hosted **partner** model (Flux / Ideogram / DALL·E / Recraft / …). **Spends Comfy credits**, unlike the local `run_workflow` / `generate_image` paths. Every call confirms the spend with you first — see [Spending credits](#spending-credits-on-partner-models) below. Runs **entirely on the partner's infrastructure** — your local ComfyUI is never in the execution path; use `emit_partner_workflow` below for the path where it is. `params` are the model's own schema-driven inputs, forwarded verbatim — `list_partner_models()` gives you the `model` aliases and `partner_model_schema(model)` the parameter list, so neither needs a terminal. `out_path` becomes comfy-cli's `--download` and is a save-path *template*: a plain path names the file, `{request_id}` / `{index}` / `{ext}` are substituted per output, and a trailing slash means "a default filename in this directory". `timeout_seconds` becomes comfy-cli's own `--timeout` so the engine — not a parent kill — owns the deadline on a job the partner may already have charged for. The result carries comfy-cli's printed text as `message` and, when it named the files it wrote, the resolved paths as `saved_paths` — so a caller reads the destination as data instead of scraping prose that rich may have wrapped mid-filename. |
| `emit_partner_workflow(model, out_path, params=None)` | `comfy generate <model> [--param=value…] --emit-workflow=<path>` | Write a runnable workflow JSON that drives the partner model's **API node** instead of calling the proxy, so **your own ComfyUI executes the partner model** (the other way there is an existing `API`-tagged gallery template via `search_templates` / `run_template`; this is the path from a model *alias*). Chain it: `emit_partner_workflow` → `run_workflow` → `fetch_outputs` (the three stay separate so the graph can be inspected, edited with `set_workflow_slot`, re-run, or embedded in a bigger pipeline). Calls no partner API, needs no API key, and **spends nothing**, so unlike `partner_generate` it has no `confirm_spend` argument and raises no confirmation prompt — *running* the emitted graph is what bills the partner node. **Coverage is narrow:** comfy-cli maps only `flux-2`, `flux-pro`, `kling-i2v`, `nano-banana` and `seedance` to a node class, a small subset of `list_partner_models()`; every other model reaches its partner through the proxy only, so send those to `partner_generate`. An unsupported model raises with comfy-cli's own `emit_workflow_failed` message, which names the supported set for the comfy-cli you actually have installed. Returns comfy-cli's envelope data — `{"out", "model", "nodes"}`. |
| `run_template(name, params=None, confirm_spend=False, wait=True, timeout_seconds=600.0, ctx=None)` | `comfy run-template <name> [--param=KEY=VALUE…] [--timeout=<s>] [--allow-spend] [--async]` | One-command template run — fetch the gallery template, fill its parameterized slots, and run it on local ComfyUI (the one-shot alternative to `fetch_template` → `run_workflow`). `params` are `{slot: value}` (slot address `6.text` or name `prompt`), JSON-encoded so types round-trip. Most templates are free OSS graphs; one embedding partner (paid) nodes spends credits and fails closed unless `confirm_spend=True` unlocks it — and on an elicitation-capable client that asks **you** per call before anything runs (same posture as `partner_generate`; a default, free run is never prompted, and `comfy generate consent always` does not apply to this verb). No capability probe is needed here (unlike `partner_generate`): this verb's gate ships inside the verb itself, so a comfy-cli that has `run-template` has the gate. `wait=True` (the default) streams the run's live progress as MCP progress notifications, the same way `run_workflow` / `watch_job` do, so a long template run is not a silent block; `wait=False` submits `--async` and returns a `prompt_id`. comfy-cli's `--timeout` for this verb is *per-event*, not a whole-run deadline, so `timeout_seconds` is forwarded only to tighten it below the engine's 120s default — prefer `wait=False` over a large `timeout_seconds` for long runs. |
| `job_status(prompt_id)` | `comfy jobs status <prompt_id>` | Poll a submitted job's status + outputs. |
| `wait_for_job(prompt_id, timeout_seconds=25.0)` | `comfy jobs status <prompt_id>` (polled) | Bounded wait until a job reaches a terminal status; returns a `{"timed_out": True, …}` payload on expiry. Chain several. |
| `watch_job(prompt_id, timeout_seconds=600.0)` | `comfy jobs watch <prompt_id>` (streamed) | Tail an async-submitted job's live execution, streaming progress notifications; bounded, returns a `{"timed_out": True, …}` payload on expiry. Streaming counterpart to `wait_for_job`. |
| `get_execution_error(prompt_id)` | `comfy jobs status <prompt_id>` | Compact failure verdict for a failed run — the failing node, `exception_type`/`exception_message`, and a bounded traceback tail — so an agent can self-repair; returns `error: None` on a healthy prompt. |
| `cancel_job(prompt_id)` | `comfy jobs cancel <prompt_id>` | Cancel a queued or running job. |
| `get_queue()` | `comfy jobs ls` | List known **local** jobs with status (pending/running/completed); cloud-tracked rows are filtered out. |
| `fetch_outputs(prompt_id, out_dir, url_only=False, inline_images=False)` | `comfy download <prompt_id> --where local -o <out_dir> [--url-only]` | Write a finished local job's outputs into `out_dir`; `url_only=True` emits the output URLs without copying bytes; `inline_images=True` also returns the copied images as inline MCP image content so the agent can see them without a second read. |

### Resource management

| Tool | Wraps | What it does |
|---|---|---|
| `system_stats()` | `comfy system-stats` | Read the live local ComfyUI's VRAM per device and system RAM. ComfyUI's whole `/system_stats` payload is forwarded **unmodified**, so treat it as a passthrough, not a fixed schema: a `devices` list plus a `system` dict, whose keys are whatever that ComfyUI reports. The ones this server's guidance reads are per-device `vram_free` / `vram_total` (byte counts, alongside e.g. `name`, `type`, `index`, `torch_vram_free`) and `system.ram_free` / `ram_total` / `comfyui_version` — examples, not an exhaustive list. Nothing is filtered, so the `system` block also carries ComfyUI's `python_version` and `argv` (its full launch command line), which reach the model's context verbatim. Call it before a heavy `run_workflow` / `run_template` to decide whether to free memory first, and again afterwards to confirm the headroom landed. Read-only. Needs a running ComfyUI (the numbers come from the server), and unlike the run/job tools it is **not** diverted by `COMFYUI_URL`/`COMFYUI_HOST` — `comfy system-stats` takes no `--host`/`--port`. |
| `free_memory(unload_models=True, free_memory=None)` | `comfy free [--unload-models\|--no-unload-models] [--free-memory]` | Ask ComfyUI to unload models from VRAM and reset its executor cache (`POST /free`). `free_memory=None` means **follow `unload_models`**, so the default call requests both — maximum headroom, and a deliberate divergence from comfy-cli's `--free-memory`, which defaults to off; pass `free_memory=False` for the CLI's lighter unload that keeps cached executor state. **The cache reset can't be had without the unload:** ComfyUI's worker resolves the pair as `flags.get("unload_models", free_memory)` and its `/free` handler only records `unload_models` when true, so `unload_models=False, free_memory=True` would unload everything — that pair is rejected rather than sent. `unload_models=False` therefore asks ComfyUI to do nothing; it's a deliberate no-op kept for symmetry with the CLI. **Not immediate and never destructive:** ComfyUI applies the request when its queue worker next iterates — immediate if idle, after the current job if busy — and it does **not** interrupt a running job, so it cannot be used to stop one (`cancel_job` does that). Returns comfy-cli's acknowledgement of what was *requested*, not a measurement; read `system_stats` afterwards to confirm. Also **not** diverted by `COMFYUI_URL`/`COMFYUI_HOST`. See [Using with local LLMs](#using-with-local-llms-vram-coordination). |

### Diagnostics

| Tool | Wraps | What it does |
|---|---|---|
| `server_info()` | `comfy env` + `comfy outdated` | Is a local ComfyUI running, where, and which workspace. **Call first.** Passes through comfy-cli's `hardware` block (GPU vendor/model, VRAM or unified memory, total RAM) when the installed comfy-cli reports one — the signal behind [When to use this server](#when-to-use-this-server). Also attaches a `freshness` block (`comfy outdated`): installed-vs-latest for ComfyUI core and each custom node pack, so a stale install is flagged before it masquerades as a missing model/node. On a comfy-cli without the `outdated` verb the block degrades to `freshness: {"error": "freshness unavailable: …", "unsupported": true}` (a benign capability gap — skip staleness advice, nothing is broken); on any other probe failure such as a network error it degrades to `freshness: {"error": …}` carrying the real reason. Either way the tool itself still succeeds. Reports the configured remote under a `comfy_target` block when `COMFYUI_URL`/`COMFYUI_HOST` is set (see [Driving a remote ComfyUI](#driving-a-remote-comfyui)). |
| `auth_status()` | `comfy cloud whoami` | Comfy Cloud credential status for partner-API nodes (read-only, never returns secrets). Adds a local `registration_env_key_present` bool for the `COMFY_API_KEY` registration-env slot whoami can't see. |
| `auth_login()` | `comfy cloud login --no-browser --timeout 600` | Start Comfy Cloud sign-in and return `{"status": "awaiting_browser", "login_url": …, "expires_in_s": …}` — the URL for the **user** to open, so an agent can get them signed in instead of telling them to run the CLI by hand. Returns as soon as comfy-cli emits the URL; the sign-in keeps running in the background (comfy-cli owns the OAuth flow and the loopback callback, so no OAuth logic lives here). Confirm the result with `auth_status`. Only one sign-in at a time: calling it again while one is pending re-reports the same URL without spawning a second flow, and calling it after the flow ended reports `completed` / `failed` once and then clears. Never returns tokens. |
| `which()` | `comfy which` | Which ComfyUI install/workspace comfy-cli currently targets (a lighter answer than `server_info`). |
| `get_logs(tail=200, port=None)` | `comfy logs --tail <tail> [--port <port>]` | Tail the background ComfyUI's captured log (`<workspace>/user/comfyui_<port>.log`) — closes the debugging loop after a detached `launch_comfyui`. Returns `{lines, path, truncated}`; a missing log file returns `{"error": "no_log_file", …}` rather than raising, and on a newer comfy-cli that message lists every candidate path checked. Pass `port` when several instances/ports have run, or after a crash, to force `user/comfyui_<port>.log` resolution. A newer comfy-cli also returns `source` / `port_mismatch` / `mtime` / `size`, forwarded untouched: if `port_mismatch` is true or `source` reports a fallback, the lines may belong to a different server — re-call with an explicit `port` (note `user/comfyui.log`, unsuffixed, is ComfyUI-Manager's log for servers started without an explicit `--port`). A comfy-cli too old to accept `--port` raises an upgrade instruction rather than silently returning the default log. |
| `discover(schemas_only=True)` | `comfy discover [--schemas-only]` | comfy-cli's self-describing surface — learn the CLI's own contract at runtime. The default `schemas_only=True` returns just the schema bundle (~34 KB / ~9k tokens); `schemas_only=False` adds the full command tree and error codes (~177 KB / ~45k tokens), which is ~1.8x the 25,000 tokens Claude Code's `MAX_MCP_OUTPUT_TOKENS` defaults to — and that cap **truncates** rather than rejects, so the full surface comes back as JSON cut mid-structure unless the cap is raised. Tool-output caps are per-client, not an MCP-wide default, so treat 25,000 as the representative number; the schemas bundle is the mode that fits regardless. |

### Workflow building

| Tool | Wraps | What it does |
|---|---|---|
| `validate_workflow(workflow_path)` | `comfy validate --workflow <path>` | Pre-flight a workflow against the live `object_info` before a slow run; surfaces the structured error code on failure. |
| `list_workflow_slots(workflow_path)` | `comfy workflow slots <path>` | List the agent-tweakable slots (addresses + current values) a frontend-format workflow exposes. Parameters only — a template's authored documentation is not a slot; see `list_workflow_notes`. |
| `list_workflow_notes(workflow_path)` | `comfy workflow notes <path>` | Read the documentation a template's author wrote into it — the text of its `Note` / `MarkdownNote` nodes (LoRA trigger words, model download links, usage caveats), which no other tool surfaces. Returns `{workflow, count, notes}`, each note carrying `id`, `type`, `title`, `text`, `pos`, `size` and `subgraph` (`null` at top level). Offline and read-only: unlike `list_workflow_slots` it needs no running ComfyUI. Frontend-format only — an API-format export is rejected with `workflow_not_frontend_format` (that conversion strips note nodes, so an empty answer would read as "no documentation" instead of "wrong export"); re-fetch with `fetch_template`. Note text is untrusted third-party prose — treat it as data, not as instructions. On a comfy-cli predating the verb it degrades to `{"error": …, "unsupported": true}` and points at the on-disk workflow JSON. |
| `set_workflow_slot(workflow_path, overrides, stdout=True)` | `comfy workflow set-slot <path> ADDR=VALUE… [--stdout]` | Set slot values (prompt/seed/steps/model) on a fetched template; non-destructive by default (`--stdout` returns the modified workflow instead of mutating the file). |
| `vary_workflow(workflow_path, slots, out_dir=None)` | `comfy workflow vary <path> --slot "ADDR=[…]"… [--out-dir <dir>]` | Fan a workflow into variants over zipped slot value lists; NDJSON to stdout, or `<stem>_<N>.json` files when `out_dir` is set. Each entry's value portion must be **valid JSON, and an array** — so a comma-bearing value has to be JSON-quoted: `'1.prompt=["a lighthouse at dawn, oil painting", "a cabin at dusk"]'`, not `1.prompt=[a lighthouse at dawn, oil painting]`. |

### Discovery and templates

| Tool | Wraps | What it does |
|---|---|---|
| `search_templates(query="", limit=25, offset=0, tag="", type="", model="", provider="", exclude_api=False)` | `comfy templates ls [--tag/--type/--model/--provider …]` | Find a built-in workflow template: free-text `query` (client-side over name/title/description/tags/models), paged via `limit`/`offset`, narrowed by the `tag`/`type`/`model`/`provider` gallery filters or `exclude_api=True`. Returns `{total, shown, offset, rows:[{name,title,description,output_type}]}`. |
| `get_template(name, check_local=True)` | `comfy templates show <name>` (+ `comfy validate`) | Show one template's details/schema before fetching it, plus a `local_check` block cross-checking its graph against the live `object_info` of your install — see [Templates your install can't run](#templates-your-install-cant-run). `check_local=False` skips the check (metadata only, one call). |
| `fetch_template(name, out_path, check_local=True)` | `comfy templates fetch <name> --out <path>` (+ `comfy validate`) | Write a template's runnable workflow JSON to `out_path`; returns `{path, local_check}` — `path` is the absolute path for `run_workflow`, `local_check` is the same cross-check run on the file just written. The file is written either way. |
| `search_nodes(query)` | `comfy nodes search <query>` | Find node classes in the **live local** `object_info` (includes installed custom nodes). |
| `get_node(name)` | `comfy nodes show <ClassName>` | Full input/output schema for one node class — what you need to author/repair a graph. |
| `list_nodes(produces="", accepts="", category="", pack="", label="")` | `comfy nodes ls [--produces/--accepts/--category/--pack/--label …]` | List node classes, filtered by output/input type, category, pack, or label; bare call lists all. Reads the **live install**. |
| `nodes_upstream(name, limit=None)` | `comfy nodes upstream <name> [--limit N]` | Nodes whose outputs can feed `<name>`'s inputs ("what wires INTO this?"). Reads the **live install**. |
| `nodes_downstream(name, limit=None)` | `comfy nodes downstream <name> [--limit N]` | Nodes that accept `<name>`'s output types ("what does this wire INTO?"). Reads the **live install**. |
| `nodes_path(from_type, to_type, max_depth=6, max_paths=10)` | `comfy nodes path <FROM> <TO> --max-depth N --max-paths N` | Node chains routing a value between two connection types (e.g. `MODEL` → `IMAGE`). Reads the **live install**. |
| `nodes_types()` | `comfy nodes types` | All connection types (`MODEL`, `IMAGE`, …) ranked by connectivity. Reads the **live install**. |
| `nodes_categories()` | `comfy nodes categories` | The node category tree. Reads the **live install**. |
| `search_models(query="", folder="")` | `comfy models search` / `models list-folder <folder>` / `models list-folders` | List/search model files on disk. **Local:** filenames only, no cloud enrichment. |
| `list_partner_models(style="", partner="", query="", limit=100, offset=0)` | `comfy generate list [--style S] [--partner P] [--query Q]` | The catalog of hosted **partner** models `partner_generate` can run — the only place that list exists (nothing in `discover` / `search_nodes` / `search_templates` carries the partner aliases). One record per model: `alias` (what you pass as `model`), `id`, `partner`, `category` (the model's style, and the axis `style` filters on — `text-to-image`, `image-edit`, `image-to-image`, `text-to-video`, `image-to-video`, `video-extend`, `controlnet`, `inpaint`, `outpaint`, `upscale`, `background`, `lipsync`, `vectorize` as this is written; comfy-cli owns that set, so read it off an unfiltered call), `mode` (`sync`/`async`, the partner's protocol — `partner_generate` waits either way) and the model's full, untruncated `summary`. Filters are forwarded to comfy-cli: `style` is exact and **case-sensitive**, `partner` exact and case-insensitive, `query` a substring over `id` + `summary`. `limit` (default 100, capped at 200) / `offset` page the result (`{total, shown, offset, filters, models}`) so a growing catalog can't trip the client's tool-output cap; 52 models as this is written, so the default returns all of them — check `shown` against `total` rather than assuming that stays true. |
| `partner_model_schema(model)` | `comfy generate schema <model>` | One partner model's callable parameters — what to put in `partner_generate`'s `params`. Returns `{model, id, partner, category, summary, mode, polling, content_type, params, example}`, where each `params` record carries `name`, `type` (`string`/`integer`/`number`/`boolean`/`enum`/`object`/`array`/`binary` — `binary` is a local file path comfy-cli uploads or inlines for you), `required`, `default`, `enum` and the spec's own `description`. Reads the spec only: no partner call, no key, no spend. |

### Lifecycle and assets

| Tool | Wraps | What it does |
|---|---|---|
| `launch_comfyui(extra_args=None)` | `comfy launch --background [-- <extras>]` | Start the local ComfyUI detached; forwards `extra_args` to ComfyUI. |
| `stop_comfyui()` | `comfy stop` | Stop the ComfyUI that comfy-cli launched (only its own recorded pid). |
| `restart_comfyui(extra_args=None)` | `comfy stop` then `comfy launch --background [-- <extras>]` | Stop-then-launch the local ComfyUI (best-effort stop); forwards `extra_args` to the fresh server. Handy for relaunching with different flags. |
| `update_comfyui(target="comfy")` | `comfy update <all\|comfy\|cli>` | Update the local install: `"comfy"` = ComfyUI core, `"all"` = the installed custom node packs, `"cli"` = comfy-cli itself. This is what `server_info`'s `freshness` block points at when it reports a stale install. Slow (a core update re-installs requirements; 30-minute timeout) and the updated code only takes effect after a `restart_comfyui`. Any other `target` is rejected before comfy-cli is invoked, and a second update requested while one is still running is refused rather than run in parallel (concurrent `git`/`pip` against one workspace can leave it half-installed). |
| `switch_comfyui_version(version, confirm_switch=False)` | `comfy update comfy --version <version>` | Move the local ComfyUI install to a **specific** version — `"nightly"`, `"latest"`, or a release like `"0.24.0"` / `"v0.24.0"` — so you can roll **back** to reproduce or rule out a regression (`update_comfyui` only ever moves forward to the latest). **Destructive:** the engine stashes any uncommitted changes in the ComfyUI checkout, moves it to that version, and reinstalls that version's Python dependencies (minutes, not seconds; 15-minute timeout). **The USER is asked to confirm every call** — on a client that supports MCP elicitation a prompt naming exactly that is raised, and a decline cancels with nothing changed; on a client that cannot show prompts the call errors unless `confirm_switch=True`, which an agent may pass **only** when the user has actually agreed. That prompt is raised even when `confirm_switch=True` is passed, so a host's "always allow this tool" toggle is not standing authority over the install. It **refuses while a local ComfyUI is running** (reinstalling under a live process can leave it serving half-replaced code) — checked both before the prompt and again immediately before the switch, since the prompt may sit unanswered for minutes, and fail-closed, so a `comfy env` this server cannot read is refused rather than read as "stopped" — and it does **not** restart anything — the flow is `stop_comfyui` → `switch_comfyui_version` → `launch_comfyui` → `server_info` to confirm what came up. Returns `{switched_to, result, restart_required: true}`. A malformed version is rejected before comfy-cli is invoked; a comfy-cli whose `comfy update` predates `--version` surfaces as an "upgrade comfy-cli" error rather than a raw usage dump; and it shares `update_comfyui`'s one-at-a-time lock. |
| `upload_file(paths, overwrite=False)` | `comfy upload <files...> [--overwrite]` | Stage source images/masks into the local `input` dir (unlocks img2img / inpaint). |
| `download_model(url, relative_path=None, filename=None, wait=True, timeout_seconds=110.0)` | `comfy model download --url <url> [--relative-path <path>] [--filename <name>] --background` | Download a model file by direct URL (HuggingFace / CivitAI) into the local models dir; download-by-URL only, not a hub search. The transfer is **submitted** to comfy-cli's background worker and returns a `download_id`, so a multi-GB checkpoint no longer holds the MCP request open past the client's deadline: `wait=True` (default) polls that id for you within a bounded budget and returns `{"timed_out": True, "download_id": …}` — not an error — if the transfer is still running, while a `failed` / `cancelled` download raises with comfy-cli's own error. On that path `timeout_seconds` is the **end-to-end** budget for the whole call, submit included, so the submit and the poll cannot add up past the client deadline the 110s default is chosen to sit under. `wait=False` returns the submit payload immediately and keeps the submit's own fixed budget. The file is written straight to its final path as it transfers, so a filesystem / `search_models` check mid-flight sees a present-but-incomplete file — `download_status` is the source of truth. `relative_path` resolves from the workspace root and must be the models dir or a subfolder of it — `models`, `models/loras` (a bare `loras` is rejected, not assumed); sibling dirs like `custom_nodes/…`, `input`, `output` are refused. Use `/` as the separator on every host, Windows included. Against a comfy-cli too old to know `--background` (releases up to 1.13.0) it falls back to the previous blocking synchronous download — which has no id to detach or poll, so it blocks even on `wait=False` and marks the payload it returns with `background_unsupported: true` to say the flag could not be honored. |
| `download_status(download_id)` | `comfy model download-status <download_id>` | Progress of one background download: `status`, `completed_bytes` / `total_bytes` / `percent`, `elapsed_seconds`, `dest`, and `error`. The only proof a model is complete and loadable. On a comfy-cli without the verb, returns `{"error": …, "unsupported": true}` instead of a raw usage dump. |
| `wait_for_download(download_id, timeout_seconds=25.0)` | `comfy model download-status <download_id>` (polled) | Bounded wait until a download reaches a terminal state (completed / failed / cancelled); returns a `{"timed_out": True, …}` payload on expiry. Chain several — the `wait_for_job` shape, for transfers. Degrades to `unsupported: true` on a comfy-cli without the verb. |
| `cancel_download(download_id)` | `comfy model download-cancel <download_id>` | Stop a running background download and remove its partial file. Degrades to `unsupported: true` on a comfy-cli without the verb. |

Node introspection (`search_nodes` / `get_node` / `list_nodes` / `nodes_upstream` /
`nodes_downstream` / `nodes_path` / `nodes_types` / `nodes_categories`) and `search_models`
read the **user's live install** (custom nodes included), not a static catalog — that's the
local differentiator from the cloud MCP's equivalents. The graph-wiring verbs (`upstream` /
`downstream` / `path`) are what an agent authoring a workflow uses to find compatible nodes.

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
timeout) are exactly the ones that leave the least behind. Set **`COMFY_LOCAL_MCP_DEBUG_LOG`** and
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
| macOS | `~/Library/Application Support/comfy-local-mcp/failures.jsonl` |
| Windows | `~/AppData/Local/comfy-local-mcp/failures.jsonl` |
| Linux / other | `~/.config/comfy-local-mcp/failures.jsonl` |

Each line records the failure `kind` (`error_envelope`, `no_json`, `timeout`, `binary_missing`,
`schema_mismatch`), a UTC `ts`, the comfy-cli `args`, its `exit_code` and the envelope's
`error_code`, the message you saw in your client, and up to 4,000 characters of `stdout_tail` /
`stderr_tail` — deliberately more output than an error message can carry:

```console
$ COMFY_LOCAL_MCP_DEBUG_LOG=1 …            # in your MCP client config's env block
$ jq -r 'select(.kind == "timeout") | .ts + "  " + (.args | join(" "))' \
    ~/Library/Application\ Support/comfy-local-mcp/failures.jsonl
```

The file rotates itself: 1 MiB per file with two older generations kept (`failures.jsonl.1`,
`failures.jsonl.2`), so it stops growing at roughly 3 MiB no matter how long you leave it on.
Successful calls are never recorded, and nothing is ever transmitted anywhere — the log is local,
full stop.

> **Privacy — review before sharing.** The log contains local file paths and comfy-cli's own
> command output, which can include the workflow or prompt text comfy-cli echoed back. Credentials
> in a URL are masked (`user:pass@` userinfo, and the whole query string, are stripped) wherever
> the URL appears — in `args`, in `message`, and in the `stdout_tail` / `stderr_tail` captures —
> but read a file over before you attach it to an issue. The log directory is created `0700` and
> its files `0600`, so on a shared machine they are readable only by you.

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
