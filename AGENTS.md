# Agent Guidelines — comfy-mcp

A short guide for AI agents (and humans). **Keep it in sync:** a PR changing the
architecture rule, the toolchain, or the tool set updates it too. `comfy-mcp` is a small,
standalone [MCP](https://modelcontextprotocol.io) server that lets an agent drive a user's
**local** ComfyUI: a **thin wrapper over
[`comfy-cli`](https://github.com/Comfy-Org/comfy-cli)**, the engine.

## The architecture rule — thin wrapper only (read this first)

Every tool is a passthrough to the `comfy` binary — the only way to reach comfy-cli is the
`_run_comfy(*args)` helper in `src/comfy_mcp/server.py`, which shells out to `comfy --json
--where local <args>` (global flags **before** the subcommand), parses comfy-cli's versioned
`envelope/1` result, and returns its `data`. Do not bypass it.

Hard guardrails — a PR breaking any of these should be rejected:

- **Every tool is a `comfy --json --where local` passthrough.** New functionality belongs in
  comfy-cli, exposed here as a thin `_run_comfy` call. A feature that can't be a `comfy`
  subcommand needs a comfy-cli change, not a workaround here.
- **No HTTP client.** This server never talks to ComfyUI (or anything else) over HTTP
  directly — no `httpx`/`requests`/`aiohttp`/`urllib` calls to a server. comfy-cli owns all
  I/O with ComfyUI (reaching a *local* process means shelling out to `comfy`, never opening
  a socket).
- **No code from the cloud MCP.** Do not copy code, patterns, or dependencies from
  `Comfy-Org/comfy-cloud-mcp-server` — a multi-tenant HTTP service with per-session state,
  signed URLs, analytics, and a cloud API client, none of which apply here. This repo is
  local-only, single-process, with no filesystem/multi-tenancy concerns to design around.

A tool may COMPOSE more than one passthrough when the value is in the sequence, not new
logic — `fetch_template` runs `templates fetch` then `validate`, telling the caller whether
the template it wrote can run; `download_model` submits `model download --background` then
polls `model download-status` so a multi-GB transfer doesn't hold the MCP request open. That
stays inside the rule: every call goes through `_run_comfy`, the verdict is comfy-cli's own,
no product behavior is added. Breaching it means deriving the answer here (parsing the
graph, keeping a table of what is "supported") instead of asking the engine — e.g.
`download_model` sizing the file on disk instead of reading its `status`. `install_node`
composes for a different reason: `comfy node install` runs Manager's `cm-cli`, which a
legacy clone under `custom_nodes/` can't provide, so it reads `comfy env`'s manager fields
BEFORE the consent prompt rather than authorize third-party code on a call that can't
succeed. It fails OPEN and shares `workflow_deps`' two helpers so the two can't drift, and
reads its VERDICT from printed text (`_extract_install_failures`) — cm-cli prints a pack's
failure before consulting `--exit-on-fail`, so a pack that never installed came back `ok:
true` — on the standing of `_extract_saved_paths`: no envelope, so text is the only channel
and the verdict is cm-cli's own sentence, matched one-directionally so a wording change
regresses rather than fails every install. `workflow_deps` reads its answer off DISK because
the engine leaves none — `comfy node deps-in-workflow` emits no envelope and REQUIRES an
`--output` path — so the temp-file round trip is the contract, and the manifest goes back as
written bar `failure_log._scrub_text` masking repo-URL credentials. `restart_comfyui`
composes a THIRD pair: `comfy stop --port <p> --dry-run` / `comfy stop --port <p>` show who
holds a leftover port to recycle — never a `psutil`/HTTP check; that verdict stays
comfy-cli's.

The one thing that legitimately lives here rather than in comfy-cli is **MCP protocol
surface** — capabilities comfy-cli can't express. Today that's the per-call confirmation on
tools that can spend money, destroy local state, run third-party code, kill a process, or
expose the machine: `partner_generate`, `run_template`, `run_workflow`,
`switch_comfyui_version`, `install_node`, `update_comfyui` when `target="all"`, the
`launch_comfyui`/`restart_comfyui` pair when `extra_args` would publish ComfyUI to the
network, and again to kill an untracked server. comfy-cli owns the credit-spend interlock
and the durable "always proceed" (`comfy generate consent always`); this server only raises
the confirmation over MCP **elicitation** — the protocol's y/N prompt — then forwards the
answer as `--yes`/`--allow-spend`, or (for the five the CLI doesn't gate at all) refuses to
run the command. It stores no consent of its own — all share one fail-closed body,
`_elicit_approval`; give a new gate its own `_ApprovalWording`, not a second copy. Adding
*product* behavior here is still a guardrail breach; adapting comfy-cli's contract to an MCP
primitive is this repo's job.

The three *spend* gates — `partner_generate`, `run_template`, `run_workflow` — differ where
the engine's own shape differs, and those differences are load-bearing: `comfy generate`
always spends, so it always prompts and honors `spend.auto_confirm`, while `comfy
run-template` and `comfy run` are usually free and never read that setting — so those two
prompt only when `confirm_spend=True` unlocks spending, treating the generate-scoped
always-proceed as granting nothing. That shared opt-in policy is one body,
`_resolve_optin_spend_consent`, whose per-verb wording is an argument the way
`_ApprovalWording` is `_elicit_approval`'s. What still differs is the capability SIGNAL:
`run_template` can trust the verb, `run_workflow` must PROBE `comfy run --help` for the flag
— its docstring says why. `switch_comfyui_version`, `install_node` and `update_comfyui` sit
outside that axis: none spends credits and the CLI gates none, so their prompt is this
server's only gate, with no always-proceed to read. TWO are ARGUMENT-scoped, the danger
being the argument not the verb: the launch pair prompts only for `extra_args` publishing an
unauthenticated ComfyUI (`_network_exposing_args`), `update_comfyui` only for
`target="all"`, which pip-installs every third-party pack (`comfy`/`cli` never prompt).
`install_node` is NOT one — it prompts on EVERY call, its `names` argument RESTRICTS to
registry slugs, refusing a URL, since the prompt promises a REGISTRY pack.
`restart_comfyui`'s kill gate is STATE-scoped instead, firing mid-sequence once a launch
loses the port. Mirror the engine's contract per tool; never generalize one gate to another.

The local differentiator: discovery (`search_nodes`, `get_node`, `search_models`) reads the
**live install** — custom nodes included — not a static catalog.

## Module layout

`server.py` holds the wrapper core (`_run_comfy`, the envelope parser, the `--json-stream`
machinery, the spend-consent plumbing) and every `@mcp.tool()`. Nine **leaf** modules sit
under it — none imports `server`, so the dependency edges only ever point one way:

| Module | Owns |
|---|---|
| `textutil.py` | pure text helpers: `_tail` / `_stream_tail` (bounded stream tails) and `_redact_url` (userinfo masking) |
| `tcc.py` | macOS protected-folder (TCC) detection + the guidance message |
| `failure_log.py` | the opt-in `COMFY_MCP_DEBUG_LOG` failure log (its config, its module state, and `_log_failure`) **and the URL scrubbers** — `_scrub_text` / `_scrubbed_stream_tail` also mask credentials on the way to the MCP CLIENT, not just to disk |
| `instructions.py` | the `INSTRUCTIONS` constant handed to `MCPServer(..., instructions=...)` — client-handshake text |
| `errors.py` | `ComfyCliError`; the "nothing recorded to stop" detector; the `error.details` renderer + per-field char cap |
| `clitext.py` | comfy-cli **human-output** parsing for verbs with no envelope — `Saved:`-block/install-failure extraction, `plain_ok` synthesis, missing-verb/-option probes, `install_node`'s per-pack verdict, echoed-argv forgery guards. Its extractors are the documented cm-cli contract (see architecture rule above) — move or edit byte-for-byte |
| `argv.py` | argument-injection and OS-limit guards for every tool-facing string headed for `subprocess`: shared primitives plus per-domain guards (workflow path, prompt id, download id, extra args, version, node names, log port, model path/filename, upload paths) |
| `target.py` | remote-target resolution/redaction/provenance for run/job tools — `COMFYUI_URL`/`HOST`/`PORT` parsing, `--host`/`--port` forwarding, the local-only `download_model` refusal, and divergence notes on `system_stats`/`free_memory` so an agent doesn't gate a remote run on local numbers |
| `params.py` | param/slot marshaling into comfy-cli argv for `generate`/`run-template`/`set-slot`/`vary`, incl. the structured slot machinery — `SlotOverride`/`SlotVariants` are this module's public TYPES (carve-out below) |

`server` reaches them **module-qualified** (e.g. `failure_log._log_failure(...)`) and
re-exports no BEHAVIOR: patching a moved name on `server` would silently patch nothing.
**Patch the owning module** (`monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", …)`),
not `server` — the wrong one now raises `AttributeError`. Carve-out: public exception/model
TYPES (`ComfyCliError`, `SlotOverride`, `SlotVariants`) ARE name-imported — they ride many
`except`/`isinstance`/tool-signature sites and hold no mutable state a test could patch the
wrong copy of, so that risk doesn't apply.

## Toolchain

Python ≥ 3.10; pip + setuptools (no `uv.lock` here — comfy-cli bundles `uv` and may write a
stray one into the working directory; gitignored).

```bash
pip install -e '.[dev]'   # install with dev extras (pytest, ruff)
pytest -q                 # run the tests
ruff check .              # lint
ruff format --check .     # format check (run `ruff format .` to fix)
```

CI (`.github/workflows/ci.yml`) runs all three on Python 3.10 and 3.14 every PR; get them
green locally. Never add a `paths`/`paths-ignore` filter to its `pull_request` trigger —
`test (py3.10)`/`test (py3.14)` must report every PR; it no-ops on Markdown-only changes.

## Tests

Tests live in `tests/` and mock comfy-cli — no real ComfyUI, no `comfy` binary. `_run_comfy`
and the parser are exercised directly (`test_wrapper.py`, `test_parser.py`); each tool group
has its own file. Add a tool's test with it.

**Mock comfy-cli via the shared fixtures in `tests/conftest.py`, never a hand-rolled stub.**
They mirror how `server` spawns the CLI, so a spawn-signature change is one edit, not a
sweep:

- `envelope(ok=…, data=…, error=…)` — build an `envelope/1` body.
- `patched_run(stdout=…, returncode=…, stderr=…, raises=…, on_spawn=…) -> calls` — the plain
  `--json` path (`subprocess.run`); `calls` records `cmd`/`env`/`timeout`/`encoding` per
  call for exact-argv checks; `on_spawn(cmd)` fires at spawn so the one verb whose answer is
  a FILE writes its `--output`.
- `patched_plain_run(returncode, stdout, stderr) -> calls` — same, for verbs that print
  human text with no envelope (`launch`/`stop`/`generate`).
- `patched_stream(stdout_text) -> procs` — the `--json-stream` NDJSON path
  (`asyncio.create_subprocess_exec`). Its fake pipes are real `asyncio.StreamReader`s from
  conftest's `stream_reader(text, limit)` helper — reuse it rather than hand-roll an
  awaitable, so fakes exercise the buffer-limit behavior too.
- `patched_async_run(stdout=…, returncode=…, stderr=…, hang=…, on_spawn=…) -> procs` — the
  plain-JSON *async* path (`_run_comfy_async`): same spawn and real `StreamReader` pipes as
  `patched_stream`, but parses the capture once at the end, not line-by-line. `hang=True`
  leaves the pipes OPEN so the fake child never finishes (timeout/cancellation cases);
  `kill()` closes them (mirroring the process-group kill so a post-kill drain reaches EOF),
  records `killed`, and fires `on_spawn(cmd)` like `patched_run`'s.

The two spawn paths differ deliberately: the plain `--json` path is synchronous
(`subprocess.Popen` + a bounded `communicate`, off-loaded to a thread pool for async
callers); anything STREAMING or long-lived spawns via `asyncio.create_subprocess_exec`
instead — nothing blocking runs on the event loop, enforced by ruff's `ASYNC` select. Two
async runners live there: `_run_comfy_streaming` (NDJSON + progress) and `_run_comfy_async`,
a plain-JSON twin of `_run_comfy` for CANCELLATION — `asyncio.to_thread` is non-blocking,
but cancellation never reaches the thread, so a client giving up left the child running. It
carries the legacy foreground `model download`, `workflow_deps`' 300s resolve, and
`upload_file`'s 300s transfer. Each stream keeps only a `_STDERR_MAX_CHARS` tail
(`_drain_capped_into`; callers widen stdout via `stdout_cap=`), never `communicate()`'s full
capture. `auth_login` (`_start_login`) is a third spawn site with its own browser flow.

A local stub is justified only where the call genuinely differs — the `comfy --version`
probe (its own kwargs) and multi-call sequenced replies.

## Destined-public hygiene

This repository is **private but destined to go public.** Treat everything you write as if
it were already public:

- **No secrets** — API keys, tokens, or credentials in code, commits, tests, fixtures, or PR
  text. Credential-in-URL fixtures use `https://<user>:<pass>@host`: a bare `user:pass@`
  fails the secret-scanning diff gate, and a fake scheme documents a scrubber gap
  (`failure_log._URL_RE` needs `https?://`).
- **No internal hostnames, IPs, or internal-only URLs** in code, comments, or commit
  messages.
- **No internal-tracker references** in commits or PR titles/bodies — describe the change on
  its own terms.
- Prefer environment variables and documented config over anything hardcoded.
