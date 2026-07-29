# Agent Guidelines — comfy-local-mcp

A short guide for AI agents (and humans) contributing to this repo. **Keep it in
sync:** if a PR changes the architecture rule, the toolchain, or the tool set,
update the relevant section in the same PR.

## What this repo is

`comfy-local-mcp` is a small, standalone [MCP](https://modelcontextprotocol.io)
server that lets an agent drive a user's **local** ComfyUI. It is a **thin
wrapper over [`comfy-cli`](https://github.com/Comfy-Org/comfy-cli)** — comfy-cli
is the engine; this repo is just the MCP surface over it.

## The architecture rule — thin wrapper only (read this first)

Every tool is a passthrough to the `comfy` binary. There is exactly one way to
reach comfy-cli: the `_run_comfy(*args)` helper in `src/comfy_local_mcp/server.py`,
which shells out to `comfy --json --where local <args>` (global flags **before**
the subcommand), parses comfy-cli's versioned `envelope/1` result, and returns
its `data`. Do not bypass it.

Hard guardrails — a PR that breaks any of these should be rejected:

- **Every tool is a `comfy --json --where local` passthrough.** New functionality
  belongs in comfy-cli; expose it here as a thin tool that calls `_run_comfy`.
  If a feature can't be expressed as a `comfy` subcommand, the fix is a comfy-cli
  change, not a workaround in this repo.
- **No HTTP client.** This server never talks to ComfyUI (or anything else) over
  HTTP directly — no `httpx`, `requests`, `aiohttp`, `urllib` calls to a server.
  comfy-cli owns all I/O with ComfyUI. (Reaching a *local* process is done by
  shelling out to `comfy`, never by opening a socket.)
- **No code from the cloud MCP.** Do not copy code, patterns, or dependencies
  from `Comfy-Org/comfy-cloud-mcp-server`. That server is a multi-tenant HTTP
  service with per-session state, signed URLs, analytics, and a cloud API client
  — none of which apply here. This repo is local-only, single-process, and has no
  filesystem/multi-tenancy concerns to design around.

A tool may COMPOSE more than one passthrough when the value is in the sequence
rather than in new logic — `fetch_template` runs `templates fetch` and then
`validate` to tell the caller whether the template it just wrote can actually
run on this install, and `download_model` submits `model download --background`
and then polls `model download-status` so a multi-GB transfer does not hold the
MCP request open. That stays inside the rule: every call still goes through
`_run_comfy`, the verdict is comfy-cli's own, and this repo adds no product
behavior of its own. What would breach it is deriving the answer here (parsing
the graph, keeping a table of what is "supported") instead of asking the engine
— for `download_model` that would mean sizing the file on disk to decide whether
it finished, rather than reading comfy-cli's `status`.

The one thing that legitimately lives here rather than in comfy-cli is **MCP
protocol surface** — capabilities that only exist between this server and its
client, and that comfy-cli has no way to express. Today that is the per-call
confirmation on the three tools that can spend money or destroy local state:
`partner_generate`, `run_template`, and `switch_comfyui_version`. comfy-cli owns
the credit-spend interlock and the durable "always proceed"
(`comfy generate consent always`), and this server only raises the confirmation
over MCP **elicitation** — the protocol's equivalent of the CLI's y/N prompt —
then forwards the answer as `--yes` / `--allow-spend`, or (for the version
switch, which the CLI does not gate at all) simply refuses to run the command.
It stores no consent of its own. All three share one fail-closed body,
`_elicit_approval`; give a new gate its own `_ApprovalWording` rather than a
second copy of that handling. Adding *product* behavior here is still a
guardrail breach; adapting comfy-cli's contract to an MCP primitive is this
repo's job.

The two differ where the engine's own shape differs, and those differences are
load-bearing rather than incidental: `comfy generate` always spends so it always
prompts and honors `spend.auto_confirm`, while `comfy run-template` is usually
free and never reads that setting — so `run_template` prompts only when
`confirm_spend=True` asks to unlock spending, and treats the generate-scoped
always-proceed as granting nothing. Mirror the engine's contract; do not
generalize one tool's consent rules onto the other.

The local differentiator: discovery tools (`search_nodes`, `get_node`,
`search_models`) read the **user's live install** — custom nodes included — via
comfy-cli, not a bundled static catalog.

## Module layout

`server.py` holds the wrapper core (`_run_comfy`, the envelope parser, the
`--json-stream` machinery, the spend-consent plumbing) and every `@mcp.tool()`.
Three **leaf** modules sit under it — nothing in them imports `server`, so the
dependency edges only ever point one way:

| Module | Owns |
|---|---|
| `textutil.py` | pure text helpers: `_tail` / `_stream_tail` (bounded stream tails) and `_redact_url` (userinfo masking) |
| `tcc.py` | macOS protected-folder (TCC) detection + the guidance message |
| `failure_log.py` | the opt-in `COMFY_LOCAL_MCP_DEBUG_LOG` failure log: its config, its module state, and `_log_failure` |

`server` reaches them **module-qualified** (`tcc._tcc_guidance(...)`,
`failure_log._log_failure(...)`) and re-exports nothing. That is deliberate: a
test that patches a moved name on `server` would otherwise silently patch a name
nothing reads. **Patch the owning module** — `monkeypatch.setattr(failure_log,
"_FAILURE_LOG_PATH", …)`, not `server`. Patching the wrong one now raises
`AttributeError` instead of passing while testing nothing.

## Toolchain

Python ≥ 3.10. Everything runs through pip + setuptools (there is no `uv.lock`
in this repo — comfy-cli bundles `uv` and may write a stray `uv.lock` into the
working directory; it is gitignored and is not ours).

```bash
pip install -e '.[dev]'   # install with dev extras (pytest, ruff)

pytest -q                 # run the tests
ruff check .              # lint
ruff format --check .     # format check (run `ruff format .` to fix)
```

CI (`.github/workflows/ci.yml`) runs all three on Python 3.10 and 3.14 for every
PR. Get them green locally before pushing. The workflow carries no path filter
on `pull_request` — the `test (py3.10)` / `test (py3.14)` contexts are required
by branch protection, so they have to report on every PR — and it decides
internally whether to run the suite, no-opping when a PR changes only Markdown.
Do not add a `paths` / `paths-ignore` filter to that trigger.

## Tests

Tests live in `tests/` and mock comfy-cli — they never require a real ComfyUI or
the `comfy` binary. `_run_comfy` and the envelope parser are exercised directly
(`test_wrapper.py`, `test_parser.py`); each tool group has its own file
(`test_discovery.py`, `test_templates.py`). When you add or change a tool, add or
update its test in the same PR.

**Mock comfy-cli through the shared fixtures in `tests/conftest.py`, never a
hand-rolled stub.** They are the single place that mirrors how `server` spawns
the CLI, so a change to the spawn signature is one edit rather than a sweep:

- `envelope(ok=…, data=…, error=…)` — build an `envelope/1` body.
- `patched_run(stdout=…, returncode=…, stderr=…, raises=…) -> calls` — the plain
  `--json` path (`subprocess.run`); `calls` records `cmd`/`env`/`timeout`/
  `encoding` per invocation for exact-argv assertions.
- `patched_plain_run(returncode, stdout, stderr) -> calls` — same, for the verbs
  that print human text and emit no envelope (`launch`/`stop`/`generate`).
- `patched_stream(stdout_text) -> procs` — the `--json-stream` NDJSON path
  (`asyncio.create_subprocess_exec`). Its fake pipes are real
  `asyncio.StreamReader`s, built by conftest's `stream_reader(text, limit)`
  helper; reuse that rather than hand-rolling an awaitable, so a fake still
  exercises the reader's buffer-limit behavior.

The two spawn paths differ deliberately: the plain `--json` path is synchronous
(`subprocess.Popen` + a bounded `communicate`, off-loaded to a thread pool by its
async callers), while every path that STREAMS or is otherwise long-lived
(`_run_comfy_streaming`, `auth_login`) spawns with `asyncio.create_subprocess_exec`
and reads the pipes as asyncio streams — nothing blocking may run on the event
loop. `ASYNC` is enabled in ruff's `select` to enforce that.

A local stub is justified only where the call genuinely differs — the
`comfy --version` probe (its own kwargs) and multi-call sequenced replies.

## Destined-public hygiene

This repository is **private but destined to go public.** Treat everything you
write as if it were already public:

- **No secrets** — API keys, tokens, or credentials in code, commits, tests,
  fixtures, or PR text.
- **No internal hostnames, IPs, or internal-only URLs** in code, comments, or
  commit messages.
- **No internal-tracker references** in commits or PR titles/bodies — describe
  the change on its own terms.
- Prefer environment variables and documented config over anything hardcoded.
