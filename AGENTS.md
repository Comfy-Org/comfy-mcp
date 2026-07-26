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

The one thing that legitimately lives here rather than in comfy-cli is **MCP
protocol surface** — capabilities that only exist between this server and its
client, and that comfy-cli has no way to express. Today that is the per-call
spend confirmation on the two tools that can spend, `partner_generate` and
`run_template`: comfy-cli owns the credit-spend interlock and the durable
"always proceed" (`comfy generate consent always`), and this server only raises
the confirmation over MCP **elicitation** — the protocol's equivalent of the
CLI's y/N prompt — then forwards the answer as `--yes` / `--allow-spend`. It
stores no consent of its own. Adding *product* behavior here is still a
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
PR. Get them green locally before pushing.

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
  (`subprocess.Popen`).

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
