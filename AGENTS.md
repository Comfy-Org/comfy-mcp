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

The local differentiator: discovery tools (`search_nodes`, `get_node`,
`search_models`) read the **user's live install** — custom nodes included — via
comfy-cli, not a bundled static catalog.

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
