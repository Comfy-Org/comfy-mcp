# Contributing to comfy-mcp

Thanks for your interest in improving `comfy-mcp`! This is a small,
standalone [MCP](https://modelcontextprotocol.io) server that lets AI agents
drive a user's **local** ComfyUI by shelling out to
[`comfy-cli`](https://github.com/Comfy-Org/comfy-cli).

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Response times

We review issues and pull requests on a best-effort basis and do not commit to a
response-time SLA.

## The one architecture rule — thin wrapper only

Before you write any code, read [`AGENTS.md`](AGENTS.md). The core rule:

**Every tool is a passthrough to the `comfy` binary.** There is exactly one way
to reach comfy-cli — the `_run_comfy(*args)` helper in
`src/comfy_mcp/server.py`, which shells out to
`comfy --json --where local <args>`, parses comfy-cli's versioned `envelope/1`
result, and returns its `data`. Do not bypass it.

This means:

- **No HTTP client.** This server never talks to ComfyUI (or anything else) over
  HTTP directly — comfy-cli owns all I/O with ComfyUI.
- **New functionality belongs in comfy-cli.** If a feature can't be expressed as
  a `comfy` subcommand, the fix is a comfy-cli change, not a workaround here.
- **No code from the cloud MCP.** Don't copy code, patterns, or dependencies
  from the Comfy Cloud MCP — this repo is local-only and single-process.

A PR that breaks any of these guardrails will be asked to change. See
[`AGENTS.md`](AGENTS.md) for the full rationale.

## Dev setup

Python ≥ 3.10. Everything runs through pip + setuptools.

```bash
pip install -e '.[dev]'    # install with dev extras (pytest, ruff)
```

## Run the checks

CI (`.github/workflows/ci.yml`) runs these three on Python 3.10 and 3.14 for
every PR. Get them green locally before pushing:

```bash
pytest                     # run the tests
ruff check .               # lint
ruff format --check .      # format check (run `ruff format .` to fix)
```

The tests mock comfy-cli — they never require a real ComfyUI or the `comfy`
binary, so `pytest` runs anywhere. There is also an opt-in end-to-end smoke test
(`./scripts/smoke.sh`) that drives the real tools against a running local
ComfyUI; it **skips** cleanly when ComfyUI or the `comfy` binary is absent.

## Adding or changing a tool

- Every tool is a thin `_run_comfy(...)` call — keep it that way.
- Add or update the tool's test in the same PR (`tests/` mirrors the tool
  groups: `test_wrapper.py`, `test_parser.py`, `test_discovery.py`,
  `test_templates.py`, …).
- Keep [`README.md`](README.md) and [`AGENTS.md`](AGENTS.md) in sync when you
  change the tool set, the architecture rule, or the toolchain.

## Opening a pull request

1. Fork and branch off `main`.
2. Make your change, with tests, and get the three checks above green.
3. Open a PR and fill in [the template](.github/PULL_REQUEST_TEMPLATE.md).

Two more things run against your work without you asking for them: a
[TruffleHog scan](.github/workflows/secret-scanning.yml) of the pushed range,
and GitHub push protection, which rejects a push carrying a recognized provider
credential to this repository outright. If a push is rejected that way, rotate
the credential and rewrite it out of your commits rather than bypassing the block —
see [`SECURITY.md`](SECURITY.md#automated-security-tooling).

## Reporting bugs and requesting features

Use the [issue templates](.github/ISSUE_TEMPLATE/) — a bug report or a feature
request. For anything security-sensitive, follow [`SECURITY.md`](SECURITY.md)
instead of opening a public issue.
