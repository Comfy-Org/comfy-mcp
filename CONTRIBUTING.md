# Contributing to comfy-mcp

Thanks for your interest in improving `comfy-mcp`! This is a small,
standalone [MCP](https://modelcontextprotocol.io) application with stdio and
Streamable HTTP adapters. Both let AI agents drive the ComfyUI owned by the
server host by shelling out to
[`comfy-cli`](https://github.com/Comfy-Org/comfy-cli).

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Response times

We review issues and pull requests on a best-effort basis and do not commit to a
response-time SLA.

## The one architecture rule — thin wrapper only

Before you write any code, read [`AGENTS.md`](AGENTS.md). The core rule:

**Every tool is a passthrough to the `comfy` binary.** Application functions
resolve a `ComfyCliClient` from `src/comfy_mcp/client/`; its concrete
implementation delegates to the established guarded runners in
`src/comfy_mcp/server/_internal.py`. Those runners are the only code that may spawn
`comfy --json --where local <args>` or
`comfy --json-stream --where local <args>`, parse comfy-cli's versioned
`envelope/1` result, and return its data. Do not bypass this boundary.

This means:

- **No outbound ComfyUI HTTP client.** comfy-cli owns all I/O with ComfyUI. The
  inbound Remote MCP listener is SDK-managed Streamable HTTP; it is not a
  second ComfyUI integration.
- **New functionality belongs in comfy-cli.** If a feature can't be expressed as
  a `comfy` subcommand, the fix is a comfy-cli change, not a workaround here.
- **No code from the cloud MCP.** Don't copy code, patterns, or dependencies
  from the Comfy Cloud MCP — this repo has no cloud API, multi-tenant storage,
  analytics, or per-session filesystem layer. Its narrow HTTP file boundary is
  implemented locally: inline bytes become temporary `comfy upload` paths and
  completed `comfy download` files receive short-lived capability URLs.
- **No transport-owned application path.** `server/remote.py` only adapts the shared
  FastMCP instance to ASGI/uvicorn. It may not construct a server, register or
  wrap tools, spawn comfy-cli, or derive product verdicts independently.

A PR that breaks any of these guardrails will be asked to change. See
[`AGENTS.md`](AGENTS.md) for the full rationale.

## FastMCP 4 and transport guardrails

The framework baseline is the exact pair in `pyproject.toml`:
`fastmcp==4.0.0b3` and `mcp==2.0.0`. They are upgraded together, with the
transport, consent, schema, lifecycle, and packaging tests updated in the same
PR.

- `McpApplicationBuilder` owns the name, version, and base instructions and is
  called once to create `server.mcp`; every tool registers on that instance.
- `comfy_mcp.server` exports only `mcp`, `main`, and the 40 tool callables.
  Runtime helpers live in `comfy_mcp.server._internal`; do not add private
  aliases to the public package. Tests of private behavior import the owning
  module explicitly.
- Bare `comfy-mcp` serves that application through stdio. `comfy-mcp serve`
  serves the same 40 tools, schemas, results, and confirmations at `/mcp`.
- Remote mode uses FastMCP's public `http_app(..., stateless_http=True)` ASGI
  surface and uvicorn lifecycle. Do not add legacy SSE, hand-written JSON-RPC,
  private session monkeypatches, or catch-all protocol-error suppression.
- The signed download route is registered on that same FastMCP application and
  served by the same ASGI listener/port. Do not add a second file server,
  transport-specific tool registry, or direct ComfyUI HTTP client.
- Both adapters use the same application functions and `ComfyCliClient`.
  `client/` must not import the MCP server, and request-local injection remains
  a `ContextVar` so concurrent HTTP requests cannot share mutable client state.
- stdio stdout is protocol-only. Diagnostics go to stderr; the optional failure
  trail goes only to its configured file.

The rationale, including the LightRAG change-history bugs this avoids, is in
[`docs/remote-http-design.md`](docs/remote-http-design.md).

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

Most tests mock or fake comfy-cli, so `pytest` runs anywhere. Before running
tests, review the files and paths affected by the change and select a focused
regression set; after it passes, run the complete three-command gate above.

Changes to the transport, shared client boundary, failure handling, or workflow
business flow must also run this focused smoke set:

```bash
pytest -q \
  tests/test_remote_http.py \
  tests/test_remote_transport.py \
  tests/test_fastmcp_app.py \
  tests/test_file_transfer.py \
  tests/test_failure_log.py
```

This uses the in-process application and a real loopback Streamable HTTP MCP
transport with the shared comfy-cli fixtures in `tests/conftest.py`; default
tests must not create a fake `comfy` executable. The opt-in live-engine smoke
(`./scripts/smoke.sh`) launches the real stdio MCP subprocess and drives the
real `comfy` binary against a running same-machine ComfyUI selected with
`COMFY_LOCAL_URL`. The live suite skips when ComfyUI or the binary is absent;
its `generate_image` case also requires the documented SD1.5 checkpoint.

For a deployed Streamable HTTP server, run the client-side live smoke through
its reachable endpoint (an SSH local-forward is supported):

```bash
COMFY_MCP_TEST_URL=http://127.0.0.1:9000/mcp \
COMFY_MCP_TEST_WORKFLOW=/absolute/server/path/tests/e2e/workflow_smoke.json \
  pytest -q tests/e2e/test_remote_mcp.py -m e2e
```

The workflow path is server-side; the generated upload and output paths are
client-side. This opt-in test performs one tiny upload and one checkpoint-free
ComfyUI job, so do not run it against a deployment where those mutations are
unexpected.

## Failure log changes

`COMFY_MCP_DEBUG_LOG` is deliberately off by default, local to the MCP server
host, owner-only, bounded by rotation, and best-effort. Runners publish one
immutable `_FailureEvent`; the JSONL writer is its sole default observer and
returns without filesystem work while disabled. Keep this small—do not turn it
into a general event bus. Any new runner failure must call
`failure_log._log_failure(...)` immediately before its exception reaches the
shared application. Preserve URL credential/query redaction for arguments,
messages, and stream tails. Update the README's kind list and both stdio/HTTP
configuration guidance whenever this contract changes, and add a regression
at the runner and observer boundary. Upload request bytes must never be
published in a failure event or written to JSONL.

## Remote file-transfer changes

`upload_file` keeps the ComfyCloud-compatible required pair `file_path` and
`client_os`. stdio passes the absolute client path to `comfy upload`; HTTP
returns a credential-free single-use PUT command on `/api/uploads/{token}`.
The route accepts only the image filename bound at mint time, caps the body at
50 MiB, writes it owner-only, and removes it after success, failure, or
cancellation. It still reaches ComfyUI only through the same
`ComfyCliClient.run_async("upload", ...)` path. Base64-in-tool upload is not a
fallback.

Both capability routes use the same MCP listener and FastMCP ASGI application.
`fetch_outputs` keeps `comfy download` as its engine. stdio writes directly to
the caller's path; HTTP downloads into owner-only scratch and returns a
five-minute HMAC-signed URL and one `client_os`-selected command on the same MCP
listener. Validate that comfy-cli reported every served path inside that
scratch directory, make the URL non-cacheable, and clean the directory on
expiry. A reverse proxy deployment must forward `/api/uploads/` and
`/downloads/` along with `/mcp`, and preserve the client-facing origin through
`Forwarded` or the `X-Forwarded-Proto` / `X-Forwarded-Host` /
`X-Forwarded-Port` trio. Tests must pin a non-default external port so a proxy
regression cannot emit an internal or incomplete capability URL. A platform
launcher whose proxy strips those fields maps its authoritative upstream URL
to `COMFY_MCP_PUBLIC_URL`; the core remains platform-neutral and validates that
override as a credential-free HTTP(S) origin.

Changes here require the complete upload → submit → poll → fetch flow over real
stdio and loopback HTTP transports, both modern and legacy HTTP negotiation,
signature tamper/expiry and path-boundary tests, failure-log non-disclosure,
and cleanup assertions.

## Adding or changing a tool

- Every tool is a thin call through the shared `ComfyCliClient` runners — keep
  it that way.
- Add or update the tool's test in the same PR (`tests/` mirrors the tool
  groups: `test_wrapper.py`, `test_parser.py`, `test_discovery.py`,
  `test_templates.py`, …).
- A registered tool is available over both transports. Review network exposure
  and confirmation semantics as part of every tool change; do not hide the
  issue by creating a transport-specific registry or wrapper.
- Keep [`README.md`](README.md) and [`AGENTS.md`](AGENTS.md) in sync when you
  change the tool set, client architecture, transport, failure-log contract,
  smoke stages, or toolchain.

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
