# Agent Guidelines — comfy-mcp

`comfy-mcp` is a small standalone MCP server for a user's local ComfyUI. It is a
thin wrapper over `comfy-cli`, which remains the engine. Keep this file in sync
when architecture, toolchain, or the public tool set changes.

Detailed design rationale and edge-case contracts live in
[`docs/agent-reference.md`](docs/agent-reference.md). Read the relevant section
before changing runners, consent, transfers, process lifecycle, or test doubles.

## Non-negotiable architecture

- Every tool reaches `comfy` only through the `ComfyCliClient` port and the
  guarded runners in `server/_internal.py`.
- Commands use `comfy --json --where local <args>` or
  `comfy --json-stream --where local <args>` with global flags first, and parse
  comfy-cli's versioned `envelope/1` result.
- New product behavior belongs in comfy-cli. A tool here may compose multiple
  passthroughs only when the value is the sequence and every verdict remains
  comfy-cli's own.
- Never contact ComfyUI or another service directly. No `httpx`, `requests`,
  `aiohttp`, `urllib`, sockets, `psutil`, or filesystem-derived engine verdicts.
- Never copy code, architecture, dependencies, or per-session state from
  `Comfy-Org/comfy-cloud-mcp-server`.
- Discovery (`nodes`, `search_models`) reads the live install, including custom
  nodes; never replace it with a static catalog.

Documented compositions are limited to the contracts in the reference:
`fetch_template`, `download_model`, `install_node`, `workflow_deps`, and
`restart_comfyui`. Preserve their engine-owned verdicts and failure behavior.

## MCP-only adaptations

MCP protocol concerns that comfy-cli cannot express may live here:

- Per-call confirmation for spending, destructive changes, third-party code,
  process kills, and network exposure. All gates share `_elicit_approval`; each
  gate has its own `_ApprovalWording.gate_id`. Never treat `confirm_*` as human
  consent when an interactive route exists.
- `partner_generate` always prompts and may honor comfy-cli's durable generate
  consent. `run_template` and `run_workflow` prompt only when
  `confirm_spend=True` and never inherit generate consent.
- `switch_comfyui_version` and `install_node` always prompt;
  `update_comfyui` prompts for `target="all"`; launch/restart prompt only for
  network-exposing arguments; restart prompts again before killing an
  untracked server. Mirror each engine contract rather than generalizing gates.
- Modern approval returns `InputRequiredResult`; legacy approval uses
  `ctx.elicit`. Classify negotiated revisions only through
  `mcp_types.version` registries. Unknown revisions fail closed.
- Project anchoring passes `cwd=` to comfy-cli; the machine snapshot quotes
  comfy-cli's hardware payload verbatim and fails open on probe failure.
- Remote HTTP transfer adapts client-local paths through same-listener upload
  and signed-download routes. Bytes still enter and leave ComfyUI only through
  comfy-cli.

For transfers: stage owner-only scratch, cap uploads at 50 MiB, publish only
contained regular files, use five-minute single-use/signed capabilities, and
always clean scratch. Derive the client-facing origin from sanitized
`Forwarded`/`X-Forwarded-*`, or use validated `COMFY_MCP_PUBLIC_URL`. Never add a
second listener, storage API, base64 fallback, or per-session filesystem.

## Module and transport boundaries

- `server/__init__.py` exports only `mcp`, `main`, and the 40 public tools;
  `server/tools.py` owns the explicit tool export list.
- `server/_internal.py` owns composition, runners, envelopes, consent, tool
  implementations, and startup. Do not re-export private helpers.
- `server/mcp_app.py` performs the only `FastMCP(...)` construction.
  `server/remote.py` only adapts the already-built application to uvicorn.
- Leaf modules do not import the server package. Access patchable leaf state
  module-qualified; import only stable exception/model types by name.
- `file_transfer.py` owns protocol filesystem adaptation, not ComfyUI I/O.
  `client/protocols.py` owns the port; `client/subprocess_client.py` owns its
  concrete runner delegation; `client/context.py` owns request-safe binding.

Stdio and HTTP use the same FastMCP instance, tool registry, schemas, consent,
instructions, and client. HTTP uses FastMCP's public `custom_route` on the same
listener for `/api/uploads/{token}` and `/downloads/{token}/{filename}`. Never
implement JSON-RPC, SSE, sessions, WebSockets, reconnect, or a second server.
Keep console logs on stderr; stdout is always reserved for protocol/data.
Loopback is the default listener. Non-loopback binds require explicit allowed
Host patterns and belong behind an authenticated TLS reverse proxy.

## Tests and test doubles

Default tests mock comfy-cli and never require a real ComfyUI or `comfy` binary.
The opt-in `tests/e2e/test_smoke.py` is the real local ComfyUI/stdio flow;
`tests/e2e/test_remote_mcp.py` is the environment-driven deployed HTTP flow.
Never label fake-engine coverage as real end-to-end coverage.

All comfy-cli doubles belong in `tests/conftest.py`; never define a local one.
Reuse or extend these fixtures:

- `fake_comfy_client`, `envelope`, `patched_comfy_run_sequence`
- `patched_run`, `patched_run_sequence`, `patched_plain_run`
- `patched_stream`, `blocking_stream`, `stderr_blocking_stream`
- `patched_async_run`, plus the shared real-`StreamReader` helpers

Plain `--json` uses synchronous `subprocess.Popen` plus bounded `communicate`,
off-loaded for async callers. Streaming and long-lived/cancellable work use
`asyncio.create_subprocess_exec`; nothing blocking runs on the event loop.
Keep stderr/stdout tails bounded and preserve process-group cancellation.

Tests for private behavior import `comfy_mcp.server._internal`; public API tests
import `comfy_mcp.server` and verify private names stay absent. Patch the module
that owns mutable state, not a copied import.

## Toolchain and public hygiene

Python >=3.10; pip + setuptools; no `uv.lock`. `comfy-cli` is intentionally not
a declared dependency. Keep `fastmcp==4.0.0b3` and `mcp==2.0.0` exact-pinned;
upgrade them only in a dedicated fully tested change.

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
ruff format --check .
```

CI runs tests, lint, and formatting on Python 3.10 and 3.14 for every PR. Never
add `paths` or `paths-ignore` to the pull-request trigger.

This private repository is destined to be public:

- Never write secrets, credentials, internal hosts/IPs, or internal tracker
  references in code, tests, fixtures, docs, commits, or PR text.
- Credential URL fixtures use `https://<user>:<pass>@host` so scrubbers and
  secret scanning remain meaningful.
- Prefer environment variables and documented configuration over hardcoding.
