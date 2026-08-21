# Agent Guidelines — comfy-mcp

A short guide for AI agents (and humans). **Keep it in sync:** a PR changing the
architecture rule, the toolchain, or the tool set updates it too. `comfy-mcp` is a small,
standalone [MCP](https://modelcontextprotocol.io) server that lets an agent drive a user's
**local** ComfyUI: a **thin wrapper over
[`comfy-cli`](https://github.com/Comfy-Org/comfy-cli)**, the engine.

## The architecture rule — thin wrapper only (read this first)

Every tool is a passthrough to the `comfy` binary. Application/tool code reaches it only
through the `ComfyCliClient` port under `src/comfy_mcp/client/`; the concrete client is
composed with the guarded `_run_comfy*_impl` runners in `server/_internal.py`. Those runners shell out
to `comfy --json[--stream] --where local <args>` (global flags **before** the subcommand),
parse comfy-cli's versioned `envelope/1`, and return its data. Do not bypass the client port
or the runners.

Hard guardrails — a PR breaking any of these should be rejected:

- **Every tool is a `comfy --json --where local` passthrough.** New functionality belongs in
  comfy-cli, exposed here as a thin `ComfyCliClient.run*` call. A feature that can't be a `comfy`
  subcommand needs a comfy-cli change, not a workaround here.
- **No HTTP client.** This server never talks to ComfyUI (or anything else) over HTTP
  directly — no `httpx`/`requests`/`aiohttp`/`urllib` calls to a server. comfy-cli owns all
  I/O with ComfyUI (reaching a *local* process means shelling out to `comfy`, never opening
  a socket).
- **No code from the cloud MCP.** Do not copy code, patterns, or dependencies from
  `Comfy-Org/comfy-cloud-mcp-server` — a multi-tenant HTTP service with per-session state,
  storage, analytics, and a cloud API client, none of which apply here. This repo's narrow
  same-listener signed download capability is locally implemented and process-scoped; it is
  not permission to import the cloud server's session/filesystem architecture.

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
primitive is this repo's job. Project anchoring (`_project_root`, `COMFY_PROJECT`, the
`project` tool) is the same kind of adaptation: comfy-cli resolves its governing `project/1`
by walking up from its own process cwd, assuming a persistent shell session an MCP client's
arbitrary per-call cwd can't provide, so this server passes `cwd=` on its own spawns instead
— the `status`/`init` verdicts stay entirely comfy-cli's own.

The second MCP-surface capability is the startup **machine snapshot**
(`_machine_snapshot_block`): comfy-cli's own `hardware` payload, quoted verbatim into the
handshake instructions, fail-open on any probe failure — no derived verdict, same guardrail.

The third is the remote MCP **filesystem boundary** (`file_transfer.py`). A path string names
the MCP server's filesystem, never an HTTP client's. `upload_file` may therefore materialize a
bounded inline `{name, mimeType, data}` object long enough to pass its path to the unchanged
`comfy upload` call. In the other direction, HTTP `fetch_outputs` must first call `comfy
download` into owner-only scratch, then publish only contained regular files through a
10-minute HMAC-signed URL on the SAME FastMCP listener. stdio keeps direct paths. This is MCP
protocol adaptation, not an alternate ComfyUI client: bytes still enter/leave ComfyUI only
through comfy-cli; never add a second server, storage API, per-session filesystem, or direct
ComfyUI HTTP call. Inline base64 must never enter subprocess argv or the failure log.

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

The local differentiator: discovery (`nodes`, `search_models`) reads the **live install** —
custom nodes included — not a static catalog.

## Module layout

`server/` is the application package. Its public package API is deliberately small:
`server/__init__.py` exports only `mcp`, `main`, and the 39 tool callables;
`server/tools.py` holds the explicit tool export list without wrapping or duplicating the
callables. The tightly coupled runner state, envelope parser, `--json-stream`
machinery, consent plumbing, tool implementations, and startup logic remain private in
`server/_internal.py`. Tests that exercise private behavior import that owning module
explicitly; do not re-export private helpers from `server/__init__.py`.

`McpApplicationBuilder` creates the one FastMCP instance in the private composition root;
stdio and HTTP are transport adapters selected after registration. Leaf business/client
modules do not import the server package, so dependency edges only point inward from the
composition root. User-entry and application-composition modules live with that root under
`server/`; public model/exception types and reusable leaf support stay at package level.
`server/remote.py` receives the already-built application and never constructs a server or
registers a tool; the private runtime imports it only inside the explicit `serve` branch,
after initialization.

| Module | Owns |
|---|---|
| `server/__init__.py` | stable public Python API: `mcp`, `main`, and the 39 registered tool callables; no private runtime names |
| `server/tools.py` | explicit public tool export list; the objects are the exact callables registered by the private runtime |
| `server/cli.py` | the console script's human argv surface — `--help`, `--version`, and validated `serve` options — plus the installed-metadata version lookup (`_version`) |
| `server/config.py` | immutable Remote MCP listener configuration (`COMFY_MCP_*`), including the non-loopback Host-header policy. It never owns or reads the `COMFYUI_*` target |
| `server/instructions.py` | the `INSTRUCTIONS` constant handed to `FastMCP(..., instructions=...)` — client-handshake text |
| `server/mcp_app.py` | the minimal `McpApplicationBuilder`; owns name/version/base instructions and performs the repository's only `FastMCP(...)` construction |
| `server/remote.py` | transport-only adapter: calls `http_app()` on the supplied shared FastMCP instance and composes it with uvicorn; owns no tools, models, client, or application factory |
| `server/_internal.py` | private composition root, guarded runners, envelope/stream machinery, consent state, tool implementations, and startup logic |
| `file_transfer.py` | remote MCP filesystem adapter: bounded inline upload staging plus same-listener signed output tokens, containment, expiry, and cleanup; no ComfyUI I/O |
| `textutil.py` | pure text helpers: `_tail` / `_stream_tail` (bounded stream tails) and `_redact_url` (userinfo masking) |
| `tcc.py` | macOS protected-folder (TCC) detection + the guidance message |
| `failure_log.py` | the opt-in `COMFY_MCP_DEBUG_LOG` observer: `_log_failure` publishes immutable `_FailureEvent`s and the JSONL observer owns config, rotation, permissions and lazy file state. Its URL scrubbers also mask credentials on the way to the MCP CLIENT and disk |
| `errors.py` | `ComfyCliError`; the "nothing recorded to stop" detector; the `error.details` renderer + per-field char cap |
| `clitext.py` | comfy-cli **human-output** parsing for verbs with no envelope — `Saved:`-block/install-failure extraction, `plain_ok` synthesis, missing-verb/-option probes, `install_node`'s per-pack verdict, echoed-argv forgery guards. Its extractors are the documented cm-cli contract (see architecture rule above) — move or edit byte-for-byte |
| `argv.py` | argument-injection and OS-limit guards for every tool-facing string headed for `subprocess`: shared primitives plus per-domain guards (workflow path, prompt id, download id, extra args, version, node names, log port, model path/filename, upload paths) |
| `target.py` | remote-target resolution/redaction/provenance for run/job tools — `COMFYUI_URL`/`HOST`/`PORT` parsing, `--host`/`--port` forwarding, the local-only `download_model` refusal, and divergence notes on `system_stats`/`free_memory` so an agent doesn't gate a remote run on local numbers |
| `params.py` | param/slot marshaling into comfy-cli argv for `generate`/`run-template`/`set-slot`/`vary`, incl. the structured slot machinery — `SlotOverride`/`SlotVariants` are this module's public TYPES (carve-out below) |
| `client/protocols.py` | `ComfyCliClient`, the outbound engine port used by all application/tool code |
| `client/subprocess_client.py` | the concrete client delegating to the guarded sync/raw/async/streaming subprocess runners; imports no MCP server |
| `client/context.py` | lazy default composition and request-safe `ContextVar` injection; concurrent requests never share a mutable current-client field |

## Transport modes and logging

No arguments serves the shared 39-tool FastMCP application over stdio. `comfy-mcp serve`
passes that exact instance to FastMCP 4's
`http_app(json_response=True, stateless_http=True)` and hands the SDK-owned ASGI app to an
explicit `uvicorn.Server`. Do not create a second FastMCP instance or transport-specific
tool registry. Do not call
`MCPServer.run(transport="streamable-http")`, copy legacy SSE, or patch a private session
method. Never implement JSON-RPC, sessions, SSE, WebSockets, or reconnect here. Both modes
therefore share tool names, schemas, returns, instructions/version, consent, and
`ComfyCliClient`; the Remote layer must never shell out independently.

The one non-MCP route is registered with FastMCP's public `custom_route` on that shared
application: `/downloads/{token}/{filename}` serves only HMAC-signed, unexpired files that
`comfy download` placed inside its owner-only scratch directory. It uses the same ASGI app and
port. Reverse proxies must forward `/downloads/` with the configured MCP path. Do not build a
second file listener or patch SDK sessions to carry files.

FastMCP 4 serves both modern sessionless and legacy handshake protocols. Legacy consent uses
`ctx.elicit`; modern consent returns `InputRequiredResult` and resumes from FastMCP-sealed
request state. Every gate remains fail-closed and request-local. Do not treat a caller's
`confirm_*` argument as human consent when either interactive route is available.

The MCP listener (`COMFY_MCP_HOST`/`PORT`/`PATH`) and the ComfyUI target
(`COMFYUI_URL` or `COMFYUI_HOST`/`PORT`) are different address spaces. Default the listener
to loopback. A non-loopback bind requires explicit allowed Host-header patterns and still is
not authentication; deployment beyond a trusted network belongs behind an authenticated TLS
reverse proxy.

Keep console logs on stderr in both modes. Under stdio, stdout is the JSON-RPC channel; under
HTTP it is technically free but deliberately stays unused so behavior is predictable across
launch modes. Child stdout remains captured data for `envelope/1` parsing only. The opt-in
failure publisher has one JSONL observer. While disabled it returns before any filesystem
effect; while enabled it remains non-propagating and file-only. Inline upload bytes/base64 are
never failure-event fields. Keep this diagnostic-only—do
not introduce a general event bus.

The private runtime reaches leaf modules **module-qualified** (for example,
`failure_log._log_failure(...)`). Patch the owning module
(`monkeypatch.setattr(failure_log, "_FAILURE_LOG_PATH", …)`), not the private runtime — the
wrong one raises `AttributeError`. Tests for runner/tool internals import
`comfy_mcp.server._internal` explicitly and patch it because that module genuinely owns the
tightly coupled runtime state. Public API tests import `comfy_mcp.server` and must assert
that private names stay absent. Carve-out: exception/model TYPES (`ComfyCliError`,
`SlotOverride`, `SlotVariants`) are name-imported inside `_internal` because they ride many
`except`/`isinstance`/tool-signature sites and hold no mutable state a test could patch the
wrong copy of.

## Toolchain

Python ≥ 3.10; pip + setuptools (no `uv.lock` here — comfy-cli bundles `uv` and may write a
stray one into the working directory; gitignored). **comfy-cli is deliberately NOT a declared
dependency** — don't "fix" that by adding one; `pyproject.toml`'s comment says why.
FastMCP is deliberately exact-pinned to `4.0.0b3` and the protocol engine to `mcp==2.0.0`
while FastMCP 4 is beta; upgrade the pair in a dedicated, fully tested change.

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

Remote configuration and same-instance adapter tests live in `test_remote_transport.py`;
in-process FastMCP business flows live in `test_fastmcp_app.py`; the real stdio process flow
lives in `test_stdio_business_flow.py`; real-loopback Streamable HTTP integration lives in
`test_remote_http.py`; transfer edge cases live in `test_file_transfer.py`. Both transport
flows cover the full upload/submit/poll/fetch path and the
same 39-tool discovery result. Also cover modern+legacy negotiation, pre-initialize/stale
requests, concurrency, native tool errors, signed-link tampering/expiry, scratch containment
and cleanup, failure observation/non-disclosure, and clean lifecycle. Never
label fake-engine coverage a real ComfyUI end-to-end test.

**Mock comfy-cli via the shared fixtures in `tests/conftest.py`, never a hand-rolled stub.**
They mirror how `server._internal` spawns the CLI: a spawn-signature change is one edit, not a sweep:

- `envelope(ok=…, data=…, error=…)` — build an `envelope/1` body.
- `patched_run(stdout=…, returncode=…, stderr=…, raises=…, on_spawn=…) -> calls` — the plain
  `--json` path (`subprocess.run`); `calls` records `cmd`/`env`/`timeout`/`encoding` per
  call for exact-argv checks; `on_spawn(cmd)` fires at spawn so the one verb whose answer is
  a FILE writes its `--output`.
- `patched_plain_run(returncode, stdout, stderr) -> calls` — same, for verbs that print
  human text with no envelope (`launch`/`stop`/`generate`).
- `patched_stream(stdout_text) -> procs` — the `--json-stream` NDJSON path
  (`asyncio.create_subprocess_exec`); fake pipes are real `asyncio.StreamReader`s from
  conftest's `stream_reader(text, limit)` — reuse it, never hand-roll an awaitable, so
  buffer limits stay exercised.
- `blocking_stream(first_lines, stderr_text=…) -> procs` — its TIMEOUT counterpart: a child
  that BLOCKS with both pipes open, `wait()` parked until `kill()` — or a 30s net — EOFs them.
- `patched_async_run(stdout=…, returncode=…, stderr=…, hang=…, on_spawn=…) -> procs` — the
  plain-JSON *async* path (`_run_comfy_async`): same spawn and real `StreamReader` pipes as
  `patched_stream`, but parses the capture once at the end, not line-by-line. `hang=True`
  leaves the pipes OPEN so the child never finishes; `kill()` closes them (the process-group
  kill, so a post-kill drain reaches EOF), records `killed`, and fires `on_spawn(cmd)`.

The two spawn paths differ deliberately: the plain `--json` path is synchronous
(`subprocess.Popen` + a bounded `communicate`, off-loaded to a thread pool for async
callers); anything STREAMING or long-lived spawns via `asyncio.create_subprocess_exec` —
nothing blocking runs on the event loop, enforced by ruff's `ASYNC` select. Two async
runners live there: `_run_comfy_streaming` (NDJSON + progress) and `_run_comfy_async`, a
plain-JSON twin of `_run_comfy` for CANCELLATION — cancellation never reaches a `to_thread`
worker, so a client giving up left the child running; it carries the longest-lived children
(foreground `model download`, `workflow_deps`, `upload_file`). Each stream keeps only a
`_STDERR_MAX_CHARS` tail (`_drain_capped_into`; callers widen stdout via `stdout_cap=`),
never `communicate()`'s full capture. `auth_login` (`_start_login`) is a third spawn site.

A local stub is justified only where the call genuinely differs — the `comfy --version` probe
(its own kwargs), multi-call sequenced replies, and `test_wrapper.py`'s `_StderrBlockingProc`
(stdout EOFs fast while `stderr.read()` blocks, which neither streaming fixture models).

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
