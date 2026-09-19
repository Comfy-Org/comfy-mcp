# Agent architecture and testing reference

This document holds the detailed rationale behind the enforceable rules in
[`../AGENTS.md`](../AGENTS.md). Keep the root guide thin; update this reference
when a change affects the contracts described below.

## Thin-wrapper compositions

Application and tool code reaches the `comfy` binary only through
`ComfyCliClient` and the guarded `_run_comfy*_impl` runners. The following
multi-call tools are allowed because their value is in orchestration while
their answers remain comfy-cli's:

- `fetch_template` runs `templates fetch` and then `validate`, returning the
  engine's validation result for the written template.
- `download_model` starts `model download --background` and polls
  `model download-status`; it must never derive progress from local file size.
- `install_node` reads `comfy env` before prompting because legacy Manager
  clones cannot provide `cm-cli`. The preflight fails open, shares helpers with
  `workflow_deps`, and the install verdict comes from cm-cli's own text through
  `_extract_install_failures`.
- `workflow_deps` must use the `comfy node deps-in-workflow --output` file
  contract because the engine emits no envelope. Return the manifest as
  written except for credential scrubbing through `failure_log._scrub_text`.
- `restart_comfyui` uses `comfy stop --dry-run` and `comfy stop` to identify and
  recycle a port. Never substitute process inspection or an HTTP probe.

Text parsing is allowed only where comfy-cli has no envelope. Preserve the
one-directional extractors in `clitext.py`: a changed engine sentence should
regress the related operation rather than create false success.

## Confirmation contracts

The MCP server adapts dangerous engine operations to interactive confirmation;
it stores no durable consent of its own. `_elicit_approval` is the single
fail-closed body and `_ApprovalWording` supplies per-gate identity and wording.

The spend gates intentionally differ:

- `partner_generate` wraps `comfy generate`, which always spends. It always
  prompts and may honor `spend.auto_confirm` / comfy-cli's durable generate
  consent before forwarding `--yes` or `--allow-spend`.
- `run_template` and `run_workflow` are normally free. They prompt only when
  `confirm_spend=True` unlocks spending and never inherit generate consent.
  Their shared policy is `_resolve_optin_spend_consent`.
- `run_template` can trust its verb's capability. `run_workflow` must probe
  `comfy run --help` for the spend flag; preserve the explaining docstring.

Non-spend gates follow the danger rather than a shared category:

- `switch_comfyui_version` and `install_node` prompt on every call.
- `install_node.names` accepts registry slugs only; the prompt never authorizes
  installing an arbitrary URL.
- `update_comfyui` prompts only for `target="all"`, which updates third-party
  packs that comfy-cli does not gate.
- Launch and restart prompt only when `extra_args` expose unauthenticated
  ComfyUI to the network.
- Restart's kill gate is state-scoped and fires only after launch loses the
  port to an untracked process.

Modern approval state is request-local and resumed from FastMCP-sealed request
state. Legacy clients use elicitation. A caller's `confirm_*` argument is never
human consent while either interactive mechanism is available.

## Project and startup adaptation

comfy-cli resolves `project/1` by walking upward from its process working
directory. MCP clients do not provide a persistent shell cwd, so this server
anchors project calls by passing `cwd=` to its own spawns. Status and init
verdicts still belong to comfy-cli.

`_machine_snapshot_block` may quote comfy-cli's hardware payload in handshake
instructions. It must not derive a machine verdict and must fail open when the
probe is unavailable.

## Remote filesystem boundary

The remote schema follows the command-oriented contract:

- `upload_file(file_path, client_os)` treats `file_path` as client-local. Stdio
  passes it to `comfy upload`. HTTP mints a five-minute, single-use PUT command
  on the shared FastMCP listener, streams at most 50 MiB to owner-only scratch,
  invokes the guarded async upload runner, and always removes scratch.
- HTTP `fetch_outputs` runs `comfy download` into owner-only scratch and
  publishes only contained regular files through five-minute HMAC-signed URLs.
  Stdio keeps direct output paths.
- Download lease release belongs in the ASGI response's `finally` path so a
  disconnect cannot bypass cleanup. Once publication succeeds, the signed URL
  store owns the directory; post-publish result construction must not clean it.

Behind a reverse proxy, commands must contain the complete client-facing
scheme, host, and port. The deployment contract requires the proxy to sanitize
and replace `Forwarded` / `X-Forwarded-*` and protect the MCP and transfer
routes. If the platform strips those headers, configure a credential-free
HTTP(S) origin through `COMFY_MCP_PUBLIC_URL`.

The headers affect only the capability command returned to that caller; they
do not redirect another client's transfer. An application-level proxy-trust
switch would duplicate the deployment's trust policy and break the convention
that resources belong to the environment where the user invoked the tool.

## Module ownership

- `server/__init__.py`: stable public application/tool API only.
- `server/tools.py`: exact public tool callable list, without wrappers.
- `server/cli.py`: human argv surface and installed metadata version.
- `server/config.py`: immutable listener configuration, never ComfyUI target
  variables.
- `server/instructions.py`: FastMCP handshake instructions.
- `server/mcp_app.py`: the only `FastMCP(...)` construction.
- `server/remote.py`: transport-only ASGI/uvicorn adapter.
- `server/_internal.py`: composition root, runners, envelopes, consent, tools,
  and startup.
- `file_transfer.py`: same-listener capability routes and scratch lifecycle.
- `failure_log.py`: opt-in non-propagating JSONL observer and scrubbing.
- `textutil.py`: bounded text tails and URL-userinfo redaction.
- `tcc.py`: macOS protected-folder detection and guidance.
- `errors.py`: `ComfyCliError`, stop-state detection, and bounded detail text.
- `clitext.py`: documented human-output extraction contracts.
- `argv.py`: injection and OS-limit guards for all subprocess-bound strings.
- `target.py`: ComfyUI target resolution, redaction, and forwarding.
- `params.py`: generate/run-template slot and parameter marshaling.
- `client/protocols.py`: outbound engine port.
- `client/subprocess_client.py`: concrete guarded-runner delegation.
- `client/context.py`: lazy defaults and request-safe `ContextVar` binding.

Leaf modules remain reusable and do not import the server package. The private
runtime accesses leaf modules module-qualified so tests patch the actual owner.
Exception/model types may be imported by name where they appear repeatedly in
exceptions, `isinstance`, and tool signatures.

## Transport and logging details

No arguments starts stdio. `comfy-mcp serve` passes the exact same application
to `http_app(json_response=True, stateless_http=True)` and then to an explicit
`uvicorn.Server`. Do not call another server runner or patch private SDK session
methods.

The shared listener registers `/api/uploads/{token}` and
`/downloads/{token}/{filename}` with FastMCP's public `custom_route`. Reverse
proxies must forward those paths with the MCP path and preserve the public
origin headers.

Console output stays on stderr in both modes. Stdio stdout is JSON-RPC; child
stdout is captured data for envelope parsing. The optional failure log returns
before any filesystem effect while disabled and remains file-only and
non-propagating while enabled. Upload bytes are never failure-event fields.

## Test organization and fixtures

Default tests use no real ComfyUI and no real `comfy` binary. Runner/parser
coverage lives in `test_wrapper.py` and `test_parser.py`; tool groups own their
tests. Remote transport/configuration tests live in
`test_remote_transport.py`; in-process FastMCP flows in
`test_fastmcp_app.py`; loopback HTTP integration in `test_remote_http.py`; and
transfer edges in `test_file_transfer.py`.

`tests/e2e/test_smoke.py` is the explicitly opt-in local live suite. It uses the
real comfy-cli and ComfyUI, including the real stdio MCP process. The deployed
HTTP suite requires `COMFY_MCP_TEST_URL` and `COMFY_MCP_TEST_WORKFLOW`, behaves
only as a client, and never hardcodes deployment details.

Shared fixtures mirror each spawn boundary:

- `fake_comfy_client` is the shared `ComfyCliClient` port double.
- `envelope` creates `envelope/1` bodies.
- `patched_comfy_run_sequence` scripts ordered composition-level replies.
- `patched_run` and `patched_run_sequence` cover bounded plain JSON spawns;
  `patched_plain_run` covers human output.
- `patched_stream`, `blocking_stream`, and `stderr_blocking_stream` use real
  `asyncio.StreamReader` pipes and model streaming timeout/cancellation edges.
- `patched_async_run` covers long-lived cancellable plain-JSON children.

If a test needs a distinct sequence, blocking behavior, port collision, or
client method, extend `tests/conftest.py`; a different call shape is not a
reason to create a local double.

Synchronous plain JSON uses `Popen` plus bounded `communicate` and moves off the
event loop for async callers. Streaming and long-lived work use
`asyncio.create_subprocess_exec`. Cancellation must kill the process group and
drain real pipes to EOF. Keep only bounded output tails unless a documented
envelope requires a wider stdout cap. `auth_login` remains its separate third
spawn site.
