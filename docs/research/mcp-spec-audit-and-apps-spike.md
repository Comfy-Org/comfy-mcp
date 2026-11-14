# MCP spec audit and MCP Apps spike

Research only. No protocol or server behavior was changed.

**Date:** 2026-11-14.
**Server:** `comfy-mcp` 0.10.0, stdio, `MCPServer` from the Python `mcp` SDK (`pyproject.toml`: `mcp>=2,<3`).
**Spec compared against:** stable revision **`2026-07-28`**, which `https://modelcontextprotocol.io/llms.txt` lists first and which the SDK reports as `LATEST_PROTOCOL_VERSION`. Previous stable revision is `2025-11-25`. A draft changelog exists after `2026-07-28`; it is not the comparison baseline.
**SDK inspected:** `mcp` **2.3.0**, the newest release on PyPI at audit time (`pip index versions mcp`). It was installed into this environment and read. It was not the version CI happens to resolve on a given day; the constraint allows any 2.x.

Line numbers refer to `src/comfy_mcp/server.py` on `main` at `02dfb3b` unless another file is named.

## What was verified, and what was inferred

| Claim | How it was checked |
| --- | --- |
| `2026-07-28` is the latest stable revision, and the changelog items cited below | Fetched `modelcontextprotocol.io/llms.txt`, `/specification/2026-07-28/changelog`, and `/specification/2026-07-28/basic/lifecycle` |
| SDK 2.3.0 is dual-era (`HANDSHAKE` = `2024-11-05` … `2025-11-25`, `MODERN` = `2026-07-28` only) and ships `mcp.server.apps` | Installed the package and read `mcp_types.version`, `mcp/server/runner.py`, `mcp/server/apps.py`, `mcp/server/elicitation.py`, `mcp/server/mcpserver/context.py` |
| This server's tools, elicitation, progress, images, and cancellation | Read `server.py` and the tests that lock those paths (`tests/test_sdk_conformance.py` and the per-gate files) |
| MCP Apps wire rules, CSP, and `data:` for images and media | Fetched the ext-apps spec `specification/2026-01-26/apps.mdx` (SEP-1865, status Stable) and the Python SDK Apps guide |
| Cloud server PRs #1894 and #2005 | Read via the GitHub API. Both are merged. Bodies were available. The TypeScript sources were not checked out |
| Which chat clients render Apps | Cursor's MCP docs (Apps row = Supported, stdio included), the 2026-01-26 MCP blog (Claude, Goose, VS Code Insiders, ChatGPT), the VS Code blog the same day, and PR #1894's live table from 2026-11-11 |
| That today's Claude Desktop and Cursor still open with `initialize` | **Inferred** from PR #1894, which records both hosts advertising `capabilities.extensions["io.modelcontextprotocol/ui"]` inside `initialize` on 2026-11-11. Not re-tested here |
| That a `2026-07-28` client would fail every consent gate closed | **Inferred** by combining the server's `ctx.elicit()` call with the SDK's `NoBackChannelError` on the modern path. Not exercised against a live modern client |
| `mcp.server.apps` on 2.0.0–2.2.0 | **Not checked.** Only 2.3.0 was installed |

This server was not connected to Claude Desktop, Cursor, or VS Code during the audit.

---

## Part 1 — Spec audit

### How the server sits on the protocol

`main()` serves stdio only (`server.py` `mcp.run(transport="stdio")`, around line 13410). The `MCPServer` is constructed at import with a name, an explicit version, and instructions (`server.py` lines 137–141). It passes none of `extensions`, `icons`, `cache_hints`, or `subscriptions`.

Every user-facing function is an `@mcp.tool()`. There is no `@mcp.resource`, `@mcp.prompt`, or `@mcp.completion`. Tool return annotations are `Any`, so the SDK does not publish an `outputSchema`. Return values that are not `str` / `Image` / `Audio` are pretty-printed JSON inside a single text content block (`mcp/server/mcpserver/utilities/func_metadata.py`, `_convert_to_content`). Exceptions, including `ComfyCliError`, become a tool result with `isError: true` rather than a JSON-RPC protocol error (`mcp/server/mcpserver/server.py` `_handle_call_tool`). That matches SEP-1303 (validation and tool failures should be tool errors so the model can correct itself).

The SDK, not this repo, implements version negotiation. `mcp` 2.3.0 answers a legacy `initialize` for any handshake revision through `2025-11-25` (unknown versions are negotiated down to `LATEST_HANDSHAKE_VERSION`, `2025-11-25`; `runner.py` `_negotiate_initialize`). A request carrying `2026-07-28` in `_meta["io.modelcontextprotocol/protocolVersion"]` is served statelessly, and `server/discover` is implemented by the SDK (`lowlevel/server.py` `_handle_discover`) with `supportedVersions: ["2026-07-28"]`. Modern results get `resultType` and an `io.modelcontextprotocol/serverInfo` stamp from the SDK. `ListToolsResult` defaults `ttlMs` to `0` and `cacheScope` to `"private"`, so the new required cache fields are present without this repo setting them. `ttlMs: 0` means "do not cache."

`2026-07-28` removed the in-request back-channel. Sampling, roots, and `elicitation/create` are not deliverable on that path (`_streamable_http_modern.py` raises `NoBackChannelError`; `resolve.py` `_uses_input_required` sends `InputRequiredResult` only when the negotiated version is at least `2026-07-28` **and** the tool uses the SDK's `Resolve(...)` markers). This server never uses `Resolve`. It calls `ctx.elicit()`, which still sends `elicitation/create` (`elicitation.py` `elicit_with_validation` → `session.elicit_form`).

### Ranked gaps

Effort is S (one module, mostly declarative), M (a few tools plus tests, behavior or schema changes), or L (every consent gate, and the tests that fake `Context.elicit`). Priority is user-facing, not spec-completeness.

| Rank | Item | State today | Gap | User impact | Effort | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Consent on `2026-07-28` (MRTR vs `elicitation/create`) | Form elicitation, one boolean schema, fail-closed. `_elicit_approval` at lines 4970–5050. Callers: spend (`partner_generate`, `run_template`, `run_workflow`), `install_node`, `switch_comfyui_version`, `update_comfyui` when `target="all"`, network-exposing `launch_comfyui` / `restart_comfyui`, and the untracked-server kill. Capability probe: `_client_elicitation_support` at 4718. Timeout 300s (`_ELICIT_TIMEOUT`, line 4763). `confirm_*` arguments are the documented fallback when the client has no elicitation capability | `ctx.elicit()` needs a server-to-client request. On `2026-07-28` the SDK has no back-channel, so the call raises `NoBackChannelError`. The `except Exception` in `_elicit_approval` turns that into "could not confirm … with the user" and does not run the command. The SDK's modern replacement is `InputRequiredResult` (`resultType: "input_required"`), which this server does not return. URL-mode elicitation and `notifications/elicitation/complete` are also absent; the latter was removed again in `2026-07-28` | None on clients that still `initialize` (see the 2026-11-11 note above). The first client that speaks only `2026-07-28` makes every gated tool unapprovable. Spend fails closed, so this does not silently charge credits. Install, version switch, network publish, and untracked kill fail closed the same way | L | **P1** |
| 2 | Tool annotations | `@mcp.tool()` is called with no `annotations=`, no `title=`, no `icons=`. The SDK decorator accepts all three (`MCPServer.tool`). `ToolAnnotations` has `title`, `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`. Omitted hints default, per the spec, to read-only **false**, destructive **true**, idempotent **false**, open-world **true** | The whole catalog is advertised as destructive and open-world. Read-only tools include `server_info`, `which`, `discover`, `nodes`, `search_templates`, `search_models`, `get_template`, `validate_workflow`, `list_workflow_slots`, `list_workflow_notes`, `system_stats`, `get_logs`, `project` (status), `job` status/error/queue, `download` status | Clients that confirm destructive or open-world tools prompt before a status check. Clients that skip confirmation for `readOnlyHint: true` cannot do that here | S | **P2** |
| 3 | `outputSchema` and `structuredContent` | Tools return `Any`. The SDK therefore publishes no `outputSchema` and puts the JSON in `content[0].text`. `Image` returns become `ImageContent` plus that text (`fetch_outputs` line 7083, `preview_media` line 7132). No tool returns a `CallToolResult` with `structuredContent` | Models re-parse a JSON string to branch on `prompt_id`, `status`, `download_id`, `cancelled`. Hosts cannot bind a widget or a form to a schema. `2026-07-28` also loosened `structuredContent` to any JSON value (SEP-2106); nothing here uses that | Agents mis-read a truncated or pretty-printed payload. An Apps viewer has no stable object to render unless it scrapes text or image blocks | M | **P2** |
| 4 | Resource links and embedded resources in tool results | Images are inlined as base64 `ImageContent`, capped at 8 files and 16 MiB raw (`_INLINE_IMAGE_MAX_COUNT` / `_INLINE_IMAGE_MAX_BYTES`, lines 6951–6952), and only for `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`. Video and audio are not inlined. Paths come back inside the JSON text. No `ResourceLink`, no `EmbeddedResource` | A client cannot open the file the tool just wrote except by reading the path out of the text. Large images past the cap, and every video, are path-only. `preview_media` covers video and audio with one PNG contact sheet or waveform, which is the right fallback and already exists | The model sees a path. The person sees a PNG when someone remembered `inline_images=True` or called `preview_media`. A video file itself is never in the tool result | M | **P2** |
| 5 | MCP Apps (`ui://` + tool `_meta.ui`) | No resources are registered. `resources/list` is an empty SDK handler. No tool sets `_meta`. Detail and a proposal are in Part 2 | Apps-capable hosts (Claude, Cursor, VS Code) have nothing to render inline | Same as rank 4, plus no playback UI even when the bytes would fit | M | **P2** |
| 6 | Cancellation of long operations | Streaming runs (`run_workflow`, `run_template`, `partner_generate` with `wait=True`, `job` watch) use `_run_comfy_streaming` / `_run_comfy_async`. `CancelledError` is left to propagate (line 2374) and the child is reaped. Several long tools still hop to a worker thread (`anyio.to_thread` / `asyncio.to_thread`) because a thread cannot be killed: `job` cancel and wait, legacy foreground `download_model`, `install_node`, `switch_comfyui_version`. Comments at lines 9026 and 9580 state that an MCP cancel does not stop the worker | A client that cancels or disconnects during a foreground model download or a threaded wait leaves the `comfy` child running until the tool's own timeout. The streaming path is fine | A cancelled multi-GB download can keep writing. A cancelled watch on the async path stops. The split is easy to get wrong when adding a tool | M | **P2** |
| 7 | Progress on downloads | `notifications/progress` is emitted for workflow and job streams via `ctx.report_progress` (line 2361). The SDK no-ops when the client sent no `progressToken` (`session.py` `report_progress`). `download_model` polls `download-status` and returns a row; it does not emit progress notifications during the transfer | A client that asked for progress on a download sees silence until the final result. `download(action="status")` is the supported poll | Low. The tool already tells the caller to poll. A notification would only help hosts that render a progress bar without a follow-up call | S | **P3** |
| 8 | `tools/list` cache hints | SDK default `ttlMs: 0`, `cacheScope: "private"` on `ListToolsResult`. The tool list does not change for the life of the process (`listChanged` is false unless something calls `ctx.notify_tools_changed`, which nothing does) | A non-zero `ttlMs` would let clients keep the catalog and stabilize prompt caches, which `2026-07-28` calls out. `0` is legal and conservative | Extra `tools/list` traffic and colder prompt caches. No functional miss | S | **P3** |
| 9 | Icons and server `title` | `MCPServer` is constructed without `title`, `website_url`, or `icons`. Tools have neither `title` nor `icons`. The SDK models both (SEP-973, in the spec since `2025-11-25`) | Clients that show an icon or a human title fall back to the function name | Cosmetic | S | **P3** |
| 10 | Logging (`notifications/message`, `logging/setLevel`) | No `ctx.info` / `ctx.log` calls. Python `logging` goes to stderr. `2026-07-28` **deprecates** the logging capability (SEP-2577) and tells stdio servers to log on stderr. `logging/setLevel` is removed; level is per-request `_meta["io.modelcontextprotocol/logLevel"]`, and the server must not emit `notifications/message` unless that field was sent | No gap against the current spec. Adding `notifications/message` now would adopt a deprecated feature | None | — | **not worth a ticket** |
| 11 | Completions | No prompts and no resource templates with arguments. `completion/complete` is only useful for those | Nothing to complete | None | — | **not worth a ticket** |
| 12 | Sampling | Unused. Deprecated in `2026-07-28` (SEP-2577). The SDK marks `sample` deprecated | Correct to leave unused | None | — | **not worth a ticket** |
| 13 | Tasks (`io.modelcontextprotocol/tasks`) | Not negotiated. Long work is already a server-minted handle: `prompt_id` on `job`, `download_id` on `download`. That is the pattern `2026-07-28` tells core servers to use (SEP-2567) after tasks left the core protocol (SEP-2663) | A tasks port would duplicate `job` / `download` and add a second status vocabulary | Clients that only poll tasks and do not call `job` would not see runs. No such client is assumed here | L | **P3, do not do** |
| 14 | URL-mode elicitation | Not used. `auth_login` drives `comfy` login, which owns the browser flow | URL mode is for secrets that must not cross the client. This server never asks for an API key inside a form | None until a tool needs an out-of-band secret the CLI cannot collect itself | — | **not worth a ticket** |
| 15 | OAuth, sessions, `Mcp-Session-Id`, SSE resumability | stdio only. No HTTP transport, no OAuth | The `2026-07-28` auth and HTTP-transport changes do not apply | None | — | **not applicable** |
| 16 | Protocol version negotiation itself | Inherited from the SDK, dual-era, as described above. Server code never reads `ctx.protocol_version`. Comments still describe identity as an `initialize` field (`_server_version`, lines 104–128) | Negotiation works without a local change. There is no test that a `2026-07-28` `server/discover` and a `2025-11-25` `initialize` both succeed against this process. Rank 1 is the behavior that negotiation change breaks | A future SDK that drops handshake support would fail closed on consent (rank 1) with no test failing first | S, as a test inside the rank-1 ticket | covered by rank 1 |

Tool names already match SEP-986 (`^[A-Za-z0-9._-]{1,128}$`). The SDK warns on a bad name at registration; nothing here needs renaming.

`_meta` on the wire: the SDK stamps `serverInfo` on modern results. This repo does not add trace context (`traceparent` / `tracestate` / `baggage`, SEP-414) and does not forward it into the `comfy` child. That is a P3 observability nicety, folded into "not worth a ticket" with logging. Workflow graphs do use a different `_meta.title` (ComfyUI canvas labels, `instructions.py` and `emit_partner_workflow`). That is ComfyUI's key, not MCP's.

### Tickets worth writing

Only ranks 1–6. Ranks 7–9 are real and small; they are listed so they are not rediscovered, and they should not jump the queue.

#### 1. Return spend and danger confirmations as `InputRequiredResult` on `2026-07-28`

`ctx.elicit()` still sends `elicitation/create`, which `mcp` 2.3.0 cannot deliver once the client speaks `2026-07-28` (no back-channel; `NoBackChannelError`). `_elicit_approval` catches that and refuses the command. Handshake clients (`2025-11-25` and earlier) keep working, and the 2026-11-11 cloud-server check still saw Claude Desktop and Cursor using `initialize`, so this is latent. It becomes user-visible the day a host stops handshaking: every gated tool (`partner_generate`, `run_template`, `run_workflow` when spending, `install_node`, `switch_comfyui_version`, `update_comfyui` for `target="all"`, network-exposing launch/restart, untracked kill) fails closed with no prompt.

Keep one fail-closed body. Branch on `ctx.protocol_version`: handshake revisions keep `ctx.elicit()`; `2026-07-28` and later return the SDK's `InputRequiredResult` (or the `Resolve` / `Elicit[...]` helper in `mcp.server.mcpserver.resolve`, which already does that split) and resume from `ctx.input_responses` / `ctx.request_state` on the retry. Do not add a second copy of the yes/no wording. `confirm_*` stays the escape hatch when the client has no elicitation capability. URL-mode elicitation is out of scope.

Acceptance:

- A fake session negotiated at `2025-11-25` still receives one `elicitation/create`, and decline / cancel / timeout still spend nothing and install nothing.
- A fake session negotiated at `2026-07-28` receives `resultType: "input_required"` and no `elicitation/create`. A retry with `approve: true` runs; a retry with decline or a missing response does not.
- `tests/test_sdk_conformance.py` grows a case for the new call shape, next to the existing `Context.elicit` signature check.
- An in-process smoke test calls `server/discover` and `initialize` against this server and records which versions come back.

#### 2. Set tool annotations so read-only tools are not advertised as destructive

Every `@mcp.tool()` omits `ToolAnnotations`. The spec's defaults mark a tool destructive and open-world. Hosts that confirm on those hints prompt before `server_info`, `nodes`, or `job` status.

Set `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint` explicitly on each tool. Reads of local state are `readOnlyHint=True`, `destructiveHint=False`, `openWorldHint=False`. `search_templates`, `search_models`, `download_model`, `partner_generate`, and `auth_login` are `openWorldHint=True`. `free_memory`, `stop_comfyui`, `restart_comfyui`, `install_node`, `switch_comfyui_version`, `set_workflow_slot`, and `update_comfyui` are `destructiveHint=True` and not read-only. `job` and `download` are one tool with several actions; annotate for the worst action (not read-only) and say so in the description, rather than inventing a per-action annotation the spec does not have. Add `title` only where the function name is opaque (`which`, `discover`, `nodes`).

Acceptance:

- `tools/list` for `server_info` has `readOnlyHint: true`, `destructiveHint: false`, `openWorldHint: false`.
- `tools/list` for `install_node` and `free_memory` has `destructiveHint: true` and `readOnlyHint: false`.
- A unit test builds the tool list from the real `MCPServer` and fails if a new `@mcp.tool()` ships with `annotations is None`.

#### 3. Publish `structuredContent` for the tools agents branch on

Tool results are pretty-printed JSON in a text block because every tool is annotated `-> Any`. Hosts and a future Apps view then have to scrape that string. `2026-07-28` still requires a useful `content` array beside `structuredContent`; do not return structured content alone.

Add real return models, or `structured_output=True` where the SDK can derive a schema, for `job`, `download` / `download_model`, `fetch_outputs`, `server_info`, and `run_workflow` / `run_template` / `partner_generate`. Keep the text block as the JSON serialization of the same object so a client that ignores `structuredContent` does not regress. Leave the long tail of catalog tools as text until one of them grows a consumer.

Acceptance:

- `tools/list` includes an `outputSchema` for `job` and `fetch_outputs`.
- A call result for those tools has `structuredContent` equal to the object that used to be the only payload, and `content` still contains that payload as text.
- Existing tests that compare the returned dict still pass, because the Python return value does not change; only the wire conversion does. Add one test that inspects `CallToolResult.structured_content`.

#### 4. Surface saved media as content the host can render, without raising the 16 MiB cap

`fetch_outputs(inline_images=True)` and `preview_media` already append `ImageContent` under a count and byte cap. Video and audio never become content blocks. Paths are buried in JSON. The SDK's `Image` helper base64-encodes the file (`mcp/server/mcpserver/utilities/types.py`). There is no video helper.

Keep the cap. For files that fit, add the bytes the widget in Part 2 will read (image blocks as today; a small video or audio clip only under the same byte budget, carried as a base64 data URL inside `structuredContent` or as `AudioContent` for audio). For files that do not fit, add a `ResourceLink`-style record in `structuredContent` with the absolute path, media type, and byte size, and keep using `preview_media`'s PNG as the inline preview. Do not start a local HTTP server and do not point the client at `127.0.0.1:8188`. That URL is what the network-exposure gate exists to keep off the LAN, and an Apps sandbox will not load it without a CSP exception hosts often refuse for loopback.

Acceptance:

- `fetch_outputs(inline_images=True)` still refuses a ninth image and still refuses once raw bytes would pass 16 MiB.
- A video result includes the on-disk path and size in `structuredContent`, and includes image bytes only for the contact sheet when `preview_media` was the tool that ran.
- No new listening socket.

#### 5. Honor MCP cancellation on the remaining long synchronous tools

Streaming tools already kill the `comfy` child when the request is cancelled. `download_model`'s legacy foreground path, `job(action="wait")` on the thread pool, `install_node`, and `switch_comfyui_version` document that a `CancelledError` does not reach the worker. Move the long ones that do not need to finish after the client has gone onto `_run_comfy_async`, matching `run_workflow`. Leave a thread only where the work must finish after cancel (a version switch that has already started rewriting the install is a judgment call; say which side you picked in the PR).

Acceptance:

- Cancelling a legacy foreground download kills the child and does not leave a truncated file without the error the tool already documents for timeout.
- A test drives `CancelledError` into the download wait and asserts the process group was signalled, using the existing fake-spawn fixtures in `tests/conftest.py`.

#### 6. MCP Apps output viewer

See Part 2. This is the same change as the proposal there; it is one ticket, not a second design.

---

## Part 2 — MCP Apps spike

### Spec, in the form a stdio server has to implement

MCP Apps is the extension `io.modelcontextprotocol/ui` (SEP-1865, stable spec dated 2026-01-26 in the `ext-apps` repo). It is transport-agnostic. The host, not the server, opens an iframe.

1. The server lists a resource whose URI uses the `ui://` scheme and whose MIME type is exactly `text/html;profile=mcp-app`.
2. A tool points at it with `_meta.ui.resourceUri`. Optional `_meta.ui.visibility` is `["model"]`, `["app"]`, or both. Default is both.
3. The host `resources/read`s that URI, puts the HTML in a sandboxed iframe, and delivers the tool result with `ui/notifications/tool-result` over `postMessage`. The server never sees `ui/*` messages.
4. The tool's `content` array must still be meaningful. The model reads `content`. The iframe is for the person. A client that did not advertise the extension ignores `_meta.ui` and shows `content`.

CSP is declared on the **resource** `_meta.ui.csp`, not on the tool. The spec's default policy is `default-src 'none'`, and `img-src` / `media-src` allow `'self'` and `data:` with no extra origins. `connect-src` allows nothing until `connectDomains` is set. `frame-src` is `'none'` until `frameDomains` is set. Inline script is allowed (`script-src 'self' 'unsafe-inline'`).

### What the Python SDK already does

Verified in `mcp` 2.3.0, module `mcp.server.apps`:

- `Apps()` is an `Extension` with `identifier == "io.modelcontextprotocol/ui"`.
- `@apps.tool(resource_uri="ui://…")` stamps `_meta.ui.resourceUri` and refuses a hand-written `meta["ui"]`.
- `add_html_resource(uri, html, csp=ResourceCsp(...), ...)` serves `text/html;profile=mcp-app` and writes CSP onto the resource `_meta.ui`.
- `MCPServer(..., extensions=[apps])` advertises the extension and refuses to construct if a tool's URI has no registered resource.
- `client_supports_apps(ctx)` is true only when client capabilities include `extensions["io.modelcontextprotocol/ui"].mimeTypes` containing `text/html;profile=mcp-app`.

`MCPServer.tool()` also accepts `meta=` directly, and `Tool.meta` serializes as `_meta`. A tool can stay on `@mcp.tool(meta={"ui": {"resourceUri": "ui://comfy-mcp/output-viewer"}})` while an `Apps()` instance that only calls `add_html_resource` serves the document and advertises the extension. That avoids moving forty tools onto a second decorator. `@apps.tool` is the better API if the tool is new; it is awkward for `fetch_outputs`, which already exists as `@mcp.tool()`.

Not verified: whether `mcp` 2.0.0 through 2.2.0 export `mcp.server.apps`. A widget change should bump the floor to the first 2.x that has it (2.3.0 is known to) rather than assume every `mcp>=2` install can import it.

`MCPServer` is built at import time (line 137), before the tool decorators run. The `Apps()` instance has to exist before that constructor so it can be passed as `extensions`. Tools may keep using the `mcp` object afterwards.

### Client support

| Client | Apps | Basis |
| --- | --- | --- |
| Claude Desktop and Claude web | Yes, as of the 2026-01-26 MCP blog | Blog. PR #1894 then **verified live** on Claude Desktop 5.1231.2 (2026-11-11): image, video, and shortlink download rendered. claude.ai web was not in that run |
| Cursor | Yes, including stdio | Cursor MCP docs: "Apps (extension) — Supported", and stdio is a first-class transport. PR #1894 verified an image on Cursor 3.21.13; video and download were not tried there |
| VS Code | Announced 2026-01-26 in Insiders, stable planned the following week | VS Code blog. Not re-checked against a current stable build in this audit |
| Goose, ChatGPT | Named by the 2026-01-26 MCP blog | Not relevant to this local stdio server unless someone points those hosts at it |
| Codex CLI | No Apps UI; text fallback worked | PR #1894 live table, Codex 0.162.0 |
| Claude Code | Not shown | PR #1894: blocked by an org setting, not by the protocol |

The official client matrix (`modelcontextprotocol.io/extensions/client-matrix`) lists the same clients. The checkmarks did not survive the fetch used here, so the table above does not cite that page for a yes/no.

PR #1894 also recorded that Desktop and Cursor advertised the extension inside `initialize`. They were still handshake-era clients on 2026-11-11. Apps does not require `2026-07-28`. Shipping a widget does not require the rank-1 MRTR work. The two efforts are independent.

### How a widget should show local images and video

Three ways to get pixels into the iframe, and only one fits this server.

**Inline base64 in the tool result, rendered as a `data:` URL.** The host already delivers the tool result to the iframe. The spec's default CSP allows `data:` for `img-src` and `media-src` with an empty `resourceDomains` and an empty `connectDomains`. `fetch_outputs` and `preview_media` already produce `ImageContent` this way, under a 16 MiB / 8-file cap. The widget reads `content` blocks of type `image` and sets `img.src` from the base64. No network, no CSP exception, no local socket. This is the right path for still images and for the PNG contact sheet `preview_media` makes from video and audio.

**A second `resources/read` of the file.** The iframe cannot read the disk. It would have to ask the host (`resources/read` or `tools/call`), and the bytes would still cross as base64. That is the same cap with an extra round trip. Skip it.

**An `http://127.0.0.1:8188` URL** (ComfyUI `/view`). Reject this. The default CSP blocks it. Adding `resourceDomains: ["http://127.0.0.1:8188"]` asks the host to allow loopback from a sandboxed origin; private-network and mixed-content rules often refuse even when the server asks. It also points a UI at the local ComfyUI port, which the launch gate treats as sensitive the moment it is not loopback-only. The cloud viewer refuses anything that is not `http(s)` and still only allowlists public storage origins. Localhost is a worse version of the problem they declined to take on.

Video files that exceed the byte cap should not be stuffed into the tool result. Show the contact-sheet PNG when the caller has one, and show the absolute path and size as text. Do not auto-call `preview_media` from inside `fetch_outputs`; that would be new product behavior. The model already has `preview_media`.

`file://` URLs in the iframe are not a plan. The sandbox origin is not the user's disk, and `media-src` does not allow `file:`.

### Graceful fallback

Do all of the following, or the widget is a regression for Codex, Claude Code, and any host that never negotiates the extension:

- Register the resource and set `_meta.ui.resourceUri` unconditionally. Hosts that do not understand it ignore the field. Gating registration on `client_supports_apps` is unnecessary and would make `resources/list` depend on which request is in flight; `2026-07-28` wants list results cacheable and identical across connections.
- Always return the same text payload the tool returns today, including paths. The model needs the path even when a person can see the picture. PR #1894 learned this the hard way: before they appended a "do not repeat the URLs" note, Claude Desktop printed signed URLs next to the viewer. For this server the path is the useful fact, so do not tell the model to hide it. A one-line note is optional and only worth it if a live host starts echoing base64.
- Keep `inline_images` default and the byte cap unchanged. Apps support must not start inlining video.
- `client_supports_apps` is available if a later change wants a shorter text on Apps hosts. It is not required for the first widget.

### What the cloud server's PRs give us

Both PRs were readable. **Do not copy their TypeScript.** `AGENTS.md` forbids taking code, patterns of structure, or dependencies from `Comfy-Org/comfy-cloud-mcp-server`. The notes below are behavior to re-decide locally, not a porting guide.

**PR #1894** (merged 2026-11-11), "MCP Apps output viewer on `get_output` / `get_batch_output`", plus an upload widget behind a flag. Useful observations:

- One resource, `ui://comfy-cloud/output-viewer`, MIME `text/html;profile=mcp-app`, linked from `get_output` and `get_batch_output` via `_meta.ui.resourceUri`. `wait_for_job` was left alone.
- The HTML is a self-contained string so the build does not need a bundler. The same constraint fits this repo (no frontend toolchain).
- The view speaks the ext-apps bridge itself: `ui/initialize` at protocol `2026-01-26`, then `ui/notifications/initialized`, then handles `tool-result`, `host-context-changed`, and `ui/resource-teardown`. It reports `ui/notifications/size-changed`. It does not call tools and does not `fetch`.
- Media comes from `structuredContent`, as `http(s)` URLs only. CSP `resourceDomains` lists `https://storage.googleapis.com` and the API origin. No `connectDomains`. Download uses `ui/open-link` and only for an `/api/s/<id>` shortlink, never a signed storage URL.
- DOM text goes through `textContent`. No `innerHTML`. A failed media element becomes "Preview unavailable."
- `content[]` stays byte-identical for non-Apps hosts. An extra note is appended only when the client advertised the extension.
- Live result on stdio against staging: Claude Desktop passed image, video, and download; Cursor passed image; Codex kept the text fallback.
- The upload widget (`ui://comfy-cloud/upload-widget`, flag off by default) exists because cloud uploads are a credential-free PUT to a grant URL from the browser. It is not a fit here. Local files are already paths the model can pass to `upload_file` / `run_workflow`. Porting it would also need a CORS origin the sandbox does not have. Skip it.

**PR #2005** (merged 2026-11-13) extracts one shared `postMessage` bridge that four HTML views had each inlined. The lesson for this repo: the first widget can keep a short bridge in its own string. A second widget should import that bridge rather than paste it. The cloud views in that PR (output viewer, catalog picker, run-settings, upload) are cloud-catalog and cloud-run UI. This server's catalog is `search_templates` text. Do not start with a picker.

### Proposal: one widget, the output viewer

Ship a single inline viewer for `fetch_outputs` and `preview_media`. Do not ship a second widget in the same change. A job-progress panel duplicates `notifications/progress`. An upload picker solves a cloud problem this process does not have.

**Behavior**

- Resource `ui://comfy-mcp/output-viewer`, title "Output preview", MIME `text/html;profile=mcp-app`.
- CSP: omit `connectDomains` and `resourceDomains`. The document only uses `data:` and text.
- On `tool-result`, walk `content` for `type: "image"` and render each with a `data:` URL. Walk `structuredContent` (once ticket 3 exists) for a list of files `{name, path, media_type, bytes, data_url?}`. Render `data_url` in `<img>` or `<video>` / `<audio>` only when it is a `data:` URL. Render `path` and size with `textContent`. If there is no image block and no `data_url`, show the path and a line that says the file is on disk and was not inlined.
- Theme: read `host-context-changed` and set `color-scheme`. No other host calls.
- No `fetch`, no `tools/call`, no `innerHTML`, no `file://`, no `http://127.0.0.1`.

Until ticket 3 lands, the viewer can ship against `ImageContent` blocks alone. `preview_media` and `fetch_outputs(inline_images=True)` already return those. `structuredContent` makes video-under-cap and the path list cleaner; it is not a blocker for stills.

**Files**

- `src/comfy_mcp/output_viewer.py` — leaf module. Constants for the URI, the MIME type, and the HTML string. No import of `server`. This matches the module rule in `AGENTS.md`.
- `src/comfy_mcp/server.py` — build `Apps()`, `add_html_resource`, pass `extensions=[apps]` into `MCPServer`. Add `meta={"ui": {"resourceUri": OUTPUT_VIEWER_URI}}` on `fetch_outputs` and `preview_media` only.
- `tests/test_output_viewer.py` — in-process, no host. Assert `resources/read` returns the MIME type and the HTML, the HTML contains no `fetch(` and no `innerHTML`, `tools/list` sets `_meta.ui.resourceUri` on those two tools and not on `run_workflow`, and a `fetch_outputs` result still includes the text payload when images are also returned.
- `pyproject.toml` — raise the `mcp` floor to the minor that exports `mcp.server.apps` (verified for 2.3.0). Keep `<3`.

**Effort:** M. Most of the work is the HTML string and the tests that pin it. Wiring is small because the SDK registers the resource and the capability. The live check is the expensive part, and it is manual: Claude Desktop or Cursor, one still image, one `preview_media` of a short video, and one client that is not Apps-capable (the text and the image block must still show).

**Risks**

- Base64 of a 16 MiB image is about 22 MiB of JSON. Hosts may truncate tool results before the iframe sees them. The cap stays; the viewer must tolerate a missing image block.
- HTML inside a Python string is invisible to ruff. The test that bans `fetch(` and `innerHTML` is the stand-in, copied in spirit from PR #1894's unit tests, not from their source.
- `MCPServer(..., extensions=[apps])` is an import-time change. A mistake fails every tool, not just the viewer. The "resource URI with no resource" check in the SDK fails at startup, which is the failure we want.
- Setting `meta.ui` on `@mcp.tool` while also decorating the same function with `@apps.tool` would register it twice. Use one of the two.
- A later `2026-07-28`-only host still needs rank 1 for consent. The viewer does not.
- Copying `output-viewer-resource.ts` would violate `AGENTS.md` and would be wrong anyway: that file's media model is public `https` URLs and shortlinks.

**Second widget, later, only if the first one is used.** A catalog browser over `search_templates` is the only other local surface that is visual. It needs `structuredContent` for template rows and a decision about whether the model or the person is picking. It is not part of this spike's recommended first change.
