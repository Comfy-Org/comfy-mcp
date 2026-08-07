# Agent Guidelines — comfy-mcp

A short guide for AI agents (and humans) here. **Keep it in sync:** a PR that
changes the architecture rule, the toolchain or the tool set updates it too.
`comfy-mcp` is a small, standalone [MCP](https://modelcontextprotocol.io) server
that lets an agent drive a user's **local** ComfyUI: a **thin wrapper over
[`comfy-cli`](https://github.com/Comfy-Org/comfy-cli)**, which is the engine.

## The architecture rule — thin wrapper only (read this first)

Every tool is a passthrough to the `comfy` binary. There is exactly one way to
reach comfy-cli: the `_run_comfy(*args)` helper in `src/comfy_mcp/server.py`,
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
rather than in new logic — `fetch_template` runs `templates fetch` then
`validate`, telling the caller whether the template it just wrote can actually
run here, and `download_model` submits `model download --background` then polls
`model download-status` so a multi-GB transfer does not hold the MCP request
open. That stays inside the rule: every call goes through `_run_comfy`, the
verdict is comfy-cli's own, and no product behavior is added here. What would
breach it is deriving the answer here (parsing the graph, keeping a table of what
is "supported") instead of asking the engine — for `download_model`, sizing the
file on disk to decide it finished rather than reading its `status`.
`install_node` composes for a different reason: `comfy node install` runs
Manager's `cm-cli`, which a legacy clone under `custom_nodes/` cannot provide, so
it reads `comfy env`'s manager fields BEFORE the consent prompt rather than ask a
user to authorize third-party code on a call that cannot succeed. It fails OPEN,
and shares `workflow_deps`' two helpers so the two cannot drift. It also reads
its VERDICT from the printed text (`_extract_install_failures`) — cm-cli prints a
pack's failure before it consults `--exit-on-fail`, so a pack that never
installed came back as `ok: true` — on the standing of `_extract_saved_paths`: no
envelope, so the text is the only channel and the verdict is cm-cli's own
sentence, matched one-directionally so a wording change regresses rather than
fails every install. `workflow_deps` reads its answer off DISK because the engine
leaves none — `comfy node deps-in-workflow` emits no envelope and REQUIRES an
`--output` path — so the temp-file round trip is that contract, and the manifest
goes back as written bar `failure_log._scrub_text` masking repo-URL credentials.
`restart_comfyui` composes a THIRD pair: `comfy stop --port <p> --dry-run` /
`comfy stop --port <p>` say who holds a leftover port and recycle it — never a
`psutil`/HTTP check here, since that verdict stays comfy-cli's.

The one thing that legitimately lives here rather than in comfy-cli is **MCP
protocol surface** — capabilities comfy-cli has no way to express. Today that is
the per-call confirmation on the tools that can spend money, destroy local state,
run third-party code, kill a process, or expose the machine: `partner_generate`,
`run_template`, `run_workflow`, `switch_comfyui_version`, `install_node`,
`update_comfyui` when `target="all"`, the `launch_comfyui` / `restart_comfyui`
pair when `extra_args` would publish ComfyUI to the network, and again to kill
an untracked server. comfy-cli owns the
credit-spend interlock and the durable "always proceed" (`comfy generate consent
always`); this server only raises the confirmation over MCP **elicitation** —
the protocol's y/N prompt — then forwards the answer as `--yes` /
`--allow-spend`, or (for the five the CLI does not gate at all) refuses to run
the command. It stores no consent of its own, and all share one fail-closed body,
`_elicit_approval`; give a new gate its own `_ApprovalWording`, not a second
copy. Adding *product* behavior here is still a guardrail breach; adapting
comfy-cli's contract to an MCP primitive is this repo's job.

The three *spend* gates — `partner_generate`, `run_template`, `run_workflow` —
differ where the engine's own shape differs, and those differences are
load-bearing: `comfy generate` always spends, so it always prompts and honors
`spend.auto_confirm`, while `comfy run-template` and `comfy run` are usually
free and never read that setting — so those two prompt only when
`confirm_spend=True` asks to unlock spending, and treat the generate-scoped
always-proceed as granting nothing. That shared opt-in policy is one body,
`_resolve_optin_spend_consent`, whose per-verb wording is an argument the way
`_ApprovalWording` is `_elicit_approval`'s. What still differs is the capability
SIGNAL: `run_template` can trust the verb, `run_workflow` must PROBE `comfy run
--help` for the flag — its docstring says why. `switch_comfyui_version`,
`install_node` and `update_comfyui` sit outside that axis: none spends credits
and the CLI gates none, so their prompt is this server's only gate, with no
always-proceed to read. TWO are ARGUMENT-scoped, the danger being the argument
and not the verb: the launch pair prompts only for `extra_args` publishing an
unauthenticated ComfyUI (`_network_exposing_args`), `update_comfyui` only for
`target="all"`, which pip-installs every third-party pack (`comfy`/`cli` never
prompt). `install_node` is NOT one — it prompts on EVERY call, its `names`
argument being a RESTRICTION to registry slugs refusing a URL, as the prompt
promises a REGISTRY pack. `restart_comfyui`'s kill gate is STATE-scoped
instead, firing mid-sequence once a launch loses the port. Mirror the engine's
contract per tool; never generalize one gate onto another.

The local differentiator: discovery (`search_nodes`, `get_node`, `search_models`)
reads the **user's live install** — custom nodes included — not a static catalog.

## Module layout

`server.py` holds the wrapper core (`_run_comfy`, the envelope parser, the
`--json-stream` machinery, the spend-consent plumbing) and every `@mcp.tool()`.
Eight **leaf** modules sit under it — none imports `server`, so the dependency
edges only ever point one way:

| Module | Owns |
|---|---|
| `textutil.py` | pure text helpers: `_tail` / `_stream_tail` (bounded stream tails) and `_redact_url` (userinfo masking) |
| `tcc.py` | macOS protected-folder (TCC) detection + the guidance message |
| `failure_log.py` | the opt-in `COMFY_MCP_DEBUG_LOG` failure log (its config, its module state, and `_log_failure`) **and the URL scrubbers** — `_scrub_text` / `_scrubbed_stream_tail` also mask credentials on the way to the MCP CLIENT, not just to disk |
| `instructions.py` | the `INSTRUCTIONS` constant handed to `MCPServer(..., instructions=...)` — the client-handshake text |
| `errors.py` | `ComfyCliError`, the "nothing recorded to stop" detector (`_is_no_recorded_server` + its markers), and the `error.details` renderer + per-field char cap (`_render_error_details`, `_MAX_ERROR_FIELD_CHARS`) |
| `clitext.py` | comfy-cli **human-output** parsing — the text channel for verbs with no envelope: the `Saved:`-block / install-failure extractors and `plain_ok` synthesis (`_extract_saved_paths`, `_extract_install_failures`, `_synthesize_plain_result`), the missing-verb/-option capability probes (`_is_missing_verb_error`, `_is_missing_option_error`, `_normalize_cli_text`), `install_node`'s per-pack verdict (`_classify_install_result`), and the echoed-argv forgery guards (`_phrase_is_only_the_caller_s`, `_is_manager_missing_error`). `_extract_install_failures` / `_extract_saved_paths` are the AGENTS.md-documented cm-cli contract (see "The architecture rule" above) — move or edit them byte-for-byte |
| `argv.py` | argument-injection and OS-limit guards for every tool-facing string headed for `subprocess`: the shared primitives (`_reject_nul`, `_reject_option_like`, `_reject_nul_deep`, `_guard_arg_len`, `_encode_argv`, `_bounded_timeout`, `_clip_for_error`) and the per-domain guards built on them — `_guard_workflow_path`, `_guard_prompt_id`, `_guard_download_id`, `_guard_extra_args`, `_guard_version`, `_guard_node_names`, `_guard_log_port` / `_render_bad_port`, `_guard_model_relative_path` / `_guard_model_filename`, `_validate_upload_paths` — plus their length/shape constants and regexes |
| `target.py` | remote-target resolution/redaction/provenance for the run/job tools: `DEFAULT_COMFYUI_PORT`, `REMOTE_SHARED_MODELS_ENV` and the `COMFYUI_URL`/`COMFYUI_HOST`/`COMFYUI_PORT` parsing (`_comfy_target`, `_redact_config_url`), the `--host`/`--port` forward (`_with_target`, `_TARGET_AWARE_SUBCOMMANDS`), the local-only `download_model` refusal (`_reject_remote_model_download`), and the divergence notes `system_stats`/`free_memory` attach so an agent does not gate a remote run on local numbers (`_annotate_comfy_target`, `_target_provenance_suffix`, `_with_target_provenance`) |

`server` reaches them **module-qualified** (`tcc._tcc_guidance(...)`,
`failure_log._log_failure(...)`) and re-exports no BEHAVIOR — deliberately: a
test patching a moved name on `server` would otherwise silently patch a name
nothing reads. **Patch the owning module** — `monkeypatch.setattr(failure_log,
"_FAILURE_LOG_PATH", …)`, not `server`; the wrong one now raises
`AttributeError`. The one carve-out: public exception and model TYPES are
name-imported (`from .errors import ComfyCliError`) rather than reached
module-qualified — `ComfyCliError` rides hundreds of `except`/`isinstance`
sites, and a type has no mutable state a test could silently patch the wrong
copy of, so the failure mode the module-qualified rule guards against does not
apply to it.

## Toolchain

Python ≥ 3.10; pip + setuptools (there is no `uv.lock` here — comfy-cli bundles
`uv` and may write a stray one into the working directory; it is gitignored).

```bash
pip install -e '.[dev]'   # install with dev extras (pytest, ruff)
pytest -q                 # run the tests
ruff check .              # lint
ruff format --check .     # format check (run `ruff format .` to fix)
```

CI (`.github/workflows/ci.yml`) runs all three on Python 3.10 and 3.14 for every
PR; get them green locally first. Never add a `paths`/`paths-ignore` filter to its
`pull_request` trigger — the required `test (py3.10)`/`test (py3.14)` contexts must
report on every PR, and the workflow already no-ops on Markdown-only changes.

## Tests

Tests live in `tests/` and mock comfy-cli — no real ComfyUI, no `comfy` binary.
`_run_comfy` and the parser are exercised directly (`test_wrapper.py`,
`test_parser.py`); each tool group has its own file. Add a tool's test with it.

**Mock comfy-cli through the shared fixtures in `tests/conftest.py`, never a
hand-rolled stub.** They are the single place that mirrors how `server` spawns
the CLI, so a change to the spawn signature is one edit rather than a sweep:

- `envelope(ok=…, data=…, error=…)` — build an `envelope/1` body.
- `patched_run(stdout=…, returncode=…, stderr=…, raises=…, on_spawn=…) -> calls`
  — the plain `--json` path (`subprocess.run`); `calls` records `cmd`/`env`/
  `timeout`/`encoding` per call for exact-argv assertions, and `on_spawn(cmd)`
  fires at spawn so the one verb whose answer is a FILE writes its `--output`.
- `patched_plain_run(returncode, stdout, stderr) -> calls` — same, for the verbs
  that print human text and emit no envelope (`launch`/`stop`/`generate`).
- `patched_stream(stdout_text) -> procs` — the `--json-stream` NDJSON path
  (`asyncio.create_subprocess_exec`). Its fake pipes are real
  `asyncio.StreamReader`s, built by conftest's `stream_reader(text, limit)`
  helper; reuse that rather than hand-rolling an awaitable, so a fake still
  exercises the reader's buffer-limit behavior.
- `patched_async_run(stdout=…, returncode=…, stderr=…, hang=…, on_spawn=…) ->
  procs` — the plain-JSON *async* path (`_run_comfy_async`): same spawn and
  real `StreamReader` pipes as `patched_stream`, but the capture is parsed
  once at the end, not line-by-line. `hang=True` leaves the pipes OPEN so the
  fake child never finishes, for timeout/cancellation cases; `kill()` closes
  them (mirroring the process-group kill that lets a post-kill drain reach
  EOF), records `killed`, and fires `on_spawn(cmd)` like `patched_run`'s.

The two spawn paths differ deliberately: the plain `--json` path is synchronous
(`subprocess.Popen` + a bounded `communicate`, off-loaded to a thread pool by
async callers); anything STREAMING or long-lived spawns via
`asyncio.create_subprocess_exec` instead — nothing blocking may run on the
event loop, enforced by ruff's `ASYNC` select. Two async runners live there:
`_run_comfy_streaming` (NDJSON + progress) and `_run_comfy_async`, a plain-JSON
twin of `_run_comfy` for CANCELLATION — `asyncio.to_thread` is non-blocking,
but cancellation never reaches the thread, so a client giving up left the
child running. It carries the legacy foreground `model download`,
`workflow_deps`' 300s resolve, and `upload_file`'s 300s transfer. Each
stream keeps only a `_STDERR_MAX_CHARS` tail (`_drain_capped_into`; callers
widen stdout via `stdout_cap=`), never `communicate()`'s full capture.
`auth_login` (`_start_login`) is a third spawn site with its own browser flow.

A local stub is justified only where the call genuinely differs — the
`comfy --version` probe (its own kwargs) and multi-call sequenced replies.

## Destined-public hygiene

This repository is **private but destined to go public.** Treat everything you
write as if it were already public:

- **No secrets** — API keys, tokens, or credentials in code, commits, tests,
  fixtures, or PR text. Credential-in-URL fixtures use `https://<user>:<pass>@host`:
  a bare `user:pass@` fails the secret-scanning diff gate, and a fake scheme
  documents behavior the scrubber lacks (`failure_log._URL_RE` needs `https?://`).
- **No internal hostnames, IPs, or internal-only URLs** in code, comments, or
  commit messages.
- **No internal-tracker references** in commits or PR titles/bodies — describe
  the change on its own terms.
- Prefer environment variables and documented config over anything hardcoded.
