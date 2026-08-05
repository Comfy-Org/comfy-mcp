# Agent Guidelines — comfy-mcp

A short guide for AI agents (and humans) here. **Keep it in sync:** a PR that
changes the architecture rule, the toolchain or the tool set updates it too.

## What this repo is

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
rather than in new logic — `fetch_template` runs `templates fetch` and then
`validate` to tell the caller whether the template it just wrote can actually
run on this install, and `download_model` submits `model download --background`
and then polls `model download-status` so a multi-GB transfer does not hold the
MCP request open. That stays inside the rule: every call still goes through
`_run_comfy`, the verdict is comfy-cli's own, and this repo adds no product
behavior of its own. What would breach it is deriving the answer here (parsing
the graph, keeping a table of what is "supported") instead of asking the engine
— for `download_model` that would mean sizing the file on disk to decide whether
it finished, rather than reading comfy-cli's `status`. `install_node` composes
the same way for a different reason: it reads `comfy env`'s
`workspace.manager_detected` / `manager_mode` BEFORE its consent prompt, because
`comfy node install` runs Manager's `cm-cli` and an install that has Manager only
as a legacy clone under `custom_nodes/` cannot run it — so the user would be
asked to authorize third-party code on a call guaranteed to fail. The verdict is
comfy-cli's own field, not a scan this repo performs, and the check fails OPEN:
an unreadable answer installs anyway and lets the engine's error stand.
`workflow_deps` describes that same environment through the same two helpers
(`_manager_state_clause` / `_MANAGER_VENV_REMEDY`) so the two cannot drift. `workflow_deps` reads its
answer off DISK because the engine leaves no choice — `comfy node
deps-in-workflow` emits no envelope and REQUIRES an `--output` path — so the
temp-file round trip is that contract, and Manager's manifest goes back as
written bar `failure_log._scrub_text` masking credentials in its repo-URL keys.

The one thing that legitimately lives here rather than in comfy-cli is **MCP
protocol surface** — capabilities that only exist between this server and its
client, and that comfy-cli has no way to express. Today that is the per-call
confirmation on the tools that can spend money, destroy local state, run
third-party code, or expose the machine: `partner_generate`, `run_template`,
`run_workflow`, `switch_comfyui_version`, `install_node`, `update_comfyui` when
`target="all"`, and the `launch_comfyui` / `restart_comfyui` pair when
`extra_args` would publish ComfyUI to the network. comfy-cli owns the
credit-spend interlock and the durable "always proceed" (`comfy generate consent
always`), and this server only raises the confirmation over MCP **elicitation**
— the protocol's equivalent of the CLI's y/N prompt — then forwards the answer
as `--yes` / `--allow-spend`, or (for the four the CLI does not gate at all)
simply refuses to run the command. It stores no consent of its own. All of them
share one fail-closed body, `_elicit_approval`; give a new gate its own
`_ApprovalWording` rather than a second copy of that handling. Adding *product*
behavior here is still a guardrail breach; adapting comfy-cli's contract to an
MCP primitive is this repo's job.

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
--help` for the flag — its docstring says why, and what an engine without it
means. `switch_comfyui_version`, `install_node` and `update_comfyui` sit outside
that axis: none spends credits, the CLI gates none of them, so their prompt is
this server's only gate, with no always-proceed to read. TWO gates are
ARGUMENT-scoped, the danger being the argument and not the verb: the launch pair
prompts only for `extra_args` that would publish an unauthenticated ComfyUI
(`_network_exposing_args`), `update_comfyui` only for `target="all"`, which
pip-installs every third-party pack (`comfy`/`cli` never prompt). `install_node`
is NOT one: it prompts on EVERY call, and its argument is a RESTRICTION — `names`
pinned to registry slugs, refusing a URL, as the prompt promises a REGISTRY pack.
Mirror the engine's contract per tool; never generalize one gate onto another.

The local differentiator: discovery (`search_nodes`, `get_node`, `search_models`)
reads the **user's live install** — custom nodes included — not a static catalog.

## Module layout

`server.py` holds the wrapper core (`_run_comfy`, the envelope parser, the
`--json-stream` machinery, the spend-consent plumbing) and every `@mcp.tool()`.
Three **leaf** modules sit under it — nothing in them imports `server`, so the
dependency edges only ever point one way:

| Module | Owns |
|---|---|
| `textutil.py` | pure text helpers: `_tail` / `_stream_tail` (bounded stream tails) and `_redact_url` (userinfo masking) |
| `tcc.py` | macOS protected-folder (TCC) detection + the guidance message |
| `failure_log.py` | the opt-in `COMFY_MCP_DEBUG_LOG` failure log (its config, its module state, and `_log_failure`) **and the URL scrubbers** — `_scrub_text` / `_scrubbed_stream_tail` also mask credentials on the way to the MCP CLIENT, not just to disk |

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
PR; get them green locally first. Never add a `paths`/`paths-ignore` filter to its
`pull_request` trigger — the required `test (py3.10)`/`test (py3.14)` contexts must
report on every PR, and the workflow already no-ops on Markdown-only changes.

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
- `patched_async_run(stdout=…, returncode=…, stderr=…, hang=…) -> procs` — the
  plain-JSON *async* path (`_run_comfy_async`): same
  `asyncio.create_subprocess_exec` spawn and the same real `StreamReader` pipes,
  but the capture is parsed once at the end instead of read line-by-line.
  `hang=True` leaves those pipes OPEN so the fake child never finishes, for the
  timeout/cancellation cases; its `kill()` closes them, mirroring the process-group
  kill that is what lets a post-kill drain reach EOF. Each `_FakeAsyncRunProc`
  records `killed`, so a test can assert the process-tree kill actually fired.

The two spawn paths differ deliberately: the plain `--json` path is synchronous
(`subprocess.Popen` + a bounded `communicate`, off-loaded to a thread pool by its
async callers), while every path that STREAMS or is otherwise long-lived spawns
with `asyncio.create_subprocess_exec` — nothing blocking may run on the event
loop, and ruff's `select` enables `ASYNC` to enforce that. Two async runners sit
on that side, not one: `_run_comfy_streaming` (NDJSON + progress notifications)
and `_run_comfy_async`, a plain-JSON twin of `_run_comfy` with the same result
contract. The twin exists for CANCELLATION, not for the event loop —
`asyncio.to_thread(_run_comfy, …)` is already non-blocking but its cancellation
never reaches the thread, so a long-lived call left the `comfy` child running
when a client gave up. Today it carries the legacy foreground `model download`
(the `--background`-less fallback); short metadata calls stay on the thread-pool
path. Being reserved for the longest-lived children, it bounds each captured
stream to its trailing `_STDERR_MAX_CHARS` (via `_drain_capped_into`) rather than
retaining everything the way `communicate()` does — the one place its contract is
narrower than `_run_comfy`'s. `auth_login` is a third async spawn site
(`_start_login`) for the same reason, but drives its own browser flow.

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
