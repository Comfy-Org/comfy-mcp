# Changelog

Notable changes per release. Dates are UTC.

The version is a **minor** bump whenever a tool's response shape or default
behaviour changes, even though that would be a major bump after 1.0 — while the
project is pre-1.0, minor is the strongest signal available, so read it as
"check the notes before upgrading".

## 0.11.0

The server now introduces itself, and a few answers that used to look finished
were not. `comfy-mcp --help` and `--version` print instead of exiting with no
output, and the MCP `initialize` handshake reports the installed package
version. Node search can tell a free node from a paid partner node,
`billing_status` is wired to `comfy cloud status` — no tagged comfy-cli
release has that verb yet — and image, video, and audio outputs can come
back as an inline preview. Minor rather than patch:
several tool responses gain fields, and `launch_comfyui` / `restart_comfyui`
stop acting on the local server when a remote ComfyUI is configured.

The comfy-cli floor stays **>= 1.14.0**. Two verbs sit beside that floor
rather than under it. `job(action="watch")` needs comfy-cli >= 1.16.0, and a
server-wide raise would refuse every other tool on an install where only
watch is broken. `billing_status` is the other caveat: `comfy cloud status`
is newer than every tagged comfy-cli release, so a current install returns
`unsupported: true` instead of a balance. The message names no version to
upgrade to, because there is not one yet.

### Added

- **`comfy-mcp --help` and `--version`.** A bare `comfy-mcp` still serves MCP
  over stdio. The two flags print a short usage screen and the installed
  version, then exit 0, so a first install is distinguishable from a process
  that started, read nothing, and left. (#228)
- **`billing_status`.** One read-only passthrough of `comfy cloud status`:
  remaining credits, plan, and how many jobs can run at once, once the
  engine has the verb. It does not spend and it does not prompt. No tagged
  comfy-cli release includes that verb yet, so on a current install the
  call returns `unsupported: true` with the balance fields null. That is
  the common result today, not a rare degrade, and the message names no
  version to upgrade to. (#250)
- **`preview_media`.** Wraps `comfy preview` and returns comfy-cli's metadata
  plus an optional bounded inline PNG — an image thumbnail, a video contact
  sheet, or an audio waveform — so an agent can look at video and audio that
  `fetch_outputs` can download but cannot show through the MCP image path.
  (#256, thanks @wgu9)
- **`is_api_node` on node rows, and `exclude_api` on `nodes(action="search")`.**
  comfy-cli tags each `nodes search` / `nodes ls` row with `is_api_node`. The
  field is documented on `nodes`, and `exclude_api=True` drops the paid rows
  when the caller asked for free, local generation. The default is `False`,
  so an existing caller sees the same rows plus the flag. A comfy-cli too old
  to send the flag is left unfiltered: a node that cannot be classified is
  shown. (#235)
- **`object_info_path` on `nodes` and `validate_workflow`.** Optional saved
  `object_info` catalog, forwarded as comfy-cli's `--input`, for installs
  whose live catalog exceeds the engine's response ceiling. Live discovery
  stays the default. (#246, thanks @wgu9)
- **PyPI project URLs.** `[project.urls]` now carries Homepage, Repository,
  Documentation, and Issues. Through 0.10.0 the published metadata had
  `project_urls: null` and a blank Home-page, so the PyPI sidebar linked
  nowhere. (#227)
- **MCP Registry listing, published with the GitHub Release.** `server.json`
  names `io.github.Comfy-Org/comfy-mcp` and points at the PyPI package over
  stdio. `publish.yml` publishes that listing after the PyPI upload, once the
  registry can see this exact version. Cutting the 0.11.0 GitHub Release is
  what sends it; this changelog entry does not. (#232)

### Changed

Check these before upgrading. The floor is unchanged; the verbs below are not.

- **`job(action="watch")` refuses a comfy-cli it can identify as older than
  1.16.0.** On those builds the watch attached as a new session, heard no
  execution events, and after its timeout returned a payload that looked like
  a job that never started. It now fails immediately, says to upgrade with
  `pip install -U comfy-cli`, and points at `job(action="wait")`, which is
  unaffected. Timeouts that still happen report
  how many events arrived, and attach a `jobs status` reading when that count
  is zero. Other tools keep working on 1.14.x and 1.15.x. (#241)
- **`launch_comfyui` and `restart_comfyui` refuse a configured remote.** With
  `COMFYUI_URL` or `COMFYUI_HOST` set, both used to act on the local server
  and report success. `launch_comfyui` now starts nothing and returns a
  structured refusal (`launched: false`). `restart_comfyui` names the remote
  it did not touch and says to restart that one on its own machine. (#274,
  #275)
- **`stop_comfyui` says a configured remote was not checked.** `comfy stop`
  only knows about background launches on this machine, so a remote used to
  come back as "No ComfyUI is running in the background". The same answer now
  says the configured remote was not inspected and may still be up. (#276)
- **A declined consent prompt no longer says "the user declined".** Some
  clients answer `decline` themselves when they cannot show a prompt. The
  error now says the prompt was declined and that a client may have done so
  on its own. The gates still fail closed, and what they allow is unchanged.
  (#236)
- **Download results spell the handle `download_id`.** Arguments already did.
  comfy-cli's own `id` stays beside the new key, so a caller reading `id`
  keeps working. (#234)
- **The template gallery cache is warmed at startup.** A missing or corrupt
  `~/.cache/comfy-cli/gallery/index.json` used to be fetched inside the first
  `search_templates` call. That fetch now runs once on a background thread at
  process start. (#243)

### Fixed

- **The MCP handshake reports this package's version.** `initialize` was
  answering `serverInfo.version: ""` because the SDK does not invent one. It
  now reports the installed distribution version, falling back to the
  `__version__` literal when there is no distribution metadata. (#230)
- **comfy-cli JSON that contains U+2028 parses.** `_last_json_object` split
  stdout with `str.splitlines()`, which also breaks on Unicode line
  separators. A one-line envelope whose string value held U+2028 (the MiniMax
  H3 image-to-video template prompt) was cut in two, and the tool raised
  "comfy-cli returned no JSON". Captured stdout now splits on `\n`, `\r\n`,
  and a bare `\r` only. (#307, fixes #296)
- **Inline output previews resolve inside the project directory.** With
  `COMFY_PROJECT` set and a relative `out_dir`, `fetch_outputs(inline_images=True)`
  wrote the files into the project and then looked for them relative to the
  MCP process directory, so it could return a stale image or omit one it had
  just downloaded. The preview directory is resolved against the download
  child's project directory. (#294, thanks @allenflux)
- **`generate_image(wait=True)` returns the `prompt_id` when the wait
  expires.** The job kept running with no handle in the error. The bounded
  wait now comes back with the id either way: finished outputs, or still
  running. (#238)
- **`download(action="cancel")` says whether the transfer actually stopped.**
  The reply used to be the same `status: "completed"` row a finished download
  returns. It now carries `cancelled`, comfy-cli's `changed`, and a one-line
  `note` for the case where the call was too late. (#237)
- **`nodes(action="path")` falls back to `--loose` when the engine's default
  walk is not type-constrained.** On those builds an exact path search
  returned nodes that do not accept the requested socket, with nothing in the
  payload saying so. The fallback is the query that engine answers correctly.
  (#239)
- **Launch and restart report the pid that holds the port.** comfy-cli records
  the detached `comfy launch` wrapper, and killing that pid left `python
  main.py` running. The reported pid is the process bound to the port; the
  wrapper pid stays beside it, labelled. (#233)
- **Racing first calls share one `comfy --version` probe.** The version guard
  memoized only after the probe finished, so several tools arriving during
  that first cold start each spawned their own. They now wait on the in-flight
  probe. (#240)

### Docs

- **Install comfy-cli next to comfy-mcp.** The wheel still does not depend on
  comfy-cli — the binary that matters is the one on `PATH` — and the README
  quickstart now installs both in one command,
  `pip install comfy-mcp "comfy-cli>=1.14.0"`, and says why a bare
  `pip install comfy-mcp` then fails on every tool call. (#229)
- **Hand-written workflow nodes get a plain-language `_meta.title`.** Agent
  instructions say to title every node, pre-flight the graph with
  `validate_workflow`, and prefer a template when one fits. (#287)
- **`AGENTS.md` reference tables moved under `docs/agents/`.** The module
  layout and the test-fixture catalogue moved verbatim so the root file stays
  inside its line ceiling. The architecture rules stay in `AGENTS.md`. (#298)
- **Vulnerability reports go to support@comfy.org.** GitHub private
  vulnerability reporting is not enabled here, so `SECURITY.md` no longer
  offers that form. The issue chooser's security contact link opens the same
  policy. (#308, fixes #295)
- **ComfyUI Desktop and the Windows portable package.** The README only
  described a comfy-cli workspace on `127.0.0.1:8188`. It now says how to
  point this server at Desktop — the app defaults to port 8000, so set
  `COMFY_LOCAL_URL` — and at `ComfyUI_windows_portable`. A comfy-cli
  background server still outranks 8188. `restart_comfyui` asks before
  killing an untracked listener on a taken port, which can be Desktop's
  server or the portable one. `comfy set-default` on the portable tree makes
  that tree the workspace for launch and node installs too, and those use
  the Python comfy-cli resolves rather than `python_embeded`. A plain file
  URL still downloads into the workspace `models` directory; a typed CivitAI
  or Hugging Face download on comfy-cli 1.25 or newer can follow the running
  server's model folder. (#309, fixes #258)

### Internal

CI and dependency bumps, folded together: shared-workflow SHA updates (#231,
#244, #249, #253, #260, #262, #263, #264, #266, #268, #269, #270, #271, #277,
#280, #282, #283, #289, #290), adoption of the public-repo-hygiene check and
its ECMA-48 allowlist (#254, #261), re-pinning `agents-md-integrity` so `uses:`
and `workflows_ref` name one commit (#279), GitHub Actions and TruffleHog
updates (#265, #288, #306), and the ruff dev extra widened to `>=0.16,<0.18`
(#292). Tests: the packaged version files are pinned to agree at PR time
(#226), and the blocking-child stub lives in one shared fixture (#242).

## 0.10.0

`search_templates` rows can now tell a paid template from a free one.

### Added

- **An `api` boolean on every `search_templates` row.** The gallery's `API` tag
  — the one meaning "this template runs the model on a hosted partner API and
  spends your credits" — was dropped on the way out. Two templates can carry
  the same title and differ only by that tag (`api_minimax_h3_t2v` vs
  `video_minimax_h3_t2v`), so an agent reading the results could not tell them
  apart, and would recommend the paid one while saying no free version existed.
  Each row now carries a plain `api: true`/`false`, derived with the exact tag
  test `exclude_api` already used and factored into one shared helper so the
  flag and the filter cannot drift apart. The raw `tags` list stays out of the
  rows, so the listing is as compact as before. This is additive — a caller
  ignoring the field sees no change — but it is a response-shape change, hence
  the minor bump.

### Documentation

- The tokenized matching semantics that `nodes` and `search_models` inherit
  from comfy-cli are now documented, so a caller knows how a multi-word query
  is matched instead of inferring it from results.
- The comfy-cli floor's rationale records that its cheap-to-raise precedent
  expired once this server began publishing to PyPI: the guard refuses up front
  in `_run_comfy` rather than degrading per verb, so a raise is a breaking
  change for anyone below the new floor. The floor itself stays at 1.14.0 —
  comfy-cli 1.15.0 and 1.16.0 both shipped fixes to verbs called here without
  warranting a move.

### Known issues from 0.9.0 — two now fixed upstream

comfy-cli 1.16.0 resolves two of the three upstream defects 0.9.0 shipped with.
Upgrade the engine (`pip install --upgrade comfy-cli`) to pick them up: this
server needs no change, and its floor is deliberately unmoved, so an install
below 1.16.0 keeps working with the old behaviour rather than being refused.

- **Progress streaming works.** `comfy jobs watch` delivered a single envelope
  at the end claiming no nodes had run, because it connected under a fresh
  client id and a local ComfyUI addresses execution events only to the session
  that submitted the prompt. It now attaches as the submitter and streams live
  progress.
- **`nodes(action="path")` respects `from_type` and `max_depth`**, and claims
  `"exact": true` only when the search actually finished. It previously
  returned identical rows for unrelated source types while reporting the answer
  as exact.
- **Still open:** the slot mis-pairing originating in `comfy workflow slots`.
  This server continues to detect and refuse it, but cannot repair the pairing.

## 0.9.0

Seven P1 defects found by two independent QA passes (Claude Code and Codex CLI).
Both clients hitting the same failures is what established them as server-side
rather than client quirks.

### Fixed

- **`generate_image` was dead.** It shelled out to `comfy run-template default`,
  but `default` had left the gallery, so every call failed with
  `template_not_found`. Reproduced on a fresh install with the cache refreshed,
  on macOS/MPS and Linux/CUDA alike. The on-ramp now runs
  `image_z_image_turbo`, and a template that goes missing in future yields an
  error naming templates that exist rather than a dead-end hint.
- **`search_templates` answered the wrong rows.** `query` was a raw substring
  test, wrong in both directions: `MiniMax Text to Video` returned nothing while
  `MiniMax H3: Text to Video` existed, and the mid-word fragment `ext to imag`
  matched 91 rows. It now runs a word-anchored phrase pass, falling back to
  all-words only when the phrase matches nothing.
- **Consent gates blamed the user for refusals nobody made.** Every non-accept
  answer collapsed into "the user declined", including clients that resolve the
  request without ever showing a prompt. Only an actual decline says so now.
  Gates still fail closed.
- **`discover` could not be called at its own default.** It returned every
  schema body — ~63 KB, ~109 KB once a client pretty-prints it — over a typical
  25,000-token cap, so the call was rejected outright.
- **Mis-paired workflow slots were relayed as fact.** On dynamic-combo partner
  templates, `list_workflow_slots` reported an `INT` slot holding a model title.
  `set_workflow_slot` would have written into the wrong field while reporting
  success, with `validate_workflow` still reporting valid.
- **`update_comfyui` leaked a traceback on a detached HEAD** — a normal state for
  a version-pinned install — instead of git's own "you are not currently on a
  branch".
- **Emitted partner workflows carried no cost provenance.** The file is the
  thing handed to `run_workflow`, where billing happens, often by someone who
  never saw the tool response that said it would bill.

### Added

- **`COMFY_MCP_ASSUME_CONSENT`** — pre-authorize specific confirmation gates from
  the server's environment, for clients that cannot display a prompt. Set by
  whoever configures the server, in a file the model cannot edit; an agent
  cannot grant itself permission. Spending is excluded by construction: no
  value, including `all`, reaches the credit gates. See the README.

### Changed — check before upgrading

- `discover()` now returns a `schema_index` of names instead of every schema
  body. Pass `command="<name>"` for one body, or `schemas_only=False` for the
  full surface.
- `generate_image` runs a different template, and `checkpoint=` is refused on
  graphs that load weights through split UNET/CLIP/VAE loaders — which is now
  the gallery norm — rather than failing mid-run.
- `search_templates` may return `match: "all-words"`, flagging a result widened
  beyond the phrase that was typed.

### Known issues — upstream in comfy-cli, not this server

- **Progress streaming delivers nothing.** `job(action="watch")` relays what the
  engine sends, and `comfy jobs watch` emits a single envelope at the end even
  for a job whose ComfyUI log shows a live progress bar. Poll
  `job(action="status")` meanwhile. Docstrings no longer promise otherwise.
- **`nodes(action="path")` ignores `from_type` and `max_depth`** while labelling
  its result `"exact": true`.
- The slot mis-pairing above originates in `comfy workflow slots`; this release
  detects and refuses it but cannot repair the pairing.
