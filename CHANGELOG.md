# Changelog

Notable changes per release. Dates are UTC.

The version is a **minor** bump whenever a tool's response shape or default
behaviour changes, even though that would be a major bump after 1.0 — while the
project is pre-1.0, minor is the strongest signal available, so read it as
"check the notes before upgrading".

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
