# Upstream comfy-cli issues

Defects that surface through comfy-mcp but originate in **comfy-cli**. They are
recorded here so a user hitting one files it against the right project, and so
nobody re-diagnoses them from scratch.

Every entry was reproduced by driving comfy-cli **directly**, with no MCP layer
involved — that is what establishes ownership. Verified against **comfy-cli
1.15.0 / ComfyUI 0.31.0** on Linux + CUDA unless noted.

Status: none filed upstream yet.

---

## 1. `jobs watch` emits no progress events

**Impact:** `job(action="watch")`, and `run_workflow`/`run_template` with
`wait=True`, deliver no live progress. comfy-mcp's relay is wired correctly —
`_run_comfy_streaming(..., ctx=ctx)` drives a pump that calls
`ctx.report_progress` on `queued` / `executing` / `executed` / `progress`
events — but the stream carries none of them, so there is nothing to relay.

**Reproduce** — submit a job that takes tens of seconds, then watch it:

```bash
comfy --skip-prompt --workspace ~/comfy/ComfyUI run-template image_z_image_turbo \
  --param=57.text="a blue teapot" --param=57.steps=8 --async
# -> prompt_id

comfy --json-stream --where local jobs watch <prompt_id>
```

**Expected:** a stream of per-node/per-step events while the job runs.

**Actual:** exactly ONE envelope, emitted at the end:

```json
{"command":"jobs watch","ok":true,
 "data":{"status":"completed","completed_nodes":[],"elapsed_seconds":30.18, ...}}
```

`completed_nodes` is empty even though every node ran. Meanwhile ComfyUI's own
log shows a live per-step tqdm bar for the whole 30 seconds — the progress
exists, it just never reaches the JSON stream.

**Consequence:** any client waiting on a long job sees silence and cannot
distinguish "working" from "hung". comfy-mcp's docstrings have been corrected to
stop promising progress, which is a workaround for a documentation problem, not
a fix for this one.

---

## 2. `nodes path` ignores `from_type` and `max_depth`, and claims `"exact": true`

**Impact:** `nodes(action="path")` returns confidently wrong graph traversals.
comfy-mcp forwards both flags and relays the payload verbatim, including the
`exact` label.

**Reproduce:**

```bash
comfy --json nodes path MODEL IMAGE --max-depth 4 --max-paths 3
comfy --json nodes path AUDIO IMAGE --max-depth 4 --max-paths 3   # same rows
comfy --json nodes path MODEL IMAGE --max-depth 1 --max-paths 3   # same rows
```

**Expected:** `from_type` constrains the source type; `max_depth` bounds path
length; `AUDIO -> IMAGE` should be empty where no such path exists.

**Actual:** `from_type="AUDIO"` returns byte-for-byte the same paths as
`from_type="MODEL"`, and `max_depth=1` equals `max_depth=4`. Every step carries
`"from_type": ""`. The walker appears to enumerate nodes that *output*
`to_type`, alphabetically, ignoring the source constraint — `ByteDanceImageNode`
has no `MODEL` input but does have a field *named* `model` that is a COMBO of
API ids, which suggests matching on input NAME rather than type.

The result is then labelled `"exact": true`, asserting a correctness it does not
have. For the `AUDIO` case the correct answer is the empty set.

**Consequence:** an agent using this to plan a graph is actively misled. Wrong
answers labelled exact are worse than an error.

---

## 3. `workflow slots` mis-pairs slot names onto values

**Impact:** `list_workflow_slots` reports slots whose declared type contradicts
their value, and `set_workflow_slot` would write a user's value into a
DIFFERENT field while reporting success.

**Reproduce** — no MCP involved:

```bash
comfy --skip-prompt templates fetch api_minimax_h3_t2v -o /tmp/h3.json
comfy --json workflow slots /tmp/h3.json
```

**Actual:**

```json
{"address":"23.seed",      "type":"INT",     "current_value":"MiniMax H3"}
{"address":"23.watermark", "type":"BOOLEAN", "current_value":"768P"}
```

The real values are `42` and `false`, and the prompt has no slot at all.

**Root cause:** slot names (from `object_info`) are zipped POSITIONALLY against
`widgets_values`. `MinimaxHailuo03TextToVideoNode` exposes 3 inputs against 8
`widgets_values`, so every pairing after the shortfall is shifted.
`api_minimax_h3_r2v` fails identically, so this is the dynamic-combo partner
nodes as a class rather than one bad template.

**Consequence:** silent data corruption. `validate_workflow` still reports
`valid: true`, and reading the slots back shows the address the caller asked
for, so the obvious check passes. comfy-mcp now detects the type contradiction
and refuses to write to such a slot — but the true pairing is not recoverable
from a payload that has already lost it, so only comfy-cli can fix this.

---

## Not comfy-cli's, recorded to prevent misfiling

- **`workflow_deps` / `install_node` returning `unsupported`** — an install-state
  precondition (a legacy-clone ComfyUI-Manager whose `cm_cli` is unimportable),
  not a tool defect. The error text already explains it.
- **`generate_image` running a retired template** — was comfy-mcp's own
  hardcoded constant; fixed in 0.9.0.
