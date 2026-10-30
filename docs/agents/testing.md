# comfy-mcp: test fixtures and the two spawn paths

Reference detail for [`AGENTS.md`](../../AGENTS.md) "Tests". The hard rule — mock comfy-cli via the
shared fixtures in `tests/conftest.py`, never a hand-rolled stub — stays in `AGENTS.md`; this file is
the fixture catalogue and the rationale behind it.

## The shared fixtures

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
callers); anything STREAMING or long-lived spawns via `asyncio.create_subprocess_exec` — nothing
blocking runs on the event loop, enforced by ruff's `ASYNC` select. Two async runners live
there: `_run_comfy_streaming` (NDJSON + progress) and `_run_comfy_async`, a plain-JSON twin of
`_run_comfy` for CANCELLATION — cancellation never reaches a `to_thread` worker, so a client
giving up left the child running; it carries the longest-lived children (foreground `model
download`, `workflow_deps`, `upload_file`, `job("watch")`'s status polls). Each stream keeps
only a `_STDERR_MAX_CHARS` tail (`_drain_capped_into`; callers widen stdout via `stdout_cap=`),
never `communicate()`'s full capture. `auth_login` (`_start_login`) is a third spawn site.

A local stub is justified only where the call genuinely differs — the `comfy --version` probe
(its own kwargs), multi-call sequenced replies, and `test_wrapper.py`'s `_StderrBlockingProc`
(stdout EOFs fast while `stderr.read()` blocks, which neither streaming fixture models).
