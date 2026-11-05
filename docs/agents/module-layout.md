# comfy-mcp: per-module ownership

Reference detail for [`AGENTS.md`](../../AGENTS.md) "Module layout". The architecture rule and the
module-qualified patching rule stay in `AGENTS.md`; this file only expands who owns what.

| Module | Owns |
|---|---|
| `textutil.py` | pure text helpers: `_tail` / `_stream_tail` (bounded stream tails) and `_redact_url` (userinfo masking) |
| `tcc.py` | macOS protected-folder (TCC) detection + the guidance message |
| `failure_log.py` | the opt-in `COMFY_MCP_DEBUG_LOG` failure log (its config, its module state, and `_log_failure`) **and the URL scrubbers** — `_scrub_text` / `_scrubbed_stream_tail` also mask credentials on the way to the MCP CLIENT, not just to disk |
| `instructions.py` | the `INSTRUCTIONS` constant handed to `MCPServer(..., instructions=...)` — client-handshake text |
| `errors.py` | `ComfyCliError`; the "nothing recorded to stop" detector; the `error.details` renderer + per-field char cap |
| `clitext.py` | comfy-cli **human-output** parsing for verbs with no envelope — `Saved:`-block/install-failure extraction, `plain_ok` synthesis, missing-verb/-option probes, `install_node`'s per-pack verdict, echoed-argv forgery guards. Its extractors are the documented cm-cli contract (see architecture rule above) — move or edit byte-for-byte |
| `argv.py` | argument-injection and OS-limit guards for every tool-facing string headed for `subprocess`: shared primitives plus per-domain guards (workflow path, prompt id, download id, extra args, version, node names, log port, model path/filename, upload paths) |
| `target.py` | remote-target resolution/redaction/provenance for run/job tools — `COMFYUI_URL`/`HOST`/`PORT` parsing, `--host`/`--port` forwarding, the local-only `download_model` refusal, and divergence notes on `system_stats`/`free_memory` (and on `stop_comfyui`'s "nothing recorded") so an agent doesn't gate a remote run on local numbers |
| `params.py` | param/slot marshaling into comfy-cli argv for `generate`/`run-template`/`set-slot`/`vary`, incl. the structured slot machinery — `SlotOverride`/`SlotVariants` are this module's public TYPES (carve-out below) |
| `cli.py` | the console script's own argv surface — the `--help` / `--version` text a HUMAN who types `comfy-mcp` in a terminal gets, plus the installed-metadata version lookup (`_version`) behind it. That lookup is the SINGLE answer to "which release is this?": `server._server_version` delegates to it for the handshake's `serverInfo.version`, so the string a client displays is the string the terminal prints |
