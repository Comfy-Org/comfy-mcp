#!/usr/bin/env bash
# One-command e2e smoke: a real no-model round-trip against a LIVE local ComfyUI.
#
#   server_info -> run_workflow (EmptyImage -> SaveImage) -> fetch_outputs -> PNG.
#
# Requires a running local ComfyUI (COMFYUI_URL, default 127.0.0.1:8188) AND the
# comfy binary on PATH (or COMFY_BIN). Without both, the test SKIPS rather than
# fails — so this is safe to run anywhere; it just reports "skipped" if you're
# not set up. Answers "does it really work on this machine?" in one shot.
set -euo pipefail

cd "$(dirname "$0")/.."
exec python -m pytest tests/e2e -m e2e "$@"
