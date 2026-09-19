#!/usr/bin/env bash
# One-command live-engine smoke against a LIVE same-machine ComfyUI.
#
# It includes the checkpoint-free workflow/output and system-stats checks plus
# generate_image, whose default template needs the documented SD1.5 checkpoint.
#
# Requires a running local ComfyUI (COMFY_LOCAL_URL, default 127.0.0.1:8188) AND the
# comfy binary on PATH (or COMFY_BIN). Without both, the test SKIPS rather than
# fails. This complements, rather than replaces, the stdio/HTTP transport smoke
# documented in README.md.
set -euo pipefail

cd "$(dirname "$0")/.."
exec python -m pytest tests/e2e -m e2e "$@"
