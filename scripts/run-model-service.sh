#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: scripts/run-model-service.sh <config> [socket] [checkpoint] [codec-socket]}"
SOCKET="${2:-}"
CHECKPOINT="${3:-}"
CODEC_SOCKET="${4:-}"
CONFIG_PATH="$CONFIG"
[[ "$CONFIG_PATH" = /* ]] || CONFIG_PATH="$REPO/$CONFIG_PATH"
ARGS=(serve --config "$CONFIG_PATH")
[[ -n "$SOCKET" ]] && ARGS+=(--socket "$SOCKET")
[[ -n "$CHECKPOINT" ]] && ARGS+=(--checkpoint "$CHECKPOINT")
[[ -n "$CODEC_SOCKET" ]] && ARGS+=(--codec-socket "$CODEC_SOCKET")
exec uv run model-service "${ARGS[@]}"
