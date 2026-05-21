#!/usr/bin/env bash
set -e
set -u
set -o pipefail

# Backward-compatible entrypoint.
exec ./run_asr.sh "$@"
