#!/usr/bin/env bash
set -euo pipefail
export LONG_DECODE_BENCH=1
exec "$(dirname "$0")/run-vlx-full-stack-ab.sh"
