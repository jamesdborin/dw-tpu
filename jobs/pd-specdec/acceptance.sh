#!/usr/bin/env bash
# Job-local sampling-equivalence gate for MTP speculative decoding.
# Runs after the runner's greedy (vs non-spec H4-r2), GPU-reference and
# suffix-causality probes. Writes $OUT/sampling-equivalent.json.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "${PYTHON:-/opt/dsv4/.venv/bin/python}" "$HERE/spec_acceptance.py"
