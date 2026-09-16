#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RELEASE_ROOT}"
export PYTHONPATH="${RELEASE_ROOT}"
exec python UI_Tool/launch_scripts/start_server_local.py \
  --config UI_Tool/launch_scripts/config/all_service_example_local_1.yaml "$@"
