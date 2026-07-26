#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"

run_adapt() {
  echo "Adapt: generating the nested 1K / 3K / 6K corpus from one 6K run."
  if [[ "${GPU_IDS:-}" == *","* || "${GPU_IDS:-}" == *" "* || "${FPRIR_PROCESSES_PER_GPU:-1}" -gt 1 ]]; then
    "${SCRIPT_DIR}/run_adapt_multi_gpu.sh" 6k
  else
    "${SCRIPT_DIR}/run_adapt_tier.sh" 6k
  fi
}

run_dist() {
  if [[ "${GPU_IDS:-}" == *","* || "${GPU_IDS:-}" == *" "* ]]; then
    "${SCRIPT_DIR}/run_dist_multi_gpu.sh"
    return
  fi
  local tier
  for tier in quick standard extended; do
    echo "Dist: running ${tier}."
    "${SCRIPT_DIR}/run_dist_tier.sh" "${tier}"
  done
}

case "${MODE}" in
  all)
    run_adapt
    run_dist
    ;;
  adapt)
    run_adapt
    ;;
  dist)
    run_dist
    ;;
  *)
    echo "Usage: $0 {all|adapt|dist}" >&2
    exit 2
    ;;
esac
