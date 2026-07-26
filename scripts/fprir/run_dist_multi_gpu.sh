#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

IFS=', ' read -r -a GPUS <<< "${GPU_IDS:-${GPU_ID}}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one physical CUDA device." >&2
  exit 2
fi

TIERS=(quick standard extended)
PIDS=()
for index in "${!TIERS[@]}"; do
  tier="${TIERS[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  tier_output="${FPRIR_OUTPUT_ROOT}/dist-${tier}"
  launcher_log="${LOG_DIR}/dist-${tier}-launcher-$(date +%Y%m%d-%H%M%S).log"
  mkdir -p "${tier_output}"
  printf "running\n" >"${tier_output}/.status"
  echo "Dist ${tier}: physical GPU ${gpu}, log ${launcher_log}"
  (
    set +e
    GPU_ID="${gpu}" \
    FPRIR_OUTPUT_ROOT="${FPRIR_OUTPUT_ROOT}" \
    LOG_DIR="${LOG_DIR}" \
    PYTHON="${PYTHON}" \
    "${SCRIPT_DIR}/run_dist_tier.sh" "${tier}"
    tier_status=$?
    printf "%d\n" "${tier_status}" >"${tier_output}/.status"
    exit "${tier_status}"
  ) >"${launcher_log}" 2>&1 &
  PIDS+=("$!")
done

set +e
"${PYTHON}" "${SCRIPT_DIR}/monitor_dist_tiers.py" \
  --output-root "${FPRIR_OUTPUT_ROOT}" \
  --tiers "${TIERS[@]}" \
  --stage "${DIST_STAGE:-all}"
monitor_status=$?
status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=1
done
set -e
if [[ "${monitor_status}" -ne 0 || "${status}" -ne 0 ]]; then
  echo "At least one Dist tier failed; inspect ${LOG_DIR}." >&2
  exit 1
fi
