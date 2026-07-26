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

TIERS=(5 10)
REQUESTED_STAGE="${DIST_STAGE:-all}"
if [[ "${REQUESTED_STAGE}" == "all" ]]; then
  STAGES=(localization beamforming)
elif [[ "${REQUESTED_STAGE}" == "localization" || "${REQUESTED_STAGE}" == "beamforming" ]]; then
  STAGES=("${REQUESTED_STAGE}")
else
  echo "DIST_STAGE must be all, localization, or beamforming" >&2
  exit 2
fi
TASK_TIERS=()
TASK_STAGES=()
for tier in "${TIERS[@]}"; do
  for stage in "${STAGES[@]}"; do
    TASK_TIERS+=("${tier}")
    TASK_STAGES+=("${stage}")
  done
done

PIDS=()
for index in "${!TASK_TIERS[@]}"; do
  tier="${TASK_TIERS[$index]}"
  stage="${TASK_STAGES[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  tier_output="${FPRIR_OUTPUT_ROOT}/dist-${tier}"
  status_path="${tier_output}/.status-${stage}"
  launcher_log="${LOG_DIR}/dist-${tier}-${stage}-launcher-$(date +%Y%m%d-%H%M%S).log"
  mkdir -p "${tier_output}"
  printf "running\n" >"${status_path}"
  echo "Dist-${tier} ${stage}: physical GPU ${gpu}, log ${launcher_log}"
  (
    set +e
    GPU_ID="${gpu}" \
    DIST_STAGE="${stage}" \
    FPRIR_OUTPUT_ROOT="${FPRIR_OUTPUT_ROOT}" \
    LOG_DIR="${LOG_DIR}" \
    PYTHON="${PYTHON}" \
    "${SCRIPT_DIR}/run_dist_tier.sh" "${tier}"
    task_status=$?
    printf "%d\n" "${task_status}" >"${status_path}"
    exit "${task_status}"
  ) >"${launcher_log}" 2>&1 &
  PIDS+=("$!")
done

set +e
"${PYTHON}" "${SCRIPT_DIR}/monitor_dist_tiers.py" \
  --output-root "${FPRIR_OUTPUT_ROOT}" \
  --tiers "${TIERS[@]}" \
  --stage "${REQUESTED_STAGE}"
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
