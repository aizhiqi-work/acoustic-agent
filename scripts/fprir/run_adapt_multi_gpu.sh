#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TIER="${1:-full}"
case "${TIER}" in
  1k)
    FLOORPLANS=1000
    NESTED_SIZES=(1000)
    ;;
  8k)
    FLOORPLANS=8000
    NESTED_SIZES=(1000 8000)
    ;;
  full)
    FLOORPLANS=15376
    NESTED_SIZES=(1000 8000 15376)
    ;;
  *)
    echo "Usage: $0 {1k|8k|full}" >&2
    exit 2
    ;;
esac

FLOORPLANS="${FPRIR_MAX_FLOORPLANS:-${FLOORPLANS}}"
QUALITY="${FPRIR_QUALITY:-simulation}"
DURATION="${FPRIR_DURATION_S:-2.0}"
MOTION_FRACTION="${FPRIR_MOTION_FRACTION:-0.3}"
SAME_ROOM_VARIANTS="${FPRIR_SAME_ROOM_VARIANTS:-2}"
CROSS_ROOM_VARIANTS="${FPRIR_CROSS_ROOM_VARIANTS:-3}"

IFS=', ' read -r -a GPUS <<< "${GPU_IDS:-${GPU_ID}}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one physical CUDA device." >&2
  exit 2
fi

PROCESSES_PER_GPU="${FPRIR_PROCESSES_PER_GPU:-1}"
if ! [[ "${PROCESSES_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FPRIR_PROCESSES_PER_GPU must be a positive integer." >&2
  exit 2
fi
WORKER_GPUS=()
for gpu in "${GPUS[@]}"; do
  for ((slot = 0; slot < PROCESSES_PER_GPU; slot++)); do
    WORKER_GPUS+=("${gpu}")
  done
done

OUTPUT="${FPRIR_OUTPUT_ROOT}/adapt-${TIER}"
PARTS_ROOT="${OUTPUT}/parts"
mkdir -p "${PARTS_ROOT}" "${LOG_DIR}"
PART_COUNT="${#WORKER_GPUS[@]}"
CPU_THREADS="${FPRIR_CPU_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu)}"
DEFAULT_NUMBA_THREADS=$((CPU_THREADS / PART_COUNT))
if [[ "${DEFAULT_NUMBA_THREADS}" -lt 1 ]]; then
  DEFAULT_NUMBA_THREADS=1
elif [[ "${DEFAULT_NUMBA_THREADS}" -gt 16 ]]; then
  DEFAULT_NUMBA_THREADS=16
fi
NUMBA_THREADS="${FPRIR_NUMBA_THREADS:-${DEFAULT_NUMBA_THREADS}}"
PIDS=()
PART_DIRS=()
EXTRA_ARGS=()
if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--plan-only)
fi

for rank in "${!WORKER_GPUS[@]}"; do
  gpu="${WORKER_GPUS[$rank]}"
  part="$(printf "%s/part-%03d" "${PARTS_ROOT}" "${rank}")"
  log_path="${LOG_DIR}/adapt-${TIER}-part-${rank}-$(date +%Y%m%d-%H%M%S).log"
  mkdir -p "${part}"
  printf "running\n" >"${part}/.status"
  PART_DIRS+=("${part}")
  echo "Part ${rank}/${PART_COUNT}: physical GPU ${gpu}, Numba threads ${NUMBA_THREADS}, log ${log_path}"
  (
    set +e
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" NUMBA_NUM_THREADS="${NUMBA_THREADS}" \
      "${PYTHON}" scripts/generate_fprir.py \
      --profile full \
      --configuration-set adapt \
      --output "${part}" \
      --max-floorplans "${FLOORPLANS}" \
      --quality "${QUALITY}" \
      --fs 16000 \
      --duration-s "${DURATION}" \
      --same-room-variants "${SAME_ROOM_VARIANTS}" \
      --cross-room-variants "${CROSS_ROOM_VARIANTS}" \
      --motion-fraction "${MOTION_FRACTION}" \
      --motion-spacing-m 0.25 \
      --shard-size 32 \
      --workers 1 \
      --intersection-backend bvh \
      --rt-accelerator cuda \
      --rt-precision float32 \
      --rt-cuda-device 0 \
      --partition-count "${PART_COUNT}" \
      --partition-rank "${rank}" \
      "${EXTRA_ARGS[@]}" \
      >"${log_path}" 2>&1
    status=$?
    printf "%d\n" "${status}" >"${part}/.status"
    exit "${status}"
  ) &
  PIDS+=("$!")
done

set +e
"${PYTHON}" "${SCRIPT_DIR}/monitor_parts.py" \
  --parts-root "${PARTS_ROOT}" \
  --part-count "${PART_COUNT}"
monitor_status=$?
worker_status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || worker_status=1
done
set -e
if [[ "${monitor_status}" -ne 0 || "${worker_status}" -ne 0 ]]; then
  echo "Adapt ${TIER} failed in at least one GPU partition; inspect ${LOG_DIR}." >&2
  exit 1
fi

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  echo "Adapt ${TIER} plan complete across ${PART_COUNT} process partitions: ${PARTS_ROOT}"
  exit 0
fi

(
  cd "${REPO_ROOT}"
  "${PYTHON}" scripts/merge_fprir_parts.py \
    --output "${OUTPUT}" \
    --parts "${PART_DIRS[@]}" \
    --physical-gpu-ids "${WORKER_GPUS[@]}" \
    --nested-tier-sizes "${NESTED_SIZES[@]}"
)
