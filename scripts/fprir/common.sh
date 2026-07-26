#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
FPRIR_OUTPUT_ROOT="${FPRIR_OUTPUT_ROOT:-${REPO_ROOT}/research/results/fprir-tiers}"
LOG_DIR="${LOG_DIR:-${FPRIR_OUTPUT_ROOT}/logs}"

mkdir -p "${FPRIR_OUTPUT_ROOT}" "${LOG_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  echo "Create .venv and install the project with research dependencies first." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

run_logged() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}-$(date +%Y%m%d-%H%M%S).log"
  echo "Repository : ${REPO_ROOT}"
  echo "Python     : ${PYTHON}"
  if [[ "${FPRIR_RT_ACCELERATOR:-cuda}" == "numba" ]]; then
    echo "Accelerator: Numba CPU (${NUMBA_NUM_THREADS:-auto} threads)"
  else
    echo "Physical GPU: ${GPU_ID} (solver device 0)"
  fi
  echo "Log        : ${log_path}"
  (
    cd "${REPO_ROOT}"
    "$@"
  ) 2>&1 | tee "${log_path}"
}
