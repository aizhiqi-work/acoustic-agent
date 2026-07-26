#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TIER="${1:-smoke}"
case "${TIER}" in
  smoke)
    FLOORPLANS=10
    QUALITY=preview
    DURATION=1.0
    MOTION_FRACTION=0.5
    NESTED_SIZES=(10)
    ;;
  1k)
    FLOORPLANS=1000
    QUALITY=simulation
    DURATION=2.0
    MOTION_FRACTION=0.1
    NESTED_SIZES=(1000)
    ;;
  3k)
    FLOORPLANS=3000
    QUALITY=simulation
    DURATION=2.0
    MOTION_FRACTION=0.1
    NESTED_SIZES=(1000 3000)
    ;;
  6k)
    FLOORPLANS=6000
    QUALITY=simulation
    DURATION=2.0
    MOTION_FRACTION=0.1
    NESTED_SIZES=(1000 3000 6000)
    ;;
  *)
    echo "Usage: $0 {smoke|1k|3k|6k}" >&2
    exit 2
    ;;
esac

RT_ACCELERATOR="${FPRIR_RT_ACCELERATOR:-cuda}"
if [[ "${RT_ACCELERATOR}" == "numba" ]]; then
  RT_PRECISION="${FPRIR_RT_PRECISION:-float64}"
else
  RT_PRECISION="${FPRIR_RT_PRECISION:-float32}"
fi

OUTPUT="${FPRIR_OUTPUT_ROOT}/adapt-${TIER}"
ARGS=(
  "${PYTHON}" scripts/generate_fprir.py
  --profile full
  --configuration-set adapt
  --output "${OUTPUT}"
  --max-floorplans "${FLOORPLANS}"
  --quality "${QUALITY}"
  --fs 16000
  --duration-s "${DURATION}"
  --motion-fraction "${MOTION_FRACTION}"
  --motion-spacing-m 0.25
  --shard-size 32
  --workers 1
  --intersection-backend bvh
  --rt-accelerator "${RT_ACCELERATOR}"
  --rt-precision "${RT_PRECISION}"
  --rt-cuda-device 0
  --nested-tier-sizes "${NESTED_SIZES[@]}"
)

if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  ARGS+=(--plan-only)
fi

run_logged "adapt-${TIER}" "${ARGS[@]}"
