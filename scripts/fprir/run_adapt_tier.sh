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
    SAME_ROOM_VARIANTS=1
    CROSS_ROOM_VARIANTS=1
    NESTED_SIZES=(10)
    ;;
  1k)
    FLOORPLANS=1000
    QUALITY=simulation
    DURATION=2.0
    MOTION_FRACTION=0.3
    SAME_ROOM_VARIANTS=2
    CROSS_ROOM_VARIANTS=3
    NESTED_SIZES=(1000)
    ;;
  8k)
    FLOORPLANS=8000
    QUALITY=simulation
    DURATION=2.0
    MOTION_FRACTION=0.3
    SAME_ROOM_VARIANTS=2
    CROSS_ROOM_VARIANTS=3
    NESTED_SIZES=(1000 8000)
    ;;
  full)
    FLOORPLANS=15376
    QUALITY=simulation
    DURATION=2.0
    MOTION_FRACTION=0.3
    SAME_ROOM_VARIANTS=2
    CROSS_ROOM_VARIANTS=3
    NESTED_SIZES=(1000 8000 15376)
    ;;
  *)
    echo "Usage: $0 {smoke|1k|8k|full}" >&2
    exit 2
    ;;
esac

RT_ACCELERATOR="${FPRIR_RT_ACCELERATOR:-cuda}"
if [[ "${RT_ACCELERATOR}" == "numba" ]]; then
  RT_PRECISION="${FPRIR_RT_PRECISION:-float64}"
else
  RT_PRECISION="${FPRIR_RT_PRECISION:-float32}"
fi

SAME_ROOM_VARIANTS="${FPRIR_SAME_ROOM_VARIANTS:-${SAME_ROOM_VARIANTS}}"
CROSS_ROOM_VARIANTS="${FPRIR_CROSS_ROOM_VARIANTS:-${CROSS_ROOM_VARIANTS}}"

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
  --same-room-variants "${SAME_ROOM_VARIANTS}"
  --cross-room-variants "${CROSS_ROOM_VARIANTS}"
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
