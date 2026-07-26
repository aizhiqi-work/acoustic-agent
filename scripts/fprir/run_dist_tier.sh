#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TIER="${1:-quick}"
STAGE="${DIST_STAGE:-all}"
ROOM_COUNTS=(4 6 8 10 12)

case "${TIER}" in
  quick)
    QUALITY=preview
    CALIBRATION=1
    VALIDATION=2
    POINTS=2
    PLANS_PER_COUNT=1
    ;;
  standard)
    QUALITY=simulation
    CALIBRATION=5
    VALIDATION=10
    POINTS=4
    PLANS_PER_COUNT=5
    ;;
  extended)
    QUALITY=simulation
    CALIBRATION=10
    VALIDATION=20
    POINTS=6
    PLANS_PER_COUNT=10
    ;;
  *)
    echo "Usage: $0 {quick|standard|extended}" >&2
    exit 2
    ;;
esac

OUTPUT="${FPRIR_OUTPUT_ROOT}/dist-${TIER}"
mkdir -p "${OUTPUT}"

run_localization() {
  if [[ "${FORCE:-0}" != "1" && -s "${OUTPUT}/localization/summary.json" ]]; then
    echo "Localization result already complete: ${OUTPUT}/localization/summary.json"
    return
  fi
  run_logged "dist-${TIER}-localization" \
    "${PYTHON}" -m research.doa.run_stratified \
    --output-dir "${OUTPUT}/localization" \
    --quality "${QUALITY}" \
    --room-counts "${ROOM_COUNTS[@]}" \
    --calibration-per-count "${CALIBRATION}" \
    --validation-per-count "${VALIDATION}" \
    --points-per-room "${POINTS}" \
    --accelerator cuda \
    --precision float32 \
    --cuda-device 0
}

run_beamforming() {
  if [[ "${FORCE:-0}" != "1" && -s "${OUTPUT}/beamforming/summary.json" ]]; then
    echo "Beamforming result already complete: ${OUTPUT}/beamforming/summary.json"
    return
  fi
  run_logged "dist-${TIER}-beamforming" \
    "${PYTHON}" -m research.beamforming.run_whole_home_benchmark \
    --output "${OUTPUT}/beamforming" \
    --room-counts "${ROOM_COUNTS[@]}" \
    --plans-per-room-count "${PLANS_PER_COUNT}" \
    --scenarios same_room cross_room \
    --quality "${QUALITY}" \
    --duration 2.5 \
    --rir-duration 1.0 \
    --interferer-snr 0 \
    --background-snr 10 \
    --sensor-noise-snr 30 \
    --rt-accelerator cuda \
    --rt-precision float32 \
    --rt-cuda-device 0 \
    --seed 20260723
}

case "${STAGE}" in
  all)
    run_localization
    run_beamforming
    ;;
  localization)
    run_localization
    ;;
  beamforming)
    run_beamforming
    ;;
  *)
    echo "DIST_STAGE must be all, localization, or beamforming" >&2
    exit 2
    ;;
esac
