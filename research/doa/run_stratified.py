from __future__ import annotations

import argparse
from pathlib import Path

from .stratified import DEFAULT_ROOM_COUNTS, run_stratified_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the room-count and area-stratified FloorPlan distributed localization study."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/results/distributed-floorplan-stratified"),
    )
    parser.add_argument("--quality", choices=("preview", "simulation", "fine", "reference"), default="preview")
    parser.add_argument("--room-counts", nargs="+", type=int, default=DEFAULT_ROOM_COUNTS)
    parser.add_argument("--calibration-per-count", type=int, default=5)
    parser.add_argument("--validation-per-count", type=int, default=10)
    parser.add_argument("--points-per-room", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run two room-count strata with one calibration and two validation FloorPlans each.",
    )
    args = parser.parse_args()
    room_counts = tuple(args.room_counts)
    calibration = max(1, int(args.calibration_per_count))
    validation = max(1, int(args.validation_per_count))
    if args.quick:
        room_counts = room_counts[:2]
        calibration = 1
        validation = 2
    payload = run_stratified_study(
        args.output_dir,
        room_counts=room_counts,
        calibration_per_count=calibration,
        validation_per_count=validation,
        quality=args.quality,
        points_per_room=max(1, int(args.points_per_room)),
        seed=int(args.seed),
    )
    print(f"Validation FloorPlans: {len(payload['validation_indices'])}")
    print(f"Acoustic cases: {len(payload['results'])}")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
