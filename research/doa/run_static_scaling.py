from __future__ import annotations

import argparse
from pathlib import Path

from .static_scaling import (
    DEFAULT_ROOM_COUNTS,
    filter_static_scaling_outputs,
    rebuild_static_scaling_outputs,
    run_static_scaling_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the minimum synchronized singles and 4-channel arrays for static whole-home localization."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/results/static-connected-floorplan-scaling"),
    )
    parser.add_argument("--quality", choices=("preview", "simulation", "fine", "reference"), default="preview")
    parser.add_argument("--room-counts", nargs="+", type=int, default=DEFAULT_ROOM_COUNTS)
    parser.add_argument("--calibration-per-count", type=int, default=5)
    parser.add_argument("--validation-per-count", type=int, default=10)
    parser.add_argument("--points-per-room", type=int, default=1)
    parser.add_argument("--positions-per-room", type=int, default=2)
    parser.add_argument("--max-single-nodes", type=int, default=8)
    parser.add_argument("--max-array-nodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--risk-quantile",
        type=float,
        default=None,
        help="Reuse a previously calibrated placement risk quantile instead of calibrating again.",
    )
    parser.add_argument("--accelerator", choices=("numba", "cuda", "auto"), default="numba")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the first two requested room counts with one calibration and two validation FloorPlans.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild CSV and Markdown outputs from an existing summary.json without rerunning localization.",
    )
    parser.add_argument(
        "--filter-from",
        type=Path,
        default=None,
        help="Build a smaller area-stratified study from an existing output directory without rerunning localization.",
    )
    args = parser.parse_args()
    if args.report_only:
        rebuild_static_scaling_outputs(args.output_dir)
        print(f"Report: {(args.output_dir / 'report.md').resolve()}")
        return
    if args.filter_from is not None:
        payload = filter_static_scaling_outputs(
            args.filter_from,
            args.output_dir,
            room_counts=tuple(args.room_counts),
            validation_per_count=max(1, int(args.validation_per_count)),
            seed=int(args.seed),
        )
        print(f"Validation FloorPlans: {len(payload['validation_indices'])}")
        print(f"Localization cases: {len(payload['results'])}")
        print(f"Report: {(args.output_dir / 'report.md').resolve()}")
        return
    if args.accelerator == "cuda" and args.precision != "float32":
        parser.error("CUDA experiments require --precision float32")
    room_counts = tuple(args.room_counts)
    calibration = max(1, int(args.calibration_per_count))
    validation = max(1, int(args.validation_per_count))
    if args.quick:
        room_counts = room_counts[:2]
        calibration = 1
        validation = 2
    payload = run_static_scaling_study(
        args.output_dir,
        room_counts=room_counts,
        calibration_per_count=calibration,
        validation_per_count=validation,
        quality=args.quality,
        points_per_room=max(1, int(args.points_per_room)),
        positions_per_room=max(1, int(args.positions_per_room)),
        max_single_nodes=max(3, int(args.max_single_nodes)),
        max_array_nodes=max(2, int(args.max_array_nodes)),
        seed=int(args.seed),
        risk_quantile=args.risk_quantile,
        rt_accelerator=args.accelerator,
        rt_precision=args.precision,
        rt_cuda_device=args.cuda_device,
    )
    print(f"Validation FloorPlans: {sum(row['split'] == 'validation' for row in payload['split'])}")
    print(f"Localization cases: {len(payload['results'])}")
    print(f"Elapsed: {payload['elapsed_s']:.1f}s")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
