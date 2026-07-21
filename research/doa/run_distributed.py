from __future__ import annotations

import argparse
from pathlib import Path

from .distributed import TEST_INDICES, TRAIN_INDICES, run_distributed_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FloorPlan distributed microphone localization study.")
    parser.add_argument("--output-dir", type=Path, default=Path("research/results/distributed-floorplan"))
    parser.add_argument("--quality", choices=("preview", "simulation", "fine", "reference"), default="preview")
    parser.add_argument("--train-indices", nargs="+", type=int, default=TRAIN_INDICES)
    parser.add_argument("--test-indices", nargs="+", type=int, default=TEST_INDICES)
    parser.add_argument("--points-per-room", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Use two train and two test FloorPlans.")
    args = parser.parse_args()
    train_indices = tuple(args.train_indices[:2]) if args.quick else tuple(args.train_indices)
    test_indices = tuple(args.test_indices[:2]) if args.quick else tuple(args.test_indices)
    payload = run_distributed_study(
        args.output_dir,
        train_indices=train_indices,
        test_indices=test_indices,
        quality=args.quality,
        points_per_room=max(1, int(args.points_per_room)),
    )
    chosen = payload["selected_configuration"]
    print(f"Selected: {chosen['configuration']} ({chosen['channels']} channels)")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
