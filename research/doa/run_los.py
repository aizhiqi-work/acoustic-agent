from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_los_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Geometry and FloorPlan LOS DOA baselines.")
    parser.add_argument("--output-dir", type=Path, default=Path("research/results/doa-los"))
    parser.add_argument("--scenes", nargs="+", choices=("geometry", "floorplan"), default=("geometry", "floorplan"))
    parser.add_argument("--conditions", nargs="+", choices=("direct", "room"), default=("direct", "room"))
    parser.add_argument("--quality", choices=("preview", "simulation", "fine", "reference"), default="preview")
    parser.add_argument("--floorplan-idx", type=int, default=0)
    parser.add_argument("--accelerator", choices=("numba", "cuda", "auto"), default="numba")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Run one placement per scene for a fast smoke study.")
    args = parser.parse_args()
    if args.accelerator == "cuda" and args.precision != "float32":
        parser.error("CUDA experiments require --precision float32")
    rows = run_los_study(
        args.output_dir,
        scenes=args.scenes,
        conditions=args.conditions,
        quality=args.quality,
        floorplan_idx=args.floorplan_idx,
        rt_accelerator=args.accelerator,
        rt_precision=args.precision,
        rt_cuda_device=args.cuda_device,
        geometry_angles_deg=(30.0,) if args.quick else (30.0, 105.0, 230.0),
        floorplan_seeds=(42,) if args.quick else (42, 43, 44),
    )
    print(f"Wrote {len(rows)} estimates to {args.output_dir.resolve()}")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
