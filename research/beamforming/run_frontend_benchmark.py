from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import DEFAULT_ROOM_COUNTS
from .frontend_benchmark import run_frontend_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete single/local/distributed audio front-end benchmark.")
    parser.add_argument("--output", type=Path, default=Path("research/results/frontend-benchmark"))
    parser.add_argument("--room-counts", type=int, nargs="+", default=DEFAULT_ROOM_COUNTS)
    parser.add_argument("--plans-per-room-count", type=int, default=5)
    parser.add_argument("--quality", choices=("preview", "simulation", "fine", "reference"), default="preview")
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--rir-duration", type=float, default=1.0)
    parser.add_argument("--rt-accelerator", choices=("numba", "cuda"), default="numba")
    parser.add_argument("--rt-precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--rt-cuda-device", type=int, default=0)
    arguments = parser.parse_args()
    payload = run_frontend_benchmark(
        arguments.output,
        room_counts=arguments.room_counts,
        plans_per_room_count=arguments.plans_per_room_count,
        quality=arguments.quality,
        duration_s=arguments.duration,
        rir_duration_s=arguments.rir_duration,
        rt_accelerator=arguments.rt_accelerator,
        rt_precision=arguments.rt_precision,
        rt_cuda_device=arguments.rt_cuda_device,
    )
    print(arguments.output / "report.md")
    print(f"floorplans={payload['floorplan_count']} rows={len(payload['results'])}")


if __name__ == "__main__":
    main()
