from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import DEFAULT_ROOM_COUNTS
from .whole_home_benchmark import run_whole_home_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-deployment whole-home TDOA and beamforming benchmarks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/whole-home-benchmark"),
    )
    parser.add_argument("--room-counts", type=int, nargs="+", default=DEFAULT_ROOM_COUNTS)
    parser.add_argument("--plans-per-room-count", type=int, default=5)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("same_room", "cross_room"),
        default=("same_room", "cross_room"),
    )
    parser.add_argument(
        "--quality",
        choices=("preview", "simulation", "fine", "reference"),
        default="preview",
    )
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--rir-duration", type=float, default=1.0)
    parser.add_argument("--interferer-snr", type=float, default=0.0)
    parser.add_argument("--background-snr", type=float, default=10.0)
    parser.add_argument("--sensor-noise-snr", type=float, default=30.0)
    parser.add_argument("--rt-accelerator", choices=("numba", "cuda"), default="numba")
    parser.add_argument("--rt-precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--rt-cuda-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    arguments = parser.parse_args()
    payload = run_whole_home_benchmark(
        arguments.output,
        room_counts=arguments.room_counts,
        plans_per_room_count=arguments.plans_per_room_count,
        scenarios=arguments.scenarios,
        quality=arguments.quality,
        duration_s=arguments.duration,
        rir_duration_s=arguments.rir_duration,
        interferer_snr_db=arguments.interferer_snr,
        background_snr_db=arguments.background_snr,
        sensor_noise_snr_db=arguments.sensor_noise_snr,
        rt_accelerator=arguments.rt_accelerator,
        rt_precision=arguments.rt_precision,
        rt_cuda_device=arguments.rt_cuda_device,
        seed=arguments.seed,
    )
    print(arguments.output / "report.md")
    print(f"floorplans={payload['floorplan_count']} cases={payload['case_count']}")


if __name__ == "__main__":
    main()
