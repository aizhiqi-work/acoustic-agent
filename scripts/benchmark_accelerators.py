from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Sequence

import numpy as np
import numba

from acoustic_agent import AcousticAgent, SimConfig, make_room, simulate_rir
from acoustic_agent.api import QUALITY_PRESETS
from acoustic_agent.models import Room
from acoustic_agent.steam_rt import RoomRayScene, trace_energy_field


@dataclass(frozen=True)
class BenchmarkScene:
    name: str
    room: Room
    source: tuple[float, float, float]
    receiver: tuple[float, float, float]
    category: str = "geometry"
    floorplan_index: int | None = None
    room_count: int | None = None
    furnishing: str | None = None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Numba and single-GPU CUDA reflection tracing.")
    parser.add_argument("--accelerator", choices=("numba", "cuda"), required=True)
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--scene-set", choices=("geometry", "floorplan", "all"), default="geometry")
    parser.add_argument("--qualities", default="preview,simulation,reference")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.accelerator == "cuda" and args.precision != "float32":
        parser.error("CUDA benchmarks currently require --precision float32")

    qualities = tuple(value.strip() for value in args.qualities.split(",") if value.strip())
    unknown = sorted(set(qualities) - set(QUALITY_PRESETS))
    if unknown:
        parser.error(f"unknown qualities: {', '.join(unknown)}")

    scenes = _benchmark_scenes(args.scene_set)
    warmup_config = _config(args.accelerator, args.precision, args.device, rays=512, bounces=4)
    warmup_scene = RoomRayScene(scenes[0].room)
    warmup_started = time.perf_counter()
    trace_energy_field(
        warmup_scene,
        np.asarray(scenes[0].source),
        np.asarray(scenes[0].receiver),
        warmup_config,
        render_ambisonics=False,
    )
    warmup_s = time.perf_counter() - warmup_started

    records: list[dict[str, Any]] = []
    for scene_case in scenes:
        ray_scene = RoomRayScene(scene_case.room)
        source = np.asarray(scene_case.source)
        receiver = np.asarray(scene_case.receiver)
        for quality in qualities:
            preset = QUALITY_PRESETS[quality]
            config = _config(
                args.accelerator,
                args.precision,
                args.device,
                rays=int(preset["rt_num_rays"]),
                bounces=int(preset["rt_num_bounces"]),
            )
            trace_times: list[float] = []
            end_to_end_times: list[float] = []
            kernel_times: list[float] = []
            transfer_times: list[float] = []
            field: dict[str, Any] | None = None
            result = None
            for _ in range(max(1, int(args.repeats))):
                started = time.perf_counter()
                field = trace_energy_field(ray_scene, source, receiver, config, render_ambisonics=False)
                trace_times.append(time.perf_counter() - started)
                if field.get("kernel_time_s") is not None:
                    kernel_times.append(float(field["kernel_time_s"]))
                    transfer_times.append(float(field["transfer_time_s"]))

                started = time.perf_counter()
                result = simulate_rir(scene_case.room, source, receiver, config=config)
                end_to_end_times.append(time.perf_counter() - started)

            assert field is not None and result is not None
            traced_rt60 = [
                float(value)
                for value in result.rt60.get("traced_rt60_bands", {}).values()
                if float(value) > 0.0
            ]
            records.append({
                "scene": scene_case.name,
                "surface_count": len(ray_scene.surfaces),
                "intersection_backend": field.get("intersection_backend"),
                "quality": quality,
                "category": scene_case.category,
                "floorplan_index": scene_case.floorplan_index,
                "room_count": scene_case.room_count,
                "furnishing": scene_case.furnishing,
                "rays": int(config.rt_num_rays),
                "bounces": int(config.rt_num_bounces),
                "trace_s_median": statistics.median(trace_times),
                "trace_s_min": min(trace_times),
                "end_to_end_s_median": statistics.median(end_to_end_times),
                "end_to_end_s_min": min(end_to_end_times),
                "kernel_s_median": statistics.median(kernel_times) if kernel_times else None,
                "transfer_s_median": statistics.median(transfer_times) if transfer_times else None,
                "echogram_energy": float(np.sum(field["echogram"], dtype=np.float64)),
                "rir_peak": float(np.max(np.abs(result.rir))),
                "traced_rt60_s_mean": float(np.mean(traced_rt60)) if traced_rt60 else 0.0,
                "rir_rt60_s": float(result.rt60.get("rir_rt60_s", 0.0)),
                "active_ray_count": int(field["active_ray_count"]),
                "actual_bounces": int(field["actual_bounces"]),
            })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accelerator": args.accelerator,
        "precision": args.precision,
        "device": int(args.device),
        "warmup_s": warmup_s,
        "repeats": max(1, int(args.repeats)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "numba_threads": int(numba.get_num_threads()),
        },
        "scenes": [
            {
                "name": scene.name,
                "source": scene.source,
                "receiver": scene.receiver,
                "shape": scene.room.metadata.get("shape"),
                "object_count": len(scene.room.metadata.get("objects", [])),
                "category": scene.category,
                "floorplan_index": scene.floorplan_index,
                "room_count": scene.room_count,
                "furnishing": scene.furnishing,
            }
            for scene in scenes
        ],
        "results": records,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _config(accelerator: str, precision: str, device: int, *, rays: int, bounces: int) -> SimConfig:
    return SimConfig(
        duration_s=1.0,
        rt_duration_s=1.0,
        rt_num_rays=int(rays),
        rt_num_bounces=int(bounces),
        rt_accelerator=accelerator,
        rt_precision=precision,
        rt_cuda_device=int(device),
        late_tail=False,
        diffraction_enabled=False,
        direct_occlusion=False,
        adaptive_geometry_bounces=False,
        adaptive_cross_room_bounces=False,
        collect_visual_paths=False,
        render_ambisonics=False,
    )


def _benchmark_scenes(scene_set: str = "geometry") -> tuple[BenchmarkScene, ...]:
    simple = make_room("rectangle", size=(8.0, 6.0, 3.0), material_seed=17)
    medium = _with_objects(
        make_room("u_shape", size=(8.0, 6.0, 3.0), material_seed=17),
        rows=2,
        columns=6,
        x_range=(1.6, 6.4),
        y_range=(2.0, 2.8),
    )
    complex_room = _with_objects(
        make_room("circle", size=(10.0, 8.0, 3.2), circle_segments=64, material_seed=17),
        rows=4,
        columns=8,
        x_range=(2.0, 8.0),
        y_range=(2.0, 6.0),
    )
    geometry_scenes = (
        BenchmarkScene("simple_rectangle", simple, (0.8, 0.8, 1.4), (7.2, 1.4, 1.4)),
        BenchmarkScene("medium_u_furnished", medium, (0.8, 0.8, 1.4), (7.2, 1.4, 1.4)),
        BenchmarkScene("complex_round_furnished", complex_room, (1.2, 4.0, 1.4), (8.8, 4.0, 1.4)),
    )
    if scene_set == "geometry":
        return geometry_scenes
    floorplan_scenes = _floorplan_benchmark_scenes()
    if scene_set == "floorplan":
        return floorplan_scenes
    return geometry_scenes + floorplan_scenes


def _floorplan_benchmark_scenes() -> tuple[BenchmarkScene, ...]:
    cases = (
        (12513, 5, None),
        (12513, 5, "balanced"),
        (11282, 10, None),
        (11282, 10, "balanced"),
    )
    scenes: list[BenchmarkScene] = []
    for index, expected_rooms, furnishing in cases:
        agent = AcousticAgent.create(
            scene="floorplan",
            idx=index,
            placement="cross_room",
            seed=20260722,
            furnishing=furnishing,
            visualization=False,
        )
        if len(agent.rooms) != expected_rooms:
            raise RuntimeError(
                f"FloorPlan {index} expected {expected_rooms} rooms, found {len(agent.rooms)}"
            )
        if agent.default_source is None or agent.default_receiver is None:
            raise RuntimeError(f"FloorPlan {index} did not resolve benchmark positions")
        suffix = "furnished" if furnishing else "empty"
        scenes.append(
            BenchmarkScene(
                name=f"floorplan_{expected_rooms}_rooms_{suffix}",
                room=agent.room,
                source=agent.default_source,
                receiver=agent.default_receiver,
                category="floorplan",
                floorplan_index=index,
                room_count=expected_rooms,
                furnishing=furnishing or "none",
            )
        )
    return tuple(scenes)


def _with_objects(
    room: Room,
    *,
    rows: int,
    columns: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> Room:
    xs = np.linspace(x_range[0], x_range[1], columns)
    ys = np.linspace(y_range[0], y_range[1], rows)
    objects = []
    for index, (x, y) in enumerate((float(x), float(y)) for y in ys for x in xs):
        objects.append({
            "id": f"cabinet_{index}",
            "type": "cabinet",
            "semantic": "cabinet_storage",
            "position": [x, y],
            "size": [0.45, 0.35, 1.2],
            "z": 0.6,
            "rotation": float((index % 4) * 15),
        })
    return replace(room, metadata={**dict(room.metadata), "objects": objects})


if __name__ == "__main__":
    raise SystemExit(main())
