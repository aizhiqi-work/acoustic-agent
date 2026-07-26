from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from research.doa.distributed import (
    SensorNode,
    TOAMeasurement,
    TargetPoint,
    candidate_nodes,
    load_model,
    localization_grid,
    localize_tdoa,
    place_nodes,
    sample_target_points,
)

from .audio_io import load_wav_mono, write_wav_mono
from .benchmark import (
    DEFAULT_ROOM_COUNTS,
    _benchmark_evaluation_slice,
    _format_metric,
    _recommended_microphone_counts,
    _validation_split,
    _write_csv,
)
from .core import estimate_gcc_phat_delays, scale_background_to_snr
from .experiment import (
    FS,
    RESOURCE_AUDIO,
    _calibration_slice,
    _distributed_rirs,
    _render_distributed,
    _select_node_indices,
    _study_audio,
    _tile_audio,
)
from .frontend_benchmark import (
    _fit_render,
    _local_array_rir,
    _local_steering,
    _run_architecture,
)


ARRAY_COUNTS_BY_ROOM_COUNT = {4: 1, 6: 1, 8: 1, 10: 2, 12: 2}
WHOLE_HOME_STRATEGIES = (
    "oracle_target_room_single_raw",
    "tdoa_routed_single_raw",
    "tdoa_routed_single_wiener",
    "oracle_target_room_array_mwf",
    "tdoa_routed_fixed_array_ds",
    "tdoa_routed_fixed_array_mwf",
    "distributed_singles_mwf",
    "equal_channel_hybrid_mwf",
    "coverage_hybrid_selected_mwf",
    "coverage_hybrid_all_mwf",
)


def run_whole_home_benchmark(
    output_dir: str | Path,
    *,
    room_counts: Sequence[int] = DEFAULT_ROOM_COUNTS,
    plans_per_room_count: int = 5,
    scenarios: Sequence[str] = ("same_room", "cross_room"),
    quality: str = "preview",
    duration_s: float = 2.5,
    rir_duration_s: float = 1.0,
    interferer_snr_db: float = 0.0,
    background_snr_db: float = 10.0,
    sensor_noise_snr_db: float = 30.0,
    rt_accelerator: str = "numba",
    rt_precision: str = "float64",
    rt_cuda_device: int = 0,
    seed: int = 20260723,
) -> dict[str, Any]:
    """Evaluate fixed whole-home microphone deployments after TDOA routing.

    Sensor positions are selected before target positions. Source gains are
    calibrated once at an oracle target-room reference and then held constant
    across every deployment so that coverage and wall attenuation remain part
    of the comparison.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_counts = tuple(int(value) for value in room_counts)
    selected_scenarios = tuple(str(value).strip().lower() for value in scenarios)
    if not selected_scenarios or set(selected_scenarios) - {"same_room", "cross_room"}:
        raise ValueError("scenarios may contain same_room and cross_room")
    missing_array_counts = sorted(set(selected_counts) - set(ARRAY_COUNTS_BY_ROOM_COUNT))
    if missing_array_counts:
        raise ValueError(f"no fixed-array policy for room counts: {missing_array_counts}")

    split = _validation_split(selected_counts, int(plans_per_room_count))
    single_counts = _recommended_microphone_counts(selected_counts)
    signature = {
        "version": 1,
        "indices": [int(row["index"]) for row in split],
        "scenarios": list(selected_scenarios),
        "quality": str(quality),
        "duration_s": float(duration_s),
        "rir_duration_s": float(rir_duration_s),
        "interferer_snr_db": float(interferer_snr_db),
        "background_snr_db": float(background_snr_db),
        "sensor_noise_snr_db": float(sensor_noise_snr_db),
        "rt_accelerator": str(rt_accelerator),
        "rt_precision": str(rt_precision),
        "rt_cuda_device": int(rt_cuda_device),
        "seed": int(seed),
    }
    rows, cases = _load_checkpoint(output, signature)
    completed = {
        (int(case["floorplan_idx"]), str(case["target_case"]), str(case["scenario"]))
        for case in cases
    }
    resumed_case_count = len(cases)
    total_cases = len(split) * 2 * len(selected_scenarios)
    target_dry, interferer_dry = _study_audio(float(duration_s), FS)
    pink_dry = _tile_audio(
        load_wav_mono(RESOURCE_AUDIO / "pink_noise_bed.wav", FS), target_dry.size
    )
    started = time.perf_counter()

    progress = tqdm(
        total=total_cases,
        initial=len(completed),
        desc="Beamforming",
        unit="case",
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.1,
    )
    for plan_number, split_row in enumerate(split, start=1):
        floorplan_idx = int(split_row["index"])
        room_count = int(split_row["room_count"])
        progress.set_postfix(
            floorplan=floorplan_idx,
            rooms=room_count,
            refresh=False,
        )
        if all(
            (floorplan_idx, target_case, scenario) in completed
            for target_case in ("array_covered", "array_uncovered")
            for scenario in selected_scenarios
        ):
            continue

        model = load_model(floorplan_idx)
        center_candidates = candidate_nodes(model, positions_per_room=1)
        all_candidates = candidate_nodes(model, positions_per_room=2)
        fixed_singles = place_nodes(
            model,
            single_counts[room_count],
            mode="single",
            risk_quantile=0.2,
            candidates=all_candidates,
        )
        fixed_arrays = place_nodes(
            model,
            ARRAY_COUNTS_BY_ROOM_COUNT[room_count],
            mode="array",
            risk_quantile=0.2,
            candidates=center_candidates,
        )
        target_cases = _target_cases(
            model,
            fixed_arrays,
            seed=int(seed) + floorplan_idx,
        )
        cache = output / "rir"

        for target_case, target in target_cases.items():
            room_points = sample_target_points(
                model,
                points_per_room=3,
                seed=int(seed) + floorplan_idx * 17 + _stable_label_seed(target_case),
            )
            same_interferer = _same_room_interferer(target, room_points)
            cross_interferer = _cross_room_interferer(model, target, room_points)
            noise_source = _noise_source(model, target, room_points)
            oracle_node = _room_center_node(model, target.room_id, center_candidates)

            oracle_target = _render_mono(
                cache, model, target, oracle_node, target_dry, quality, rir_duration_s,
                rt_accelerator, rt_precision, rt_cuda_device,
            )
            oracle_pink = _render_mono(
                cache, model, noise_source, oracle_node, pink_dry, quality, rir_duration_s,
                rt_accelerator, rt_precision, rt_cuda_device,
            )

            fixed_target = _render_mono_group(
                cache, model, target, fixed_singles, target_dry, quality, rir_duration_s,
                rt_accelerator, rt_precision, rt_cuda_device,
            )
            fixed_pink = _render_mono_group(
                cache, model, noise_source, fixed_singles, pink_dry, quality, rir_duration_s,
                rt_accelerator, rt_precision, rt_cuda_device,
            )
            array_target, array_models = _render_array_group(
                cache, model, target, fixed_arrays, target_dry, quality, rir_duration_s,
                rt_accelerator, rt_precision, rt_cuda_device,
            )
            array_pink, _ = _render_array_group(
                cache, model, noise_source, fixed_arrays, pink_dry, quality, rir_duration_s,
                rt_accelerator, rt_precision, rt_cuda_device,
            )
            oracle_array_target, oracle_array_model = _render_array(
                cache, model, target, oracle_node, target_dry, quality, rir_duration_s,
                rt_accelerator, rt_precision, rt_cuda_device,
            )
            oracle_array_pink, _ = _render_array(
                cache, model, noise_source, oracle_node, pink_dry, quality, rir_duration_s,
                rt_accelerator, rt_precision, rt_cuda_device,
            )

            for scenario in selected_scenarios:
                case_key = (floorplan_idx, target_case, scenario)
                if case_key in completed:
                    continue
                interferer = same_interferer if scenario == "same_room" else cross_interferer
                oracle_interferer = _render_mono(
                    cache, model, interferer, oracle_node, interferer_dry, quality,
                    rir_duration_s, rt_accelerator, rt_precision, rt_cuda_device,
                )
                gains = _global_source_gains(
                    oracle_target,
                    oracle_interferer,
                    oracle_pink,
                    target_dry.size,
                    interferer_snr_db,
                    background_snr_db,
                    sensor_noise_snr_db,
                )
                oracle_components = _mix_with_global_gains(
                    oracle_target,
                    oracle_interferer,
                    oracle_pink,
                    gains,
                    seed + floorplan_idx * 101 + len(cases),
                )

                fixed_interferer = _render_mono_group(
                    cache, model, interferer, fixed_singles, interferer_dry, quality,
                    rir_duration_s, rt_accelerator, rt_precision, rt_cuda_device,
                )
                fixed_components = _mix_with_global_gains(
                    fixed_target,
                    fixed_interferer,
                    fixed_pink,
                    gains,
                    seed + floorplan_idx * 103 + len(cases),
                )
                array_interferer, _ = _render_array_group(
                    cache, model, interferer, fixed_arrays, interferer_dry, quality,
                    rir_duration_s, rt_accelerator, rt_precision, rt_cuda_device,
                )
                array_components = [
                    _mix_with_global_gains(
                        array_target[index],
                        array_interferer[index],
                        array_pink[index],
                        gains,
                        seed + floorplan_idx * 107 + len(cases) + index,
                    )
                    for index in range(len(fixed_arrays))
                ]
                oracle_array_interferer, _ = _render_array(
                    cache, model, interferer, oracle_node, interferer_dry, quality,
                    rir_duration_s, rt_accelerator, rt_precision, rt_cuda_device,
                )
                oracle_array_components = _mix_with_global_gains(
                    oracle_array_target,
                    oracle_array_interferer,
                    oracle_array_pink,
                    gains,
                    seed + floorplan_idx * 109 + len(cases),
                )

                arrivals, estimated_position, estimated_room, localization_metadata = _tdoa_localize(
                    model,
                    fixed_singles,
                    fixed_components,
                    target,
                )
                activity = _target_activity(fixed_components)
                selected_single = _select_node_indices(
                    model,
                    fixed_singles,
                    (*estimated_position, 1.5),
                    estimated_room,
                    activity,
                    1,
                )[0]
                selected_array = _select_fixed_node(
                    model,
                    fixed_arrays,
                    estimated_position,
                    estimated_room,
                )
                common = {
                    "floorplan_idx": floorplan_idx,
                    "room_count": room_count,
                    "area_m2": round(float(split_row["area_m2"]), 4),
                    "target_case": target_case,
                    "scenario": scenario,
                    "target_room": target.room_id,
                    "interferer_room": interferer.room_id,
                    "noise_room": noise_source.room_id,
                    "estimated_room": estimated_room,
                    "room_correct": bool(estimated_room == target.room_id),
                    "localization_error_m": localization_metadata["localization_error_m"],
                    "target_room_has_fixed_single": any(
                        node.room_id == target.room_id for node in fixed_singles
                    ),
                    "target_room_has_fixed_array": any(
                        node.room_id == target.room_id for node in fixed_arrays
                    ),
                    "fixed_single_count": len(fixed_singles),
                    "fixed_array_count": len(fixed_arrays),
                }
                captured: dict[str, np.ndarray] | None = (
                    {} if plan_number == 1 and target_case == "array_uncovered" and scenario == "cross_room" else None
                )

                rows.extend(
                    _strategy_rows(
                        strategy="oracle_target_room_single_raw",
                        deployment="oracle_local_single",
                        processing_architecture="single",
                        processing_pipeline="single_raw",
                        components=oracle_components,
                        arrivals=np.zeros(1),
                        target_dry=target_dry,
                        physical_devices=1,
                        physical_channels=1,
                        common={**common, "selection_mode": "oracle", "selected_room": target.room_id},
                        captured=captured,
                    )
                )
                routed_single_components = _take_channels(fixed_components, [selected_single])
                for strategy, pipeline in (
                    ("tdoa_routed_single_raw", "single_raw"),
                    ("tdoa_routed_single_wiener", "single_wiener"),
                ):
                    rows.extend(
                        _strategy_rows(
                            strategy=strategy,
                            deployment="fixed_distributed_singles",
                            processing_architecture="single",
                            processing_pipeline=pipeline,
                            components=routed_single_components,
                            arrivals=np.zeros(1),
                            target_dry=target_dry,
                            physical_devices=len(fixed_singles),
                            physical_channels=len(fixed_singles),
                            common={
                                **common,
                                "selection_mode": "tdoa_room",
                                "selected_room": fixed_singles[selected_single].room_id,
                            },
                            captured=captured,
                        )
                    )

                oracle_array_arrivals, _ = _local_steering(
                    oracle_node,
                    oracle_array_model,
                    oracle_array_components,
                    target.position,
                )
                rows.extend(
                    _strategy_rows(
                        strategy="oracle_target_room_array_mwf",
                        deployment="oracle_local_array",
                        processing_architecture="local_array_4ch",
                        processing_pipeline="local_mwf",
                        components=oracle_array_components,
                        arrivals=oracle_array_arrivals,
                        target_dry=target_dry,
                        physical_devices=1,
                        physical_channels=4,
                        common={**common, "selection_mode": "oracle", "selected_room": target.room_id},
                        captured=captured,
                    )
                )

                selected_array_components = array_components[selected_array]
                selected_array_arrivals, _ = _local_steering(
                    fixed_arrays[selected_array],
                    array_models[selected_array],
                    selected_array_components,
                    target.position,
                )
                for strategy, pipeline in (
                    ("tdoa_routed_fixed_array_ds", "local_ds"),
                    ("tdoa_routed_fixed_array_mwf", "local_mwf"),
                ):
                    rows.extend(
                        _strategy_rows(
                            strategy=strategy,
                            deployment="fixed_arrays",
                            processing_architecture="local_array_4ch",
                            processing_pipeline=pipeline,
                            components=selected_array_components,
                            arrivals=selected_array_arrivals,
                            target_dry=target_dry,
                            physical_devices=len(fixed_singles) + len(fixed_arrays),
                            physical_channels=len(fixed_singles) + 4 * len(fixed_arrays),
                            common={
                                **common,
                                "selection_mode": "tdoa_room",
                                "selected_room": fixed_arrays[selected_array].room_id,
                            },
                            captured=captured,
                        )
                    )

                rows.extend(
                    _strategy_rows(
                        strategy="distributed_singles_mwf",
                        deployment="fixed_distributed_singles",
                        processing_architecture="distributed_singles",
                        processing_pipeline="distributed_mwf",
                        components=fixed_components,
                        arrivals=arrivals,
                        target_dry=target_dry,
                        physical_devices=len(fixed_singles),
                        physical_channels=len(fixed_singles),
                        common={**common, "selection_mode": "all", "selected_room": "multiple"},
                        captured=captured,
                    )
                )

                fair_single_count = max(1, len(fixed_singles) - 4)
                fair_components = _concatenate_components(
                    _take_channels(fixed_components, range(fair_single_count)),
                    array_components[0],
                )
                fair_arrivals = _measured_arrivals(fair_components)
                rows.extend(
                    _strategy_rows(
                        strategy="equal_channel_hybrid_mwf",
                        deployment="one_array_plus_budget_singles",
                        processing_architecture="hybrid_array_singles",
                        processing_pipeline="distributed_mwf",
                        components=fair_components,
                        arrivals=fair_arrivals,
                        target_dry=target_dry,
                        physical_devices=fair_single_count + 1,
                        physical_channels=fair_single_count + 4,
                        common={
                            **common,
                            "selection_mode": "fixed_equal_channel",
                            "selected_room": "multiple",
                        },
                        captured=captured,
                    )
                )

                selected_single_indices = _select_node_indices(
                    model,
                    fixed_singles,
                    (*estimated_position, 1.5),
                    estimated_room,
                    activity,
                    min(3, len(fixed_singles)),
                )
                selected_hybrid = _concatenate_components(
                    _take_channels(fixed_components, selected_single_indices),
                    selected_array_components,
                )
                selected_hybrid_arrivals = _measured_arrivals(selected_hybrid)
                rows.extend(
                    _strategy_rows(
                        strategy="coverage_hybrid_selected_mwf",
                        deployment="coverage_singles_plus_arrays",
                        processing_architecture="hybrid_array_singles",
                        processing_pipeline="distributed_mwf",
                        components=selected_hybrid,
                        arrivals=selected_hybrid_arrivals,
                        target_dry=target_dry,
                        physical_devices=len(fixed_singles) + len(fixed_arrays),
                        physical_channels=len(fixed_singles) + 4 * len(fixed_arrays),
                        common={
                            **common,
                            "selection_mode": "tdoa_room_subset",
                            "selected_room": fixed_arrays[selected_array].room_id,
                        },
                        captured=captured,
                    )
                )

                all_hybrid = _concatenate_components(
                    fixed_components,
                    *array_components,
                )
                all_hybrid_arrivals = _measured_arrivals(all_hybrid)
                rows.extend(
                    _strategy_rows(
                        strategy="coverage_hybrid_all_mwf",
                        deployment="coverage_singles_plus_arrays",
                        processing_architecture="hybrid_array_singles",
                        processing_pipeline="distributed_mwf",
                        components=all_hybrid,
                        arrivals=all_hybrid_arrivals,
                        target_dry=target_dry,
                        physical_devices=len(fixed_singles) + len(fixed_arrays),
                        physical_channels=len(fixed_singles) + 4 * len(fixed_arrays),
                        common={**common, "selection_mode": "all", "selected_room": "multiple"},
                        captured=captured,
                    )
                )

                cases.append(
                    {
                        "floorplan_idx": floorplan_idx,
                        "room_count": room_count,
                        "target_case": target_case,
                        "scenario": scenario,
                        "target_room": target.room_id,
                        "estimated_room": estimated_room,
                        "room_correct": bool(estimated_room == target.room_id),
                        "localization_error_m": localization_metadata["localization_error_m"],
                        "fixed_single_rooms": [node.room_id for node in fixed_singles],
                        "fixed_array_rooms": [node.room_id for node in fixed_arrays],
                    }
                )
                completed.add(case_key)
                _write_checkpoint(output, signature, rows, cases)
                progress.update(1)
                if captured is not None:
                    audio_dir = output / "audio" / f"floorplan-{floorplan_idx}-{target_case}-{scenario}"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    write_wav_mono(audio_dir / "clean-dry-reference.wav", target_dry, FS)
                    for name, samples in captured.items():
                        write_wav_mono(audio_dir / f"{name}.wav", samples, FS)

    progress.close()

    overall = _summarize(rows, ("strategy",))
    scenario_strategy = _summarize(rows, ("scenario", "strategy"))
    coverage_strategy = _summarize(rows, ("target_case", "strategy"))
    room_strategy = _summarize(rows, ("room_count", "strategy"))
    localization = _localization_summary(cases)
    paired_comparisons = _paired_comparisons(
        rows,
        baseline="distributed_singles_mwf",
        seed=int(seed),
    )
    payload = {
        "study": "fixed_whole_home_microphone_benchmark_v1",
        "floorplan_count": len(split),
        "case_count": len(cases),
        "plans_per_room_count": int(plans_per_room_count),
        "room_counts": list(selected_counts),
        "strategies": list(WHOLE_HOME_STRATEGIES),
        "single_microphones_by_room_count": single_counts,
        "fixed_arrays_by_room_count": {
            count: ARRAY_COUNTS_BY_ROOM_COUNT[count] for count in selected_counts
        },
        "quality": str(quality),
        "duration_s": float(duration_s),
        "rir_duration_s": float(rir_duration_s),
        "interferer_snr_db_at_oracle_reference": float(interferer_snr_db),
        "background_snr_db_at_oracle_reference": float(background_snr_db),
        "sensor_noise_snr_db_at_oracle_reference": float(sensor_noise_snr_db),
        "resumed_case_count": resumed_case_count,
        "invocation_elapsed_s": round(time.perf_counter() - started, 4),
        "localization": localization,
        "overall": overall,
        "scenario_strategy": scenario_strategy,
        "coverage_strategy": coverage_strategy,
        "room_strategy": room_strategy,
        "paired_comparisons": paired_comparisons,
        "cases": cases,
        "results": rows,
    }
    _write_report(output, payload)
    return payload


def _target_cases(
    model: Any,
    arrays: Sequence[SensorNode],
    *,
    seed: int,
) -> dict[str, TargetPoint]:
    points = sample_target_points(model, points_per_room=3, seed=seed)
    array_rooms = {node.room_id for node in arrays}
    covered_rooms = sorted(array_rooms, key=lambda room: float(model.rooms[room]["area_m2"]), reverse=True)
    covered_room = covered_rooms[0]
    uncovered_rooms = [room for room in model.rooms if room not in array_rooms]
    uncovered_room = max(
        uncovered_rooms,
        key=lambda room: (
            min(_room_hops(model, room, node.room_id) for node in arrays),
            float(model.rooms[room]["area_m2"]),
        ),
    )
    return {
        "array_covered": _point_farthest_from_nodes(points, covered_room, arrays),
        "array_uncovered": _point_farthest_from_nodes(points, uncovered_room, arrays),
    }


def _point_farthest_from_nodes(
    points: Sequence[TargetPoint],
    room_id: str,
    nodes: Sequence[SensorNode],
) -> TargetPoint:
    candidates = [point for point in points if point.room_id == room_id]
    return max(
        candidates,
        key=lambda point: min(
            float(np.linalg.norm(np.asarray(point.position[:2]) - np.asarray(node.position[:2])))
            for node in nodes
        ),
    )


def _same_room_interferer(target: TargetPoint, points: Sequence[TargetPoint]) -> TargetPoint:
    candidates = [point for point in points if point.room_id == target.room_id and point.id != target.id]
    return max(
        candidates,
        key=lambda point: float(
            np.linalg.norm(np.asarray(point.position[:2]) - np.asarray(target.position[:2]))
        ),
    )


def _cross_room_interferer(
    model: Any,
    target: TargetPoint,
    points: Sequence[TargetPoint],
) -> TargetPoint:
    adjacent = [room for room, _ in model.adjacency.get(target.room_id, [])]
    rooms = adjacent or [room for room in model.rooms if room != target.room_id]
    room = max(rooms, key=lambda value: float(model.rooms[value]["area_m2"]))
    return next(point for point in points if point.room_id == room)


def _noise_source(
    model: Any,
    target: TargetPoint,
    points: Sequence[TargetPoint],
) -> TargetPoint:
    room = max(
        model.rooms,
        key=lambda value: (
            _room_hops(model, target.room_id, value),
            float(model.rooms[value]["area_m2"]),
        ),
    )
    return next(point for point in points if point.room_id == room)


def _room_hops(model: Any, first: str, second: str) -> int:
    try:
        return len(model.route(first, second))
    except ValueError:
        return 1_000


def _room_center_node(
    model: Any,
    room_id: str,
    candidates: Sequence[SensorNode],
) -> SensorNode:
    return next(node for node in candidates if node.room_id == room_id)


def _render_mono(
    cache: Path,
    model: Any,
    source: TargetPoint,
    node: SensorNode,
    dry: np.ndarray,
    quality: str,
    rir_duration_s: float,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> np.ndarray:
    return _render_mono_group(
        cache, model, source, [node], dry, quality, rir_duration_s,
        rt_accelerator, rt_precision, rt_cuda_device,
    )


def _render_mono_group(
    cache: Path,
    model: Any,
    source: TargetPoint,
    nodes: Sequence[SensorNode],
    dry: np.ndarray,
    quality: str,
    rir_duration_s: float,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> np.ndarray:
    rirs = _distributed_rirs(
        cache,
        model,
        source,
        nodes,
        quality=quality,
        rir_duration_s=rir_duration_s,
        rt_accelerator=rt_accelerator,
        rt_precision=rt_precision,
        rt_cuda_device=rt_cuda_device,
    )
    return _render_distributed(dry, rirs, dry.size)


def _render_array(
    cache: Path,
    model: Any,
    source: TargetPoint,
    node: SensorNode,
    dry: np.ndarray,
    quality: str,
    rir_duration_s: float,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rir, receiver_model = _local_array_rir(
        cache,
        model,
        source,
        node,
        quality=quality,
        rir_duration_s=rir_duration_s,
        rt_accelerator=rt_accelerator,
        rt_precision=rt_precision,
        rt_cuda_device=rt_cuda_device,
    )
    return _fit_render(dry, rir, dry.size), receiver_model


def _render_array_group(
    cache: Path,
    model: Any,
    source: TargetPoint,
    nodes: Sequence[SensorNode],
    dry: np.ndarray,
    quality: str,
    rir_duration_s: float,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    captures = []
    models = []
    for node in nodes:
        capture, receiver_model = _render_array(
            cache, model, source, node, dry, quality, rir_duration_s,
            rt_accelerator, rt_precision, rt_cuda_device,
        )
        captures.append(capture)
        models.append(receiver_model)
    return captures, models


def _global_source_gains(
    target: np.ndarray,
    interferer: np.ndarray,
    pink: np.ndarray,
    sample_count: int,
    interferer_snr_db: float,
    background_snr_db: float,
    sensor_noise_snr_db: float,
) -> dict[str, float]:
    evaluation = _benchmark_evaluation_slice(sample_count)
    _, interferer_gain = scale_background_to_snr(
        target, interferer, interferer_snr_db, sample_slice=evaluation
    )
    _, pink_gain = scale_background_to_snr(
        target, pink, background_snr_db, sample_slice=evaluation
    )
    target_rms = float(
        np.sqrt(np.mean(np.square(np.asarray(target)[:, evaluation], dtype=np.float64)))
    )
    sensor_std = target_rms / (10.0 ** (float(sensor_noise_snr_db) / 20.0))
    return {
        "interferer": float(interferer_gain),
        "pink": float(pink_gain),
        "sensor_std": float(sensor_std),
    }


def _mix_with_global_gains(
    target: np.ndarray,
    interferer: np.ndarray,
    pink: np.ndarray,
    gains: Mapping[str, float],
    seed: int,
) -> dict[str, np.ndarray]:
    target_values = np.asarray(target, dtype=np.float32)
    interference = np.asarray(interferer, dtype=np.float32) * float(gains["interferer"])
    background = np.asarray(pink, dtype=np.float32) * float(gains["pink"])
    rng = np.random.default_rng(int(seed))
    sensor = rng.normal(
        0.0,
        float(gains["sensor_std"]),
        size=target_values.shape,
    ).astype(np.float32)
    noise = np.asarray(interference + background + sensor, dtype=np.float32)
    return {
        "target": target_values,
        "noise": noise,
        "mixture": np.asarray(target_values + noise, dtype=np.float32),
        "sensor": sensor,
    }


def _tdoa_localize(
    model: Any,
    nodes: Sequence[SensorNode],
    components: Mapping[str, np.ndarray],
    target: TargetPoint,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    calibration = _calibration_slice(components["target"].shape[1])
    observation = components["target"][:, calibration] + components["sensor"][:, calibration]
    reference = int(np.argmax(np.sqrt(np.mean(np.square(observation), axis=1))))
    delays, confidence = estimate_gcc_phat_delays(
        observation,
        FS,
        max_delay_s=0.2,
        reference_channel=reference,
    )
    measurements = [
        TOAMeasurement(node.id, float(delays[index]), float(confidence[index]))
        for index, node in enumerate(nodes)
    ]
    position, room, _ = localize_tdoa(
        model,
        nodes,
        measurements,
        localization_grid(model, spacing_m=0.45),
    )
    return delays, position, room, {
        "localization_error_m": round(
            float(np.linalg.norm(np.asarray(position) - np.asarray(target.position[:2]))),
            4,
        ),
        "room_correct": bool(room == target.room_id),
    }


def _target_activity(components: Mapping[str, np.ndarray]) -> np.ndarray:
    calibration = _calibration_slice(components["target"].shape[1])
    values = components["target"][:, calibration] + components["sensor"][:, calibration]
    return np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=1))


def _select_fixed_node(
    model: Any,
    nodes: Sequence[SensorNode],
    estimated_position: Sequence[float],
    estimated_room: str,
) -> int:
    ranking = []
    for index, node in enumerate(nodes):
        try:
            _, distance, hops = model.propagation(
                (*estimated_position, 1.5), estimated_room, node.position, node.room_id
            )
        except ValueError:
            distance, hops = 1e6, 1_000
        ranking.append(((0 if node.room_id == estimated_room else 1, hops, distance, node.id), index))
    return min(ranking)[1]


def _take_channels(
    components: Mapping[str, np.ndarray],
    indices: Sequence[int] | range,
) -> dict[str, np.ndarray]:
    selected = list(indices)
    return {name: np.asarray(value)[selected] for name, value in components.items()}


def _concatenate_components(
    *components: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: np.vstack([np.asarray(component[name]) for component in components]).astype(np.float32)
        for name in ("target", "noise", "mixture", "sensor")
    }


def _measured_arrivals(components: Mapping[str, np.ndarray]) -> np.ndarray:
    calibration = _calibration_slice(components["target"].shape[1])
    observation = components["target"][:, calibration] + components["sensor"][:, calibration]
    reference = int(np.argmax(np.sqrt(np.mean(np.square(observation), axis=1))))
    delays, _ = estimate_gcc_phat_delays(
        observation,
        FS,
        max_delay_s=0.2,
        reference_channel=reference,
    )
    return delays


def _strategy_rows(
    *,
    strategy: str,
    deployment: str,
    processing_architecture: str,
    processing_pipeline: str,
    components: Mapping[str, np.ndarray],
    arrivals: np.ndarray,
    target_dry: np.ndarray,
    physical_devices: int,
    physical_channels: int,
    common: Mapping[str, Any],
    captured: dict[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    local_capture: dict[str, np.ndarray] | None = {} if captured is not None else None
    generated = _run_architecture(
        architecture=processing_architecture,
        pipeline_names=(processing_pipeline,),
        components=components,
        arrival_times_s=np.asarray(arrivals, dtype=np.float64),
        target_dry=target_dry,
        physical_microphones=int(physical_channels),
        common=common,
        captured_audio=local_capture,
    )
    for row in generated:
        row["strategy"] = strategy
        row["deployment"] = deployment
        row["algorithm"] = processing_pipeline.rsplit("_", 1)[-1]
        row["physical_devices"] = int(physical_devices)
        row["physical_channels"] = int(physical_channels)
        row["processing_architecture"] = processing_architecture
        row["processing_pipeline"] = processing_pipeline
        row["architecture"] = deployment
        row["pipeline"] = strategy
    if captured is not None and local_capture:
        captured[strategy] = next(iter(local_capture.values()))
    return generated


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = (
        "input_snr_db",
        "output_snr_db",
        "snr_improvement_db",
        "si_sdr_improvement_db",
        "output_pesq",
        "pesq_improvement",
        "output_stoi",
        "output_dry_pesq",
        "output_dry_stoi",
        "rtf",
    )
    summaries = []
    for group, items in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = {key: value for key, value in zip(keys, group)}
        result["cases"] = len(items)
        result["room_accuracy"] = round(
            float(np.mean([bool(item["room_correct"]) for item in items])), 5
        )
        result["positive_si_sdr_rate"] = round(
            float(np.mean([float(item["si_sdr_improvement_db"]) > 0.0 for item in items])), 5
        )
        result["mean_physical_devices"] = round(
            float(np.mean([int(item["physical_devices"]) for item in items])), 3
        )
        result["mean_physical_channels"] = round(
            float(np.mean([int(item["physical_channels"]) for item in items])), 3
        )
        for metric in metrics:
            values = np.asarray(
                [float(item[metric]) for item in items if item.get(metric) is not None],
                dtype=np.float64,
            )
            result[f"mean_{metric}"] = None if not values.size else round(float(np.mean(values)), 5)
            result[f"p10_{metric}"] = None if not values.size else round(float(np.quantile(values, 0.1)), 5)
        summaries.append(result)
    return summaries


def _localization_summary(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {"all": list(cases)}
    for target_case in ("array_covered", "array_uncovered"):
        groups[target_case] = [case for case in cases if case["target_case"] == target_case]
    output = []
    for label, items in groups.items():
        errors = np.asarray([float(item["localization_error_m"]) for item in items])
        output.append(
            {
                "target_case": label,
                "cases": len(items),
                "room_accuracy": round(float(np.mean([bool(item["room_correct"]) for item in items])), 5),
                "median_error_m": round(float(np.median(errors)), 5),
                "p90_error_m": round(float(np.quantile(errors, 0.9)), 5),
            }
        )
    return output


def _paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    seed: int,
    bootstrap_samples: int = 5_000,
) -> list[dict[str, Any]]:
    case_rows: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["floorplan_idx"]), str(row["target_case"]), str(row["scenario"]))
        case_rows.setdefault(key, {})[str(row["strategy"])] = row
    metrics = ("output_pesq", "output_stoi", "si_sdr_improvement_db")
    rng = np.random.default_rng(int(seed))
    comparisons = []
    for strategy in WHOLE_HOME_STRATEGIES:
        if strategy == baseline:
            continue
        matched = [
            (key, values[strategy], values[baseline])
            for key, values in case_rows.items()
            if strategy in values and baseline in values
        ]
        if not matched:
            continue
        row: dict[str, Any] = {
            "baseline": baseline,
            "strategy": strategy,
            "cases": len(matched),
            "floorplans": len({key[0] for key, _, _ in matched}),
        }
        floorplans = sorted({key[0] for key, _, _ in matched})
        for metric in metrics:
            deltas = np.asarray(
                [float(candidate[metric]) - float(reference[metric]) for _, candidate, reference in matched],
                dtype=np.float64,
            )
            per_floorplan = {
                floorplan: float(
                    np.mean(
                        [
                            float(candidate[metric]) - float(reference[metric])
                            for key, candidate, reference in matched
                            if key[0] == floorplan
                        ]
                    )
                )
                for floorplan in floorplans
            }
            bootstrap = np.empty(max(1, int(bootstrap_samples)), dtype=np.float64)
            for index in range(bootstrap.size):
                sampled = rng.choice(floorplans, len(floorplans), replace=True)
                bootstrap[index] = float(
                    np.mean([per_floorplan[int(floorplan)] for floorplan in sampled])
                )
            row[f"mean_delta_{metric}"] = round(float(np.mean(deltas)), 5)
            row[f"ci95_low_delta_{metric}"] = round(float(np.quantile(bootstrap, 0.025)), 5)
            row[f"ci95_high_delta_{metric}"] = round(float(np.quantile(bootstrap, 0.975)), 5)
            row[f"win_rate_{metric}"] = round(float(np.mean(deltas > 0.0)), 5)
        comparisons.append(row)
    return comparisons


def _stable_label_seed(label: str) -> int:
    return sum((index + 1) * ord(value) for index, value in enumerate(str(label)))


def _write_checkpoint(
    output: Path,
    signature: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> None:
    (output / "checkpoint.json").write_text(
        json.dumps(
            {"signature": dict(signature), "results": list(rows), "cases": list(cases)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_checkpoint(
    output: Path,
    signature: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = output / "checkpoint.json"
    if not path.is_file():
        return [], []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []
    if payload.get("signature") != dict(signature):
        return [], []
    return list(payload.get("results", [])), list(payload.get("cases", []))


def _write_report(output: Path, payload: Mapping[str, Any]) -> None:
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / "details.csv", payload["results"])
    _write_csv(output / "overall.csv", payload["overall"])
    _write_csv(output / "scenario-strategy.csv", payload["scenario_strategy"])
    _write_csv(output / "coverage-strategy.csv", payload["coverage_strategy"])
    _write_csv(output / "room-strategy.csv", payload["room_strategy"])
    _write_csv(output / "localization.csv", payload["localization"])
    _write_csv(output / "paired-comparisons.csv", payload["paired_comparisons"])
    lines = [
        "# Fixed Whole-Home Microphone Benchmark",
        "",
        f"- FloorPlans: `{payload['floorplan_count']}`",
        f"- Cases: `{payload['case_count']}`",
        f"- Room counts: `{', '.join(str(value) for value in payload['room_counts'])}`",
        f"- RIR quality: `{payload['quality']}`",
        f"- Invocation elapsed: `{payload['invocation_elapsed_s']:.2f} s` "
        f"(`{payload['resumed_case_count']}` cases resumed from checkpoint)",
        "- Deployment is fixed before target sampling; every source keeps one globally calibrated emission gain.",
        "",
        "## Overall",
        "",
        "| Strategy | Devices | Channels | SNR in | SNR change | SI-SDR change | PESQ | P10 PESQ | STOI | Dry PESQ | Positive SI-SDR | RTF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["overall"]:
        lines.append(
            f"| {row['strategy']} | {_format_metric(row['mean_physical_devices'], digits=1)} | "
            f"{_format_metric(row['mean_physical_channels'], digits=1)} | "
            f"{_format_metric(row['mean_input_snr_db'])} | "
            f"{_format_metric(row['mean_snr_improvement_db'])} | "
            f"{_format_metric(row['mean_si_sdr_improvement_db'])} | "
            f"{_format_metric(row['mean_output_pesq'])} | "
            f"{_format_metric(row['p10_output_pesq'])} | "
            f"{_format_metric(row['mean_output_stoi'])} | "
            f"{_format_metric(row['mean_output_dry_pesq'])} | "
            f"{100.0 * float(row['positive_si_sdr_rate']):.1f}% | "
            f"{_format_metric(row['mean_rtf'], digits=4)} |"
        )
    lines.extend(
        [
            "",
            "## TDOA Localization",
            "",
            "| Target coverage | Cases | Room accuracy | Median error | P90 error |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["localization"]:
        lines.append(
            f"| {row['target_case']} | {row['cases']} | {100.0 * row['room_accuracy']:.1f}% | "
            f"{row['median_error_m']:.3f} m | {row['p90_error_m']:.3f} m |"
        )
    lines.extend(
        [
            "",
            "## Paired Against Distributed Singles MWF",
            "",
            "Confidence intervals resample complete FloorPlans, preserving the four correlated cases within each layout.",
            "",
            "| Strategy | PESQ delta [95% CI] | PESQ wins | STOI delta [95% CI] | STOI wins | SI-SDR delta [95% CI] | SI-SDR wins |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    highlighted = {
        "tdoa_routed_fixed_array_mwf",
        "equal_channel_hybrid_mwf",
        "coverage_hybrid_selected_mwf",
        "coverage_hybrid_all_mwf",
        "oracle_target_room_array_mwf",
    }
    for row in payload["paired_comparisons"]:
        if row["strategy"] not in highlighted:
            continue
        lines.append(
            f"| {row['strategy']} | {row['mean_delta_output_pesq']:.3f} "
            f"[{row['ci95_low_delta_output_pesq']:.3f}, {row['ci95_high_delta_output_pesq']:.3f}] | "
            f"{100.0 * row['win_rate_output_pesq']:.1f}% | "
            f"{row['mean_delta_output_stoi']:.3f} "
            f"[{row['ci95_low_delta_output_stoi']:.3f}, {row['ci95_high_delta_output_stoi']:.3f}] | "
            f"{100.0 * row['win_rate_output_stoi']:.1f}% | "
            f"{row['mean_delta_si_sdr_improvement_db']:.3f} dB "
            f"[{row['ci95_low_delta_si_sdr_improvement_db']:.3f}, "
            f"{row['ci95_high_delta_si_sdr_improvement_db']:.3f}] | "
            f"{100.0 * row['win_rate_si_sdr_improvement_db']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `oracle_target_room_*` moves one reference device into the true target room and is an upper bound, not a fixed whole-home deployment.",
            "- `tdoa_routed_*` selects among devices that were fixed before the target was sampled.",
            "- `equal_channel_hybrid_mwf` spends the same channel budget as distributed singles: one four-channel array plus `N-4` singles.",
            "- `coverage_hybrid_*` keeps all `N` singles and adds one array for 4/6/8-room homes or two arrays for 10/12-room homes.",
            "- Wet-reference PESQ measures enhancement without penalizing the room response. Dry PESQ additionally exposes coloration and reverberation.",
            "- Target-active and target-silent calibration segments are oracle VAD baselines; no clean evaluation waveform is used to estimate weights.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
