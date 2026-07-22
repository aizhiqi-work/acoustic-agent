from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from acoustic_agent.floorplan_resource import FloorplanResource

from .distributed import (
    AcousticMeasurementGenerator,
    TOAMeasurement,
    candidate_nodes,
    load_model,
    localization_grid,
    localize_doa,
    localize_hybrid,
    localize_tdoa,
    place_nodes,
    sample_target_points,
    tune_risk_quantile,
)
from .stratified import (
    ADEQUACY_MEDIAN_M,
    ADEQUACY_P90_M,
    ADEQUACY_ROOM_ACCURACY,
    FloorplanProfile,
    _cluster_bootstrap,
    build_stratified_split,
    population_summary,
    scan_floorplan_population,
)


DEFAULT_ROOM_COUNTS = tuple(range(2, 15))
ARRAY_CHANNELS = 4
MIN_SINGLE_NODES = 3
MIN_ARRAY_NODES = 2
CONFIRMATORY_CALIBRATION_FLOORPLANS = 5
CONFIRMATORY_VALIDATION_FLOORPLANS = 10


def rebuild_static_scaling_outputs(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    selected_counts = {
        (int(row["room_count"]), str(row["sensor_kind"])): int(row["recommended_nodes"])
        for row in payload["recommendations"]
    }
    recommended_rows = [
        row
        for row in payload["results"]
        if int(row["node_count"])
        == selected_counts[(int(row["room_count"]), str(row["sensor_kind"]))]
    ]
    payload["recommended_by_relative_area"] = _group_metrics(
        recommended_rows,
        ("sensor_kind", "relative_area_bin"),
    )
    payload["recommended_by_absolute_area"] = _group_metrics(
        recommended_rows,
        ("sensor_kind", "absolute_area_bin"),
    )
    _write_outputs(output, payload)
    return payload


def filter_static_scaling_outputs(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    room_counts: Sequence[int],
    validation_per_count: int = 5,
    seed: int = 20260722,
) -> dict[str, Any]:
    source = Path(source_dir)
    output = Path(output_dir)
    payload = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    requested_counts = tuple(int(value) for value in room_counts)
    selected_validation: list[dict[str, Any]] = []
    for room_count in requested_counts:
        candidates = [
            row
            for row in payload["split"]
            if row["split"] == "validation" and int(row["room_count"]) == room_count
        ]
        if len(candidates) < int(validation_per_count):
            raise ValueError(
                f"room_count={room_count} has only {len(candidates)} source validation FloorPlans; "
                f"{validation_per_count} are required"
            )
        selected_validation.extend(
            _area_stratified_rows(
                candidates,
                int(validation_per_count),
                seed=seed + room_count * 101,
            )
        )
    selected_ids = {int(row["index"]) for row in selected_validation}
    selected_calibration = [
        row
        for row in payload["split"]
        if row["split"] == "calibration" and int(row["room_count"]) in requested_counts
    ]
    evidence_tier = f"{int(validation_per_count)}-plan"
    split_rows = [
        {
            **row,
            "evidence_tier": evidence_tier,
            "stratum_validation_floorplans": int(validation_per_count),
        }
        for row in (*selected_calibration, *selected_validation)
    ]
    results = [
        {**row, "evidence_tier": evidence_tier}
        for row in payload["results"]
        if int(row["floorplan_idx"]) in selected_ids
    ]
    placements = [
        {**row, "evidence_tier": evidence_tier}
        for row in payload["placements"]
        if int(row["floorplan_idx"]) in selected_ids
    ]
    runtimes = [
        row for row in payload["runtimes"] if int(row["floorplan_idx"]) in selected_ids
    ]
    recommendations = _recommend(results, seed=seed)
    selected_counts = {
        (int(row["room_count"]), str(row["sensor_kind"])): int(row["recommended_nodes"])
        for row in recommendations
    }
    recommended_rows = [
        row
        for row in results
        if int(row["node_count"])
        == selected_counts[(int(row["room_count"]), str(row["sensor_kind"]))]
    ]
    filtered = {
        **payload,
        "study": "static_connected_floorplan_sensor_scaling_filtered",
        "source_study": payload["study"],
        "source_output_dir": str(source),
        "room_counts": list(requested_counts),
        "validation_per_room_count": int(validation_per_count),
        "validation_indices": sorted(selected_ids),
        "split": sorted(
            split_rows,
            key=lambda row: (str(row["split"]), int(row["room_count"]), float(row["area_m2"])),
        ),
        "results": results,
        "placements": placements,
        "runtimes": runtimes,
        "metrics": _group_metrics(results, ("room_count", "sensor_kind", "node_count")),
        "recommendations": recommendations,
        "recommended_by_relative_area": _group_metrics(
            recommended_rows,
            ("sensor_kind", "relative_area_bin"),
        ),
        "recommended_by_absolute_area": _group_metrics(
            recommended_rows,
            ("sensor_kind", "absolute_area_bin"),
        ),
        "source_elapsed_s": payload["elapsed_s"],
        "elapsed_s": round(float(sum(float(row["elapsed_s"]) for row in runtimes)), 4),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_outputs(output, filtered)
    return filtered


def _area_stratified_rows(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["area_m2"]), int(row["index"])))
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    edges = np.linspace(0, len(ordered), count + 1)
    for index in range(count):
        lower = int(np.floor(edges[index]))
        upper = max(int(np.floor(edges[index + 1])), lower + 1)
        chosen = int(rng.integers(lower, min(upper, len(ordered))))
        selected.append(dict(ordered[chosen]))
    return selected


def build_adaptive_split(
    profiles: Sequence[FloorplanProfile],
    *,
    room_counts: Sequence[int] = DEFAULT_ROOM_COUNTS,
    calibration_per_count: int = 5,
    validation_per_count: int = 10,
    seed: int = 20260722,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for room_count in room_counts:
        population_size = sum(
            profile.room_count == int(room_count) and profile.eligible for profile in profiles
        )
        if population_size < 3:
            raise ValueError(
                f"room_count={room_count} has {population_size} eligible FloorPlans; at least 3 are required"
            )
        calibration, validation = _adaptive_sample_counts(
            population_size,
            calibration_per_count=calibration_per_count,
            validation_per_count=validation_per_count,
        )
        stratum = build_stratified_split(
            profiles,
            room_counts=(int(room_count),),
            calibration_per_count=calibration,
            validation_per_count=validation,
            seed=seed,
        )
        evidence_tier = (
            "confirmatory"
            if calibration >= CONFIRMATORY_CALIBRATION_FLOORPLANS
            and validation >= CONFIRMATORY_VALIDATION_FLOORPLANS
            else "exploratory"
        )
        rows.extend(
            {
                **row,
                "evidence_tier": evidence_tier,
                "stratum_calibration_floorplans": calibration,
                "stratum_validation_floorplans": validation,
            }
            for row in stratum
        )
    return sorted(rows, key=lambda row: (str(row["split"]), int(row["room_count"]), float(row["area_m2"])))


def run_static_scaling_study(
    output_dir: str | Path,
    *,
    room_counts: Sequence[int] = DEFAULT_ROOM_COUNTS,
    calibration_per_count: int = 5,
    validation_per_count: int = 10,
    quality: str = "preview",
    points_per_room: int = 1,
    positions_per_room: int = 2,
    max_single_nodes: int = 8,
    max_array_nodes: int = 8,
    seed: int = 20260722,
    risk_quantile: float | None = None,
    rt_accelerator: str = "numba",
    rt_precision: str = "float64",
    rt_cuda_device: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resource = FloorplanResource()
    profiles = scan_floorplan_population(resource)
    split_rows = build_adaptive_split(
        profiles,
        room_counts=room_counts,
        calibration_per_count=calibration_per_count,
        validation_per_count=validation_per_count,
        seed=seed,
    )
    calibration_indices = [int(row["index"]) for row in split_rows if row["split"] == "calibration"]
    validation_rows = [row for row in split_rows if row["split"] == "validation"]
    split_by_index = {int(row["index"]): row for row in split_rows}
    profile_by_index = {profile.index: profile for profile in profiles}

    if risk_quantile is None:
        selected_risk_quantile, tuning_rows = tune_risk_quantile(calibration_indices, resource=resource)
        risk_quantile_source = "calibrated"
    else:
        selected_risk_quantile = float(risk_quantile)
        if not 0.0 <= selected_risk_quantile <= 1.0:
            raise ValueError("risk_quantile must be between 0 and 1")
        tuning_rows = []
        risk_quantile_source = "provided"
    generator = AcousticMeasurementGenerator(
        output,
        quality=quality,
        rt_accelerator=rt_accelerator,
        rt_precision=rt_precision,
        rt_cuda_device=rt_cuda_device,
    )
    results: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_plans = len(validation_rows)

    for plan_number, split in enumerate(validation_rows, start=1):
        plan_started = time.perf_counter()
        floorplan_index = int(split["index"])
        profile = profile_by_index[floorplan_index]
        model = load_model(floorplan_index, resource)
        grid = localization_grid(model, spacing_m=0.55)
        targets = sample_target_points(
            model,
            points_per_room=max(1, int(points_per_room)),
            seed=seed + floorplan_index * 17,
            spacing_m=0.52,
        )
        candidates = candidate_nodes(model, positions_per_room=max(1, int(positions_per_room)))
        single_count = min(max(int(max_single_nodes), MIN_SINGLE_NODES), len(candidates))
        array_count = min(max(int(max_array_nodes), MIN_ARRAY_NODES), len(candidates))
        single_nodes = place_nodes(
            model,
            single_count,
            mode="single",
            risk_quantile=selected_risk_quantile,
            candidates=candidates,
        )
        array_nodes = place_nodes(
            model,
            array_count,
            mode="array",
            risk_quantile=selected_risk_quantile,
            candidates=candidates,
        )
        placements.extend(
            _placement_rows(model.index, profile, split, "single", single_nodes)
        )
        placements.extend(
            _placement_rows(model.index, profile, split, "array_4ch", array_nodes)
        )

        for target in targets:
            truth = np.asarray(target.position[:2], dtype=float)
            single_measurements = [generator.single(model, target, node) for node in single_nodes]
            array_measurements = [
                generator.array(model, target, node, channels=ARRAY_CHANNELS) for node in array_nodes
            ]
            for node_count in range(MIN_SINGLE_NODES, len(single_nodes) + 1):
                nodes = single_nodes[:node_count]
                estimate, estimated_room, _ = localize_tdoa(
                    model,
                    nodes,
                    single_measurements[:node_count],
                    grid,
                )
                results.append(
                    _result_row(
                        profile,
                        split,
                        target,
                        truth,
                        estimate,
                        estimated_room,
                        sensor_kind="single",
                        algorithm="tdoa",
                        node_count=node_count,
                        physical_microphones=node_count,
                        local_sensor=any(node.room_id == target.room_id for node in nodes),
                    )
                )
            for node_count in range(1, len(array_nodes) + 1):
                nodes = array_nodes[:node_count]
                doa = array_measurements[:node_count]
                if node_count == 1:
                    estimate, estimated_room, _ = localize_doa(model, nodes, doa, grid)
                    algorithm = "doa"
                else:
                    toa = [
                        TOAMeasurement(measurement.node_id, measurement.arrival_s, measurement.confidence)
                        for measurement in doa
                    ]
                    estimate, estimated_room, _ = localize_hybrid(
                        model,
                        nodes,
                        doa,
                        nodes,
                        toa,
                        grid,
                    )
                    algorithm = "doa+inter_array_tdoa"
                results.append(
                    _result_row(
                        profile,
                        split,
                        target,
                        truth,
                        estimate,
                        estimated_room,
                        sensor_kind="array_4ch",
                        algorithm=algorithm,
                        node_count=node_count,
                        physical_microphones=node_count * ARRAY_CHANNELS,
                        local_sensor=any(node.room_id == target.room_id for node in nodes),
                    )
                )

        elapsed = time.perf_counter() - plan_started
        runtimes.append(
            {
                "floorplan_idx": floorplan_index,
                "room_count": profile.room_count,
                "area_m2": round(profile.area_m2, 4),
                "targets": len(targets),
                "single_nodes_simulated": len(single_nodes),
                "array_nodes_simulated": len(array_nodes),
                "elapsed_s": round(elapsed, 4),
            }
        )
        print(
            f"[{plan_number:03d}/{total_plans:03d}] FloorPlan {floorplan_index}: "
            f"{profile.room_count} rooms, {profile.area_m2:.1f} m2, {len(targets)} targets, {elapsed:.1f}s",
            flush=True,
        )
        next_room_count = (
            int(validation_rows[plan_number]["room_count"]) if plan_number < total_plans else None
        )
        if next_room_count != profile.room_count:
            completed = sum(int(row["room_count"]) == profile.room_count for row in validation_rows[:plan_number])
            print(
                f"[room-count complete] {profile.room_count} rooms: {completed} FloorPlans",
                flush=True,
            )

    metrics = _group_metrics(results, ("room_count", "sensor_kind", "node_count"))
    recommendations = _recommend(results, seed=seed)
    selected_counts = {
        (int(row["room_count"]), str(row["sensor_kind"])): int(row["recommended_nodes"])
        for row in recommendations
    }
    recommended_rows = [
        row
        for row in results
        if int(row["node_count"])
        == selected_counts[(int(row["room_count"]), str(row["sensor_kind"]))]
    ]
    recommended_by_relative_area = _group_metrics(
        recommended_rows,
        ("sensor_kind", "relative_area_bin"),
    )
    recommended_by_absolute_area = _group_metrics(
        recommended_rows,
        ("sensor_kind", "absolute_area_bin"),
    )
    payload = {
        "study": "static_connected_floorplan_sensor_scaling",
        "task": {"outputs": ["global_x_m", "global_y_m", "room_id"], "motion": False},
        "database_records": len(profiles),
        "room_counts": [int(value) for value in room_counts],
        "calibration_per_room_count": int(calibration_per_count),
        "validation_per_room_count": int(validation_per_count),
        "quality": quality,
        "rt_accelerator": rt_accelerator,
        "rt_precision": rt_precision,
        "rt_cuda_device": int(rt_cuda_device),
        "points_per_room": int(points_per_room),
        "positions_per_room": int(positions_per_room),
        "max_single_nodes": int(max_single_nodes),
        "max_array_nodes": int(max_array_nodes),
        "array_channels": ARRAY_CHANNELS,
        "selected_risk_quantile": selected_risk_quantile,
        "risk_quantile_source": risk_quantile_source,
        "adequacy_thresholds": {
            "median_error_m": ADEQUACY_MEDIAN_M,
            "p90_error_m": ADEQUACY_P90_M,
            "room_accuracy": ADEQUACY_ROOM_ACCURACY,
        },
        "population": population_summary(profiles),
        "split": split_rows,
        "tuning": tuning_rows,
        "results": results,
        "placements": placements,
        "runtimes": runtimes,
        "metrics": metrics,
        "recommendations": recommendations,
        "recommended_by_relative_area": recommended_by_relative_area,
        "recommended_by_absolute_area": recommended_by_absolute_area,
        "elapsed_s": round(time.perf_counter() - started, 4),
    }
    _write_outputs(output, payload)
    return payload


def _adaptive_sample_counts(
    population_size: int,
    *,
    calibration_per_count: int,
    validation_per_count: int,
) -> tuple[int, int]:
    requested = int(calibration_per_count) + int(validation_per_count)
    if population_size >= requested:
        return int(calibration_per_count), int(validation_per_count)
    calibration = max(1, min(int(calibration_per_count), population_size // 3))
    validation = min(int(validation_per_count), population_size - calibration)
    return calibration, validation


def _placement_rows(
    floorplan_index: int,
    profile: FloorplanProfile,
    split: Mapping[str, Any],
    sensor_kind: str,
    nodes: Sequence[Any],
) -> list[dict[str, Any]]:
    return [
        {
            "floorplan_idx": floorplan_index,
            "room_count": profile.room_count,
            "area_m2": round(profile.area_m2, 4),
            "evidence_tier": split["evidence_tier"],
            "sensor_kind": sensor_kind,
            "order": order,
            "node_id": node.id,
            "room_id": node.room_id,
            "x_m": round(node.position[0], 5),
            "y_m": round(node.position[1], 5),
            "z_m": round(node.position[2], 5),
        }
        for order, node in enumerate(nodes, start=1)
    ]


def _result_row(
    profile: FloorplanProfile,
    split: Mapping[str, Any],
    target: Any,
    truth: np.ndarray,
    estimate: np.ndarray,
    estimated_room: str,
    *,
    sensor_kind: str,
    algorithm: str,
    node_count: int,
    physical_microphones: int,
    local_sensor: bool,
) -> dict[str, Any]:
    return {
        "split": "validation",
        "floorplan_idx": profile.index,
        "room_count": profile.room_count,
        "area_m2": round(profile.area_m2, 4),
        "relative_area_bin": split["relative_area_bin"],
        "absolute_area_bin": split["absolute_area_bin"],
        "evidence_tier": split["evidence_tier"],
        "target_id": target.id,
        "target_room": target.room_id,
        "sensor_kind": sensor_kind,
        "algorithm": algorithm,
        "node_count": node_count,
        "channels_per_node": 1 if sensor_kind == "single" else ARRAY_CHANNELS,
        "physical_microphones": physical_microphones,
        "configuration": f"{sensor_kind}_{node_count}",
        "local_sensor": bool(local_sensor),
        "true_x_m": round(float(truth[0]), 5),
        "true_y_m": round(float(truth[1]), 5),
        "estimated_x_m": round(float(estimate[0]), 5),
        "estimated_y_m": round(float(estimate[1]), 5),
        "position_error_m": round(float(np.linalg.norm(estimate - truth)), 5),
        "estimated_room": estimated_room,
        "room_correct": bool(estimated_room == target.room_id),
    }


def _group_metrics(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for values, subset in grouped.items():
        errors = np.asarray([float(row["position_error_m"]) for row in subset], dtype=float)
        cross = [row for row in subset if not bool(row["local_sensor"])]
        node_counts = [int(row["node_count"]) for row in subset]
        microphone_counts = [int(row["physical_microphones"]) for row in subset]
        summary = {key: value for key, value in zip(keys, values)}
        summary.update(
            {
                "node_count_min": min(node_counts),
                "node_count_max": max(node_counts),
                "physical_microphones": microphone_counts[0]
                if len(set(microphone_counts)) == 1
                else None,
                "physical_microphones_min": min(microphone_counts),
                "physical_microphones_max": max(microphone_counts),
                "floorplans": len({int(row["floorplan_idx"]) for row in subset}),
                "cases": len(subset),
                "median_error_m": round(float(np.median(errors)), 4),
                "p90_error_m": round(float(np.percentile(errors, 90)), 4),
                "room_accuracy": round(float(np.mean([bool(row["room_correct"]) for row in subset])), 4),
                "cross_room_cases": len(cross),
                "cross_room_median_error_m": None
                if not cross
                else round(float(np.median([float(row["position_error_m"]) for row in cross])), 4),
                "cross_room_room_accuracy": None
                if not cross
                else round(float(np.mean([bool(row["room_correct"]) for row in cross])), 4),
            }
        )
        summaries.append(summary)
    return sorted(summaries, key=lambda row: tuple(str(row[key]) for key in keys))


def _recommend(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    room_counts = sorted({int(row["room_count"]) for row in rows})
    for room_count in room_counts:
        for sensor_kind in ("single", "array_4ch"):
            subset = [
                row
                for row in rows
                if int(row["room_count"]) == room_count and row["sensor_kind"] == sensor_kind
            ]
            summaries = _group_metrics(subset, ("node_count",))
            minimum = MIN_SINGLE_NODES if sensor_kind == "single" else MIN_ARRAY_NODES
            admissible = [row for row in summaries if int(row["node_count"]) >= minimum]
            evaluated = []
            for candidate in admissible:
                selected = [
                    row for row in subset if int(row["node_count"]) == int(candidate["node_count"])
                ]
                confidence = _cluster_bootstrap(
                    selected,
                    seed=(
                        seed
                        + room_count * 313
                        + int(candidate["node_count"]) * 17
                        + (0 if sensor_kind == "single" else 100_000)
                    ),
                )
                observed_meets = (
                    float(candidate["median_error_m"]) <= ADEQUACY_MEDIAN_M
                    and float(candidate["p90_error_m"]) <= ADEQUACY_P90_M
                    and float(candidate["room_accuracy"]) >= ADEQUACY_ROOM_ACCURACY
                )
                ci95_supports = (
                    float(confidence["median_error_ci95_high_m"]) <= ADEQUACY_MEDIAN_M
                    and float(confidence["p90_error_ci95_high_m"]) <= ADEQUACY_P90_M
                    and float(confidence["room_accuracy_ci95_low"]) >= ADEQUACY_ROOM_ACCURACY
                )
                evaluated.append(
                    {
                        **candidate,
                        **confidence,
                        "observed_meets_thresholds": observed_meets,
                        "ci95_supports_thresholds": ci95_supports,
                    }
                )
            confidence_supported = [row for row in evaluated if bool(row["ci95_supports_thresholds"])]
            point_supported = [row for row in evaluated if bool(row["observed_meets_thresholds"])]
            if confidence_supported:
                chosen = min(confidence_supported, key=lambda row: int(row["node_count"]))
                selection_basis = "ci95"
            elif point_supported:
                chosen = min(point_supported, key=lambda row: int(row["node_count"]))
                selection_basis = "point_estimate_only"
            else:
                chosen = min(
                    evaluated,
                    key=lambda row: (
                        float(row["median_error_m"])
                        + 0.5 * float(row["p90_error_m"])
                        + 2.0 * (1.0 - float(row["room_accuracy"])),
                        int(row["node_count"]),
                    ),
                )
                selection_basis = "best_tested_not_adequate"
            selected = [row for row in subset if int(row["node_count"]) == int(chosen["node_count"])]
            recommendations.append(
                {
                    "room_count": room_count,
                    "evidence_tier": selected[0]["evidence_tier"],
                    "sensor_kind": sensor_kind,
                    "recommended_nodes": int(chosen["node_count"]),
                    "physical_microphones": int(chosen["physical_microphones"]),
                    "meets_thresholds": bool(chosen["observed_meets_thresholds"]),
                    "ci95_supports_thresholds": bool(chosen["ci95_supports_thresholds"]),
                    "selection_basis": selection_basis,
                    **{
                        key: chosen[key]
                        for key in (
                            "floorplans",
                            "cases",
                            "median_error_m",
                            "p90_error_m",
                            "room_accuracy",
                            "cross_room_cases",
                            "cross_room_median_error_m",
                            "cross_room_room_accuracy",
                        )
                    },
                    **{
                        key: chosen[key]
                        for key in (
                            "median_error_ci95_low_m",
                            "median_error_ci95_high_m",
                            "p90_error_ci95_low_m",
                            "p90_error_ci95_high_m",
                            "room_accuracy_ci95_low",
                            "room_accuracy_ci95_high",
                        )
                    },
                }
            )
    return recommendations


def _write_outputs(output: Path, payload: Mapping[str, Any]) -> None:
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, key in (
        ("population.csv", "population"),
        ("split.csv", "split"),
        ("results.csv", "results"),
        ("placements.csv", "placements"),
        ("runtimes.csv", "runtimes"),
        ("metrics.csv", "metrics"),
        ("recommendations.csv", "recommendations"),
        ("recommended_by_relative_area.csv", "recommended_by_relative_area"),
        ("recommended_by_absolute_area.csv", "recommended_by_absolute_area"),
    ):
        _write_csv(output / name, payload[key])

    room_count_text = ", ".join(str(value) for value in payload["room_counts"])
    validation_count = payload.get("validation_per_room_count")
    sample_limit = (
        f"This report uses {validation_count} validation FloorPlans per listed room-count stratum. "
        if validation_count is not None
        else "Validation counts vary by room-count stratum. "
    )
    lines = [
        "# Static connected-FloorPlan localization scaling study",
        "",
        "## Protocol",
        "",
        "- Task output: global `(x, y)` and `room_id`; motion is not modeled.",
        "- Only fully connected FloorPlans with valid room geometry are eligible.",
        f"- Room-count strata: `{room_count_text}`; validation FloorPlans per stratum: "
        f"`{validation_count if validation_count is not None else 'adaptive'}`.",
        f"- Accuracy gate: median <= {ADEQUACY_MEDIAN_M:.1f} m, P90 <= {ADEQUACY_P90_M:.1f} m, "
        f"room accuracy >= {100.0 * ADEQUACY_ROOM_ACCURACY:.0f}%.",
        f"- Each room contributes up to `{payload['positions_per_room']}` candidate installation positions.",
        "- Singles are globally synchronized and localized with onset TDOA.",
        f"- Each array node has `{ARRAY_CHANNELS}` microphones; one node is a DOA-only baseline and two or more "
        "use DOA plus synchronized inter-array TDOA.",
        f"- Reflection tracer: `{payload['rt_accelerator']}` / `{payload['rt_precision']}` / device "
        f"`{payload['rt_cuda_device']}`.",
        "",
        "## Minimum tested configurations",
        "",
        "| Rooms | Evidence | Sensor | Nodes | Physical microphones | Observed gate | 95% CI support | Median | P90 | Room accuracy |",
        "|---:|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in payload["recommendations"]:
        sensor = "single microphones" if row["sensor_kind"] == "single" else "4-channel arrays"
        lines.append(
            f"| {row['room_count']} | {row['evidence_tier']} | {sensor} | {row['recommended_nodes']} | "
            f"{row['physical_microphones']} | {'yes' if row['meets_thresholds'] else 'no'} | "
            f"{'yes' if row['ci95_supports_thresholds'] else 'no'} | "
            f"{row['median_error_m']:.2f} m | {row['p90_error_m']:.2f} m | "
            f"{100.0 * row['room_accuracy']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Area sensitivity at the selected room-count configurations",
            "",
            "| Sensor | Relative area within room-count stratum | Selected node range | Median | P90 | Room accuracy |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["recommended_by_relative_area"]:
        sensor = "single microphones" if row["sensor_kind"] == "single" else "4-channel arrays"
        node_range = (
            str(row["node_count_min"])
            if int(row["node_count_min"]) == int(row["node_count_max"])
            else f"{row['node_count_min']}-{row['node_count_max']}"
        )
        lines.append(
            f"| {sensor} | {row['relative_area_bin']} | {node_range} | {row['median_error_m']:.2f} m | "
            f"{row['p90_error_m']:.2f} m | {100.0 * row['room_accuracy']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Selection first uses the smallest configuration whose 95% cluster-bootstrap interval supports all three "
            "accuracy gates. If no tested configuration has that support, the table falls back to the smallest point "
            "estimate that passes and marks 95% CI support as `no`; if even the point estimate fails, it reports the "
            f"best tested configuration. {sample_limit}Results assume one active source, known FloorPlan geometry, known "
            "sensor coordinates, open interior portals, and global clock synchronization. Real-device clock drift, "
            "sensor placement error, noise, and simulation-to-real transfer require separate validation.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
