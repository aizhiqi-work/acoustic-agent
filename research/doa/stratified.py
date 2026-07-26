from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
from urllib.parse import quote
import zlib

import numpy as np
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from acoustic_agent.floorplan_resource import FloorplanResource

from .distributed import (
    AcousticMeasurementGenerator,
    FloorplanModel,
    TargetPoint,
    load_model,
    localization_grid,
    localize_tdoa,
    place_nodes,
    sample_target_points,
    tune_risk_quantile,
)


DEFAULT_ROOM_COUNTS = tuple(range(4, 15))
ADEQUACY_MEDIAN_M = 1.0
ADEQUACY_P90_M = 2.0
ADEQUACY_ROOM_ACCURACY = 0.85


@dataclass(frozen=True)
class FloorplanProfile:
    index: int
    room_count: int
    area_m2: float
    portal_count: int
    connected: bool
    geometry_valid: bool

    @property
    def eligible(self) -> bool:
        return self.connected and self.geometry_valid


def scan_floorplan_population(resource: FloorplanResource | None = None) -> list[FloorplanProfile]:
    loader = resource or FloorplanResource()
    uri = f"file:{quote(str(loader.path))}?mode=ro&immutable=1"
    profiles: list[FloorplanProfile] = []
    with sqlite3.connect(uri, uri=True) as connection:
        for index, payload in connection.execute("SELECT idx, payload FROM scenes ORDER BY idx"):
            record = json.loads(zlib.decompress(payload))
            rooms = list(record.get("rooms") or [])
            room_ids = {str(room.get("id")) for room in rooms}
            adjacency = {room_id: set() for room_id in room_ids}
            portal_count = 0
            for portal in record.get("portals") or []:
                if not bool(portal.get("open", True)):
                    continue
                ids = [str(value) for value in portal.get("room_ids", []) if str(value) in room_ids]
                if len(ids) != 2:
                    continue
                portal_count += 1
                adjacency[ids[0]].add(ids[1])
                adjacency[ids[1]].add(ids[0])
            seen: set[str] = set()
            stack = [next(iter(room_ids))] if room_ids else []
            while stack:
                room_id = stack.pop()
                if room_id in seen:
                    continue
                seen.add(room_id)
                stack.extend(adjacency[room_id] - seen)
            area_m2 = float(sum(float(room.get("area_m2") or 0.0) for room in rooms))
            geometry_valid = (
                bool(rooms)
                and _valid_polygon_coordinates(record.get("corners") or [])
                and all(
                    _valid_polygon_coordinates(room.get("corners") or [])
                    and math.isfinite(float(room.get("area_m2") or 0.0))
                    and float(room.get("area_m2") or 0.0) >= 1.0
                    for room in rooms
                )
            )
            profiles.append(
                FloorplanProfile(
                    index=int(index),
                    room_count=len(rooms),
                    area_m2=area_m2,
                    portal_count=portal_count,
                    connected=bool(room_ids) and len(seen) == len(room_ids),
                    geometry_valid=geometry_valid,
                )
            )
    return profiles


def build_stratified_split(
    profiles: Sequence[FloorplanProfile],
    *,
    room_counts: Sequence[int] = DEFAULT_ROOM_COUNTS,
    calibration_per_count: int = 5,
    validation_per_count: int = 10,
    seed: int = 20260722,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for room_count in room_counts:
        population = sorted(
            (profile for profile in profiles if profile.room_count == int(room_count) and profile.eligible),
            key=lambda profile: (profile.area_m2, profile.index),
        )
        required = int(calibration_per_count) + int(validation_per_count)
        if len(population) < required:
            raise ValueError(
                f"room_count={room_count} has {len(population)} eligible FloorPlans; {required} are required"
            )
        validation = _area_stratified_sample(
            population,
            int(validation_per_count),
            seed=seed + int(room_count) * 101,
        )
        validation_ids = {profile.index for profile in validation}
        remaining = [profile for profile in population if profile.index not in validation_ids]
        calibration = _area_stratified_sample(
            remaining,
            int(calibration_per_count),
            seed=seed + int(room_count) * 101 + 1,
        )
        area_values = np.asarray([profile.area_m2 for profile in population], dtype=float)
        q33, q67 = np.quantile(area_values, [1.0 / 3.0, 2.0 / 3.0])
        for split, selected in (("calibration", calibration), ("validation", validation)):
            for profile in selected:
                rows.append(
                    {
                        **asdict(profile),
                        "split": split,
                        "relative_area_bin": _relative_area_bin(profile.area_m2, float(q33), float(q67)),
                        "absolute_area_bin": _absolute_area_bin(profile.area_m2),
                    }
                )
    return sorted(rows, key=lambda row: (str(row["split"]), int(row["room_count"]), float(row["area_m2"])))


def population_summary(profiles: Sequence[FloorplanProfile]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[int, list[FloorplanProfile]] = defaultdict(list)
    for profile in profiles:
        grouped[profile.room_count].append(profile)
    for room_count in sorted(grouped):
        group = grouped[room_count]
        eligible = [profile for profile in group if profile.eligible]
        areas = np.asarray([profile.area_m2 for profile in group], dtype=float)
        rows.append(
            {
                "room_count": room_count,
                "records": len(group),
                "connected_records": sum(profile.connected for profile in group),
                "eligible_records": len(eligible),
                "connected_rate": round(float(np.mean([profile.connected for profile in group])), 6),
                "area_min_m2": round(float(np.min(areas)), 3),
                "area_p10_m2": round(float(np.percentile(areas, 10)), 3),
                "area_median_m2": round(float(np.median(areas)), 3),
                "area_p90_m2": round(float(np.percentile(areas, 90)), 3),
                "area_max_m2": round(float(np.max(areas)), 3),
            }
        )
    return rows


def run_stratified_study(
    output_dir: str | Path,
    *,
    room_counts: Sequence[int] = DEFAULT_ROOM_COUNTS,
    calibration_per_count: int = 5,
    validation_per_count: int = 10,
    quality: str = "preview",
    points_per_room: int = 1,
    seed: int = 20260722,
    rt_accelerator: str = "numba",
    rt_precision: str = "float64",
    rt_cuda_device: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resource = FloorplanResource()
    profiles = scan_floorplan_population(resource)
    population_rows = population_summary(profiles)
    split_rows = build_stratified_split(
        profiles,
        room_counts=room_counts,
        calibration_per_count=calibration_per_count,
        validation_per_count=validation_per_count,
        seed=seed,
    )
    calibration_indices = [int(row["index"]) for row in split_rows if row["split"] == "calibration"]
    validation_rows = [row for row in split_rows if row["split"] == "validation"]
    validation_indices = [int(row["index"]) for row in validation_rows]
    profile_by_index = {profile.index: profile for profile in profiles}
    split_by_index = {int(row["index"]): row for row in split_rows}

    risk_quantile, tuning_rows = tune_risk_quantile(calibration_indices, resource=resource)
    generator = AcousticMeasurementGenerator(
        output,
        quality=quality,
        rt_accelerator=rt_accelerator,
        rt_precision=rt_precision,
        rt_cuda_device=rt_cuda_device,
    )
    results: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    planned_rirs = sum(
        int(row["room_count"]) ** 2 * max(1, int(points_per_room))
        for row in validation_rows
    )

    with tqdm(
        total=planned_rirs,
        desc="Localization RIR",
        unit="rir",
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.1,
    ) as progress:
        for floorplan_index in validation_indices:
            model = load_model(floorplan_index, resource)
            profile = profile_by_index[floorplan_index]
            split = split_by_index[floorplan_index]
            grid = localization_grid(model, spacing_m=0.55)
            ordered_nodes = place_nodes(
                model,
                len(model.rooms),
                mode="single",
                risk_quantile=risk_quantile,
            )
            targets = sample_target_points(
                model,
                points_per_room=max(1, int(points_per_room)),
                seed=seed + floorplan_index * 17,
                spacing_m=0.52,
            )
            progress.set_postfix(
                floorplan=floorplan_index,
                rooms=profile.room_count,
                refresh=False,
            )
            for order, node in enumerate(ordered_nodes, start=1):
                placements.append(
                    {
                        "floorplan_idx": floorplan_index,
                        "room_count": profile.room_count,
                        "area_m2": round(profile.area_m2, 4),
                        "order": order,
                        "node_id": node.id,
                        "room_id": node.room_id,
                        "x_m": round(node.position[0], 5),
                        "y_m": round(node.position[1], 5),
                        "z_m": round(node.position[2], 5),
                    }
                )
            for target in targets:
                measurements = []
                for node in ordered_nodes:
                    measurements.append(generator.single(model, target, node))
                    progress.update(1)
                truth = np.asarray(target.position[:2], dtype=float)
                for microphone_count in range(3, len(ordered_nodes) + 1):
                    nodes = ordered_nodes[:microphone_count]
                    estimate, estimated_room, _ = localize_tdoa(
                        model,
                        nodes,
                        measurements[:microphone_count],
                        grid,
                    )
                    results.append(
                        {
                            "split": "validation",
                            "floorplan_idx": floorplan_index,
                            "room_count": profile.room_count,
                            "area_m2": round(profile.area_m2, 4),
                            "relative_area_bin": split["relative_area_bin"],
                            "absolute_area_bin": split["absolute_area_bin"],
                            "target_id": target.id,
                            "target_room": target.room_id,
                            "microphones": microphone_count,
                            "configuration": f"single_{microphone_count}x1",
                            "local_microphone": any(node.room_id == target.room_id for node in nodes),
                            "true_x_m": round(float(truth[0]), 5),
                            "true_y_m": round(float(truth[1]), 5),
                            "estimated_x_m": round(float(estimate[0]), 5),
                            "estimated_y_m": round(float(estimate[1]), 5),
                            "position_error_m": round(float(np.linalg.norm(estimate - truth)), 5),
                            "estimated_room": estimated_room,
                            "room_correct": bool(estimated_room == target.room_id),
                        }
                    )

    by_room_count = _group_metrics(results, ("room_count", "microphones"))
    by_relative_area = _group_metrics(results, ("relative_area_bin", "microphones"))
    by_absolute_area = _group_metrics(results, ("absolute_area_bin", "microphones"))
    overall = _group_metrics(results, ("microphones",))
    recommendations = _recommend_by_room_count(results, seed=seed)
    recommended_microphones = {
        int(row["room_count"]): int(row["recommended_microphones"]) for row in recommendations
    }
    recommended_rows = [
        row
        for row in results
        if int(row["microphones"]) == recommended_microphones[int(row["room_count"])]
    ]
    recommended_by_relative_area = _group_metrics(recommended_rows, ("relative_area_bin",))
    recommended_by_absolute_area = _group_metrics(recommended_rows, ("absolute_area_bin",))
    payload = {
        "study": "stratified_floorplan_distributed_localization",
        "database_records": len(profiles),
        "room_counts": [int(value) for value in room_counts],
        "calibration_per_room_count": int(calibration_per_count),
        "validation_per_room_count": int(validation_per_count),
        "calibration_indices": calibration_indices,
        "validation_indices": validation_indices,
        "quality": quality,
        "rt_accelerator": rt_accelerator,
        "rt_precision": rt_precision,
        "rt_cuda_device": int(rt_cuda_device),
        "points_per_room": int(points_per_room),
        "selected_risk_quantile": risk_quantile,
        "adequacy_thresholds": {
            "median_error_m": ADEQUACY_MEDIAN_M,
            "p90_error_m": ADEQUACY_P90_M,
            "room_accuracy": ADEQUACY_ROOM_ACCURACY,
        },
        "population": population_rows,
        "split": split_rows,
        "tuning": tuning_rows,
        "results": results,
        "placements": placements,
        "by_room_count": by_room_count,
        "by_relative_area": by_relative_area,
        "by_absolute_area": by_absolute_area,
        "overall": overall,
        "recommendations": recommendations,
        "recommended_by_relative_area": recommended_by_relative_area,
        "recommended_by_absolute_area": recommended_by_absolute_area,
    }
    _write_outputs(output, payload)
    return payload


def _area_stratified_sample(
    profiles: Sequence[FloorplanProfile],
    count: int,
    *,
    seed: int,
) -> list[FloorplanProfile]:
    if count < 1 or count > len(profiles):
        raise ValueError("count must be between 1 and the population size")
    ordered = sorted(profiles, key=lambda profile: (profile.area_m2, profile.index))
    rng = np.random.default_rng(seed)
    selected: list[FloorplanProfile] = []
    edges = np.linspace(0, len(ordered), count + 1)
    for index in range(count):
        lower = int(math.floor(edges[index]))
        upper = int(math.floor(edges[index + 1]))
        upper = max(upper, lower + 1)
        chosen = int(rng.integers(lower, min(upper, len(ordered))))
        selected.append(ordered[chosen])
    return selected


def _valid_polygon_coordinates(coordinates: Sequence[Sequence[float]]) -> bool:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        return False
    values = values[:, :2]
    if not np.all(np.isfinite(values)):
        return False
    if np.linalg.norm(values[0] - values[-1]) <= 1e-7:
        values = values[:-1]
    if values.shape[0] < 3:
        return False
    edges = np.linalg.norm(np.roll(values, -1, axis=0) - values, axis=1)
    if float(np.min(edges)) <= 1e-7:
        return False
    polygon = Polygon(values)
    return bool(polygon.is_valid and polygon.area > 1e-4)


def _relative_area_bin(area_m2: float, q33: float, q67: float) -> str:
    if area_m2 <= q33:
        return "small"
    if area_m2 <= q67:
        return "medium"
    return "large"


def _absolute_area_bin(area_m2: float) -> str:
    if area_m2 < 60.0:
        return "compact_lt60"
    if area_m2 < 100.0:
        return "medium_60_100"
    if area_m2 < 150.0:
        return "large_100_150"
    return "very_large_ge150"


def _group_metrics(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for values, subset in groups.items():
        errors = np.asarray([float(row["position_error_m"]) for row in subset], dtype=float)
        cross = [row for row in subset if not bool(row["local_microphone"])]
        summary = {key: value for key, value in zip(keys, values)}
        summary.update(
            {
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


def _recommend_by_room_count(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    room_counts = sorted({int(row["room_count"]) for row in rows})
    for room_count in room_counts:
        room_rows = [row for row in rows if int(row["room_count"]) == room_count]
        summaries = _group_metrics(room_rows, ("microphones",))
        adequate = [
            row
            for row in summaries
            if float(row["median_error_m"]) <= ADEQUACY_MEDIAN_M
            and float(row["p90_error_m"]) <= ADEQUACY_P90_M
            and float(row["room_accuracy"]) >= ADEQUACY_ROOM_ACCURACY
        ]
        if adequate:
            chosen = min(adequate, key=lambda row: int(row["microphones"]))
            adequate_flag = True
        else:
            chosen = min(
                summaries,
                key=lambda row: (
                    float(row["median_error_m"])
                    + 0.5 * float(row["p90_error_m"])
                    + 2.0 * (1.0 - float(row["room_accuracy"])),
                    int(row["microphones"]),
                ),
            )
            adequate_flag = False
        selected = [row for row in room_rows if int(row["microphones"]) == int(chosen["microphones"])]
        confidence = _cluster_bootstrap(selected, seed=seed + room_count * 313)
        recommendations.append(
            {
                "room_count": room_count,
                "recommended_microphones": int(chosen["microphones"]),
                "meets_thresholds": adequate_flag,
                **{key: chosen[key] for key in (
                    "floorplans",
                    "cases",
                    "median_error_m",
                    "p90_error_m",
                    "room_accuracy",
                    "cross_room_cases",
                    "cross_room_median_error_m",
                    "cross_room_room_accuracy",
                )},
                **confidence,
            }
        )
    return recommendations


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    iterations: int = 1000,
) -> dict[str, float]:
    by_plan: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_plan[int(row["floorplan_idx"])].append(row)
    plan_ids = sorted(by_plan)
    rng = np.random.default_rng(seed)
    medians = []
    p90s = []
    accuracies = []
    for _ in range(iterations):
        sampled = rng.choice(plan_ids, size=len(plan_ids), replace=True)
        values = [row for plan_id in sampled for row in by_plan[int(plan_id)]]
        errors = np.asarray([float(row["position_error_m"]) for row in values], dtype=float)
        medians.append(float(np.median(errors)))
        p90s.append(float(np.percentile(errors, 90)))
        accuracies.append(float(np.mean([bool(row["room_correct"]) for row in values])))
    return {
        "median_error_ci95_low_m": round(float(np.percentile(medians, 2.5)), 4),
        "median_error_ci95_high_m": round(float(np.percentile(medians, 97.5)), 4),
        "p90_error_ci95_low_m": round(float(np.percentile(p90s, 2.5)), 4),
        "p90_error_ci95_high_m": round(float(np.percentile(p90s, 97.5)), 4),
        "room_accuracy_ci95_low": round(float(np.percentile(accuracies, 2.5)), 4),
        "room_accuracy_ci95_high": round(float(np.percentile(accuracies, 97.5)), 4),
    }


def _write_outputs(output: Path, payload: Mapping[str, Any]) -> None:
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, key in (
        ("population.csv", "population"),
        ("split.csv", "split"),
        ("tuning.csv", "tuning"),
        ("placements.csv", "placements"),
        ("results.csv", "results"),
        ("by_room_count.csv", "by_room_count"),
        ("by_relative_area.csv", "by_relative_area"),
        ("by_absolute_area.csv", "by_absolute_area"),
        ("overall.csv", "overall"),
        ("recommendations.csv", "recommendations"),
        ("recommended_by_relative_area.csv", "recommended_by_relative_area"),
        ("recommended_by_absolute_area.csv", "recommended_by_absolute_area"),
    ):
        _write_csv(output / name, payload[key])
    lines = [
        "# Stratified FloorPlan Distributed Localization",
        "",
        f"- Database: **{payload['database_records']:,}** FloorPlans.",
        f"- Main room-count strata: **{payload['room_counts'][0]}-{payload['room_counts'][-1]} rooms**.",
        f"- Calibration: **{payload['calibration_per_room_count']} FloorPlans per room count**.",
        f"- Unseen validation: **{payload['validation_per_room_count']} FloorPlans per room count**.",
        f"- Acoustic cases: **{payload['points_per_room']}** source point(s) per room "
        f"at `{payload['quality']}` RIR quality.",
        f"- Reflection tracer: `{payload['rt_accelerator']}` / `{payload['rt_precision']}` / device `{payload['rt_cuda_device']}`.",
        "- Validation FloorPlans are area-stratified within each room-count stratum.",
        "- Confidence intervals resample entire FloorPlans, not individual source points.",
        "",
        "## Database population",
        "",
        "| Rooms | Records | Connected | Eligible | Area P10 | Median | Area P90 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["population"]:
        lines.append(
            f"| {row['room_count']} | {row['records']} | {row['connected_records']} "
            f"({100.0 * row['connected_rate']:.1f}%) | {row['eligible_records']} | "
            f"{row['area_p10_m2']:.1f} m2 | {row['area_median_m2']:.1f} m2 | {row['area_p90_m2']:.1f} m2 |"
        )
    lines.extend(
        [
            "",
            "Room counts 2 and 3 do not contain enough connected records for a disjoint 5+10 split and are reported "
            "in the population table but excluded from the balanced main experiment.",
            "",
            "## Minimum synchronized single-microphone deployment",
            "",
            "Adequacy requires median error <= 1.0 m, P90 <= 2.0 m, and room accuracy >= 85%.",
            "",
            "| Rooms | Recommended mics | Adequate | Median (floorplan-bootstrap 95% CI) | P90 | Room accuracy | Cross-room median |",
            "|---:|---:|:---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["recommendations"]:
        cross = "n/a" if row["cross_room_median_error_m"] is None else f"{row['cross_room_median_error_m']:.2f} m"
        lines.append(
            f"| {row['room_count']} | {row['recommended_microphones']} | "
            f"{'yes' if row['meets_thresholds'] else 'no'} | {row['median_error_m']:.2f} m "
            f"[{row['median_error_ci95_low_m']:.2f}, {row['median_error_ci95_high_m']:.2f}] | "
            f"{row['p90_error_m']:.2f} m | {100.0 * row['room_accuracy']:.1f}% | {cross} |"
        )
    relative_rows = {row["relative_area_bin"]: row for row in payload["recommended_by_relative_area"]}
    absolute_rows = {row["absolute_area_bin"]: row for row in payload["recommended_by_absolute_area"]}
    lines.extend(
        [
            "",
            "## Floor-area sensitivity",
            "",
            "Each row below uses the recommended microphone count for its own room-count stratum, so the relative-area "
            "comparison does not simply reward smaller homes for having fewer rooms.",
            "",
            "| Within-room-count area bin | FloorPlans | Cases | Median | P90 | Room accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    _append_area_metric_rows(lines, relative_rows, ("small", "medium", "large"))
    lines.extend(
        [
            "",
            "| Absolute floor area | FloorPlans | Cases | Median | P90 | Room accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    _append_area_metric_rows(
        lines,
        absolute_rows,
        ("compact_lt60", "medium_60_100", "large_100_150", "very_large_ge150"),
    )
    observed_rule = all(
        int(row["recommended_microphones"])
        == min(7, max(4, int(math.ceil(int(row["room_count"]) / 2.0)) + 1))
        for row in payload["recommendations"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The recommendation is the smallest tested topology-aware deployment that satisfies all three fixed "
            "validation thresholds. If none does, the table reports the lowest fixed accuracy/cost penalty and marks "
            "the row as not adequate. These are simulation conclusions for one active source, known microphone "
            "coordinates, known FloorPlan geometry, open interior portals, and globally synchronized microphones.",
            "" if not observed_rule else "Across these strata, the observed minimum follows `min(7, max(4, ceil(rooms / 2) + 1))`: "
            "four microphones up to six rooms, five for seven to eight, six for nine to ten, and seven for eleven to "
            "fourteen rooms. This compact rule is an empirical summary of this validation set, not a universal law.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _append_area_metric_rows(
    lines: list[str],
    rows_by_label: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
) -> None:
    for label in labels:
        row = rows_by_label.get(label)
        if row is None:
            lines.append(f"| {label} | 0 | 0 | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {row['floorplans']} | {row['cases']} | {row['median_error_m']:.2f} m | "
            f"{row['p90_error_m']:.2f} m | {100.0 * row['room_accuracy']:.1f}% |"
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
