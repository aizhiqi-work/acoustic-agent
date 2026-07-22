from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from acoustic_agent import AcousticAgent, microphone_array
from acoustic_agent.floorplan_resource import FloorplanResource
from acoustic_agent.mic import channel_positions

from .estimators import angular_error_deg, estimate_srp_phat
from .experiment import C, FS, broadband_probe, render_observation


TRAIN_INDICES = (0, 2, 7, 13)
TEST_INDICES = (20, 29, 41, 60)
RISK_QUANTILES = (0.05, 0.20, 0.35)


@dataclass(frozen=True)
class SensorNode:
    id: str
    room_id: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class TargetPoint:
    id: str
    room_id: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class DOAMeasurement:
    node_id: str
    bearing_deg: float
    confidence: float
    peak_ratio: float
    arrival_s: float = 0.0


@dataclass(frozen=True)
class TOAMeasurement:
    node_id: str
    arrival_s: float
    confidence: float


class FloorplanModel:
    def __init__(self, index: int, record: Mapping[str, Any]) -> None:
        self.index = int(index)
        self.record = dict(record)
        self.rooms = {str(room["id"]): dict(room) for room in record["rooms"]}
        self.polygons: dict[str, Polygon] = {}
        for room_id, room in self.rooms.items():
            polygon = Polygon(room["corners"])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.geom_type == "MultiPolygon":
                polygon = max(polygon.geoms, key=lambda item: item.area)
            self.polygons[room_id] = polygon
        self.portals = [dict(portal) for portal in record.get("portals", []) if bool(portal.get("open", True))]
        self.adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {room_id: [] for room_id in self.rooms}
        for portal in self.portals:
            room_ids = [str(value) for value in portal.get("room_ids", []) if str(value) in self.rooms]
            if len(room_ids) != 2:
                continue
            first, second = room_ids
            self.adjacency[first].append((second, portal))
            self.adjacency[second].append((first, portal))
        self._route_cache: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}

    def route(self, source_room: str, receiver_room: str) -> tuple[dict[str, Any], ...]:
        key = (str(source_room), str(receiver_room))
        if key in self._route_cache:
            return self._route_cache[key]
        if key[0] == key[1]:
            self._route_cache[key] = ()
            return ()
        queue: deque[tuple[str, tuple[dict[str, Any], ...]]] = deque([(key[0], ())])
        seen = {key[0]}
        while queue:
            room_id, route = queue.popleft()
            for neighbor, portal in self.adjacency.get(room_id, []):
                if neighbor in seen:
                    continue
                next_route = (*route, portal)
                if neighbor == key[1]:
                    self._route_cache[key] = next_route
                    return next_route
                seen.add(neighbor)
                queue.append((neighbor, next_route))
        raise ValueError(f"rooms {key[0]!r} and {key[1]!r} are disconnected")

    def propagation(
        self,
        source_xy: Sequence[float],
        source_room: str,
        receiver_xy: Sequence[float],
        receiver_room: str,
    ) -> tuple[float, float, int]:
        """Return arrival bearing at receiver, polyline distance, and portal hops."""
        source = np.asarray(source_xy[:2], dtype=float)
        receiver = np.asarray(receiver_xy[:2], dtype=float)
        route = self.route(source_room, receiver_room)
        if not route:
            anchor = source
            points = [source, receiver]
        else:
            points = [source]
            current_room = str(source_room)
            for portal in route:
                room_points = portal.get("room_points", {})
                point = room_points.get(current_room, portal.get("center"))
                points.append(np.asarray(point[:2], dtype=float))
                room_ids = [str(value) for value in portal.get("room_ids", [])]
                current_room = room_ids[1] if room_ids[0] == current_room else room_ids[0]
            final_portal = route[-1]
            final_points = final_portal.get("room_points", {})
            anchor = np.asarray(final_points.get(str(receiver_room), final_portal.get("center"))[:2], dtype=float)
            points[-1] = anchor
            points.append(receiver)
        delta = anchor - receiver
        bearing = float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0)
        distance = float(sum(np.linalg.norm(points[index + 1] - points[index]) for index in range(len(points) - 1)))
        return bearing, distance, len(route)

    def first_leg_direction(
        self,
        source_xy: Sequence[float],
        source_room: str,
        receiver_room: str,
        receiver_xy: Sequence[float],
    ) -> np.ndarray:
        source = np.asarray(source_xy[:2], dtype=float)
        route = self.route(source_room, receiver_room)
        if route:
            portal = route[0]
            destination = np.asarray(portal.get("room_points", {}).get(source_room, portal.get("center"))[:2], dtype=float)
        else:
            destination = np.asarray(receiver_xy[:2], dtype=float)
        delta = source - destination
        norm = max(float(np.linalg.norm(delta)), 1e-6)
        return delta / norm


def load_model(index: int, resource: FloorplanResource | None = None) -> FloorplanModel:
    loader = resource or FloorplanResource()
    return FloorplanModel(index, loader.record(index))


def candidate_nodes(
    model: FloorplanModel,
    *,
    height_m: float = 2.2,
    positions_per_room: int = 1,
) -> list[SensorNode]:
    nodes: list[SensorNode] = []
    positions_per_room = max(1, int(positions_per_room))
    for room_id in sorted(model.rooms):
        polygon = _safe_polygon(model.polygons[room_id], 0.28)
        center = polygon.representative_point()
        nodes.append(SensorNode(f"{room_id}:center", room_id, (float(center.x), float(center.y), float(height_m))))
        if positions_per_room == 1:
            continue
        points = _polygon_grid(model.polygons[room_id], spacing_m=0.7, margin_m=0.32)
        selected = [np.asarray([center.x, center.y], dtype=float)]
        remaining = [np.asarray(point, dtype=float) for point in points]
        for position_index in range(1, positions_per_room):
            if remaining:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: min(
                        float(np.linalg.norm(remaining[index] - other)) for other in selected
                    ),
                )
                chosen = remaining.pop(chosen_index)
            else:
                chosen = selected[-1]
            selected.append(chosen)
            nodes.append(
                SensorNode(
                    f"{room_id}:aux:{position_index}",
                    room_id,
                    (float(chosen[0]), float(chosen[1]), float(height_m)),
                )
            )
    return nodes


def sample_target_points(
    model: FloorplanModel,
    *,
    points_per_room: int,
    seed: int,
    height_m: float = 1.5,
    spacing_m: float = 0.55,
) -> list[TargetPoint]:
    rng = np.random.default_rng(seed)
    targets: list[TargetPoint] = []
    for room_id in sorted(model.rooms):
        points = _polygon_grid(model.polygons[room_id], spacing_m=spacing_m, margin_m=0.3)
        if not points:
            point = _safe_polygon(model.polygons[room_id], 0.15).representative_point()
            points = [(float(point.x), float(point.y))]
        order = rng.permutation(len(points))
        chosen = [points[int(index)] for index in order[: min(points_per_room, len(points))]]
        while len(chosen) < points_per_room:
            chosen.append(points[len(chosen) % len(points)])
        for point_index, (x_value, y_value) in enumerate(chosen):
            targets.append(TargetPoint(f"{room_id}:target:{point_index}", room_id, (x_value, y_value, float(height_m))))
    return targets


def localization_grid(model: FloorplanModel, spacing_m: float = 0.45) -> list[TargetPoint]:
    grid: list[TargetPoint] = []
    for room_id in sorted(model.rooms):
        points = _polygon_grid(model.polygons[room_id], spacing_m=spacing_m, margin_m=0.18)
        if not points:
            center = _safe_polygon(model.polygons[room_id], 0.08).representative_point()
            points = [(float(center.x), float(center.y))]
        for point_index, (x_value, y_value) in enumerate(points):
            grid.append(TargetPoint(f"{room_id}:grid:{point_index}", room_id, (x_value, y_value, 1.5)))
    return grid


def place_nodes(
    model: FloorplanModel,
    count: int,
    *,
    mode: str,
    risk_quantile: float,
    method: str = "topology_greedy",
    candidates: Sequence[SensorNode] | None = None,
) -> list[SensorNode]:
    available = list(candidates) if candidates is not None else candidate_nodes(model)
    count = min(max(int(count), 1), len(available))
    if method == "largest_rooms":
        return sorted(available, key=lambda node: float(model.rooms[node.room_id]["area_m2"]), reverse=True)[:count]
    if method == "farthest_rooms":
        selected = [max(available, key=lambda node: float(model.rooms[node.room_id]["area_m2"]))]
        while len(selected) < count:
            remaining = [node for node in available if node not in selected]
            selected.append(
                max(
                    remaining,
                    key=lambda node: min(
                        np.linalg.norm(np.asarray(node.position[:2]) - np.asarray(other.position[:2])) for other in selected
                    ),
                )
            )
        return selected
    if method != "topology_greedy":
        raise ValueError("method must be topology_greedy, largest_rooms, or farthest_rooms")

    structural_points = sample_target_points(model, points_per_room=3, seed=17_000 + model.index, spacing_m=0.7)
    selected: list[SensorNode] = []
    while len(selected) < count:
        remaining = [node for node in available if node not in selected]
        selected.append(
            max(
                remaining,
                key=lambda node: _deployment_score(model, [*selected, node], structural_points, mode, risk_quantile),
            )
        )
    return selected


def tune_risk_quantile(
    train_indices: Sequence[int] = TRAIN_INDICES,
    *,
    resource: FloorplanResource | None = None,
) -> tuple[float, list[dict[str, float]]]:
    loader = resource or FloorplanResource()
    rows: list[dict[str, float]] = []
    for quantile in RISK_QUANTILES:
        plan_scores = []
        for index in train_indices:
            model = load_model(index, loader)
            points = sample_target_points(model, points_per_room=3, seed=31_000 + index, spacing_m=0.65)
            arrays = place_nodes(model, 2, mode="array", risk_quantile=quantile)
            singles = place_nodes(model, min(4, len(model.rooms)), mode="single", risk_quantile=quantile)
            score = _deployment_score(model, arrays, points, "array", 0.1)
            score += _deployment_score(model, singles, points, "single", 0.1)
            plan_scores.append(score)
        rows.append(
            {
                "risk_quantile": float(quantile),
                "mean_training_score": float(np.mean(plan_scores)),
                "minimum_training_score": float(np.min(plan_scores)),
            }
        )
    best = max(rows, key=lambda row: (row["mean_training_score"], row["minimum_training_score"]))
    return float(best["risk_quantile"]), rows


def localize_doa(
    model: FloorplanModel,
    nodes: Sequence[SensorNode],
    measurements: Sequence[DOAMeasurement],
    grid: Sequence[TargetPoint],
) -> tuple[np.ndarray, str, np.ndarray]:
    node_by_id = {node.id: node for node in nodes}
    losses = np.zeros(len(grid), dtype=np.float64)
    total_weight = np.zeros(len(grid), dtype=np.float64)
    for measurement in measurements:
        node = node_by_id[measurement.node_id]
        for index, point in enumerate(grid):
            predicted, _, hops = model.propagation(point.position, point.room_id, node.position, node.room_id)
            sigma_deg = 4.0 + 5.0 * hops
            weight = max(0.05, float(measurement.confidence)) * (0.72 ** hops)
            residual = angular_error_deg(measurement.bearing_deg, predicted) / sigma_deg
            losses[index] += weight * _huber(residual)
            total_weight[index] += weight
    losses /= np.maximum(total_weight, 1e-9)
    position, room_id = _estimate_grid_position(grid, losses)
    return position, room_id, losses


def localize_tdoa(
    model: FloorplanModel,
    nodes: Sequence[SensorNode],
    measurements: Sequence[TOAMeasurement],
    grid: Sequence[TargetPoint],
) -> tuple[np.ndarray, str, np.ndarray]:
    node_by_id = {node.id: node for node in nodes}
    ordered = [measurement for measurement in measurements if measurement.node_id in node_by_id]
    observed = np.asarray([measurement.arrival_s for measurement in ordered], dtype=float)
    observed -= np.mean(observed)
    losses = np.zeros(len(grid), dtype=np.float64)
    for index, point in enumerate(grid):
        predicted = []
        weights = []
        for measurement in ordered:
            node = node_by_id[measurement.node_id]
            _, distance, hops = model.propagation(point.position, point.room_id, node.position, node.room_id)
            predicted.append(distance / C)
            weights.append(max(0.05, measurement.confidence) * (0.78 ** hops))
        predicted_values = np.asarray(predicted, dtype=float)
        predicted_values -= np.mean(predicted_values)
        residual_us = (observed - predicted_values) * 1e6
        losses[index] = float(np.average(_huber(residual_us / 120.0), weights=np.asarray(weights)))
    position, room_id = _estimate_grid_position(grid, losses)
    return position, room_id, losses


def localize_hybrid(
    model: FloorplanModel,
    array_nodes: Sequence[SensorNode],
    doa: Sequence[DOAMeasurement],
    single_nodes: Sequence[SensorNode],
    toa: Sequence[TOAMeasurement],
    grid: Sequence[TargetPoint],
) -> tuple[np.ndarray, str, np.ndarray]:
    _, _, doa_loss = localize_doa(model, array_nodes, doa, grid)
    _, _, tdoa_loss = localize_tdoa(model, single_nodes, toa, grid)
    doa_scale = max(float(np.percentile(doa_loss, 75) - np.min(doa_loss)), 1e-6)
    tdoa_scale = max(float(np.percentile(tdoa_loss, 75) - np.min(tdoa_loss)), 1e-6)
    combined = 0.20 * (doa_loss - np.min(doa_loss)) / doa_scale
    combined += 0.80 * (tdoa_loss - np.min(tdoa_loss)) / tdoa_scale
    position, room_id = _estimate_grid_position(grid, combined)
    return position, room_id, combined


class AcousticMeasurementGenerator:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        quality: str = "preview",
        rt_accelerator: str = "numba",
        rt_precision: str = "float64",
        rt_cuda_device: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cache_dir = self.output_dir / "measurement-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quality = str(quality)
        self.rt_accelerator = str(rt_accelerator)
        self.rt_precision = str(rt_precision)
        self.rt_cuda_device = int(rt_cuda_device)
        self.probe = broadband_probe(FS, duration_s=0.35, seed=20260722)

    def array(
        self,
        model: FloorplanModel,
        target: TargetPoint,
        node: SensorNode,
        *,
        channels: int,
    ) -> DOAMeasurement:
        key = self._key("array", model, target, node, channels)
        cached = self._load(key)
        if cached is not None:
            return DOAMeasurement(
                node.id,
                float(cached["bearing_deg"]),
                float(cached["confidence"]),
                float(cached["peak_ratio"]),
                float(cached["arrival_s"]),
            )
        receiver_model = microphone_array("circular", count=int(channels), radius_m=0.05)
        result = self._simulate(model, target, node, receiver_model)
        observation = render_observation(np.asarray(result.rir), self.probe)
        positions = np.asarray(channel_positions(node.position, receiver_model), dtype=float)
        bearing, spectrum = estimate_srp_phat(
            observation,
            positions,
            fs=FS,
            search_deg=np.arange(0.0, 360.0, 1.0),
            speed_of_sound_m_s=C,
        )
        confidence, peak_ratio = _spectrum_confidence(spectrum)
        response = np.asarray(result.rir, dtype=float)
        channel_arrivals = []
        for channel in response:
            peak = max(float(np.max(np.abs(channel))), 1e-12)
            indices = np.flatnonzero(np.abs(channel) >= peak * 0.025)
            channel_arrivals.append((int(indices[0]) if indices.size else int(np.argmax(np.abs(channel)))) / FS)
        arrival_s = float(np.median(channel_arrivals))
        payload = {
            "bearing_deg": bearing,
            "confidence": confidence,
            "peak_ratio": peak_ratio,
            "arrival_s": arrival_s,
        }
        self._save(key, payload)
        return DOAMeasurement(node.id, bearing, confidence, peak_ratio, arrival_s)

    def single(self, model: FloorplanModel, target: TargetPoint, node: SensorNode) -> TOAMeasurement:
        key = self._key("single", model, target, node, 1)
        cached = self._load(key)
        if cached is not None:
            return TOAMeasurement(node.id, float(cached["arrival_s"]), float(cached["confidence"]))
        result = self._simulate(model, target, node, microphone_array("mono"))
        rir = np.asarray(result.rir, dtype=float).reshape(-1)
        peak = max(float(np.max(np.abs(rir))), 1e-12)
        indices = np.flatnonzero(np.abs(rir) >= peak * 0.025)
        arrival_sample = int(indices[0]) if indices.size else int(np.argmax(np.abs(rir)))
        direct = result.metadata.get("steam_audio", {}).get("direct", {})
        visibility = float(direct.get("occlusion", 0.0))
        confidence = float(np.clip(0.45 + 0.5 * visibility, 0.15, 0.95))
        payload = {"arrival_s": arrival_sample / FS, "confidence": confidence}
        self._save(key, payload)
        return TOAMeasurement(node.id, arrival_sample / FS, confidence)

    def _simulate(
        self,
        model: FloorplanModel,
        target: TargetPoint,
        node: SensorNode,
        receiver_model: Mapping[str, Any],
    ) -> Any:
        agent = AcousticAgent.create(
            scene="floorplan",
            idx=model.index,
            placement="same_room" if target.room_id == node.room_id else "cross_room",
            source=target.position,
            receiver=node.position,
            source_room=target.room_id,
            receiver_room=node.room_id,
            seed=77_000 + model.index,
            material_seed=2026,
            receiver_model=receiver_model,
            source_model="omni",
            quality=self.quality,
            duration_s=0.55,
            fs=FS,
            visualization=False,
        )
        config = replace(
            agent.config,
            duration_s=0.55,
            rt_duration_s=0.55,
            late_tail=False,
            collect_visual_paths=False,
            render_ambisonics=False,
            rt_accelerator=self.rt_accelerator,
            rt_precision=self.rt_precision,
            rt_cuda_device=self.rt_cuda_device,
        )
        return agent.run(config=config)

    def _key(self, kind: str, model: FloorplanModel, target: TargetPoint, node: SensorNode, channels: int) -> str:
        value = json.dumps(
            {
                "kind": kind,
                "idx": model.index,
                "target": target.position,
                "target_room": target.room_id,
                "node": node.position,
                "node_room": node.room_id,
                "channels": channels,
                "quality": self.quality,
                "rt_accelerator": self.rt_accelerator,
                "rt_precision": self.rt_precision,
                "rt_cuda_device": self.rt_cuda_device,
                "version": 4,
            },
            sort_keys=True,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _load(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _save(self, key: str, payload: Mapping[str, Any]) -> None:
        (self.cache_dir / f"{key}.json").write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")


def run_distributed_study(
    output_dir: str | Path,
    *,
    train_indices: Sequence[int] = TRAIN_INDICES,
    test_indices: Sequence[int] = TEST_INDICES,
    quality: str = "preview",
    points_per_room: int = 1,
    rt_accelerator: str = "numba",
    rt_precision: str = "float64",
    rt_cuda_device: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resource = FloorplanResource()
    risk_quantile, tuning_rows = tune_risk_quantile(train_indices, resource=resource)
    generator = AcousticMeasurementGenerator(
        output,
        quality=quality,
        rt_accelerator=rt_accelerator,
        rt_precision=rt_precision,
        rt_cuda_device=rt_cuda_device,
    )
    result_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []

    configurations = (
        ("array_1x8", "doa", 1, 0, 8, "topology_greedy"),
        ("array_2x4_async", "doa", 2, 0, 8, "topology_greedy"),
        ("array_3x4_async", "doa", 3, 0, 12, "topology_greedy"),
        ("array_4x4_async", "doa", 4, 0, 16, "topology_greedy"),
        ("array_2x4_sync", "sync_array", 2, 0, 8, "topology_greedy"),
        ("array_3x4_sync", "sync_array", 3, 0, 12, "topology_greedy"),
        ("array_4x4_sync", "sync_array", 4, 0, 16, "topology_greedy"),
        ("single_3x1", "tdoa", 0, 3, 3, "topology_greedy"),
        ("single_4x1", "tdoa", 0, 4, 4, "topology_greedy"),
        ("single_6x1", "tdoa", 0, 6, 6, "topology_greedy"),
        ("single_6x1_100us", "tdoa_100us", 0, 6, 6, "topology_greedy"),
        ("single_6x1_500us", "tdoa_500us", 0, 6, 6, "topology_greedy"),
        ("single_8x1", "tdoa", 0, 8, 8, "topology_greedy"),
        ("single_6x1_largest", "tdoa", 0, 6, 6, "largest_rooms"),
        ("single_6x1_farthest", "tdoa", 0, 6, 6, "farthest_rooms"),
        ("hybrid_2x4_4x1", "hybrid", 2, 4, 12, "topology_greedy"),
    )

    for floorplan_index in test_indices:
        model = load_model(floorplan_index, resource)
        grid = localization_grid(model)
        max_arrays = min(4, len(model.rooms))
        max_singles = min(8, len(model.rooms))
        array_nodes = place_nodes(model, max_arrays, mode="array", risk_quantile=risk_quantile)
        single_layouts = {
            "topology_greedy": place_nodes(model, max_singles, mode="single", risk_quantile=risk_quantile),
            "largest_rooms": place_nodes(
                model,
                min(6, len(model.rooms)),
                mode="single",
                risk_quantile=risk_quantile,
                method="largest_rooms",
            ),
            "farthest_rooms": place_nodes(
                model,
                min(6, len(model.rooms)),
                mode="single",
                risk_quantile=risk_quantile,
                method="farthest_rooms",
            ),
        }
        unique_single_nodes = list(
            {node.id: node for layout in single_layouts.values() for node in layout}.values()
        )
        targets = sample_target_points(
            model,
            points_per_room=points_per_room,
            seed=90_000 + floorplan_index,
            spacing_m=0.48,
        )
        for node_index, node in enumerate(array_nodes):
            placement_rows.append(_placement_row(model, node, "array4", node_index, "topology_greedy"))
        for layout_name, layout_nodes in single_layouts.items():
            for node_index, node in enumerate(layout_nodes):
                placement_rows.append(_placement_row(model, node, "single", node_index, layout_name))

        for target in targets:
            array4 = [generator.array(model, target, node, channels=4) for node in array_nodes]
            array8 = generator.array(model, target, array_nodes[0], channels=8)
            single_measurements = {
                node.id: generator.single(model, target, node) for node in unique_single_nodes
            }
            for config_name, algorithm, array_count, single_count, channels, layout_name in configurations:
                layout_nodes = single_layouts[layout_name]
                if array_count > len(array_nodes) or single_count > len(layout_nodes):
                    continue
                if config_name == "array_1x8":
                    used_arrays = array_nodes[:1]
                    used_doa = [array8]
                else:
                    used_arrays = array_nodes[:array_count]
                    used_doa = array4[:array_count]
                used_singles = layout_nodes[:single_count]
                used_toa = [single_measurements[node.id] for node in used_singles]
                estimates: list[tuple[np.ndarray, str, int]] = []
                if algorithm == "doa":
                    estimated, room_id, _ = localize_doa(model, used_arrays, used_doa, grid)
                    estimates.append((estimated, room_id, 0))
                elif algorithm.startswith("tdoa"):
                    jitter_us = 0.0
                    if algorithm == "tdoa_100us":
                        jitter_us = 100.0
                    elif algorithm == "tdoa_500us":
                        jitter_us = 500.0
                    trials = range(8) if jitter_us > 0.0 else range(1)
                    for trial in trials:
                        actual_toa = _with_clock_offsets(
                            used_toa,
                            jitter_us=jitter_us,
                            floorplan_index=floorplan_index,
                            trial=trial,
                        )
                        estimated, room_id, _ = localize_tdoa(model, used_singles, actual_toa, grid)
                        estimates.append((estimated, room_id, trial))
                elif algorithm == "sync_array":
                    array_toa = [
                        TOAMeasurement(measurement.node_id, measurement.arrival_s, measurement.confidence)
                        for measurement in used_doa
                    ]
                    estimated, room_id, _ = localize_hybrid(
                        model,
                        used_arrays,
                        used_doa,
                        used_arrays,
                        array_toa,
                        grid,
                    )
                    estimates.append((estimated, room_id, 0))
                else:
                    array_toa = [
                        TOAMeasurement(measurement.node_id, measurement.arrival_s, measurement.confidence)
                        for measurement in used_doa
                    ]
                    estimated, room_id, _ = localize_hybrid(
                        model,
                        used_arrays,
                        used_doa,
                        [*used_arrays, *used_singles],
                        [*array_toa, *used_toa],
                        grid,
                    )
                    estimates.append((estimated, room_id, 0))
                truth = np.asarray(target.position[:2], dtype=float)
                local_sensor = any(node.room_id == target.room_id for node in (*used_arrays, *used_singles))
                for estimated, room_id, trial in estimates:
                    result_rows.append(
                        {
                            "split": "test",
                            "floorplan_idx": floorplan_index,
                            "target_id": target.id,
                            "target_room": target.room_id,
                            "configuration": config_name,
                            "algorithm": algorithm,
                            "placement_method": layout_name,
                            "clock_trial": trial,
                            "channels": channels,
                            "array_nodes": array_count,
                            "single_nodes": single_count,
                            "coverage": "local_node" if local_sensor else "cross_room_only",
                            "true_x_m": round(float(truth[0]), 5),
                            "true_y_m": round(float(truth[1]), 5),
                            "estimated_x_m": round(float(estimated[0]), 5),
                            "estimated_y_m": round(float(estimated[1]), 5),
                            "position_error_m": round(float(np.linalg.norm(estimated - truth)), 5),
                            "estimated_room": room_id,
                            "room_correct": bool(room_id == target.room_id),
                        }
                    )

    summary_rows = _aggregate_results(result_rows)
    selected = _select_minimal_configuration(summary_rows)
    motion_rows, motion_summary = _run_motion_study(
        test_indices,
        resource=resource,
        generator=generator,
        risk_quantile=risk_quantile,
    )
    payload = {
        "study": "floorplan_distributed_localization",
        "method": "position-independent topology-aware minimax greedy placement",
        "train_indices": list(train_indices),
        "test_indices": list(test_indices),
        "selected_risk_quantile": risk_quantile,
        "tuning": tuning_rows,
        "quality": quality,
        "rt_accelerator": rt_accelerator,
        "rt_precision": rt_precision,
        "rt_cuda_device": int(rt_cuda_device),
        "points_per_room": points_per_room,
        "selected_configuration": selected,
        "summary": summary_rows,
        "results": result_rows,
        "placements": placement_rows,
        "motion_summary": motion_summary,
        "motion_results": motion_rows,
    }
    _write_study(output, payload)
    return payload


def _run_motion_study(
    test_indices: Sequence[int],
    *,
    resource: FloorplanResource,
    generator: AcousticMeasurementGenerator,
    risk_quantile: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trajectory_summaries = []
    for floorplan_index in test_indices:
        model = load_model(floorplan_index, resource)
        grid = localization_grid(model)
        nodes = place_nodes(
            model,
            min(6, len(model.rooms)),
            mode="single",
            risk_quantile=risk_quantile,
        )
        targets, destination_room = _portal_trajectory(model)
        raw_positions = []
        estimated_rooms = []
        for target in targets:
            measurements = [generator.single(model, target, node) for node in nodes]
            position, room_id, _ = localize_tdoa(model, nodes, measurements, grid)
            raw_positions.append(position)
            estimated_rooms.append(room_id)
        raw = np.asarray(raw_positions, dtype=float)
        filtered, velocities = _kalman_constant_velocity(raw)
        truth = np.asarray([target.position[:2] for target in targets], dtype=float)
        raw_errors = np.linalg.norm(raw - truth, axis=1)
        filtered_errors = np.linalg.norm(filtered - truth, axis=1)
        true_direction = _bearing_vector(truth[-1] - truth[0])
        estimated_direction = _bearing_vector(filtered[-1] - filtered[0])
        direction_error = angular_error_deg(estimated_direction, true_direction)
        true_transition = next(
            (index for index, target in enumerate(targets) if target.room_id == destination_room),
            len(targets) - 1,
        )
        estimated_transition = next(
            (index for index, room_id in enumerate(estimated_rooms) if room_id == destination_room),
            len(targets),
        )
        room_accuracy = float(np.mean([room_id == target.room_id for room_id, target in zip(estimated_rooms, targets)]))
        trajectory_summaries.append(
            {
                "floorplan_idx": floorplan_index,
                "frames": len(targets),
                "raw_median_error_m": float(np.median(raw_errors)),
                "filtered_median_error_m": float(np.median(filtered_errors)),
                "trend_direction_error_deg": float(direction_error),
                "room_accuracy": room_accuracy,
                "portal_transition_delay_frames": int(estimated_transition - true_transition),
            }
        )
        for frame_index, target in enumerate(targets):
            rows.append(
                {
                    "floorplan_idx": floorplan_index,
                    "frame": frame_index,
                    "target_room": target.room_id,
                    "estimated_room": estimated_rooms[frame_index],
                    "true_x_m": round(float(truth[frame_index, 0]), 5),
                    "true_y_m": round(float(truth[frame_index, 1]), 5),
                    "raw_x_m": round(float(raw[frame_index, 0]), 5),
                    "raw_y_m": round(float(raw[frame_index, 1]), 5),
                    "filtered_x_m": round(float(filtered[frame_index, 0]), 5),
                    "filtered_y_m": round(float(filtered[frame_index, 1]), 5),
                    "velocity_x_m_frame": round(float(velocities[frame_index, 0]), 5),
                    "velocity_y_m_frame": round(float(velocities[frame_index, 1]), 5),
                    "raw_error_m": round(float(raw_errors[frame_index]), 5),
                    "filtered_error_m": round(float(filtered_errors[frame_index]), 5),
                    "room_correct": bool(estimated_rooms[frame_index] == target.room_id),
                }
            )
    return rows, {
        "configuration": "single_6x1 + constant_velocity_kalman",
        "trajectory_count": len(trajectory_summaries),
        "frames": int(sum(item["frames"] for item in trajectory_summaries)),
        "raw_median_error_m": round(float(np.median([item["raw_median_error_m"] for item in trajectory_summaries])), 4),
        "filtered_median_error_m": round(float(np.median([item["filtered_median_error_m"] for item in trajectory_summaries])), 4),
        "median_trend_direction_error_deg": round(float(np.median([item["trend_direction_error_deg"] for item in trajectory_summaries])), 4),
        "room_accuracy": round(float(np.mean([item["room_accuracy"] for item in trajectory_summaries])), 4),
        "median_absolute_portal_transition_delay_frames": round(
            float(np.median([abs(item["portal_transition_delay_frames"]) for item in trajectory_summaries])),
            4,
        ),
        "per_trajectory": trajectory_summaries,
    }


def _portal_trajectory(model: FloorplanModel) -> tuple[list[TargetPoint], str]:
    candidates = []
    for portal in model.portals:
        room_ids = [str(value) for value in portal.get("room_ids", []) if str(value) in model.rooms]
        if len(room_ids) != 2:
            continue
        score = sum(float(model.rooms[room_id].get("area_m2", 0.0)) for room_id in room_ids)
        candidates.append((score, portal, room_ids))
    if not candidates:
        raise ValueError(f"FloorPlan {model.index} has no two-room portal")
    _, portal, room_ids = max(candidates, key=lambda item: item[0])
    source_room, destination_room = room_ids
    source_center = _safe_polygon(model.polygons[source_room], 0.3).representative_point()
    destination_center = _safe_polygon(model.polygons[destination_room], 0.3).representative_point()
    source_anchor = np.asarray([source_center.x, source_center.y], dtype=float)
    destination_anchor = np.asarray([destination_center.x, destination_center.y], dtype=float)
    room_points = portal.get("room_points", {})
    source_portal = np.asarray(room_points.get(source_room, portal.get("center"))[:2], dtype=float)
    destination_portal = np.asarray(room_points.get(destination_room, portal.get("center"))[:2], dtype=float)
    source_near = 0.82 * source_portal + 0.18 * source_anchor
    destination_near = 0.82 * destination_portal + 0.18 * destination_anchor
    source_mid = 0.5 * source_anchor + 0.5 * source_near
    destination_mid = 0.5 * destination_anchor + 0.5 * destination_near
    raw = (
        (source_room, source_anchor),
        (source_room, source_mid),
        (source_room, source_near),
        (destination_room, destination_near),
        (destination_room, destination_mid),
        (destination_room, destination_anchor),
    )
    targets = []
    for frame_index, (room_id, position) in enumerate(raw):
        polygon = model.polygons[room_id]
        if not polygon.buffer(1e-6).covers(Point(float(position[0]), float(position[1]))):
            fallback = _safe_polygon(polygon, 0.12).representative_point()
            position = np.asarray([fallback.x, fallback.y], dtype=float)
        targets.append(
            TargetPoint(
                f"trajectory:{model.index}:{frame_index}",
                room_id,
                (float(position[0]), float(position[1]), 1.5),
            )
        )
    return targets, destination_room


def _kalman_constant_velocity(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(observations, dtype=float)
    state = np.asarray([values[0, 0], values[0, 1], 0.0, 0.0], dtype=float)
    covariance = np.diag([0.4, 0.4, 1.0, 1.0])
    transition = np.asarray(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    observation = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    process_noise = np.diag([0.08, 0.08, 0.18, 0.18])
    measurement_noise = np.eye(2, dtype=float) * 0.55**2
    positions = []
    velocities = []
    for frame_index, measured in enumerate(values):
        if frame_index:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process_noise
        innovation = measured - observation @ state
        innovation_covariance = observation @ covariance @ observation.T + measurement_noise
        gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        state = state + gain @ innovation
        covariance = (np.eye(4) - gain @ observation) @ covariance
        positions.append(state[:2].copy())
        velocities.append(state[2:].copy())
    return np.asarray(positions), np.asarray(velocities)


def _bearing_vector(vector: Sequence[float]) -> float:
    values = np.asarray(vector, dtype=float)
    return float(math.degrees(math.atan2(float(values[1]), float(values[0]))) % 360.0)


def _deployment_score(
    model: FloorplanModel,
    nodes: Sequence[SensorNode],
    points: Sequence[TargetPoint],
    mode: str,
    risk_quantile: float,
) -> float:
    scores = []
    covered_rooms: set[str] = set()
    for point in points:
        if mode == "array":
            information = np.eye(2, dtype=float) * 1e-4
            cross_evidence = 0.0
            for node in nodes:
                _, distance, hops = model.propagation(point.position, point.room_id, node.position, node.room_id)
                if hops == 0:
                    delta = np.asarray(point.position[:2]) - np.asarray(node.position[:2])
                    radius_sq = max(float(np.dot(delta, delta)), 0.05)
                    gradient = np.asarray([-delta[1], delta[0]]) / radius_sq
                    information += np.outer(gradient, gradient) / math.radians(5.0) ** 2
                    covered_rooms.add(point.room_id)
                else:
                    cross_evidence += (0.70 ** hops) / max(distance, 1.0)
            sign, logdet = np.linalg.slogdet(information)
            scores.append(float(logdet if sign > 0 else -20.0) + 0.15 * cross_evidence)
        elif mode == "single":
            rows = []
            weights = []
            for node in nodes:
                _, _, hops = model.propagation(point.position, point.room_id, node.position, node.room_id)
                direction = model.first_leg_direction(point.position, point.room_id, node.room_id, node.position)
                rows.append([float(direction[0]), float(direction[1]), 1.0])
                weights.append(0.76 ** hops)
                if hops == 0:
                    covered_rooms.add(point.room_id)
            if len(rows) < 3:
                scores.append(-20.0 + 0.1 * len(rows))
                continue
            design = np.asarray(rows, dtype=float)
            fisher = design.T @ (np.asarray(weights)[:, None] * design) + np.eye(3) * 1e-5
            position = fisher[:2, :2] - np.outer(fisher[:2, 2], fisher[2, :2]) / max(float(fisher[2, 2]), 1e-9)
            sign, logdet = np.linalg.slogdet(position + np.eye(2) * 1e-4)
            scores.append(float(logdet if sign > 0 else -20.0))
        else:
            raise ValueError("mode must be array or single")
    room_fraction = len(covered_rooms) / max(len(model.rooms), 1)
    return float(np.quantile(np.asarray(scores), risk_quantile) + 0.5 * np.mean(scores) + 0.75 * room_fraction)


def _estimate_grid_position(grid: Sequence[TargetPoint], losses: np.ndarray) -> tuple[np.ndarray, str]:
    best_index = int(np.argmin(losses))
    best_room = grid[best_index].room_id
    room_indices = np.asarray([index for index, point in enumerate(grid) if point.room_id == best_room], dtype=int)
    ranked = room_indices[np.argsort(losses[room_indices])[: min(5, room_indices.size)]]
    weights = np.exp(-2.0 * (losses[ranked] - float(np.min(losses[ranked]))))
    positions = np.asarray([grid[int(index)].position[:2] for index in ranked], dtype=float)
    estimate = np.average(positions, axis=0, weights=np.maximum(weights, 1e-9))
    return np.asarray(estimate, dtype=float), best_room


def _safe_polygon(polygon: Polygon, margin_m: float) -> Polygon:
    inset = polygon.buffer(-float(margin_m), join_style=2)
    if inset.is_empty:
        inset = polygon.buffer(-min(0.08, float(margin_m) * 0.25), join_style=2)
    if inset.is_empty:
        inset = polygon
    if inset.geom_type == "MultiPolygon":
        inset = max(inset.geoms, key=lambda item: item.area)
    return inset


def _polygon_grid(polygon: Polygon, *, spacing_m: float, margin_m: float) -> list[tuple[float, float]]:
    domain = _safe_polygon(polygon, margin_m)
    min_x, min_y, max_x, max_y = domain.bounds
    if max_x <= min_x or max_y <= min_y:
        return []
    x_values = np.arange(min_x + spacing_m * 0.25, max_x + 1e-9, spacing_m)
    y_values = np.arange(min_y + spacing_m * 0.25, max_y + 1e-9, spacing_m)
    points = [(float(x), float(y)) for x in x_values for y in y_values if domain.covers(Point(float(x), float(y)))]
    if not points:
        point = domain.representative_point()
        points = [(float(point.x), float(point.y))]
    return points


def _huber(value: float | np.ndarray, delta: float = 1.5) -> float | np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute <= delta, 0.5 * absolute * absolute, delta * (absolute - 0.5 * delta))


def _spectrum_confidence(spectrum: np.ndarray) -> tuple[float, float]:
    values = np.asarray(spectrum, dtype=float).reshape(-1)
    peak = float(np.max(values))
    median = float(np.median(values))
    spread = max(float(np.percentile(values, 90) - np.percentile(values, 10)), 1e-9)
    confidence = float(np.clip((peak - median) / (2.5 * spread), 0.1, 1.0))
    sorted_values = np.sort(values)
    exclusion = max(1, int(round(values.size * 4.0 / 360.0)))
    peak_index = int(np.argmax(values))
    mask = np.ones(values.size, dtype=bool)
    for offset in range(-exclusion, exclusion + 1):
        mask[(peak_index + offset) % values.size] = False
    second = float(np.max(values[mask])) if np.any(mask) else float(sorted_values[-2])
    peak_ratio = float((peak - median) / max(second - median, 1e-9))
    return confidence, peak_ratio


def _placement_row(
    model: FloorplanModel,
    node: SensorNode,
    kind: str,
    order: int,
    method: str,
) -> dict[str, Any]:
    return {
        "floorplan_idx": model.index,
        "kind": kind,
        "method": method,
        "order": order + 1,
        "node_id": node.id,
        "room_id": node.room_id,
        "x_m": round(node.position[0], 5),
        "y_m": round(node.position[1], 5),
        "z_m": round(node.position[2], 5),
    }


def _aggregate_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    configurations = list(dict.fromkeys(str(row["configuration"]) for row in rows))
    for configuration in configurations:
        subset = [row for row in rows if row["configuration"] == configuration]
        errors = np.asarray([float(row["position_error_m"]) for row in subset], dtype=float)
        cross = [row for row in subset if row["coverage"] == "cross_room_only"]
        cross_errors = np.asarray([float(row["position_error_m"]) for row in cross], dtype=float)
        summaries.append(
            {
                "configuration": configuration,
                "algorithm": subset[0]["algorithm"],
                "placement_method": subset[0]["placement_method"],
                "channels": int(subset[0]["channels"]),
                "array_nodes": int(subset[0]["array_nodes"]),
                "single_nodes": int(subset[0]["single_nodes"]),
                "cases": len(subset),
                "median_error_m": round(float(np.median(errors)), 4),
                "p90_error_m": round(float(np.percentile(errors, 90)), 4),
                "room_accuracy": round(float(np.mean([bool(row["room_correct"]) for row in subset])), 4),
                "cross_room_cases": len(cross),
                "cross_room_median_error_m": None if not cross else round(float(np.median(cross_errors)), 4),
                "cross_room_room_accuracy": None if not cross else round(float(np.mean([bool(row["room_correct"]) for row in cross])), 4),
            }
        )
    return summaries


def _with_clock_offsets(
    measurements: Sequence[TOAMeasurement],
    *,
    jitter_us: float,
    floorplan_index: int,
    trial: int,
) -> list[TOAMeasurement]:
    if jitter_us <= 0.0:
        return list(measurements)
    adjusted = []
    for measurement in measurements:
        digest = hashlib.sha256(
            f"{floorplan_index}:{measurement.node_id}:{jitter_us}:{trial}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "little")
        offset_s = float(np.random.default_rng(seed).normal(0.0, float(jitter_us) * 1e-6))
        adjusted.append(
            TOAMeasurement(
                measurement.node_id,
                measurement.arrival_s + offset_s,
                measurement.confidence,
            )
        )
    return adjusted


def _select_minimal_configuration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = [
        row
        for row in rows
        if str(row.get("placement_method")) == "topology_greedy"
        and str(row.get("algorithm")) in {"doa", "tdoa", "sync_array", "hybrid"}
    ]
    feasible = [
        row
        for row in primary
        if float(row["median_error_m"]) <= 1.0
        and float(row["p90_error_m"]) <= 2.0
        and float(row["room_accuracy"]) >= 0.85
    ]
    if feasible:
        return dict(min(feasible, key=lambda row: (int(row["channels"]), float(row["median_error_m"]))))
    return dict(
        min(
            primary,
            key=lambda row: (
                float(row["median_error_m"]) + 0.5 * float(row["p90_error_m"]) + 2.0 * (1.0 - float(row["room_accuracy"])),
                int(row["channels"]),
            ),
        )
    )


def _write_study(output: Path, payload: Mapping[str, Any]) -> None:
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output / "results.csv", payload["results"])
    _write_csv(output / "summary.csv", payload["summary"])
    _write_csv(output / "placements.csv", payload["placements"])
    _write_csv(output / "tuning.csv", payload["tuning"])
    _write_csv(output / "motion_results.csv", payload["motion_results"])
    lines = [
        "# FloorPlan Distributed Localization Study",
        "",
        f"- Training FloorPlans: `{payload['train_indices']}`",
        f"- Unseen test FloorPlans: `{payload['test_indices']}`",
        f"- Selected minimax risk quantile: `{payload['selected_risk_quantile']}`",
        f"- RIR quality: `{payload['quality']}`",
        f"- Reflection tracer: `{payload['rt_accelerator']}` / `{payload['rt_precision']}` / device `{payload['rt_cuda_device']}`",
        "- Placement uses only room polygons, room areas, and portal topology. Test target positions are never used.",
        "- Array nodes use SRP-PHAT DOA. Synchronized single microphones use onset TDOA. Hybrid fusion uses both.",
        "",
        "## Static localization",
        "",
        "| Configuration | Channels | Median | P90 | Room accuracy | Cross-room-only median | Cross-room room accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        cross_error = "n/a" if row["cross_room_median_error_m"] is None else f"{row['cross_room_median_error_m']:.2f} m"
        cross_room = "n/a" if row["cross_room_room_accuracy"] is None else f"{100.0 * row['cross_room_room_accuracy']:.1f}%"
        lines.append(
            f"| {row['configuration']} | {row['channels']} | {row['median_error_m']:.2f} m | "
            f"{row['p90_error_m']:.2f} m | {100.0 * row['room_accuracy']:.1f}% | {cross_error} | {cross_room} |"
        )
    chosen = payload["selected_configuration"]
    motion = payload["motion_summary"]
    lines.extend(
        [
            "",
            "## Selected configuration",
            "",
            f"The study selector chose **{chosen['configuration']}** with {chosen['channels']} total channels.",
            "The selector first enforces median <= 1.0 m, P90 <= 2.0 m, and room accuracy >= 85%; "
            "if no configuration satisfies all thresholds, it minimizes a fixed accuracy/cost penalty.",
            "",
            "## Motion and portal crossing",
            "",
            f"The selected six-single-microphone deployment was evaluated on {motion['trajectory_count']} unseen "
            f"door-crossing trajectories ({motion['frames']} frames). A constant-velocity Kalman filter was applied "
            "after independent frame localization.",
            "",
            "| Raw median | Filtered median | Trend direction error | Room accuracy | Portal transition delay |",
            "|---:|---:|---:|---:|---:|",
            f"| {motion['raw_median_error_m']:.2f} m | {motion['filtered_median_error_m']:.2f} m | "
            f"{motion['median_trend_direction_error_deg']:.1f} deg | {100.0 * motion['room_accuracy']:.1f}% | "
            f"{motion['median_absolute_portal_transition_delay_frames']:.1f} frames |",
            "",
            "## Interpretation limits",
            "",
            "This is a simulation study with known sensor coordinates and synchronized single microphones. "
            "The matched FloorPlan and portal graph are available to the estimator. Clock drift, sensor position error, "
            "multiple simultaneous speakers, and simulation-to-real mismatch remain separate experiments.",
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
