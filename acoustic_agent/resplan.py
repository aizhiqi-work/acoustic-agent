from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
import math
import pickle
import secrets
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry


DEFAULT_RESPLAN_PATH = Path(__file__).resolve().parents[2] / "ResPlan.pkl"
_ROOM_TYPES = {"living", "kitchen", "bedroom", "bathroom", "storage", "balcony"}
_INTERIOR_ROOM_TYPES = _ROOM_TYPES - {"balcony"}
_OUTDOOR_LAYERS = ("balcony", "garden", "veranda", "parking", "pool")
_ROOM_AREA_LIMITS = {
    "living": (6.0, 120.0),
    "kitchen": (2.0, 40.0),
    "bedroom": (4.0, 60.0),
    "bathroom": (1.0, 25.0),
    "storage": (0.5, 30.0),
    "balcony": (0.5, 40.0),
}
_ROOM_MIN_WIDTH_M = {
    "living": 1.5,
    "kitchen": 0.8,
    "bedroom": 1.2,
    "bathroom": 0.5,
    "storage": 0.4,
    "balcony": 0.3,
}
_ALLOWED_PICKLE_GLOBALS = {
    ("shapely.io", "from_wkb"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "dtype"),
    ("networkx.classes.graph", "Graph"),
    ("networkx.classes.reportviews", "NodeView"),
}


class _ResPlanUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"unsupported ResPlan pickle global: {module}.{name}")
        return super().find_class(module, name)


class ResPlanDataset:
    def __init__(self, path: str | Path = DEFAULT_RESPLAN_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"ResPlan dataset not found: {self.path}")
        with self.path.open("rb") as handle:
            records = _ResPlanUnpickler(handle).load()
        if not isinstance(records, list):
            raise ValueError("ResPlan dataset must contain a list of floor plans")
        self.records: list[Mapping[str, Any]] = records
        self.profiles = [_plan_profile(record, index=index) for index, record in enumerate(self.records)]
        self.eligible_indices = [profile["index"] for profile in self.profiles if profile["eligible"]]
        if not self.eligible_indices:
            raise ValueError("ResPlan dataset contains no eligible single-floor scenes")
        self._stats = _dataset_stats(self.records, self.profiles)

    def __len__(self) -> int:
        return len(self.records)

    def scene(self, index: int, room_id: str | None = None, *, height_m: float = 2.8) -> dict[str, Any]:
        if index < 0 or index >= len(self.records):
            raise IndexError(f"ResPlan index must be between 0 and {max(0, len(self.records) - 1)}")
        profile = self.profiles[index]
        if not profile["eligible"]:
            reasons = ", ".join(profile["filter_reasons"])
            raise ValueError(f"ResPlan[{index}] is filtered: {reasons}")
        scene = scene_from_record(
            self.records[index],
            index=index,
            room_id=room_id,
            height_m=height_m,
            profile=profile,
        )
        scene["dataset"]["count"] = len(self.records)
        scene["dataset"]["eligible_count"] = len(self.eligible_indices)
        scene["dataset"]["filtered_count"] = len(self.records) - len(self.eligible_indices)
        return scene

    def resolve_index(self, index: int, direction: str = "nearest") -> int:
        direction = str(direction).lower()
        if direction == "random":
            return int(secrets.choice(self.eligible_indices))
        bounded = min(max(int(index), 0), len(self.records) - 1)
        if direction == "next":
            position = bisect_right(self.eligible_indices, bounded)
            return int(self.eligible_indices[position] if position < len(self.eligible_indices) else self.eligible_indices[-1])
        if direction == "previous":
            position = bisect_left(self.eligible_indices, bounded) - 1
            return int(self.eligible_indices[position] if position >= 0 else self.eligible_indices[0])
        if direction != "nearest":
            raise ValueError("direction must be nearest, next, previous, or random")
        position = bisect_left(self.eligible_indices, bounded)
        if position < len(self.eligible_indices) and self.eligible_indices[position] == bounded:
            return bounded
        if position >= len(self.eligible_indices):
            return int(self.eligible_indices[-1])
        if position == 0:
            return int(self.eligible_indices[0])
        before = self.eligible_indices[position - 1]
        after = self.eligible_indices[position]
        return int(before if bounded - before <= after - bounded else after)

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)


def scene_from_record(
    record: Mapping[str, Any],
    *,
    index: int,
    room_id: str | None = None,
    height_m: float = 2.8,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    graph = record.get("graph")
    if not isinstance(graph, nx.Graph):
        raise ValueError(f"ResPlan[{index}] has no valid room graph")
    profile = dict(profile or _plan_profile(record, index=index))
    scale = float(profile["meters_per_unit"])
    all_rooms = _room_records(graph, scale=scale)
    valid_room_ids = set(profile["valid_room_ids"])
    rooms = [room for room in all_rooms if room["id"] in valid_room_ids]
    if not rooms:
        raise ValueError(f"ResPlan[{index}] contains no valid supported rooms")
    selected = _select_room(rooms, room_id)
    selected_raw = selected["geometry"]
    room_min_x, room_min_y, room_max_x, room_max_y = selected_raw.bounds
    selected_metric = _metric_geometry(selected_raw, scale, room_min_x, room_min_y)
    selected_polygon = _largest_polygon(selected_metric)
    if selected_polygon is None or selected_polygon.area <= 1e-6:
        raise ValueError(f"ResPlan[{index}] room {selected['id']!r} has invalid geometry")

    corners = _polygon_coordinates(selected_polygon)
    width = max(room_max_x - room_min_x, 1e-6) * scale
    depth = max(room_max_y - room_min_y, 1e-6) * scale
    features = _boundary_features(
        record,
        selected_raw,
        rooms=all_rooms,
        scale=scale,
        origin=(room_min_x, room_min_y),
    )
    surface_segments = _surface_segments(
        selected_raw,
        features,
        scale=scale,
        origin=(room_min_x, room_min_y),
    )
    source, receiver = _interior_pair(selected_polygon, float(height_m))
    plan = _plan_overview(record, rooms, selected["id"], scale)
    connections = _selected_connections(record, graph, all_rooms, selected)
    exterior_exposures = [
        {
            "feature_index": feature_index,
            "type": feature["type"],
            "connection": feature["connection"],
        }
        for feature_index, feature in enumerate(features)
        if str(feature.get("connection", "")).startswith("outdoor")
    ]

    return {
        "dataset": {
            "index": int(index),
            "sample_id": record.get("id"),
            "count": None,
            "unit_type": str(record.get("unitType", "Unknown")),
            "net_area_m2": _finite_float(record.get("net_area")),
            "gross_area_m2": _finite_float(record.get("area")),
            "meters_per_unit": float(scale),
            "scale_source": profile["scale_source"],
            "wall_depth_m": float(record.get("wall_depth", 0.0)) * scale,
            "eligible": bool(profile["eligible"]),
            "filter_reasons": list(profile["filter_reasons"]),
            "dropped_duplicate_rooms": int(profile["dropped_duplicate_rooms"]),
            "dropped_invalid_rooms": int(profile["dropped_invalid_rooms"]),
        },
        "rooms": [
            {
                "id": room["id"],
                "type": room["type"],
                "area_m2": float(room["geometry"].area * scale * scale),
            }
            for room in rooms
        ],
        "selected_room": {
            "id": selected["id"],
            "type": selected["type"],
            "area_m2": float(selected_raw.area * scale * scale),
            "connections": connections,
            "exterior_exposures": exterior_exposures,
        },
        "room": {
            "shape": "resplan",
            "size": [float(width), float(depth), float(height_m)],
            "corners": corners,
            "metadata": {
                "shape": "resplan",
                "geometry_model": "resplan_room_extrusion",
                "opening_model": "full_height_equivalent_boundary_material_v1",
                "connectivity_model": "resplan_graph_plus_wall_boolean_v1",
                "connections": connections,
                "exterior_exposures": exterior_exposures,
                "resplan": {
                    "index": int(index),
                    "sample_id": record.get("id"),
                    "room_id": selected["id"],
                    "room_type": selected["type"],
                    "meters_per_unit": float(scale),
                },
                "boundary_features": [_public_feature(feature) for feature in features],
                "surface_segments": surface_segments,
            },
        },
        "source": source,
        "receiver": receiver,
        "plan": plan,
    }


def _room_records(graph: nx.Graph, *, scale: float | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node_id, attributes in graph.nodes(data=True):
        if not isinstance(attributes, Mapping):
            continue
        room_type = str(attributes.get("type", "")).lower()
        geometry = _largest_polygon(attributes.get("geometry"))
        if room_type not in _ROOM_TYPES or geometry is None or geometry.is_empty:
            continue
        candidates.append({"id": str(node_id), "type": room_type, "geometry": geometry, "aliases": [str(node_id)]})
    candidates.sort(key=lambda room: (_room_type_rank(room["type"]), -float(room["geometry"].area), room["id"]))
    rooms: list[dict[str, Any]] = []
    for candidate in candidates:
        duplicate = next(
            (
                room for room in rooms
                if room["type"] == candidate["type"] and _near_duplicate(room["geometry"], candidate["geometry"])
            ),
            None,
        )
        if duplicate is not None:
            duplicate["aliases"].append(candidate["id"])
            continue
        candidate["issues"] = _room_quality_issues(candidate, scale) if scale is not None else []
        rooms.append(candidate)
    rooms.sort(key=lambda room: (_room_type_rank(room["type"]), -float(room["geometry"].area), room["id"]))
    return rooms


def _room_type_rank(room_type: str) -> int:
    order = ("living", "bedroom", "kitchen", "bathroom", "storage", "balcony")
    return order.index(room_type) if room_type in order else len(order)


def _select_room(rooms: Sequence[dict[str, Any]], room_id: str | None) -> dict[str, Any]:
    if room_id is not None:
        for room in rooms:
            if room["id"] == room_id:
                return room
        raise KeyError(f"unknown room {room_id!r}")
    return rooms[0]


def _meters_per_unit(record: Mapping[str, Any]) -> float:
    return _metric_scale(record)[0]


def _metric_scale(record: Mapping[str, Any]) -> tuple[float, str]:
    inner = record.get("inner")
    net_area = _finite_float(record.get("net_area"))
    gross_area = _finite_float(record.get("area"))
    wall_depth = max(float(record.get("wall_depth", 0.0)), 0.0)
    if not isinstance(inner, BaseGeometry) or inner.is_empty or inner.area <= 1e-9:
        return ((0.2 / wall_depth), "wall_depth_fallback") if wall_depth > 0.0 else (0.05, "constant_fallback")
    span = max(inner.bounds[2] - inner.bounds[0], inner.bounds[3] - inner.bounds[1])

    def plausible(candidate: float) -> bool:
        return (
            math.isfinite(candidate)
            and candidate > 0.0
            and 0.08 <= wall_depth * candidate <= 0.40
            and 3.0 <= span * candidate <= 80.0
        )

    ratio = net_area / gross_area if net_area is not None and gross_area is not None and gross_area > 0.0 else 0.0
    net_scale = math.sqrt(net_area / float(inner.area)) if net_area is not None and net_area > 0.0 else 0.0
    if 0.2 <= ratio <= 1.15 and plausible(net_scale):
        return float(net_scale), "net_area"
    if gross_area is not None and gross_area > 0.0:
        gross_scale = math.sqrt((0.75 * gross_area) / float(inner.area))
        if plausible(gross_scale):
            return float(gross_scale), "gross_area_proxy"
    if wall_depth > 0.0:
        return 0.2 / wall_depth, "wall_depth_fallback"
    return 0.05, "constant_fallback"


def _plan_profile(record: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    graph = record.get("graph")
    scale, scale_source = _metric_scale(record)
    if not isinstance(graph, nx.Graph):
        return {
            "index": int(index),
            "eligible": False,
            "filter_reasons": ["invalid_graph"],
            "meters_per_unit": float(scale),
            "scale_source": scale_source,
            "valid_room_ids": [],
            "raw_room_count": 0,
            "deduplicated_room_count": 0,
            "dropped_duplicate_rooms": 0,
            "dropped_invalid_rooms": 0,
            "room_issue_counts": {},
        }
    raw_room_count = sum(
        1 for _, attributes in graph.nodes(data=True)
        if str(attributes.get("type", "")).lower() in _ROOM_TYPES
    )
    rooms = _room_records(graph, scale=scale)
    valid_rooms = [room for room in rooms if not room["issues"]]
    reasons: list[str] = []
    room_issue_counts = Counter(issue for room in rooms for issue in room["issues"])
    stair = record.get("stair")
    if isinstance(stair, BaseGeometry) and not stair.is_empty:
        reasons.append("stair_or_multilevel")
    inner = record.get("inner")
    inner_parts = len(inner.geoms) if isinstance(inner, BaseGeometry) and hasattr(inner, "geoms") else 1
    if inner_parts > 1:
        reasons.append("disconnected_inner_geometry")
    if len(valid_rooms) > 14:
        reasons.append("too_many_rooms")
    if not any(room["type"] == "living" for room in valid_rooms):
        reasons.append("no_valid_living_room")
    if _rooms_overlap(valid_rooms, scale):
        reasons.append("overlapping_rooms")
    return {
        "index": int(index),
        "eligible": not reasons,
        "filter_reasons": reasons,
        "meters_per_unit": float(scale),
        "scale_source": scale_source,
        "valid_room_ids": [room["id"] for room in valid_rooms],
        "raw_room_count": int(raw_room_count),
        "deduplicated_room_count": len(rooms),
        "dropped_duplicate_rooms": max(0, raw_room_count - len(rooms)),
        "dropped_invalid_rooms": max(0, len(rooms) - len(valid_rooms)),
        "room_issue_counts": dict(room_issue_counts),
    }


def _dataset_stats(records: Sequence[Mapping[str, Any]], profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    filter_reasons = Counter()
    scale_sources = Counter()
    room_issues = Counter()
    edge_types = Counter()
    outdoor_layers = Counter()
    duplicate_plans = 0
    invalid_room_plans = 0
    raw_rooms = deduplicated_rooms = valid_rooms = 0
    for record, profile in zip(records, profiles):
        filter_reasons.update(profile["filter_reasons"])
        scale_sources[profile["scale_source"]] += 1
        room_issues.update(profile["room_issue_counts"])
        raw_rooms += int(profile["raw_room_count"])
        deduplicated_rooms += int(profile["deduplicated_room_count"])
        valid_rooms += int(profile["deduplicated_room_count"]) - int(profile["dropped_invalid_rooms"])
        duplicate_plans += int(profile["dropped_duplicate_rooms"] > 0)
        invalid_room_plans += int(profile["dropped_invalid_rooms"] > 0)
        graph = record.get("graph")
        if isinstance(graph, nx.Graph):
            edge_types.update(str(attributes.get("type", "unknown")) for _, _, attributes in graph.edges(data=True))
        for layer in _OUTDOOR_LAYERS:
            geometry = record.get(layer)
            if isinstance(geometry, BaseGeometry) and not geometry.is_empty:
                outdoor_layers[layer] += 1
    eligible = sum(bool(profile["eligible"]) for profile in profiles)
    return {
        "records": len(records),
        "eligible_records": int(eligible),
        "filtered_records": int(len(records) - eligible),
        "filter_reasons": dict(filter_reasons),
        "scale_sources": dict(scale_sources),
        "rooms": {
            "raw": int(raw_rooms),
            "deduplicated": int(deduplicated_rooms),
            "valid": int(valid_rooms),
            "dropped_duplicates": int(raw_rooms - deduplicated_rooms),
            "dropped_invalid": int(deduplicated_rooms - valid_rooms),
            "plans_with_duplicates": int(duplicate_plans),
            "plans_with_invalid_rooms": int(invalid_room_plans),
            "quality_issues": dict(room_issues),
        },
        "legacy_edge_types": dict(edge_types),
        "outdoor_layer_records": dict(outdoor_layers),
    }


def _near_duplicate(first: Polygon, second: Polygon) -> bool:
    if first.wkb == second.wkb or first.equals(second):
        return True
    intersection = first.intersection(second).area
    return intersection > 0.0 and intersection / max(first.union(second).area, 1e-9) >= 0.95


def _room_quality_issues(room: Mapping[str, Any], scale: float | None) -> list[str]:
    if scale is None:
        return []
    geometry = room["geometry"]
    room_type = room["type"]
    area_m2 = float(geometry.area * scale * scale)
    min_x, min_y, max_x, max_y = geometry.bounds
    width_m = (max_x - min_x) * scale
    depth_m = (max_y - min_y) * scale
    min_width_m = min(width_m, depth_m)
    aspect = max(width_m, depth_m) / max(min_width_m, 1e-9)
    issues: list[str] = []
    min_area, max_area = _ROOM_AREA_LIMITS[room_type]
    if not min_area <= area_m2 <= max_area:
        issues.append("implausible_area")
    if min_width_m < _ROOM_MIN_WIDTH_M[room_type]:
        issues.append("too_narrow")
    if len(geometry.exterior.coords) - 1 > 64:
        issues.append("too_many_vertices")
    if geometry.interiors:
        issues.append("interior_holes")
    if aspect > 10.0:
        issues.append("extreme_aspect_ratio")
    return issues


def _rooms_overlap(rooms: Sequence[Mapping[str, Any]], scale: float) -> bool:
    for index, first in enumerate(rooms):
        for second in rooms[index + 1:]:
            if first["geometry"].intersection(second["geometry"]).area * scale * scale > 0.05:
                return True
    return False


def _selected_connections(
    record: Mapping[str, Any],
    graph: nx.Graph,
    rooms: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    alias_map = {alias: room for room in rooms for alias in room.get("aliases", [room["id"]])}
    selected_aliases = set(selected.get("aliases", [selected["id"]]))
    connections: dict[tuple[str, str], dict[str, Any]] = {}
    connected_room_ids: set[str] = set()
    for source, target, attributes in graph.edges(data=True):
        source_id, target_id = str(source), str(target)
        if source_id not in selected_aliases and target_id not in selected_aliases:
            continue
        other_id = target_id if source_id in selected_aliases else source_id
        other = alias_map.get(other_id)
        other_attributes = graph.nodes[other_id]
        target_room_id = other["id"] if other is not None else other_id
        target_type = other["type"] if other is not None else str(other_attributes.get("type", "unknown"))
        if target_room_id == selected["id"]:
            continue
        edge_type = str(attributes.get("type", "adjacency"))
        normalized = edge_type
        verified = True
        if edge_type == "adjacency" and other is not None:
            verified = _is_open_passage(record, selected["geometry"], other["geometry"])
            normalized = "via_opening" if verified else "blocked_adjacency"
        outdoor = target_type in {"balcony", "front_door", "garden", "veranda"}
        item = {
            "type": normalized,
            "legacy_type": edge_type,
            "target_room_id": target_room_id,
            "target_type": target_type,
            "walkable": normalized in {"via_door", "via_opening", "direct"},
            "outdoor": outdoor,
            "verified": bool(verified),
        }
        connections[(target_room_id, normalized)] = item
        if other is not None:
            connected_room_ids.add(other["id"])
    for other in rooms:
        if other["id"] == selected["id"] or other["id"] in connected_room_ids:
            continue
        if _is_open_passage(record, selected["geometry"], other["geometry"]):
            connections[(other["id"], "via_opening_detected")] = {
                "type": "via_opening_detected",
                "legacy_type": None,
                "target_room_id": other["id"],
                "target_type": other["type"],
                "walkable": True,
                "outdoor": other["type"] == "balcony",
                "verified": True,
            }
    return sorted(connections.values(), key=lambda item: (not item["outdoor"], item["target_type"], item["target_room_id"], item["type"]))


def _is_open_passage(record: Mapping[str, Any], first: Polygon, second: Polygon) -> bool:
    wall_depth = max(float(record.get("wall_depth", 0.0)), 1.0)
    minimum_length = 2.0 * wall_depth
    walls = record.get("wall")
    try:
        gap = first.buffer(wall_depth * 0.55).intersection(second.buffer(wall_depth * 0.55))
        if isinstance(walls, BaseGeometry) and not walls.is_empty:
            gap = gap.difference(walls)
        for piece in (list(gap.geoms) if hasattr(gap, "geoms") else [gap]):
            if piece.is_empty:
                continue
            min_x, min_y, max_x, max_y = piece.bounds
            if max(max_x - min_x, max_y - min_y) >= minimum_length and piece.area >= wall_depth * wall_depth * 0.25:
                return True
    except Exception:
        pass
    try:
        contact = first.boundary.intersection(second.buffer(wall_depth * 0.6))
        if contact.is_empty or contact.length < minimum_length:
            return False
        if isinstance(walls, BaseGeometry) and not walls.is_empty:
            contact = contact.difference(walls.buffer(0.5))
        lengths = [part.length for part in contact.geoms] if hasattr(contact, "geoms") else [contact.length]
        return max(lengths, default=0.0) >= minimum_length
    except Exception:
        return False


def _boundary_features(
    record: Mapping[str, Any],
    room: Polygon,
    *,
    rooms: Sequence[Mapping[str, Any]],
    scale: float,
    origin: tuple[float, float],
) -> list[dict[str, Any]]:
    wall_depth = max(float(record.get("wall_depth", 0.0)), 1.0)
    proximity = max(wall_depth * 1.1, 1.25)
    features: list[dict[str, Any]] = []
    raw_layers = (
        ("door", record.get("front_door"), "outdoor_entry"),
        ("door", record.get("door"), None),
        ("window", record.get("window"), None),
    )
    seen: list[Polygon] = []
    for kind, layer, forced_connection in raw_layers:
        for polygon in _iter_polygons(layer):
            if any(_near_duplicate(polygon, previous) for previous in seen):
                continue
            if polygon.distance(room.boundary) > proximity:
                continue
            contact = room.boundary.intersection(polygon.buffer(max(wall_depth * 0.28, 0.4), cap_style=2, join_style=2))
            raw_segments = _line_segments(contact)
            if sum(LineString(segment).length for segment in raw_segments) <= 0.25:
                continue
            seen.append(polygon)
            metric_polygon = _metric_geometry(polygon, scale, *origin)
            metric_segments = [_metric_segment(segment, scale, origin) for segment in raw_segments]
            connection = forced_connection or _connector_connection(kind, polygon, room, rooms, proximity)
            features.append({
                "type": kind,
                "raw_geometry": polygon,
                "raw_zone": polygon.buffer(max(wall_depth * 0.28, 0.4), cap_style=2, join_style=2),
                "polygon": _polygon_coordinates(_largest_polygon(metric_polygon)),
                "segments": metric_segments,
                "sill_height_m": 0.0 if kind == "door" else 0.9,
                "height_m": 2.1 if kind == "door" else 1.2,
                "connection": connection,
            })
    return features


def _connector_connection(
    kind: str,
    connector: Polygon,
    selected_room: Polygon,
    rooms: Sequence[Mapping[str, Any]],
    proximity: float,
) -> str:
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    for room in rooms:
        geometry = room["geometry"]
        if geometry.equals(selected_room) or connector.distance(geometry) > proximity:
            continue
        score = connector.boundary.intersection(geometry.buffer(proximity * 0.55)).length
        if score > 0.0:
            candidates.append((float(score), room))
    if candidates:
        target = max(candidates, key=lambda item: item[0])[1]
        return "outdoor_balcony" if target["type"] == "balcony" else "interior_room"
    return "outdoor_facade" if kind == "window" else "outdoor_or_unmatched"


def _surface_segments(
    room: Polygon,
    features: Sequence[dict[str, Any]],
    *,
    scale: float,
    origin: tuple[float, float],
) -> list[dict[str, Any]]:
    remaining: BaseGeometry = room.boundary
    segments: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        contact = remaining.intersection(feature["raw_zone"])
        for segment in _line_segments(contact):
            metric = _metric_segment(segment, scale, origin)
            if _segment_length(metric) < 0.015:
                continue
            segments.append({
                "a": metric[0],
                "b": metric[1],
                "type": feature["type"],
                "feature_index": feature_index,
            })
        remaining = remaining.difference(feature["raw_zone"])
    for segment in _line_segments(remaining):
        metric = _metric_segment(segment, scale, origin)
        if _segment_length(metric) < 0.015:
            continue
        segments.append({"a": metric[0], "b": metric[1], "type": "wall"})
    return segments


def _plan_overview(
    record: Mapping[str, Any],
    rooms: Sequence[dict[str, Any]],
    selected_room_id: str,
    scale: float,
) -> dict[str, Any]:
    inner = _largest_polygon(record.get("inner"))
    if inner is None:
        inner = rooms[0]["geometry"]
    min_x, min_y, max_x, max_y = inner.bounds
    room_payload = []
    for room in rooms:
        geometry = _metric_geometry(room["geometry"], scale, min_x, min_y)
        polygon = _largest_polygon(geometry)
        if polygon is None:
            continue
        room_payload.append({
            "id": room["id"],
            "type": room["type"],
            "selected": room["id"] == selected_room_id,
            "polygon": _polygon_coordinates(polygon),
        })
    apertures = []
    for kind, layer in (("door", record.get("door")), ("window", record.get("window")), ("front_door", record.get("front_door"))):
        for polygon in _iter_polygons(layer):
            metric = _largest_polygon(_metric_geometry(polygon, scale, min_x, min_y))
            if metric is not None:
                apertures.append({"type": kind, "polygon": _polygon_coordinates(metric)})
    return {
        "size": [float((max_x - min_x) * scale), float((max_y - min_y) * scale)],
        "rooms": room_payload,
        "apertures": apertures,
    }


def _interior_pair(room: Polygon, height_m: float) -> tuple[list[float], list[float]]:
    min_x, min_y, max_x, max_y = room.bounds
    margin = min(0.3, max(0.08, min(max_x - min_x, max_y - min_y) * 0.08))
    candidates: list[Point] = [room.representative_point()]
    for yi in range(1, 10):
        for xi in range(1, 10):
            point = Point(min_x + (max_x - min_x) * xi / 10.0, min_y + (max_y - min_y) * yi / 10.0)
            if room.covers(point) and room.boundary.distance(point) >= margin:
                candidates.append(point)
    if len(candidates) == 1:
        candidates.append(candidates[0])
    first, second = max(
        ((a, b) for index, a in enumerate(candidates) for b in candidates[index + 1:]),
        key=lambda pair: pair[0].distance(pair[1]),
    )
    z = min(1.4, max(0.1, height_m - 0.2))
    return (
        [round(float(first.x), 3), round(float(first.y), 3), round(float(z), 3)],
        [round(float(second.x), 3), round(float(second.y), 3), round(float(z), 3)],
    )


def _metric_geometry(geometry: BaseGeometry, scale: float, origin_x: float, origin_y: float) -> BaseGeometry:
    return affinity.affine_transform(geometry, [scale, 0.0, 0.0, scale, -origin_x * scale, -origin_y * scale])


def _metric_segment(
    segment: tuple[tuple[float, float], tuple[float, float]],
    scale: float,
    origin: tuple[float, float],
) -> list[list[float]]:
    return [
        [round((float(point[0]) - origin[0]) * scale, 6), round((float(point[1]) - origin[1]) * scale, 6)]
        for point in segment
    ]


def _largest_polygon(geometry: Any) -> Polygon | None:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, BaseGeometry):
        polygons = list(_iter_polygons(geometry))
        return max(polygons, key=lambda item: item.area) if polygons else None
    return None


def _iter_polygons(geometry: Any) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        if not geometry.is_empty:
            yield geometry
        return
    if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        return
    for part in getattr(geometry, "geoms", ()):
        yield from _iter_polygons(part)


def _line_segments(geometry: BaseGeometry) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if geometry.is_empty:
        return segments
    if geometry.geom_type in {"LineString", "LinearRing"}:
        coordinates = list(geometry.coords)
        segments.extend((coordinates[index], coordinates[index + 1]) for index in range(len(coordinates) - 1))
        return segments
    for part in getattr(geometry, "geoms", ()):
        segments.extend(_line_segments(part))
    return segments


def _polygon_coordinates(polygon: Polygon | None) -> list[list[float]]:
    if polygon is None:
        return []
    coordinates: list[list[float]] = []
    for x, y in list(polygon.exterior.coords)[:-1]:
        point = [round(float(x), 6), round(float(y), 6)]
        if not coordinates or _segment_length([coordinates[-1], point]) > 1e-7:
            coordinates.append(point)
    return coordinates


def _public_feature(feature: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": feature["type"],
        "polygon": feature["polygon"],
        "segments": feature["segments"],
        "sill_height_m": feature["sill_height_m"],
        "height_m": feature["height_m"],
        "connection": feature.get("connection", "unknown"),
    }


def _segment_length(segment: Sequence[Sequence[float]]) -> float:
    return float(math.hypot(float(segment[1][0]) - float(segment[0][0]), float(segment[1][1]) - float(segment[0][1])))


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
