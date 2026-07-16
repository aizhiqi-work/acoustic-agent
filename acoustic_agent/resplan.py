from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry


DEFAULT_RESPLAN_PATH = Path(__file__).resolve().parents[2] / "ResPlan.pkl"
_ROOM_TYPES = {"living", "kitchen", "bedroom", "bathroom", "storage", "balcony"}
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

    def __len__(self) -> int:
        return len(self.records)

    def scene(self, index: int, room_id: str | None = None, *, height_m: float = 2.8) -> dict[str, Any]:
        if index < 0 or index >= len(self.records):
            raise IndexError(f"ResPlan index must be between 0 and {max(0, len(self.records) - 1)}")
        scene = scene_from_record(self.records[index], index=index, room_id=room_id, height_m=height_m)
        scene["dataset"]["count"] = len(self.records)
        return scene


def scene_from_record(
    record: Mapping[str, Any],
    *,
    index: int,
    room_id: str | None = None,
    height_m: float = 2.8,
) -> dict[str, Any]:
    graph = record.get("graph")
    if not isinstance(graph, nx.Graph):
        raise ValueError(f"ResPlan[{index}] has no valid room graph")
    rooms = _room_records(graph)
    if not rooms:
        raise ValueError(f"ResPlan[{index}] contains no supported rooms")
    selected = _select_room(rooms, room_id)
    selected_raw = selected["geometry"]
    scale = _meters_per_unit(record)
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

    return {
        "dataset": {
            "index": int(index),
            "sample_id": record.get("id"),
            "count": None,
            "unit_type": str(record.get("unitType", "Unknown")),
            "net_area_m2": _finite_float(record.get("net_area")),
            "gross_area_m2": _finite_float(record.get("area")),
            "meters_per_unit": float(scale),
            "wall_depth_m": float(record.get("wall_depth", 0.0)) * scale,
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
        },
        "room": {
            "shape": "resplan",
            "size": [float(width), float(depth), float(height_m)],
            "corners": corners,
            "metadata": {
                "shape": "resplan",
                "geometry_model": "resplan_room_extrusion",
                "opening_model": "full_height_equivalent_boundary_material_v1",
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


def _room_records(graph: nx.Graph) -> list[dict[str, Any]]:
    rooms: list[dict[str, Any]] = []
    for node_id, attributes in graph.nodes(data=True):
        if not isinstance(attributes, Mapping):
            continue
        room_type = str(attributes.get("type", "")).lower()
        geometry = _largest_polygon(attributes.get("geometry"))
        if room_type not in _ROOM_TYPES or geometry is None or geometry.is_empty:
            continue
        rooms.append({"id": str(node_id), "type": room_type, "geometry": geometry})
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
    inner = record.get("inner")
    net_area = _finite_float(record.get("net_area"))
    if isinstance(inner, BaseGeometry) and not inner.is_empty and inner.area > 1e-9 and net_area is not None and net_area > 0.0:
        return float(math.sqrt(net_area / float(inner.area)))
    return 0.05


def _boundary_features(
    record: Mapping[str, Any],
    room: Polygon,
    *,
    scale: float,
    origin: tuple[float, float],
) -> list[dict[str, Any]]:
    wall_depth = max(float(record.get("wall_depth", 0.0)), 1.0)
    proximity = max(wall_depth * 1.1, 1.25)
    features: list[dict[str, Any]] = []
    raw_layers = (
        ("door", record.get("door")),
        ("door", record.get("front_door")),
        ("window", record.get("window")),
    )
    for kind, layer in raw_layers:
        for polygon in _iter_polygons(layer):
            if polygon.distance(room.boundary) > proximity:
                continue
            contact = room.boundary.intersection(polygon.buffer(max(wall_depth * 0.28, 0.4), cap_style=2, join_style=2))
            raw_segments = _line_segments(contact)
            if sum(LineString(segment).length for segment in raw_segments) <= 0.25:
                continue
            metric_polygon = _metric_geometry(polygon, scale, *origin)
            metric_segments = [_metric_segment(segment, scale, origin) for segment in raw_segments]
            features.append({
                "type": kind,
                "raw_geometry": polygon,
                "raw_zone": polygon.buffer(max(wall_depth * 0.28, 0.4), cap_style=2, join_style=2),
                "polygon": _polygon_coordinates(_largest_polygon(metric_polygon)),
                "segments": metric_segments,
                "sill_height_m": 0.0 if kind == "door" else 0.9,
                "height_m": 2.1 if kind == "door" else 1.2,
            })
    return features


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
    }


def _segment_length(segment: Sequence[Sequence[float]]) -> float:
    return float(math.hypot(float(segment[1][0]) - float(segment[0][0]), float(segment[1][1]) - float(segment[0][1])))


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
