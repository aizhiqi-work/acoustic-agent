from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import random
import re
from typing import Any, Mapping, Sequence

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .floorplan_resource import FloorplanResource


SCHEMA_VERSION = 1
SUPPORTED_ROOM_TYPES = {"living", "kitchen", "bedroom", "bathroom", "storage", "balcony"}
_ROOM_WEIGHTS = {
    "living": 24.0,
    "kitchen": 10.0,
    "bedroom": 14.0,
    "bathroom": 6.0,
    "storage": 4.0,
    "balcony": 6.0,
}
_CHINESE_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_ENGLISH_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def generate_floorplan_from_text(
    description: str,
    *,
    seed: int = 42,
    width_m: float | None = None,
    depth_m: float | None = None,
    height_m: float | None = None,
) -> dict[str, Any]:
    """Generate a deterministic solver-ready floor-plan specification locally."""
    text = str(description or "").strip()
    if not text:
        text = "10m x 8m, two bedrooms, one living room, one kitchen, one bathroom"
    parsed_width, parsed_depth = _parse_dimensions(text)
    width = _bounded_dimension(width_m if width_m is not None else parsed_width, 10.0)
    depth = _bounded_dimension(depth_m if depth_m is not None else parsed_depth, 8.0)
    height = _bounded_height(height_m if height_m is not None else _parse_height(text))
    rooms = _room_program(text)
    if len(rooms) > 12:
        raise ValueError("custom floor plans support at most 12 rooms")
    if width * depth / len(rooms) < 3.0:
        raise ValueError("the requested dimensions are too small for the requested room count")

    rng = random.Random(int(seed))
    partitions = _partition_rooms(rooms, (0.0, 0.0, width, depth), rng)
    room_records = [
        {
            "id": room["id"],
            "type": room["type"],
            "corners": _rectangle_corners(rect),
        }
        for room, rect in partitions
    ]
    openings = _generate_openings(room_records, width, depth, rng)
    spec = {
        "schema_version": SCHEMA_VERSION,
        "title": _short_title(text),
        "units": "m",
        "height_m": height,
        "wall_depth_m": 0.12,
        "outer_boundary": [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]],
        "rooms": room_records,
        "openings": openings,
        "provenance": {
            "source": "local_text_generator",
            "description": text,
            "seed": int(seed),
            "model": None,
            "prompt_version": None,
        },
    }
    report = validate_floorplan_spec(spec)
    if not report["valid"]:
        raise ValueError("generated floor plan is invalid: " + "; ".join(report["errors"]))
    return report["spec"]


def validate_floorplan_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        spec = _normalize_spec(raw)
    except (TypeError, ValueError) as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "summary": {"rooms": 0, "doors": 0, "windows": 0, "area_m2": 0.0},
            "spec": None,
        }

    outer = Polygon(spec["outer_boundary"])
    if not outer.is_valid or outer.area < 4.0:
        errors.append("outer_boundary must be a valid polygon of at least 4 m2")
    room_polygons: dict[str, Polygon] = {}
    for room in spec["rooms"]:
        polygon = Polygon(room["corners"])
        room_id = room["id"]
        if not polygon.is_valid or polygon.area < 1.0:
            errors.append(f"room {room_id!r} must be a valid polygon of at least 1 m2")
            continue
        if outer.is_valid and not outer.buffer(0.02).covers(polygon):
            errors.append(f"room {room_id!r} extends outside outer_boundary")
        room_polygons[room_id] = polygon

    room_items = list(room_polygons.items())
    for index, (first_id, first) in enumerate(room_items):
        for second_id, second in room_items[index + 1:]:
            overlap = first.intersection(second).area
            if overlap > 1e-4:
                errors.append(f"rooms {first_id!r} and {second_id!r} overlap by {overlap:.3f} m2")
    if outer.is_valid and room_polygons:
        uncovered = outer.difference(unary_union(list(room_polygons.values()))).area
        if uncovered > max(0.05, outer.area * 0.002):
            errors.append(f"room polygons leave {uncovered:.3f} m2 of the indoor boundary uncovered")

    adjacency: dict[str, set[str]] = {room_id: set() for room_id in room_polygons}
    for opening in spec["openings"]:
        opening_id = opening["id"]
        segment = LineString(opening["segment"])
        room_ids = opening["room_ids"]
        if segment.length < 0.35:
            errors.append(f"opening {opening_id!r} is shorter than 0.35 m")
        expected_rooms = 2 if opening["connection"] == "interior_room" else 1
        if len(room_ids) != expected_rooms:
            errors.append(f"opening {opening_id!r} must reference {expected_rooms} room(s)")
        for room_id in room_ids:
            polygon = room_polygons.get(room_id)
            if polygon is None:
                errors.append(f"opening {opening_id!r} references unknown room {room_id!r}")
            elif segment.distance(polygon.boundary) > 0.025:
                errors.append(f"opening {opening_id!r} is not on room {room_id!r}'s boundary")
        if opening["connection"] == "interior_room" and opening["open"] and len(room_ids) == 2:
            adjacency.setdefault(room_ids[0], set()).add(room_ids[1])
            adjacency.setdefault(room_ids[1], set()).add(room_ids[0])

    if adjacency:
        start = next(iter(adjacency))
        reached = {start}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        missing = sorted(set(adjacency) - reached)
        if missing:
            errors.append("rooms are not connected through open interior doors: " + ", ".join(missing))

    entrance_count = sum(item["connection"] == "outdoor_entry" for item in spec["openings"])
    if entrance_count == 0:
        warnings.append("no exterior entrance door is defined")
    window_count = sum(item["type"] == "window" for item in spec["openings"])
    if window_count == 0:
        warnings.append("no exterior windows are defined")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "rooms": len(spec["rooms"]),
            "doors": sum(item["type"] == "door" for item in spec["openings"]),
            "windows": window_count,
            "area_m2": round(float(outer.area), 3) if outer.is_valid else 0.0,
        },
        "spec": spec,
    }


def compile_floorplan_spec(
    raw: Mapping[str, Any],
    *,
    source_room: str | None = None,
    receiver_room: str | None = None,
    seed: int = 42,
    height_m: float | None = None,
) -> dict[str, Any]:
    report = validate_floorplan_spec(raw)
    if not report["valid"]:
        raise ValueError("invalid custom floor plan: " + "; ".join(report["errors"]))
    spec = deepcopy(report["spec"])
    if height_m is not None:
        spec["height_m"] = _bounded_height(height_m)
    record = _compiled_record(spec)
    resource = _MemoryFloorplanResource(record)
    room_ids = {room["id"] for room in record["rooms"]}
    default_source = next((room["id"] for room in record["rooms"] if room["type"] == "living"), record["rooms"][0]["id"])
    source_id = source_room if source_room in room_ids else default_source
    receiver_id = receiver_room if receiver_room in room_ids else source_id
    placement = resource.sample_placement(
        0,
        placement="same_room" if source_id == receiver_id else "cross_room",
        seed=int(seed),
        source_room=source_id,
        receiver_room=receiver_id,
    )
    scene = resource.scene(
        0,
        source_id,
        receiver_room_id=receiver_id,
        height_m=float(spec["height_m"]),
        source=placement["source"],
        receiver=placement["receiver"],
    )
    scene["dataset"].update({
        "generator": str(spec.get("provenance", {}).get("source", "custom")),
        "title": str(spec.get("title", "Custom floor plan")),
        "description": str(spec.get("provenance", {}).get("description", "")),
        "custom": True,
    })
    scene["room"]["metadata"]["custom_floorplan"] = {
        "schema_version": SCHEMA_VERSION,
        "title": spec.get("title"),
        "provenance": deepcopy(spec.get("provenance", {})),
        "validation": {"warnings": report["warnings"], "summary": report["summary"]},
        "spec": spec,
    }
    return scene


class _MemoryFloorplanResource(FloorplanResource):
    def __init__(self, record: Mapping[str, Any]) -> None:
        self.path = None
        self._record = deepcopy(dict(record))
        self.count = 1
        self.metadata = {"schema_version": 1, "source_record_count": 1}

    def record(self, index: int) -> dict[str, Any]:
        if int(index) != 0:
            raise IndexError("custom floor plan contains one scene at index 0")
        return deepcopy(self._record)


def _normalize_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("floor plan specification must be an object")
    if int(raw.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported custom floor plan schema {raw.get('schema_version')!r}")
    if str(raw.get("units", "m")).lower() not in {"m", "meter", "meters"}:
        raise ValueError("custom floor plan units must be meters")
    outer_raw = raw.get("outer_boundary")
    if not isinstance(outer_raw, Sequence) or isinstance(outer_raw, (str, bytes)):
        raise ValueError("outer_boundary must contain polygon points")
    outer = _points(outer_raw, "outer_boundary")
    if len(outer) < 3:
        raise ValueError("outer_boundary must contain at least three points")
    min_x = min(point[0] for point in outer)
    min_y = min(point[1] for point in outer)
    shift = lambda point: [round(float(point[0]) - min_x, 6), round(float(point[1]) - min_y, 6)]
    outer = [shift(point) for point in outer]

    raw_rooms = raw.get("rooms")
    if not isinstance(raw_rooms, Sequence) or isinstance(raw_rooms, (str, bytes)) or not raw_rooms:
        raise ValueError("rooms must contain at least one room")
    if len(raw_rooms) > 12:
        raise ValueError("custom floor plans support at most 12 rooms")
    rooms: list[dict[str, Any]] = []
    room_ids: set[str] = set()
    for index, item in enumerate(raw_rooms):
        if not isinstance(item, Mapping):
            raise ValueError(f"room {index} must be an object")
        room_type = str(item.get("type", "room")).strip().lower()
        if room_type not in SUPPORTED_ROOM_TYPES:
            raise ValueError(f"room {index} has unsupported type {room_type!r}")
        room_id = str(item.get("id", f"{room_type}_{index}")).strip()
        if not room_id or room_id in room_ids:
            raise ValueError(f"room id {room_id!r} is empty or duplicated")
        corners = _points(item.get("corners", []), f"room {room_id} corners")
        if len(corners) < 3:
            raise ValueError(f"room {room_id!r} must contain at least three corners")
        room_ids.add(room_id)
        rooms.append({"id": room_id, "type": room_type, "corners": [shift(point) for point in corners]})

    raw_openings = raw.get("openings", [])
    if not isinstance(raw_openings, Sequence) or isinstance(raw_openings, (str, bytes)):
        raise ValueError("openings must be a list")
    openings: list[dict[str, Any]] = []
    opening_ids: set[str] = set()
    height = _bounded_height(raw.get("height_m"))
    for index, item in enumerate(raw_openings):
        if not isinstance(item, Mapping):
            raise ValueError(f"opening {index} must be an object")
        kind = str(item.get("type", "door")).strip().lower()
        if kind not in {"door", "window", "opening"}:
            raise ValueError(f"opening {index} has unsupported type {kind!r}")
        opening_id = str(item.get("id", f"{kind}_{index}")).strip()
        if not opening_id or opening_id in opening_ids:
            raise ValueError(f"opening id {opening_id!r} is empty or duplicated")
        segment = _points(item.get("segment", []), f"opening {opening_id} segment")
        if len(segment) != 2:
            raise ValueError(f"opening {opening_id!r} segment must contain two points")
        room_refs = [str(value) for value in item.get("room_ids", [])]
        connection = str(item.get("connection", "interior_room" if len(room_refs) == 2 else "outdoor_facade"))
        sill = max(0.0, float(item.get("sill_height_m", 0.9 if kind == "window" else 0.0)))
        opening_height = float(item.get("height_m", 1.2 if kind == "window" else min(2.1, height)))
        opening_ids.add(opening_id)
        openings.append({
            "id": opening_id,
            "type": kind,
            "room_ids": room_refs,
            "segment": [shift(point) for point in segment],
            "width_m": round(LineString([shift(point) for point in segment]).length, 6),
            "height_m": round(min(max(0.05, opening_height), max(0.05, height - sill)), 6),
            "sill_height_m": round(min(sill, height - 0.05), 6),
            "connection": connection,
            "open": bool(item.get("open", kind == "opening" or connection == "interior_room")),
            "confidence": round(min(1.0, max(0.0, float(item.get("confidence", 1.0)))), 4),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "title": str(raw.get("title", "Custom floor plan"))[:120],
        "units": "m",
        "height_m": height,
        "wall_depth_m": round(min(0.5, max(0.03, float(raw.get("wall_depth_m", 0.12)))), 6),
        "outer_boundary": outer,
        "rooms": rooms,
        "openings": openings,
        "provenance": deepcopy(dict(raw.get("provenance", {}))) if isinstance(raw.get("provenance"), Mapping) else {},
    }


def _compiled_record(spec: Mapping[str, Any]) -> dict[str, Any]:
    height = float(spec["height_m"])
    outer = Polygon(spec["outer_boundary"])
    min_x, min_y, max_x, max_y = outer.bounds
    rooms = []
    for item in spec["rooms"]:
        polygon = Polygon(item["corners"])
        rooms.append({
            "id": item["id"],
            "type": item["type"],
            "area_m2": float(polygon.area),
            "corners": deepcopy(item["corners"]),
        })
    portals: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    room_by_id = {room["id"]: room for room in rooms}
    for opening in spec["openings"]:
        segment = opening["segment"]
        center = [round((segment[0][0] + segment[1][0]) * 0.5, 6), round((segment[0][1] + segment[1][1]) * 0.5, 6)]
        feature = {
            "id": opening["id"],
            "type": opening["type"],
            "segments": [deepcopy(segment)],
            "sill_height_m": opening["sill_height_m"],
            "height_m": opening["height_m"],
            "connection": opening["connection"],
            "open": opening["open"],
            "room_ids": list(opening["room_ids"]),
        }
        features.append(feature)
        if opening["connection"] == "interior_room" and len(opening["room_ids"]) == 2 and opening["open"]:
            room_points = {
                room_id: _portal_room_point(center, room_by_id[room_id]["corners"])
                for room_id in opening["room_ids"]
            }
            portals.append({
                "id": opening["id"],
                "type": opening["type"],
                "room_ids": list(opening["room_ids"]),
                "room_points": room_points,
                "center": center,
                "width_m": opening["width_m"],
                "sill_height_m": opening["sill_height_m"],
                "height_m": opening["height_m"],
                "open": True,
            })
    surfaces = _surface_segments(rooms, spec["openings"], height)
    digest = hashlib.sha256(repr((spec["outer_boundary"], spec["rooms"], spec["openings"])).encode()).hexdigest()[:16]
    return {
        "source_index": 0,
        "sample_id": f"custom-{digest}",
        "unit_type": "Custom",
        "net_area_m2": float(sum(room["area_m2"] for room in rooms)),
        "gross_area_m2": float(outer.area),
        "meters_per_unit": 1.0,
        "scale_source": "custom_metric",
        "wall_depth_m": float(spec["wall_depth_m"]),
        "height_m": height,
        "size": [float(max_x - min_x), float(max_y - min_y)],
        "corners": deepcopy(spec["outer_boundary"]),
        "rooms": rooms,
        "portals": portals,
        "features": features,
        "surfaces": surfaces,
    }


def _surface_segments(
    rooms: Sequence[Mapping[str, Any]],
    openings: Sequence[Mapping[str, Any]],
    height: float,
) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for room in rooms:
        corners = room["corners"]
        room_openings = [item for item in openings if room["id"] in item["room_ids"]]
        for edge_index, start in enumerate(corners):
            end = corners[(edge_index + 1) % len(corners)]
            edge_length = math.dist(start, end)
            if edge_length <= 1e-8:
                continue
            intervals: list[tuple[float, float, Mapping[str, Any]]] = []
            for opening in room_openings:
                if not _segment_on_edge(opening["segment"], start, end):
                    continue
                values = sorted((_edge_parameter(opening["segment"][0], start, end), _edge_parameter(opening["segment"][1], start, end)))
                intervals.append((max(0.0, values[0]), min(1.0, values[1]), opening))
            intervals.sort(key=lambda value: value[0])
            cursor = 0.0
            for begin, finish, opening in intervals:
                if begin > cursor + 1e-6:
                    _append_surface(surfaces, _edge_slice(start, end, cursor, begin), "wall", 0.0, height, room["id"])
                segment = _edge_slice(start, end, begin, finish)
                sill = float(opening["sill_height_m"])
                top = min(height, sill + float(opening["height_m"]))
                if opening["type"] == "window":
                    if sill > 1e-6:
                        _append_surface(surfaces, segment, "wall", 0.0, sill, room["id"])
                    _append_surface(surfaces, segment, "window", sill, top, room["id"])
                    if top < height - 1e-6:
                        _append_surface(surfaces, segment, "wall", top, height, room["id"])
                elif opening["connection"] != "interior_room" or not opening["open"]:
                    _append_surface(surfaces, segment, "door", sill, top, room["id"])
                    if top < height - 1e-6:
                        _append_surface(surfaces, segment, "wall", top, height, room["id"])
                elif top < height - 1e-6:
                    _append_surface(surfaces, segment, "wall", top, height, room["id"])
                cursor = max(cursor, finish)
            if cursor < 1.0 - 1e-6:
                _append_surface(surfaces, _edge_slice(start, end, cursor, 1.0), "wall", 0.0, height, room["id"])
    return surfaces


def _append_surface(
    target: list[dict[str, Any]],
    segment: Sequence[Sequence[float]],
    kind: str,
    z_min: float,
    z_max: float,
    room_id: str,
) -> None:
    if math.dist(segment[0], segment[1]) < 0.015 or z_max - z_min < 0.015:
        return
    target.append({
        "a": [round(float(value), 6) for value in segment[0]],
        "b": [round(float(value), 6) for value in segment[1]],
        "type": kind,
        "z_min": round(float(z_min), 6),
        "z_max": round(float(z_max), 6),
        "name": f"{room_id}_{kind}_{len(target)}",
        "room_id": room_id,
    })


def _room_program(text: str) -> list[dict[str, Any]]:
    counts = {
        "bedroom": _room_count(text, "bedroom", 2),
        "living": _room_count(text, "living", 1),
        "kitchen": _room_count(text, "kitchen", 1),
        "bathroom": _room_count(text, "bathroom", 1),
        "storage": _room_count(text, "storage", 0),
        "balcony": _room_count(text, "balcony", 0),
    }
    rooms: list[dict[str, Any]] = []
    for room_type in ("living", "kitchen", "bedroom", "bathroom", "storage", "balcony"):
        for index in range(max(0, counts[room_type])):
            rooms.append({"id": f"{room_type}_{index}", "type": room_type, "weight": _ROOM_WEIGHTS[room_type]})
    if not rooms:
        rooms.append({"id": "living_0", "type": "living", "weight": _ROOM_WEIGHTS["living"]})
    return rooms


def _room_count(text: str, room_type: str, default: int) -> int:
    lower = text.lower()
    count = r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
    english = {
        "bedroom": rf"{count}\s*(?:bedrooms?|beds?)\b",
        "living": rf"{count}\s*(?:living rooms?|lounges?)\b",
        "kitchen": rf"{count}\s*kitchens?\b",
        "bathroom": rf"{count}\s*(?:bathrooms?|baths?)\b",
        "storage": rf"{count}\s*(?:storage rooms?|storerooms?)\b",
        "balcony": rf"{count}\s*balcon(?:y|ies)\b",
    }[room_type]
    match = re.search(english, lower)
    if match:
        return min(8, _parse_count(match.group(1)))
    chinese_patterns = {
        "bedroom": [r"([零一二两三四五六七八九十\d]+)\s*(?:个)?(?:卧室|房间)", r"([零一二两三四五六七八九十\d]+)\s*室"],
        "living": [r"([零一二两三四五六七八九十\d]+)\s*(?:个)?(?:客厅|厅)"],
        "kitchen": [r"([零一二两三四五六七八九十\d]+)\s*(?:个)?(?:厨房|厨)"],
        "bathroom": [r"([零一二两三四五六七八九十\d]+)\s*(?:个)?(?:卫生间|浴室|卫)"],
        "storage": [r"([零一二两三四五六七八九十\d]+)\s*(?:个)?(?:储物间|储藏室)"],
        "balcony": [r"([零一二两三四五六七八九十\d]+)\s*(?:个)?阳台"],
    }[room_type]
    for pattern in chinese_patterns:
        match = re.search(pattern, text)
        if match:
            return min(8, _parse_count(match.group(1)))
    return default


def _partition_rooms(
    rooms: Sequence[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    rng: random.Random,
) -> list[tuple[dict[str, Any], tuple[float, float, float, float]]]:
    if len(rooms) == 1:
        return [(dict(rooms[0]), bounds)]
    total = sum(float(room["weight"]) for room in rooms)
    cumulative = 0.0
    split_index = 1
    best = math.inf
    for index in range(1, len(rooms)):
        cumulative += float(rooms[index - 1]["weight"])
        difference = abs(cumulative / total - 0.5)
        if difference < best:
            best = difference
            split_index = index
    first = rooms[:split_index]
    second = rooms[split_index:]
    first_weight = sum(float(room["weight"]) for room in first)
    ratio = min(0.72, max(0.28, first_weight / total + rng.uniform(-0.025, 0.025)))
    x0, y0, x1, y1 = bounds
    if x1 - x0 >= y1 - y0:
        split = x0 + (x1 - x0) * ratio
        first_bounds = (x0, y0, split, y1)
        second_bounds = (split, y0, x1, y1)
    else:
        split = y0 + (y1 - y0) * ratio
        first_bounds = (x0, y0, x1, split)
        second_bounds = (x0, split, x1, y1)
    return _partition_rooms(first, first_bounds, rng) + _partition_rooms(second, second_bounds, rng)


def _generate_openings(
    rooms: Sequence[Mapping[str, Any]],
    width: float,
    depth: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    shared: list[tuple[str, str, list[list[float]]]] = []
    for index, first in enumerate(rooms):
        for second in rooms[index + 1:]:
            segment = _shared_boundary(first["corners"], second["corners"])
            if segment is not None and math.dist(segment[0], segment[1]) >= 0.75:
                shared.append((str(first["id"]), str(second["id"]), segment))
    adjacency: dict[str, list[tuple[str, list[list[float]]]]] = {str(room["id"]): [] for room in rooms}
    for first, second, segment in shared:
        adjacency[first].append((second, segment))
        adjacency[second].append((first, segment))
    root = next((str(room["id"]) for room in rooms if room["type"] == "living"), str(rooms[0]["id"]))
    visited = {root}
    queue = [root]
    door_pairs: list[tuple[str, str, list[list[float]]]] = []
    while queue:
        current = queue.pop(0)
        candidates = list(adjacency.get(current, []))
        rng.shuffle(candidates)
        for neighbor, segment in candidates:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
            door_pairs.append((current, neighbor, segment))
    if len(visited) != len(rooms):
        raise ValueError("the generated room partition has no connected door graph")

    openings: list[dict[str, Any]] = []
    for index, (first, second, wall) in enumerate(door_pairs):
        door = _centered_segment(wall, min(0.9, max(0.7, math.dist(wall[0], wall[1]) - 0.15)), 0.5)
        openings.append({
            "id": f"door_{index}",
            "type": "door",
            "room_ids": [first, second],
            "segment": door,
            "height_m": 2.1,
            "sill_height_m": 0.0,
            "connection": "interior_room",
            "open": True,
            "confidence": 1.0,
        })

    living = next((room for room in rooms if room["type"] == "living"), rooms[0])
    living_edges = _exterior_edges(living["corners"], width, depth)
    entrance_edge = max(living_edges, key=lambda edge: math.dist(edge[0], edge[1])) if living_edges else None
    if entrance_edge is not None:
        entrance = _centered_segment(entrance_edge, min(1.0, math.dist(entrance_edge[0], entrance_edge[1]) * 0.28), 0.2)
        openings.append({
            "id": "entrance_0",
            "type": "door",
            "room_ids": [str(living["id"])],
            "segment": entrance,
            "height_m": 2.1,
            "sill_height_m": 0.0,
            "connection": "outdoor_entry",
            "open": False,
            "confidence": 1.0,
        })

    window_index = 0
    for room in rooms:
        if room["type"] == "storage":
            continue
        edges = sorted(_exterior_edges(room["corners"], width, depth), key=lambda edge: math.dist(edge[0], edge[1]), reverse=True)
        if not edges:
            continue
        edge = edges[0]
        edge_length = math.dist(edge[0], edge[1])
        fraction = 0.72 if room["id"] == living["id"] and edge == entrance_edge else 0.5
        window_width = min(1.6, max(0.6, edge_length * 0.35))
        window = _centered_segment(edge, window_width, fraction)
        if any(_segments_overlap(window, opening["segment"], 0.12) for opening in openings):
            continue
        openings.append({
            "id": f"window_{window_index}",
            "type": "window",
            "room_ids": [str(room["id"])],
            "segment": window,
            "height_m": 1.2,
            "sill_height_m": 0.9,
            "connection": "outdoor_facade",
            "open": False,
            "confidence": 1.0,
        })
        window_index += 1
    return openings


def _parse_dimensions(text: str) -> tuple[float | None, float | None]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|米)?\s*(?:[x×*乘]|by)\s*(\d+(?:\.\d+)?)\s*(?:m|米)?", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    length = re.search(r"(?:长|length)\s*(\d+(?:\.\d+)?)\s*(?:m|米)?", text, re.IGNORECASE)
    width = re.search(r"(?:宽|width)\s*(\d+(?:\.\d+)?)\s*(?:m|米)?", text, re.IGNORECASE)
    return (float(length.group(1)) if length else None, float(width.group(1)) if width else None)


def _parse_height(text: str) -> float | None:
    match = re.search(r"(?:层高|height)\s*(\d+(?:\.\d+)?)\s*(?:m|米)?", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _bounded_dimension(value: Any, fallback: float) -> float:
    numeric = fallback if value is None else float(value)
    if not math.isfinite(numeric) or not 3.0 <= numeric <= 40.0:
        raise ValueError("floor plan width and depth must be between 3 and 40 meters")
    return round(numeric, 4)


def _bounded_height(value: Any) -> float:
    numeric = 2.8 if value is None else float(value)
    if not math.isfinite(numeric) or not 2.0 <= numeric <= 6.0:
        raise ValueError("floor plan height must be between 2 and 6 meters")
    return round(numeric, 4)


def _points(raw: Any, label: str) -> list[list[float]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} must be a list of 2D points")
    points: list[list[float]] = []
    for point in raw:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError(f"{label} contains an invalid point")
        values = [float(point[0]), float(point[1])]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{label} contains a non-finite point")
        points.append(values)
    return points


def _parse_count(value: str) -> int:
    value = value.lower()
    if value.isdigit():
        return int(value)
    if value in _ENGLISH_NUMBERS:
        return _ENGLISH_NUMBERS[value]
    if value in _CHINESE_NUMBERS:
        return _CHINESE_NUMBERS[value]
    if value.startswith("十"):
        return 10 + _CHINESE_NUMBERS.get(value[1:], 0)
    if "十" in value:
        first, second = value.split("十", 1)
        return _CHINESE_NUMBERS.get(first, 1) * 10 + _CHINESE_NUMBERS.get(second, 0)
    return 1


def _rectangle_corners(bounds: Sequence[float]) -> list[list[float]]:
    x0, y0, x1, y1 = bounds
    return [[round(x0, 6), round(y0, 6)], [round(x1, 6), round(y0, 6)], [round(x1, 6), round(y1, 6)], [round(x0, 6), round(y1, 6)]]


def _shared_boundary(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> list[list[float]] | None:
    intersection = Polygon(first).boundary.intersection(Polygon(second).boundary)
    lines = [intersection] if isinstance(intersection, LineString) else [item for item in getattr(intersection, "geoms", []) if isinstance(item, LineString)]
    lines = [line for line in lines if line.length > 1e-6]
    if not lines:
        return None
    line = max(lines, key=lambda item: item.length)
    coordinates = list(line.coords)
    return [[round(float(coordinates[0][0]), 6), round(float(coordinates[0][1]), 6)], [round(float(coordinates[-1][0]), 6), round(float(coordinates[-1][1]), 6)]]


def _exterior_edges(corners: Sequence[Sequence[float]], width: float, depth: float) -> list[list[list[float]]]:
    edges = []
    for index, start in enumerate(corners):
        end = corners[(index + 1) % len(corners)]
        if (
            abs(start[0]) < 1e-6 and abs(end[0]) < 1e-6
            or abs(start[0] - width) < 1e-6 and abs(end[0] - width) < 1e-6
            or abs(start[1]) < 1e-6 and abs(end[1]) < 1e-6
            or abs(start[1] - depth) < 1e-6 and abs(end[1] - depth) < 1e-6
        ):
            edges.append([list(start), list(end)])
    return edges


def _centered_segment(edge: Sequence[Sequence[float]], width: float, fraction: float) -> list[list[float]]:
    length = math.dist(edge[0], edge[1])
    span = min(max(0.35, width), max(0.35, length - 0.08)) / max(length, 1e-9)
    center = min(1.0 - span * 0.5, max(span * 0.5, fraction))
    return [_edge_slice(edge[0], edge[1], center - span * 0.5, center + span * 0.5)[0], _edge_slice(edge[0], edge[1], center - span * 0.5, center + span * 0.5)[1]]


def _segment_on_edge(segment: Sequence[Sequence[float]], start: Sequence[float], end: Sequence[float]) -> bool:
    edge = LineString([start, end])
    line = LineString(segment)
    return line.distance(edge) < 1e-5 and line.difference(edge.buffer(1e-5, cap_style=2)).length < 1e-5


def _edge_parameter(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    denominator = dx * dx + dy * dy
    return 0.0 if denominator <= 1e-12 else ((float(point[0]) - float(start[0])) * dx + (float(point[1]) - float(start[1])) * dy) / denominator


def _edge_slice(start: Sequence[float], end: Sequence[float], first: float, second: float) -> list[list[float]]:
    return [
        [round(float(start[0]) + (float(end[0]) - float(start[0])) * first, 6), round(float(start[1]) + (float(end[1]) - float(start[1])) * first, 6)],
        [round(float(start[0]) + (float(end[0]) - float(start[0])) * second, 6), round(float(start[1]) + (float(end[1]) - float(start[1])) * second, 6)],
    ]


def _portal_room_point(center: Sequence[float], corners: Sequence[Sequence[float]]) -> list[float]:
    centroid = Polygon(corners).representative_point()
    dx = float(centroid.x) - float(center[0])
    dy = float(centroid.y) - float(center[1])
    length = max(math.hypot(dx, dy), 1e-9)
    return [round(float(center[0]) + dx / length * 0.08, 6), round(float(center[1]) + dy / length * 0.08, 6)]


def _segments_overlap(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]], margin: float) -> bool:
    return LineString(first).buffer(margin, cap_style=2).intersects(LineString(second))


def _short_title(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:72] + ("..." if len(compact) > 72 else "")


def floorplan_vlm_prompt() -> str:
    """Return the provider-independent image-to-floor-plan extraction contract."""
    return """You are converting an attached residential floor-plan image into Acoustic Agent Floorplan JSON.

Return exactly one JSON object and no Markdown. Follow these rules:
1. Use meters. Set the lower-left of outer_boundary to [0, 0], x to the right, and y upward.
2. If a printed scale or dimensions exist, use them. Otherwise set the outer width to 10 m, preserve the image aspect ratio, and record that assumption in provenance.scale_assumption.
3. Trace the indoor outer boundary and every indoor room polygon. Room polygons must not overlap, must remain inside outer_boundary, and together must cover the indoor boundary.
4. Supported room types are living, kitchen, bedroom, bathroom, storage, and balcony. Give every room a unique id such as bedroom_0.
5. Represent each door, window, or wall-free opening as a two-point segment lying exactly on the relevant room boundary.
6. An interior connection references exactly two room_ids and uses connection=\"interior_room\". An entry door references one room and uses connection=\"outdoor_entry\". An exterior window references one room and uses connection=\"outdoor_facade\".
7. Set interior doors and wall-free openings to open=true. Set exterior entry doors and windows to open=false. Do not invent an opening that is not visible.
8. Use confidence between 0 and 1. Use sill_height_m=0 for doors/openings and approximately 0.9 for windows unless the drawing specifies otherwise.
9. Preserve uncertain geometry conservatively and describe uncertainties in provenance.notes.

Required JSON shape:
{
  \"schema_version\": 1,
  \"title\": \"...\",
  \"units\": \"m\",
  \"height_m\": 2.8,
  \"wall_depth_m\": 0.12,
  \"outer_boundary\": [[x, y], ...],
  \"rooms\": [
    {\"id\": \"living_0\", \"type\": \"living\", \"corners\": [[x, y], ...]}
  ],
  \"openings\": [
    {\"id\": \"door_0\", \"type\": \"door\", \"room_ids\": [\"living_0\", \"bedroom_0\"], \"segment\": [[x1, y1], [x2, y2]], \"height_m\": 2.1, \"sill_height_m\": 0.0, \"connection\": \"interior_room\", \"open\": true, \"confidence\": 0.9}
  ],
  \"provenance\": {\"source\": \"vlm_image\", \"scale_assumption\": null, \"notes\": []}
}

Before returning, check polygon coverage, overlap, boundary alignment, room references, and open-door connectivity."""


class FloorplanBuilder:
    """Provider-independent entry point for custom floor-plan generation."""

    from_text = staticmethod(generate_floorplan_from_text)
    validate = staticmethod(validate_floorplan_spec)
    compile = staticmethod(compile_floorplan_spec)
    vlm_prompt = staticmethod(floorplan_vlm_prompt)
