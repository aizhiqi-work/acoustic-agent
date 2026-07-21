from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Mapping, Sequence

from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box


FURNITURE_CATALOG: dict[str, dict[str, Any]] = {
    "sofa": {"title": "Sofa", "semantic": "sofa_couch", "size": (2.05, 0.90, 0.72)},
    "bed": {"title": "Bed", "semantic": "bed_mattress", "size": (2.10, 1.55, 0.55)},
    "table": {"title": "Table", "semantic": "table_desk_counter", "size": (1.35, 0.78, 0.74)},
    "cabinet": {"title": "Cabinet", "semantic": "cabinet_shelf_wardrobe", "size": (1.25, 0.42, 1.75)},
    "chair": {"title": "Chair", "semantic": "chair_seating", "size": (0.55, 0.55, 0.86)},
    "rug": {"title": "Rug", "semantic": "carpet_rug", "size": (1.85, 1.20, 0.04), "z": 0.02},
    "curtain": {"title": "Curtain", "semantic": "curtain_blind", "size": (1.75, 0.06, 2.10), "z": 1.05},
    "tv_mirror": {"title": "TV / Mirror", "semantic": "screen_mirror", "size": (1.10, 0.05, 0.65), "z": 1.15},
    "fridge": {"title": "Fridge", "semantic": "appliance", "size": (0.75, 0.68, 1.75)},
    "washing_machine": {"title": "Washing Machine", "semantic": "appliance", "size": (0.65, 0.62, 0.86)},
}


_ROOM_PROGRAMS: dict[str, dict[str, tuple[str, ...]]] = {
    "sparse": {
        "living": ("sofa", "table"),
        "bedroom": ("bed",),
        "kitchen": ("fridge", "cabinet"),
        "bathroom": ("washing_machine",),
        "storage": ("cabinet",),
        "balcony": ("chair",),
    },
    "balanced": {
        "living": ("sofa", "table", "rug", "tv_mirror", "cabinet"),
        "bedroom": ("bed", "cabinet", "rug"),
        "kitchen": ("fridge", "cabinet", "table"),
        "bathroom": ("washing_machine", "cabinet"),
        "storage": ("cabinet", "cabinet"),
        "balcony": ("chair", "table"),
    },
    "compact": {
        "living": ("sofa", "table", "rug", "tv_mirror", "cabinet", "chair", "chair"),
        "bedroom": ("bed", "cabinet", "rug", "chair"),
        "kitchen": ("fridge", "cabinet", "table", "cabinet", "chair"),
        "bathroom": ("washing_machine", "cabinet"),
        "storage": ("cabinet", "cabinet", "cabinet"),
        "balcony": ("chair", "chair", "table"),
    },
}

_WALL_TYPES = {"sofa", "bed", "cabinet", "tv_mirror", "fridge", "washing_machine"}
_OVERLAP_TYPES = {"rug", "curtain"}


def normalize_compactness(value: str) -> str:
    key = str(value or "balanced").strip().lower().replace("-", "_")
    aliases = {"normal": "balanced", "medium": "balanced", "dense": "compact", "high": "compact", "low": "sparse"}
    key = aliases.get(key, key)
    if key not in _ROOM_PROGRAMS:
        raise ValueError("compactness must be sparse, balanced, or compact")
    return key


def generate_floorplan_furniture(
    room_metadata: Mapping[str, Any],
    *,
    compactness: str = "balanced",
    seed: int = 42,
    exclude_points: Sequence[Sequence[float]] = (),
    existing_objects: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Generate deterministic semantic furniture without blocking floor-plan portals."""
    level = normalize_compactness(compactness)
    multi_room = room_metadata.get("multi_room") if isinstance(room_metadata, Mapping) else None
    if not isinstance(multi_room, Mapping):
        raise ValueError("room_metadata.multi_room is required for semantic furnishing")
    raw_rooms = multi_room.get("rooms", ())
    if not isinstance(raw_rooms, Sequence) or isinstance(raw_rooms, (str, bytes)):
        raise ValueError("room_metadata.multi_room.rooms must be a list")

    rooms: list[tuple[str, str, Polygon]] = []
    for raw_room in raw_rooms:
        if not isinstance(raw_room, Mapping):
            continue
        corners = raw_room.get("corners", ())
        try:
            polygon = Polygon([(float(point[0]), float(point[1])) for point in corners]).buffer(0)
        except (TypeError, ValueError, IndexError):
            continue
        if polygon.is_empty or not isinstance(polygon, Polygon) or polygon.area < 1.0:
            continue
        rooms.append((str(raw_room.get("id", "room")), str(raw_room.get("type", "living")), polygon))
    if not rooms:
        raise ValueError("no valid semantic rooms are available for furnishing")

    rng = random.Random(int(seed))
    feature_map = _features_by_room(room_metadata.get("boundary_features", ()))
    excluded = [Point(float(point[0]), float(point[1])).buffer(0.48) for point in exclude_points if len(point) >= 2]
    occupied = [footprint for item in existing_objects if (footprint := _object_footprint(item)) is not None]
    objects: list[dict[str, Any]] = []
    placed: list[tuple[str, str, tuple[float, float], Polygon]] = []
    skipped: Counter[str] = Counter()
    room_counts: Counter[str] = Counter()

    for room_id, room_type, room in sorted(rooms, key=lambda item: item[0]):
        room_features = feature_map.get(room_id, ())
        door_zones = [LineString(feature["segment"]).buffer(0.72, cap_style=2) for feature in room_features if feature["type"] in {"door", "opening"}]
        window_zones = [LineString(feature["segment"]).buffer(0.20, cap_style=2) for feature in room_features if feature["type"] == "window"]
        program = _ROOM_PROGRAMS[level].get(room_type, ())
        for item_type in program:
            spec = FURNITURE_CATALOG[item_type]
            candidate = _place_item(
                item_type,
                spec,
                room,
                room_id,
                occupied,
                excluded,
                door_zones,
                window_zones,
                placed,
                rng,
                compactness=level,
            )
            if candidate is None:
                skipped[item_type] += 1
                continue
            position, rotation, footprint = candidate
            item = _furniture_object(item_type, spec, position, rotation, room_id, level, int(seed), len(objects))
            objects.append(item)
            room_counts[room_id] += 1
            placed.append((room_id, item_type, position, footprint))
            if item_type not in _OVERLAP_TYPES:
                occupied.append(footprint)

        curtain_limit = 0 if level == "sparse" else 1 if level == "balanced" else 2
        if room_type not in {"living", "bedroom"}:
            curtain_limit = 0
        for feature in [item for item in room_features if item["type"] == "window"][:curtain_limit]:
            candidate = _curtain_candidate(room, feature["segment"], door_zones)
            if candidate is None:
                skipped["curtain"] += 1
                continue
            position, rotation, size, footprint = candidate
            spec = {**FURNITURE_CATALOG["curtain"], "size": size}
            objects.append(_furniture_object("curtain", spec, position, rotation, room_id, level, int(seed), len(objects)))
            room_counts[room_id] += 1
            placed.append((room_id, "curtain", position, footprint))

    return {
        "objects": objects,
        "summary": {
            "compactness": level,
            "seed": int(seed),
            "object_count": len(objects),
            "room_count": len(rooms),
            "objects_by_room": dict(room_counts),
            "skipped_by_type": dict(skipped),
        },
    }


def _features_by_room(raw_features: Any) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        return output
    for feature in raw_features:
        if not isinstance(feature, Mapping):
            continue
        room_ids = feature.get("room_ids", ())
        segments = feature.get("segments", ())
        if not isinstance(room_ids, Sequence) or not isinstance(segments, Sequence):
            continue
        for index, room_id in enumerate(room_ids):
            if not segments:
                continue
            raw_segment = segments[min(index, len(segments) - 1)]
            try:
                segment = (
                    (float(raw_segment[0][0]), float(raw_segment[0][1])),
                    (float(raw_segment[1][0]), float(raw_segment[1][1])),
                )
            except (TypeError, ValueError, IndexError):
                continue
            output.setdefault(str(room_id), []).append({
                "id": str(feature.get("id", "feature")),
                "type": str(feature.get("type", "wall")),
                "segment": segment,
                "open": bool(feature.get("open", False)),
            })
    return output


def _place_item(
    item_type: str,
    spec: Mapping[str, Any],
    room: Polygon,
    room_id: str,
    occupied: Sequence[Polygon],
    excluded: Sequence[Polygon],
    door_zones: Sequence[Polygon],
    window_zones: Sequence[Polygon],
    placed: Sequence[tuple[str, str, tuple[float, float], Polygon]],
    rng: random.Random,
    *,
    compactness: str,
) -> tuple[tuple[float, float], float, Polygon] | None:
    size = tuple(float(value) for value in spec["size"])
    candidates = _wall_candidates(room, size, item_type, rng) if item_type in _WALL_TYPES else _free_candidates(room, size, rng)
    clearance = {"sparse": 0.20, "balanced": 0.13, "compact": 0.07}[compactness]
    valid: list[tuple[float, tuple[float, float], float, Polygon]] = []
    domain = room.buffer(-0.055, join_style=2)
    if domain.is_empty:
        domain = room
    for position, rotation in candidates:
        footprint = _rect_footprint(position, size, rotation)
        if not domain.covers(footprint):
            continue
        if any(footprint.intersects(zone) for zone in door_zones):
            continue
        if item_type not in {"curtain", "rug"} and size[2] > 1.0 and any(footprint.intersects(zone) for zone in window_zones):
            continue
        if any(footprint.buffer(clearance).intersects(zone) for zone in excluded):
            continue
        if item_type not in _OVERLAP_TYPES and any(footprint.buffer(clearance).intersects(other) for other in occupied):
            continue
        score = _layout_score(item_type, position, room, room_id, placed) + rng.random() * 0.03
        valid.append((score, position, rotation, footprint))
    if not valid:
        return None
    _score, position, rotation, footprint = max(valid, key=lambda item: item[0])
    return position, rotation, footprint


def _wall_candidates(
    room: Polygon,
    size: tuple[float, float, float],
    item_type: str,
    rng: random.Random,
) -> list[tuple[tuple[float, float], float]]:
    coordinates = list(room.exterior.coords)
    candidates: list[tuple[tuple[float, float], float]] = []
    depth_along_wall = item_type == "bed"
    along = size[1] if depth_along_wall else size[0]
    inward = size[0] if depth_along_wall else size[1]
    for start, end in zip(coordinates, coordinates[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        length = math.hypot(dx, dy)
        if length < along + 0.18:
            continue
        ux, uy = dx / length, dy / length
        midpoint = ((float(start[0]) + float(end[0])) * 0.5, (float(start[1]) + float(end[1])) * 0.5)
        normals = ((-uy, ux), (uy, -ux))
        inward_normal = next((normal for normal in normals if room.covers(Point(
            midpoint[0] + normal[0] * (inward * 0.5 + 0.08),
            midpoint[1] + normal[1] * (inward * 0.5 + 0.08),
        ))), None)
        if inward_normal is None:
            continue
        margin = along * 0.5 + 0.10
        usable = max(0.0, length - margin * 2)
        sample_count = max(2, min(9, int(length / 0.45)))
        distances = [margin + usable * index / max(sample_count - 1, 1) for index in range(sample_count)]
        rng.shuffle(distances)
        rotation = math.degrees(math.atan2(dy, dx)) - (90.0 if depth_along_wall else 0.0)
        for distance in distances:
            candidates.append(((
                float(start[0]) + ux * distance + inward_normal[0] * (inward * 0.5 + 0.06),
                float(start[1]) + uy * distance + inward_normal[1] * (inward * 0.5 + 0.06),
            ), rotation))
    rng.shuffle(candidates)
    return candidates


def _free_candidates(room: Polygon, size: tuple[float, float, float], rng: random.Random) -> list[tuple[tuple[float, float], float]]:
    min_x, min_y, max_x, max_y = room.bounds
    center = room.representative_point()
    candidates = [((float(center.x), float(center.y)), rotation) for rotation in (0.0, 90.0)]
    for _ in range(120):
        position = (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        if room.covers(Point(position)):
            candidates.append((position, rng.choice((0.0, 90.0))))
    return candidates


def _layout_score(
    item_type: str,
    position: tuple[float, float],
    room: Polygon,
    room_id: str,
    placed: Sequence[tuple[str, str, tuple[float, float], Polygon]],
) -> float:
    center = room.centroid
    score = -0.05 * math.hypot(position[0] - center.x, position[1] - center.y)
    targets = {
        "table": ({"sofa"}, 1.35, 3.5),
        "rug": ({"sofa", "bed"}, 0.65, 2.2),
        "tv_mirror": ({"sofa"}, 2.5, 1.7),
        "chair": ({"table"}, 0.85, 2.5),
    }
    if item_type in targets:
        target_types, target_distance, weight = targets[item_type]
        distances = [
            math.hypot(position[0] - other_position[0], position[1] - other_position[1])
            for other_room, other_type, other_position, _footprint in placed
            if other_room == room_id and other_type in target_types
        ]
        if distances:
            score -= abs(min(distances) - target_distance) * weight
    return score


def _curtain_candidate(
    room: Polygon,
    segment: Sequence[Sequence[float]],
    door_zones: Sequence[Polygon],
) -> tuple[tuple[float, float], float, tuple[float, float, float], Polygon] | None:
    start, end = segment
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length < 0.35:
        return None
    ux, uy = dx / length, dy / length
    midpoint = ((float(start[0]) + float(end[0])) * 0.5, (float(start[1]) + float(end[1])) * 0.5)
    normals = ((-uy, ux), (uy, -ux))
    normal = next((item for item in normals if room.covers(Point(midpoint[0] + item[0] * 0.05, midpoint[1] + item[1] * 0.05))), None)
    if normal is None:
        return None
    size = (min(1.75, max(0.35, length)), 0.06, 2.10)
    position = (midpoint[0] + normal[0] * 0.04, midpoint[1] + normal[1] * 0.04)
    rotation = math.degrees(math.atan2(dy, dx))
    footprint = _rect_footprint(position, size, rotation)
    if any(footprint.intersects(zone) for zone in door_zones):
        return None
    return position, rotation, size, footprint


def _rect_footprint(position: Sequence[float], size: Sequence[float], rotation: float) -> Polygon:
    footprint = box(-float(size[0]) * 0.5, -float(size[1]) * 0.5, float(size[0]) * 0.5, float(size[1]) * 0.5)
    footprint = affinity.rotate(footprint, float(rotation), origin=(0.0, 0.0), use_radians=False)
    return affinity.translate(footprint, float(position[0]), float(position[1]))


def _object_footprint(item: Mapping[str, Any]) -> Polygon | None:
    try:
        position = item.get("position", (0.0, 0.0))
        size = item.get("size", (1.0, 1.0, 1.0))
        return _rect_footprint(position, size, float(item.get("rotation", item.get("rotation_deg", 0.0))))
    except (TypeError, ValueError, IndexError):
        return None


def _furniture_object(
    item_type: str,
    spec: Mapping[str, Any],
    position: tuple[float, float],
    rotation: float,
    room_id: str,
    compactness: str,
    seed: int,
    index: int,
) -> dict[str, Any]:
    size = [round(float(value), 4) for value in spec["size"]]
    return {
        "id": f"auto_{seed}_{index}",
        "type": item_type,
        "title": str(spec["title"]),
        "semantic": str(spec["semantic"]),
        "position": [round(float(position[0]), 4), round(float(position[1]), 4)],
        "rotation": round(float(rotation), 3),
        "size": size,
        "z": round(float(spec.get("z", size[2] * 0.5)), 4),
        "absorption_class": "auto",
        "placement": {
            "source": "semantic_auto",
            "room_id": room_id,
            "compactness": compactness,
            "seed": int(seed),
        },
    }
