from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import sqlite3
from typing import Any, Mapping, Sequence
from urllib.parse import quote
import zlib

from .geometry import point_in_polygon


RESOURCE_SCHEMA_VERSION = 1
DEFAULT_RESPLAN_RESOURCE = Path(__file__).resolve().parent / "resources" / "resplan" / "resplan_v1.sqlite3"
_EXTERIOR_CONNECTIONS = {"outdoor_entry", "outdoor_facade", "outdoor_balcony"}


class ResPlanResource:
    """Random-access loader for precompiled, losslessly compressed ResPlan scenes."""

    def __init__(self, path: str | Path = DEFAULT_RESPLAN_RESOURCE) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"compiled ResPlan resource not found: {self.path}; "
                "run scripts/build_resplan_resource.py"
            )
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM metadata").fetchall()
            count = connection.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
        self.metadata = {str(key): json.loads(value) for key, value in rows}
        if int(self.metadata.get("schema_version", -1)) != RESOURCE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ResPlan resource schema {self.metadata.get('schema_version')!r}; "
                f"expected {RESOURCE_SCHEMA_VERSION}"
            )
        self.count = int(count)

    def __len__(self) -> int:
        return self.count

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{quote(str(self.path))}?mode=ro&immutable=1"
        return sqlite3.connect(uri, uri=True)

    @lru_cache(maxsize=64)
    def record(self, index: int) -> dict[str, Any]:
        resolved = self.resolve_index(index)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM scenes WHERE idx = ?",
                (resolved,),
            ).fetchone()
        if row is None:
            raise IndexError(f"ResPlan resource index must be between 0 and {self.count - 1}")
        return json.loads(zlib.decompress(row[0]))

    def resolve_index(self, index: int, direction: str = "nearest") -> int:
        if self.count < 1:
            raise IndexError("ResPlan resource is empty")
        direction = str(direction).lower()
        if direction == "random":
            return secrets.randbelow(self.count)
        bounded = min(max(int(index), 0), self.count - 1)
        if direction == "next":
            return min(bounded + 1, self.count - 1)
        if direction == "previous":
            return max(bounded - 1, 0)
        if direction != "nearest":
            raise ValueError("direction must be nearest, next, previous, or random")
        return bounded

    def stats(self) -> dict[str, Any]:
        source_stats = self.metadata.get("source_stats", {})
        return {
            **(dict(source_stats) if isinstance(source_stats, Mapping) else {}),
            "records": self.count,
            "eligible_records": self.count,
            "compiled_records": self.count,
            "source_records": int(self.metadata.get("source_record_count", self.count)),
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "storage": "sqlite_zlib_json",
        }

    def rooms(self, index: int) -> list[dict[str, Any]]:
        return [
            {"id": room["id"], "type": room["type"], "area_m2": room["area_m2"]}
            for room in self.record(index)["rooms"]
        ]

    def sample_placement(
        self,
        index: int,
        *,
        placement: str = "random",
        seed: int | None = None,
        source_room: str | None = None,
        receiver_room: str | None = None,
        height_m: float = 1.4,
        wall_margin_m: float = 0.3,
        min_distance_m: float = 1.0,
    ) -> dict[str, Any]:
        record = self.record(index)
        rooms = [room for room in record["rooms"] if len(room.get("corners", [])) >= 3]
        if not rooms:
            raise ValueError(f"ResPlan resource index {index} contains no usable rooms")
        room_by_id = {str(room["id"]): room for room in rooms}
        rng = random.Random(seed)
        mode = str(placement).lower().replace("-", "_")
        aliases = {"any": "random", "same": "same_room", "different": "cross_room", "cross": "cross_room"}
        mode = aliases.get(mode, mode)
        if mode not in {"random", "same_room", "cross_room"}:
            raise ValueError("placement must be random, same_room, or cross_room")

        if source_room:
            source_spec = _room_by_id(room_by_id, source_room)
        elif receiver_room:
            receiver_spec = _room_by_id(room_by_id, receiver_room)
            reachable_to_receiver = _reachable_room_ids(record.get("portals", []), str(receiver_spec["id"]))
            source_candidates = [room for room in rooms if str(room["id"]) in reachable_to_receiver]
            if mode == "same_room":
                source_spec = receiver_spec
            elif mode == "cross_room":
                cross_candidates = [room for room in source_candidates if str(room["id"]) != str(receiver_spec["id"])]
                if not cross_candidates:
                    raise ValueError(f"room {receiver_spec['id']!r} has no connected source room")
                source_spec = rng.choice(cross_candidates)
            else:
                source_spec = rng.choice(source_candidates)
        else:
            source_spec = rng.choice(rooms)
        reachable_ids = _reachable_room_ids(record.get("portals", []), str(source_spec["id"]))
        reachable = [room for room in rooms if str(room["id"]) in reachable_ids]
        if receiver_room:
            receiver_spec = _room_by_id(room_by_id, receiver_room)
            if str(receiver_spec["id"]) not in reachable_ids:
                raise ValueError(
                    f"rooms {source_spec['id']!r} and {receiver_spec['id']!r} have no verified open route"
                )
        elif mode == "same_room":
            receiver_spec = source_spec
        elif mode == "cross_room":
            candidates = [room for room in reachable if room["id"] != source_spec["id"]]
            if not candidates:
                raise ValueError(f"room {source_spec['id']!r} has no connected receiver room")
            receiver_spec = rng.choice(candidates)
        else:
            receiver_spec = rng.choice(reachable)

        source_xy = _sample_room_xy(source_spec["corners"], rng, wall_margin_m)
        receiver_xy = _sample_room_xy(receiver_spec["corners"], rng, wall_margin_m)
        if source_spec["id"] == receiver_spec["id"]:
            best = receiver_xy
            best_distance = _distance_2d(source_xy, receiver_xy)
            for _ in range(96):
                candidate = _sample_room_xy(receiver_spec["corners"], rng, wall_margin_m)
                distance = _distance_2d(source_xy, candidate)
                if distance > best_distance:
                    best, best_distance = candidate, distance
                if distance >= min_distance_m:
                    best = candidate
                    break
            receiver_xy = best
        z_value = max(0.1, float(height_m))
        return {
            "mode": mode,
            "seed": seed,
            "source_room": str(source_spec["id"]),
            "receiver_room": str(receiver_spec["id"]),
            "source": [round(source_xy[0], 3), round(source_xy[1], 3), round(z_value, 3)],
            "receiver": [round(receiver_xy[0], 3), round(receiver_xy[1], 3), round(z_value, 3)],
        }

    def scene(
        self,
        index: int,
        room_id: str | None = None,
        *,
        receiver_room_id: str | None = None,
        height_m: float = 2.8,
        source: Sequence[float] | None = None,
        receiver: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        resolved = self.resolve_index(index)
        record = self.record(resolved)
        rooms = record["rooms"]
        room_by_id = {str(room["id"]): room for room in rooms}
        selected = _room_by_id(room_by_id, room_id) if room_id else _default_room(rooms)
        if receiver_room_id == "auto":
            receiver_room = _auto_receiver_room(record, selected)
        elif receiver_room_id:
            receiver_room = _room_by_id(room_by_id, receiver_room_id)
        else:
            receiver_room = selected
        route_rooms, route_portals = _portal_route(
            record.get("portals", []),
            str(selected["id"]),
            str(receiver_room["id"]),
        )
        if not route_rooms:
            raise ValueError(
                f"rooms {selected['id']!r} and {receiver_room['id']!r} have no verified open route"
            )

        if source is None or receiver is None:
            deterministic_seed = _stable_seed(resolved, str(selected["id"]), str(receiver_room["id"]))
            placement = self.sample_placement(
                resolved,
                placement="same_room" if selected["id"] == receiver_room["id"] else "cross_room",
                seed=deterministic_seed,
                source_room=str(selected["id"]),
                receiver_room=str(receiver_room["id"]),
            )
            if source is None:
                source = placement["source"]
            if receiver is None:
                receiver = placement["receiver"]

        adjusted_surfaces = _surfaces_at_height(record["surfaces"], record["height_m"], height_m)
        adjusted_features = _features_at_height(record["features"], height_m)
        portals = _portals_at_height(record["portals"], height_m)
        selected_connections = _connections_for_room(rooms, portals, adjusted_features, str(selected["id"]))
        receiver_connections = _connections_for_room(rooms, portals, adjusted_features, str(receiver_room["id"]))
        selected_exterior = _exterior_exposures(adjusted_features, str(selected["id"]))
        receiver_exterior = _exterior_exposures(adjusted_features, str(receiver_room["id"]))
        compact_rooms = [
            {"id": room["id"], "type": room["type"], "area_m2": room["area_m2"]}
            for room in rooms
        ]
        multi_room = {
            "enabled": True,
            "accelerator": "numba_jit",
            "rooms": rooms,
            "portals": portals,
            "source_room_id": selected["id"],
            "receiver_room_id": receiver_room["id"],
            "route_room_ids": route_rooms,
            "route_portal_ids": route_portals,
            "door_state": "interior_doors_open",
            "window_state": "closed",
        }
        room_metadata = {
            "shape": "resplan",
            "geometry_model": "resplan_multi_room_extrusion",
            "opening_model": "vertical_portal_apertures_v1",
            "connectivity_model": "verified_resplan_portal_graph_v1",
            "source_room_id": selected["id"],
            "receiver_room_id": receiver_room["id"],
            "connections": selected_connections,
            "exterior_exposures": selected_exterior,
            "boundary_features": adjusted_features,
            "surface_segments": adjusted_surfaces,
            "resplan": {
                "index": resolved,
                "source_index": record["source_index"],
                "sample_id": record.get("sample_id"),
                "room_id": selected["id"],
                "receiver_room_id": receiver_room["id"],
                "room_type": selected["type"],
                "receiver_room_type": receiver_room["type"],
                "meters_per_unit": record["meters_per_unit"],
            },
            "multi_room": multi_room,
        }
        return {
            "dataset": {
                "index": resolved,
                "source_index": record["source_index"],
                "sample_id": record.get("sample_id"),
                "count": self.count,
                "eligible_count": self.count,
                "filtered_count": 0,
                "source_count": int(self.metadata.get("source_record_count", self.count)),
                "unit_type": record.get("unit_type", "Unknown"),
                "net_area_m2": record.get("net_area_m2"),
                "gross_area_m2": record.get("gross_area_m2"),
                "meters_per_unit": record["meters_per_unit"],
                "scale_source": record["scale_source"],
                "wall_depth_m": record["wall_depth_m"],
                "eligible": True,
                "resource_schema": RESOURCE_SCHEMA_VERSION,
            },
            "rooms": compact_rooms,
            "selected_room": {
                "id": selected["id"],
                "type": selected["type"],
                "area_m2": selected["area_m2"],
                "connections": selected_connections,
                "exterior_exposures": selected_exterior,
            },
            "receiver_room": {
                "id": receiver_room["id"],
                "type": receiver_room["type"],
                "area_m2": receiver_room["area_m2"],
                "connections": receiver_connections,
                "exterior_exposures": receiver_exterior,
            },
            "room": {
                "shape": "resplan",
                "size": [float(record["size"][0]), float(record["size"][1]), float(height_m)],
                "corners": record["corners"],
                "metadata": room_metadata,
            },
            "source": [float(value) for value in source],
            "receiver": [float(value) for value in receiver],
            "plan": {
                "size": record["size"],
                "simulation_origin": [0.0, 0.0],
                "rooms": [
                    {
                        "id": room["id"],
                        "type": room["type"],
                        "selected": room["id"] == selected["id"],
                        "receiver": room["id"] == receiver_room["id"],
                        "polygon": room["corners"],
                    }
                    for room in rooms
                ],
                "apertures": [],
            },
        }


def _room_by_id(room_by_id: Mapping[str, dict[str, Any]], room_id: str | None) -> dict[str, Any]:
    try:
        return room_by_id[str(room_id)]
    except KeyError as exc:
        raise ValueError(f"unknown room {room_id!r}; expected one of: {', '.join(room_by_id)}") from exc


def _default_room(rooms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    preference = {"living": 0, "bedroom": 1, "kitchen": 2, "bathroom": 3, "storage": 4, "balcony": 5}
    return min(rooms, key=lambda room: (preference.get(str(room["type"]), 9), -float(room["area_m2"]), str(room["id"])))


def _auto_receiver_room(record: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    rooms = record["rooms"]
    room_by_id = {str(room["id"]): room for room in rooms}
    reachable = _reachable_room_ids(record.get("portals", []), str(selected["id"])) - {str(selected["id"])}
    candidates = [room_by_id[room_id] for room_id in reachable if room_id in room_by_id]
    if not candidates:
        return dict(selected)
    preference = {"bedroom": 0, "kitchen": 1, "living": 2, "bathroom": 3, "storage": 4, "balcony": 5}
    return min(candidates, key=lambda room: (preference.get(str(room["type"]), 9), -float(room["area_m2"]), str(room["id"])))


def _reachable_room_ids(portals: Sequence[Mapping[str, Any]], start: str) -> set[str]:
    adjacency: dict[str, set[str]] = {start: set()}
    for portal in portals:
        room_ids = [str(value) for value in portal.get("room_ids", [])]
        if len(room_ids) != 2 or not bool(portal.get("open", False)):
            continue
        first, second = room_ids
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    seen = {start}
    queue = [start]
    while queue:
        room_id = queue.pop(0)
        for neighbor in adjacency.get(room_id, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _portal_route(
    portals: Sequence[Mapping[str, Any]],
    source_room: str,
    receiver_room: str,
) -> tuple[list[str], list[str]]:
    if source_room == receiver_room:
        return [source_room], []
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for portal in portals:
        room_ids = [str(value) for value in portal.get("room_ids", [])]
        if len(room_ids) != 2 or not bool(portal.get("open", False)):
            continue
        first, second = room_ids
        portal_id = str(portal["id"])
        adjacency.setdefault(first, []).append((second, portal_id))
        adjacency.setdefault(second, []).append((first, portal_id))
    queue = [source_room]
    parent: dict[str, tuple[str, str] | None] = {source_room: None}
    while queue and receiver_room not in parent:
        current = queue.pop(0)
        for neighbor, portal_id in adjacency.get(current, []):
            if neighbor not in parent:
                parent[neighbor] = (current, portal_id)
                queue.append(neighbor)
    if receiver_room not in parent:
        return [], []
    rooms = [receiver_room]
    portal_ids: list[str] = []
    while rooms[-1] != source_room:
        previous, portal_id = parent[rooms[-1]]  # type: ignore[misc]
        rooms.append(previous)
        portal_ids.append(portal_id)
    rooms.reverse()
    portal_ids.reverse()
    return rooms, portal_ids


def _sample_room_xy(corners: Sequence[Sequence[float]], rng: random.Random, margin: float) -> tuple[float, float]:
    min_x = min(float(point[0]) for point in corners)
    max_x = max(float(point[0]) for point in corners)
    min_y = min(float(point[1]) for point in corners)
    max_y = max(float(point[1]) for point in corners)
    best: tuple[float, float] | None = None
    best_clearance = -1.0
    for required_margin in (max(0.0, margin), max(0.0, margin * 0.5), 0.0):
        for _ in range(768):
            candidate = (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            if not point_in_polygon(candidate, corners):
                continue
            clearance = _boundary_distance(candidate, corners)
            if clearance > best_clearance:
                best, best_clearance = candidate, clearance
            if clearance >= required_margin:
                return candidate
    if best is not None:
        return best
    raise ValueError("could not sample an interior point from the room polygon")


def _boundary_distance(point: Sequence[float], corners: Sequence[Sequence[float]]) -> float:
    return min(
        _point_segment_distance(point, corners[index], corners[(index + 1) % len(corners)])
        for index in range(len(corners))
    )


def _point_segment_distance(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _stable_seed(index: int, source_room: str, receiver_room: str) -> int:
    digest = hashlib.sha256(f"{index}:{source_room}:{receiver_room}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _surfaces_at_height(
    surfaces: Sequence[Mapping[str, Any]],
    base_height: float,
    target_height: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in surfaces:
        item = dict(raw)
        z_min = float(item.get("z_min", 0.0))
        z_max = float(item.get("z_max", base_height))
        if abs(z_min - base_height) < 1e-6:
            z_min = float(target_height)
        if abs(z_max - base_height) < 1e-6:
            z_max = float(target_height)
        z_min = min(max(0.0, z_min), float(target_height))
        z_max = min(max(0.0, z_max), float(target_height))
        if z_max - z_min < 0.015:
            continue
        item["z_min"] = round(z_min, 6)
        item["z_max"] = round(z_max, 6)
        out.append(item)
    return out


def _features_at_height(features: Sequence[Mapping[str, Any]], target_height: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in features:
        item = dict(raw)
        sill = min(max(0.0, float(item.get("sill_height_m", 0.0))), float(target_height))
        item["sill_height_m"] = round(sill, 6)
        item["height_m"] = round(min(float(item.get("height_m", target_height)), max(0.0, target_height - sill)), 6)
        out.append(item)
    return out


def _portals_at_height(portals: Sequence[Mapping[str, Any]], target_height: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in portals:
        item = dict(raw)
        sill = min(max(0.0, float(item.get("sill_height_m", 0.0))), float(target_height))
        item["sill_height_m"] = round(sill, 6)
        item["height_m"] = round(min(float(item.get("height_m", target_height)), max(0.0, target_height - sill)), 6)
        out.append(item)
    return out


def _connections_for_room(
    rooms: Sequence[Mapping[str, Any]],
    portals: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    room_id: str,
) -> list[dict[str, Any]]:
    room_by_id = {str(room["id"]): room for room in rooms}
    connections: list[dict[str, Any]] = []
    for portal in portals:
        room_ids = [str(value) for value in portal.get("room_ids", [])]
        if room_id not in room_ids or len(room_ids) != 2:
            continue
        target_id = room_ids[1] if room_ids[0] == room_id else room_ids[0]
        target = room_by_id.get(target_id)
        if target is None:
            continue
        connections.append({
            "type": "via_door" if portal.get("type") == "door" else "via_opening",
            "legacy_type": None,
            "target_room_id": target_id,
            "target_type": target["type"],
            "walkable": bool(portal.get("open", False)),
            "outdoor": target.get("type") == "balcony",
            "verified": True,
            "portal_id": portal.get("id"),
        })
    for feature in features:
        if (
            room_id in feature.get("room_ids", [])
            and feature.get("type") == "door"
            and feature.get("connection") == "outdoor_entry"
        ):
            connections.append({
                "type": "outdoor_entry",
                "legacy_type": "direct",
                "target_room_id": feature.get("id", "front_door"),
                "target_type": "front_door",
                "walkable": bool(feature.get("open", False)),
                "outdoor": True,
                "verified": True,
                "portal_id": None,
            })
    return sorted(connections, key=lambda item: (not item["outdoor"], item["target_type"], item["target_room_id"], item["type"]))


def _exterior_exposures(features: Sequence[Mapping[str, Any]], room_id: str) -> list[dict[str, Any]]:
    return [
        {
            "feature_id": feature.get("id"),
            "feature_index": index,
            "type": feature["type"],
            "connection": feature["connection"],
        }
        for index, feature in enumerate(features)
        if room_id in feature.get("room_ids", []) and feature.get("connection") in _EXTERIOR_CONNECTIONS
    ]
