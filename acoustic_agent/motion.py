from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import heapq
import math
from typing import Any, Mapping, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .geometry import point_in_polygon
from .models import Room


@dataclass(frozen=True)
class MotionFrame:
    phase: float
    source: tuple[float, float, float]
    receiver: tuple[float, float, float]


def sample_motion(
    room: Room,
    source: Sequence[float],
    receiver: Sequence[float],
    *,
    mode: str = "approach",
    moving: str = "source",
    distance_m: float = 0.8,
    keyframes: int | None = None,
    keyframe_spacing_m: float = 0.25,
    seed: int = 42,
    wall_margin_m: float = 0.15,
) -> dict[str, Any]:
    """Sample a short, room-constrained source or receiver trajectory."""
    motion_mode = str(mode).lower()
    moving_role = str(moving).lower()
    if motion_mode not in {"static", "approach", "random", "recede", "pass_by", "through_portal"}:
        raise ValueError("motion mode must be static, approach, random, recede, pass_by, or through_portal")
    if moving_role not in {"source", "receiver"}:
        raise ValueError("moving must be source or receiver")

    src = _point3(source)
    rcv = _point3(receiver)
    if motion_mode == "static":
        frame = MotionFrame(0.0, src, rcv)
        return _motion_payload(motion_mode, moving_role, 0.0, 0.0, (frame,), "static")

    requested = max(0.05, min(6.0, float(distance_m)))
    frame_count = lambda distance: _motion_frame_count(distance, keyframes, keyframe_spacing_m)
    moving_start = src if moving_role == "source" else rcv
    target = rcv if moving_role == "source" else src
    corners = _role_corners(room, moving_role)
    portal_route = _portal_motion_route(room, src, rcv, moving_role)
    if motion_mode == "random":
        route, actual = _random_room_route(
            moving_start,
            corners,
            requested,
            wall_margin_m,
            seed,
        )
        count = frame_count(actual)
        positions = tuple(reversed(_sample_polyline(route, actual, count, eased=False)))
        frames = tuple(
            MotionFrame(
                index / max(count - 1, 1),
                position if moving_role == "source" else src,
                position if moving_role == "receiver" else rcv,
            )
            for index, position in enumerate(positions)
        )
        payload = _motion_payload(motion_mode, moving_role, requested, actual, frames, "random_room_route")
        payload["random_seed"] = int(seed)
        return payload
    if motion_mode == "through_portal":
        if len(portal_route) < 2:
            raise ValueError("through_portal motion requires connected source and receiver rooms")
        route_length = _polyline_length(portal_route)
        actual = min(
            max(0.05, route_length - 0.3),
            _first_portal_crossing_distance(room, portal_route, moving_role) + 0.6,
        )
        count = frame_count(actual)
        positions = _snap_positions_to_rooms(room, _sample_polyline(portal_route, actual, count))
        frames = tuple(
            MotionFrame(
                index / max(count - 1, 1),
                position if moving_role == "source" else src,
                position if moving_role == "receiver" else rcv,
            )
            for index, position in enumerate(positions)
        )
        return _motion_payload(
            motion_mode,
            moving_role,
            actual,
            actual,
            frames,
            "portal_crossing_smoothstep",
        )
    if motion_mode == "approach" and len(portal_route) >= 2:
        route_length = _polyline_length(portal_route)
        actual = min(requested, max(0.05, route_length - 0.3))
        count = frame_count(actual)
        positions = _snap_positions_to_rooms(room, _sample_polyline(portal_route, actual, count))
        frames = tuple(
            MotionFrame(
                index / max(count - 1, 1),
                position if moving_role == "source" else src,
                position if moving_role == "receiver" else rcv,
            )
            for index, position in enumerate(positions)
        )
        return _motion_payload(motion_mode, moving_role, requested, actual, frames, "portal_route_smoothstep")

    if motion_mode == "approach":
        route_xy = _visibility_path(
            np.asarray(moving_start[:2], dtype=float),
            np.asarray(target[:2], dtype=float),
            corners,
            wall_margin_m=wall_margin_m,
        )
        route = tuple((float(point[0]), float(point[1]), moving_start[2]) for point in route_xy)
        route_length = _polyline_length(route)
        actual = min(requested, max(0.05, route_length - 0.3))
        count = frame_count(actual)
        positions = _sample_polyline(route, actual, count, eased=False)
        frames = tuple(
            MotionFrame(
                index / max(count - 1, 1),
                position if moving_role == "source" else src,
                position if moving_role == "receiver" else rcv,
            )
            for index, position in enumerate(positions)
        )
        return _motion_payload(motion_mode, moving_role, requested, actual, frames, "room_shortest_path")

    direction_target = portal_route[1] if len(portal_route) >= 2 else target
    direction = _unit2((direction_target[0] - moving_start[0], direction_target[1] - moving_start[1]))
    if motion_mode == "recede":
        direction = (-direction[0], -direction[1])
    elif motion_mode == "pass_by":
        direction = (-direction[1], direction[0])

    count = frame_count(requested)

    def positions_for(distance: float) -> tuple[tuple[float, float, float], ...]:
        positions = []
        for index in range(count):
            phase = index / max(count - 1, 1)
            eased = _smootherstep(phase)
            offset = (eased - 0.5) * distance if motion_mode == "pass_by" else eased * distance
            positions.append((
                moving_start[0] + direction[0] * offset,
                moving_start[1] + direction[1] * offset,
                moving_start[2],
            ))
        return tuple(positions)

    actual = _maximum_safe_distance(positions_for, requested, corners, wall_margin_m)
    if keyframes is None:
        count = frame_count(actual)
    positions = positions_for(actual)
    frames = tuple(
        MotionFrame(
            index / max(count - 1, 1),
            position if moving_role == "source" else src,
            position if moving_role == "receiver" else rcv,
        )
        for index, position in enumerate(positions)
    )
    return _motion_payload(motion_mode, moving_role, requested, actual, frames, "local_smoothstep")


def _motion_payload(
    mode: str,
    moving: str,
    requested_distance_m: float,
    actual_distance_m: float,
    frames: tuple[MotionFrame, ...],
    path_model: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "moving": moving,
        "requested_distance_m": round(float(requested_distance_m), 4),
        "distance_m": round(float(actual_distance_m), 4),
        "keyframes": len(frames),
        "keyframe_spacing_m": round(
            float(actual_distance_m) / max(len(frames) - 1, 1),
            4,
        ),
        "path_model": path_model,
        "frames": [
            {
                "phase": round(frame.phase, 6),
                "source": [round(value, 6) for value in frame.source],
                "receiver": [round(value, 6) for value in frame.receiver],
            }
            for frame in frames
        ],
    }


def _motion_frame_count(distance_m: float, keyframes: int | None, spacing_m: float) -> int:
    if keyframes is not None:
        count = max(3, min(17, int(keyframes)))
        return count + 1 if count % 2 == 0 else count
    spacing = max(0.05, min(2.0, float(spacing_m)))
    return max(3, min(65, int(math.ceil(max(0.0, float(distance_m)) / spacing)) + 1))


def _random_room_route(
    anchor: tuple[float, float, float],
    corners: Sequence[Sequence[float]],
    requested_distance_m: float,
    wall_margin_m: float,
    seed: int,
) -> tuple[tuple[tuple[float, float, float], ...], float]:
    polygon = Polygon(corners)
    inset = polygon.buffer(-max(0.0, float(wall_margin_m)), join_style=2)
    domain = inset if isinstance(inset, Polygon) and not inset.is_empty else polygon
    random_state = int(seed) & 0xFFFFFFFF

    def next_random() -> float:
        nonlocal random_state
        random_state = (1664525 * random_state + 1013904223) & 0xFFFFFFFF
        return random_state / 4294967296.0

    min_x, min_y, max_x, max_y = domain.bounds
    best_route: tuple[tuple[float, float, float], ...] | None = None
    best_length = 0.0
    qualified_routes: dict[tuple[float, float], tuple[tuple[float, float, float], ...]] = {}
    anchor_xy = np.asarray(anchor[:2], dtype=float)
    for _ in range(32):
        candidate = np.asarray([
            min_x + next_random() * (max_x - min_x),
            min_y + next_random() * (max_y - min_y),
        ], dtype=float)
        if not domain.covers(Point(candidate)):
            continue
        route_xy = _visibility_path(
            anchor_xy,
            candidate,
            corners,
            wall_margin_m=wall_margin_m,
        )
        route = tuple((float(point[0]), float(point[1]), anchor[2]) for point in route_xy)
        route_length = _polyline_length(route)
        if route_length > best_length:
            best_route = route
            best_length = route_length
        if route_length + 1e-9 >= requested_distance_m:
            sampled_start = _sample_polyline(route, requested_distance_m, 2, eased=False)[-1]
            key = (round(sampled_start[0], 3), round(sampled_start[1], 3))
            qualified_routes.setdefault(key, route)
    if qualified_routes:
        routes = list(qualified_routes.values())
        return routes[min(len(routes) - 1, int(next_random() * len(routes)))], float(requested_distance_m)
    if best_route is None or best_length <= 1e-9:
        fallback = (anchor, anchor)
        return fallback, 0.0
    return best_route, min(float(requested_distance_m), best_length)


def _portal_motion_route(
    room: Room,
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
    moving: str,
) -> tuple[tuple[float, float, float], ...]:
    metadata = room.metadata if isinstance(room.metadata, Mapping) else {}
    multi_room = metadata.get("multi_room") if isinstance(metadata.get("multi_room"), Mapping) else {}
    route_rooms = [str(value) for value in multi_room.get("route_room_ids", [])]
    route_portals = [str(value) for value in multi_room.get("route_portal_ids", [])]
    if len(route_rooms) != len(route_portals) + 1 or not route_portals:
        return ()
    room_by_id = {
        str(item.get("id")): item
        for item in multi_room.get("rooms", [])
        if isinstance(item, Mapping)
    }
    portal_by_id = {
        str(item.get("id")): item
        for item in multi_room.get("portals", [])
        if isinstance(item, Mapping) and bool(item.get("open", False))
    }
    if any(room_id not in room_by_id for room_id in route_rooms) or any(portal_id not in portal_by_id for portal_id in route_portals):
        return ()

    xy_points: list[np.ndarray] = [np.asarray(source[:2], dtype=float)]
    current = xy_points[0]
    for index, portal_id in enumerate(route_portals):
        current_room = route_rooms[index]
        next_room = route_rooms[index + 1]
        portal = portal_by_id[portal_id]
        room_points = portal.get("room_points") if isinstance(portal.get("room_points"), Mapping) else {}
        current_side = np.asarray(room_points.get(current_room, portal.get("center")), dtype=float)
        next_side = np.asarray(room_points.get(next_room, portal.get("center")), dtype=float)
        segment = _visibility_path(current, current_side, room_by_id[current_room].get("corners", []))
        xy_points.extend(segment[1:])
        if float(np.linalg.norm(next_side - xy_points[-1])) > 1e-6:
            xy_points.append(next_side)
        current = next_side
    final_segment = _visibility_path(current, np.asarray(receiver[:2], dtype=float), room_by_id[route_rooms[-1]].get("corners", []))
    xy_points.extend(final_segment[1:])
    deduplicated = _deduplicate_xy(xy_points)
    source_to_receiver = tuple((float(point[0]), float(point[1]), source[2]) for point in deduplicated)
    if moving == "receiver":
        return tuple((point[0], point[1], receiver[2]) for point in reversed(source_to_receiver))
    return source_to_receiver


def _visibility_path(
    start: np.ndarray,
    end: np.ndarray,
    corners: Sequence[Sequence[float]],
    *,
    wall_margin_m: float = 0.0,
) -> list[np.ndarray]:
    polygon = Polygon(corners)
    if polygon.is_empty or not polygon.is_valid:
        return [start, end]
    domain = polygon.buffer(1e-4, join_style=2)
    visibility_boundary = polygon
    margin = max(0.0, float(wall_margin_m))
    if margin > 0.0:
        inset = polygon.buffer(-margin, join_style=2)
        if (
            isinstance(inset, Polygon)
            and not inset.is_empty
            and inset.covers(Point(start))
            and inset.covers(Point(end))
        ):
            domain = inset
            visibility_boundary = inset
    if domain.covers(LineString((start, end))):
        return [start, end]
    nodes = [start, end, *(np.asarray(point, dtype=float) for point in list(visibility_boundary.exterior.coords)[:-1])]
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for first, point in enumerate(nodes):
        for second in range(first + 1, len(nodes)):
            if not domain.covers(LineString((point, nodes[second]))):
                continue
            length = float(np.linalg.norm(nodes[second] - point))
            adjacency[first].append((second, length))
            adjacency[second].append((first, length))
    queue = [(0.0, 0)]
    distances = {0: 0.0}
    previous: dict[int, int] = {}
    while queue:
        cost, node = heapq.heappop(queue)
        if node == 1:
            break
        if cost > distances.get(node, math.inf):
            continue
        for neighbor, edge_length in adjacency[node]:
            candidate = cost + edge_length
            if candidate >= distances.get(neighbor, math.inf):
                continue
            distances[neighbor] = candidate
            previous[neighbor] = node
            heapq.heappush(queue, (candidate, neighbor))
    if 1 not in distances:
        return [start, end]
    indices = [1]
    while indices[-1] != 0:
        indices.append(previous[indices[-1]])
    return [nodes[index] for index in reversed(indices)]


def _deduplicate_xy(points: Sequence[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for point in points:
        if not result or float(np.linalg.norm(point - result[-1])) > 1e-6:
            result.append(point)
    return result


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(points[index], points[index + 1]) for index in range(len(points) - 1))


def _first_portal_crossing_distance(
    room: Room,
    route: Sequence[Sequence[float]],
    moving: str,
) -> float:
    metadata = room.metadata if isinstance(room.metadata, Mapping) else {}
    multi_room = metadata.get("multi_room") if isinstance(metadata.get("multi_room"), Mapping) else {}
    room_ids = [str(value) for value in multi_room.get("route_room_ids", [])]
    portal_ids = [str(value) for value in multi_room.get("route_portal_ids", [])]
    if not portal_ids or len(room_ids) != len(portal_ids) + 1:
        return 0.0
    portal_by_id = {
        str(item.get("id")): item
        for item in multi_room.get("portals", [])
        if isinstance(item, Mapping)
    }
    portal_id = portal_ids[0] if moving == "source" else portal_ids[-1]
    next_room = room_ids[1] if moving == "source" else room_ids[-2]
    portal = portal_by_id.get(portal_id, {})
    room_points = portal.get("room_points") if isinstance(portal.get("room_points"), Mapping) else {}
    target = room_points.get(next_room, portal.get("center"))
    if not isinstance(target, Sequence) or len(target) < 2:
        return 0.0
    return _distance_along_polyline(route, (float(target[0]), float(target[1])))


def _distance_along_polyline(points: Sequence[Sequence[float]], target: tuple[float, float]) -> float:
    nearest_distance = math.inf
    nearest_travel = 0.0
    traveled = 0.0
    target_xy = np.asarray(target, dtype=float)
    for index in range(len(points) - 1):
        start = np.asarray(points[index][:2], dtype=float)
        end = np.asarray(points[index + 1][:2], dtype=float)
        vector = end - start
        squared = float(np.dot(vector, vector))
        mix = 0.0 if squared <= 1e-12 else max(0.0, min(1.0, float(np.dot(target_xy - start, vector) / squared)))
        projection = start + mix * vector
        distance = float(np.linalg.norm(target_xy - projection))
        segment_length = float(np.linalg.norm(vector))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_travel = traveled + mix * segment_length
        traveled += segment_length
    return nearest_travel


def _sample_polyline(
    points: Sequence[tuple[float, float, float]],
    distance: float,
    count: int,
    *,
    eased: bool = True,
) -> tuple[tuple[float, float, float], ...]:
    segment_lengths = [math.dist(points[index], points[index + 1]) for index in range(len(points) - 1)]
    output = []
    for index in range(count):
        phase = index / max(count - 1, 1)
        travel = (_smootherstep(phase) if eased else phase) * distance
        segment_index = 0
        while segment_index < len(segment_lengths) - 1 and travel > segment_lengths[segment_index]:
            travel -= segment_lengths[segment_index]
            segment_index += 1
        length = max(segment_lengths[segment_index], 1e-12)
        mix = min(1.0, travel / length)
        start, end = points[segment_index], points[segment_index + 1]
        output.append(tuple(start[axis] + (end[axis] - start[axis]) * mix for axis in range(3)))
    return tuple(output)


def _snap_positions_to_rooms(
    room: Room,
    positions: Sequence[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], ...]:
    metadata = room.metadata if isinstance(room.metadata, Mapping) else {}
    multi_room = metadata.get("multi_room") if isinstance(metadata.get("multi_room"), Mapping) else {}
    rooms = [item for item in multi_room.get("rooms", []) if isinstance(item, Mapping)]
    portal_points = [
        point
        for portal in multi_room.get("portals", [])
        if isinstance(portal, Mapping) and bool(portal.get("open", False))
        for point in (portal.get("room_points", {}) or {}).values()
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if not rooms or not portal_points:
        return tuple(positions)
    snapped = []
    for position in positions:
        if any(
            isinstance(item.get("corners"), Sequence)
            and point_in_polygon(position[:2], item["corners"])
            for item in rooms
        ):
            snapped.append(position)
            continue
        nearest = min(portal_points, key=lambda point: math.hypot(position[0] - float(point[0]), position[1] - float(point[1])))
        snapped.append((float(nearest[0]), float(nearest[1]), position[2]))
    return tuple(snapped)


def _smootherstep(value: float) -> float:
    bounded = max(0.0, min(1.0, float(value)))
    return bounded**3 * (bounded * (bounded * 6.0 - 15.0) + 10.0)


def _maximum_safe_distance(factory: Any, requested: float, corners: Sequence[Sequence[float]], margin: float) -> float:
    if all(_safe_point(point, corners, margin) for point in factory(requested)):
        return requested
    low, high = 0.0, requested
    for _ in range(20):
        middle = 0.5 * (low + high)
        if all(_safe_point(point, corners, margin) for point in factory(middle)):
            low = middle
        else:
            high = middle
    return low


def _role_corners(room: Room, role: str) -> tuple[tuple[float, float], ...]:
    metadata = room.metadata if isinstance(room.metadata, Mapping) else {}
    room_id = metadata.get(f"{role}_room_id")
    multi_room = metadata.get("multi_room") if isinstance(metadata.get("multi_room"), Mapping) else {}
    for candidate in multi_room.get("rooms", []):
        if isinstance(candidate, Mapping) and candidate.get("id") == room_id:
            corners = candidate.get("corners")
            if isinstance(corners, Sequence) and len(corners) >= 3:
                return tuple((float(point[0]), float(point[1])) for point in corners)
    return tuple((float(point[0]), float(point[1])) for point in room.corners)


def _safe_point(point: Sequence[float], corners: Sequence[Sequence[float]], margin: float) -> bool:
    return point_in_polygon((float(point[0]), float(point[1])), corners) and _boundary_distance(point, corners) >= margin


def _boundary_distance(point: Sequence[float], corners: Sequence[Sequence[float]]) -> float:
    return min(
        _segment_distance(point, corners[index], corners[(index + 1) % len(corners)])
        for index in range(len(corners))
    )


def _segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    factor = 0.0 if denominator <= 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + factor * dx), py - (ay + factor * dy))


def _unit2(value: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(value[0], value[1])
    return (1.0, 0.0) if length <= 1e-9 else (value[0] / length, value[1] / length)


def _point3(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("motion points must contain x, y, z")
    point = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in point):
        raise ValueError("motion point contains a non-finite value")
    return point


def motion_room_metadata(
    metadata: Mapping[str, Any],
    source: Sequence[float],
    receiver: Sequence[float],
) -> dict[str, Any]:
    """Update room ownership and portal route for a moving source/receiver frame."""
    updated = copy.deepcopy(dict(metadata))
    multi_room = updated.get("multi_room")
    if not isinstance(multi_room, dict) or not bool(multi_room.get("enabled", False)):
        return updated
    rooms = [item for item in multi_room.get("rooms", []) if isinstance(item, Mapping)]
    portals = [
        item
        for item in multi_room.get("portals", [])
        if isinstance(item, Mapping) and bool(item.get("open", False))
    ]
    source_room = _room_id_for_motion_point(source, rooms, str(multi_room.get("source_room_id", "")))
    receiver_room = _room_id_for_motion_point(receiver, rooms, str(multi_room.get("receiver_room_id", "")))
    route_rooms, route_portals = _open_portal_route(source_room, receiver_room, portals)
    if not route_rooms:
        return updated
    updated["source_room_id"] = source_room
    updated["receiver_room_id"] = receiver_room
    multi_room["source_room_id"] = source_room
    multi_room["receiver_room_id"] = receiver_room
    multi_room["route_room_ids"] = route_rooms
    multi_room["route_portal_ids"] = route_portals
    return updated


def room_for_motion_frame(room: Room, source: Sequence[float], receiver: Sequence[float]) -> Room:
    metadata = room.metadata if isinstance(room.metadata, Mapping) else {}
    updated = motion_room_metadata(metadata, source, receiver)
    if updated == metadata:
        return room
    return Room(
        id=room.id,
        name=room.name,
        # Multi-room tracing must retain the complete apartment footprint.
        # Room ownership only selects portal topology and material statistics;
        # replacing this polygon at a doorway would also replace the traced
        # floor and ceiling on a single motion frame.
        corners=room.corners,
        height_m=room.height_m,
        materials=room.materials,
        metadata=updated,
    )


def _room_id_for_motion_point(
    point: Sequence[float],
    rooms: Sequence[Mapping[str, Any]],
    fallback: str,
) -> str:
    candidates = []
    for item in rooms:
        corners = item.get("corners")
        if not isinstance(corners, Sequence) or len(corners) < 3:
            continue
        if point_in_polygon((float(point[0]), float(point[1])), corners):
            candidates.append((_boundary_distance(point, corners), str(item.get("id"))))
    if candidates:
        return max(candidates)[1]
    return fallback


def _open_portal_route(
    source_room: str,
    receiver_room: str,
    portals: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    if not source_room or not receiver_room:
        return [], []
    if source_room == receiver_room:
        return [source_room], []
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for portal in portals:
        room_ids = portal.get("room_ids")
        if not isinstance(room_ids, Sequence) or len(room_ids) != 2:
            continue
        first, second = str(room_ids[0]), str(room_ids[1])
        portal_id = str(portal.get("id"))
        adjacency.setdefault(first, []).append((second, portal_id))
        adjacency.setdefault(second, []).append((first, portal_id))
    queue = deque([source_room])
    previous: dict[str, tuple[str, str]] = {}
    visited = {source_room}
    while queue:
        current = queue.popleft()
        if current == receiver_room:
            break
        for neighbor, portal_id in adjacency.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            previous[neighbor] = (current, portal_id)
            queue.append(neighbor)
    if receiver_room not in visited:
        return [], []
    route_rooms = [receiver_room]
    route_portals = []
    while route_rooms[-1] != source_room:
        prior, portal_id = previous[route_rooms[-1]]
        route_portals.append(portal_id)
        route_rooms.append(prior)
    route_rooms.reverse()
    route_portals.reverse()
    return route_rooms, route_portals
