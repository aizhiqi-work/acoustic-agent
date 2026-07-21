from __future__ import annotations

from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box

from acoustic_agent import AcousticAgent, FloorplanBuilder, SimConfig, generate_floorplan_furniture


def _footprint(item):
    width, depth = item["size"][:2]
    shape = box(-width * 0.5, -depth * 0.5, width * 0.5, depth * 0.5)
    shape = affinity.rotate(shape, item.get("rotation", 0.0), origin=(0.0, 0.0))
    return affinity.translate(shape, *item["position"])


def _sample_scene():
    spec = FloorplanBuilder.from_text(
        "10m x 8m, two bedrooms, one living room, one kitchen, one bathroom",
        seed=42,
    )
    return spec, FloorplanBuilder.compile(spec, source_room="living_0", receiver_room="bedroom_1", seed=42)


def test_semantic_furnishing_is_deterministic_inside_rooms_and_clear_of_doors() -> None:
    _spec, scene = _sample_scene()
    metadata = scene["room"]["metadata"]
    first = generate_floorplan_furniture(
        metadata,
        compactness="balanced",
        seed=91,
        exclude_points=(scene["source"], scene["receiver"]),
    )
    second = generate_floorplan_furniture(
        metadata,
        compactness="balanced",
        seed=91,
        exclude_points=(scene["source"], scene["receiver"]),
    )

    assert first == second
    assert first["summary"]["object_count"] >= 12
    assert {"sofa", "bed", "table", "cabinet", "fridge", "rug", "curtain", "washing_machine"}.issubset(
        {item["type"] for item in first["objects"]}
    )
    rooms = {
        room["id"]: Polygon(room["corners"])
        for room in metadata["multi_room"]["rooms"]
    }
    door_zones = {}
    for feature in metadata["boundary_features"]:
        if feature["type"] not in {"door", "opening"}:
            continue
        for index, room_id in enumerate(feature["room_ids"]):
            segment = feature["segments"][min(index, len(feature["segments"]) - 1)]
            door_zones.setdefault(room_id, []).append(LineString(segment).buffer(0.72, cap_style=2))
    excluded = [Point(point[:2]).buffer(0.48) for point in (scene["source"], scene["receiver"])]
    for item in first["objects"]:
        footprint = _footprint(item)
        room_id = item["placement"]["room_id"]
        assert rooms[room_id].buffer(1e-7).covers(footprint)
        assert not any(footprint.intersects(zone) for zone in door_zones.get(room_id, ()))
        assert not any(footprint.buffer(0.07).intersects(zone) for zone in excluded)


def test_compactness_increases_layout_and_manual_objects_are_respected() -> None:
    _spec, scene = _sample_scene()
    metadata = scene["room"]["metadata"]
    manual = [{
        "id": "manual_table",
        "type": "table",
        "position": [2.5, 2.5],
        "size": [1.35, 0.78, 0.74],
        "rotation": 0.0,
    }]
    sparse = generate_floorplan_furniture(metadata, compactness="sparse", seed=12, existing_objects=manual)
    balanced = generate_floorplan_furniture(metadata, compactness="balanced", seed=12, existing_objects=manual)
    compact = generate_floorplan_furniture(metadata, compactness="compact", seed=12, existing_objects=manual)

    assert sparse["summary"]["object_count"] <= balanced["summary"]["object_count"]
    assert balanced["summary"]["object_count"] <= compact["summary"]["object_count"]
    manual_footprint = _footprint(manual[0])
    for item in compact["objects"]:
        if item["type"] in {"rug", "curtain"}:
            continue
        assert not _footprint(item).buffer(0.07).intersects(manual_footprint)


def test_floorplan_agent_accepts_automatic_furnishing_configuration() -> None:
    spec, _scene = _sample_scene()
    agent = AcousticAgent.from_floorplan_spec(
        spec,
        source_room="living_0",
        receiver_room="bedroom_1",
        furnishing={"mode": "auto", "compactness": "sparse", "seed": 37},
        config=SimConfig(
            fs=8000,
            duration_s=0.02,
            reflections_enabled=False,
            diffraction_enabled=False,
        ),
    )
    result = agent.run()

    assert agent.furnishing is not None
    assert agent.furnishing["summary"]["compactness"] == "sparse"
    assert len(agent.room.metadata["objects"]) == agent.furnishing["summary"]["object_count"]
    assert all(item["placement"]["source"] == "semantic_auto" for item in agent.room.metadata["objects"])
    assert result.rir.shape == (1, 160)
