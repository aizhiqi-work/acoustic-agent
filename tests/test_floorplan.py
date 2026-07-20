import math

import networkx as nx
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, box

import acoustic_agent.steam_rt as steam_rt
from acoustic_agent.engine import _estimate_material_rt60, simulate_rir
from acoustic_agent.geometry import make_room
from acoustic_agent.models import SimConfig
from acoustic_agent.floorplan import FloorplanDataset, _metric_scale, _plan_profile, scene_from_record
from acoustic_agent.floorplan_web_server import _floorplan_viewer_html
from acoustic_agent.steam_rt import RoomRayScene


def _record():
    living = box(0.0, 0.0, 50.0, 50.0)
    bedroom = box(50.0, 0.0, 100.0, 50.0)
    graph = nx.Graph()
    graph.add_node("living_0", geometry=living, type="living", area=living.area)
    graph.add_node("bedroom_0", geometry=bedroom, type="bedroom", area=bedroom.area)
    graph.add_edge("living_0", "bedroom_0", type="via_door")
    return {
        "id": 42,
        "unitType": "Apartment",
        "inner": MultiPolygon([box(0.0, 0.0, 100.0, 50.0)]),
        "net_area": 50.0,
        "area": 55.0,
        "wall_depth": 2.0,
        "door": MultiPolygon([box(48.0, 20.0, 52.0, 30.0)]),
        "front_door": Polygon(),
        "window": MultiPolygon([box(-1.0, 10.0, 1.0, 20.0)]),
        "graph": graph,
    }


def test_floorplan_same_room_uses_global_scene_and_zero_portal_route():
    scene = scene_from_record(_record(), index=7, room_id="living_0")
    multi_room = scene["room"]["metadata"]["multi_room"]

    assert scene["dataset"]["meters_per_unit"] == 0.1
    assert scene["selected_room"]["area_m2"] == 25.0
    assert scene["receiver_room"]["id"] == "living_0"
    assert scene["room"]["size"] == [10.0, 5.0, 2.8]
    assert scene["room"]["metadata"]["opening_model"] == "vertical_portal_apertures_v1"
    assert scene["room"]["metadata"]["geometry_model"] == "floorplan_multi_room_extrusion"
    assert multi_room["route_room_ids"] == ["living_0"]
    assert multi_room["route_portal_ids"] == []
    assert {feature["type"] for feature in scene["room"]["metadata"]["boundary_features"]} == {"door", "window"}
    assert next(feature for feature in scene["room"]["metadata"]["boundary_features"] if feature["type"] == "door")["open"] is True
    assert scene["selected_room"]["exterior_exposures"] == [{
        "feature_id": "window_0_living_0",
        "feature_index": 1,
        "type": "window",
        "connection": "outdoor_facade",
    }]

    room_polygon = Polygon(next(room["corners"] for room in multi_room["rooms"] if room["id"] == "living_0"))
    assert room_polygon.covers(Point(scene["source"][:2]))
    assert room_polygon.covers(Point(scene["receiver"][:2]))


def test_floorplan_same_room_uses_open_door_aperture_and_vertical_window_surfaces():
    scene = scene_from_record(_record(), index=7, room_id="living_0")
    room = make_room(
        "floorplan",
        size=scene["room"]["size"],
        corners=scene["room"]["corners"],
    )
    room.metadata.update(scene["room"]["metadata"])

    ray_scene = RoomRayScene(room)
    names = [surface.name for surface in ray_scene.surfaces]

    assert not any(name.endswith("_door") for name in names)
    assert any("_window_0_glass" in surface.name and surface.z_min == 0.9 and surface.z_max == 2.1 for surface in ray_scene.surfaces)
    assert any(name.endswith("_lintel") for name in names)
    assert any("_window_0_lower" in surface.name and surface.z_min == 0.0 and surface.z_max == 0.9 for surface in ray_scene.surfaces)


def test_floorplan_same_and_cross_room_modes_share_identical_geometry():
    same_room = scene_from_record(_record(), index=7, room_id="bedroom_0", receiver_room_id="bedroom_0")
    cross_room = scene_from_record(_record(), index=7, room_id="bedroom_0", receiver_room_id="living_0")
    same_metadata = same_room["room"]["metadata"]
    cross_metadata = cross_room["room"]["metadata"]

    assert same_room["room"]["size"] == cross_room["room"]["size"]
    assert same_room["room"]["corners"] == cross_room["room"]["corners"]
    assert same_metadata["surface_segments"] == cross_metadata["surface_segments"]
    assert same_metadata["boundary_features"] == cross_metadata["boundary_features"]
    assert same_metadata["multi_room"]["rooms"] == cross_metadata["multi_room"]["rooms"]
    assert same_metadata["multi_room"]["portals"] == cross_metadata["multi_room"]["portals"]
    assert same_metadata["multi_room"]["route_portal_ids"] == []
    assert cross_metadata["multi_room"]["route_portal_ids"] == ["door_0"]
    assert same_room["source"] == cross_room["source"]
    assert same_room["receiver"] != cross_room["receiver"]


def test_floorplan_isolated_same_room_has_zero_length_route():
    record = _record()
    record["inner"] = MultiPolygon([box(0.0, 0.0, 50.0, 50.0)])
    record["door"] = Polygon()
    record["graph"].remove_node("bedroom_0")

    scene = scene_from_record(record, index=7, room_id="living_0", receiver_room_id="living_0")

    assert scene["room"]["metadata"]["multi_room"]["route_room_ids"] == ["living_0"]
    assert scene["room"]["metadata"]["multi_room"]["route_portal_ids"] == []


def test_floorplan_workbench_reuses_the_main_layout_with_dataset_controls():
    html = _floorplan_viewer_html()

    assert 'data-scene-source="floorplan"' in html
    assert 'id="floorplanIdx"' in html
    assert 'id="floorplanRoom"' in html
    assert 'id="floorplanReceiverRoom"' in html
    assert 'id="layerPortal"' in html
    assert 'id="sourceDirectivityPane"' in html
    assert 'id="receiverPane"' in html


def test_floorplan_uses_gross_area_proxy_when_net_area_is_corrupt():
    record = _record()
    record["net_area"] = 5_000_000.0

    scale, source = _metric_scale(record)

    assert source == "gross_area_proxy"
    assert math.isclose(scale, math.sqrt(0.75 * 55.0 / 5_000.0))


def test_floorplan_deduplicates_room_nodes_and_normalizes_open_adjacency():
    record = _record()
    bathroom = box(10.0, 10.0, 20.0, 20.0)
    record["graph"].add_node("bathroom_0", geometry=bathroom, type="bathroom", area=bathroom.area)
    record["graph"].add_node("bathroom_1", geometry=bathroom, type="bathroom", area=bathroom.area)
    record["graph"]["living_0"]["bedroom_0"]["type"] = "adjacency"

    scene = scene_from_record(record, index=7, room_id="living_0")

    assert [room["id"] for room in scene["rooms"]].count("bathroom_0") == 1
    assert "bathroom_1" not in {room["id"] for room in scene["rooms"]}
    connection = next(item for item in scene["selected_room"]["connections"] if item["target_room_id"] == "bedroom_0")
    assert connection["type"] == "via_door"
    assert connection["portal_id"] == "door_0"


def test_floorplan_profile_filters_stairs_and_index_navigation_skips_them():
    record = _record()
    record["stair"] = MultiPolygon([box(20.0, 20.0, 30.0, 30.0)])
    profile = _plan_profile(record, index=2)
    dataset = FloorplanDataset.__new__(FloorplanDataset)
    dataset.records = [None] * 6
    dataset.eligible_indices = [0, 1, 3, 5]

    assert not profile["eligible"]
    assert "stair_or_multilevel" in profile["filter_reasons"]
    assert dataset.resolve_index(2, "nearest") == 1
    assert dataset.resolve_index(1, "next") == 3
    assert dataset.resolve_index(3, "previous") == 1


def test_floorplan_cross_room_scene_builds_global_portals_and_vertical_openings():
    scene = scene_from_record(
        _record(),
        index=7,
        room_id="living_0",
        receiver_room_id="bedroom_0",
    )
    multi_room = scene["room"]["metadata"]["multi_room"]

    assert scene["room"]["size"] == [10.0, 5.0, 2.8]
    assert scene["receiver_room"]["id"] == "bedroom_0"
    assert multi_room["enabled"] is True
    assert multi_room["accelerator"] == "numba_jit"
    assert multi_room["route_room_ids"] == ["living_0", "bedroom_0"]
    assert multi_room["route_portal_ids"] == ["door_0"]
    assert multi_room["portals"][0]["height_m"] == 2.1
    assert multi_room["portals"][0]["width_m"] > 0.8

    room_polygons = {item["id"]: Polygon(item["corners"]) for item in multi_room["rooms"]}
    assert room_polygons["living_0"].covers(Point(scene["source"][:2]))
    assert room_polygons["bedroom_0"].covers(Point(scene["receiver"][:2]))
    lintels = [
        item for item in scene["room"]["metadata"]["surface_segments"]
        if item["name"].endswith("_lintel")
    ]
    assert len(lintels) == 2
    assert all(item["z_min"] == 2.1 and item["z_max"] == 2.8 for item in lintels)


def test_floorplan_global_scene_keeps_entry_doors_closed_but_opens_balcony_doors():
    record = _record()
    balcony = box(100.0, 0.0, 120.0, 50.0)
    record["inner"] = MultiPolygon([box(0.0, 0.0, 120.0, 50.0)])
    record["graph"].add_node("balcony_0", geometry=balcony, type="balcony", area=balcony.area)
    record["graph"].add_edge("bedroom_0", "balcony_0", type="via_door")
    record["door"] = MultiPolygon([
        box(48.0, 20.0, 52.0, 30.0),
        box(98.0, 20.0, 102.0, 30.0),
    ])
    record["front_door"] = box(10.0, -2.0, 20.0, 2.0)

    scene = scene_from_record(
        record,
        index=7,
        room_id="living_0",
        receiver_room_id="bedroom_0",
    )
    multi_room = scene["room"]["metadata"]["multi_room"]
    features = scene["room"]["metadata"]["boundary_features"]
    surfaces = scene["room"]["metadata"]["surface_segments"]

    assert multi_room["route_portal_ids"] == ["door_0"]
    balcony_door = next(feature for feature in features if feature["id"] == "door_1")
    front_door = next(feature for feature in features if feature["id"] == "front_door_0")
    assert balcony_door["open"] is True
    assert front_door["open"] is False
    assert set(balcony_door["room_ids"]) == {"bedroom_0", "balcony_0"}
    assert front_door["room_ids"] == ["living_0"]
    assert not any(item["name"] == "bedroom_0_door_1_door" for item in surfaces)
    assert any(item["name"] == "bedroom_0_door_1_lintel" and item["type"] == "wall" for item in surfaces)
    assert any(item["name"] == "living_0_front_door_0_door" and item["type"] == "door" for item in surfaces)

    balcony_scene = scene_from_record(
        record,
        index=7,
        room_id="balcony_0",
        receiver_room_id="bedroom_0",
    )
    balcony_route = balcony_scene["room"]["metadata"]["multi_room"]
    assert balcony_scene["selected_room"]["id"] == "balcony_0"
    assert balcony_scene["receiver_room"]["id"] == "bedroom_0"
    assert balcony_route["route_portal_ids"] == ["door_1"]
    assert any(room["id"] == "balcony_0" for room in balcony_scene["rooms"])


def test_cross_room_solver_routes_around_wall_through_open_door_with_shared_energy_trace_paths():
    scene = scene_from_record(
        _record(),
        index=7,
        room_id="living_0",
        receiver_room_id="bedroom_0",
    )
    room = make_room("floorplan", size=scene["room"]["size"], corners=scene["room"]["corners"])
    room.metadata.update(scene["room"]["metadata"])
    result = simulate_rir(
        room,
        (1.0, 0.8, 1.4),
        (9.0, 0.8, 1.4),
        config=SimConfig(
            fs=8000,
            duration_s=0.2,
            rt_num_rays=256,
            rt_num_bounces=3,
            rt_duration_s=0.2,
            rt_visual_num_rays=128,
            rt_visual_num_bounces=3,
            late_tail=False,
        ),
    )

    direct = next(path for path in result.paths if path.kind == "direct_transmitted")
    portal = next(path for path in result.paths if path.kind == "portal_path")
    steam = result.metadata["steam_audio"]
    assert portal.metadata["route_portal_ids"] == ["door_0"]
    assert portal.metadata["aperture_pressure_gain"] == 1.0
    assert 0.0 < portal.metadata["aperture_pressure_gain_estimate"] < 1.0
    assert portal.metadata["aperture_attenuation_applied"] is False
    assert portal.metadata["segment_visibility_verified"] is True
    assert portal.distance_m > direct.distance_m
    assert portal.gain > direct.gain * 10.0
    assert steam["portal_propagation"]["contributes_to_rir"] is True
    assert steam["portal_propagation"]["accelerator"] == "python_visibility_graph"
    assert steam["reflections"]["accelerator"] == "numba"
    assert steam["rt_visual"]["accelerator"] == "numba"
    assert steam["rt_visual"]["model"] == "listener_space_energy_trace_representatives"
    assert steam["rt_visual"]["shares_energy_trace"] is True
    assert "footprint_filter" not in steam["rt_visual"]
    assert steam["diffraction"]["path_count"] == 0
    assert result.metadata["solver_pipeline"][1] == "portal_pathing"
    assert np.isfinite(result.rir).all()
    assert float(np.sum(result.rir * result.rir)) > 0.0


def test_cross_room_simulation_adapts_to_at_least_96_bounces():
    scene = scene_from_record(_record(), index=7, room_id="living_0", receiver_room_id="bedroom_0")
    room = make_room("floorplan", size=scene["room"]["size"], corners=scene["room"]["corners"])
    room.metadata.update(scene["room"]["metadata"])

    effective, metadata = steam_rt._adaptive_reflection_config(
        RoomRayScene(room),
        SimConfig(rt_num_rays=32768, rt_num_bounces=64),
    )

    assert metadata["applied"] is True
    assert metadata["requested"] == 64
    assert metadata["effective"] >= 96
    assert effective.rt_num_bounces == metadata["effective"]


def test_coupled_room_late_decay_includes_semantic_furniture_absorption():
    scene = scene_from_record(_record(), index=7, room_id="living_0", receiver_room_id="bedroom_0")
    empty = make_room("floorplan", size=scene["room"]["size"], corners=scene["room"]["corners"])
    empty.metadata.update(scene["room"]["metadata"])
    furnished = make_room("floorplan", size=scene["room"]["size"], corners=scene["room"]["corners"])
    furnished.metadata.update(scene["room"]["metadata"])
    furnished.metadata["objects"] = [{
        "id": "sofa_0",
        "type": "sofa",
        "semantic": "sofa_couch",
        "absorption_class": "highly_absorptive",
        "position": [2.0, 2.0],
        "size": [2.2, 0.9, 0.8],
        "z": 0.4,
        "rotation": 0.0,
    }]

    empty_rt = _estimate_material_rt60(empty)
    furnished_rt = _estimate_material_rt60(furnished)
    object_areas = furnished_rt["coupled_decay"]["object_absorption_area_m2"]

    assert max(object_areas.values()) > 0.0
    assert np.mean(list(furnished_rt["coupled_rt60_bands"].values())) < np.mean(list(empty_rt["coupled_rt60_bands"].values()))


def test_same_room_global_solver_uses_multi_room_rt_without_portal_path():
    scene = scene_from_record(
        _record(),
        index=7,
        room_id="living_0",
        receiver_room_id="living_0",
    )
    room = make_room("floorplan", size=scene["room"]["size"], corners=scene["room"]["corners"])
    room.metadata.update(scene["room"]["metadata"])
    ray_scene = RoomRayScene(room)
    result = simulate_rir(
        room,
        scene["source"],
        scene["receiver"],
        config=SimConfig(
            fs=8000,
            duration_s=0.2,
            rt_num_rays=256,
            rt_num_bounces=3,
            rt_duration_s=0.2,
            rt_visual_num_rays=128,
            rt_visual_num_bounces=3,
            late_tail=False,
        ),
    )

    assert ray_scene.is_multi_room is True
    assert ray_scene.is_cross_room is False
    assert not any(path.kind == "portal_path" for path in result.paths)
    assert result.metadata["steam_audio"]["portal_propagation"]["path_count"] == 0
    assert np.isfinite(result.rir).all()
    assert float(np.sum(result.rir * result.rir)) > 0.0


def test_cross_room_jit_intersections_match_python_for_portal_wall_spans():
    scene = scene_from_record(
        _record(),
        index=7,
        room_id="living_0",
        receiver_room_id="bedroom_0",
    )
    room = make_room("floorplan", size=scene["room"]["size"], corners=scene["room"]["corners"])
    room.metadata.update(scene["room"]["metadata"])
    ray_scene = RoomRayScene(room)
    arrays = steam_rt._scene_kernel_arrays(ray_scene)
    rng = np.random.default_rng(4107)
    directions = rng.normal(size=(256, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = (
        np.asarray([1.0, 2.5, 1.4]),
        np.asarray([1.0, 2.5, 2.5]),
        np.asarray([9.0, 2.5, 1.4]),
    )

    for origin in origins:
        for direction in directions:
            expected = ray_scene.closest_hit(origin, direction)
            surface_index, distance, *_normal = steam_rt._closest_hit_jit(
                origin,
                direction,
                arrays["kinds"],
                arrays["wall_a"],
                arrays["wall_delta"],
                arrays["wall_z"],
                arrays["z_values"],
                arrays["box_center"],
                arrays["box_axis_u"],
                arrays["box_axis_v"],
                arrays["box_half"],
                arrays["box_z"],
                arrays["normals"],
                arrays["corners"],
                arrays["height"],
            )
            assert (surface_index >= 0) is bool(expected["valid"])
            if expected["valid"]:
                assert np.isclose(distance, expected["distance"], rtol=1e-10, atol=1e-10)
