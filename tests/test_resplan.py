import math

import networkx as nx
from shapely.geometry import MultiPolygon, Point, Polygon, box

from acoustic_agent.geometry import make_room
from acoustic_agent.resplan import ResPlanDataset, _metric_scale, _plan_profile, scene_from_record
from acoustic_agent.resplan_web_server import _resplan_viewer_html
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


def test_resplan_scene_restores_metric_scale_and_same_room_points():
    scene = scene_from_record(_record(), index=7, room_id="living_0")

    assert scene["dataset"]["meters_per_unit"] == 0.1
    assert scene["selected_room"]["area_m2"] == 25.0
    assert scene["room"]["size"] == [5.0, 5.0, 2.8]
    assert scene["room"]["metadata"]["opening_model"] == "full_height_equivalent_boundary_material_v1"
    assert {feature["type"] for feature in scene["room"]["metadata"]["boundary_features"]} == {"door", "window"}
    assert any(exposure["type"] == "window" for exposure in scene["selected_room"]["exterior_exposures"])

    room_polygon = Polygon(scene["room"]["corners"])
    assert room_polygon.covers(Point(scene["source"][:2]))
    assert room_polygon.covers(Point(scene["receiver"][:2]))


def test_resplan_surface_segments_become_door_and_window_acoustic_surfaces():
    scene = scene_from_record(_record(), index=7, room_id="living_0")
    room = make_room(
        "resplan",
        size=scene["room"]["size"],
        corners=scene["room"]["corners"],
    )
    room.metadata.update(scene["room"]["metadata"])

    names = [surface.name for surface in RoomRayScene(room).surfaces]

    assert any(name.startswith("door_") for name in names)
    assert any(name.startswith("window_") for name in names)
    assert any(name.startswith("wall_") for name in names)


def test_resplan_workbench_reuses_the_main_layout_with_dataset_controls():
    html = _resplan_viewer_html()

    assert 'data-scene-source="resplan"' in html
    assert 'id="resplanIdx"' in html
    assert 'id="resplanRoom"' in html
    assert 'id="sourceDirectivityPane"' in html
    assert 'id="receiverPane"' in html


def test_resplan_uses_gross_area_proxy_when_net_area_is_corrupt():
    record = _record()
    record["net_area"] = 5_000_000.0

    scale, source = _metric_scale(record)

    assert source == "gross_area_proxy"
    assert math.isclose(scale, math.sqrt(0.75 * 55.0 / 5_000.0))


def test_resplan_deduplicates_room_nodes_and_normalizes_open_adjacency():
    record = _record()
    bathroom = box(10.0, 10.0, 20.0, 20.0)
    record["graph"].add_node("bathroom_0", geometry=bathroom, type="bathroom", area=bathroom.area)
    record["graph"].add_node("bathroom_1", geometry=bathroom, type="bathroom", area=bathroom.area)
    record["graph"]["living_0"]["bedroom_0"]["type"] = "adjacency"

    scene = scene_from_record(record, index=7, room_id="living_0")

    assert [room["id"] for room in scene["rooms"]].count("bathroom_0") == 1
    assert "bathroom_1" not in {room["id"] for room in scene["rooms"]}
    assert any(connection["type"] == "via_opening" for connection in scene["selected_room"]["connections"])


def test_resplan_profile_filters_stairs_and_index_navigation_skips_them():
    record = _record()
    record["stair"] = MultiPolygon([box(20.0, 20.0, 30.0, 30.0)])
    profile = _plan_profile(record, index=2)
    dataset = ResPlanDataset.__new__(ResPlanDataset)
    dataset.records = [None] * 6
    dataset.eligible_indices = [0, 1, 3, 5]

    assert not profile["eligible"]
    assert "stair_or_multilevel" in profile["filter_reasons"]
    assert dataset.resolve_index(2, "nearest") == 1
    assert dataset.resolve_index(1, "next") == 3
    assert dataset.resolve_index(3, "previous") == 1
