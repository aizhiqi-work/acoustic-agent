import json
from importlib.resources import files

from acoustic_agent import AcousticAgent, FloorplanResource, ResPlanResource, SimConfig
from acoustic_agent.engine import _estimate_material_rt60
from acoustic_agent.geometry import point_in_polygon


def test_floorplan_resource_loads_compiled_scene_by_dense_index():
    resource = FloorplanResource()

    assert len(resource) == 15376
    assert resource.stats()["storage"] == "sqlite_zlib_json"
    assert resource.resolve_index(-10) == 0
    assert resource.resolve_index(len(resource) + 10) == len(resource) - 1

    scene = resource.scene(0)
    assert scene["dataset"]["index"] == 0
    assert scene["dataset"]["source_index"] == 0
    assert scene["dataset"]["resource_schema"] == 1
    assert scene["room"]["metadata"]["geometry_model"] == "floorplan_multi_room_extrusion"
    assert len(scene["rooms"]) >= 1


def test_floorplan_resource_carries_resplan_citation_and_data_terms():
    resource_dir = files("acoustic_agent").joinpath("resources", "floorplan")
    source = json.loads(resource_dir.joinpath("source.json").read_text(encoding="utf-8"))
    terms = resource_dir.joinpath("DATA_LICENSE.md").read_text(encoding="utf-8")

    assert source["upstream_dataset"] == "ResPlan"
    assert source["authors"] == ["Mohamed Abouagour", "Eleftherios Garyfallidis"]
    assert source["doi"] == "10.48550/arXiv.2508.14006"
    assert source["upstream_license"] == "CC-BY-NC-SA-4.0"
    assert source["compiled_record_count"] == len(FloorplanResource())
    assert "CC BY-NC-SA 4.0" in terms


def test_pre_floorplan_python_names_remain_compatible():
    assert ResPlanResource is FloorplanResource
    agent = AcousticAgent.from_resplan(
        0,
        seed=123,
        config=SimConfig(
            fs=8000,
            duration_s=0.01,
            reflections_enabled=False,
            diffraction_enabled=False,
        ),
    )

    assert agent.resplan == agent.floorplan
    assert agent.run().rir.shape == (1, 80)


def test_floorplan_resource_samples_reproducible_room_placements():
    resource = FloorplanResource()

    first = resource.sample_placement(0, placement="random", seed=42)
    second = resource.sample_placement(0, placement="random", seed=42)
    same = resource.sample_placement(0, placement="same_room", seed=7)
    cross = resource.sample_placement(0, placement="cross_room", seed=7)

    assert first == second
    assert same["source_room"] == same["receiver_room"]
    assert cross["source_room"] != cross["receiver_room"]

    record = resource.record(0)
    room_by_id = {room["id"]: room for room in record["rooms"]}
    for placement in (same, cross):
        assert point_in_polygon(placement["source"][:2], room_by_id[placement["source_room"]]["corners"])
        assert point_in_polygon(placement["receiver"][:2], room_by_id[placement["receiver_room"]]["corners"])


def test_acoustic_agent_from_floorplan_runs_without_manual_positions():
    agent = AcousticAgent.from_floorplan(
        0,
        seed=123,
        config=SimConfig(
            fs=8000,
            duration_s=0.04,
            reflections_enabled=False,
            diffraction_enabled=False,
        ),
    )

    result = agent.run()

    assert result.rir.shape == (1, 320)
    assert len(agent.rooms) >= 1
    assert result.metadata["placement"]["source_room"] == agent.placement["source_room"]
    assert result.metadata["floorplan"]["index"] == 0


def test_floorplan_material_rt60_uses_source_room_surfaces_and_open_portals():
    agent = AcousticAgent.from_floorplan(
        0,
        placement="same_room",
        seed=42,
        material_seed=1451557868,
        material_profile={
            "wall": "auto",
            "floor": "auto",
            "ceiling": "auto",
            "door": "auto",
            "window": "auto",
        },
    )

    estimate = _estimate_material_rt60(agent.room)

    assert estimate["scope"] == "source_room"
    assert estimate["room_id"] == "bedroom_0"
    assert estimate["model"] == "sabine_explicit_surfaces_with_opening_loss"
    assert estimate["opening_area_m2"] == 6.5632
    assert estimate["rt60_s"] == 0.585
    assert estimate["coupled_decay"]["model"] == "coupled_room_energy_matrix"
    assert estimate["coupled_decay"]["portal_count"] >= 1
    assert set(estimate["coupled_rt60_bands"]) == {"125", "250", "500", "1000", "2000", "4000"}
    assert all(value > 0.0 for value in estimate["coupled_rt60_bands"].values())
