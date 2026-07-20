from __future__ import annotations

from copy import deepcopy

from acoustic_agent import AcousticAgent, FloorplanBuilder, SimConfig
from acoustic_agent.custom_floorplan import compile_floorplan_spec, validate_floorplan_spec


def test_local_text_generator_is_deterministic_and_understands_chinese_room_counts() -> None:
    first = FloorplanBuilder.from_text("12米 x 9米，三室两厅一厨两卫，一个储物间", seed=17)
    second = FloorplanBuilder.from_text("12米 x 9米，三室两厅一厨两卫，一个储物间", seed=17)

    assert first == second
    assert first["outer_boundary"] == [[0.0, 0.0], [12.0, 0.0], [12.0, 9.0], [0.0, 9.0]]
    room_types = [room["type"] for room in first["rooms"]]
    assert room_types.count("bedroom") == 3
    assert room_types.count("living") == 2
    assert room_types.count("bathroom") == 2
    assert room_types.count("storage") == 1


def test_local_text_generator_understands_english_number_words_and_by_dimensions() -> None:
    spec = FloorplanBuilder.from_text(
        "12m by 9m, three bedrooms, two living rooms, one kitchen, two bathrooms, one storage room",
        seed=17,
    )

    assert spec["outer_boundary"] == [[0.0, 0.0], [12.0, 0.0], [12.0, 9.0], [0.0, 9.0]]
    room_types = [room["type"] for room in spec["rooms"]]
    assert room_types.count("bedroom") == 3
    assert room_types.count("living") == 2
    assert room_types.count("bathroom") == 2
    assert room_types.count("storage") == 1


def test_generated_spec_has_connected_doors_windows_and_solver_surfaces() -> None:
    spec = FloorplanBuilder.from_text("10m x 8m, two bedrooms, one living room, one kitchen, one bathroom", seed=42)
    report = FloorplanBuilder.validate(spec)
    scene = FloorplanBuilder.compile(spec, source_room="living_0", receiver_room="bedroom_1", seed=11)

    assert report["valid"] is True
    assert report["summary"] == {"rooms": 5, "doors": 5, "windows": 5, "area_m2": 80.0}
    metadata = scene["room"]["metadata"]
    assert metadata["multi_room"]["route_portal_ids"]
    assert metadata["custom_floorplan"]["validation"]["summary"] == report["summary"]
    assert {surface["type"] for surface in metadata["surface_segments"]} == {"wall", "door", "window"}
    assert scene["selected_room"]["id"] == "living_0"
    assert scene["receiver_room"]["id"] == "bedroom_1"


def test_validator_rejects_overlap_and_disconnected_rooms() -> None:
    spec = FloorplanBuilder.from_text("8m x 6m，一室一厅一厨一卫", seed=2)
    broken = deepcopy(spec)
    broken["rooms"][1]["corners"] = deepcopy(broken["rooms"][0]["corners"])
    broken["openings"] = [item for item in broken["openings"] if item["connection"] != "interior_room"]

    report = validate_floorplan_spec(broken)

    assert report["valid"] is False
    assert any("overlap" in message for message in report["errors"])
    assert any("not connected" in message for message in report["errors"])


def test_floorplan_spec_api_runs_without_a_model_provider() -> None:
    spec = FloorplanBuilder.from_text("9m x 7m，两室一厅一厨一卫", seed=8)
    agent = AcousticAgent.from_floorplan_spec(
        spec,
        seed=8,
        source_room="living_0",
        receiver_room="bedroom_0",
        config=SimConfig(
            fs=8000,
            duration_s=0.02,
            reflections_enabled=False,
            diffraction_enabled=False,
        ),
    )

    result = agent.run()

    assert result.rir.shape == (1, 160)
    assert agent.placement["mode"] == "cross_room"
    assert result.metadata["floorplan"]["custom"] is True


def test_compile_uses_normalized_metric_coordinates() -> None:
    spec = FloorplanBuilder.from_text("8m x 6m，一室一厅一厨一卫", seed=3)
    shifted = deepcopy(spec)
    shifted["outer_boundary"] = [[x + 100, y - 20] for x, y in shifted["outer_boundary"]]
    for room in shifted["rooms"]:
        room["corners"] = [[x + 100, y - 20] for x, y in room["corners"]]
    for opening in shifted["openings"]:
        opening["segment"] = [[x + 100, y - 20] for x, y in opening["segment"]]

    scene = compile_floorplan_spec(shifted)

    assert scene["room"]["corners"] == [[0.0, 0.0], [8.0, 0.0], [8.0, 6.0], [0.0, 6.0]]


def test_bottom_left_vlm_coordinates_are_normalized_without_a_mirrored_plan() -> None:
    expected = FloorplanBuilder.from_text("9m x 7m，两室一厅一厨一卫", seed=9)
    raw = deepcopy(expected)
    raw["coordinate_system"] = "cartesian_bottom_left"
    depth = 7.0
    raw["outer_boundary"] = [[x, depth - y] for x, y in raw["outer_boundary"]]
    for room in raw["rooms"]:
        room["corners"] = [[x, depth - y] for x, y in room["corners"]]
    for opening in raw["openings"]:
        opening["segment"] = [[x, depth - y] for x, y in opening["segment"]]

    normalized = FloorplanBuilder.validate(raw)["spec"]

    assert normalized["coordinate_system"] == "image_top_left"
    assert normalized["outer_boundary"] == expected["outer_boundary"]
    assert normalized["rooms"] == expected["rooms"]
    assert normalized["openings"] == expected["openings"]
    assert normalized["provenance"]["coordinate_transform"] == "cartesian_bottom_left_to_image_top_left"


def test_codex_handoff_prompt_describes_the_validated_schema() -> None:
    prompt = FloorplanBuilder.vlm_prompt()

    assert "Return exactly one JSON object" in prompt
    assert '"schema_version": 1' in prompt
    assert '"coordinate_system": "image_top_left"' in prompt
    assert "Do not rotate, mirror, or vertically flip" in prompt
    assert 'connection="interior_room"' in prompt
    assert "check polygon coverage" in prompt
