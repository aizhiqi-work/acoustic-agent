import math
import json

import numpy as np

from acoustic_agent import AcousticAgent, FloorplanBuilder, SimConfig
from acoustic_agent.geometry import point_in_polygon
from acoustic_agent.motion import room_for_motion_frame


def test_acoustic_agent_runs_the_common_workflow_directly():
    agent = AcousticAgent(
        room=[3.0, 2.5, 2.4],
        config=SimConfig(
            fs=8000,
            duration_s=0.04,
            rt_num_rays=1,
            rt_num_bounces=1,
            rt_duration_s=0.04,
            diffraction_enabled=False,
        ),
    )

    result = agent.run(source=[0.7, 0.8, 1.2], receiver=[2.2, 1.7, 1.2])

    assert result.rir.shape == (1, 320)


def test_create_unifies_geometry_floorplan_and_custom_scene_inputs():
    geometry = AcousticAgent.create(
        scene="geometry",
        room=[3.0, 2.5, 2.4],
        source=[0.7, 0.8, 1.2],
        mic=[2.2, 1.7, 1.2],
        microphone="mono",
        directivity="omni",
        sample_rate=8000,
        rir_length=0.04,
        seed=7,
    )
    floorplan = AcousticAgent.create(
        scene="floorplan",
        idx=0,
        placement="same_room",
        seed=7,
        config=SimConfig(fs=8000, duration_s=0.02, reflections_enabled=False, diffraction_enabled=False),
    )
    spec = FloorplanBuilder.from_text("6m x 5m, one bedroom and one living room", seed=7)
    custom = AcousticAgent.create(
        scene="custom",
        floorplan_spec=spec,
        seed=7,
        config=SimConfig(fs=8000, duration_s=0.02, reflections_enabled=False, diffraction_enabled=False),
    )

    assert geometry.scene_type == "geometry"
    assert geometry.default_source == (0.7, 0.8, 1.2)
    assert geometry.default_receiver == (2.2, 1.7, 1.2)
    assert geometry.config.seed == 7
    assert geometry.config.fs == 8000
    assert floorplan.scene_type == "floorplan"
    assert custom.scene_type == "custom"


def test_run_accepts_a_compact_motion_description():
    agent = AcousticAgent.create(
        room=[4.0, 3.0, 2.8],
        source=[0.6, 1.5, 1.4],
        receiver=[3.2, 1.5, 1.4],
        config=SimConfig(fs=8000, duration_s=0.02, reflections_enabled=False, diffraction_enabled=False),
    )

    result = agent.run(motion={"mode": "approach", "moving": "receiver", "distance_m": 0.5, "keyframes": 3})

    assert len(result.frames) == 3
    assert result.motion["mode"] == "approach"
    assert all(frame.rir.shape == (1, 160) for frame in result.frames)


def test_agent_batch_accepts_plain_coordinate_pairs():
    agent = AcousticAgent.create(
        room=[4.0, 3.0, 2.8],
        source=[0.6, 1.5, 1.4],
        receiver=[3.2, 1.5, 1.4],
        config=SimConfig(fs=8000, duration_s=0.02, reflections_enabled=False, diffraction_enabled=False),
    )
    batch = agent.run_batch([
        ([0.6, 1.5, 1.4], [3.2, 1.5, 1.4]),
        {"id": "second", "source": [0.8, 1.0, 1.4], "mic": [2.8, 2.0, 1.4], "seed": 9},
    ])

    assert len(batch.items) == 2
    assert batch.pairs[1].id == "second"
    assert all(rir.shape == (1, 160) for rir in batch.rirs)


def test_run_many_supports_mixed_static_dynamic_jobs_and_saves_manifest(tmp_path):
    lightweight = SimConfig(fs=8000, duration_s=0.02, reflections_enabled=False, diffraction_enabled=False)
    batch = AcousticAgent.run_many([
        {
            "id": "geometry_static",
            "scene": "geometry",
            "room": [4.0, 3.0, 2.8],
            "source": [0.6, 1.5, 1.4],
            "receiver": [3.2, 1.5, 1.4],
            "config": lightweight,
        },
        {
            "id": "floorplan_dynamic",
            "scene": "floorplan",
            "idx": 0,
            "placement": "same_room",
            "seed": 7,
            "config": lightweight,
            "motion": {"mode": "approach", "distance_m": 0.25, "keyframes": 2},
        },
    ])
    destination = batch.save_npz(tmp_path / "dataset.npz")

    assert len(batch.items) == 2
    assert batch.items[0].rir.shape == (1, 160)
    assert len(batch.items[1].rirs) == 3
    with np.load(destination) as archive:
        manifest = json.loads(str(archive["manifest"]))
        assert manifest["results"][0]["id"] == "geometry_static"
        assert manifest["results"][1]["dynamic"] is True
        assert manifest["jobs"][0]["config"]["fs"] == 8000
        assert "rir_000001_frame_0002" in archive


def test_run_many_can_keep_successful_jobs_when_one_job_is_invalid():
    lightweight = SimConfig(fs=8000, duration_s=0.02, reflections_enabled=False, diffraction_enabled=False)
    batch = AcousticAgent.run_many([
        {
            "id": "valid",
            "room": [4.0, 3.0, 2.8],
            "source": [0.6, 1.5, 1.4],
            "receiver": [3.2, 1.5, 1.4],
            "config": lightweight,
        },
        {"id": "invalid", "scene": "floorplan"},
    ], on_error="skip")

    assert batch.succeeded == 1
    assert batch.failed == 1
    assert batch.jobs[0]["id"] == "valid"
    assert batch.errors[0]["id"] == "invalid"
    assert batch.errors[0]["type"] == "TypeError"


def test_acoustic_agent_quality_names_select_the_expected_tiers():
    assert AcousticAgent(quality="preview").config.rt_num_rays == 8192
    assert AcousticAgent(quality="simulation").config.rt_num_bounces == 64
    assert AcousticAgent(quality="fine").config.rt_num_rays == 65536
    assert AcousticAgent(quality="reference").config.rt_num_rays == 131072


def test_acoustic_agent_uses_shape_specific_room_parameters():
    cases = [
        ({"shape": "triangle", "size": [6, 4, 2.8], "apex": 0.25}, 3),
        ({"shape": "circle", "size": [6, 4, 2.8], "segments": 20}, 20),
        ({"shape": "polygon", "size": [6, 4, 2.8], "sides": 7, "irregularity": 0.2, "skew": 0.1}, 7),
        ({"shape": "l_shape", "size": [6, 4, 2.8], "cutout_width": 0.4, "cutout_depth": 0.5}, 6),
        ({"shape": "t_shape", "size": [6, 4, 2.8], "head_depth": 0.4, "stem_width": 0.3, "stem_offset": 0.6}, 8),
        ({"shape": "trapezoid", "size": [6, 4, 2.8], "top_width": 0.7, "top_offset": 0.4}, 4),
        ({"shape": "u_shape", "size": [6, 4, 2.8], "opening_width": 0.4, "opening_depth": 0.5, "opening_offset": 0.6}, 8),
        ({"shape": "fan_shape", "size": [6, 4, 2.8], "angle_deg": 100, "inner_radius": 0.3, "segments": 12}, 26),
    ]

    for room, corner_count in cases:
        assert len(AcousticAgent(room=room).room.corners) == corner_count

    triangle = AcousticAgent(room={"shape": "triangle", "size": [6, 4, 2.8], "apex": 0.25})
    assert triangle.room.corners[2] == (1.5, 4.0)


def test_acoustic_agent_uses_microphone_specific_parameters():
    linear = AcousticAgent(receiver_model={
        "type": "linear",
        "count": 3,
        "spacing_m": 0.1,
        "orientation_deg": 30,
    })
    circular = AcousticAgent(receiver_model={
        "type": "circular",
        "count": 6,
        "radius_m": 0.2,
        "orientation_deg": 15,
    })

    assert linear.receiver_model["type"] == "linear_array"
    assert len(linear.receiver_model["channels"]) == 3
    assert circular.receiver_model["type"] == "circular_array"
    assert len(circular.receiver_model["channels"]) == 6


def test_acoustic_agent_accepts_acoustic_geometry_as_a_public_parameter():
    geometry = [{
        "type": "panel",
        "semantic": "acoustic_treatment",
        "material": "plaster",
        "position": [2.0, 1.5],
        "z": 1.0,
        "size": [0.1, 2.0, 2.0],
        "rotation_deg": 0.0,
    }]

    agent = AcousticAgent(room=[4.0, 3.0, 2.8], acoustic_geometry=geometry)
    embedded = AcousticAgent(room={"shape": "rectangle", "size": [4.0, 3.0, 2.8], "objects": geometry})

    assert agent.room.metadata["objects"][0]["type"] == "panel"
    assert agent.room.metadata["objects"][0]["semantic"] == "acoustic_treatment"
    assert agent.room.metadata["objects"][0]["rotation"] == 0.0
    assert embedded.room.metadata["objects"] == agent.room.metadata["objects"]


def test_acoustic_agent_from_floorplan_accepts_semantic_furniture():
    agent = AcousticAgent.from_floorplan(
        0,
        seed=42,
        acoustic_geometry=[{
            "type": "sofa",
            "semantic": "sofa_couch",
            "material": "chairs_heavy_upholstered",
            "position": [2.5, 2.0],
            "z": 0.36,
            "size": [2.0, 0.9, 0.72],
            "rotation_deg": 10,
        }],
        config=SimConfig(
            fs=8000,
            duration_s=0.02,
            reflections_enabled=False,
            diffraction_enabled=False,
        ),
    )

    result = agent.run()

    assert agent.room.metadata["objects"][0]["semantic"] == "sofa_couch"
    assert agent.room.metadata["objects"][0]["material"] == "chairs_heavy_upholstered"
    assert result.rir.shape == (1, 160)


def test_agent_samples_safe_multi_keyframe_motion_and_runs_every_frame():
    config = SimConfig(
        fs=8000,
        duration_s=0.02,
        reflections_enabled=False,
        diffraction_enabled=False,
    )
    agent = AcousticAgent(room=[4.0, 3.0, 2.8], config=config)
    motion = agent.sample_motion(
        source=[0.4, 1.5, 1.4],
        receiver=[3.0, 1.5, 1.4],
        mode="recede",
        moving="source",
        distance_m=1.5,
        keyframes=9,
    )

    assert motion["keyframes"] == 9
    assert 0.0 < motion["distance_m"] < 1.5
    assert all(point_in_polygon(frame["source"], agent.room.corners) for frame in motion["frames"])
    assert [frame["phase"] for frame in motion["frames"]] == sorted(frame["phase"] for frame in motion["frames"])

    result = agent.run_dynamic(motion)
    assert len(result.frames) == 9
    assert all(frame.rir.shape == (1, 160) for frame in result.frames)


def test_geometry_travel_uses_room_route_and_quarter_meter_frame_spacing():
    agent = AcousticAgent(room={
        "shape": "u_shape",
        "size": [6.0, 10.0, 2.8],
        "opening_width": 0.42,
        "opening_depth": 0.82,
        "opening_offset": 0.5,
    })

    motion = agent.sample_motion(
        source=[5.766, 2.587, 1.231],
        receiver=[1.079, 6.252, 1.348],
        moving="receiver",
        distance_m=5.5998,
        keyframe_spacing_m=0.25,
    )

    receiver_points = [frame["receiver"] for frame in motion["frames"]]
    assert motion["path_model"] == "room_shortest_path"
    assert motion["distance_m"] == 5.5998
    assert motion["keyframes"] == 24
    assert motion["keyframe_spacing_m"] == 0.2435
    assert all(point_in_polygon(point, agent.room.corners) for point in receiver_points)
    assert all(
        point_in_polygon(
            [(first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5],
            agent.room.corners,
        )
        for first, second in zip(receiver_points, receiver_points[1:])
    )


def test_random_geometry_travel_keeps_length_and_returns_to_current_position():
    agent = AcousticAgent(room={
        "shape": "u_shape",
        "size": [6.0, 10.0, 2.8],
        "opening_width": 0.42,
        "opening_depth": 0.82,
        "opening_offset": 0.5,
    })
    arguments = {
        "source": [5.766, 2.587, 1.231],
        "receiver": [1.079, 6.252, 1.348],
        "mode": "random",
        "moving": "receiver",
        "distance_m": 3.0,
        "keyframe_spacing_m": 0.25,
    }

    first = agent.sample_motion(**arguments, seed=42)
    repeated = agent.sample_motion(**arguments, seed=42)
    different = agent.sample_motion(**arguments, seed=43)

    assert first["path_model"] == "random_room_route"
    assert first["distance_m"] == 3.0
    assert first["keyframes"] == 13
    assert first["frames"] == repeated["frames"]
    assert first["frames"][0]["receiver"] != different["frames"][0]["receiver"]
    assert first["frames"][-1]["receiver"] == arguments["receiver"]
    assert all(point_in_polygon(frame["receiver"], agent.room.corners) for frame in first["frames"])
    receiver_points = [frame["receiver"] for frame in first["frames"]]
    sampled_length = sum(math.dist(first_point, second_point) for first_point, second_point in zip(receiver_points, receiver_points[1:]))
    assert abs(sampled_length - first["distance_m"]) < 1e-5


def test_cross_room_approach_follows_portals_with_eased_keyframes():
    agent = AcousticAgent.from_floorplan(
        0,
        source_room="balcony_0",
        receiver_room="balcony_1",
        seed=42,
    )
    motion = agent.sample_motion(
        mode="approach",
        moving="source",
        distance_m=3.0,
        keyframes=13,
    )
    rooms = {room["id"]: room for room in agent.room.metadata["multi_room"]["rooms"]}
    positions = [frame["source"] for frame in motion["frames"]]
    steps = [
        ((positions[index + 1][0] - positions[index][0]) ** 2 + (positions[index + 1][1] - positions[index][1]) ** 2) ** 0.5
        for index in range(len(positions) - 1)
    ]

    assert motion["path_model"] == "portal_route_smoothstep"
    assert point_in_polygon(positions[0], rooms["balcony_0"]["corners"])
    assert any(point_in_polygon(position, rooms["bedroom_0"]["corners"]) for position in positions[1:])
    assert steps[0] < max(steps)
    assert steps[-1] < max(steps)


def test_floorplan_exact_positions_update_room_ownership_and_defaults():
    reference = AcousticAgent.from_floorplan(
        0,
        source_room="balcony_0",
        receiver_room="balcony_1",
        seed=42,
    )
    source = list(reference.default_receiver)
    receiver = list(reference.default_source)

    exact = AcousticAgent.from_floorplan(0, source=source, receiver=receiver, seed=7)

    assert list(exact.default_source) == source
    assert list(exact.default_receiver) == receiver
    assert exact.placement["source_room"] == "balcony_1"
    assert exact.placement["receiver_room"] == "balcony_0"
    assert exact.room.metadata["multi_room"]["route_room_ids"][0] == "balcony_1"


def test_cross_room_through_portal_crosses_next_door_in_both_directions():
    agent = AcousticAgent.from_floorplan(
        0,
        source_room="balcony_0",
        receiver_room="balcony_1",
        seed=42,
    )
    rooms = {room["id"]: room for room in agent.room.metadata["multi_room"]["rooms"]}

    source_motion = agent.sample_motion(mode="through_portal", moving="source", keyframes=13)
    receiver_motion = agent.sample_motion(mode="through_portal", moving="receiver", keyframes=13)

    assert source_motion["path_model"] == "portal_crossing_smoothstep"
    assert point_in_polygon(source_motion["frames"][-1]["source"], rooms["bedroom_0"]["corners"])
    assert point_in_polygon(receiver_motion["frames"][-1]["receiver"], rooms["bedroom_1"]["corners"])
    assert source_motion["distance_m"] != 0.8
    assert receiver_motion["distance_m"] != 0.8

    lightweight = SimConfig(
        fs=8000,
        duration_s=0.02,
        reflections_enabled=False,
        diffraction_enabled=False,
        late_tail=False,
        rt_num_rays=64,
        rt_num_bounces=4,
    )
    dynamic = agent.run_dynamic(source_motion, config=lightweight)
    final_topology = dynamic.frames[-1].metadata["multi_room"]
    assert final_topology["source_room_id"] == "bedroom_0"
    assert final_topology["route_room_ids"][0] == "bedroom_0"
    assert len(final_topology["route_portal_ids"]) == 3

    # Crossing a doorway changes topology, not the apartment's traced floor
    # and ceiling footprint.
    final_room = room_for_motion_frame(
        agent.room,
        source_motion["frames"][-1]["source"],
        source_motion["frames"][-1]["receiver"],
    )
    assert final_room.corners == agent.room.corners
    rerun = agent.run(
        source=source_motion["frames"][-1]["source"],
        receiver=source_motion["frames"][-1]["receiver"],
        config=lightweight,
    )
    assert rerun.metadata["multi_room"]["source_room_id"] == "bedroom_0"
    assert rerun.metadata["multi_room"]["route_room_ids"][0] == "bedroom_0"
