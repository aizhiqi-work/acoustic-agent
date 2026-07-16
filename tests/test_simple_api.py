from acoustic_agent import AcousticAgent, SimConfig


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
        "material": "plaster",
        "position": [2.0, 1.5],
        "z": 1.0,
        "size": [0.1, 2.0, 2.0],
        "rotation_deg": 0.0,
    }]

    agent = AcousticAgent(room=[4.0, 3.0, 2.8], acoustic_geometry=geometry)
    embedded = AcousticAgent(room={"shape": "rectangle", "size": [4.0, 3.0, 2.8], "objects": geometry})

    assert agent.room.metadata["objects"][0]["type"] == "panel"
    assert agent.room.metadata["objects"][0]["rotation"] == 0.0
    assert embedded.room.metadata["objects"] == agent.room.metadata["objects"]
