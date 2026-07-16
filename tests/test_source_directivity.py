import numpy as np

from acoustic_agent import AcousticAgent, SimConfig, make_room, simulate_rir, source_directivity
from acoustic_agent.directivity import source_directivity_gain


def _direct_path(result):
    return next(path for path in result.paths if path.kind.startswith("direct"))


def test_weighted_dipole_matches_steam_audio_cardioid_formula():
    model = source_directivity("cardioid")

    assert source_directivity_gain((1.0, 0.0, 0.0), model) == 1.0
    assert np.isclose(source_directivity_gain((0.0, 1.0, 0.0), model), 0.5)
    assert np.isclose(source_directivity_gain((-1.0, 0.0, 0.0), model), 0.0)


def test_cardioid_direct_sound_has_front_response_and_rear_null():
    room = make_room("rectangle", size=(6.0, 4.0, 2.8))
    config = SimConfig(
        duration_s=0.1,
        reflections_enabled=False,
        diffraction_enabled=False,
        rt_visual_num_rays=16,
        rt_visual_num_bounces=1,
    )
    source = (3.0, 2.0, 1.4)
    front = simulate_rir(room, source, (4.0, 2.0, 1.4), config=config, source_model="cardioid")
    rear = simulate_rir(room, source, (2.0, 2.0, 1.4), config=config, source_model="cardioid")

    assert _direct_path(front).metadata["source_directivity_gain"] == 1.0
    assert np.isclose(_direct_path(rear).metadata["source_directivity_gain"], 0.0)
    assert np.max(np.abs(front.rir)) > 0.0
    assert np.max(np.abs(rear.rir)) == 0.0


def test_omitted_source_model_is_sample_equivalent_to_explicit_omni():
    room = make_room("rectangle", size=(4.0, 3.0, 2.8))
    config = SimConfig(
        duration_s=0.12,
        rt_duration_s=0.12,
        rt_num_rays=256,
        rt_num_bounces=2,
        rt_visual_num_rays=64,
        rt_visual_num_bounces=2,
        late_tail=False,
    )
    args = (room, (1.1, 1.2, 1.4), (3.1, 2.1, 1.4))

    implicit = simulate_rir(*args, config=config)
    explicit = simulate_rir(*args, config=config, source_model="omni")

    np.testing.assert_array_equal(implicit.rir, explicit.rir)
    np.testing.assert_array_equal(implicit.ambisonic_rir, explicit.ambisonic_rir)


def test_agent_accepts_source_model_at_construction_and_per_run():
    agent = AcousticAgent(
        room=[4.0, 3.0, 2.8],
        source_model={"type": "cardioid", "orientation_deg": 30.0},
        config=SimConfig(
            duration_s=0.05,
            reflections_enabled=False,
            diffraction_enabled=False,
            rt_visual_num_rays=8,
            rt_visual_num_bounces=1,
        ),
    )

    result = agent.run(
        source=[1.0, 1.0, 1.4],
        receiver=[2.0, 1.0, 1.4],
        source_model={"type": "focused", "orientation_deg": 0.0},
    )

    assert agent.source_model["pattern"] == "cardioid"
    assert result.source_model["pattern"] == "focused"
    assert result.metadata["steam_audio"]["source_directivity"]["dipole_power"] == 4.0
