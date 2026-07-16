import base64

import numpy as np

from acoustic_agent import SimConfig, SimulationPair, make_room, microphone_array, simulate_batch, simulate_rir
from acoustic_agent.geometry import point_in_polygon, polygon_area
from acoustic_agent.hrtf import render_binaural_sofa
from acoustic_agent.materials import MaterialLibrary
from acoustic_agent.web_export import scene_payload


def test_rectangle_mono_rir_has_direct_and_reflections():
    room = make_room("rectangle", size=(5.0, 4.0, 2.8))
    result = simulate_rir(room, (1.0, 1.0, 1.4), (3.0, 2.5, 1.3), config=SimConfig(duration_s=0.5, late_tail=False, rt_num_rays=512, rt_num_bounces=2, rt_duration_s=0.5))
    assert result.rir.shape == (1, 8000)
    assert np.isfinite(result.rir).all()
    assert result.metadata["steam_audio"]["model"] == "steam_audio_style_direct_plus_pathtraced_reflections"
    assert result.metadata["steam_audio"]["reflections"]["enabled"] is True
    assert result.metadata["steam_audio"]["rt_visual"]["retain_limit"] == 512
    assert result.metadata["steam_audio"]["rt_visual"]["retained_path_count"] <= 512
    assert result.metadata["steam_audio"]["rt_visual"]["ray_count"] == result.metadata["steam_audio"]["reflections"]["num_rays"]
    assert result.metadata["steam_audio"]["rt_visual"]["max_bounces"] == result.metadata["steam_audio"]["reflections"]["num_bounces"]
    assert result.metadata["steam_audio"]["rt_visual"]["follows_simulation"] is True
    assert result.metadata["steam_audio"]["rt_visual"].get("accelerator") == "numba"
    assert result.metadata["steam_audio"]["rt_visual"]["retention_policy"] == "stratified_order_then_strongest_gain"
    assert any(path.kind == "direct" for path in result.paths)
    assert any(path.kind == "rt_reflection" for path in result.paths)
    assert not any(path.kind == "ism_reflection" for path in result.paths)
    assert result.metadata["solver_pipeline"] == ["direct", "diffraction", "rt_energy_field", "reverb_estimate", "hybrid_late_reverb", "receiver_reconstruction"]


def test_concave_room_uses_transmitted_direct_and_utd_diffraction():
    room = make_room("l_shape", size=(6.0, 5.0, 2.8))
    config = SimConfig(
        duration_s=0.35,
        rt_num_rays=512,
        rt_num_bounces=2,
        rt_duration_s=0.35,
        rt_visual_num_rays=512,
        rt_visual_num_bounces=4,
    )
    result = simulate_rir(room, (5.0, 2.0, 1.4), (2.0, 4.0, 1.4), config=config)
    direct = next(path for path in result.paths if path.kind == "direct_transmitted")
    diffraction = [path for path in result.paths if path.kind == "diffraction"]
    direct_meta = result.metadata["steam_audio"]["direct"]
    assert direct_meta["occlusion"] < 1.0
    assert max(direct_meta["transmission"].values()) < 0.01
    assert direct.gain < diffraction[0].gain
    assert diffraction[0].metadata["model"] == "steam_audio_utd_deviation"
    assert diffraction[0].metadata["contributes_to_rir"] is True
    assert len(diffraction[0].points) == 3


def test_web_payload_contains_true_rir_and_band_path_diagnostics():
    room = make_room("rectangle", size=(4.0, 3.0, 2.8))
    result = simulate_rir(
        room,
        (1.0, 1.0, 1.3),
        (3.0, 2.0, 1.3),
        config=SimConfig(duration_s=0.2, rt_num_rays=512, rt_num_bounces=2, rt_duration_s=0.2),
    )
    payload = scene_payload(room, sources=[(1.0, 1.0, 1.3)], receivers=[(3.0, 2.0, 1.3)], result=result)
    assert payload["rir"]["samples"]
    assert payload["rir"]["fs"] == 16000
    assert payload["rir"]["channel_count"] == 1
    assert payload["rir"]["shape"] == list(result.rir.shape)
    exact_rir = np.frombuffer(base64.b64decode(payload["rir"]["f32_base64"]), dtype="<f4").reshape(payload["rir"]["shape"])
    np.testing.assert_array_equal(exact_rir, np.asarray(result.rir, dtype=np.float32))
    assert set(payload["paths"][0]["band_gains"]) == {"125", "250", "500", "1000", "2000", "4000"}
    assert result.ambisonic_rir is not None
    assert result.ambisonic_rir.shape == (4, result.rir.shape[-1])
    assert result.metadata["steam_audio"]["reflections"]["ambisonics"]["order"] == 1
    assert result.metadata["steam_audio"]["reflections"]["ambisonics"]["energy"] > 0.0


def test_hrtf_render_uses_path_aware_binaural_model():
    room = make_room("rectangle", size=(4.0, 3.0, 2.8))
    result = simulate_rir(
        room,
        (1.0, 1.0, 1.3),
        (3.0, 2.0, 1.3),
        config=SimConfig(
            duration_s=0.15,
            reflections_enabled=False,
            rt_visual_num_rays=64,
            rt_visual_num_bounces=2,
        ),
        receiver_model=microphone_array("hrtf", interpolation="nearest"),
    )
    hrtf_meta = result.receiver_model["render_metadata"]
    assert result.rir.shape == (2, 2400)
    assert np.isfinite(result.rir).all()
    assert hrtf_meta["model"] == "sofa_path_aware_direct_diffraction_plus_decorrelated_residual"
    assert hrtf_meta["directional_path_count"] >= 1
    assert not np.allclose(result.rir[0], result.rir[1])


def test_hrtf_render_decodes_foa_reflections_when_available():
    room = make_room("rectangle", size=(4.0, 3.0, 2.8))
    result = simulate_rir(
        room,
        (1.0, 1.0, 1.3),
        (3.0, 2.0, 1.3),
        config=SimConfig(
            duration_s=0.2,
            rt_num_rays=512,
            rt_num_bounces=2,
            rt_duration_s=0.2,
            rt_visual_num_rays=64,
            rt_visual_num_bounces=2,
        ),
        receiver_model=microphone_array("hrtf", interpolation="nearest"),
    )
    hrtf_meta = result.receiver_model["render_metadata"]
    assert result.rir.shape == (2, 3200)
    assert hrtf_meta["model"] == "sofa_path_aware_direct_diffraction_plus_foa_reflections"
    assert hrtf_meta["ambisonic_order"] == 1
    assert hrtf_meta["ambisonic_energy"] > 0.0


def test_hrtf_loudness_normalization_preserves_average_binaural_energy():
    mono = np.zeros(1024, dtype=np.float32)
    mono[16] = 0.5
    normalized, metadata = render_binaural_sofa(
        mono,
        source=(0.0, 1.0, 0.0),
        receiver=(0.0, 0.0, 0.0),
        fs=16000,
    )
    raw, raw_metadata = render_binaural_sofa(
        mono,
        source=(0.0, 1.0, 0.0),
        receiver=(0.0, 0.0, 0.0),
        fs=16000,
        loudness_normalization="none",
    )
    target = float(np.sum(mono * mono))
    normalized_energy = 0.5 * float(np.sum(normalized[0] ** 2) + np.sum(normalized[1] ** 2))
    raw_energy = 0.5 * float(np.sum(raw[0] ** 2) + np.sum(raw[1] ** 2))
    assert np.isclose(normalized_energy, target, rtol=1e-5)
    assert raw_energy < target
    assert metadata["loudness_normalization"] == "energy"
    assert metadata["binaural_energy_db"] == 0.0
    assert raw_metadata["loudness_normalization"] == "none"
    assert raw_metadata["loudness_gain_db"] == 0.0


def test_default_hrtf_matches_steam_audio_cipic_and_preserves_handedness():
    impulse = np.zeros(1024, dtype=np.float32)
    impulse[16] = 1.0
    right_source, metadata = render_binaural_sofa(
        impulse,
        source=(1.0, 0.0, 0.0),
        receiver=(0.0, 0.0, 0.0),
        fs=16000,
        interpolation="nearest",
        loudness_normalization="none",
    )
    left_energy = float(np.sum(right_source[0] ** 2))
    right_energy = float(np.sum(right_source[1] ** 2))
    assert metadata["sofa_path"].endswith("cipic_124.sofa")
    assert metadata["sofa_database"] == "cipic_124"
    assert metadata["coordinate_model"] == "world_x_right_y_front_z_up_to_sofa_x_front_y_left_z_up"
    assert right_energy > left_energy * 5.0


def test_phase_aware_hrtf_interpolation_avoids_near_front_cancellation():
    impulse = np.zeros(1024, dtype=np.float32)
    impulse[16] = 1.0
    rendered, _ = render_binaural_sofa(
        impulse,
        source=(-0.02736, 0.99362, -0.10944),
        receiver=(0.0, 0.0, 0.0),
        fs=16000,
        interpolation="bilinear",
        loudness_normalization="none",
    )
    ear_energy = np.sum(rendered.astype(np.float64) ** 2, axis=1)
    ild_db = abs(10.0 * np.log10(ear_energy[0] / ear_energy[1]))
    assert ild_db < 1.0


def test_l_shape_and_array_batch():
    room = make_room("l_shape", size=(6.0, 5.0, 2.8))
    pairs = [SimulationPair((1.0, 1.0, 1.3), (2.8, 3.6, 1.3)), SimulationPair((1.5, 3.8, 1.4), (2.7, 3.4, 1.3))]
    batch = simulate_batch(
        room,
        pairs,
        config=SimConfig(
            duration_s=0.25,
            rt_num_rays=512,
            rt_num_bounces=2,
            rt_duration_s=0.25,
            rt_visual_num_rays=128,
            rt_visual_num_bounces=2,
        ),
        receiver_model=microphone_array("linear", count=3),
        workers=1,
    )
    assert len(batch.items) == 2
    assert batch.items[0].rir.shape == (3, 4000)
    assert np.isfinite(batch.items[0].rir).all()


def test_new_geometry_presets_create_valid_rooms():
    cases = {
        "trapezoid": (1.5, 1.0, 1.4),
        "u_shape": (1.5, 1.0, 1.4),
        "fan_shape": (3.0, 1.4, 1.4),
    }
    for shape, probe in cases.items():
        room = make_room(shape, size=(6.0, 4.0, 2.8))
        assert len(room.corners) >= 4
        assert polygon_area(room.corners) > 0.0
        assert point_in_polygon(probe[:2], room.corners)
        assert room.metadata["shape"] == shape


def test_material_library_load_reuses_process_cache():
    first = MaterialLibrary.load()
    second = MaterialLibrary.load()
    assert first is second
    assert first.records


def test_u_shape_nlos_uses_multi_order_diffraction():
    room = make_room("u_shape", size=(6.0, 4.0, 2.8))
    result = simulate_rir(
        room,
        (0.8, 3.2, 1.4),
        (5.2, 3.2, 1.4),
        config=SimConfig(
            duration_s=0.25,
            rt_num_rays=512,
            rt_num_bounces=2,
            rt_duration_s=0.25,
            diffraction_order=3,
            max_diffraction_paths=8,
        ),
    )
    direct = next(path for path in result.paths if path.kind == "direct_transmitted")
    diffraction = [path for path in result.paths if path.kind == "diffraction"]
    assert direct.metadata["occlusion"] == 0.0
    assert diffraction
    assert diffraction[0].metadata["model"] == "steam_audio_utd_multi_edge"
    assert diffraction[0].metadata["diffraction_order"] == 2
    assert len(diffraction[0].points) == 4
    assert result.metadata["steam_audio"]["diffraction"]["order_counts"]["2"] >= 1
