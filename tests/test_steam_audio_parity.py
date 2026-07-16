import math

import numpy as np
import pytest

from acoustic_agent import SimConfig, make_room
from acoustic_agent.acoustics import (
    AIR_ABSORPTION_NP_PER_M,
    propagation_band_gains,
    steam_audio_pathing_deviation,
    steam_audio_utd_deviation,
)
from acoustic_agent.hrtf import _render_directional_path_signal
from acoustic_agent.models import FREQUENCY_BANDS, AcousticPath
from acoustic_agent.rir import render_impulses
from acoustic_agent.steam_rt import (
    RoomRayScene,
    _boundary_diffraction_paths,
    _extend_energy_field_late_tail,
    _schroeder_fit_rt60,
    _segment_inside_room,
    _sphere_samples,
    _trace_energy_field_numba,
    _trace_energy_field_numpy,
    _volumetric_occlusion,
    bandlimit_band_signals,
    reconstruct_band_irs,
    simulate_direct,
)
from acoustic_agent.web_server import _quality_config


def test_schroeder_fit_matches_steam_audio_extrapolation():
    bin_duration = 0.01
    target_rt60 = 1.0
    times = np.arange(400, dtype=np.float64) * bin_duration
    energy = np.power(10.0, -6.0 * times / target_rt60)

    estimated = _schroeder_fit_rt60(energy, bin_duration)

    assert estimated == pytest.approx(target_rt60, rel=0.01)


def test_solver_presets_match_quality_tiers():
    presets = {name: _quality_config(name) for name in ("preview", "simulation", "fine", "reference")}

    assert {name: int(preset["rt_num_rays"]) for name, preset in presets.items()} == {
        "preview": 8192,
        "simulation": 32768,
        "fine": 65536,
        "reference": 131072,
    }
    assert {name: int(preset["rt_num_bounces"]) for name, preset in presets.items()} == {
        "preview": 32,
        "simulation": 64,
        "fine": 96,
        "reference": 96,
    }


def test_air_absorption_uses_steam_audio_octave_defaults():
    assert AIR_ABSORPTION_NP_PER_M == {
        "125": 0.0,
        "250": 0.00011513,
        "500": 0.00034539,
        "1000": 0.00057565,
        "2000": 0.0011513,
        "4000": 0.0034539,
    }


def test_disabling_transmission_does_not_restore_occluded_direct_sound():
    room = make_room("u_shape", size=(6.0, 4.0, 2.8))
    scene = RoomRayScene(room)
    source = np.asarray((0.8, 3.2, 1.4))
    listener = np.asarray((5.2, 3.2, 1.4))

    direct = simulate_direct(
        scene,
        source,
        listener,
        SimConfig(direct_occlusion_mode="raycast", direct_transmission=False),
    )

    assert direct["occlusion"] == 0.0
    assert np.allclose(direct["transmission"], 0.0)
    assert max(direct["band_gains"].values()) == 0.0


def test_volumetric_occlusion_discards_samples_hidden_from_source():
    room = make_room("rectangle", size=(4.0, 4.0, 2.8))
    scene = RoomRayScene(room)
    source = np.asarray((0.01, 2.0, 1.4))
    listener = np.asarray((1.0, 2.0, 1.4))
    config = SimConfig(direct_occlusion_radius_m=0.1, direct_occlusion_samples=256)

    assert _volumetric_occlusion(scene, listener, source, config) == pytest.approx(1.0)


def test_transmission_accumulates_each_physical_barrier_once():
    room = make_room("u_shape", size=(6.0, 4.0, 2.8))
    scene = RoomRayScene(room)
    source = np.asarray((0.8, 3.2, 1.4))
    listener = np.asarray((5.2, 3.2, 1.4))
    direct = simulate_direct(
        scene,
        source,
        listener,
        SimConfig(direct_occlusion_mode="raycast", direct_transmission=True),
    )

    wall_transmission = np.asarray([room.materials["wall"].transmission[band] for band in FREQUENCY_BANDS])
    np.testing.assert_allclose(direct["transmission"], wall_transmission ** 2, rtol=1e-12, atol=0.0)


def test_multi_edge_diffraction_uses_total_reference_normalized_deviation():
    room = make_room("u_shape", size=(6.0, 4.0, 2.8))
    scene = RoomRayScene(room)
    source = np.asarray((0.8, 3.2, 1.4))
    listener = np.asarray((5.2, 3.2, 1.4))
    config = SimConfig(direct_occlusion_mode="raycast", diffraction_order=3)
    direct = simulate_direct(scene, source, listener, config)
    path = _boundary_diffraction_paths(room, scene, source, listener, direct, config)[0]
    deviations = [float(value) for value in path.metadata["per_edge_deviation_rad"]]
    expected = propagation_band_gains(path.distance_m, min_distance_m=config.min_distance_m)
    deviation = steam_audio_pathing_deviation(sum(deviations))

    for band in FREQUENCY_BANDS:
        assert path.band_gains[band] == pytest.approx(expected[band] * deviation[band], rel=1e-6)

    per_edge_raw = steam_audio_utd_deviation(deviations[0])
    assert path.band_gains["1000"] > expected["1000"] * per_edge_raw["1000"] ** len(deviations)


def test_six_band_synthesis_is_perfectly_reconstructing_for_equal_inputs():
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(4096).astype(np.float32)
    bands = np.tile(signal, (len(FREQUENCY_BANDS), 1))

    reconstructed = np.sum(bandlimit_band_signals(bands, 16000), axis=0)

    np.testing.assert_allclose(reconstructed, signal, rtol=1e-6, atol=2e-6)


def test_six_band_synthesis_is_calibrated_at_band_centers():
    fs = 16000
    length = 65536
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0
    branches = bandlimit_band_signals(np.tile(impulse, (len(FREQUENCY_BANDS), 1)), fs)
    spectra = np.abs(np.fft.rfft(branches, axis=1))
    frequencies = np.fft.rfftfreq(length, 1.0 / fs)

    for band_index, band in enumerate(FREQUENCY_BANDS):
        bin_index = int(np.argmin(np.abs(frequencies - float(band))))
        gain_db = 20.0 * math.log10(max(float(spectra[band_index, bin_index]), 1e-20))
        assert abs(gain_db) < 0.25


def test_reconstructor_uses_coherent_noise_across_bands(monkeypatch):
    for band in FREQUENCY_BANDS:
        monkeypatch.setitem(AIR_ABSORPTION_NP_PER_M, band, 0.0)
    config = SimConfig(fs=16000, duration_s=0.04, seed=91)
    field = {
        "echogram": np.ones((len(FREQUENCY_BANDS), 4), dtype=np.float64),
        "num_bins": 4,
        "bin_duration_s": 0.01,
    }

    reconstructed = np.sum(reconstruct_band_irs(field, config), axis=0)
    white = np.random.default_rng(config.seed + 7).uniform(-1.0, 1.0, size=reconstructed.size)
    expected = white / math.sqrt(4.0 * math.pi)

    np.testing.assert_allclose(reconstructed, expected, rtol=1e-5, atol=2e-6)


def test_foa_energy_field_keeps_the_initial_listener_ray_direction():
    room = make_room("rectangle", size=(6.0, 4.0, 2.8))
    config = SimConfig(
        duration_s=1.0,
        rt_duration_s=1.0,
        rt_num_rays=1,
        rt_num_bounces=12,
        seed=0,
        late_tail=False,
    )
    field = _trace_energy_field_numba(
        RoomRayScene(room),
        np.asarray((4.0, 2.0, 1.4)),
        np.asarray((1.0, 1.0, 1.4)),
        config,
    )
    energy = field["echogram"][0]
    valid = energy > 0.0
    initial_direction = _sphere_samples(1, config.seed)[0]

    assert np.count_nonzero(valid) > 1
    np.testing.assert_allclose(field["ambisonic_echogram"][0, 0, valid] / energy[valid], 1.0, atol=1e-12)
    for channel, expected in enumerate(initial_direction, start=1):
        np.testing.assert_allclose(
            field["ambisonic_echogram"][0, channel, valid] / energy[valid],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


def test_numba_energy_field_matches_numpy_scattering_sequence():
    room = make_room("u_shape", size=(6.0, 4.0, 2.8))
    scene = RoomRayScene(room)
    config = SimConfig(
        duration_s=0.4,
        rt_duration_s=0.4,
        rt_num_rays=512,
        rt_num_bounces=12,
        seed=4107,
        late_tail=False,
    )
    source = np.asarray((0.8, 3.2, 1.4))
    listener = np.asarray((5.2, 3.2, 1.4))

    numpy_field = _trace_energy_field_numpy(scene, source, listener, config)
    numba_field = _trace_energy_field_numba(scene, source, listener, config)

    np.testing.assert_allclose(numba_field["echogram"], numpy_field["echogram"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        numba_field["ambisonic_echogram"],
        numpy_field["ambisonic_echogram"],
        rtol=1e-12,
        atol=1e-15,
    )
    assert numba_field["active_ray_count"] == numpy_field["active_ray_count"]
    assert numba_field["actual_bounces"] == numpy_field["actual_bounces"]
    assert numba_field["surface_hit_count"] == numpy_field["surface_hit_count"]
    assert numba_field["surface_contribution_count"] == numpy_field["surface_contribution_count"]


def test_hrtf_directional_path_preserves_six_band_gains():
    fs = 16000
    length = 4096
    delay_s = 0.03
    band_gains = {band: 0.5 ** index for index, band in enumerate(FREQUENCY_BANDS)}
    path = AcousticPath(
        "direct_transmitted",
        3.0,
        delay_s,
        float(np.mean(list(band_gains.values()))),
        band_gains,
        ((3.0, 1.0, 1.4), (1.0, 1.0, 1.4)),
        {"contributes_to_rir": True},
    )
    expected_bands = np.zeros((len(FREQUENCY_BANDS), length), dtype=np.float32)
    for band_index, band in enumerate(FREQUENCY_BANDS):
        expected_bands[band_index] = render_impulses(
            np.asarray([delay_s]),
            np.asarray([band_gains[band]]),
            fs=fs,
            duration_s=length / fs,
            fractional=True,
        )
    expected = np.sum(bandlimit_band_signals(expected_bands, fs), axis=0)

    rendered = _render_directional_path_signal(path, fs, length)

    np.testing.assert_allclose(rendered, expected, rtol=1e-6, atol=1e-7)


def test_default_distance_attenuation_matches_steam_audio_unit_distance():
    room = make_room("rectangle", size=(4.0, 4.0, 2.8))
    direct = simulate_direct(
        RoomRayScene(room),
        np.asarray((1.5, 1.0, 1.4)),
        np.asarray((1.0, 1.0, 1.4)),
        SimConfig(direct_occlusion=False),
    )

    assert direct["distance_m"] == pytest.approx(0.5)
    assert direct["distance_attenuation"] == pytest.approx(1.0)


def test_hybrid_tail_extrapolates_when_trace_ends_before_transition():
    num_bins = 200
    traced = np.zeros((len(FREQUENCY_BANDS), num_bins), dtype=np.float64)
    times = np.arange(50, dtype=np.float64) * 0.01
    traced[:, :50] = np.exp(-6.0 * math.log(10.0) * times / 1.5)
    field = {
        "echogram": traced,
        "ambisonic_echogram": np.zeros((len(FREQUENCY_BANDS), 4, num_bins), dtype=np.float64),
        "num_bins": num_bins,
        "bin_duration_s": 0.01,
    }

    projected, metadata = _extend_energy_field_late_tail(
        field,
        {band: 1.5 for band in FREQUENCY_BANDS},
        SimConfig(hybrid_transition_s=1.0, hybrid_overlap_fraction=0.25),
    )

    assert metadata["applied"] is True
    assert metadata["added"] is True
    assert set(metadata["anchor_model_by_band"].values()) == {"rt60_extrapolated_recent_bins"}
    assert np.any(projected["echogram"][:, 75:] > 0.0)


def test_diffraction_segment_validation_detects_a_narrow_notch():
    room = make_room(
        "polygon",
        size=(10.0, 10.0, 2.8),
        corners=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.13, 10.0), (5.13, 5.0), (5.11, 5.0), (5.11, 10.0), (0.0, 10.0)),
    )

    assert not _segment_inside_room(np.asarray((1.0, 9.0, 1.4)), np.asarray((9.0, 9.0, 1.4)), room)


def test_room_rejects_self_intersecting_polygon():
    corners = []
    for index in (0, 2, 4, 1, 3):
        angle = 2.0 * math.pi * index / 5.0 - math.pi * 0.5
        corners.append((2.0 + 1.8 * math.cos(angle), 2.0 + 1.8 * math.sin(angle)))

    with pytest.raises(ValueError, match="self-intersect"):
        make_room("polygon", size=(4.0, 4.0, 2.8), corners=corners)
