from dataclasses import replace
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
from acoustic_agent.engine import simulate_rir
from acoustic_agent.models import FREQUENCY_BANDS, AcousticPath
from acoustic_agent.rir import render_impulses
from acoustic_agent.steam_rt import (
    RoomRayScene,
    _air_absorption_amplitude,
    _air_absorption_energy_weights,
    _apply_coupled_late_reverb_prior,
    _adaptive_reflection_config,
    _bandlimit_band_signals_serial,
    _boundary_diffraction_paths,
    _energy_decay_profile,
    _render_parametric_fdn_late_reverb,
    _schroeder_fit_rt60,
    _segment_inside_room,
    _sphere_samples,
    _steam_compensated_multiband_coefficients,
    _trace_energy_field_numba,
    _trace_energy_field_numpy,
    _volumetric_occlusion,
    bandlimit_band_signals,
    estimate_late_reverb_times,
    estimate_reconstructed_reverb_times,
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


def test_compensated_fdn_filters_preserve_sharp_six_band_feedback_targets():
    fs = 16000
    desired = np.asarray((0.81, 0.72, 0.93, 0.76, 0.66, 0.79), dtype=np.float64)
    coeffs = _steam_compensated_multiband_coefficients(desired, fs)
    centers = np.asarray([float(band) for band in FREQUENCY_BANDS], dtype=np.float64)
    z = np.exp(-2.0j * math.pi * centers / fs)
    transfer = np.ones(len(FREQUENCY_BANDS), dtype=np.complex128)
    for band_coeffs in coeffs:
        transfer *= (
            band_coeffs[0] + band_coeffs[1] * z + band_coeffs[2] * z * z
        ) / (
            1.0 + band_coeffs[3] * z + band_coeffs[4] * z * z
        )

    np.testing.assert_allclose(np.abs(transfer), desired, rtol=0.02, atol=0.002)


def test_low_energy_cross_room_decay_uses_wide_range_fit_instead_of_zero_rt(monkeypatch):
    for band in FREQUENCY_BANDS:
        monkeypatch.setitem(AIR_ABSORPTION_NP_PER_M, band, 0.0)
    bin_duration = 0.01
    target_rt60 = 0.8
    times = np.arange(200, dtype=np.float64) * bin_duration
    energy = 1e-8 * np.power(10.0, -6.0 * times / target_rt60)
    field = {
        "echogram": np.repeat(energy[None, :], len(FREQUENCY_BANDS), axis=0),
        "bin_duration_s": bin_duration,
        "num_bins": energy.size,
    }

    targets, profiles = estimate_late_reverb_times(
        field,
        SimConfig(),
        fallback={band: 0.0 for band in FREQUENCY_BANDS},
    )

    for band in FREQUENCY_BANDS:
        assert targets[band] == pytest.approx(target_rt60, rel=0.03)
        assert profiles[band]["selected_target_source"] == "fitted_wide_range_single_slope"


def _energy_from_target_edc_db(edc_db):
    edc = np.power(10.0, np.asarray(edc_db, dtype=np.float64) / 10.0)
    return np.maximum(0.0, np.concatenate((edc[:-1] - edc[1:], edc[-1:])))


def test_decay_profile_keeps_single_exponential_as_one_slope():
    times = np.arange(200, dtype=np.float64) * 0.01
    energy = _energy_from_target_edc_db(-60.0 * times)

    profile = _energy_decay_profile(energy, 0.01)

    assert profile["model"] == "single_slope"
    assert len(profile["segments"]) == 1
    assert profile["segments"][0]["equivalent_rt60_s"] == pytest.approx(1.0, rel=0.01)


def test_decay_profile_detects_significant_coupled_space_break():
    times = np.arange(200, dtype=np.float64) * 0.01
    transition_s = 0.25
    edc_db = np.where(
        times <= transition_s,
        -60.0 * times,
        -60.0 * transition_s - 120.0 * (times - transition_s),
    )
    energy = _energy_from_target_edc_db(edc_db)

    profile = _energy_decay_profile(energy, 0.01)

    assert profile["model"] == "double_slope"
    assert profile["transition_time_s"] == pytest.approx(transition_s, abs=0.02)
    assert profile["segments"][0]["equivalent_rt60_s"] == pytest.approx(1.0, rel=0.03)
    assert profile["segments"][1]["equivalent_rt60_s"] == pytest.approx(0.5, rel=0.03)


def test_coupled_space_tail_target_uses_the_fitted_late_slope(monkeypatch):
    for band in FREQUENCY_BANDS:
        monkeypatch.setitem(AIR_ABSORPTION_NP_PER_M, band, 0.0)
    times = np.arange(200, dtype=np.float64) * 0.01
    transition_s = 0.2
    edc_db = np.where(
        times <= transition_s,
        -100.0 * times,
        -100.0 * transition_s - 50.0 * (times - transition_s),
    )
    energy = _energy_from_target_edc_db(edc_db)
    field = {
        "echogram": np.tile(energy, (len(FREQUENCY_BANDS), 1)),
        "num_bins": energy.size,
        "bin_duration_s": 0.01,
    }
    fallback = {band: 0.6 for band in FREQUENCY_BANDS}

    targets, profiles = estimate_late_reverb_times(field, SimConfig(), fallback=fallback)

    for band in FREQUENCY_BANDS:
        assert profiles[band]["model"] == "double_slope"
        assert profiles[band]["selected_target_source"] == "fitted_late_slope"
        assert targets[band] == pytest.approx(1.2, rel=0.03)


def test_coupled_room_prior_stabilizes_only_the_late_tail_target():
    traced = {band: 0.7 + 0.1 * index for index, band in enumerate(FREQUENCY_BANDS)}
    prior = {band: 1.4 + 0.05 * index for index, band in enumerate(FREQUENCY_BANDS)}
    profiles = {
        band: {"selected_target_source": "fitted_late_slope", "selected_target_rt60_s": traced[band]}
        for band in FREQUENCY_BANDS
    }

    targets, updated = _apply_coupled_late_reverb_prior(traced, profiles, prior)

    assert targets == prior
    for band in FREQUENCY_BANDS:
        assert updated[band]["traced_target_rt60_s"] == pytest.approx(traced[band])
        assert updated[band]["selected_target_source"] == "coupled_room_energy_matrix"


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


def test_reflection_air_absorption_uses_distance_and_energy_units():
    coefficient = 0.01
    travel_time_s = 0.2
    speed = 343.0
    expected_amplitude = math.exp(-coefficient * speed * travel_time_s)

    amplitude = _air_absorption_amplitude(coefficient, travel_time_s, speed)
    energy = _air_absorption_energy_weights(coefficient, np.asarray([travel_time_s]), speed)[0]

    assert amplitude == pytest.approx(expected_amplitude)
    assert energy == pytest.approx(expected_amplitude ** 2)


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


def test_closest_hit_does_not_accept_infinite_surface_intersections():
    room = make_room("rectangle", size=(4.0, 4.0, 2.8))
    scene = RoomRayScene(room)
    hit = scene.closest_hit(
        np.asarray((10.0, 10.0, 1.4)),
        np.asarray((1.0, 0.0, 0.0)),
    )

    assert hit["valid"] is False
    assert math.isinf(hit["distance"])


def test_volumetric_occlusion_discards_samples_hidden_from_source():
    room = make_room("rectangle", size=(4.0, 4.0, 2.8))
    scene = RoomRayScene(room)
    source = np.asarray((0.01, 2.0, 1.4))
    listener = np.asarray((1.0, 2.0, 1.4))
    config = SimConfig(direct_occlusion_radius_m=0.1, direct_occlusion_samples=256)

    assert _volumetric_occlusion(scene, listener, source, config) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", ("linear", "bvh"))
def test_batched_volumetric_occlusion_matches_serial_rays(backend):
    room = make_room("u_shape", size=(6.0, 5.0, 2.8))
    scene = RoomRayScene(room)
    scene.configure_intersection(backend, 1)
    source = np.asarray((0.7, 4.2, 1.4))
    listener = np.asarray((5.1, 1.0, 1.4))
    config = SimConfig(direct_occlusion_radius_m=0.18, direct_occlusion_samples=128)

    radius = float(config.direct_occlusion_radius_m)
    from acoustic_agent.steam_rt import _sphere_volume_samples

    samples = _sphere_volume_samples(config.direct_occlusion_samples) * radius + source[None, :]
    visible = 0
    valid = 0
    for sample in samples:
        source_leg = sample - source
        source_distance = float(np.linalg.norm(source_leg))
        if (
            source_distance > 1e-9
            and scene.any_hit(source, source_leg / source_distance, source_distance)
        ):
            continue
        listener_leg = sample - listener
        listener_distance = float(np.linalg.norm(listener_leg))
        valid += 1
        if listener_distance <= 1e-9 or not scene.any_hit(
            listener,
            listener_leg / max(listener_distance, 1e-9),
            listener_distance,
        ):
            visible += 1
    expected = float(visible / valid) if valid else 0.0

    assert _volumetric_occlusion(scene, listener, source, config) == expected


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


def test_parallel_six_band_synthesis_is_sample_exact_to_serial():
    rng = np.random.default_rng(20260726)
    signals = rng.standard_normal((len(FREQUENCY_BANDS), 32000))

    serial = _bandlimit_band_signals_serial(signals, 16000)
    parallel = bandlimit_band_signals(signals, 16000)

    np.testing.assert_array_equal(parallel, serial)


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


def test_reflective_geometry_adapts_simulation_bounces_for_rt_convergence():
    room = make_room(
        "rectangle",
        size=(6.0, 4.0, 2.8),
        materials={"wall": "wall", "floor": "floor", "ceiling": "ceiling"},
    )
    requested = SimConfig(rt_num_rays=32768, rt_num_bounces=64)

    effective, metadata = _adaptive_reflection_config(RoomRayScene(room), requested)

    assert metadata["applied"] is True
    assert metadata["reason"] == "geometry_tail_convergence"
    assert metadata["requested"] == 64
    assert metadata["effective"] == 128
    assert effective.rt_num_bounces == 128

    disabled, disabled_metadata = _adaptive_reflection_config(
        RoomRayScene(room),
        replace(requested, adaptive_geometry_bounces=False),
    )
    assert disabled_metadata["applied"] is False
    assert disabled_metadata["reason"] == "disabled"
    assert disabled.rt_num_bounces == 64


def test_parametric_fdn_on_off_must_change_the_late_tail():
    fs = 8000
    num_samples = 12000
    rng = np.random.default_rng(81)
    times = np.arange(num_samples, dtype=np.float64) / fs
    envelope = np.exp(-3.0 * math.log(10.0) * times / 1.2)
    traced = np.vstack([rng.uniform(-1.0, 1.0, num_samples) * envelope for _ in FREQUENCY_BANDS])
    field = {"bin_duration_s": 0.01}
    rt60 = {band: 1.2 for band in FREQUENCY_BANDS}

    enabled, fdn, metadata = _render_parametric_fdn_late_reverb(
        traced,
        field,
        rt60,
        SimConfig(fs=fs, duration_s=1.5, hybrid_transition_s=0.5, hybrid_overlap_fraction=0.25),
    )
    disabled, disabled_fdn, disabled_metadata = _render_parametric_fdn_late_reverb(
        traced,
        field,
        rt60,
        SimConfig(fs=fs, duration_s=1.5, hybrid_transition_s=0.5, hybrid_overlap_fraction=0.25, late_tail=False),
    )

    start = int(metadata["transition_start_s"] * fs)
    assert metadata["applied"] is True
    assert metadata["added"] is True
    assert metadata["delay_line_count"] == 16
    assert metadata["fdn_tail_energy"] > 0.0
    assert disabled_metadata["applied"] is False
    np.testing.assert_allclose(disabled, traced, rtol=1e-6, atol=1e-7)
    assert not np.any(disabled_fdn)
    assert float(np.sum(np.square(enabled[:, start:] - disabled[:, start:]))) > 1e-6
    assert float(np.sum(np.square(fdn[:, start:]))) > 1e-6


def test_hybrid_multiband_fdn_tracks_rt60_targets_at_production_sample_rate():
    fs = 16000
    num_samples = 2 * fs
    target_rt60 = 1.2
    rng = np.random.default_rng(83)
    times = np.arange(num_samples, dtype=np.float64) / fs
    envelope = np.exp(-3.0 * math.log(10.0) * times / target_rt60)
    traced = np.vstack([rng.uniform(-1.0, 1.0, num_samples) * envelope for _ in FREQUENCY_BANDS])
    config = SimConfig(
        fs=fs,
        duration_s=2.0,
        hybrid_transition_s=0.5,
        hybrid_overlap_fraction=0.25,
    )

    rendered, _, metadata = _render_parametric_fdn_late_reverb(
        traced,
        {"bin_duration_s": 0.01},
        {band: target_rt60 for band in FREQUENCY_BANDS},
        config,
    )

    assert metadata["transition_start_s"] > 0.0
    measured = estimate_reconstructed_reverb_times(rendered, config)
    for band in FREQUENCY_BANDS:
        assert measured[band] == pytest.approx(target_rt60, rel=0.12)


def test_coupled_space_fdn_crossfade_finishes_at_the_fitted_breakpoint():
    fs = 8000
    num_samples = int(1.5 * fs)
    rng = np.random.default_rng(89)
    traced = rng.uniform(-1.0, 1.0, (len(FREQUENCY_BANDS), num_samples))
    traced *= np.exp(-5.0 * np.arange(num_samples, dtype=np.float64) / fs)[None, :]
    config = SimConfig(
        fs=fs,
        duration_s=1.5,
        hybrid_transition_s=1.0,
        hybrid_overlap_fraction=0.25,
    )

    _, _, metadata = _render_parametric_fdn_late_reverb(
        traced,
        {"bin_duration_s": 0.01},
        {band: 1.0 for band in FREQUENCY_BANDS},
        config,
        decay_profiles={
            "125": {
                "model": "double_slope",
                "transition_time_s": 0.4,
                "selected_target_source": "fitted_late_slope",
            },
        },
    )

    assert metadata["transition_by_band"]["125"] == {
        "start_s": 0.3,
        "end_s": 0.4,
        "model": "coupled_space_breakpoint",
        "anchor_s": 0.4,
    }
    assert metadata["transition_by_band"]["250"] == {
        "start_s": 0.75,
        "end_s": 1.0,
        "model": "configured_hybrid_transition",
        "anchor_s": 0.75,
    }


def test_full_solver_fdn_toggle_changes_the_rendered_tail():
    room = make_room("rectangle", size=(5.0, 4.0, 2.8))
    config = SimConfig(
        fs=8000,
        duration_s=0.9,
        rt_duration_s=0.9,
        rt_num_rays=1024,
        rt_num_bounces=24,
        rt_visual_num_rays=32,
        rt_visual_num_bounces=2,
        hybrid_transition_s=0.3,
    )

    enabled = simulate_rir(room, (1.0, 1.0, 1.4), (4.0, 3.0, 1.4), config=config)
    disabled = simulate_rir(room, (1.0, 1.0, 1.4), (4.0, 3.0, 1.4), config=replace(config, late_tail=False))

    metadata = enabled.metadata["steam_audio"]["reflections"]["late_tail"]
    start = int(float(metadata["transition_start_s"]) * config.fs)
    assert metadata["model"] == "steam_style_16_line_hadamard_fdn"
    assert metadata["applied"] is True
    assert float(np.sum(np.square(enabled.rir[0, start:] - disabled.rir[0, start:]))) > 1e-6


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
