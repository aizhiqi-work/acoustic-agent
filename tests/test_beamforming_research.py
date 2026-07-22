from __future__ import annotations

import numpy as np

from research.beamforming.core import (
    align_channels,
    apply_stft_gain,
    apply_stft_beamformer,
    apply_wpd_beamformer,
    apply_wpe,
    aligned_si_sdr_db,
    arrival_times_from_azimuth,
    delay_and_sum,
    estimate_gcc_phat_delays,
    estimate_adaptive_beamformer_weights,
    estimate_wpd_weights,
    estimate_wpe_filters,
    estimate_wiener_gain,
    fractional_delay,
    snr_db,
    steering_vector,
)
from research.beamforming.benchmark import _recommended_microphone_counts, _validation_split
from research.beamforming.experiment import _select_node_indices
from research.beamforming.whole_home_benchmark import (
    ARRAY_COUNTS_BY_ROOM_COUNT,
    _global_source_gains,
    _mix_with_global_gains,
    _paired_comparisons,
    _target_cases,
)
from research.doa.distributed import SensorNode, candidate_nodes, load_model, place_nodes


def test_gcc_phat_recovers_relative_delay() -> None:
    rng = np.random.default_rng(12)
    source = rng.normal(size=4096)
    first = fractional_delay(source, 3.25)
    second = fractional_delay(source, 9.5)
    delays, confidence = estimate_gcc_phat_delays(
        np.vstack([first, second]),
        16_000,
        max_delay_s=0.005,
    )
    assert abs(delays[1] * 16_000 - 6.25) < 0.35
    assert confidence[1] > 0.2


def test_delay_and_sum_improves_uncorrelated_noise_snr() -> None:
    rng = np.random.default_rng(13)
    source = rng.normal(size=8192)
    arrival_samples = np.asarray([2.0, 5.5, 9.25, 12.0])
    target = np.vstack([fractional_delay(source, value) for value in arrival_samples])
    noise = rng.normal(size=target.shape)
    input_snr = snr_db(target[0], noise[0])
    output_target = delay_and_sum(target, arrival_samples / 16_000, 16_000)
    output_noise = delay_and_sum(noise, arrival_samples / 16_000, 16_000)
    assert snr_db(output_target, output_noise) > input_snr + 4.0


def test_align_channels_removes_distributed_target_delays() -> None:
    rng = np.random.default_rng(131)
    source = rng.normal(size=8192)
    arrival_samples = np.asarray([2.0, 93.5, 407.25, 1201.0])
    signals = np.vstack([fractional_delay(source, value) for value in arrival_samples])
    aligned = align_channels(signals, arrival_samples / 16_000, 16_000)
    delays, _ = estimate_gcc_phat_delays(aligned, 16_000, max_delay_s=0.005)
    assert np.max(np.abs(delays * 16_000)) < 0.4


def test_azimuth_steering_delays_near_side_first() -> None:
    microphones = np.asarray([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
    arrivals = arrival_times_from_azimuth(0.0, microphones)
    assert arrivals[0] < arrivals[1]
    assert np.isclose(arrivals[1] - arrivals[0], 0.1 / 343.0)


def test_wiener_gain_can_be_applied_to_matching_components() -> None:
    rng = np.random.default_rng(14)
    target = np.sin(2.0 * np.pi * 440.0 * np.arange(4096) / 16_000)
    noise = rng.normal(scale=0.4, size=target.size)
    gain = estimate_wiener_gain(target + noise, noise[:1024])
    enhanced_target = apply_stft_gain(target, gain)
    enhanced_noise = apply_stft_gain(noise, gain)
    assert enhanced_target.shape == target.shape
    assert enhanced_noise.shape == noise.shape
    assert np.all(np.isfinite(enhanced_target))


def test_stft_identity_has_no_boundary_impulse() -> None:
    rng = np.random.default_rng(141)
    signal = rng.normal(size=4096)
    probe_gain = estimate_wiener_gain(signal, rng.normal(size=1024))
    reconstructed = apply_stft_gain(signal, np.ones_like(probe_gain))
    assert np.max(np.abs(reconstructed)) < np.max(np.abs(signal)) * 1.01
    assert np.allclose(reconstructed[1:-1], signal[1:-1], atol=1e-5)


def test_aligned_si_sdr_removes_only_global_delay() -> None:
    rng = np.random.default_rng(15)
    reference = rng.normal(size=4096)
    estimate = fractional_delay(reference, 11.0) + rng.normal(scale=0.01, size=reference.size)
    assert aligned_si_sdr_db(estimate, reference, max_shift_samples=32) > 30.0


def test_floorplan_selection_prefers_a_source_room_microphone() -> None:
    model = load_model(0)
    source_room = max(model.rooms, key=lambda room_id: float(model.rooms[room_id]["area_m2"]))
    source_polygon = model.polygons[source_room]
    source_point = source_polygon.representative_point()
    local = SensorNode("local", source_room, (source_point.x, source_point.y, 2.2))
    remote_room = next(room_id for room_id in model.rooms if room_id != source_room)
    remote_point = model.polygons[remote_room].representative_point()
    remote = SensorNode("remote", remote_room, (remote_point.x, remote_point.y, 2.2))
    selected = _select_node_indices(
        model,
        [remote, local],
        (source_point.x, source_point.y, 1.5),
        source_room,
        [1.0, 0.2],
        1,
    )
    assert selected == [1]


def test_frequency_domain_beamformers_are_finite_and_distortionless_when_required() -> None:
    rng = np.random.default_rng(16)
    sample_count = 8192
    target_source = rng.normal(size=sample_count)
    noise_source = rng.normal(size=sample_count)
    target_arrivals = np.asarray([0.0, 1.5, 3.25, 4.75])
    noise_arrivals = np.asarray([4.5, 3.0, 1.25, 0.0])
    target = np.vstack([fractional_delay(target_source, value) for value in target_arrivals])
    noise = np.vstack([fractional_delay(noise_source, value) for value in noise_arrivals])
    guide = steering_vector(target_arrivals / 16_000, 16_000)
    for algorithm in ("ds", "weighted_ds", "mvdr", "gev", "mwf"):
        weights = estimate_adaptive_beamformer_weights(
            target[:, :4096],
            noise[:, :4096],
            target_arrivals / 16_000,
            16_000,
            algorithm=algorithm,
            reliability=np.ones(4),
        )
        output = apply_stft_beamformer(target + noise, weights)
        assert output.shape == (sample_count,)
        assert np.all(np.isfinite(output))
        if algorithm in {"ds", "weighted_ds", "mvdr", "gev"}:
            response = np.einsum("fm,fm->f", np.conj(weights), guide)
            assert np.allclose(response, 1.0, atol=1e-5)


def test_beamforming_benchmark_uses_five_fixed_plans_and_tdoa_mic_counts() -> None:
    split = _validation_split((4, 6, 8, 10, 12), 5)
    assert len(split) == 25
    assert {int(row["room_count"]) for row in split} == {4, 6, 8, 10, 12}
    assert _recommended_microphone_counts((4, 6, 8, 10, 12)) == {
        4: 5,
        6: 7,
        8: 8,
        10: 8,
        12: 8,
    }


def test_whole_home_target_cases_cover_fixed_array_and_blind_rooms() -> None:
    model = load_model(12178)
    arrays = place_nodes(
        model,
        ARRAY_COUNTS_BY_ROOM_COUNT[4],
        mode="array",
        risk_quantile=0.2,
        candidates=candidate_nodes(model, positions_per_room=1),
    )
    cases = _target_cases(model, arrays, seed=20260723)
    array_rooms = {node.room_id for node in arrays}
    assert cases["array_covered"].room_id in array_rooms
    assert cases["array_uncovered"].room_id not in array_rooms
    assert cases["array_covered"].room_id != cases["array_uncovered"].room_id


def test_whole_home_global_source_gains_are_shared_across_deployments() -> None:
    sample_count = 16_000
    target = np.full((1, sample_count), 2.0, dtype=np.float32)
    interferer = np.full((1, sample_count), 0.5, dtype=np.float32)
    pink = np.full((1, sample_count), 0.25, dtype=np.float32)
    gains = _global_source_gains(target, interferer, pink, sample_count, 0.0, 10.0, 120.0)
    near = _mix_with_global_gains(target, interferer, pink, gains, seed=1)
    far = _mix_with_global_gains(target * 0.1, interferer * 0.4, pink * 0.2, gains, seed=2)
    assert np.isclose(
        np.mean(far["noise"] - far["sensor"]),
        np.mean(interferer * 0.4) * gains["interferer"]
        + np.mean(pink * 0.2) * gains["pink"],
        rtol=1e-5,
    )
    assert np.isclose(
        np.mean(near["noise"] - near["sensor"]),
        np.mean(interferer) * gains["interferer"] + np.mean(pink) * gains["pink"],
        rtol=1e-5,
    )


def test_whole_home_paired_comparison_resamples_floorplans() -> None:
    rows = []
    for floorplan, candidate in ((1, 1.5), (2, 2.5)):
        for scenario in ("same_room", "cross_room"):
            common = {
                "floorplan_idx": floorplan,
                "target_case": "array_covered",
                "scenario": scenario,
                "output_stoi": 0.5,
                "si_sdr_improvement_db": 1.0,
            }
            rows.append({**common, "strategy": "distributed_singles_mwf", "output_pesq": 1.0})
            rows.append(
                {
                    **common,
                    "strategy": "coverage_hybrid_all_mwf",
                    "output_pesq": candidate,
                    "output_stoi": 0.6,
                    "si_sdr_improvement_db": 3.0,
                }
            )
    comparisons = _paired_comparisons(
        rows,
        baseline="distributed_singles_mwf",
        seed=7,
        bootstrap_samples=200,
    )
    result = next(row for row in comparisons if row["strategy"] == "coverage_hybrid_all_mwf")
    assert result["cases"] == 4
    assert result["floorplans"] == 2
    assert result["mean_delta_output_pesq"] == 1.0
    assert result["win_rate_output_stoi"] == 1.0


def test_wpe_and_wpd_are_finite_without_boundary_impulses() -> None:
    rng = np.random.default_rng(17)
    signals = rng.normal(size=(4, 8192))
    filters = estimate_wpe_filters(signals)
    dereverberated = apply_wpe(signals, filters)
    weights = estimate_wpd_weights(signals, [0.0, 0.00005, 0.0001, 0.00015], 16_000)
    enhanced = apply_wpd_beamformer(signals, weights)
    assert dereverberated.shape == signals.shape
    assert enhanced.shape == (signals.shape[1],)
    assert np.all(np.isfinite(dereverberated))
    assert np.all(np.isfinite(enhanced))
    assert np.max(np.abs(dereverberated)) < np.max(np.abs(signals)) * 2.0
