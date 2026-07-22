from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from acoustic_agent import AcousticAgent, microphone_array
from acoustic_agent.audio import render_audio
from acoustic_agent.mic import channel_positions
from research.doa.distributed import (
    FloorplanModel,
    SensorNode,
    TOAMeasurement,
    candidate_nodes,
    load_model,
    localization_grid,
    localize_tdoa,
    place_nodes,
    sample_target_points,
)
from research.doa.estimators import angular_error_deg, azimuth_deg, estimate_srp_phat

from .audio_io import load_wav_mono, write_wav_mono
from .core import (
    SPEED_OF_SOUND_M_S,
    apply_stft_gain,
    align_signals,
    aligned_si_sdr_db,
    arrival_times_from_azimuth,
    beamform_components,
    estimate_gcc_phat_delays,
    estimate_wiener_gain,
    optional_stoi,
    rir_arrival_times,
    scale_background_to_snr,
    snr_db,
)


FS = 16_000
RESOURCE_AUDIO = Path(__file__).resolve().parents[2] / "acoustic_agent" / "resources" / "audio"


def run_beamforming_study(
    output_dir: str | Path,
    *,
    study: str = "all",
    quality: str = "preview",
    floorplan_idx: int = 0,
    distributed_nodes: int = 8,
    subset_counts: Sequence[int] = (2, 4, 6, 8),
    duration_s: float = 2.4,
    rir_duration_s: float = 1.0,
    target_interferer_snr_db: float = 0.0,
    target_sensor_noise_snr_db: float = 18.0,
    rt_accelerator: str = "numba",
    rt_precision: str = "float64",
    rt_cuda_device: int = 0,
) -> dict[str, Any]:
    """Run the local-array and distributed synchronized-single baselines."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_study = str(study).strip().lower()
    if selected_study not in {"all", "local", "distributed"}:
        raise ValueError("study must be all, local, or distributed")
    target_dry, interferer_dry = _study_audio(float(duration_s), FS)
    rows: list[dict[str, Any]] = []
    distributed_metadata = None
    if selected_study in {"all", "local"}:
        rows.extend(
            run_local_array_study(
                output / "local-array",
                target_dry=target_dry,
                interferer_dry=interferer_dry,
                quality=quality,
                rir_duration_s=rir_duration_s,
                target_interferer_snr_db=target_interferer_snr_db,
                target_sensor_noise_snr_db=target_sensor_noise_snr_db,
                rt_accelerator=rt_accelerator,
                rt_precision=rt_precision,
                rt_cuda_device=rt_cuda_device,
            )
        )
    if selected_study in {"all", "distributed"}:
        distributed_rows, distributed_metadata = run_distributed_study(
            output / "distributed",
            target_dry=target_dry,
            interferer_dry=interferer_dry,
            floorplan_idx=floorplan_idx,
            quality=quality,
            rir_duration_s=rir_duration_s,
            distributed_nodes=distributed_nodes,
            subset_counts=subset_counts,
            target_interferer_snr_db=target_interferer_snr_db,
            target_sensor_noise_snr_db=target_sensor_noise_snr_db,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        rows.extend(distributed_rows)
    payload = {
        "study": "classical_beamforming_baselines_v1",
        "study_scope": selected_study,
        "sample_rate_hz": FS,
        "quality": quality,
        "floorplan_idx": int(floorplan_idx),
        "distributed_nodes": int(distributed_nodes),
        "subset_counts": [int(value) for value in subset_counts],
        "target_interferer_snr_db": float(target_interferer_snr_db),
        "target_sensor_noise_snr_db": float(target_sensor_noise_snr_db),
        "rt_accelerator": str(rt_accelerator),
        "rt_precision": str(rt_precision),
        "rt_cuda_device": int(rt_cuda_device),
        "distributed": distributed_metadata,
        "results": rows,
    }
    _write_results(output, payload)
    return payload


def run_local_array_study(
    output_dir: str | Path,
    *,
    target_dry: np.ndarray,
    interferer_dry: np.ndarray,
    quality: str,
    rir_duration_s: float,
    target_interferer_snr_db: float,
    target_sensor_noise_snr_db: float,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    receiver = np.asarray([3.0, 2.5, 1.4], dtype=np.float64)
    target = receiver + _bearing_vector(35.0) * 1.7
    interferer = receiver + _bearing_vector(140.0) * 1.8
    receiver_model = microphone_array("circular", count=4, radius_m=0.05)
    room = {
        "shape": "rectangle",
        "size": [6.0, 5.0, 2.8],
        "material_profile": {"wall": "auto", "floor": "auto", "ceiling": "auto"},
        "material_seed": 2026,
    }
    agent = AcousticAgent.create(
        scene="geometry",
        room=room,
        source=target.tolist(),
        receiver=receiver.tolist(),
        receiver_model=receiver_model,
        source_model="omni",
        quality=quality,
        duration_s=rir_duration_s,
        fs=FS,
        seed=20260722,
        visualization=False,
    )
    config = replace(
        agent.config,
        duration_s=float(rir_duration_s),
        rt_duration_s=float(rir_duration_s),
        collect_visual_paths=False,
        render_ambisonics=False,
        rt_accelerator=str(rt_accelerator),
        rt_precision=str(rt_precision),
        rt_cuda_device=int(rt_cuda_device),
    )
    simulation = agent.run_sources(
        {"target": target.tolist(), "interferer": interferer.tolist()},
        receiver=receiver.tolist(),
        config=config,
        receiver_model=receiver_model,
    )
    target_rir = np.asarray(simulation["target"].rir, dtype=np.float64)
    interferer_rir = np.asarray(simulation["interferer"].rir, dtype=np.float64)
    target_audio = _fit_length(render_audio(target_dry, target_rir), target_dry.size)
    interferer_audio = _fit_length(render_audio(interferer_dry, interferer_rir), target_dry.size)
    evaluation = _evaluation_slice(target_dry.size)
    interferer_audio, _ = scale_background_to_snr(
        target_audio,
        interferer_audio,
        target_interferer_snr_db,
        sample_slice=evaluation,
    )
    sensor_noise = _sensor_noise(target_audio.shape, seed=41)
    sensor_noise, _ = scale_background_to_snr(
        target_audio,
        sensor_noise,
        target_sensor_noise_snr_db,
        sample_slice=evaluation,
    )
    noise_audio = np.asarray(interferer_audio + sensor_noise, dtype=np.float32)
    mixture = np.asarray(target_audio + noise_audio, dtype=np.float32)
    calibration = _calibration_slice(target_dry.size)
    channel_rms = np.sqrt(np.mean(np.square(target_audio[:, calibration], dtype=np.float64), axis=1))
    reference_channel = int(np.argmax(channel_rms))
    baseline = {
        "target": target_audio[reference_channel],
        "noise": noise_audio[reference_channel],
        "mixture": mixture[reference_channel],
    }
    input_snr = snr_db(baseline["target"], baseline["noise"], sample_slice=evaluation)
    input_si_sdr = aligned_si_sdr_db(
        baseline["mixture"][evaluation],
        target_dry[evaluation],
        max_shift_samples=int(0.25 * FS),
    )
    common = {
        "scene": "geometry",
        "scenario": "same_room_interferer",
        "physical_microphones": 4,
        "input_reference": f"array_channel_{reference_channel}",
        "true_azimuth_deg": round(azimuth_deg(receiver, target), 4),
        "estimated_azimuth_deg": None,
        "doa_error_deg": None,
        "selected_microphones": [reference_channel],
    }
    rows = [
        _metric_row(
            "single_raw",
            baseline,
            baseline,
            input_snr,
            input_si_sdr,
            evaluation,
            clean_reference=target_dry,
            processing_s=0.0,
            fs=FS,
            metadata=common,
        )
    ]

    start = time.perf_counter()
    wiener_gain = estimate_wiener_gain(
        baseline["mixture"],
        baseline["noise"][: max(512, calibration.stop - calibration.start)],
    )
    wiener = {
        name: apply_stft_gain(value, wiener_gain)
        for name, value in baseline.items()
    }
    elapsed = time.perf_counter() - start
    rows.append(
        _metric_row(
            "single_wiener",
            wiener,
            baseline,
            input_snr,
            input_si_sdr,
            evaluation,
            clean_reference=target_dry,
            processing_s=elapsed,
            fs=FS,
            metadata=common,
        )
    )

    microphones = np.asarray(channel_positions(receiver, receiver_model), dtype=np.float64)
    oracle_arrivals = rir_arrival_times(target_rir, FS)
    start = time.perf_counter()
    oracle = beamform_components(
        {"target": target_audio, "noise": noise_audio, "mixture": mixture},
        oracle_arrivals,
        FS,
    )
    elapsed = time.perf_counter() - start
    rows.append(
        _metric_row(
            "array4_ds_oracle",
            oracle,
            baseline,
            input_snr,
            input_si_sdr,
            evaluation,
            clean_reference=target_dry,
            processing_s=elapsed,
            fs=FS,
            metadata={**common, "selected_microphones": list(range(4))},
        )
    )

    estimated_azimuth, _ = estimate_srp_phat(
        target_audio[:, calibration] + sensor_noise[:, calibration],
        microphones,
        fs=FS,
        search_deg=np.arange(0.0, 360.0, 1.0),
        speed_of_sound_m_s=SPEED_OF_SOUND_M_S,
    )
    estimated_arrivals = arrival_times_from_azimuth(estimated_azimuth, microphones)
    start = time.perf_counter()
    estimated = beamform_components(
        {"target": target_audio, "noise": noise_audio, "mixture": mixture},
        estimated_arrivals,
        FS,
    )
    elapsed = time.perf_counter() - start
    truth_azimuth = azimuth_deg(receiver, target)
    rows.append(
        _metric_row(
            "array4_ds_estimated_doa",
            estimated,
            baseline,
            input_snr,
            input_si_sdr,
            evaluation,
            clean_reference=target_dry,
            processing_s=elapsed,
            fs=FS,
            metadata={
                **common,
                "estimated_azimuth_deg": round(float(estimated_azimuth), 4),
                "doa_error_deg": round(angular_error_deg(estimated_azimuth, truth_azimuth), 4),
                "selected_microphones": list(range(4)),
            },
        )
    )
    _write_audio_rows(output, rows, {
        "single_raw": baseline["mixture"],
        "single_wiener": wiener["mixture"],
        "array4_ds_oracle": oracle["mixture"],
        "array4_ds_estimated_doa": estimated["mixture"],
    })
    return rows


def run_distributed_study(
    output_dir: str | Path,
    *,
    target_dry: np.ndarray,
    interferer_dry: np.ndarray,
    floorplan_idx: int,
    quality: str,
    rir_duration_s: float,
    distributed_nodes: int,
    subset_counts: Sequence[int],
    target_interferer_snr_db: float,
    target_sensor_noise_snr_db: float,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(int(floorplan_idx))
    candidates = candidate_nodes(model, positions_per_room=2)
    nodes = place_nodes(
        model,
        min(max(int(distributed_nodes), 2), len(candidates)),
        mode="single",
        risk_quantile=0.2,
        candidates=candidates,
    )
    targets = sample_target_points(model, points_per_room=3, seed=20260722)
    source_room = max(model.rooms, key=lambda room_id: float(model.rooms[room_id]["area_m2"]))
    source_points = [point for point in targets if point.room_id == source_room]
    target = source_points[0]
    same_room_interferer = max(
        source_points[1:],
        key=lambda point: float(np.linalg.norm(np.asarray(point.position) - np.asarray(target.position))),
    )
    adjacent_rooms = [room_id for room_id, _ in model.adjacency[source_room]]
    cross_room = max(adjacent_rooms, key=lambda room_id: float(model.rooms[room_id]["area_m2"]))
    cross_room_interferer = next(point for point in targets if point.room_id == cross_room)
    rows: list[dict[str, Any]] = []
    scenario_metadata = []
    target_rirs = _distributed_rirs(
        output,
        model,
        target,
        nodes,
        quality=quality,
        rir_duration_s=rir_duration_s,
        rt_accelerator=rt_accelerator,
        rt_precision=rt_precision,
        rt_cuda_device=rt_cuda_device,
    )
    target_audio = _render_distributed(target_dry, target_rirs, target_dry.size)

    for scenario_name, interferer in (
        ("same_room_interferer", same_room_interferer),
        ("cross_room_interferer", cross_room_interferer),
    ):
        scenario_outputs: dict[str, np.ndarray] = {}
        interferer_rirs = _distributed_rirs(
            output,
            model,
            interferer,
            nodes,
            quality=quality,
            rir_duration_s=rir_duration_s,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        interferer_audio = _render_distributed(interferer_dry, interferer_rirs, target_dry.size)
        evaluation = _evaluation_slice(target_dry.size)
        interferer_audio, _ = scale_background_to_snr(
            target_audio,
            interferer_audio,
            target_interferer_snr_db,
            sample_slice=evaluation,
        )
        sensor_noise = _sensor_noise(target_audio.shape, seed=101 + len(rows))
        sensor_noise, _ = scale_background_to_snr(
            target_audio,
            sensor_noise,
            target_sensor_noise_snr_db,
            sample_slice=evaluation,
        )
        noise_audio = np.asarray(interferer_audio + sensor_noise, dtype=np.float32)
        mixture = np.asarray(target_audio + noise_audio, dtype=np.float32)
        calibration = _calibration_slice(target_dry.size)
        # These represent observable target-active and target-silent calibration
        # intervals, not clean-component access in the estimated pipeline.
        target_calibration = target_audio[:, calibration] + sensor_noise[:, calibration]
        noise_calibration = noise_audio[:, calibration]
        activity_rms = np.sqrt(np.mean(np.square(target_calibration, dtype=np.float64), axis=1))
        noise_rms = np.sqrt(np.mean(np.square(noise_calibration, dtype=np.float64), axis=1))
        reliability = activity_rms / np.maximum(noise_rms, 1e-12)
        reference_index = int(np.argmax(reliability))
        baseline = {
            "target": target_audio[reference_index],
            "noise": noise_audio[reference_index],
            "mixture": mixture[reference_index],
        }
        input_snr = snr_db(baseline["target"], baseline["noise"], sample_slice=evaluation)
        input_si_sdr = aligned_si_sdr_db(
            baseline["mixture"][evaluation],
            target_dry[evaluation],
            max_shift_samples=int(0.35 * FS),
        )
        common = {
            "scene": "floorplan",
            "floorplan_idx": int(floorplan_idx),
            "scenario": scenario_name,
            "physical_microphones": len(nodes),
            "target_room": target.room_id,
            "interferer_room": interferer.room_id,
            "input_reference": nodes[reference_index].id,
            "estimated_room": None,
            "localization_error_m": None,
            "room_correct": None,
        }
        rows.append(
            _metric_row(
                "distributed_best_single",
                baseline,
                baseline,
                input_snr,
                input_si_sdr,
                evaluation,
                clean_reference=target_dry,
                processing_s=0.0,
                fs=FS,
                metadata={**common, "selected_microphones": [nodes[reference_index].id]},
            )
        )

        measured_delays, confidence = estimate_gcc_phat_delays(
            target_calibration,
            FS,
            max_delay_s=0.15,
            reference_channel=reference_index,
        )
        measurements = [
            TOAMeasurement(node.id, float(measured_delays[index]), float(confidence[index]))
            for index, node in enumerate(nodes)
        ]
        estimated_xy, estimated_room, _ = localize_tdoa(
            model,
            nodes,
            measurements,
            localization_grid(model, spacing_m=0.35),
        )
        estimated_position = np.asarray([estimated_xy[0], estimated_xy[1], target.position[2]], dtype=np.float64)
        localization_error = float(np.linalg.norm(estimated_position[:2] - np.asarray(target.position[:2])))

        usable_counts = sorted({min(max(int(value), 2), len(nodes)) for value in subset_counts})
        for microphone_count in usable_counts:
            oracle_indices = _select_node_indices(
                model,
                nodes,
                target.position,
                target.room_id,
                reliability,
                microphone_count,
            )
            oracle_weights = _selection_weights(reliability[oracle_indices])
            oracle_arrivals = np.asarray(
                [rir_arrival_times(target_rirs[index], FS)[0] for index in oracle_indices],
                dtype=np.float64,
            )
            start = time.perf_counter()
            oracle = beamform_components(
                {
                    "target": target_audio[oracle_indices],
                    "noise": noise_audio[oracle_indices],
                    "mixture": mixture[oracle_indices],
                },
                oracle_arrivals,
                FS,
                weights=oracle_weights,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                _metric_row(
                    f"distributed_ds_oracle_{microphone_count}",
                    oracle,
                    baseline,
                    input_snr,
                    input_si_sdr,
                    evaluation,
                    clean_reference=target_dry,
                    processing_s=elapsed,
                    fs=FS,
                    metadata={
                        **common,
                        "selected_microphones": [nodes[index].id for index in oracle_indices],
                        "source_room_microphones": sum(nodes[index].room_id == target.room_id for index in oracle_indices),
                    },
                )
            )
            scenario_outputs[f"distributed_ds_oracle_{microphone_count}"] = oracle["mixture"]

            estimated_indices = _select_node_indices(
                model,
                nodes,
                estimated_position,
                estimated_room,
                reliability,
                microphone_count,
            )
            estimated_weights = _selection_weights(reliability[estimated_indices])
            estimated_arrivals = np.asarray(
                [
                    model.propagation(
                        estimated_position,
                        estimated_room,
                        nodes[index].position,
                        nodes[index].room_id,
                    )[1]
                    / SPEED_OF_SOUND_M_S
                    for index in estimated_indices
                ],
                dtype=np.float64,
            )
            start = time.perf_counter()
            estimated = beamform_components(
                {
                    "target": target_audio[estimated_indices],
                    "noise": noise_audio[estimated_indices],
                    "mixture": mixture[estimated_indices],
                },
                estimated_arrivals,
                FS,
                weights=estimated_weights,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                _metric_row(
                    f"distributed_ds_estimated_{microphone_count}",
                    estimated,
                    baseline,
                    input_snr,
                    input_si_sdr,
                    evaluation,
                    clean_reference=target_dry,
                    processing_s=elapsed,
                    fs=FS,
                    metadata={
                        **common,
                        "estimated_room": estimated_room,
                        "localization_error_m": round(localization_error, 4),
                        "room_correct": bool(estimated_room == target.room_id),
                        "selected_microphones": [nodes[index].id for index in estimated_indices],
                        "source_room_microphones": sum(nodes[index].room_id == target.room_id for index in estimated_indices),
                    },
                )
            )
            scenario_outputs[f"distributed_ds_estimated_{microphone_count}"] = estimated["mixture"]

        without_local = [index for index, node in enumerate(nodes) if node.room_id != target.room_id]
        if len(without_local) >= 2:
            count = min(4, len(without_local))
            ranked_without_local = _select_node_indices(
                model,
                [nodes[index] for index in without_local],
                target.position,
                target.room_id,
                reliability[without_local],
                count,
            )
            selected = [without_local[index] for index in ranked_without_local]
            no_local_weights = _selection_weights(reliability[selected])
            arrivals = np.asarray([rir_arrival_times(target_rirs[index], FS)[0] for index in selected])
            start = time.perf_counter()
            no_local = beamform_components(
                {
                    "target": target_audio[selected],
                    "noise": noise_audio[selected],
                    "mixture": mixture[selected],
                },
                arrivals,
                FS,
                weights=no_local_weights,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                _metric_row(
                    f"distributed_ds_oracle_{count}_no_source_room",
                    no_local,
                    baseline,
                    input_snr,
                    input_si_sdr,
                    evaluation,
                    clean_reference=target_dry,
                    processing_s=elapsed,
                    fs=FS,
                    metadata={
                        **common,
                        "selected_microphones": [nodes[index].id for index in selected],
                        "source_room_microphones": 0,
                    },
                )
            )
        scenario_rows = [row for row in rows if row["scenario"] == scenario_name]
        best = max(scenario_rows, key=lambda row: float(row["snr_improvement_db"]))
        audio_by_name = {"distributed_best_single": baseline["mixture"]}
        if best["configuration"] in scenario_outputs:
            audio_by_name[str(best["configuration"])] = scenario_outputs[str(best["configuration"])]
        scenario_metadata.append(
            {
                "scenario": scenario_name,
                "interferer_room": interferer.room_id,
                "estimated_target_room": estimated_room,
                "localization_error_m": round(localization_error, 4),
                "best_configuration": best["configuration"],
                "best_snr_improvement_db": best["snr_improvement_db"],
            }
        )
        _write_audio_rows(output / scenario_name, scenario_rows, audio_by_name)

    metadata = {
        "floorplan_idx": int(floorplan_idx),
        "target_room": target.room_id,
        "target_position_m": list(target.position),
        "nodes": [
            {"id": node.id, "room_id": node.room_id, "position_m": list(node.position)}
            for node in nodes
        ],
        "scenarios": scenario_metadata,
    }
    return rows, metadata


def _distributed_rirs(
    output: Path,
    model: FloorplanModel,
    source: Any,
    nodes: Sequence[SensorNode],
    *,
    quality: str,
    rir_duration_s: float,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> list[np.ndarray]:
    cache = output / "rir-cache"
    cache.mkdir(parents=True, exist_ok=True)
    responses = []
    for node in nodes:
        signature = json.dumps(
            {
                "version": 1,
                "floorplan_idx": model.index,
                "source": source.position,
                "source_room": source.room_id,
                "receiver": node.position,
                "receiver_room": node.room_id,
                "quality": quality,
                "duration_s": rir_duration_s,
                "accelerator": rt_accelerator,
                "precision": rt_precision,
                "device": rt_cuda_device,
            },
            sort_keys=True,
        )
        path = cache / f"{hashlib.sha256(signature.encode('utf-8')).hexdigest()}.npz"
        if path.is_file():
            responses.append(np.asarray(np.load(path)["rir"], dtype=np.float32))
            continue
        agent = AcousticAgent.create(
            scene="floorplan",
            idx=model.index,
            placement="same_room" if source.room_id == node.room_id else "cross_room",
            source=source.position,
            receiver=node.position,
            source_room=source.room_id,
            receiver_room=node.room_id,
            seed=77_000 + model.index,
            material_seed=2026,
            receiver_model="mono",
            source_model="omni",
            quality=quality,
            duration_s=rir_duration_s,
            fs=FS,
            visualization=False,
        )
        config = replace(
            agent.config,
            duration_s=float(rir_duration_s),
            rt_duration_s=float(rir_duration_s),
            late_tail=True,
            collect_visual_paths=False,
            render_ambisonics=False,
            rt_accelerator=str(rt_accelerator),
            rt_precision=str(rt_precision),
            rt_cuda_device=int(rt_cuda_device),
        )
        response = np.asarray(agent.run(config=config).rir, dtype=np.float32).reshape(1, -1)
        np.savez_compressed(path, rir=response)
        responses.append(response)
    return responses


def _render_distributed(dry: np.ndarray, rirs: Sequence[np.ndarray], length: int) -> np.ndarray:
    return np.vstack([_fit_length(render_audio(dry, rir), length)[0] for rir in rirs]).astype(np.float32)


def _select_node_indices(
    model: FloorplanModel,
    nodes: Sequence[SensorNode],
    source_position: Sequence[float],
    source_room: str,
    activity_rms: Sequence[float],
    count: int,
) -> list[int]:
    levels = np.asarray(activity_rms, dtype=np.float64)
    level_db = 20.0 * np.log10(np.maximum(levels, 1e-12))
    best_level = float(np.max(level_db))
    ranking = []
    for index, node in enumerate(nodes):
        try:
            _, distance, hops = model.propagation(source_position, source_room, node.position, node.room_id)
        except ValueError:
            distance, hops = 1e6, 1_000
        local_penalty = 0 if node.room_id == source_room else 1
        weak_penalty = max(0.0, best_level - float(level_db[index]))
        ranking.append(((local_penalty, hops, weak_penalty, distance, node.id), index))
    return [index for _, index in sorted(ranking)[: min(max(int(count), 1), len(nodes))]]


def _metric_row(
    configuration: str,
    output: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    input_snr: float,
    input_si_sdr: float,
    evaluation: slice,
    *,
    clean_reference: np.ndarray,
    processing_s: float,
    fs: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output_snr = snr_db(output["target"], output["noise"], sample_slice=evaluation)
    output_si_sdr = aligned_si_sdr_db(
        output["mixture"][evaluation],
        clean_reference[evaluation],
        max_shift_samples=int(0.35 * fs),
    )
    aligned_output, aligned_reference, _ = align_signals(
        output["mixture"][evaluation],
        clean_reference[evaluation],
        max_shift_samples=int(0.35 * fs),
    )
    stoi_value = optional_stoi(aligned_output, aligned_reference, fs)
    duration_s = max(1, output["mixture"][evaluation].size) / fs
    return {
        **dict(metadata),
        "configuration": str(configuration),
        "input_snr_db": round(float(input_snr), 4),
        "output_snr_db": round(float(output_snr), 4),
        "snr_improvement_db": round(float(output_snr - input_snr), 4),
        "input_si_sdr_db": round(float(input_si_sdr), 4),
        "output_si_sdr_db": round(float(output_si_sdr), 4),
        "si_sdr_improvement_db": round(float(output_si_sdr - input_si_sdr), 4),
        "stoi": None if stoi_value is None else round(float(stoi_value), 5),
        "processing_s": round(float(processing_s), 6),
        "rtf": round(float(processing_s) / duration_s, 6),
    }


def _study_audio(duration_s: float, fs: int) -> tuple[np.ndarray, np.ndarray]:
    sample_count = max(int(round(duration_s * fs)), fs)
    target = load_wav_mono(RESOURCE_AUDIO / "main_voice.wav", fs)
    interferer = load_wav_mono(RESOURCE_AUDIO / "background_speech.wav", fs)
    return _tile_audio(target, sample_count), _tile_audio(interferer, sample_count)


def _selection_weights(reliability: Sequence[float]) -> np.ndarray:
    values = np.asarray(reliability, dtype=np.float64)
    positive = np.maximum(values, 1e-12)
    ceiling = max(float(np.percentile(positive, 75)), float(np.min(positive)))
    return np.clip(positive, ceiling * 0.1, ceiling)


def _tile_audio(samples: np.ndarray, sample_count: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    repeats = max(1, int(math.ceil(sample_count / values.size)))
    return np.tile(values, repeats)[:sample_count].astype(np.float32)


def _sensor_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.normal(0.0, 1.0, shape).astype(np.float32)


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.shape[1] >= length:
        return values[:, :length]
    return np.pad(values, ((0, 0), (0, length - values.shape[1])))


def _calibration_slice(sample_count: int) -> slice:
    return slice(int(round(sample_count * 0.10)), int(round(sample_count * 0.35)))


def _evaluation_slice(sample_count: int) -> slice:
    return slice(int(round(sample_count * 0.45)), sample_count)


def _bearing_vector(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    return np.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64)


def _write_audio_rows(output: Path, rows: Sequence[Mapping[str, Any]], audio: Mapping[str, np.ndarray]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for configuration, samples in audio.items():
        if any(row["configuration"] == configuration for row in rows):
            write_wav_mono(output / f"{configuration}.wav", np.asarray(samples), FS)


def _write_results(output: Path, payload: Mapping[str, Any]) -> None:
    rows = list(payload["results"])
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Classical Beamforming Baselines",
        "",
        f"- Quality: `{payload['quality']}`",
        f"- FloorPlan index: `{payload['floorplan_idx']}`",
        f"- Distributed microphones: `{payload['distributed_nodes']}`",
        f"- Target/interferer input SNR: `{payload['target_interferer_snr_db']} dB`",
        "",
        "| Scene | Interferer | Configuration | SNR in | SNR out | SNR improvement | SI-SDR improvement | RTF |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['scenario']} | {row['configuration']} | "
            f"{row['input_snr_db']:.2f} | {row['output_snr_db']:.2f} | "
            f"{row['snr_improvement_db']:.2f} | {row['si_sdr_improvement_db']:.2f} | {row['rtf']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`STOI` is populated when the optional `pystoi` package is installed.",
            "RIR simulation time is excluded from RTF; RTF measures enhancement processing only.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
