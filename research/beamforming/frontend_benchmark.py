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
from .benchmark import (
    DEFAULT_ROOM_COUNTS,
    _aggregate,
    _benchmark_evaluation_slice,
    _farthest_room,
    _format_metric,
    _quality_metrics,
    _recommended_microphone_counts,
    _round_optional,
    _validation_split,
    _write_csv,
)
from .core import (
    align_channels,
    apply_stft_beamformer,
    apply_stft_gain,
    apply_wpd_beamformer,
    apply_wpe,
    arrival_times_from_azimuth,
    estimate_adaptive_beamformer_weights,
    estimate_gcc_phat_delays,
    estimate_wiener_gain,
    estimate_wpd_weights,
    estimate_wpe_filters,
    scale_background_to_snr,
)
from .experiment import (
    FS,
    RESOURCE_AUDIO,
    _calibration_slice,
    _distributed_rirs,
    _render_distributed,
    _sensor_noise,
    _study_audio,
    _tile_audio,
)


FRONTEND_PIPELINES = (
    "single_raw",
    "single_wiener",
    "local_ds",
    "local_mvdr",
    "local_mwf",
    "local_wpe_mvdr",
    "local_wpe_mwf",
    "local_wpd",
    "distributed_weighted_ds",
    "distributed_mvdr",
    "distributed_mwf",
    "distributed_wpe_mvdr",
    "distributed_wpe_mwf",
    "distributed_wpd",
)


def run_frontend_benchmark(
    output_dir: str | Path,
    *,
    room_counts: Sequence[int] = DEFAULT_ROOM_COUNTS,
    plans_per_room_count: int = 5,
    quality: str = "preview",
    duration_s: float = 2.5,
    rir_duration_s: float = 1.0,
    interferer_snr_db: float = 0.0,
    background_snr_db: float = 10.0,
    sensor_noise_snr_db: float = 30.0,
    rt_accelerator: str = "numba",
    rt_precision: str = "float64",
    rt_cuda_device: int = 0,
    seed: int = 20260723,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_counts = tuple(int(value) for value in room_counts)
    split = _validation_split(selected_counts, int(plans_per_room_count))
    distributed_counts = _recommended_microphone_counts(selected_counts)
    target_dry, interferer_dry = _study_audio(float(duration_s), FS)
    pink_noise = _tile_audio(load_wav_mono(RESOURCE_AUDIO / "pink_noise_bed.wav", FS), target_dry.size)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for plan_number, split_row in enumerate(split, start=1):
        floorplan_idx = int(split_row["index"])
        room_count = int(split_row["room_count"])
        model = load_model(floorplan_idx)
        points = sample_target_points(model, points_per_room=3, seed=seed + floorplan_idx)
        source_room = max(model.rooms, key=lambda room_id: float(model.rooms[room_id]["area_m2"]))
        source_points = [point for point in points if point.room_id == source_room]
        target = source_points[0]
        same_interferer = max(
            source_points[1:],
            key=lambda point: float(np.linalg.norm(np.asarray(point.position) - np.asarray(target.position))),
        )
        adjacent_rooms = [room_id for room_id, _ in model.adjacency[source_room]]
        cross_room = max(adjacent_rooms, key=lambda room_id: float(model.rooms[room_id]["area_m2"]))
        cross_interferer = next(point for point in points if point.room_id == cross_room)
        noise_source = next(
            point for point in points if point.room_id == _farthest_room(model, source_room)
        )
        candidates = candidate_nodes(model, positions_per_room=2)
        distributed_nodes = place_nodes(
            model,
            distributed_counts[room_count],
            mode="single",
            risk_quantile=0.2,
            candidates=candidates,
        )
        local_node = next(
            (node for node in candidates if node.room_id == source_room and node.id.endswith(":center")),
            next(node for node in candidates if node.room_id == source_room),
        )
        distributed_cache = output / "distributed"
        local_cache = output / "local-array"
        distributed_target_rirs = _distributed_rirs(
            distributed_cache,
            model,
            target,
            distributed_nodes,
            quality=quality,
            rir_duration_s=rir_duration_s,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        distributed_noise_rirs = _distributed_rirs(
            distributed_cache,
            model,
            noise_source,
            distributed_nodes,
            quality=quality,
            rir_duration_s=rir_duration_s,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        local_target_rir, local_model = _local_array_rir(
            local_cache,
            model,
            target,
            local_node,
            quality=quality,
            rir_duration_s=rir_duration_s,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        local_noise_rir, _ = _local_array_rir(
            local_cache,
            model,
            noise_source,
            local_node,
            quality=quality,
            rir_duration_s=rir_duration_s,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        distributed_target = _render_distributed(
            target_dry, distributed_target_rirs, target_dry.size
        )
        distributed_pink = _render_distributed(
            pink_noise, distributed_noise_rirs, target_dry.size
        )
        local_target = _fit_render(target_dry, local_target_rir, target_dry.size)
        local_pink = _fit_render(pink_noise, local_noise_rir, target_dry.size)

        for scenario in ("same_room", "cross_room"):
            captured_audio: dict[str, np.ndarray] | None = {} if plan_number == 1 else None
            interferer = same_interferer if scenario == "same_room" else cross_interferer
            distributed_interferer_rirs = _distributed_rirs(
                distributed_cache,
                model,
                interferer,
                distributed_nodes,
                quality=quality,
                rir_duration_s=rir_duration_s,
                rt_accelerator=rt_accelerator,
                rt_precision=rt_precision,
                rt_cuda_device=rt_cuda_device,
            )
            local_interferer_rir, _ = _local_array_rir(
                local_cache,
                model,
                interferer,
                local_node,
                quality=quality,
                rir_duration_s=rir_duration_s,
                rt_accelerator=rt_accelerator,
                rt_precision=rt_precision,
                rt_cuda_device=rt_cuda_device,
            )
            distributed_interferer = _render_distributed(
                interferer_dry, distributed_interferer_rirs, target_dry.size
            )
            local_interferer = _fit_render(
                interferer_dry, local_interferer_rir, target_dry.size
            )
            common = {
                "floorplan_idx": floorplan_idx,
                "room_count": room_count,
                "area_m2": round(float(split_row["area_m2"]), 4),
                "scenario": scenario,
                "target_room": target.room_id,
                "interferer_room": interferer.room_id,
                "noise_room": noise_source.room_id,
            }
            distributed_components = _mix_components(
                distributed_target,
                distributed_interferer,
                distributed_pink,
                target_dry.size,
                interferer_snr_db,
                background_snr_db,
                sensor_noise_snr_db,
                seed + floorplan_idx * 31 + len(rows),
            )
            distributed_arrivals, distributed_metadata = _distributed_steering(
                model,
                distributed_nodes,
                distributed_components,
                target.position,
                target.room_id,
            )
            rows.extend(
                _run_architecture(
                    architecture="single",
                    pipeline_names=("single_raw", "single_wiener"),
                    components=distributed_components,
                    arrival_times_s=distributed_arrivals,
                    target_dry=target_dry,
                    physical_microphones=1,
                    common={**common, **distributed_metadata},
                    captured_audio=captured_audio,
                )
            )
            rows.extend(
                _run_architecture(
                    architecture="distributed_singles",
                    pipeline_names=(
                        "distributed_weighted_ds",
                        "distributed_mvdr",
                        "distributed_mwf",
                        "distributed_wpe_mvdr",
                        "distributed_wpe_mwf",
                        "distributed_wpd",
                    ),
                    components=distributed_components,
                    arrival_times_s=distributed_arrivals,
                    target_dry=target_dry,
                    physical_microphones=len(distributed_nodes),
                    common={**common, **distributed_metadata},
                    captured_audio=captured_audio,
                )
            )
            local_components = _mix_components(
                local_target,
                local_interferer,
                local_pink,
                target_dry.size,
                interferer_snr_db,
                background_snr_db,
                sensor_noise_snr_db,
                seed + floorplan_idx * 37 + len(rows),
            )
            local_arrivals, local_metadata = _local_steering(
                local_node,
                local_model,
                local_components,
                target.position,
            )
            local_rows = _run_architecture(
                architecture="local_array_4ch",
                pipeline_names=(
                    "local_ds",
                    "local_mvdr",
                    "local_mwf",
                    "local_wpe_mvdr",
                    "local_wpe_mwf",
                    "local_wpd",
                ),
                components=local_components,
                arrival_times_s=local_arrivals,
                target_dry=target_dry,
                physical_microphones=4,
                common={**common, **local_metadata},
                captured_audio=captured_audio,
            )
            rows.extend(local_rows)
            best = max(
                [row for row in rows if row["floorplan_idx"] == floorplan_idx and row["scenario"] == scenario],
                key=lambda row: float(row["output_pesq"] or -math.inf),
            )
            case_dir = output / "audio" / f"rooms-{room_count}" / f"floorplan-{floorplan_idx}" / scenario
            (case_dir / "best.json").parent.mkdir(parents=True, exist_ok=True)
            (case_dir / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
            if captured_audio is not None:
                write_wav_mono(case_dir / "clean-dry-reference.wav", target_dry, FS)
                for pipeline, samples in captured_audio.items():
                    write_wav_mono(case_dir / f"{pipeline}.wav", samples, FS)
        _write_frontend_checkpoint(output, rows)
        print(
            f"[{plan_number}/{len(split)}] floorplan={floorplan_idx} rooms={room_count} rows={len(rows)}",
            flush=True,
        )

    overall = _aggregate(rows, ("architecture", "pipeline"))
    room_pipeline = _aggregate(rows, ("room_count", "architecture", "pipeline"))
    scenario_pipeline = _aggregate(rows, ("scenario", "architecture", "pipeline"))
    payload = {
        "study": "complete_audio_frontend_benchmark_v1",
        "floorplan_count": len(split),
        "plans_per_room_count": int(plans_per_room_count),
        "room_counts": list(selected_counts),
        "pipelines": list(FRONTEND_PIPELINES),
        "distributed_microphones_by_room_count": distributed_counts,
        "quality": quality,
        "duration_s": float(duration_s),
        "rir_duration_s": float(rir_duration_s),
        "interferer_snr_db": float(interferer_snr_db),
        "background_snr_db": float(background_snr_db),
        "sensor_noise_snr_db": float(sensor_noise_snr_db),
        "elapsed_s": round(time.perf_counter() - started, 4),
        "overall": overall,
        "room_pipeline": room_pipeline,
        "scenario_pipeline": scenario_pipeline,
        "results": rows,
    }
    _write_frontend_report(output, payload)
    return payload


def _run_architecture(
    *,
    architecture: str,
    pipeline_names: Sequence[str],
    components: Mapping[str, np.ndarray],
    arrival_times_s: np.ndarray,
    target_dry: np.ndarray,
    physical_microphones: int,
    common: Mapping[str, Any],
    captured_audio: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    target = np.asarray(components["target"], dtype=np.float32)
    noise = np.asarray(components["noise"], dtype=np.float32)
    mixture = np.asarray(components["mixture"], dtype=np.float32)
    calibration = _calibration_slice(target.shape[1])
    evaluation = _benchmark_evaluation_slice(target.shape[1])
    sensor = np.asarray(components["sensor"], dtype=np.float32)
    target_calibration = target[:, calibration] + sensor[:, calibration]
    noise_calibration = noise[:, calibration]
    target_level = np.sqrt(np.mean(np.square(target_calibration, dtype=np.float64), axis=1))
    noise_level = np.sqrt(np.mean(np.square(noise_calibration, dtype=np.float64), axis=1))
    reliability = target_level / np.maximum(noise_level, 1e-12)
    reference = int(np.argmax(reliability))
    input_target = target[reference]
    input_noise = noise[reference]
    input_mixture = mixture[reference]
    input_metrics = _quality_metrics(
        input_mixture,
        input_target,
        input_noise,
        target_dry,
        input_target,
        evaluation,
    )
    prediction_components = {"target": target, "noise": noise, "mixture": mixture}
    prediction_sensor = sensor
    prediction_arrivals = np.asarray(arrival_times_s, dtype=np.float64)
    wpd_components = prediction_components
    wpd_sensor = prediction_sensor
    wpd_arrivals = prediction_arrivals
    if architecture == "distributed_singles":
        wpd_components = {
            name: align_channels(value, wpd_arrivals, FS)
            for name, value in wpd_components.items()
        }
        wpd_sensor = align_channels(sensor, wpd_arrivals, FS)
        wpd_arrivals = np.zeros_like(wpd_arrivals)
    prediction_target_calibration = (
        prediction_components["target"][:, calibration] + prediction_sensor[:, calibration]
    )
    prediction_noise_calibration = prediction_components["noise"][:, calibration]
    wpd_target_calibration = wpd_components["target"][:, calibration] + wpd_sensor[:, calibration]
    wpd_noise_calibration = wpd_components["noise"][:, calibration]
    wpe_needed = any("wpe" in pipeline for pipeline in pipeline_names)
    dereverberated: dict[str, np.ndarray] = {}
    dereverberated_target_calibration = None
    dereverberated_noise_calibration = None
    wpe_estimation_s = 0.0
    if wpe_needed:
        wpe_started = time.perf_counter()
        wpe_options = {"taps": 10 if architecture == "distributed_singles" else 5}
        if architecture == "distributed_singles":
            filters = [
                estimate_wpe_filters(
                    prediction_components["mixture"][index : index + 1], **wpe_options
                )
                for index in range(prediction_components["mixture"].shape[0])
            ]
        else:
            filters = estimate_wpe_filters(
                prediction_components["mixture"], **wpe_options
            )
        wpe_estimation_s = time.perf_counter() - wpe_started
        if architecture == "distributed_singles":
            dereverberated = {
                name: np.vstack(
                    [
                        apply_wpe(
                            value[index : index + 1], filters[index], **wpe_options
                        )[0]
                        for index in range(value.shape[0])
                    ]
                )
                for name, value in prediction_components.items()
            }
            dereverberated_target_calibration = np.vstack(
                [
                    apply_wpe(
                        prediction_target_calibration[index : index + 1],
                        filters[index],
                        **wpe_options,
                    )[0]
                    for index in range(prediction_target_calibration.shape[0])
                ]
            )
            dereverberated_noise_calibration = np.vstack(
                [
                    apply_wpe(
                        prediction_noise_calibration[index : index + 1],
                        filters[index],
                        **wpe_options,
                    )[0]
                    for index in range(prediction_noise_calibration.shape[0])
                ]
            )
        else:
            dereverberated = {
                name: apply_wpe(value, filters, **wpe_options)
                for name, value in prediction_components.items()
            }
            dereverberated_target_calibration = apply_wpe(
                prediction_target_calibration, filters, **wpe_options
            )
            dereverberated_noise_calibration = apply_wpe(
                prediction_noise_calibration, filters, **wpe_options
            )
    rows = []
    for pipeline in pipeline_names:
        processing_started = time.perf_counter()
        if pipeline == "single_raw":
            output_target, output_noise, output_mixture = input_target, input_noise, input_mixture
            processing_s = 0.0
        elif pipeline == "single_wiener":
            gain = estimate_wiener_gain(input_mixture, noise_calibration[reference])
            output_target = apply_stft_gain(input_target, gain)
            output_noise = apply_stft_gain(input_noise, gain)
            output_mixture = apply_stft_gain(input_mixture, gain)
            processing_s = time.perf_counter() - processing_started
        elif pipeline in {"local_ds", "distributed_weighted_ds"}:
            method = "ds" if pipeline == "local_ds" else "weighted_ds"
            weights = estimate_adaptive_beamformer_weights(
                target_calibration,
                noise_calibration,
                arrival_times_s,
                FS,
                algorithm=method,
                reference_channel=reference,
                reliability=reliability,
            )
            output_mixture = apply_stft_beamformer(mixture, weights)
            processing_s = time.perf_counter() - processing_started
            output_target = apply_stft_beamformer(target, weights)
            output_noise = apply_stft_beamformer(noise, weights)
        elif pipeline in {
            "local_mvdr",
            "local_mwf",
            "distributed_mvdr",
            "distributed_mwf",
        }:
            method = pipeline.rsplit("_", 1)[-1]
            weights = estimate_adaptive_beamformer_weights(
                target_calibration,
                noise_calibration,
                arrival_times_s,
                FS,
                algorithm=method,
                reference_channel=reference,
                reliability=reliability,
            )
            output_mixture = apply_stft_beamformer(mixture, weights)
            processing_s = time.perf_counter() - processing_started
            output_target = apply_stft_beamformer(target, weights)
            output_noise = apply_stft_beamformer(noise, weights)
        elif "wpe_" in pipeline:
            method = pipeline.rsplit("_", 1)[-1]
            weights = estimate_adaptive_beamformer_weights(
                dereverberated_target_calibration,
                dereverberated_noise_calibration,
                prediction_arrivals,
                FS,
                algorithm=method,
                reference_channel=reference,
                reliability=reliability,
            )
            output_mixture = apply_stft_beamformer(dereverberated["mixture"], weights)
            processing_s = wpe_estimation_s + time.perf_counter() - processing_started
            output_target = apply_stft_beamformer(dereverberated["target"], weights)
            output_noise = apply_stft_beamformer(dereverberated["noise"], weights)
        elif pipeline.endswith("_wpd"):
            weights = estimate_wpd_weights(
                np.concatenate(
                    [wpd_target_calibration, wpd_noise_calibration], axis=1
                ),
                wpd_arrivals,
                FS,
            )
            output_mixture = apply_wpd_beamformer(wpd_components["mixture"], weights)
            processing_s = time.perf_counter() - processing_started
            output_target = apply_wpd_beamformer(wpd_components["target"], weights)
            output_noise = apply_wpd_beamformer(wpd_components["noise"], weights)
        else:
            raise ValueError(f"unsupported front-end pipeline: {pipeline}")
        output_metrics = _quality_metrics(
            output_mixture,
            output_target,
            output_noise,
            target_dry,
            input_target,
            evaluation,
        )
        if captured_audio is not None:
            captured_audio[pipeline] = np.asarray(output_mixture, dtype=np.float32)
        duration_s = (evaluation.stop - evaluation.start) / FS
        rows.append(
            {
                **dict(common),
                "architecture": architecture,
                "pipeline": pipeline,
                "physical_microphones": int(physical_microphones),
                "reference_channel": reference,
                "input_snr_db": round(float(input_metrics["snr_db"]), 4),
                "output_snr_db": round(float(output_metrics["snr_db"]), 4),
                "snr_improvement_db": round(float(output_metrics["snr_db"] - input_metrics["snr_db"]), 4),
                "input_si_sdr_db": round(float(input_metrics["si_sdr_db"]), 4),
                "output_si_sdr_db": round(float(output_metrics["si_sdr_db"]), 4),
                "si_sdr_improvement_db": round(
                    float(output_metrics["si_sdr_db"] - input_metrics["si_sdr_db"]), 4
                ),
                "input_pesq": _round_optional(input_metrics["pesq"]),
                "output_pesq": _round_optional(output_metrics["pesq"]),
                "pesq_improvement": _optional_difference(output_metrics["pesq"], input_metrics["pesq"]),
                "input_stoi": _round_optional(input_metrics["stoi"]),
                "output_stoi": _round_optional(output_metrics["stoi"]),
                "stoi_improvement": _optional_difference(output_metrics["stoi"], input_metrics["stoi"]),
                "output_dry_pesq": _round_optional(output_metrics["dry_pesq"]),
                "output_dry_stoi": _round_optional(output_metrics["dry_stoi"]),
                "output_dry_si_sdr_db": round(float(output_metrics["dry_si_sdr_db"]), 4),
                "processing_s": round(float(processing_s), 6),
                "rtf": round(float(processing_s) / max(duration_s, 1e-9), 6),
            }
        )
    return rows


def _mix_components(
    target: np.ndarray,
    interferer: np.ndarray,
    pink: np.ndarray,
    sample_count: int,
    interferer_snr_db: float,
    background_snr_db: float,
    sensor_noise_snr_db: float,
    seed: int,
) -> dict[str, np.ndarray]:
    evaluation = _benchmark_evaluation_slice(sample_count)
    scaled_interferer, _ = scale_background_to_snr(
        target, interferer, interferer_snr_db, sample_slice=evaluation
    )
    scaled_pink, _ = scale_background_to_snr(
        target, pink, background_snr_db, sample_slice=evaluation
    )
    sensor = _sensor_noise(target.shape, seed)
    sensor, _ = scale_background_to_snr(
        target, sensor, sensor_noise_snr_db, sample_slice=evaluation
    )
    noise = np.asarray(scaled_interferer + scaled_pink + sensor, dtype=np.float32)
    return {
        "target": np.asarray(target, dtype=np.float32),
        "noise": noise,
        "mixture": np.asarray(target + noise, dtype=np.float32),
        "sensor": np.asarray(sensor, dtype=np.float32),
    }


def _distributed_steering(
    model: Any,
    nodes: Sequence[SensorNode],
    components: Mapping[str, np.ndarray],
    target_position: Sequence[float],
    target_room: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    calibration = _calibration_slice(components["target"].shape[1])
    observation = components["target"][:, calibration] + components["sensor"][:, calibration]
    reference = int(np.argmax(np.sqrt(np.mean(np.square(observation), axis=1))))
    delays, confidence = estimate_gcc_phat_delays(
        observation,
        FS,
        max_delay_s=0.2,
        reference_channel=reference,
    )
    measurements = [
        TOAMeasurement(node.id, float(delays[index]), float(confidence[index]))
        for index, node in enumerate(nodes)
    ]
    position, room_id, _ = localize_tdoa(
        model,
        nodes,
        measurements,
        localization_grid(model, spacing_m=0.45),
    )
    return delays, {
        "estimated_room": room_id,
        "room_correct": bool(room_id == target_room),
        "localization_error_m": round(
            float(np.linalg.norm(np.asarray(position) - np.asarray(target_position[:2]))), 4
        ),
        "doa_error_deg": None,
    }


def _local_steering(
    node: SensorNode,
    receiver_model: Mapping[str, Any],
    components: Mapping[str, np.ndarray],
    target_position: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    calibration = _calibration_slice(components["target"].shape[1])
    observation = components["target"][:, calibration] + components["sensor"][:, calibration]
    positions = np.asarray(channel_positions(node.position, receiver_model), dtype=np.float64)
    estimated, _ = estimate_srp_phat(
        observation,
        positions,
        fs=FS,
        search_deg=np.arange(0.0, 360.0, 1.0),
    )
    truth = azimuth_deg(node.position, target_position)
    return arrival_times_from_azimuth(estimated, positions), {
        "estimated_room": node.room_id,
        "room_correct": True,
        "localization_error_m": None,
        "doa_error_deg": round(angular_error_deg(estimated, truth), 4),
    }


def _local_array_rir(
    output: Path,
    model: Any,
    source: Any,
    node: SensorNode,
    *,
    quality: str,
    rir_duration_s: float,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    receiver_model = microphone_array("circular", count=4, radius_m=0.05)
    cache = output / "rir-cache"
    cache.mkdir(parents=True, exist_ok=True)
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
            "receiver_model": receiver_model,
        },
        sort_keys=True,
    )
    path = cache / f"{hashlib.sha256(signature.encode('utf-8')).hexdigest()}.npz"
    if path.is_file():
        return np.asarray(np.load(path)["rir"], dtype=np.float32), receiver_model
    agent = AcousticAgent.create(
        scene="floorplan",
        idx=model.index,
        placement="same_room" if source.room_id == node.room_id else "cross_room",
        source=source.position,
        receiver=node.position,
        source_room=source.room_id,
        receiver_room=node.room_id,
        seed=91_000 + model.index,
        material_seed=2026,
        receiver_model=receiver_model,
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
        collect_visual_paths=False,
        render_ambisonics=False,
        rt_accelerator=str(rt_accelerator),
        rt_precision=str(rt_precision),
        rt_cuda_device=int(rt_cuda_device),
    )
    rir = np.asarray(agent.run(config=config).rir, dtype=np.float32)
    np.savez_compressed(path, rir=rir)
    return rir, receiver_model


def _fit_render(dry: np.ndarray, rir: np.ndarray, sample_count: int) -> np.ndarray:
    audio = np.asarray(render_audio(dry, rir), dtype=np.float32)
    if audio.shape[1] >= sample_count:
        return audio[:, :sample_count]
    return np.pad(audio, ((0, 0), (0, sample_count - audio.shape[1])))


def _optional_difference(output: float | None, input_value: float | None) -> float | None:
    if output is None or input_value is None:
        return None
    return round(float(output - input_value), 5)


def _write_frontend_checkpoint(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    (output / "checkpoint.json").write_text(
        json.dumps({"results": list(rows)}, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_frontend_report(output: Path, payload: Mapping[str, Any]) -> None:
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output / "details.csv", payload["results"])
    _write_csv(output / "overall.csv", payload["overall"])
    _write_csv(output / "room-pipeline.csv", payload["room_pipeline"])
    _write_csv(output / "scenario-pipeline.csv", payload["scenario_pipeline"])
    lines = [
        "# Complete Audio Front-End Benchmark",
        "",
        f"- FloorPlans: `{payload['floorplan_count']}`",
        f"- Room counts: `{', '.join(str(value) for value in payload['room_counts'])}`",
        f"- RIR quality: `{payload['quality']}`",
        f"- Elapsed: `{payload['elapsed_s']:.2f} s`",
        "",
        "| Architecture | Pipeline | Mics | Input PESQ | PESQ | PESQ change | STOI | STOI change | SNR change | SI-SDR change | Dry PESQ | RTF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["overall"]:
        microphone_values = [
            int(item["physical_microphones"])
            for item in payload["results"]
            if item["architecture"] == row["architecture"] and item["pipeline"] == row["pipeline"]
        ]
        mic_label = str(int(round(float(np.mean(microphone_values)))))
        lines.append(
            f"| {row['architecture']} | {row['pipeline']} | {mic_label} | "
            f"{_format_metric(row['mean_input_pesq'])} | {_format_metric(row['mean_output_pesq'])} | "
            f"{_format_metric(row['mean_pesq_improvement'])} | {_format_metric(row['mean_output_stoi'])} | "
            f"{_format_metric(row['mean_stoi_improvement'])} | "
            f"{_format_metric(row['mean_snr_improvement_db'])} | "
            f"{_format_metric(row['mean_si_sdr_improvement_db'])} | "
            f"{_format_metric(row['mean_output_dry_pesq'])} | {_format_metric(row['mean_rtf'], digits=4)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Primary perceptual metrics use each architecture's fixed clean target image at its reference microphone.",
            "- Each architecture is level-calibrated over its own microphone set. Absolute scores compare complete systems; change metrics isolate processing benefit.",
            "- Dry PESQ uses the original anechoic source and therefore measures the harder denoising-plus-dereverberation task.",
            "- WPE is estimated from the observed mixture. Distributed WPE is applied independently at each node before TDOA beamforming.",
            "- RTF covers enhancement only and excludes RIR simulation, localization, and file I/O.",
            "",
            "## Recommended Pipelines by Scenario",
            "",
            "| Scenario | Pipeline | SNR change | SI-SDR change | PESQ | STOI |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    recommended = {"local_mwf", "distributed_mwf"}
    for row in payload["scenario_pipeline"]:
        if row["pipeline"] not in recommended:
            continue
        lines.append(
            f"| {row['scenario']} | {row['pipeline']} | "
            f"{_format_metric(row['mean_snr_improvement_db'])} | "
            f"{_format_metric(row['mean_si_sdr_improvement_db'])} | "
            f"{_format_metric(row['mean_output_pesq'])} | "
            f"{_format_metric(row['mean_output_stoi'])} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
