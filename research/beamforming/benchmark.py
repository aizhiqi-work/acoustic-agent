from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from research.doa.distributed import (
    TOAMeasurement,
    candidate_nodes,
    load_model,
    localization_grid,
    localize_tdoa,
    place_nodes,
    sample_target_points,
)

from .audio_io import load_wav_mono, write_wav_mono
from .core import (
    align_signals,
    aligned_si_sdr_db,
    apply_stft_beamformer,
    estimate_adaptive_beamformer_weights,
    estimate_gcc_phat_delays,
    optional_pesq,
    optional_stoi,
    scale_background_to_snr,
    snr_db,
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


DEFAULT_ROOM_COUNTS = (4, 6, 8, 10, 12)
DEFAULT_ALGORITHMS = ("best_single", "ds", "weighted_ds", "mvdr", "gev", "mwf")
BENCHMARK_MICROPHONE_COUNTS = {4: 5, 6: 7, 8: 8, 10: 8, 12: 8}
SPLIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "doa"
    / "evidence"
    / "cuda-4090"
    / "static-connected-floorplan-scaling"
    / "split.csv"
)


def run_stratified_beamforming_benchmark(
    output_dir: str | Path,
    *,
    room_counts: Sequence[int] = DEFAULT_ROOM_COUNTS,
    plans_per_room_count: int = 5,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    scenarios: Sequence[str] = ("same_room", "cross_room"),
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
    selected_algorithms = tuple(str(value).strip().lower() for value in algorithms)
    unknown = sorted(set(selected_algorithms) - set(DEFAULT_ALGORITHMS))
    if unknown:
        raise ValueError(f"unknown algorithms: {', '.join(unknown)}")
    selected_scenarios = tuple(str(value).strip().lower() for value in scenarios)
    if not selected_scenarios or set(selected_scenarios) - {"same_room", "cross_room"}:
        raise ValueError("scenarios may contain same_room and cross_room")
    split = _validation_split(selected_counts, int(plans_per_room_count))
    microphone_counts = _recommended_microphone_counts(selected_counts)
    signature = {
        "version": 2,
        "indices": [int(row["index"]) for row in split],
        "algorithms": list(selected_algorithms),
        "scenarios": list(selected_scenarios),
        "quality": quality,
        "duration_s": float(duration_s),
        "rir_duration_s": float(rir_duration_s),
        "interferer_snr_db": float(interferer_snr_db),
        "background_snr_db": float(background_snr_db),
        "sensor_noise_snr_db": float(sensor_noise_snr_db),
        "rt_accelerator": rt_accelerator,
        "rt_precision": rt_precision,
        "rt_cuda_device": int(rt_cuda_device),
        "seed": int(seed),
    }
    target_dry, interferer_dry = _study_audio(float(duration_s), FS)
    pink_noise = _tile_audio(load_wav_mono(RESOURCE_AUDIO / "pink_noise_bed.wav", FS), target_dry.size)
    rows, case_summaries = _load_checkpoint(output, signature)
    resumed_case_count = len(case_summaries)
    completed_cases = {
        (int(row["floorplan_idx"]), str(row["scenario"])) for row in case_summaries
    }
    started = time.perf_counter()

    for plan_number, split_row in enumerate(split, start=1):
        floorplan_idx = int(split_row["index"])
        room_count = int(split_row["room_count"])
        node_count = microphone_counts[room_count]
        missing_scenarios = [
            scenario
            for scenario in selected_scenarios
            if (floorplan_idx, scenario) not in completed_cases
        ]
        if not missing_scenarios:
            print(
                f"[{plan_number}/{len(split)}] floorplan={floorplan_idx} rooms={room_count} cached",
                flush=True,
            )
            continue
        model = load_model(floorplan_idx)
        nodes = place_nodes(
            model,
            node_count,
            mode="single",
            risk_quantile=0.2,
            candidates=candidate_nodes(model, positions_per_room=2),
        )
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
        noise_room = _farthest_room(model, source_room)
        noise_source = next(point for point in points if point.room_id == noise_room)
        plan_cache = output / "rir-cache"
        target_rirs = _distributed_rirs(
            plan_cache,
            model,
            target,
            nodes,
            quality=quality,
            rir_duration_s=rir_duration_s,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        noise_rirs = _distributed_rirs(
            plan_cache,
            model,
            noise_source,
            nodes,
            quality=quality,
            rir_duration_s=rir_duration_s,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        target_audio = _render_distributed(target_dry, target_rirs, target_dry.size)
        pink_audio = _render_distributed(pink_noise, noise_rirs, target_dry.size)

        for scenario in missing_scenarios:
            interferer = same_interferer if scenario == "same_room" else cross_interferer
            interferer_rirs = _distributed_rirs(
                plan_cache,
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
            evaluation = _benchmark_evaluation_slice(target_dry.size)
            interferer_audio, _ = scale_background_to_snr(
                target_audio,
                interferer_audio,
                interferer_snr_db,
                sample_slice=evaluation,
            )
            scaled_pink, _ = scale_background_to_snr(
                target_audio,
                pink_audio,
                background_snr_db,
                sample_slice=evaluation,
            )
            sensor_noise = _sensor_noise(target_audio.shape, seed + floorplan_idx * 11 + len(rows))
            sensor_noise, _ = scale_background_to_snr(
                target_audio,
                sensor_noise,
                sensor_noise_snr_db,
                sample_slice=evaluation,
            )
            noise_audio = np.asarray(interferer_audio + scaled_pink + sensor_noise, dtype=np.float32)
            mixture = np.asarray(target_audio + noise_audio, dtype=np.float32)
            calibration = _calibration_slice(target_dry.size)
            target_calibration = target_audio[:, calibration] + sensor_noise[:, calibration]
            noise_calibration = noise_audio[:, calibration]
            target_level = np.sqrt(
                np.mean(np.square(target_calibration, dtype=np.float64), axis=1)
            )
            noise_level = np.sqrt(
                np.mean(np.square(noise_calibration, dtype=np.float64), axis=1)
            )
            reliability = target_level / np.maximum(noise_level, 1e-12)
            reference_channel = int(np.argmax(reliability))
            relative_delays, confidence = estimate_gcc_phat_delays(
                target_calibration,
                FS,
                max_delay_s=0.2,
                reference_channel=reference_channel,
            )
            measurements = [
                TOAMeasurement(node.id, float(relative_delays[index]), float(confidence[index]))
                for index, node in enumerate(nodes)
            ]
            estimated_xy, estimated_room, _ = localize_tdoa(
                model,
                nodes,
                measurements,
                localization_grid(model, spacing_m=0.45),
            )
            localization_error = float(
                np.linalg.norm(np.asarray(estimated_xy) - np.asarray(target.position[:2]))
            )
            input_target = target_audio[reference_channel]
            input_noise = noise_audio[reference_channel]
            input_mixture = mixture[reference_channel]
            input_metrics = _quality_metrics(
                input_mixture,
                input_target,
                input_noise,
                target_dry,
                input_target,
                evaluation,
            )
            outputs: dict[str, np.ndarray] = {"best_single": input_mixture}
            case_rows: list[dict[str, Any]] = []
            for algorithm in selected_algorithms:
                if algorithm == "best_single":
                    output_target = input_target
                    output_noise = input_noise
                    output_mixture = input_mixture
                    processing_s = 0.0
                else:
                    processing_started = time.perf_counter()
                    weights = estimate_adaptive_beamformer_weights(
                        target_calibration,
                        noise_calibration,
                        relative_delays,
                        FS,
                        algorithm=algorithm,
                        reference_channel=reference_channel,
                        reliability=reliability,
                    )
                    output_mixture = apply_stft_beamformer(mixture, weights)
                    processing_s = time.perf_counter() - processing_started
                    output_target = apply_stft_beamformer(target_audio, weights)
                    output_noise = apply_stft_beamformer(noise_audio, weights)
                    outputs[algorithm] = output_mixture
                output_metrics = _quality_metrics(
                    output_mixture,
                    output_target,
                    output_noise,
                    target_dry,
                    input_target,
                    evaluation,
                )
                evaluation_duration_s = (evaluation.stop - evaluation.start) / FS
                case_rows.append(
                    {
                        "floorplan_idx": floorplan_idx,
                        "room_count": room_count,
                        "area_m2": round(float(split_row["area_m2"]), 4),
                        "scenario": scenario,
                        "algorithm": algorithm,
                        "microphones": len(nodes),
                        "target_room": target.room_id,
                        "interferer_room": interferer.room_id,
                        "noise_room": noise_source.room_id,
                        "estimated_room": estimated_room,
                        "room_correct": bool(estimated_room == target.room_id),
                        "localization_error_m": round(localization_error, 4),
                        "reference_microphone": nodes[reference_channel].id,
                        "input_snr_db": round(input_metrics["snr_db"], 4),
                        "output_snr_db": round(output_metrics["snr_db"], 4),
                        "snr_improvement_db": round(output_metrics["snr_db"] - input_metrics["snr_db"], 4),
                        "input_si_sdr_db": round(input_metrics["si_sdr_db"], 4),
                        "output_si_sdr_db": round(output_metrics["si_sdr_db"], 4),
                        "si_sdr_improvement_db": round(
                            output_metrics["si_sdr_db"] - input_metrics["si_sdr_db"], 4
                        ),
                        "input_stoi": _round_optional(input_metrics["stoi"]),
                        "output_stoi": _round_optional(output_metrics["stoi"]),
                        "stoi_improvement": _difference_optional(output_metrics["stoi"], input_metrics["stoi"]),
                        "input_pesq": _round_optional(input_metrics["pesq"]),
                        "output_pesq": _round_optional(output_metrics["pesq"]),
                        "pesq_improvement": _difference_optional(output_metrics["pesq"], input_metrics["pesq"]),
                        "input_dry_si_sdr_db": round(float(input_metrics["dry_si_sdr_db"]), 4),
                        "output_dry_si_sdr_db": round(float(output_metrics["dry_si_sdr_db"]), 4),
                        "input_dry_stoi": _round_optional(input_metrics["dry_stoi"]),
                        "output_dry_stoi": _round_optional(output_metrics["dry_stoi"]),
                        "input_dry_pesq": _round_optional(input_metrics["dry_pesq"]),
                        "output_dry_pesq": _round_optional(output_metrics["dry_pesq"]),
                        "processing_s": round(processing_s, 6),
                        "rtf": round(processing_s / max(evaluation_duration_s, 1e-9), 6),
                    }
                )
            rows.extend(case_rows)
            best_row = max(case_rows, key=lambda row: float(row["snr_improvement_db"]))
            case_summaries.append(
                {
                    "floorplan_idx": floorplan_idx,
                    "room_count": room_count,
                    "scenario": scenario,
                    "localization_error_m": round(localization_error, 4),
                    "room_correct": bool(estimated_room == target.room_id),
                    "best_snr_algorithm": best_row["algorithm"],
                    "best_snr_improvement_db": best_row["snr_improvement_db"],
                }
            )
            completed_cases.add((floorplan_idx, scenario))
            case_dir = output / "audio" / f"rooms-{room_count}" / f"floorplan-{floorplan_idx}" / scenario
            write_wav_mono(case_dir / "mixture-best-single.wav", input_mixture, FS)
            if best_row["algorithm"] in outputs:
                write_wav_mono(
                    case_dir / f"enhanced-{best_row['algorithm']}.wav",
                    outputs[str(best_row["algorithm"])],
                    FS,
                )
        _write_checkpoint(output, rows, case_summaries, signature)
        print(
            f"[{plan_number}/{len(split)}] floorplan={floorplan_idx} rooms={room_count} "
            f"nodes={len(nodes)} rows={len(rows)}",
            flush=True,
        )

    aggregate = _aggregate(rows, ("room_count", "scenario", "algorithm"))
    scenario_algorithm = _aggregate(rows, ("scenario", "algorithm"))
    room_algorithm = _aggregate(rows, ("room_count", "algorithm"))
    overall = _aggregate(rows, ("algorithm",))
    localization = _localization_aggregate(case_summaries)
    payload = {
        "study": "stratified_distributed_beamforming_benchmark_v1",
        "room_counts": list(selected_counts),
        "plans_per_room_count": int(plans_per_room_count),
        "floorplan_count": len(split),
        "case_count": len(case_summaries),
        "algorithms": list(selected_algorithms),
        "scenarios": list(selected_scenarios),
        "microphones_by_room_count": microphone_counts,
        "quality": quality,
        "duration_s": float(duration_s),
        "rir_duration_s": float(rir_duration_s),
        "interferer_snr_db": float(interferer_snr_db),
        "background_snr_db": float(background_snr_db),
        "sensor_noise_snr_db": float(sensor_noise_snr_db),
        "rt_accelerator": rt_accelerator,
        "rt_precision": rt_precision,
        "rt_cuda_device": int(rt_cuda_device),
        "invocation_elapsed_s": round(time.perf_counter() - started, 4),
        "resumed_case_count": resumed_case_count,
        "split": split,
        "cases": case_summaries,
        "localization": localization,
        "overall": overall,
        "scenario_algorithm": scenario_algorithm,
        "room_algorithm": room_algorithm,
        "aggregate": aggregate,
        "results": rows,
    }
    _write_benchmark(output, payload)
    return payload


def _validation_split(room_counts: Sequence[int], plans_per_room_count: int) -> list[dict[str, Any]]:
    if plans_per_room_count < 1:
        raise ValueError("plans_per_room_count must be positive")
    with SPLIT_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for room_count in room_counts:
        candidates = [
            row
            for row in rows
            if row["split"] == "validation" and int(row["room_count"]) == int(room_count)
        ]
        if len(candidates) < plans_per_room_count:
            raise ValueError(
                f"room_count={room_count} has {len(candidates)} validation FloorPlans; "
                f"{plans_per_room_count} requested"
            )
        selected.extend(candidates[:plans_per_room_count])
    return selected


def _recommended_microphone_counts(room_counts: Sequence[int]) -> dict[int, int]:
    missing = sorted(set(room_counts) - set(BENCHMARK_MICROPHONE_COUNTS))
    if missing:
        raise ValueError(f"no TDOA microphone recommendation for room counts: {missing}")
    return {
        int(room_count): BENCHMARK_MICROPHONE_COUNTS[int(room_count)]
        for room_count in room_counts
    }


def _farthest_room(model: Any, source_room: str) -> str:
    candidates = []
    for room_id in model.rooms:
        try:
            hops = len(model.route(source_room, room_id))
        except ValueError:
            continue
        candidates.append((hops, float(model.rooms[room_id]["area_m2"]), room_id))
    return max(candidates)[2]


def _quality_metrics(
    mixture: np.ndarray,
    target_component: np.ndarray,
    noise_component: np.ndarray,
    dry_reference: np.ndarray,
    enhancement_reference: np.ndarray,
    evaluation: slice,
) -> dict[str, float | None]:
    estimate = np.asarray(mixture)[evaluation]
    wet_reference = np.asarray(enhancement_reference)[evaluation]
    aligned_estimate, aligned_wet_reference, _ = align_signals(
        estimate,
        wet_reference,
        max_shift_samples=int(0.35 * FS),
    )
    dry = np.asarray(dry_reference)[evaluation]
    dry_estimate, aligned_dry_reference, _ = align_signals(
        estimate,
        dry,
        max_shift_samples=int(0.35 * FS),
    )
    return {
        "snr_db": snr_db(target_component, noise_component, sample_slice=evaluation),
        "si_sdr_db": aligned_si_sdr_db(
            estimate,
            wet_reference,
            max_shift_samples=int(0.35 * FS),
        ),
        "stoi": optional_stoi(aligned_estimate, aligned_wet_reference, FS),
        "pesq": optional_pesq(aligned_estimate, aligned_wet_reference, FS),
        "dry_si_sdr_db": aligned_si_sdr_db(
            estimate,
            dry,
            max_shift_samples=int(0.35 * FS),
        ),
        "dry_stoi": optional_stoi(dry_estimate, aligned_dry_reference, FS),
        "dry_pesq": optional_pesq(dry_estimate, aligned_dry_reference, FS),
    }


def _benchmark_evaluation_slice(sample_count: int) -> slice:
    return slice(min(int(round(0.15 * FS)), max(sample_count // 4, 1)), sample_count)


def _aggregate(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    metrics = (
        "input_snr_db",
        "input_si_sdr_db",
        "input_stoi",
        "input_pesq",
        "snr_improvement_db",
        "si_sdr_improvement_db",
        "stoi_improvement",
        "output_stoi",
        "pesq_improvement",
        "output_pesq",
        "rtf",
        "localization_error_m",
        "output_dry_si_sdr_db",
        "output_dry_stoi",
        "output_dry_pesq",
    )
    for group, items in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = {key: value for key, value in zip(keys, group)}
        result["cases"] = len(items)
        result["room_accuracy"] = round(float(np.mean([bool(item["room_correct"]) for item in items])), 5)
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            result[f"mean_{metric}"] = None if not values else round(float(np.mean(values)), 5)
            result[f"median_{metric}"] = None if not values else round(float(np.median(values)), 5)
        output.append(result)
    return output


def _localization_aggregate(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {"all": list(cases)}
    for room_count in sorted({int(case["room_count"]) for case in cases}):
        groups[str(room_count)] = [case for case in cases if int(case["room_count"]) == room_count]
    rows = []
    for label, items in groups.items():
        errors = np.asarray([float(item["localization_error_m"]) for item in items], dtype=np.float64)
        rows.append(
            {
                "room_count": label,
                "cases": len(items),
                "room_accuracy": round(float(np.mean([bool(item["room_correct"]) for item in items])), 5),
                "mean_error_m": round(float(np.mean(errors)), 5),
                "median_error_m": round(float(np.median(errors)), 5),
                "p90_error_m": round(float(np.quantile(errors, 0.9)), 5),
            }
        )
    return rows


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 5)


def _difference_optional(output: float | None, input_value: float | None) -> float | None:
    if output is None or input_value is None:
        return None
    return round(float(output - input_value), 5)


def _write_checkpoint(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    signature: Mapping[str, Any],
) -> None:
    (output / "checkpoint.json").write_text(
        json.dumps(
            {"signature": dict(signature), "results": list(rows), "cases": list(cases)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_checkpoint(
    output: Path,
    signature: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = output / "checkpoint.json"
    if not path.is_file():
        return [], []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []
    if payload.get("signature") != dict(signature):
        return [], []
    return list(payload.get("results", [])), list(payload.get("cases", []))


def _write_benchmark(output: Path, payload: Mapping[str, Any]) -> None:
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output / "details.csv", payload["results"])
    _write_csv(output / "aggregate.csv", payload["aggregate"])
    _write_csv(output / "scenario-algorithm.csv", payload["scenario_algorithm"])
    _write_csv(output / "room-algorithm.csv", payload["room_algorithm"])
    _write_csv(output / "overall.csv", payload["overall"])
    _write_csv(output / "localization.csv", payload["localization"])
    lines = [
        "# Distributed Beamforming Benchmark",
        "",
        f"- FloorPlans: `{payload['floorplan_count']}` ({payload['plans_per_room_count']} per room-count stratum)",
        f"- Room counts: `{', '.join(str(value) for value in payload['room_counts'])}`",
        f"- Algorithms: `{', '.join(payload['algorithms'])}`",
        f"- Scenarios: `{', '.join(payload['scenarios'])}`",
        f"- RIR quality: `{payload['quality']}`",
        f"- Invocation elapsed: `{payload['invocation_elapsed_s']:.2f} s` "
        f"(`{payload['resumed_case_count']}` cases resumed from checkpoint)",
        "",
        "## Overall",
        "",
        "| Algorithm | Cases | SNR improvement | SI-SDR improvement | PESQ | PESQ improvement | STOI | RTF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["overall"]:
        lines.append(
            f"| {row['algorithm']} | {row['cases']} | {_format_metric(row['mean_snr_improvement_db'])} | "
            f"{_format_metric(row['mean_si_sdr_improvement_db'])} | {_format_metric(row['mean_output_pesq'])} | "
            f"{_format_metric(row['mean_pesq_improvement'])} | {_format_metric(row['mean_output_stoi'])} | "
            f"{_format_metric(row['mean_rtf'], digits=4)} |"
        )
    lines.extend(
        [
            "",
            "## TDOA Localization",
            "",
            "| Rooms | Cases | Room accuracy | Mean error | Median error | P90 error |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["localization"]:
        lines.append(
            f"| {row['room_count']} | {row['cases']} | {100.0 * row['room_accuracy']:.1f}% | "
            f"{row['mean_error_m']:.3f} m | {row['median_error_m']:.3f} m | {row['p90_error_m']:.3f} m |"
        )
    lines.extend(
        [
            "",
            "## By Interference Topology",
            "",
            "| Scenario | Algorithm | Cases | SNR improvement | SI-SDR improvement | PESQ | STOI |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["scenario_algorithm"]:
        lines.append(
            f"| {row['scenario']} | {row['algorithm']} | {row['cases']} | "
            f"{_format_metric(row['mean_snr_improvement_db'])} | "
            f"{_format_metric(row['mean_si_sdr_improvement_db'])} | "
            f"{_format_metric(row['mean_output_pesq'])} | {_format_metric(row['mean_output_stoi'])} |"
        )
    lines.extend(
        [
            "",
            "## By Room Count",
            "",
            "| Rooms | Mics | Algorithm | Cases | SNR improvement | SI-SDR improvement | PESQ | STOI |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    microphones = {int(key): value for key, value in payload["microphones_by_room_count"].items()}
    for row in payload["room_algorithm"]:
        room_count = int(row["room_count"])
        lines.append(
            f"| {room_count} | {microphones[room_count]} | {row['algorithm']} | {row['cases']} | "
            f"{_format_metric(row['mean_snr_improvement_db'])} | "
            f"{_format_metric(row['mean_si_sdr_improvement_db'])} | "
            f"{_format_metric(row['mean_output_pesq'])} | {_format_metric(row['mean_output_stoi'])} |"
        )
    lines.extend(
        [
            "",
            "PESQ and STOI are `n/a` unless the optional research dependencies are installed.",
            "Primary SI-SDR, PESQ, and STOI use the fixed clean target image at the calibrated reference microphone. "
            "Dry-source-reference diagnostics remain available in details.csv. ",
            "DS, weighted DS, MVDR, and GEV use the same GCC-PHAT TDOA steering per case. "
            "MWF uses the same microphones, calibration observations, and reference channel without explicit steering.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: float | None, *, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"
