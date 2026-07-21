from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from acoustic_agent import AcousticAgent, microphone_array
from acoustic_agent.mic import channel_positions

from .estimators import (
    angular_error_deg,
    azimuth_deg,
    estimate_hrtf_template,
    estimate_srp_phat,
    linear_equivalent_azimuth_deg,
    listener_relative_azimuth_deg,
)


FS = 16_000
C = 343.0
RECEIVERS: dict[str, dict[str, Any]] = {
    "hrtf": microphone_array("hrtf", orientation_deg=0.0, interpolation="bilinear"),
    "linear4": microphone_array("linear", count=4, spacing_m=0.04, orientation_deg=0.0),
    "circular8": microphone_array("circular", count=8, radius_m=0.05, orientation_deg=0.0),
}


def run_los_study(
    output_dir: str | Path,
    *,
    scenes: Iterable[str] = ("geometry", "floorplan"),
    conditions: Iterable[str] = ("direct", "room"),
    quality: str = "preview",
    floorplan_idx: int = 0,
    geometry_angles_deg: Iterable[float] = (30.0, 105.0, 230.0),
    floorplan_seeds: Iterable[int] = (42, 43, 44),
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    probe = broadband_probe(FS, duration_s=0.55, seed=20260722)
    rows: list[dict[str, Any]] = []

    for scene_name in scenes:
        if scene_name == "geometry":
            placements = [_geometry_placement(angle) for angle in geometry_angles_deg]
        elif scene_name == "floorplan":
            placements = [_floorplan_placement(floorplan_idx, seed, quality) for seed in floorplan_seeds]
        else:
            raise ValueError("scenes may only contain geometry and floorplan")

        for placement in placements:
            for condition in conditions:
                for receiver_name, receiver_model in RECEIVERS.items():
                    agent = _agent_for_placement(placement, receiver_model, quality)
                    config = _condition_config(agent.config, condition)
                    result = agent.run(config=config)
                    visibility = float(result.metadata.get("steam_audio", {}).get("direct", {}).get("occlusion", 0.0))
                    if visibility < 0.999:
                        raise RuntimeError(f"LOS case became occluded: {placement['id']} ({visibility:.3f})")
                    observation = render_observation(np.asarray(result.rir), probe)
                    source = np.asarray(placement["source"], dtype=float)
                    receiver = np.asarray(placement["receiver"], dtype=float)
                    truth = azimuth_deg(receiver, source)
                    orientation = float(receiver_model.get("orientation_deg", 0.0))

                    if receiver_name == "hrtf":
                        search = np.arange(0.0, 360.0, 2.0)
                        estimate, spectrum = estimate_hrtf_template(
                            observation,
                            fs=FS,
                            search_deg=search,
                            orientation_deg=orientation,
                            interpolation=str(receiver_model.get("interpolation", "bilinear")),
                        )
                        evaluation_truth = truth
                        ambiguity = "none (generic HRTF template)"
                        method = "binaural_hrtf_template"
                    else:
                        positions = np.asarray(channel_positions(receiver, receiver_model), dtype=float)
                        if receiver_name.startswith("linear"):
                            search = np.arange(0.0, 181.0, 1.0)
                            evaluation_truth = linear_equivalent_azimuth_deg(truth, orientation)
                            ambiguity = "mirror across array axis"
                        else:
                            search = np.arange(0.0, 360.0, 1.0)
                            evaluation_truth = truth
                            ambiguity = "none"
                        estimate, spectrum = estimate_srp_phat(
                            observation,
                            positions,
                            fs=FS,
                            search_deg=search,
                            speed_of_sound_m_s=C,
                        )
                        method = "srp_phat"

                    stem = f"{placement['id']}-{condition}-{receiver_name}"
                    np.savez_compressed(
                        output / f"{stem}.npz",
                        rir=np.asarray(result.rir, dtype=np.float32),
                        received=observation,
                        spectrum=spectrum,
                        search_deg=search,
                        source_position_m=source,
                        receiver_position_m=receiver,
                    )
                    rows.append(
                        {
                            "scene": scene_name,
                            "case": placement["id"],
                            "condition": condition,
                            "receiver": receiver_name,
                            "method": method,
                            "true_world_azimuth_deg": round(truth, 4),
                            "true_listener_azimuth_deg": round(listener_relative_azimuth_deg(truth, orientation), 4),
                            "evaluation_azimuth_deg": round(evaluation_truth, 4),
                            "estimated_azimuth_deg": round(estimate, 4),
                            "absolute_error_deg": round(angular_error_deg(estimate, evaluation_truth), 4),
                            "ambiguity": ambiguity,
                            "distance_m": round(float(np.linalg.norm(source - receiver)), 4),
                            "direct_visibility": round(visibility, 4),
                            "floorplan_idx": placement.get("floorplan_idx"),
                            "seed": placement.get("seed"),
                            "artifact": f"{stem}.npz",
                        }
                    )

    _write_results(output, rows, quality)
    return rows


def broadband_probe(fs: int, duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = max(512, int(round(fs * duration_s)))
    signal = rng.normal(0.0, 1.0, count)
    fade = min(int(round(0.02 * fs)), count // 4)
    if fade:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, fade)) ** 2
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    signal /= max(float(np.max(np.abs(signal))), 1e-12)
    return signal.astype(np.float32)


def render_observation(rir: np.ndarray, probe: np.ndarray) -> np.ndarray:
    responses = np.asarray(rir, dtype=np.float64)
    if responses.ndim == 1:
        responses = responses.reshape(1, -1)
    length = responses.shape[1] + len(probe) - 1
    fft_size = 1 << max(1, int(length - 1).bit_length())
    source_spectrum = np.fft.rfft(np.asarray(probe, dtype=np.float64), fft_size)
    rendered = np.fft.irfft(np.fft.rfft(responses, fft_size, axis=1) * source_spectrum[None, :], fft_size, axis=1)
    return np.asarray(rendered[:, :length], dtype=np.float32)


def _geometry_placement(angle_deg: float) -> dict[str, Any]:
    receiver = np.asarray([3.0, 2.5, 1.4], dtype=float)
    radians = math.radians(float(angle_deg))
    source = receiver + 1.5 * np.asarray([math.cos(radians), math.sin(radians), 0.0])
    return {
        "id": f"geometry-{int(round(angle_deg)):03d}deg",
        "scene": "geometry",
        "room": {
            "shape": "rectangle",
            "size": [6.0, 5.0, 2.8],
            "material_profile": {"wall": "auto", "floor": "auto", "ceiling": "auto"},
            "material_seed": 2026,
        },
        "source": source.tolist(),
        "receiver": receiver.tolist(),
        "seed": 20260722 + int(round(angle_deg)),
    }


def _floorplan_placement(idx: int, seed: int, quality: str) -> dict[str, Any]:
    sample = AcousticAgent.create(
        scene="floorplan",
        idx=int(idx),
        placement="same_room",
        seed=int(seed),
        material_seed=2026,
        receiver_model="mono",
        source_model="omni",
        quality=quality,
        duration_s=1.2,
        fs=FS,
        visualization=False,
    )
    return {
        "id": f"floorplan-{int(idx)}-seed-{int(seed)}",
        "scene": "floorplan",
        "floorplan_idx": int(idx),
        "source": list(sample.default_source or ()),
        "receiver": list(sample.default_receiver or ()),
        "source_room": str(sample.placement["source_room"]),
        "receiver_room": str(sample.placement["receiver_room"]),
        "seed": int(seed),
    }


def _agent_for_placement(placement: Mapping[str, Any], receiver_model: Mapping[str, Any], quality: str) -> AcousticAgent:
    common = {
        "source": placement["source"],
        "receiver": placement["receiver"],
        "receiver_model": receiver_model,
        "source_model": "omni",
        "quality": quality,
        "duration_s": 1.2,
        "fs": FS,
        "visualization": False,
    }
    if placement["scene"] == "geometry":
        return AcousticAgent.create(scene="geometry", room=placement["room"], seed=placement["seed"], **common)
    return AcousticAgent.create(
        scene="floorplan",
        idx=placement["floorplan_idx"],
        placement="same_room",
        seed=placement["seed"],
        material_seed=2026,
        source_room=placement["source_room"],
        receiver_room=placement["receiver_room"],
        **common,
    )


def _condition_config(config: Any, condition: str) -> Any:
    if condition == "direct":
        return replace(
            config,
            duration_s=0.25,
            rt_duration_s=0.25,
            late_tail=False,
            reflections_enabled=False,
            diffraction_enabled=False,
            diffraction_audio_enabled=False,
            rt_num_rays=64,
            rt_num_bounces=1,
            collect_visual_paths=False,
        )
    if condition == "room":
        return replace(config, duration_s=1.2, rt_duration_s=1.2, collect_visual_paths=False)
    raise ValueError("conditions may only contain direct and room")


def _write_results(output: Path, rows: list[dict[str, Any]], quality: str) -> None:
    fields = list(rows[0]) if rows else []
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps({"configuration": {"fs": FS, "speed_of_sound_m_s": C, "quality": quality}, "results": rows}, indent=2),
        encoding="utf-8",
    )

    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["scene"]), str(row["condition"]), str(row["receiver"]))
        groups.setdefault(key, []).append(float(row["absolute_error_deg"]))
    lines = [
        "# LOS DOA Report",
        "",
        f"- Sample rate: {FS} Hz",
        f"- Speed of sound: {C:.1f} m/s",
        f"- Simulation quality: `{quality}`",
        "- Linear-array errors are evaluated against the mirror-equivalent half-plane bearing.",
        "- `direct` isolates the direct path; `room` retains reflections and late reverberation while LOS remains open.",
        "",
        "| Scene | Condition | Receiver | Cases | Mean error | Max error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, errors in sorted(groups.items()):
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(errors)} | {np.mean(errors):.2f} deg | {np.max(errors):.2f} deg |")
    lines.extend(["", "Detailed per-case values are stored in `summary.csv`; RIRs, observations, search grids, and spectra are stored in the NPZ artifacts.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
