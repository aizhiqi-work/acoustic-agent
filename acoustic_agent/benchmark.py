from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import __version__
from .acoustics import AIR_ABSORPTION_NP_PER_M, apply_surface_reflection, propagation_band_gains
from .api import AcousticAgent
from .engine import simulate_rir, solve_paths
from .models import FREQUENCY_BANDS, Material, Room, SimConfig
from .geometry import make_room
from .steam_rt import (
    _render_parametric_fdn_late_reverb,
)


STATUS_ORDER = {"fail": 0, "error": 1, "skip": 2, "pass": 3}


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    rays: int
    bounces: int
    duration_s: float
    steam_rays: int
    steam_bounces: int
    description: str


PROFILES: dict[str, BenchmarkProfile] = {
    "quick": BenchmarkProfile(
        name="quick",
        rays=8192,
        bounces=32,
        duration_s=1.2,
        steam_rays=8192,
        steam_bounces=32,
        description="Fast deterministic regression profile for local development and CI.",
    ),
    "full": BenchmarkProfile(
        name="full",
        rays=131072,
        bounces=96,
        duration_s=2.0,
        steam_rays=131072,
        steam_bounces=96,
        description="Reference profile for release evidence and native Steam Audio comparison.",
    ),
}


@dataclass
class BenchmarkMetric:
    name: str
    measured: float | int | str | bool | None
    expected: float | int | str | bool | None = None
    tolerance: float | str | None = None
    unit: str = ""
    passed: bool | None = None
    note: str = ""


@dataclass
class BenchmarkCaseResult:
    id: str
    name: str
    category: str
    status: str
    summary: str
    duration_s: float
    metrics: list[BenchmarkMetric] = field(default_factory=list)
    scene: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyBenchmarkReport:
    schema_version: int
    generated_at: str
    acoustic_agent_version: str
    profile: Mapping[str, Any]
    environment: Mapping[str, Any]
    cases: list[BenchmarkCaseResult]

    @property
    def summary(self) -> dict[str, Any]:
        counts = {status: sum(case.status == status for case in self.cases) for status in STATUS_ORDER}
        return {
            **counts,
            "total": len(self.cases),
            "required_passed": counts["fail"] == 0 and counts["error"] == 0,
            "duration_s": round(sum(case.duration_s for case in self.cases), 4),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "acoustic_agent_version": self.acoustic_agent_version,
            "profile": dict(self.profile),
            "environment": dict(self.environment),
            "summary": self.summary,
            "cases": [asdict(case) for case in self.cases],
        }


@dataclass(frozen=True)
class _BenchmarkContext:
    profile: BenchmarkProfile
    steam_audio_root: Path | None
    work_dir: Path


CaseRunner = Callable[[_BenchmarkContext], BenchmarkCaseResult]


def run_accuracy_benchmark(
    *,
    profile: str = "quick",
    output_dir: str | Path | None = None,
    steam_audio_root: str | Path | None = None,
    case_ids: Sequence[str] | None = None,
) -> AccuracyBenchmarkReport:
    """Run the fixed acoustic-accuracy suite and optionally write reports."""
    profile_key = str(profile).strip().lower()
    if profile_key not in PROFILES:
        raise ValueError(f"unknown benchmark profile {profile!r}; expected quick or full")
    destination = Path(output_dir or "benchmark-results").expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    context = _BenchmarkContext(
        profile=PROFILES[profile_key],
        steam_audio_root=_resolve_steam_audio_root(steam_audio_root),
        work_dir=destination / ".cache",
    )
    selected = set(str(value) for value in case_ids) if case_ids else None
    cases = []
    for case_id, runner in _CASE_RUNNERS:
        if selected is not None and case_id not in selected:
            continue
        started = time.perf_counter()
        try:
            result = runner(context)
        except Exception as exc:  # The report must survive an individual benchmark failure.
            result = BenchmarkCaseResult(
                id=case_id,
                name=case_id.replace("_", " ").title(),
                category="infrastructure",
                status="error",
                summary=f"{type(exc).__name__}: {exc}",
                duration_s=0.0,
                details={"exception_type": type(exc).__name__},
            )
        result.duration_s = round(time.perf_counter() - started, 4)
        cases.append(result)
    report = AccuracyBenchmarkReport(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        acoustic_agent_version=__version__,
        profile=asdict(context.profile),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "steam_audio_root": str(context.steam_audio_root) if context.steam_audio_root else None,
        },
        cases=cases,
    )
    if output_dir is not None:
        write_accuracy_report(report, destination)
    return report


def write_accuracy_report(report: AccuracyBenchmarkReport, output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / "accuracy-benchmark.json",
        "markdown": destination / "accuracy-benchmark.md",
        "html": destination / "accuracy-benchmark.html",
    }
    paths["json"].write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown_report(report), encoding="utf-8")
    paths["html"].write_text(_html_report(report), encoding="utf-8")
    return paths


def _direct_arrival(context: _BenchmarkContext) -> BenchmarkCaseResult:
    fs = 48000
    c = 343.0
    source = (1.0, 1.0, 1.4)
    receiver = (4.0, 1.0, 1.4)
    distance = math.dist(source, receiver)
    config = SimConfig(
        fs=fs,
        c=c,
        duration_s=0.08,
        min_distance_m=0.1,
        reflections_enabled=False,
        diffraction_enabled=False,
    )
    result = simulate_rir(make_room("rectangle", size=(6.0, 4.0, 2.8)), source, receiver, config=config)
    direct = next(path for path in result.paths if path.kind == "direct")
    measured_peak_s = int(np.argmax(np.abs(result.rir[0]))) / fs
    expected_s = distance / c
    path_error_samples = abs(direct.delay_s - expected_s) * fs
    peak_error_samples = abs(measured_peak_s - expected_s) * fs
    passed = path_error_samples <= 1e-8 and peak_error_samples <= 0.55
    return _case(
        "direct_arrival",
        "Direct sound arrival",
        "direct sound",
        passed,
        "Direct path time and rendered peak follow distance / c.",
        [
            _metric("path arrival", direct.delay_s, expected_s, 1e-8, "s", path_error_samples <= 1e-8),
            _metric("rendered peak error", peak_error_samples, 0.0, 0.55, "samples", peak_error_samples <= 0.55),
        ],
        scene={"shape": "rectangle", "size_m": [6.0, 4.0, 2.8], "source": source, "receiver": receiver, "c_m_per_s": c},
    )


def _distance_attenuation(context: _BenchmarkContext) -> BenchmarkCaseResult:
    room = make_room("rectangle", size=(10.0, 4.0, 2.8))
    config = SimConfig(
        fs=16000,
        duration_s=0.08,
        min_distance_m=0.1,
        reflections_enabled=False,
        diffraction_enabled=False,
    )
    source = (1.0, 2.0, 1.4)
    near = simulate_rir(room, source, (3.0, 2.0, 1.4), config=config)
    far = simulate_rir(room, source, (5.0, 2.0, 1.4), config=config)
    near_direct = near.metadata["steam_audio"]["direct"]
    far_direct = far.metadata["steam_audio"]["direct"]
    ratio = float(near_direct["band_gains"]["125"]) / max(float(far_direct["band_gains"]["125"]), 1e-20)
    delta_db = 20.0 * math.log10(ratio)
    expected_db = 20.0 * math.log10(2.0)
    error_db = abs(delta_db - expected_db)
    passed = error_db <= 0.05
    return _case(
        "distance_attenuation",
        "Distance doubling attenuation",
        "direct sound",
        passed,
        "The non-air-absorbing 125 Hz band follows inverse-distance pressure decay.",
        [_metric("2 m to 4 m level change", delta_db, expected_db, 0.05, "dB", passed)],
        scene={"distances_m": [2.0, 4.0], "band_hz": 125, "min_distance_m": 0.1},
        details={"pressure_ratio": ratio},
    )


def _shoebox_rt60(context: _BenchmarkContext) -> BenchmarkCaseResult:
    alpha = 0.25
    material = _uniform_material(alpha)
    size = (6.0, 4.0, 2.8)
    room = make_room("rectangle", size=size, materials={"wall": material, "floor": material, "ceiling": material})
    config = _reflection_config(context, adaptive=False)
    result = simulate_rir(room, (1.3, 1.1, 1.4), (4.7, 2.9, 1.4), config=config)
    volume = size[0] * size[1] * size[2]
    surface = 2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2])
    sabine = 0.161 * volume / (surface * alpha)
    eyring = 0.161 * volume / (-surface * math.log(1.0 - alpha))
    traced = np.asarray([float(result.rt60["traced_rt60_bands"][band]) for band in FREQUENCY_BANDS])
    valid = traced[traced > 0.0]
    measured = float(np.median(valid)) if valid.size else 0.0
    eyring_error = abs(measured - eyring) / eyring
    tolerance = 0.15 if context.profile.name == "quick" else 0.10
    passed = valid.size == len(FREQUENCY_BANDS) and eyring_error <= tolerance
    return _case(
        "shoebox_rt60",
        "Shoebox RT60",
        "reverberation",
        passed,
        "Path-traced octave-band decay is compared with Sabine and Eyring references.",
        [
            _metric("median traced RT60", measured, eyring, eyring * tolerance, "s", passed, "Absolute equivalent of the relative Eyring tolerance."),
            _metric("relative Eyring error", eyring_error * 100.0, 0.0, tolerance * 100.0, "%", passed),
            _metric("Sabine reference", sabine, sabine, None, "s", None),
        ],
        scene={"shape": "shoebox", "size_m": size, "uniform_absorption": alpha},
        details={"traced_rt60_bands_s": dict(result.rt60["traced_rt60_bands"]), "eyring_rt60_s": eyring, "sabine_rt60_s": sabine},
    )


def _early_reflections(context: _BenchmarkContext) -> BenchmarkCaseResult:
    room = make_room("rectangle", size=(6.0, 4.0, 2.8))
    source = np.asarray((1.2, 1.1, 1.3), dtype=float)
    receiver = np.asarray((4.4, 2.8, 1.5), dtype=float)
    config = replace(
        _reflection_config(context, adaptive=False),
        duration_s=0.5,
        rt_duration_s=0.5,
        rt_num_bounces=min(8, context.profile.bounces),
        late_tail=False,
        diffraction_enabled=False,
    )
    paths = solve_paths(room, source, receiver, config)
    expected = _shoebox_first_order_paths(source, receiver, (6.0, 4.0, 2.8), room)
    measured_rows = []
    distance_errors = []
    amplitude_law_errors = []
    missing = []
    for item in expected:
        candidates = [
            path for path in paths
            if path.kind == "rt_reflection"
            and int(path.metadata.get("order", -1)) == 1
            and item["surface"] in path.metadata.get("surfaces", [])
        ]
        if not candidates:
            missing.append(item["surface"])
            continue
        match = min(candidates, key=lambda path: abs(path.distance_m - item["distance_m"]))
        distance_error = abs(match.distance_m - item["distance_m"])
        distance_errors.append(distance_error)
        # The visual representative gain is a Monte Carlo contribution, not a discrete ISM pressure.
        # Verify the actual surface pressure law independently and report the sampled contribution separately.
        expected_pressure = item["pressure_gain"]
        recomputed_pressure = apply_surface_reflection(
            propagation_band_gains(item["distance_m"], min_distance_m=0.1),
            room.materials[item["material_semantic"]],
        )["1000"]
        amplitude_law_errors.append(abs(recomputed_pressure - expected_pressure))
        measured_rows.append({
            **item,
            "measured_distance_m": match.distance_m,
            "distance_error_m": distance_error,
            "measured_delay_s": match.delay_s,
            "sampled_mc_gain": match.gain,
            "order": int(match.metadata.get("order", -1)),
        })
    max_distance_error = max(distance_errors, default=math.inf)
    max_amplitude_error = max(amplitude_law_errors, default=math.inf)
    distance_tolerance = float(config.rt_receiver_radius_m) + 0.02
    passed = not missing and max_distance_error <= distance_tolerance and max_amplitude_error <= 1e-12
    return _case(
        "early_reflections",
        "First-order early reflections",
        "early reflections",
        passed,
        "Six shoebox image-source paths are checked for distance, arrival time, order, and surface pressure law.",
        [
            _metric("matched surfaces", len(measured_rows), 6, 0, "count", not missing),
            _metric("maximum distance error", max_distance_error, 0.0, distance_tolerance, "m", max_distance_error <= distance_tolerance),
            _metric("maximum pressure-law error", max_amplitude_error, 0.0, 1e-12, "linear", max_amplitude_error <= 1e-12),
        ],
        scene={"shape": "shoebox", "size_m": [6.0, 4.0, 2.8], "source": source.tolist(), "receiver": receiver.tolist()},
        details={
            "paths": measured_rows,
            "missing_surfaces": missing,
            "amplitude_scope": "Analytic first-order pressure law; Monte Carlo representative gains are diagnostic, not ISM impulses.",
        },
    )


def _fdn_isolation(context: _BenchmarkContext) -> BenchmarkCaseResult:
    fs = 16000
    length = int(1.2 * fs)
    target_rt60 = 0.9
    rng = np.random.default_rng(81)
    times = np.arange(length, dtype=np.float64) / fs
    envelope = np.exp(-3.0 * math.log(10.0) * times / target_rt60)
    traced = np.vstack([rng.uniform(-1.0, 1.0, length) * envelope for _ in FREQUENCY_BANDS])
    base = SimConfig(fs=fs, duration_s=1.2, hybrid_transition_s=0.45, hybrid_overlap_fraction=0.25)
    enabled, _, metadata = _render_parametric_fdn_late_reverb(
        traced, {"bin_duration_s": 0.01}, {band: target_rt60 for band in FREQUENCY_BANDS}, base
    )
    disabled, _, _ = _render_parametric_fdn_late_reverb(
        traced, {"bin_duration_s": 0.01}, {band: target_rt60 for band in FREQUENCY_BANDS}, replace(base, late_tail=False)
    )
    start = int(float(metadata["transition_start_s"]) * fs)
    pre_error = float(np.max(np.abs(enabled[:, :start] - disabled[:, :start]))) if start else 0.0
    post_delta = float(np.sum((enabled[:, start:] - disabled[:, start:]) ** 2))
    passed = bool(metadata.get("applied")) and pre_error <= 1e-7 and post_delta > 1e-6
    return _case(
        "fdn_isolation",
        "FDN early/late isolation",
        "late reverberation",
        passed,
        "FDN on/off leaves the pre-transition response unchanged and changes the late tail.",
        [
            _metric("pre-transition maximum delta", pre_error, 0.0, 1e-7, "linear", pre_error <= 1e-7),
            _metric("post-transition delta energy", post_delta, "> 1e-6", None, "energy", post_delta > 1e-6),
        ],
        scene={"synthetic_traced_rt60_s": target_rt60, "transition_s": base.hybrid_transition_s, "fs": fs},
        details={"fdn_metadata": metadata},
    )


def _portal_coupling(context: _BenchmarkContext) -> BenchmarkCaseResult:
    spec = _two_room_spec()
    source = (2.0, 2.0, 1.4)
    receiver = (6.0, 2.0, 1.4)
    config = replace(_reflection_config(context, adaptive=False), diffraction_enabled=False)
    agent = AcousticAgent.from_floorplan_spec(
        spec,
        source_room="living_0",
        receiver_room="bedroom_0",
        source=source,
        receiver=receiver,
        config=config,
    )
    open_room = agent.room
    open_result = simulate_rir(open_room, source, receiver, config=config)
    closed_room = _closed_portal_room(open_room)
    closed_result = simulate_rir(closed_room, source, receiver, config=config)
    open_energy = float(np.sum(open_result.rir[0].astype(np.float64) ** 2))
    closed_energy = float(np.sum(closed_result.rir[0].astype(np.float64) ** 2))
    energy_delta_db = 10.0 * math.log10(max(open_energy, 1e-30) / max(closed_energy, 1e-30))
    open_profile = dict(open_result.rt60.get("decay_profile", {}))
    closed_profile = dict(closed_result.rt60.get("decay_profile", {}))
    open_direct = open_result.metadata["steam_audio"]["direct"]
    closed_direct = closed_result.metadata["steam_audio"]["direct"]
    open_portal_paths = sum(path.kind == "portal_path" for path in open_result.paths)
    closed_portal_paths = sum(path.kind == "portal_path" for path in closed_result.paths)
    # A source aligned with the opening uses the unobstructed direct path. An
    # off-axis source would use portal pathing instead; both are valid open-door routes.
    open_route_count = open_portal_paths + int(float(open_direct["occlusion"]) >= 1.0)
    closed_route_count = closed_portal_paths + int(float(closed_direct["occlusion"]) >= 1.0)
    topology_pass = open_route_count >= 1 and closed_route_count == 0
    energy_pass = energy_delta_db >= 3.0
    slope_changed = _decay_signature(open_profile) != _decay_signature(closed_profile)
    passed = topology_pass and energy_pass and slope_changed
    return _case(
        "portal_coupling",
        "Open-door coupled rooms",
        "coupled spaces",
        passed,
        "Opening the interior portal must add a verified path, raise cross-room energy, and change decay behavior.",
        [
            _metric("open/closed energy change", energy_delta_db, ">= 3", None, "dB", energy_pass),
            _metric("open acoustic route count", open_route_count, ">= 1", None, "count", open_route_count >= 1),
            _metric("closed clear-route count", closed_route_count, 0, 0, "count", closed_route_count == 0),
            _metric("decay signature changed", slope_changed, True, None, "", slope_changed),
        ],
        scene={"rooms": 2, "portal_width_m": 1.0, "source_room": "living_0", "receiver_room": "bedroom_0"},
        details={
            "open_decay": open_profile,
            "closed_decay": closed_profile,
            "open_energy": open_energy,
            "closed_energy": closed_energy,
            "open_direct": open_direct,
            "closed_direct": closed_direct,
            "open_portal_path_count": open_portal_paths,
            "closed_portal_path_count": closed_portal_paths,
        },
    )


def _hrtf_consistency(context: _BenchmarkContext) -> BenchmarkCaseResult:
    room = make_room("rectangle", size=(6.0, 4.0, 2.8))
    source = (4.0, 2.0, 1.4)
    receiver = (2.0, 2.0, 1.4)
    config = SimConfig(
        fs=16000,
        duration_s=0.15,
        min_distance_m=0.1,
        reflections_enabled=False,
        diffraction_enabled=False,
    )
    mono = simulate_rir(room, source, receiver, config=config)
    binaural = simulate_rir(
        room,
        source,
        receiver,
        config=config,
        receiver_model={"type": "hrtf", "interpolation": "nearest", "loudness_normalization": "energy"},
    )
    left, right = (np.asarray(channel, dtype=np.float64) for channel in binaural.rir)
    left_peak = int(np.argmax(np.abs(left)))
    right_peak = int(np.argmax(np.abs(right)))
    itd_s = (right_peak - left_peak) / config.fs
    left_energy = float(np.sum(left * left))
    right_energy = float(np.sum(right * right))
    ild_db = 10.0 * math.log10(max(left_energy, 1e-30) / max(right_energy, 1e-30))
    mono_energy = float(np.sum(np.asarray(mono.rir[0], dtype=np.float64) ** 2))
    loudness_db = 10.0 * math.log10(max(0.5 * (left_energy + right_energy), 1e-30) / max(mono_energy, 1e-30))
    itd_pass = 0.0001 <= abs(itd_s) <= 0.001
    ild_pass = 1.0 <= abs(ild_db) <= 30.0
    loudness_pass = abs(loudness_db) <= 0.25
    passed = itd_pass and ild_pass and loudness_pass
    return _case(
        "hrtf_consistency",
        "HRTF spatial and loudness consistency",
        "binaural rendering",
        passed,
        "A lateral direct source must produce plausible ITD/ILD while energy normalization preserves loudness.",
        [
            _metric("ITD magnitude", abs(itd_s) * 1000.0, "[0.1, 1.0]", None, "ms", itd_pass),
            _metric("ILD magnitude", abs(ild_db), "[1, 30]", None, "dB", ild_pass),
            _metric("binaural loudness delta", loudness_db, 0.0, 0.25, "dB", loudness_pass),
        ],
        scene={"source_azimuth_deg": 90, "distance_m": 2.0, "sofa": "cipic_124"},
        details={"signed_itd_s": itd_s, "signed_ild_db_left_over_right": ild_db, "render_metadata": binaural.receiver_model.get("render_metadata", {})},
    )


def _dynamic_continuity(context: _BenchmarkContext) -> BenchmarkCaseResult:
    config = SimConfig(
        fs=16000,
        duration_s=0.15,
        min_distance_m=0.1,
        reflections_enabled=False,
        diffraction_enabled=False,
    )
    agent = AcousticAgent(room=(6.0, 4.0, 2.8), config=config)
    motion = agent.sample_motion(
        source=(1.0, 2.0, 1.4),
        receiver=(4.5, 2.0, 1.4),
        mode="approach",
        moving="receiver",
        distance_m=1.0,
        keyframe_spacing_m=0.1,
    )
    result = agent.run_dynamic(motion)
    peaks = np.asarray([float(np.max(np.abs(frame.rir))) for frame in result.frames])
    peak_db = 20.0 * np.log10(np.maximum(peaks, 1e-30))
    jumps = np.abs(np.diff(peak_db))
    direct_delays = np.asarray([
        next(path.delay_s for path in frame.paths if path.kind in {"direct", "direct_transmitted"})
        for frame in result.frames
    ])
    delay_step_jitter = float(np.ptp(np.diff(direct_delays))) if direct_delays.size > 2 else 0.0
    max_peak_jump = float(np.max(jumps)) if jumps.size else 0.0
    peak_pass = max_peak_jump <= 1.5
    delay_pass = delay_step_jitter <= 1e-9
    passed = peak_pass and delay_pass
    return _case(
        "dynamic_continuity",
        "Adjacent dynamic-frame continuity",
        "dynamic RIR",
        passed,
        "Uniform 0.1 m motion is checked for direct-delay continuity and raw RIR peak jumps.",
        [
            _metric("maximum adjacent peak jump", max_peak_jump, 0.0, 1.5, "dB", peak_pass),
            _metric("delay-step jitter", delay_step_jitter, 0.0, 1e-9, "s", delay_pass),
        ],
        scene={"motion": "approach", "travel_m": 1.0, "spacing_m": 0.1, "frames": len(result.frames)},
        details={"peak_db": peak_db.tolist(), "adjacent_peak_jumps_db": jumps.tolist(), "direct_delays_s": direct_delays.tolist()},
    )


def _steam_audio_native(context: _BenchmarkContext) -> BenchmarkCaseResult:
    if context.steam_audio_root is None:
        return BenchmarkCaseResult(
            id="steam_audio_native",
            name="Native Steam Audio same-scene comparison",
            category="external reference",
            status="skip",
            summary="Steam Audio SDK root was not found; pass --steam-audio-root to enable this reference.",
            duration_s=0.0,
        )
    native = _run_steam_audio_reference(context)
    alpha = float(native["absorption"])
    material = _uniform_material(alpha)
    size = tuple(float(value) for value in native["size_m"])
    room = make_room("rectangle", size=size, materials={"wall": material, "floor": material, "ceiling": material})
    config = SimConfig(
        fs=int(native["sample_rate"]),
        duration_s=float(native["duration_s"]),
        rt_duration_s=float(native["duration_s"]),
        rt_num_rays=context.profile.steam_rays,
        rt_num_bounces=context.profile.steam_bounces,
        late_tail=False,
        diffraction_enabled=False,
        adaptive_geometry_bounces=False,
        seed=1729,
    )
    result = simulate_rir(room, tuple(native["source_m"]), tuple(native["listener_m"]), config=config)
    native_values = np.asarray(native["reverb_times_s"], dtype=float)
    if int(native.get("band_count", native_values.size)) == 11:
        steam = native_values[3:9]
        compared_bands = list(FREQUENCY_BANDS)
    else:
        steam = native_values[:3]
        compared_bands = ["500", "2000", "4000"]
    acoustic = np.asarray([float(result.rt60["traced_rt60_bands"][band]) for band in compared_bands])
    relative = np.abs(acoustic - steam) / np.maximum(steam, 1e-9)
    max_error = float(np.max(relative))
    tolerance = 0.25 if context.profile.name == "quick" else 0.15
    passed = bool(np.all(steam > 0.0)) and max_error <= tolerance
    return _case(
        "steam_audio_native",
        "Native Steam Audio same-scene comparison",
        "external reference",
        passed,
        "The same uniform-material shoebox is traced by Acoustic Agent and the native Steam Audio SDK.",
        [
            _metric("maximum matched-band RT60 error", max_error * 100.0, 0.0, tolerance * 100.0, "%", passed),
        ],
        scene={"shape": "shoebox", "size_m": size, "uniform_absorption": alpha, "rays": context.profile.steam_rays, "bounces": context.profile.steam_bounces},
        details={
            "steam_audio_rt60_s": steam.tolist(),
            "acoustic_agent_rt60_s": acoustic.tolist(),
            "compared_bands_hz": compared_bands,
            "relative_errors": relative.tolist(),
            "native": native,
        },
    )


_CASE_RUNNERS: tuple[tuple[str, CaseRunner], ...] = (
    ("direct_arrival", _direct_arrival),
    ("distance_attenuation", _distance_attenuation),
    ("shoebox_rt60", _shoebox_rt60),
    ("early_reflections", _early_reflections),
    ("fdn_isolation", _fdn_isolation),
    ("portal_coupling", _portal_coupling),
    ("hrtf_consistency", _hrtf_consistency),
    ("dynamic_continuity", _dynamic_continuity),
    ("steam_audio_native", _steam_audio_native),
)


def _reflection_config(context: _BenchmarkContext, *, adaptive: bool) -> SimConfig:
    return SimConfig(
        fs=16000,
        duration_s=context.profile.duration_s,
        rt_duration_s=context.profile.duration_s,
        rt_num_rays=context.profile.rays,
        rt_num_bounces=context.profile.bounces,
        rt_visual_num_rays=None,
        rt_visual_num_bounces=None,
        adaptive_geometry_bounces=adaptive,
        adaptive_cross_room_bounces=adaptive,
        late_tail=True,
        seed=1729,
    )


def _uniform_material(alpha: float) -> Material:
    return Material(
        id=f"benchmark_uniform_{alpha:.3f}",
        name=f"Uniform alpha={alpha:.3f}",
        semantic="benchmark_surface",
        absorption={band: float(alpha) for band in FREQUENCY_BANDS},
        scattering={band: 0.05 for band in FREQUENCY_BANDS},
        transmission_loss_db={band: 40.0 for band in FREQUENCY_BANDS},
        source="benchmark",
    )


def _shoebox_first_order_paths(source: np.ndarray, receiver: np.ndarray, size: Sequence[float], room: Room) -> list[dict[str, Any]]:
    width, depth, height = (float(value) for value in size)
    images = (
        ("wall_0", np.asarray((source[0], -source[1], source[2])), "wall"),
        ("wall_1", np.asarray((2.0 * width - source[0], source[1], source[2])), "wall"),
        ("wall_2", np.asarray((source[0], 2.0 * depth - source[1], source[2])), "wall"),
        ("wall_3", np.asarray((-source[0], source[1], source[2])), "wall"),
        ("floor", np.asarray((source[0], source[1], -source[2])), "floor"),
        ("ceiling", np.asarray((source[0], source[1], 2.0 * height - source[2])), "ceiling"),
    )
    output = []
    for surface, image_source, semantic in images:
        distance = float(np.linalg.norm(image_source - receiver))
        reflection = float(room.materials[semantic].reflection["1000"])
        output.append({
            "surface": surface,
            "distance_m": distance,
            "delay_s": distance / 343.0,
            "order": 1,
            "reflection_coefficient": reflection,
            "material_semantic": semantic,
            "pressure_gain": reflection / distance * math.exp(-AIR_ABSORPTION_NP_PER_M["1000"] * distance),
        })
    return output


def _two_room_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "title": "Benchmark two-room portal",
        "units": "m",
        "coordinate_system": "image_top_left",
        "height_m": 2.8,
        "wall_depth_m": 0.12,
        "outer_boundary": [[0, 0], [8, 0], [8, 4], [0, 4]],
        "rooms": [
            {"id": "living_0", "type": "living", "corners": [[0, 0], [4, 0], [4, 4], [0, 4]]},
            {"id": "bedroom_0", "type": "bedroom", "corners": [[4, 0], [8, 0], [8, 4], [4, 4]]},
        ],
        "openings": [{
            "id": "door_0",
            "type": "door",
            "room_ids": ["living_0", "bedroom_0"],
            "segment": [[4, 1.5], [4, 2.5]],
            "height_m": 2.1,
            "sill_height_m": 0.0,
            "connection": "interior_room",
            "open": True,
            "confidence": 1.0,
        }],
        "provenance": {"source": "benchmark"},
    }


def _closed_portal_room(room: Room) -> Room:
    metadata = deepcopy(dict(room.metadata))
    multi_room = deepcopy(dict(metadata.get("multi_room", {})))
    for portal in multi_room.get("portals", []):
        portal["open"] = False
    multi_room["route_room_ids"] = []
    multi_room["route_portal_ids"] = []
    multi_room["door_state"] = "benchmark_closed"
    metadata["multi_room"] = multi_room
    objects = list(metadata.get("objects", []))
    objects.append({
        "id": "benchmark_closed_door",
        "type": "cuboid",
        "semantic": "door",
        "position": [4.0, 2.0],
        "size": [0.16, 1.0, 2.1],
        "z": 1.05,
        "rotation": 0.0,
    })
    metadata["objects"] = objects
    return replace(room, metadata=metadata)


def _decay_signature(profile: Mapping[str, Any]) -> tuple[Any, ...]:
    segments = profile.get("segments", []) if isinstance(profile, Mapping) else []
    values = tuple(round(float(item.get("equivalent_rt60_s", 0.0)), 2) for item in segments if isinstance(item, Mapping))
    return (str(profile.get("model", "unknown")), values)


def _resolve_steam_audio_root(value: str | Path | None) -> Path | None:
    candidates = []
    if value:
        candidates.append(Path(value))
    if os.environ.get("STEAM_AUDIO_ROOT"):
        candidates.append(Path(os.environ["STEAM_AUDIO_ROOT"]))
    candidates.extend((Path.cwd().parent / "steam-audio", Path(__file__).resolve().parents[2] / "steam-audio"))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if _steam_sdk_paths(resolved) is not None:
            return resolved
    return None


def _steam_sdk_paths(root: Path) -> tuple[Path, Path] | None:
    candidates = (
        (root / "unity" / "include" / "phonon", root / "unity" / "lib" / "osx" / "libphonon.dylib"),
        (root / "fmod" / "include" / "phonon", root / "fmod" / "lib" / "osx" / "libphonon.dylib"),
    )
    return next(((include, library) for include, library in candidates if (include / "phonon.h").is_file() and library.is_file()), None)


def _run_steam_audio_reference(context: _BenchmarkContext) -> dict[str, Any]:
    assert context.steam_audio_root is not None
    sdk = _steam_sdk_paths(context.steam_audio_root)
    if sdk is None:
        raise RuntimeError("Steam Audio headers or libphonon were not found")
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++ compiler is required for the native Steam Audio reference")
    include_dir, library = sdk
    source = Path(__file__).resolve().parent / "resources" / "benchmark" / "steam_audio_reference.cpp"
    context.work_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    # Steam Audio can be compiled with either its public 3-band ABI or the
    # experimental 11-band ABI. A mismatched public struct layout crashes
    # inside the SDK, so probe each ABI in a child process and retain the one
    # reported by the successful executable.
    for abi, define in (("octave", "-DIPL_ENABLE_OCTAVE_BANDS"), ("default", None)):
        executable = context.work_dir / f"steam_audio_reference_{abi}"
        rebuild = not executable.exists() or executable.stat().st_mtime < source.stat().st_mtime
        if rebuild:
            command = [
                compiler,
                "-std=c++17",
                "-O2",
            ]
            if define:
                command.append(define)
            command.extend([
                str(source),
                "-I",
                str(include_dir),
                str(library),
                "-Wl,-rpath," + str(library.parent),
                "-o",
                str(executable),
            ])
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if compiled.returncode != 0:
                errors.append(f"{abi} compile: {compiled.stderr.strip() or compiled.returncode}")
                continue
        command = [
            str(executable),
            "--rays", str(context.profile.steam_rays),
            "--bounces", str(context.profile.steam_bounces),
            "--duration", str(context.profile.duration_s),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            payload["detected_abi"] = abi
            return payload
        diagnostic = completed.stderr.strip() or completed.stdout.strip() or f"exit status {completed.returncode}"
        errors.append(f"{abi} run: {diagnostic}")
    raise RuntimeError("Steam Audio reference failed for both ABIs: " + "; ".join(errors))


def _metric(
    name: str,
    measured: Any,
    expected: Any,
    tolerance: Any,
    unit: str,
    passed: bool | None,
    note: str = "",
) -> BenchmarkMetric:
    return BenchmarkMetric(name, _plain(measured), _plain(expected), _plain(tolerance), unit, passed, note)


def _case(
    case_id: str,
    name: str,
    category: str,
    passed: bool,
    summary: str,
    metrics: list[BenchmarkMetric],
    *,
    scene: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(case_id, name, category, "pass" if passed else "fail", summary, 0.0, metrics, scene or {}, details or {})


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _markdown_report(report: AccuracyBenchmarkReport) -> str:
    summary = report.summary
    lines = [
        "# Acoustic Accuracy Benchmark",
        "",
        f"Generated: `{report.generated_at}`  ",
        f"Acoustic Agent: `{report.acoustic_agent_version}`  ",
        f"Profile: `{report.profile['name']}` - {report.profile['description']}",
        "",
        "## Summary",
        "",
        f"**{summary['pass']} passed, {summary['fail']} failed, {summary['error']} errors, {summary['skip']} skipped** in {summary['duration_s']:.2f} s.",
        "",
        "| Status | Check | Category | Time |",
        "| --- | --- | --- | ---: |",
    ]
    for case in report.cases:
        lines.append(f"| {case.status.upper()} | [{case.name}](#{_anchor(case.name)}) | {case.category} | {case.duration_s:.2f} s |")
    for case in report.cases:
        lines.extend(["", f"## {case.name}", "", f"**{case.status.upper()}** - {case.summary}", ""])
        if case.metrics:
            lines.extend(["| Metric | Measured | Expected | Tolerance | Result |", "| --- | ---: | ---: | ---: | --- |"])
            for metric in case.metrics:
                outcome = "-" if metric.passed is None else ("PASS" if metric.passed else "FAIL")
                lines.append(
                    f"| {metric.name} | {_format_value(metric.measured, metric.unit)} | "
                    f"{_format_value(metric.expected, metric.unit)} | {_format_value(metric.tolerance, metric.unit)} | {outcome} |"
                )
        if case.scene:
            lines.extend(["", "<details><summary>Scene configuration</summary>", "", "```json", json.dumps(case.scene, indent=2, ensure_ascii=False), "```", "", "</details>"])
        if case.details:
            lines.extend(["", "<details><summary>Measurements</summary>", "", "```json", json.dumps(case.details, indent=2, ensure_ascii=False), "```", "", "</details>"])
    lines.extend(["", "## Interpretation", "", "A skipped external reference is not counted as a pass. A failing check is retained as regression evidence and should be investigated rather than hidden by changing the report.", ""])
    return "\n".join(lines)


def _html_report(report: AccuracyBenchmarkReport) -> str:
    summary = report.summary
    cards = "".join(
        f'<div class="count {status}"><strong>{summary[status]}</strong><span>{status}</span></div>'
        for status in ("pass", "fail", "error", "skip")
    )
    sections = []
    for case in report.cases:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(metric.name)}</td>"
            f"<td>{html.escape(_format_value(metric.measured, metric.unit))}</td>"
            f"<td>{html.escape(_format_value(metric.expected, metric.unit))}</td>"
            f"<td>{html.escape(_format_value(metric.tolerance, metric.unit))}</td>"
            f'<td class="metric-{str(metric.passed).lower()}">{"-" if metric.passed is None else ("PASS" if metric.passed else "FAIL")}</td>'
            "</tr>"
            for metric in case.metrics
        )
        table = (
            "<table><thead><tr><th>Metric</th><th>Measured</th><th>Expected</th><th>Tolerance</th><th>Result</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            if rows else ""
        )
        details = html.escape(json.dumps({"scene": case.scene, "measurements": case.details}, indent=2, ensure_ascii=False))
        sections.append(
            f'<section id="{_anchor(case.name)}">'
            f'<header><span class="badge {case.status}">{case.status}</span><div><h2>{html.escape(case.name)}</h2><p>{html.escape(case.category)} · {case.duration_s:.2f} s</p></div></header>'
            f'<p>{html.escape(case.summary)}</p>{table}'
            f'<details><summary>Scene and raw measurements</summary><pre>{details}</pre></details></section>'
        )
    overall = "pass" if summary["required_passed"] else "fail"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Acoustic Accuracy Benchmark</title>
<style>
:root{{--ink:#172126;--muted:#607078;--line:#d8e0e3;--paper:#f5f7f7;--white:#fff;--pass:#147d64;--fail:#c33f45;--skip:#9a6b16;--error:#8b3fb0}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,sans-serif;letter-spacing:0}}
main{{width:min(1120px,calc(100% - 32px));margin:36px auto 80px}} .hero{{border-top:6px solid var(--{overall});padding:28px 30px;background:var(--white);box-shadow:0 1px 5px #17303a12}}
h1{{font-size:30px;margin:0 0 8px}} h2{{font-size:19px;margin:0}} p{{color:var(--muted)}} .counts{{display:grid;grid-template-columns:repeat(4,minmax(90px,1fr));gap:10px;margin-top:22px}}
.count{{padding:12px 15px;border:1px solid var(--line);color:#fff}} .count strong{{display:block;font-size:25px}} .count span{{text-transform:uppercase;font-size:11px;color:#fff;opacity:.82}}
section{{margin-top:16px;padding:22px 24px;background:var(--white);border:1px solid var(--line)}} section header{{display:flex;gap:13px;align-items:flex-start}} section header p{{margin:2px 0 0}}
.badge{{min-width:54px;text-align:center;padding:3px 7px;color:#fff;text-transform:uppercase;font-size:10px;font-weight:700}} .pass{{background:var(--pass)}} .fail{{background:var(--fail)}} .skip{{background:var(--skip)}} .error{{background:var(--error)}}
table{{width:100%;border-collapse:collapse;margin-top:16px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}} th{{font-size:11px;text-transform:uppercase;color:var(--muted)}}
.metric-true{{color:var(--pass);font-weight:700}} .metric-false{{color:var(--fail);font-weight:700}} details{{margin-top:15px}} summary{{cursor:pointer;color:var(--muted)}} pre{{overflow:auto;padding:14px;background:#eef2f3;font:12px/1.5 ui-monospace,monospace}}
@media(max-width:650px){{main{{width:min(100% - 18px,1120px);margin-top:9px}}.hero,section{{padding:18px}}.counts{{grid-template-columns:repeat(2,1fr)}}table{{display:block;overflow-x:auto}}}}
</style></head><body><main><div class="hero"><h1>Acoustic Accuracy Benchmark</h1>
<p>Acoustic Agent {html.escape(report.acoustic_agent_version)} · {html.escape(str(report.profile['name']))} · {html.escape(report.generated_at)}</p>
<div class="counts">{cards}</div></div>{''.join(sections)}</main></body></html>"""


def _format_value(value: Any, unit: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def _anchor(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")
