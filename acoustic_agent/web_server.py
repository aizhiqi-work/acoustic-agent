from __future__ import annotations

import argparse
import io
import json
import struct
import wave
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np

from .api import quality_preset
from .directivity import source_directivity
from .engine import SimulationResult, simulate_rir
from .geometry import make_room
from .materials import MaterialLibrary, material_summary
from .mic import microphone_array
from .models import Room, SimConfig
from .motion import motion_room_metadata
from .web_export import scene_payload


WEB_ROOT = Path(__file__).resolve().parent / "web"
CALIBRATION_AUDIO_PATH = Path(__file__).resolve().parents[2] / "reading.wav"
_SIMULATION_LOCK = Lock()
_RESULT_LOCK = Lock()
_RESULT_LIMIT = 64


@dataclass(frozen=True)
class PayloadSimulation:
    room: Room
    source: tuple[float, float, float]
    receiver: tuple[float, float, float]
    objects: list[dict[str, Any]]
    geometry: dict[str, Any]
    receiver_model: dict[str, Any]
    source_model: dict[str, Any]
    result: SimulationResult


@dataclass(frozen=True)
class StoredResult:
    metadata: dict[str, Any]
    wav: bytes
    npy: bytes


_RESULTS: OrderedDict[str, StoredResult] = OrderedDict()


class AcousticWorkbenchHandler(SimpleHTTPRequestHandler):
    floorplan_dataset: Any | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/geometry", "/floorplan", "/resplan", "/custom"} or Path(path).suffix in {".js", ".css", ".html"}:
            self.send_header("Cache-Control", "no-cache, max-age=0, must-revalidate")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/api/v1/floorplan/scene", "/api/v1/resplan/scene"}:
            try:
                dataset = self._require_floorplan_dataset()
                query = parse_qs(parsed.query)
                index = int(query.get("idx", ["0"])[0])
                room_id = query.get("room", [None])[0]
                receiver_room_id = query.get("receiver_room", [None])[0]
                height = float(query.get("height", ["2.8"])[0])
                self._send_json(dataset.scene(
                    index,
                    room_id,
                    receiver_room_id=receiver_room_id,
                    height_m=height,
                ))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path in {"/api/v1/floorplan/index", "/api/v1/resplan/index"}:
            try:
                dataset = self._require_floorplan_dataset()
                query = parse_qs(parsed.query)
                index = int(query.get("idx", ["0"])[0])
                direction = query.get("direction", ["nearest"])[0]
                self._send_json({"index": dataset.resolve_index(index, direction)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path in {"/api/v1/floorplan/stats", "/api/v1/resplan/stats"}:
            try:
                self._send_json(self._require_floorplan_dataset().stats())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/v1/custom/capabilities":
            self._send_json({
                "local_text_generation": True,
                "image_overlay": False,
                "json_editing": True,
                "chatgpt_handoff": True,
                "input_modes": ["image", "text"],
                "vlm": {
                    "available": False,
                    "provider": None,
                    "reason": "No VLM runtime is configured. Upload and edit locally, or add a provider later.",
                },
            })
            return
        if parsed.path == "/api/v1/custom/prompt":
            from .custom_floorplan import floorplan_text_prompt, floorplan_vlm_prompt

            try:
                query = parse_qs(parsed.query)
                mode = str(query.get("mode", ["image"])[0]).strip().lower()
                if mode == "text":
                    prompt = floorplan_text_prompt(query.get("description", [""])[0])
                elif mode == "image":
                    prompt = floorplan_vlm_prompt()
                else:
                    raise ValueError("mode must be image or text")
                self._send_json({"prompt": prompt, "mode": mode, "schema_version": 1})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path.startswith("/api/v1/results/"):
            self._send_stored_result(parsed.path)
            return
        if parsed.path == "/api/v1/materials/semantics":
            library = MaterialLibrary.load()
            self._send_json({"stats": library.stats(), "semantics": library.catalog()})
            return
        if parsed.path == "/api/calibration-audio":
            try:
                query = parse_qs(parsed.query)
                fs = int(query.get("fs", ["16000"])[0])
                if fs < 8000 or fs > 192000:
                    raise ValueError("sample rate must be between 8000 and 192000 Hz")
                self._send_audio(_calibration_audio_wav(fs))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.send_error(400, str(exc))
            return
        if parsed.path == "/":
            self.send_response(302)
            self.send_header("location", "/geometry")
            self.send_header("content-length", "0")
            self.end_headers()
            return
        if parsed.path == "/geometry":
            self._send_html((WEB_ROOT / "viewer.html").read_text(encoding="utf-8"))
            return
        if parsed.path in {"/floorplan", "/resplan"}:
            from .floorplan_web_server import _floorplan_viewer_html

            self._send_html(_floorplan_viewer_html())
            return
        if parsed.path == "/custom":
            from .custom_floorplan_web import custom_viewer_html

            self._send_html(custom_viewer_html())
            return
        return super().do_GET()

    def _require_floorplan_dataset(self) -> Any:
        if self.floorplan_dataset is None:
            raise RuntimeError("Floorplan resource is not configured")
        return self.floorplan_dataset

    def do_OPTIONS(self) -> None:
        if not urlparse(self.path).path.startswith("/api/"):
            self.send_error(404, "unknown API endpoint")
            return
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("content-length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/simulate", "/api/v1/simulate", "/api/v1/workbench", "/api/v1/dynamic-workbench", "/api/v1/custom/generate", "/api/v1/custom/compile", "/api/v1/custom/validate", "/api/rir.wav", "/api/rir.npy"}:
            self.send_error(404, "unknown API endpoint")
            return
        try:
            payload = self._read_json()
            if parsed.path.startswith("/api/v1/custom/"):
                from .custom_floorplan import compile_floorplan_spec, generate_floorplan_from_text, validate_floorplan_spec

                if parsed.path == "/api/v1/custom/generate":
                    spec = generate_floorplan_from_text(
                        str(payload.get("description", "")),
                        seed=int(payload.get("seed", 42)),
                        width_m=payload.get("width_m"),
                        depth_m=payload.get("depth_m"),
                        height_m=payload.get("height_m"),
                    )
                    validation = validate_floorplan_spec(spec)
                    scene = compile_floorplan_spec(
                        spec,
                        source_room=payload.get("source_room"),
                        receiver_room=payload.get("receiver_room"),
                        seed=int(payload.get("placement_seed", payload.get("seed", 42))),
                    )
                    self._send_json({"spec": spec, "validation": validation, "scene": scene})
                elif parsed.path == "/api/v1/custom/validate":
                    self._send_json(validate_floorplan_spec(payload.get("spec", {})))
                else:
                    spec = payload.get("spec")
                    if not isinstance(spec, dict):
                        raise ValueError("spec must be a JSON object")
                    validation = validate_floorplan_spec(spec)
                    scene = compile_floorplan_spec(
                        spec,
                        source_room=payload.get("source_room"),
                        receiver_room=payload.get("receiver_room"),
                        seed=int(payload.get("seed", 42)),
                        height_m=payload.get("height_m"),
                    )
                    self._send_json({"spec": validation.get("spec"), "validation": validation, "scene": scene})
                return
            with _SIMULATION_LOCK:
                if parsed.path == "/api/simulate":
                    response = simulate_from_payload(payload)
                    self._send_json(response)
                elif parsed.path == "/api/v1/simulate":
                    response = simulate_api_from_payload(payload)
                    self._send_json(response)
                elif parsed.path == "/api/v1/workbench":
                    response = simulate_workbench_from_payload(payload)
                    self._send_json(response)
                elif parsed.path == "/api/v1/dynamic-workbench":
                    response = simulate_dynamic_workbench_from_payload(payload)
                    self._send_json(response)
                else:
                    simulation = _simulate_payload(payload)
                    if parsed.path.endswith(".wav"):
                        self._send_binary(_float32_wav(simulation.result), "audio/wav", "rir.wav")
                    else:
                        self._send_binary(_npy_bytes(simulation.result.rir), "application/x-npy", "rir.npy")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self._send_json({"error": str(exc)}, status=400)
            except (BrokenPipeError, ConnectionResetError):
                return

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length < 1 or length > 10 * 1024 * 1024:
            raise ValueError("request body must be JSON between 1 byte and 10 MB")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send_json(self, value: Any, *, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), check_circular=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_audio(self, data: bytes) -> None:
        self._send_binary(data, "audio/wav")

    def _send_binary(self, data: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        if filename:
            self.send_header("content-disposition", f'attachment; filename="{filename}"')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_cors_headers(self) -> None:
        self.send_header("access-control-allow-origin", "*")

    def _send_stored_result(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) not in {4, 5} or parts[:3] != ["api", "v1", "results"]:
            self.send_error(404, "result not found")
            return
        result = _get_stored_result(parts[3])
        if result is None:
            self.send_error(404, "result not found or expired")
            return
        if len(parts) == 4:
            self._send_json(result.metadata)
        elif parts[4] == "rir.wav":
            self._send_binary(result.wav, "audio/wav", "rir.wav")
        elif parts[4] == "rir.npy":
            self._send_binary(result.npy, "application/x-npy", "rir.npy")
        else:
            self.send_error(404, "result artifact not found")


@lru_cache(maxsize=8)
def _calibration_audio_wav(fs: int) -> bytes:
    samples, source_fs = _read_wav_mono(CALIBRATION_AUDIO_PATH)
    resampled = _resample_linear(samples, source_fs, int(fs))
    pcm = np.rint(np.clip(resampled, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(fs))
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"calibration audio not found: {path}")
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("calibration audio is not a RIFF/WAVE file")

    fmt: tuple[int, int, int, int] | None = None
    audio_bytes: bytes | None = None
    offset = 12
    while offset + 8 <= len(raw):
        chunk_id, chunk_size = struct.unpack_from("<4sI", raw, offset)
        start = offset + 8
        end = min(start + int(chunk_size), len(raw))
        chunk = raw[start:end]
        if chunk_id == b"fmt " and len(chunk) >= 16:
            audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from("<HHIIHH", chunk, 0)
            fmt = (int(audio_format), int(channels), int(sample_rate), int(bits_per_sample))
        elif chunk_id == b"data":
            audio_bytes = chunk
        offset = start + int(chunk_size) + (int(chunk_size) & 1)

    if fmt is None or audio_bytes is None:
        raise ValueError("calibration WAV is missing fmt or data")
    audio_format, channels, sample_rate, bits_per_sample = fmt
    if channels < 1 or sample_rate < 1:
        raise ValueError("calibration WAV has invalid channel or sample-rate metadata")

    if audio_format == 3 and bits_per_sample == 32:
        values = np.frombuffer(audio_bytes, dtype="<f4").astype(np.float64)
    elif audio_format == 3 and bits_per_sample == 64:
        values = np.frombuffer(audio_bytes, dtype="<f8").astype(np.float64)
    elif audio_format == 1 and bits_per_sample == 16:
        values = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float64) / 32768.0
    elif audio_format == 1 and bits_per_sample == 32:
        values = np.frombuffer(audio_bytes, dtype="<i4").astype(np.float64) / 2147483648.0
    elif audio_format == 1 and bits_per_sample == 8:
        values = (np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported calibration WAV format: tag={audio_format}, bits={bits_per_sample}")

    frame_count = len(values) // channels
    if frame_count < 1:
        raise ValueError("calibration WAV contains no samples")
    frames = values[:frame_count * channels].reshape(frame_count, channels)
    mono = np.mean(frames, axis=1)
    return np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0), sample_rate


def _resample_linear(samples: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if source_fs == target_fs:
        return values.copy()
    output_length = max(1, int(round(len(values) * float(target_fs) / float(source_fs))))
    source_positions = np.arange(output_length, dtype=np.float64) * float(source_fs) / float(target_fs)
    return np.interp(source_positions, np.arange(len(values), dtype=np.float64), values)


def simulate_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    simulation = _simulate_payload(payload)
    return _scene_response(simulation, include_exact_rir=True)


def simulate_api_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    simulation = _simulate_payload(payload)
    _, result_metadata = _store_result(simulation.result)
    return result_metadata


def simulate_workbench_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    simulation = _simulate_payload(payload)
    result_id, result_metadata = _store_result(simulation.result)
    out = _scene_response(simulation, include_exact_rir=False)
    out["result_id"] = result_id
    out["result"] = result_metadata
    out["rir"].update({
        "encoding": "float32-wav",
        "wav_url": result_metadata["files"]["wav"],
        "npy_url": result_metadata["files"]["npy"],
    })
    return out


def simulate_dynamic_workbench_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    motion = payload.get("motion") if isinstance(payload.get("motion"), dict) else {}
    raw_frames = motion.get("frames") if isinstance(motion.get("frames"), list) else []
    if len(raw_frames) < 2 or len(raw_frames) > 65:
        raise ValueError("dynamic motion must contain between 2 and 65 frames")

    base_payload = dict(payload)
    base_payload.pop("motion", None)
    simulations: list[PayloadSimulation] = []
    frame_results: list[dict[str, Any]] = []
    previous_phase = -1.0
    for index, frame in enumerate(raw_frames):
        if not isinstance(frame, dict):
            raise ValueError("each dynamic motion frame must be an object")
        phase = float(frame.get("phase", index / max(len(raw_frames) - 1, 1)))
        if not 0.0 <= phase <= 1.0 or phase <= previous_phase:
            raise ValueError("dynamic motion frame phases must increase from 0 to 1")
        previous_phase = phase
        frame_payload = dict(base_payload)
        frame_payload["source"] = _float_list(frame.get("source"), 3)
        frame_payload["receiver"] = _float_list(frame.get("receiver"), 3)
        if isinstance(base_payload.get("room_metadata"), dict):
            frame_payload["room_metadata"] = motion_room_metadata(
                base_payload["room_metadata"],
                frame_payload["source"],
                frame_payload["receiver"],
            )
        simulation = _simulate_payload(frame_payload)
        simulations.append(simulation)
        result_id, result_metadata = _store_result(simulation.result)
        frame_results.append({
            "index": index,
            "phase": round(phase, 6),
            "source": [float(value) for value in simulation.source],
            "receiver": [float(value) for value in simulation.receiver],
            "result_id": result_id,
            "rir": {
                "fs": result_metadata["sample_rate"],
                "shape": result_metadata["shape"],
                "duration_s": result_metadata["duration_s"],
                "wav_url": result_metadata["files"]["wav"],
                "npy_url": result_metadata["files"]["npy"],
            },
            "rt60": dict(simulation.result.rt60),
        })

    reference = simulations[0]
    reference_result = frame_results[0]
    out = _scene_response(reference, include_exact_rir=False)
    out["result_id"] = reference_result["result_id"]
    out["result"] = _get_stored_result(reference_result["result_id"]).metadata
    out["rir"].update({
        "encoding": "float32-wav",
        "wav_url": reference_result["rir"]["wav_url"],
        "npy_url": reference_result["rir"]["npy_url"],
    })
    out["dynamic"] = {
        "mode": str(motion.get("mode", "approach")),
        "moving": str(motion.get("moving", "source")),
        "distance_m": float(motion.get("distance_m", 0.0)),
        "requested_distance_m": float(motion.get("requested_distance_m", motion.get("distance_m", 0.0))),
        "keyframes": len(frame_results),
        "path_model": str(motion.get("path_model", "local_smoothstep")),
        "frames": frame_results,
        "reference_frame": 0,
        "renderer": "time_varying_rir_snapshot_interpolation",
    }
    return out


def _simulate_payload(payload: dict[str, Any]) -> PayloadSimulation:
    shape = str(payload.get("shape", "rectangle"))
    size = _float_list(payload.get("size", (6.0, 4.0, 2.8)), 3)
    source = tuple(_float_list(payload.get("source", (1.2, 1.1, 1.5)), 3))
    receiver = tuple(_float_list(payload.get("receiver", (4.7, 2.8, 1.4)), 3))
    materials = payload.get("materials") if isinstance(payload.get("materials"), dict) else {}
    material_profile = payload.get("material_profile")
    if not isinstance(material_profile, (str, dict)):
        material_profile = None
    material_seed = int(payload.get("material_seed", 0))
    objects = _object_list(payload.get("objects"))
    geometry = payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}
    room_metadata = payload.get("room_metadata") if isinstance(payload.get("room_metadata"), dict) else {}
    config_raw = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    receiver_raw = payload.get("receiver_model") if isinstance(payload.get("receiver_model"), dict) else {"type": "mono"}
    source_raw = payload.get("source_model", "omni")
    if not isinstance(source_raw, (str, dict)):
        raise ValueError("source_model must be a preset name or an object")

    room_kwargs: dict[str, Any] = {}
    if isinstance(payload.get("corners"), list):
        room_kwargs["corners"] = _corner_list(payload["corners"])
    elif shape == "polygon":
        room_kwargs["corners"] = _polygon_corners(size)
    room = make_room(
        shape,
        size=size,
        materials=materials,
        material_profile=material_profile,
        material_seed=material_seed,
        **room_kwargs,
    )
    if isinstance(room.metadata, dict):
        library = MaterialLibrary.load()
        for index, item in enumerate(objects):
            if item.get("semantic") == "small_objects_ignore":
                continue
            item["material_selection"] = material_summary(library.sample_geometry(item, seed=material_seed + index + 1))
        room.metadata["objects"] = objects
        for key in (
            "floorplan",
            "boundary_features",
            "surface_segments",
            "opening_model",
            "connectivity_model",
            "connections",
            "source_room_id",
            "receiver_room_id",
            "multi_room",
        ):
            if key in room_metadata:
                room.metadata[key] = room_metadata[key]
    quality = str(config_raw.get("quality", payload.get("quality", "simulation")))
    quality_config = _quality_config(quality)
    rt_num_rays = int(config_raw.get("rt_num_rays", quality_config["rt_num_rays"]))
    rt_num_bounces = int(config_raw.get("rt_num_bounces", quality_config["rt_num_bounces"]))
    config = SimConfig(
        fs=int(config_raw.get("fs", 16000)),
        duration_s=float(config_raw.get("duration_s", 2.0)),
        seed=int(config_raw.get("seed", 1729)),
        direct_occlusion=bool(config_raw.get("direct_occlusion", True)),
        direct_transmission=bool(config_raw.get("direct_transmission", True)),
        direct_occlusion_mode=str(config_raw.get("direct_occlusion_mode", "volumetric")),
        direct_occlusion_radius_m=float(config_raw.get("direct_occlusion_radius_m", 0.1)),
        direct_occlusion_samples=int(config_raw.get("direct_occlusion_samples", 32)),
        num_transmission_rays=int(config_raw.get("num_transmission_rays", 8)),
        reflections_enabled=bool(config_raw.get("reflections_enabled", True)),
        rt_num_rays=rt_num_rays,
        rt_num_bounces=rt_num_bounces,
        rt_duration_s=float(config_raw.get("rt_duration_s", quality_config["rt_duration_s"])),
        rt_visual_num_rays=int(config_raw["rt_visual_num_rays"]) if "rt_visual_num_rays" in config_raw else None,
        rt_visual_num_bounces=int(config_raw["rt_visual_num_bounces"]) if "rt_visual_num_bounces" in config_raw else None,
        adaptive_geometry_bounces=bool(config_raw.get("adaptive_geometry_bounces", True)),
        geometry_max_bounces=int(config_raw.get("geometry_max_bounces", 128)),
        adaptive_cross_room_bounces=bool(config_raw.get("adaptive_cross_room_bounces", True)),
        cross_room_min_bounces=int(config_raw.get("cross_room_min_bounces", 96)),
        cross_room_max_bounces=int(config_raw.get("cross_room_max_bounces", 128)),
        portal_aperture_attenuation=bool(config_raw.get("portal_aperture_attenuation", False)),
        late_tail=bool(config_raw.get("late_tail", True)),
        late_tail_cutoff_s=float(config_raw.get("late_tail_cutoff_s", 0.08)),
        hybrid_transition_s=float(config_raw.get("hybrid_transition_s", 1.0)),
        hybrid_overlap_fraction=float(config_raw.get("hybrid_overlap_fraction", 0.25)),
        diffraction_enabled=bool(config_raw.get("diffraction_enabled", True)),
        diffraction_audio_enabled=bool(config_raw.get("diffraction_audio_enabled", True)),
        diffraction_order=int(config_raw.get("diffraction_order", 3)),
        max_diffraction_paths=int(config_raw.get("max_diffraction_paths", 8)),
    )
    receiver_model = _receiver_model(receiver_raw)
    source_model = source_directivity(source_raw)
    result = simulate_rir(
        room,
        source,
        receiver,
        config=config,
        receiver_model=receiver_model,
        source_model=source_model,
    )
    return PayloadSimulation(room, source, receiver, objects, geometry, receiver_model, source_model, result)


def _scene_response(simulation: PayloadSimulation, *, include_exact_rir: bool) -> dict[str, Any]:
    out = scene_payload(
        simulation.room,
        sources=[simulation.source],
        receivers=[simulation.receiver],
        result=simulation.result,
        include_exact_rir=include_exact_rir,
    )
    out["objects"] = simulation.objects
    out.setdefault("room", {}).setdefault("metadata", {})["geometry_params"] = simulation.geometry
    out["metadata"] = {
        **dict(out.get("metadata", {})),
        "rir_shape": list(simulation.result.rir.shape),
        "receiver_model": simulation.receiver_model,
        "source_model": simulation.source_model,
    }
    return out


def _store_result(result: SimulationResult) -> tuple[str, dict[str, Any]]:
    result_id = uuid4().hex
    fs = int(result.metadata.get("sample_rate", 16000))
    values = np.asarray(result.rir, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    base_url = f"/api/v1/results/{result_id}"
    metadata = {
        "id": result_id,
        "sample_rate": fs,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "duration_s": float(values.shape[1] / max(fs, 1)),
        "receiver_model": result.receiver_model,
        "source_model": result.source_model,
        "rt60": dict(result.rt60),
        "files": {
            "wav": f"{base_url}/rir.wav",
            "npy": f"{base_url}/rir.npy",
        },
    }
    stored = StoredResult(metadata, _float32_wav(result), _npy_bytes(values))
    with _RESULT_LOCK:
        _RESULTS[result_id] = stored
        _RESULTS.move_to_end(result_id)
        while len(_RESULTS) > _RESULT_LIMIT:
            _RESULTS.popitem(last=False)
    return result_id, metadata


def _get_stored_result(result_id: str) -> StoredResult | None:
    with _RESULT_LOCK:
        result = _RESULTS.get(result_id)
        if result is not None:
            _RESULTS.move_to_end(result_id)
        return result


def _float32_wav(result: SimulationResult) -> bytes:
    values = np.asarray(result.rir, dtype="<f4")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("RIR must have shape [channels, samples]")
    channels, frames = (int(values.shape[0]), int(values.shape[1]))
    fs = int(result.metadata.get("sample_rate", 16000))
    interleaved = np.ascontiguousarray(values.T, dtype="<f4").tobytes(order="C")
    block_align = channels * 4
    fmt = struct.pack("<HHIIHH", 3, channels, fs, fs * block_align, block_align, 32)
    fact = struct.pack("<I", frames)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"fact" + struct.pack("<I", len(fact)) + fact
    body += b"data" + struct.pack("<I", len(interleaved)) + interleaved
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def _npy_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return output.getvalue()


def _receiver_model(raw: dict[str, Any]) -> dict[str, Any]:
    kind = str(raw.get("type", "mono"))
    orientation = float(raw.get("orientation_deg", 0.0))
    if kind in {"linear", "linear_array"}:
        return microphone_array("linear", count=int(raw.get("count", 4)), spacing_m=float(raw.get("spacing_m", 0.08)), orientation_deg=orientation)
    if kind in {"circular", "circular_array"}:
        return microphone_array("circular", count=int(raw.get("count", 8)), radius_m=float(raw.get("radius_m", 0.12)), orientation_deg=orientation)
    if kind == "hrtf":
        return microphone_array(
            "hrtf",
            orientation_deg=orientation,
            interpolation=str(raw.get("interpolation", "bilinear")),
            spatial_blend=float(raw.get("spatial_blend", 1.0)),
            loudness_normalization=str(raw.get("loudness_normalization", "energy")),
            sofa_path=raw.get("sofa_path"),
        )
    return microphone_array("mono", orientation_deg=orientation)


def _object_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    objects: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        size = _float_list(item.get("size", (1.0, 1.0, 1.0)), 3)
        position_raw = item.get("position", (0.0, 0.0))
        if not isinstance(position_raw, (list, tuple)) or len(position_raw) < 2:
            position = [0.0, 0.0]
        else:
            position = [float(position_raw[0]), float(position_raw[1])]
        output = {
            "id": str(item.get("id", f"object_{index}")),
            "type": str(item.get("type", "cabinet")),
            "title": str(item.get("title", item.get("type", "Object"))),
            "semantic": str(item.get("semantic", item.get("type", "structural_element"))),
            "absorption_class": str(item.get("absorption_class", item.get("absorption_level", "auto"))),
            "position": position,
            "rotation": float(item.get("rotation", 0.0)),
            "size": size,
            "z": float(item.get("z", size[2] * 0.5)),
        }
        if item.get("material") or item.get("material_id"):
            output["material"] = str(item.get("material") or item.get("material_id"))
        if item.get("material_type"):
            output["material_type"] = str(item["material_type"])
        objects.append(output)
    return objects


def _quality_config(quality: str) -> dict[str, float | int]:
    return quality_preset(quality)


def _warm_simulation_kernels() -> None:
    simulate_from_payload({
        "shape": "rectangle",
        "size": [1.0, 1.0, 1.0],
        "source": [0.25, 0.25, 0.5],
        "receiver": [0.75, 0.75, 0.5],
        "config": {
            "fs": 8000,
            "duration_s": 0.02,
            "rt_num_rays": 1,
            "rt_num_bounces": 1,
            "rt_duration_s": 0.02,
            "diffraction_enabled": False,
        },
        "receiver_model": {"type": "mono"},
    })


def _float_list(value: Any, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"expected a list of {length} numbers")
    return [float(item) for item in value]


def _polygon_corners(size: list[float]) -> list[tuple[float, float]]:
    x, y = float(size[0]), float(size[1])
    return [
        (0.0, 0.0),
        (x * 0.72, 0.0),
        (x, y * 0.42),
        (x * 0.62, y),
        (x * 0.12, y * 0.86),
        (-x * 0.02, y * 0.26),
    ]


def _corner_list(value: Any) -> list[tuple[float, float]]:
    if len(value) < 3:
        raise ValueError("corners must contain at least three points")
    corners: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("each corner must contain x and y")
        corners.append((float(point[0]), float(point[1])))
    return corners


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    floorplan_resource: str | Path | None = None,
    floorplan_dataset: str | Path | None = None,
    warmup: bool = True,
) -> None:
    if floorplan_dataset is not None:
        from .floorplan import FloorplanDataset

        AcousticWorkbenchHandler.floorplan_dataset = FloorplanDataset(floorplan_dataset)
    else:
        from .floorplan_resource import DEFAULT_FLOORPLAN_RESOURCE, FloorplanResource

        resource_path = floorplan_resource or DEFAULT_FLOORPLAN_RESOURCE
        AcousticWorkbenchHandler.floorplan_dataset = FloorplanResource(resource_path)
    if warmup:
        _warm_simulation_kernels()
    server = ThreadingHTTPServer((host, int(port)), AcousticWorkbenchHandler)
    print(f"Acoustic Agent geometry: http://{host}:{port}/geometry")
    print(f"Acoustic Agent Floorplan:  http://{host}:{port}/floorplan")
    print(f"Acoustic Agent Custom:     http://{host}:{port}/custom")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Acoustic Agent WebGL workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--floorplan-resource", "--resplan-resource",
        dest="floorplan_resource",
        type=Path,
        default=None,
        help="Compiled Floorplan SQLite resource for the unified Floorplan route.",
    )
    parser.add_argument(
        "--floorplan-dataset", "--resplan-dataset",
        dest="floorplan_dataset",
        type=Path,
        default=None,
        help="Optional legacy Floorplan pickle instead of the compiled resource.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip startup JIT warmup; the first simulation will compile kernels.",
    )
    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        floorplan_resource=args.floorplan_resource,
        floorplan_dataset=args.floorplan_dataset,
        warmup=not args.no_warmup,
    )


if __name__ == "__main__":
    main()
