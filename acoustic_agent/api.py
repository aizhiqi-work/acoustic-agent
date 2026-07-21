from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .directivity import source_directivity
from .engine import SimulationResult, simulate_rir
from .geometry import make_room
from .materials import MaterialLibrary, material_summary
from .mic import microphone_array
from .models import Material, Room, SimConfig
from .motion import room_for_motion_frame, sample_motion
from .floorplan_resource import DEFAULT_FLOORPLAN_RESOURCE, FloorplanResource
from .furnishing import generate_floorplan_furniture


QUALITY_PRESETS: dict[str, dict[str, int | float]] = {
    "preview": {"rt_num_rays": 8192, "rt_num_bounces": 32, "rt_duration_s": 2.0},
    "simulation": {"rt_num_rays": 32768, "rt_num_bounces": 64, "rt_duration_s": 2.0},
    "fine": {"rt_num_rays": 65536, "rt_num_bounces": 96, "rt_duration_s": 2.0},
    "reference": {"rt_num_rays": 131072, "rt_num_bounces": 96, "rt_duration_s": 2.0},
}


@dataclass(frozen=True)
class DynamicSimulationResult:
    motion: Mapping[str, Any]
    frames: tuple[SimulationResult, ...]

    @property
    def rirs(self) -> tuple[Any, ...]:
        return tuple(frame.rir for frame in self.frames)


def quality_preset(quality: str) -> dict[str, int | float]:
    key = str(quality).lower()
    if key == "offline_reference":
        key = "reference"
    if key not in QUALITY_PRESETS:
        choices = ", ".join(QUALITY_PRESETS)
        raise ValueError(f"unknown quality {quality!r}; expected one of: {choices}")
    return dict(QUALITY_PRESETS[key])


def _floorplan_position(value: Sequence[float], label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} must contain [x, y, z]")
    point = [float(item) for item in value]
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{label} contains a non-finite value")
    return point


class AcousticAgent:
    """Small object-oriented facade for the common single-room workflow."""

    def __init__(
        self,
        room: Room | Sequence[float] | Mapping[str, Any] = (6.0, 4.0, 2.8),
        *,
        shape: str = "rectangle",
        quality: str = "simulation",
        materials: Mapping[str, str | Material] | None = None,
        material_profile: str | Mapping[str, Any] | None = None,
        material_seed: int = 0,
        seed: int | None = None,
        fs: int = 16000,
        duration_s: float = 2.0,
        config: SimConfig | None = None,
        receiver_model: str | Mapping[str, Any] = "mono",
        source_model: str | Mapping[str, Any] = "omni",
        acoustic_geometry: Sequence[Mapping[str, Any]] | None = None,
        visualization: bool = False,
    ) -> None:
        self.room = _make_agent_room(
            room,
            shape=shape,
            materials=materials,
            material_profile=material_profile,
            material_seed=material_seed,
        )
        if acoustic_geometry is not None:
            self.room = _room_with_acoustic_geometry(self.room, acoustic_geometry)
        self.receiver_model = _microphone_model(receiver_model)
        receiver_kind = str(self.receiver_model.get("type", "mono"))
        preset = quality_preset(quality)
        if config is None:
            self.config = SimConfig(
                fs=int(fs),
                duration_s=float(duration_s),
                rt_num_rays=int(preset["rt_num_rays"]),
                rt_num_bounces=int(preset["rt_num_bounces"]),
                rt_duration_s=float(preset["rt_duration_s"]),
                seed=SimConfig.seed if seed is None else int(seed),
                collect_visual_paths=bool(visualization),
                render_ambisonics=receiver_kind == "hrtf",
            )
        else:
            self.config = replace(
                config,
                collect_visual_paths=bool(visualization),
                render_ambisonics=receiver_kind == "hrtf",
            )
        self.source_model = source_directivity(source_model)
        self.default_source: tuple[float, float, float] | None = None
        self.default_receiver: tuple[float, float, float] | None = None
        self.rooms: list[dict[str, Any]] = []
        self.placement: dict[str, Any] | None = None
        self.floorplan: dict[str, Any] | None = None
        self.furnishing: dict[str, Any] | None = None
        self.scene_type = "geometry"

    @classmethod
    def create(cls, scene: str = "geometry", **options: Any) -> "AcousticAgent":
        """Create any supported scene through one compact public entry point."""
        kind = str(scene).strip().lower().replace("-", "_")
        aliases = {
            "sample_rate": "fs",
            "rir_length": "duration_s",
            "mic": "receiver",
            "microphone": "receiver_model",
            "directivity": "source_model",
            "objects": "acoustic_geometry",
            "floorplan_spec": "spec",
        }
        values = dict(options)
        for alias, canonical in aliases.items():
            if alias not in values:
                continue
            if canonical in values:
                raise TypeError(f"use either {alias!r} or {canonical!r}, not both")
            values[canonical] = values.pop(alias)

        if kind in {"geometry", "room", "geometric"}:
            source = values.pop("source", None)
            receiver = values.pop("receiver", None)
            if values.get("seed") is not None and "material_seed" not in values:
                values["material_seed"] = int(values["seed"])
            agent = cls(**values)
            agent.default_source = _optional_position(source, "source")
            agent.default_receiver = _optional_position(receiver, "receiver")
            return agent

        if kind in {"floorplan", "floor_plan", "resplan"}:
            if "idx" not in values:
                raise TypeError("floorplan scenes require idx")
            idx = int(values.pop("idx"))
            agent = cls.from_floorplan(idx=idx, **values)
            agent.scene_type = "floorplan"
            return agent

        if kind in {"custom", "custom_floorplan", "custom_floor_plan"}:
            if "spec" not in values:
                raise TypeError("custom scenes require spec")
            spec = values.pop("spec")
            if not isinstance(spec, Mapping):
                raise TypeError("spec must be a floorplan mapping")
            agent = cls.from_floorplan_spec(spec, **values)
            agent.scene_type = "custom"
            return agent

        raise ValueError("scene must be geometry, floorplan, or custom")

    @classmethod
    def from_floorplan(
        cls,
        idx: int,
        *,
        placement: str = "random",
        seed: int | None = None,
        source: Sequence[float] | None = None,
        receiver: Sequence[float] | None = None,
        source_room: str | None = None,
        receiver_room: str | None = None,
        quality: str = "simulation",
        materials: Mapping[str, str | Material] | None = None,
        material_profile: str | Mapping[str, Any] | None = None,
        material_seed: int | None = None,
        fs: int = 16000,
        duration_s: float = 2.0,
        config: SimConfig | None = None,
        receiver_model: str | Mapping[str, Any] = "mono",
        source_model: str | Mapping[str, Any] = "omni",
        acoustic_geometry: Sequence[Mapping[str, Any]] | None = None,
        furnishing: str | Mapping[str, Any] | None = None,
        visualization: bool = False,
        mic_type: str | None = None,
        source_directivity: str | None = None,
        room_height_m: float = 2.8,
        position_height_m: float = 1.4,
        resource: FloorplanResource | str | Path | None = None,
    ) -> "AcousticAgent":
        loader = resource if isinstance(resource, FloorplanResource) else FloorplanResource(resource or DEFAULT_FLOORPLAN_RESOURCE)
        sampled = loader.sample_placement(
            idx,
            placement=placement,
            seed=seed,
            source_room=source_room,
            receiver_room=receiver_room,
            height_m=position_height_m,
        )
        if source is not None:
            sampled["source"] = _floorplan_position(source, "source")
        if receiver is not None:
            sampled["receiver"] = _floorplan_position(receiver, "receiver")
        scene = loader.scene(
            idx,
            sampled["source_room"],
            receiver_room_id=sampled["receiver_room"],
            height_m=room_height_m,
            source=sampled["source"],
            receiver=sampled["receiver"],
        )
        actual_material_seed = int(seed if material_seed is None and seed is not None else material_seed or 0)
        furnishing_layout = _resolve_furnishing(
            scene["room"]["metadata"],
            furnishing,
            default_seed=int(seed or 0),
            exclude_points=(scene["source"], scene["receiver"]),
            existing_objects=acoustic_geometry or (),
        )
        resolved_geometry = list(acoustic_geometry or ()) + list(furnishing_layout.get("objects", ()) if furnishing_layout else ())
        agent = cls(
            room=scene["room"],
            quality=quality,
            materials=materials,
            material_profile=material_profile,
            material_seed=actual_material_seed,
            fs=fs,
            duration_s=duration_s,
            config=config,
            receiver_model=mic_type or receiver_model,
            source_model=source_directivity or source_model,
            acoustic_geometry=resolved_geometry or None,
            visualization=visualization,
        )
        agent.default_source = tuple(float(value) for value in scene["source"])
        agent.default_receiver = tuple(float(value) for value in scene["receiver"])
        agent.room = room_for_motion_frame(agent.room, agent.default_source, agent.default_receiver)
        agent.rooms = [dict(room) for room in scene["rooms"]]
        multi_room = agent.room.metadata.get("multi_room") if isinstance(agent.room.metadata, Mapping) else None
        resolved_source_room = str(multi_room.get("source_room_id", sampled["source_room"])) if isinstance(multi_room, Mapping) else str(sampled["source_room"])
        resolved_receiver_room = str(multi_room.get("receiver_room_id", sampled["receiver_room"])) if isinstance(multi_room, Mapping) else str(sampled["receiver_room"])
        agent.placement = {
            **sampled,
            "source_room": resolved_source_room,
            "receiver_room": resolved_receiver_room,
            "source": list(agent.default_source),
            "receiver": list(agent.default_receiver),
        }
        agent.floorplan = dict(scene["dataset"])
        agent.furnishing = furnishing_layout
        agent.scene_type = "floorplan"
        return agent

    @classmethod
    def from_floorplan_spec(
        cls,
        spec: Mapping[str, Any],
        *,
        seed: int = 42,
        source: Sequence[float] | None = None,
        receiver: Sequence[float] | None = None,
        source_room: str | None = None,
        receiver_room: str | None = None,
        quality: str = "simulation",
        materials: Mapping[str, str | Material] | None = None,
        material_profile: str | Mapping[str, Any] | None = None,
        material_seed: int | None = None,
        fs: int = 16000,
        duration_s: float = 2.0,
        config: SimConfig | None = None,
        receiver_model: str | Mapping[str, Any] = "mono",
        source_model: str | Mapping[str, Any] = "omni",
        acoustic_geometry: Sequence[Mapping[str, Any]] | None = None,
        furnishing: str | Mapping[str, Any] | None = None,
        visualization: bool = False,
    ) -> "AcousticAgent":
        from .custom_floorplan import compile_floorplan_spec

        scene = compile_floorplan_spec(
            spec,
            source_room=source_room,
            receiver_room=receiver_room,
            seed=int(seed),
        )
        actual_source = _floorplan_position(source, "source") if source is not None else list(scene["source"])
        actual_receiver = _floorplan_position(receiver, "receiver") if receiver is not None else list(scene["receiver"])
        actual_material_seed = int(seed if material_seed is None else material_seed)
        furnishing_layout = _resolve_furnishing(
            scene["room"]["metadata"],
            furnishing,
            default_seed=int(seed),
            exclude_points=(actual_source, actual_receiver),
            existing_objects=acoustic_geometry or (),
        )
        resolved_geometry = list(acoustic_geometry or ()) + list(furnishing_layout.get("objects", ()) if furnishing_layout else ())
        agent = cls(
            room=scene["room"],
            quality=quality,
            materials=materials,
            material_profile=material_profile,
            material_seed=actual_material_seed,
            fs=fs,
            duration_s=duration_s,
            config=config,
            receiver_model=receiver_model,
            source_model=source_model,
            acoustic_geometry=resolved_geometry or None,
            visualization=visualization,
        )
        agent.default_source = tuple(float(value) for value in actual_source)
        agent.default_receiver = tuple(float(value) for value in actual_receiver)
        agent.room = room_for_motion_frame(agent.room, agent.default_source, agent.default_receiver)
        agent.rooms = [dict(room) for room in scene["rooms"]]
        multi_room = agent.room.metadata.get("multi_room") if isinstance(agent.room.metadata, Mapping) else None
        source_id = str(scene["selected_room"]["id"])
        receiver_id = str(scene["receiver_room"]["id"])
        agent.placement = {
            "mode": "same_room" if source_id == receiver_id else "cross_room",
            "seed": int(seed),
            "source_room": str(multi_room.get("source_room_id", source_id)) if isinstance(multi_room, Mapping) else source_id,
            "receiver_room": str(multi_room.get("receiver_room_id", receiver_id)) if isinstance(multi_room, Mapping) else receiver_id,
            "source": list(agent.default_source),
            "receiver": list(agent.default_receiver),
        }
        agent.floorplan = dict(scene["dataset"])
        agent.furnishing = furnishing_layout
        agent.scene_type = "custom"
        return agent

    # Compatibility alias for releases before the public scene name became Floorplan.
    from_resplan = from_floorplan

    @property
    def resplan(self) -> dict[str, Any] | None:
        return self.floorplan

    @resplan.setter
    def resplan(self, value: Mapping[str, Any] | None) -> None:
        self.floorplan = dict(value) if value is not None else None

    def run(
        self,
        source: Sequence[float] | None = None,
        receiver: Sequence[float] | None = None,
        *,
        config: SimConfig | None = None,
        receiver_model: str | Mapping[str, Any] | None = None,
        source_model: str | Mapping[str, Any] | None = None,
        motion: str | Mapping[str, Any] | None = None,
    ) -> SimulationResult | DynamicSimulationResult:
        actual_source = source if source is not None else self.default_source
        actual_receiver = receiver if receiver is not None else self.default_receiver
        if actual_source is None or actual_receiver is None:
            raise ValueError("source and receiver are required; pass them to create() or run()")
        if motion is not None:
            motion_spec = {"mode": motion} if isinstance(motion, str) else dict(motion)
            mode = str(motion_spec.get("mode", "approach")).strip().lower()
            if mode not in {"", "static", "none", "off"}:
                if "frames" in motion_spec:
                    sampled_motion = motion_spec
                else:
                    supported = {
                        "mode",
                        "moving",
                        "distance_m",
                        "keyframes",
                        "keyframe_spacing_m",
                        "seed",
                    }
                    unknown = sorted(set(motion_spec) - supported)
                    if unknown:
                        raise TypeError(f"unknown motion option(s): {', '.join(unknown)}")
                    sampled_motion = self.sample_motion(
                        source=actual_source,
                        receiver=actual_receiver,
                        **motion_spec,
                    )
                return self.run_dynamic(
                    sampled_motion,
                    config=config,
                    receiver_model=receiver_model,
                    source_model=source_model,
                )
        model = self.receiver_model
        if receiver_model is not None:
            model = _microphone_model(receiver_model)
        emitter = self.source_model if source_model is None else source_directivity(source_model)
        result = simulate_rir(
            self.room,
            actual_source,
            actual_receiver,
            config=config or self.config,
            receiver_model=model,
            source_model=emitter,
        )
        if self.placement is None:
            return result
        placement = {
            **self.placement,
            "source": [float(value) for value in actual_source],
            "receiver": [float(value) for value in actual_receiver],
        }
        return SimulationResult(
            result.rir,
            result.paths,
            result.rt60,
            result.receiver_model,
            {**dict(result.metadata), "placement": placement, "floorplan": dict(self.floorplan or {})},
            result.ambisonic_rir,
            result.source_model,
        )

    def run_batch(
        self,
        pairs: Sequence[Any],
        *,
        workers: int = 1,
        config: SimConfig | None = None,
        receiver_model: str | Mapping[str, Any] | None = None,
        source_model: str | Mapping[str, Any] | None = None,
    ) -> Any:
        """Simulate many source/receiver pairs in this already-built scene."""
        from .batch import BatchResult, SimulationPair

        jobs = tuple(SimulationPair.from_value(pair, index) for index, pair in enumerate(pairs))
        base_config = config or self.config
        model = self.receiver_model if receiver_model is None else _microphone_model(receiver_model)
        emitter = self.source_model if source_model is None else source_directivity(source_model)

        def solve(index_pair: tuple[int, SimulationPair]) -> SimulationResult:
            index, pair = index_pair
            seed = pair.seed if pair.seed is not None else int(base_config.seed + index)
            item_config = SimConfig(**{**base_config.__dict__, "seed": int(seed)})
            return self.run(
                source=pair.source,
                receiver=pair.receiver,
                config=item_config,
                receiver_model=model,
                source_model=emitter,
            )

        indexed = tuple(enumerate(jobs))
        worker_count = max(1, int(workers))
        if worker_count == 1:
            items = tuple(solve(item) for item in indexed)
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="acoustic-agent-pairs") as pool:
                items = tuple(pool.map(solve, indexed))
        return BatchResult(
            items=items,
            pairs=jobs,
            metadata={
                "model": "acoustic_agent_batch_v2",
                "scene": self.scene_type,
                "count": len(items),
                "workers": worker_count,
                "sample_rate": int(base_config.fs),
            },
        )

    @classmethod
    def run_many(
        cls,
        jobs: Sequence[Mapping[str, Any]],
        *,
        workers: int = 1,
        on_error: str = "raise",
    ) -> Any:
        """Run independent mixed-scene jobs for reproducible dataset production."""
        from .batch import ProductionResult

        error_mode = str(on_error).strip().lower()
        if error_mode not in {"raise", "skip"}:
            raise ValueError("on_error must be 'raise' or 'skip'")
        normalized = []
        for index, job in enumerate(jobs):
            if not isinstance(job, Mapping):
                raise TypeError(f"jobs[{index}] must be a mapping")
            normalized.append(dict(job))

        def solve(index_job: tuple[int, Mapping[str, Any]]) -> tuple[int, Any, Mapping[str, Any] | None]:
            index, job = index_job
            options = dict(job)
            options.pop("id", None)
            motion = options.pop("motion", None)
            try:
                return index, cls.create(**options).run(motion=motion), None
            except Exception as error:
                if error_mode == "raise":
                    raise RuntimeError(f"jobs[{index}] failed: {error}") from error
                return index, None, {
                    "index": index,
                    "id": str(job.get("id", index)),
                    "type": type(error).__name__,
                    "message": str(error),
                }

        indexed = tuple(enumerate(normalized))
        worker_count = max(1, int(workers))
        if worker_count == 1:
            outcomes = tuple(solve(item) for item in indexed)
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="acoustic-agent-scenes") as pool:
                outcomes = tuple(pool.map(solve, indexed))
        successful = tuple((index, item) for index, item, error in outcomes if error is None)
        errors = tuple(error for _, _, error in outcomes if error is not None)
        return ProductionResult(
            items=tuple(item for _, item in successful),
            jobs=tuple(normalized[index] for index, _ in successful),
            metadata={
                "model": "acoustic_agent_production_v1",
                "requested": len(normalized),
                "count": len(successful),
                "failed": len(errors),
                "workers": worker_count,
            },
            errors=errors,
        )

    def sample_motion(
        self,
        *,
        mode: str = "approach",
        moving: str = "source",
        source: Sequence[float] | None = None,
        receiver: Sequence[float] | None = None,
        distance_m: float = 0.8,
        keyframes: int | None = None,
        keyframe_spacing_m: float = 0.25,
        seed: int = 42,
    ) -> dict[str, Any]:
        actual_source = source if source is not None else self.default_source
        actual_receiver = receiver if receiver is not None else self.default_receiver
        if actual_source is None or actual_receiver is None:
            raise ValueError("source and receiver are required to sample motion")
        return sample_motion(
            self.room,
            actual_source,
            actual_receiver,
            mode=mode,
            moving=moving,
            distance_m=distance_m,
            keyframes=keyframes,
            keyframe_spacing_m=keyframe_spacing_m,
            seed=seed,
        )

    def run_dynamic(
        self,
        motion: Mapping[str, Any],
        *,
        config: SimConfig | None = None,
        receiver_model: str | Mapping[str, Any] | None = None,
        source_model: str | Mapping[str, Any] | None = None,
    ) -> DynamicSimulationResult:
        raw_frames = motion.get("frames") if isinstance(motion, Mapping) else None
        if not isinstance(raw_frames, Sequence) or not raw_frames:
            raise ValueError("motion must contain at least one frame")
        results = []
        model = self.receiver_model if receiver_model is None else _microphone_model(receiver_model)
        emitter = self.source_model if source_model is None else source_directivity(source_model)
        for frame in raw_frames:
            if not isinstance(frame, Mapping):
                raise ValueError("each motion frame must be an object")
            actual_source = frame.get("source")
            actual_receiver = frame.get("receiver")
            dynamic_room = room_for_motion_frame(self.room, actual_source, actual_receiver)
            result = simulate_rir(
                dynamic_room,
                actual_source,
                actual_receiver,
                config=config or self.config,
                receiver_model=model,
                source_model=emitter,
            )
            if self.placement is not None:
                placement = {
                    **self.placement,
                    "source": [float(value) for value in actual_source],
                    "receiver": [float(value) for value in actual_receiver],
                }
                result = SimulationResult(
                    result.rir,
                    result.paths,
                    result.rt60,
                    result.receiver_model,
                    {**dict(result.metadata), "placement": placement, "floorplan": dict(self.floorplan or {})},
                    result.ambisonic_rir,
                    result.source_model,
                )
            results.append(result)
        return DynamicSimulationResult(dict(motion), tuple(results))

    __call__ = run


def _microphone_model(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return microphone_array(value)
    model = dict(value)
    if "channels" in model:
        return model
    kind = str(model.pop("type", "mono"))
    return microphone_array(kind, **model)


def _optional_position(value: Sequence[float] | None, label: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return tuple(_floorplan_position(value, label))


def _make_agent_room(
    room: Room | Sequence[float] | Mapping[str, Any],
    *,
    shape: str,
    materials: Mapping[str, str | Material] | None,
    material_profile: str | Mapping[str, Any] | None,
    material_seed: int,
) -> Room:
    if isinstance(room, Room):
        return room
    if not isinstance(room, Mapping):
        return make_room(
            shape,
            size=room,
            materials=materials,
            material_profile=material_profile,
            material_seed=material_seed,
        )

    spec = dict(room)
    room_shape = str(spec.pop("shape", shape))
    size = spec.pop("size", (6.0, 4.0, 2.8))
    room_materials = spec.pop("materials", materials)
    room_material_profile = spec.pop("material_profile", material_profile)
    room_material_seed = int(spec.pop("material_seed", material_seed))
    explicit_corners = spec.pop("corners", None)
    acoustic_geometry = spec.pop("acoustic_geometry", spec.pop("objects", None))
    room_metadata = spec.pop("metadata", None)
    corners = explicit_corners if explicit_corners is not None else _parametric_corners(room_shape, size, spec)
    result = make_room(
        room_shape,
        size=size,
        corners=corners,
        materials=room_materials,
        material_profile=room_material_profile,
        material_seed=room_material_seed,
    )
    if isinstance(result.metadata, dict):
        result.metadata["geometry_params"] = spec
        if isinstance(room_metadata, Mapping):
            result.metadata.update(dict(room_metadata))
    if acoustic_geometry is not None:
        result = _room_with_acoustic_geometry(result, acoustic_geometry)
    return result


def _room_with_acoustic_geometry(
    room: Room,
    acoustic_geometry: Sequence[Mapping[str, Any]],
) -> Room:
    objects = [_acoustic_object(item, index) for index, item in enumerate(acoustic_geometry)]
    library = MaterialLibrary.load()
    material_seed = int(room.metadata.get("material_seed", 0))
    for index, item in enumerate(objects):
        if item.get("semantic") == "small_objects_ignore":
            continue
        item["material_selection"] = material_summary(library.sample_geometry(item, seed=material_seed + index + 1))
    return Room(
        id=room.id,
        name=room.name,
        corners=room.corners,
        height_m=room.height_m,
        materials=room.materials,
        metadata={**dict(room.metadata), "objects": objects},
    )


def _resolve_furnishing(
    room_metadata: Mapping[str, Any],
    furnishing: str | Mapping[str, Any] | None,
    *,
    default_seed: int,
    exclude_points: Sequence[Sequence[float]],
    existing_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if furnishing is None or furnishing is False:
        return None
    if isinstance(furnishing, str):
        config: Mapping[str, Any] = {"compactness": furnishing}
    elif isinstance(furnishing, Mapping):
        config = furnishing
    else:
        raise TypeError("furnishing must be a compactness string or an object")
    mode = str(config.get("mode", "auto")).strip().lower()
    if mode in {"none", "off", "manual"}:
        return None
    if mode != "auto":
        raise ValueError("furnishing.mode must be auto, manual, or none")
    return generate_floorplan_furniture(
        room_metadata,
        compactness=str(config.get("compactness", config.get("density", "balanced"))),
        seed=int(config.get("seed", default_seed)),
        exclude_points=exclude_points,
        existing_objects=existing_objects,
    )


def _acoustic_object(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise TypeError(f"acoustic_geometry[{index}] must be an object")
    raw_size = item.get("size", (1.0, 1.0, 1.0))
    raw_position = item.get("position", (0.0, 0.0))
    if not isinstance(raw_size, Sequence) or isinstance(raw_size, (str, bytes)) or len(raw_size) != 3:
        raise ValueError(f"acoustic_geometry[{index}].size must contain width, depth, and height")
    if not isinstance(raw_position, Sequence) or isinstance(raw_position, (str, bytes)) or len(raw_position) < 2:
        raise ValueError(f"acoustic_geometry[{index}].position must contain x and y")
    size = [float(value) for value in raw_size]
    position = [float(raw_position[0]), float(raw_position[1])]
    rotation = float(item.get("rotation", item.get("rotation_deg", 0.0)))
    z = float(item.get("z", size[2] * 0.5))
    values = (*size, *position, rotation, z)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"acoustic_geometry[{index}] contains non-finite values")
    if min(size) <= 0.0:
        raise ValueError(f"acoustic_geometry[{index}].size values must be positive")
    output = {
        "id": str(item.get("id", f"object_{index}")),
        "type": str(item.get("type", "cuboid")),
        "semantic": str(item.get("semantic", item.get("type", "furniture"))),
        "title": str(item.get("title", item.get("type", "Acoustic object"))),
        "absorption_class": str(item.get("absorption_class", item.get("absorption_level", "auto"))),
        "position": position,
        "rotation": rotation,
        "size": size,
        "z": z,
    }
    if item.get("material") or item.get("material_id"):
        output["material"] = str(item.get("material") or item.get("material_id"))
    if item.get("material_type"):
        output["material_type"] = str(item["material_type"])
    if isinstance(item.get("placement"), Mapping):
        output["placement"] = dict(item["placement"])
    return output


def _parametric_corners(
    shape: str,
    size: Sequence[float],
    params: Mapping[str, Any],
) -> list[tuple[float, float]] | None:
    width, depth = float(size[0]), float(size[1])
    key = shape.lower().replace("-", "_")
    clamp = lambda value, lo, hi: max(lo, min(hi, float(value)))

    if key in {"rectangle", "shoebox", "box"}:
        return None
    if key == "triangle":
        apex = clamp(params.get("apex", 0.5), 0.05, 0.95)
        return [(0.0, 0.0), (width, 0.0), (width * apex, depth)]
    if key == "circle":
        segments = max(12, int(round(float(params.get("segments", 36)))))
        return [
            (
                width * 0.5 + math.cos(2.0 * math.pi * index / segments) * width * 0.5,
                depth * 0.5 + math.sin(2.0 * math.pi * index / segments) * depth * 0.5,
            )
            for index in range(segments)
        ]
    if key == "polygon":
        sides = max(5, min(12, int(round(float(params.get("sides", 6))))))
        irregularity = clamp(params.get("irregularity", 0.18), 0.0, 0.35)
        skew = clamp(params.get("skew", 0.0), -0.3, 0.3)
        cx, cy = width * 0.5, depth * 0.5
        raw = []
        for index in range(sides):
            angle = -math.pi * 0.5 + math.pi * 2.0 * index / sides
            ripple = math.sin((index + 1) * 1.7) * 0.5 + math.cos((index + 2) * 2.3) * 0.5
            scale = 1.0 - irregularity * 0.5 + ripple * irregularity
            x = cx + math.cos(angle) * width * 0.47 * scale
            y = cy + math.sin(angle) * depth * 0.47 * scale
            raw.append((x + (y - cy) * skew, y))
        return _normalize_corners(raw, width, depth, 0.02)
    if key == "l_shape":
        cutout_w = clamp(params.get("cutout_width", 0.45), 0.15, 0.8)
        cutout_d = clamp(params.get("cutout_depth", 0.45), 0.15, 0.8)
        inner_x, inner_y = width * (1.0 - cutout_w), depth * (1.0 - cutout_d)
        return [(0.0, 0.0), (width, 0.0), (width, inner_y), (inner_x, inner_y), (inner_x, depth), (0.0, depth)]
    if key == "t_shape":
        stem_w = width * clamp(params.get("stem_width", 0.34), 0.18, 0.85)
        stem_x = (width - stem_w) * clamp(params.get("stem_offset", 0.5), 0.0, 1.0)
        head_d = depth * clamp(params.get("head_depth", 0.38), 0.15, 0.65)
        return [(0.0, 0.0), (width, 0.0), (width, head_d), (stem_x + stem_w, head_d), (stem_x + stem_w, depth), (stem_x, depth), (stem_x, head_d), (0.0, head_d)]
    if key == "trapezoid":
        top_w = width * clamp(params.get("top_width", 0.62), 0.2, 1.0)
        top_x = (width - top_w) * clamp(params.get("top_offset", 0.5), 0.0, 1.0)
        return [(0.0, 0.0), (width, 0.0), (top_x + top_w, depth), (top_x, depth)]
    if key == "u_shape":
        opening_w = width * clamp(params.get("opening_width", 0.42), 0.2, 0.72)
        opening_d = depth * clamp(params.get("opening_depth", 0.48), 0.18, 0.82)
        opening_x = (width - opening_w) * clamp(params.get("opening_offset", 0.5), 0.0, 1.0)
        left_x, right_x, inner_y = opening_x, opening_x + opening_w, depth - opening_d
        return [(0.0, 0.0), (width, 0.0), (width, depth), (right_x, depth), (right_x, inner_y), (left_x, inner_y), (left_x, depth), (0.0, depth)]
    if key == "fan_shape":
        angle = math.radians(clamp(params.get("angle_deg", 90.0), 45.0, 150.0)) * 0.5
        inner = clamp(params.get("inner_radius", 0.28), 0.05, 0.55)
        segments = max(8, min(48, int(round(float(params.get("segments", 24))))))
        outer = [(-math.sin(angle - 2.0 * angle * index / segments), math.cos(-angle + 2.0 * angle * index / segments)) for index in range(segments + 1)]
        inner_points = [(math.sin(angle - 2.0 * angle * index / segments) * inner, math.cos(angle - 2.0 * angle * index / segments) * inner) for index in range(segments + 1)]
        return _normalize_corners(outer + inner_points, width, depth, 0.02)
    return None


def _normalize_corners(
    corners: Sequence[Sequence[float]],
    width: float,
    depth: float,
    pad_ratio: float,
) -> list[tuple[float, float]]:
    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    x0, y0 = min(xs), min(ys)
    span_x, span_y = max(max(xs) - x0, 1e-6), max(max(ys) - y0, 1e-6)
    pad_x, pad_y = width * pad_ratio, depth * pad_ratio
    return [
        (
            pad_x + (float(x) - x0) / span_x * max(width - pad_x * 2.0, 0.1),
            pad_y + (float(y) - y0) / span_y * max(depth - pad_y * 2.0, 0.1),
        )
        for x, y in corners
    ]
