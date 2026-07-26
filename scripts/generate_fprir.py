#!/usr/bin/env python3
"""Generate the FP-RIR floorplan RIR dataset and its paper statistics.

The generator is deliberately self-contained:

* floorplan-disjoint train/validation/test splits;
* same-room mono, cross-room compact-array, and distributed-microphone items;
* an optional moving-source companion subset;
* compressed, resumable HDF5 shards with a JSONL index;
* a terminal progress bar with throughput and ETA;
* paper-ready SVG statistics and LaTeX/JSON summaries.

Run a representative pilot first:

    .venv/bin/python scripts/generate_fprir.py \
        --profile pilot --output benchmark-results/fprir-pilot

The full corpus is intentionally explicit because it is a long-running job:

    .venv/bin/python scripts/generate_fprir.py \
        --profile full --output /path/to/FP-RIR
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
from tqdm.auto import tqdm

from acoustic_agent import AcousticAgent
from acoustic_agent.floorplan_resource import FloorplanResource


DATASET_VERSION = "fprir_v1"
GENERATOR_REVISION = "2026-07-26.3"
MATERIAL_PROFILE = {
    "wall": "auto",
    "floor": "auto",
    "ceiling": "auto",
    "door": "auto",
    "window": "auto",
}
PILOT_ROOM_TARGETS = (4, 6, 8, 10, 12)
HABITABLE_TYPES = {
    "living",
    "bedroom",
    "kitchen",
    "dining",
    "study",
    "office",
    "children",
    "guest",
}


@dataclass(frozen=True)
class DatasetJob:
    item_id: str
    floorplan_idx: int
    split: str
    kind: str
    seed: int
    material_seed: int
    room_count: int
    source_room: str
    source_room_type: str
    source: tuple[float, float, float]
    receiver_rooms: tuple[str, ...]
    receiver_room_types: tuple[str, ...]
    receivers: tuple[tuple[float, float, float], ...]
    graph_distances: tuple[int, ...]
    euclidean_distances_m: tuple[float, ...]
    receiver_model: Mapping[str, Any]
    motion: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "floorplan_idx": self.floorplan_idx,
            "split": self.split,
            "kind": self.kind,
            "seed": self.seed,
            "material_seed": self.material_seed,
            "room_count": self.room_count,
            "source_room": self.source_room,
            "source_room_type": self.source_room_type,
            "source": list(self.source),
            "receiver_rooms": list(self.receiver_rooms),
            "receiver_room_types": list(self.receiver_room_types),
            "receivers": [list(point) for point in self.receivers],
            "graph_distances": list(self.graph_distances),
            "euclidean_distances_m": list(self.euclidean_distances_m),
            "receiver_model": dict(self.receiver_model),
            "motion": dict(self.motion) if self.motion else None,
        }


@dataclass
class GeneratedItem:
    job: DatasetJob
    rir: np.ndarray
    metadata: dict[str, Any]


class ProgressBar:
    """Small tqdm adapter retained to keep the generator call sites concise."""

    def __init__(self, total: int, label: str, *, unit: str = "item") -> None:
        self.total = max(0, int(total))
        self.current = 0
        self._bar = tqdm(
            total=self.total,
            desc=str(label),
            unit=unit,
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.1,
            bar_format=(
                "{desc:<10} {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )

    def update(self, amount: int = 1, *, detail: str = "") -> None:
        increment = max(0, min(int(amount), self.total - self.current))
        if detail:
            self._bar.set_postfix_str(detail[:48], refresh=False)
        self.current += increment
        self._bar.update(increment)

    def finish(self, *, detail: str = "") -> None:
        if detail:
            self._bar.set_postfix_str(detail[:48], refresh=False)
        if self.current < self.total:
            self.update(self.total - self.current)
        self._bar.close()


def _stable_int(*parts: Any) -> int:
    text = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big", signed=False)


def _split_for_floorplan(index: int, ratios: Sequence[float]) -> str:
    value = _stable_int(DATASET_VERSION, "split", index) / float(2**64)
    train_end = float(ratios[0])
    validation_end = train_end + float(ratios[1])
    if value < train_end:
        return "train"
    if value < validation_end:
        return "validation"
    return "test"


def _room_graph(record: Mapping[str, Any]) -> dict[str, set[str]]:
    graph = {str(room["id"]): set() for room in record.get("rooms", [])}
    for portal in record.get("portals", []):
        if not bool(portal.get("open", True)):
            continue
        room_ids = [str(value) for value in portal.get("room_ids", []) if str(value) in graph]
        for left_index, left in enumerate(room_ids):
            for right in room_ids[left_index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    return graph


def _graph_distances(graph: Mapping[str, set[str]], start: str) -> dict[str, int]:
    distances = {str(start): 0}
    queue: deque[str] = deque((str(start),))
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph.get(current, ())):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def _connected_components(graph: Mapping[str, set[str]]) -> list[set[str]]:
    remaining = set(graph)
    components: list[set[str]] = []
    while remaining:
        start = min(remaining)
        component = set(_graph_distances(graph, start))
        components.append(component)
        remaining -= component
    return sorted(components, key=lambda values: (-len(values), sorted(values)))


def _room_by_id(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(room["id"]): room for room in record.get("rooms", [])}


def _candidate_source_rooms(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rooms = [
        room
        for room in record.get("rooms", [])
        if len(room.get("corners", [])) >= 3 and str(room.get("type", "")).lower() in HABITABLE_TYPES
    ]
    if not rooms:
        rooms = [
            room
            for room in record.get("rooms", [])
            if len(room.get("corners", [])) >= 3 and str(room.get("type", "")).lower() != "balcony"
        ]
    if not rooms:
        rooms = [room for room in record.get("rooms", []) if len(room.get("corners", [])) >= 3]
    return sorted(rooms, key=lambda room: (str(room.get("type", "")), str(room.get("id", ""))))


def _choose_stable(values: Sequence[Any], *seed_parts: Any) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return values[_stable_int(*seed_parts) % len(values)]


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _placement(
    resource: FloorplanResource,
    index: int,
    source_room: str,
    receiver_room: str,
    seed: int,
) -> dict[str, Any]:
    return resource.sample_placement(
        index,
        placement="same_room" if source_room == receiver_room else "cross_room",
        seed=int(seed & 0x7FFFFFFF),
        source_room=source_room,
        receiver_room=receiver_room,
        height_m=1.4,
        wall_margin_m=0.3,
        min_distance_m=1.0,
    )


def _receiver_for_fixed_source(
    resource: FloorplanResource,
    index: int,
    source_room: str,
    receiver_room: str,
    source: Sequence[float],
    seed: int,
    *,
    min_distance_m: float = 1.0,
) -> tuple[float, float, float]:
    best_receiver: tuple[float, float, float] | None = None
    best_distance = -1.0
    for attempt in range(16):
        sampled = _placement(
            resource,
            index,
            source_room,
            receiver_room,
            seed + attempt * 997,
        )
        receiver = tuple(float(value) for value in sampled["receiver"])
        distance = _euclidean(source, receiver)
        if distance > best_distance:
            best_receiver = receiver
            best_distance = distance
        if distance >= float(min_distance_m):
            return receiver
    if best_receiver is None:
        raise ValueError(f"could not sample receiver in room {receiver_room!r}")
    return best_receiver


def _build_jobs_for_floorplan(
    resource: FloorplanResource,
    index: int,
    *,
    split: str,
    max_distributed_mics: int,
    include_motion: bool,
    motion_distance_m: float,
    motion_spacing_m: float,
) -> list[DatasetJob]:
    record = resource.record(index)
    room_lookup = _room_by_id(record)
    graph = _room_graph(record)
    sources = _candidate_source_rooms(record)
    if not sources:
        return []
    room_count = len(room_lookup)
    base_seed = _stable_int(DATASET_VERSION, "scene", index) & 0x7FFFFFFF
    material_seed = _stable_int(DATASET_VERSION, "material", index) & 0x7FFFFFFF
    jobs: list[DatasetJob] = []

    same_room = _choose_stable(sources, DATASET_VERSION, "same", index)
    same_id = str(same_room["id"])
    same = _placement(resource, index, same_id, same_id, base_seed + 11)
    same_source = tuple(float(value) for value in same["source"])
    same_receiver = tuple(float(value) for value in same["receiver"])
    jobs.append(
        DatasetJob(
            item_id=f"fp{index:05d}_same_mono",
            floorplan_idx=index,
            split=split,
            kind="same_room_mono",
            seed=base_seed + 11,
            material_seed=material_seed,
            room_count=room_count,
            source_room=same_id,
            source_room_type=str(same_room.get("type", "unknown")),
            source=same_source,
            receiver_rooms=(same_id,),
            receiver_room_types=(str(same_room.get("type", "unknown")),),
            receivers=(same_receiver,),
            graph_distances=(0,),
            euclidean_distances_m=(_euclidean(same_source, same_receiver),),
            receiver_model={"type": "mono"},
        )
    )

    cross_pairs: list[tuple[int, str, str]] = []
    for source_room in sources:
        source_id = str(source_room["id"])
        distances = _graph_distances(graph, source_id)
        for receiver_id, distance in distances.items():
            if distance > 0:
                cross_pairs.append((distance, source_id, receiver_id))
    if cross_pairs:
        available_distances = sorted({distance for distance, _, _ in cross_pairs})
        target_distance = _choose_stable(
            available_distances,
            DATASET_VERSION,
            "cross-distance",
            index,
        )
        candidates = sorted(
            (pair for pair in cross_pairs if pair[0] == target_distance),
            key=lambda pair: (pair[1], pair[2]),
        )
        graph_distance, cross_source_room, cross_receiver_room = _choose_stable(
            candidates,
            DATASET_VERSION,
            "cross-pair",
            index,
        )
        cross = _placement(
            resource,
            index,
            cross_source_room,
            cross_receiver_room,
            base_seed + 23,
        )
        cross_source = tuple(float(value) for value in cross["source"])
        cross_receiver = tuple(float(value) for value in cross["receiver"])
        jobs.append(
            DatasetJob(
                item_id=f"fp{index:05d}_cross_circular4",
                floorplan_idx=index,
                split=split,
                kind="cross_room_circular4",
                seed=base_seed + 23,
                material_seed=material_seed,
                room_count=room_count,
                source_room=cross_source_room,
                source_room_type=str(room_lookup[cross_source_room].get("type", "unknown")),
                source=cross_source,
                receiver_rooms=(cross_receiver_room,),
                receiver_room_types=(str(room_lookup[cross_receiver_room].get("type", "unknown")),),
                receivers=(cross_receiver,),
                graph_distances=(int(graph_distance),),
                euclidean_distances_m=(_euclidean(cross_source, cross_receiver),),
                receiver_model={"type": "circular", "count": 4, "radius_m": 0.04},
            )
        )

    reachability = {
        str(room["id"]): len(_graph_distances(graph, str(room["id"])))
        for room in sources
    }
    maximum_reachability = max(reachability.values(), default=0)
    distributed_source_candidates = [
        room
        for room in sources
        if reachability[str(room["id"])] == maximum_reachability
    ]
    distributed_source = _choose_stable(
        distributed_source_candidates,
        DATASET_VERSION,
        "distributed-source",
        index,
    )
    distributed_source_id = str(distributed_source["id"])
    distances = _graph_distances(graph, distributed_source_id)
    if len(distances) >= 2:
        source_placement = _placement(
            resource,
            index,
            distributed_source_id,
            distributed_source_id,
            base_seed + 37,
        )
        distributed_source_point = tuple(float(value) for value in source_placement["source"])
        room_ids = sorted(
            distances,
            key=lambda room_id: (
                0 if room_id == distributed_source_id else 1,
                distances[room_id],
                _stable_int(DATASET_VERSION, "distributed-room", index, room_id),
            ),
        )[: max(2, int(max_distributed_mics))]
        receivers = []
        for mic_index, receiver_room in enumerate(room_ids):
            if receiver_room == distributed_source_id:
                receiver = tuple(float(value) for value in source_placement["receiver"])
            else:
                receiver = _receiver_for_fixed_source(
                    resource,
                    index,
                    distributed_source_id,
                    receiver_room,
                    distributed_source_point,
                    base_seed + 41 + mic_index,
                )
            receivers.append(receiver)
        jobs.append(
            DatasetJob(
                item_id=f"fp{index:05d}_distributed{len(receivers)}",
                floorplan_idx=index,
                split=split,
                kind="distributed_mono",
                seed=base_seed + 37,
                material_seed=material_seed,
                room_count=room_count,
                source_room=distributed_source_id,
                source_room_type=str(distributed_source.get("type", "unknown")),
                source=distributed_source_point,
                receiver_rooms=tuple(room_ids),
                receiver_room_types=tuple(
                    str(room_lookup[room_id].get("type", "unknown")) for room_id in room_ids
                ),
                receivers=tuple(receivers),
                graph_distances=tuple(int(distances[room_id]) for room_id in room_ids),
                euclidean_distances_m=tuple(
                    _euclidean(distributed_source_point, receiver) for receiver in receivers
                ),
                receiver_model={"type": "distributed_mono", "count": len(receivers)},
            )
        )

    if include_motion:
        jobs.append(
            DatasetJob(
                item_id=f"fp{index:05d}_moving_source",
                floorplan_idx=index,
                split=split,
                kind="moving_source_mono",
                seed=base_seed + 53,
                material_seed=material_seed,
                room_count=room_count,
                source_room=same_id,
                source_room_type=str(same_room.get("type", "unknown")),
                source=same_source,
                receiver_rooms=(same_id,),
                receiver_room_types=(str(same_room.get("type", "unknown")),),
                receivers=(same_receiver,),
                graph_distances=(0,),
                euclidean_distances_m=(_euclidean(same_source, same_receiver),),
                receiver_model={"type": "mono"},
                motion={
                    "mode": "random",
                    "moving": "source",
                    "distance_m": float(motion_distance_m),
                    "keyframe_spacing_m": float(motion_spacing_m),
                    "seed": base_seed + 53,
                },
            )
        )
    return jobs


def _material_metadata(agent: AcousticAgent) -> dict[str, Any]:
    return {
        str(semantic): {
            "material_id": material.id,
            "name": material.name,
            "source": material.source,
            "absorption": {str(key): float(value) for key, value in material.absorption.items()},
            "scattering": {str(key): float(value) for key, value in material.scattering.items()},
            "transmission_loss_db": {
                str(key): float(value) for key, value in material.transmission_loss_db.items()
            },
        }
        for semantic, material in agent.room.materials.items()
    }


def _new_agent(
    job: DatasetJob,
    receiver: Sequence[float],
    receiver_room: str,
    receiver_model: Mapping[str, Any],
    *,
    quality: str,
    duration_s: float,
    fs: int,
    intersection_backend: str,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> AcousticAgent:
    agent = AcousticAgent.from_floorplan(
        idx=job.floorplan_idx,
        placement="same_room" if job.source_room == receiver_room else "cross_room",
        seed=job.seed,
        material_seed=job.material_seed,
        material_profile=MATERIAL_PROFILE,
        source=job.source,
        receiver=receiver,
        source_room=job.source_room,
        receiver_room=receiver_room,
        receiver_model=receiver_model,
        source_model={"type": "omni"},
        quality=quality,
        duration_s=duration_s,
        fs=fs,
        visualization=False,
        intersection_backend=intersection_backend,
    )
    agent.config = replace(
        agent.config,
        rt_accelerator=str(rt_accelerator),
        rt_precision=str(rt_precision),
        rt_cuda_device=int(rt_cuda_device),
    )
    return agent


def _rt60_value(result: Any) -> float:
    return float(result.rt60.get("rir_rt60_s", result.rt60.get("rt60_s", 0.0)))


def _result_metadata(result: Any) -> dict[str, Any]:
    return {
        "rt60_s": _rt60_value(result),
        "rt60_bands": {
            str(key): float(value)
            for key, value in result.rt60.get(
                "rir_rt60_bands",
                result.rt60.get("rt60_bands", {}),
            ).items()
        },
        "path_count": len(result.paths),
    }


def _generate_item(
    job: DatasetJob,
    *,
    quality: str,
    duration_s: float,
    fs: int,
    intersection_backend: str,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
) -> GeneratedItem:
    started = time.perf_counter()
    common = {
        **job.as_dict(),
        "dataset_version": DATASET_VERSION,
        "generator_revision": GENERATOR_REVISION,
        "quality": quality,
        "duration_s": float(duration_s),
        "fs": int(fs),
        "intersection_backend": intersection_backend,
        "rt_accelerator": rt_accelerator,
        "rt_precision": rt_precision,
        "rt_cuda_device": int(rt_cuda_device),
        "material_profile": dict(MATERIAL_PROFILE),
    }

    if job.kind == "distributed_mono":
        channels = []
        channel_metrics = []
        material_metadata = None
        for receiver, receiver_room in zip(job.receivers, job.receiver_rooms):
            agent = _new_agent(
                job,
                receiver,
                receiver_room,
                {"type": "mono"},
                quality=quality,
                duration_s=duration_s,
                fs=fs,
                intersection_backend=intersection_backend,
                rt_accelerator=rt_accelerator,
                rt_precision=rt_precision,
                rt_cuda_device=rt_cuda_device,
            )
            result = agent.run()
            channels.append(np.asarray(result.rir[0], dtype=np.float32))
            channel_metrics.append(_result_metadata(result))
            if material_metadata is None:
                material_metadata = _material_metadata(agent)
        rir = np.stack(channels, axis=0).astype(np.float32)
        rt_values = [item["rt60_s"] for item in channel_metrics if item["rt60_s"] > 0]
        metadata = {
            **common,
            "channel_count": int(rir.shape[0]),
            "frame_count": 1,
            "channel_metrics": channel_metrics,
            "rt60_s": float(np.mean(rt_values)) if rt_values else 0.0,
            "materials": material_metadata or {},
        }
    else:
        receiver_model = dict(job.receiver_model)
        agent = _new_agent(
            job,
            job.receivers[0],
            job.receiver_rooms[0],
            receiver_model,
            quality=quality,
            duration_s=duration_s,
            fs=fs,
            intersection_backend=intersection_backend,
            rt_accelerator=rt_accelerator,
            rt_precision=rt_precision,
            rt_cuda_device=rt_cuda_device,
        )
        if job.kind == "moving_source_mono":
            motion = agent.sample_motion(**dict(job.motion or {}))
            dynamic = agent.run_dynamic(motion)
            rir = np.stack([np.asarray(frame.rir, dtype=np.float32) for frame in dynamic.frames], axis=0)
            frame_metrics = [_result_metadata(frame) for frame in dynamic.frames]
            rt_values = [item["rt60_s"] for item in frame_metrics if item["rt60_s"] > 0]
            metadata = {
                **common,
                "motion": dict(dynamic.motion),
                "channel_count": int(rir.shape[1]),
                "frame_count": int(rir.shape[0]),
                "frame_metrics": frame_metrics,
                "rt60_s": float(np.mean(rt_values)) if rt_values else 0.0,
                "materials": _material_metadata(agent),
            }
        else:
            result = agent.run()
            rir = np.asarray(result.rir, dtype=np.float32)
            metrics = _result_metadata(result)
            metadata = {
                **common,
                **metrics,
                "channel_count": int(rir.shape[0]),
                "frame_count": 1,
                "materials": _material_metadata(agent),
            }
    metadata["generation_seconds"] = round(time.perf_counter() - started, 6)
    metadata["rir_shape"] = list(rir.shape)
    return GeneratedItem(job=job, rir=rir, metadata=metadata)


def _write_shard(
    path: Path,
    index_path: Path,
    items: Sequence[GeneratedItem],
    *,
    compression: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary_index = index_path.with_suffix(index_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    temporary_index.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle, temporary_index.open("w", encoding="utf-8") as index_file:
        handle.attrs["dataset"] = "FP-RIR"
        handle.attrs["version"] = DATASET_VERSION
        handle.attrs["generator_revision"] = GENERATOR_REVISION
        for item in items:
            group = handle.create_group(item.job.item_id)
            group.create_dataset(
                "rir",
                data=item.rir,
                dtype=np.float32,
                compression="gzip",
                compression_opts=int(compression),
                shuffle=True,
            )
            metadata_json = json.dumps(item.metadata, ensure_ascii=True, separators=(",", ":"))
            group.attrs["metadata_json"] = metadata_json
            index_file.write(
                json.dumps(
                    {
                        "shard": path.name,
                        "group": item.job.item_id,
                        **item.metadata,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)
    os.replace(temporary_index, index_path)


def _load_index(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _shard_index_is_complete(
    records: Sequence[Mapping[str, Any]],
    jobs: Sequence[DatasetJob],
) -> bool:
    expected = {job.item_id for job in jobs}
    actual = {str(record.get("group", "")) for record in records}
    return len(records) == len(jobs) and actual == expected


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(size))
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _run_jobs(
    jobs: Sequence[DatasetJob],
    output_dir: Path,
    *,
    quality: str,
    duration_s: float,
    fs: int,
    intersection_backend: str,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
    shard_size: int,
    workers: int,
    compression: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    progress = ProgressBar(len(jobs), "Generate", unit="config")
    error_path = output_dir / "errors.jsonl"
    if error_path.exists():
        errors.extend(_load_index(error_path))
    for shard_index, shard_jobs in enumerate(_chunks(jobs, shard_size)):
        shard_path = shard_dir / f"fprir-{shard_index:05d}.h5"
        index_path = shard_dir / f"fprir-{shard_index:05d}.jsonl"
        if shard_path.is_file() and index_path.is_file():
            existing = _load_index(index_path)
            if _shard_index_is_complete(existing, shard_jobs):
                all_records.extend(existing)
                progress.update(len(shard_jobs), detail=f"resume shard {shard_index:05d}")
                continue
            progress.update(0, detail=f"rebuild incomplete shard {shard_index:05d}")
        shard_item_ids = {job.item_id for job in shard_jobs}
        errors = [
            error
            for error in errors
            if str(error.get("item_id", "")) not in shard_item_ids
        ]
        generated: list[GeneratedItem] = []
        if int(workers) <= 1:
            for job in shard_jobs:
                try:
                    generated.append(
                        _generate_item(
                            job,
                            quality=quality,
                            duration_s=duration_s,
                            fs=fs,
                            intersection_backend=intersection_backend,
                            rt_accelerator=rt_accelerator,
                            rt_precision=rt_precision,
                            rt_cuda_device=rt_cuda_device,
                        )
                    )
                except Exception as exc:  # Batch generation must preserve other valid items.
                    errors.append(
                        {
                            "item_id": job.item_id,
                            "floorplan_idx": job.floorplan_idx,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                progress.update(1, detail=job.item_id)
        else:
            with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="fprir") as pool:
                futures: dict[Future[GeneratedItem], DatasetJob] = {
                    pool.submit(
                        _generate_item,
                        job,
                        quality=quality,
                        duration_s=duration_s,
                        fs=fs,
                        intersection_backend=intersection_backend,
                        rt_accelerator=rt_accelerator,
                        rt_precision=rt_precision,
                        rt_cuda_device=rt_cuda_device,
                    ): job
                    for job in shard_jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        generated.append(future.result())
                    except Exception as exc:
                        errors.append(
                            {
                                "item_id": job.item_id,
                                "floorplan_idx": job.floorplan_idx,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    progress.update(1, detail=job.item_id)
        generated.sort(key=lambda item: item.job.item_id)
        _write_shard(shard_path, index_path, generated, compression=compression)
        all_records.extend(_load_index(index_path))
        with error_path.open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, ensure_ascii=True, separators=(",", ":")) + "\n")
    progress.finish()
    return all_records, errors


def _scan_resource(
    resource: FloorplanResource,
    ratios: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    room_types: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    progress = ProgressBar(len(resource), "Scan", unit="scene")
    for index in range(len(resource)):
        record = resource.record(index)
        rooms = record.get("rooms", [])
        graph = _room_graph(record)
        components = _connected_components(graph)
        split = _split_for_floorplan(index, ratios)
        split_counts[split] += 1
        room_types.update(str(room.get("type", "unknown")) for room in rooms)
        rows.append(
            {
                "idx": index,
                "room_count": len(rooms),
                "portal_count": sum(bool(portal.get("open", True)) for portal in record.get("portals", [])),
                "component_count": len(components),
                "largest_component": max((len(component) for component in components), default=0),
                "net_area_m2": float(record.get("net_area_m2") or 0.0),
                "split": split,
            }
        )
        progress.update(1)
    progress.finish()
    summary = {
        "floorplans": len(rows),
        "rooms": int(sum(row["room_count"] for row in rows)),
        "open_portals": int(sum(row["portal_count"] for row in rows)),
        "split_floorplans": dict(split_counts),
        "room_types": dict(room_types.most_common()),
        "room_count": _describe([row["room_count"] for row in rows]),
        "net_area_m2": _describe(
            [
                row["net_area_m2"]
                for row in rows
                if 5.0 <= float(row["net_area_m2"]) <= 1000.0
            ]
        ),
    }
    return rows, summary


def _select_floorplans(rows: Sequence[Mapping[str, Any]], maximum: int | None) -> list[int]:
    if maximum is None or maximum <= 0 or maximum >= len(rows):
        return [int(row["idx"]) for row in rows]
    target_total = int(maximum)
    buckets: dict[int, list[tuple[int, int]]] = {target: [] for target in PILOT_ROOM_TARGETS}
    for row in rows:
        room_count = int(row["room_count"])
        target = min(PILOT_ROOM_TARGETS, key=lambda value: (abs(value - room_count), value))
        index = int(row["idx"])
        buckets[target].append((_stable_int(DATASET_VERSION, "pilot", index), index))
    for values in buckets.values():
        values.sort()
    selected: list[int] = []
    cursor = {target: 0 for target in PILOT_ROOM_TARGETS}
    while len(selected) < target_total:
        progressed = False
        for target in PILOT_ROOM_TARGETS:
            position = cursor[target]
            if position >= len(buckets[target]):
                continue
            selected.append(buckets[target][position][1])
            cursor[target] += 1
            progressed = True
            if len(selected) >= target_total:
                break
        if not progressed:
            break
    return sorted(selected)


def _describe(values: Sequence[float | int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _summarize(
    resource_summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    selected_floorplans: Sequence[int],
    fs: int,
    duration_s: float,
    quality: str,
) -> dict[str, Any]:
    static = [record for record in records if record.get("kind") != "moving_source_mono"]
    motion = [record for record in records if record.get("kind") == "moving_source_mono"]
    channels = [int(record.get("channel_count", 0)) for record in static]
    graph_distances = [
        int(value)
        for record in static
        for value in record.get("graph_distances", [])
    ]
    euclidean_distances = [
        float(value)
        for record in static
        for value in record.get("euclidean_distances_m", [])
    ]
    rt60_values = [float(record.get("rt60_s", 0.0)) for record in static if float(record.get("rt60_s", 0.0)) > 0]
    kind_counts = Counter(str(record.get("kind")) for record in records)
    split_configurations = Counter(str(record.get("split")) for record in records)
    floorplan_split = {
        int(record["floorplan_idx"]): str(record.get("split"))
        for record in records
    }
    split_floorplans = Counter(floorplan_split.values())
    static_rir_channels = sum(int(record.get("channel_count", 0)) for record in static)
    motion_keyframes = sum(int(record.get("frame_count", 0)) for record in motion)
    return {
        "dataset": "FP-RIR",
        "version": DATASET_VERSION,
        "generator_revision": GENERATOR_REVISION,
        "scope": "full" if len(selected_floorplans) == int(resource_summary["floorplans"]) else "pilot",
        "resource": dict(resource_summary),
        "generated": {
            "selected_floorplans": len(selected_floorplans),
            "completed_floorplans": len(floorplan_split),
            "split_floorplans": dict(split_floorplans),
            "configurations": len(records),
            "split_configurations": dict(split_configurations),
            "configuration_types": dict(kind_counts),
            "static_configurations": len(static),
            "static_rir_channels": int(static_rir_channels),
            "moving_source_trajectories": len(motion),
            "moving_source_rir_keyframes": int(motion_keyframes),
            "microphones_per_static_configuration": _describe(channels),
            "room_graph_distance": _describe(graph_distances),
            "euclidean_distance_m": _describe(euclidean_distances),
            "rt60_s": _describe(rt60_values),
            "failed_configurations": len(errors),
            "sample_rate_hz": int(fs),
            "rir_duration_s": float(duration_s),
            "quality": quality,
            "storage_bytes": _directory_size(output_dir / "shards"),
            "storage_scope": "HDF5 shards and JSONL shard indices",
        },
    }


def _nice_ticks(maximum: float, count: int = 4) -> list[float]:
    if maximum <= 0:
        return [0.0, 1.0]
    raw = maximum / max(1, count)
    exponent = 10 ** math.floor(math.log10(raw))
    fraction = raw / exponent
    nice_fraction = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    step = nice_fraction * exponent
    upper = math.ceil(maximum / step) * step
    return [step * index for index in range(int(round(upper / step)) + 1)]


def _format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k" if value % 1000 == 0 else f"{value / 1000:.1f}k"
    if abs(value) >= 10 or float(value).is_integer():
        return f"{value:.0f}"
    if abs(value) < 1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _histogram(
    values: Sequence[float | int],
    *,
    discrete: bool,
    bins: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([0.0]), np.asarray([-0.5, 0.5])
    if discrete:
        minimum = int(math.floor(float(np.min(array))))
        maximum = int(math.ceil(float(np.max(array))))
        edges = np.arange(minimum - 0.5, maximum + 1.5, 1.0)
        counts, edges = np.histogram(array, bins=edges)
        return counts.astype(float), edges
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if math.isclose(minimum, maximum):
        minimum = max(0.0, minimum - 0.5)
        maximum += 0.5
    counts, edges = np.histogram(array, bins=min(int(bins), max(5, int(math.sqrt(array.size)))))
    return counts.astype(float), edges


def _plot_panel(
    x: float,
    y: float,
    width: float,
    height: float,
    values: Sequence[float | int],
    *,
    label: str,
    title: str,
    x_label: str,
    color: str,
    discrete: bool,
) -> str:
    counts, edges = _histogram(values, discrete=discrete)
    margin_left, margin_right, margin_top, margin_bottom = 58.0, 18.0, 55.0, 52.0
    plot_x = x + margin_left
    plot_y = y + margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    maximum = max(float(np.max(counts)), 1.0)
    ticks = _nice_ticks(maximum, 4)
    y_max = max(ticks[-1], 1.0)
    parts = [
        f'<g aria-label="{html.escape(title)}">',
        f'<text x="{x + 4:.1f}" y="{y + 20:.1f}" class="panel-label">({label})</text>',
        f'<text x="{x + 38:.1f}" y="{y + 20:.1f}" class="panel-title">{html.escape(title)}</text>',
        f'<text x="{x + width - 18:.1f}" y="{y + 20:.1f}" text-anchor="end" class="sample">n = {len(values):,}</text>',
    ]
    for tick in ticks:
        tick_y = plot_y + plot_h - (tick / y_max) * plot_h
        parts.append(
            f'<line x1="{plot_x:.1f}" y1="{tick_y:.1f}" x2="{plot_x + plot_w:.1f}" '
            f'y2="{tick_y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{plot_x - 9:.1f}" y="{tick_y + 4:.1f}" text-anchor="end" class="tick">'
            f'{html.escape(_format_tick(tick))}</text>'
        )
    bar_width = plot_w / max(len(counts), 1)
    for index, count in enumerate(counts):
        bar_height = (float(count) / y_max) * plot_h
        parts.append(
            f'<rect x="{plot_x + index * bar_width + 1:.1f}" '
            f'y="{plot_y + plot_h - bar_height:.1f}" '
            f'width="{max(1.0, bar_width - 2):.1f}" height="{bar_height:.1f}" '
            f'fill="{color}" rx="1"/>'
        )
    parts.extend(
        [
            f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" '
            f'x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}" class="axis"/>',
            f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" '
            f'x2="{plot_x:.1f}" y2="{plot_y + plot_h:.1f}" class="axis"/>',
        ]
    )
    x_tick_indices = (
        list(range(len(counts)))
        if discrete and len(counts) <= 8
        else sorted({0, max(0, len(counts) // 2), max(0, len(counts) - 1)})
    )
    for index in x_tick_indices:
        center = 0.5 * (edges[index] + edges[index + 1])
        tick_x = plot_x + (index + 0.5) * bar_width
        parts.append(
            f'<text x="{tick_x:.1f}" y="{plot_y + plot_h + 20:.1f}" '
            f'text-anchor="middle" class="tick">{html.escape(_format_tick(center))}</text>'
        )
    parts.extend(
        [
            f'<text x="{plot_x + plot_w / 2:.1f}" y="{y + height - 8:.1f}" '
            f'text-anchor="middle" class="axis-label">{html.escape(x_label)}</text>',
            f'<text x="{x + 13:.1f}" y="{plot_y + plot_h / 2:.1f}" '
            f'text-anchor="middle" transform="rotate(-90 {x + 13:.1f} {plot_y + plot_h / 2:.1f})" '
            f'class="axis-label">Count</text>',
            "</g>",
        ]
    )
    return "".join(parts)


def _write_statistics_svg(
    path: Path,
    scan_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    static = [record for record in records if record.get("kind") != "moving_source_mono"]
    rooms = [int(row["room_count"]) for row in scan_rows]
    graph = [int(value) for record in static for value in record.get("graph_distances", [])]
    distance = [float(value) for record in static for value in record.get("euclidean_distances_m", [])]
    rt60 = [float(record.get("rt60_s", 0.0)) for record in static if float(record.get("rt60_s", 0.0)) > 0]
    width, height = 1800, 470
    outer = 28
    gap = 22
    panel_width = (width - 2 * outer - 3 * gap) / 4
    panels = [
        _plot_panel(
            outer,
            42,
            panel_width,
            390,
            rooms,
            label="a",
            title="Floorplan complexity",
            x_label="Rooms per floorplan",
            color="#2878B5",
            discrete=True,
        ),
        _plot_panel(
            outer + panel_width + gap,
            42,
            panel_width,
            390,
            graph,
            label="b",
            title="Topological separation",
            x_label="Room-graph distance",
            color="#E07A1F",
            discrete=True,
        ),
        _plot_panel(
            outer + 2 * (panel_width + gap),
            42,
            panel_width,
            390,
            distance,
            label="c",
            title="Metric separation",
            x_label="Source-microphone distance (m)",
            color="#238B75",
            discrete=False,
        ),
        _plot_panel(
            outer + 3 * (panel_width + gap),
            42,
            panel_width,
            390,
            rt60,
            label="d",
            title="Reverberation",
            x_label="Broadband RT60 (s)",
            color="#7651A8",
            discrete=False,
        ),
    ]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #1b2430; letter-spacing: 0; }}
.panel-label {{ font-size: 19px; font-weight: 700; }}
.panel-title {{ font-size: 18px; font-weight: 700; }}
.sample {{ font-size: 13px; fill: #5d6875; }}
.axis-label {{ font-size: 14px; font-weight: 600; }}
.tick {{ font-size: 12px; fill: #4f5965; }}
.axis {{ stroke: #28323d; stroke-width: 1.4; }}
.grid {{ stroke: #dfe4e8; stroke-width: 1; }}
</style>
{''.join(panels)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _render_statistics_asset(svg_path: Path, output_path: Path) -> bool:
    executable = shutil.which("magick")
    if not executable:
        return False
    completed = subprocess.run(
        [
            executable,
            "-background",
            "white",
            "-density",
            "160",
            str(svg_path),
            "-alpha",
            "remove",
            "-alpha",
            "off",
            str(output_path),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0 and output_path.is_file()


def _write_latex_summary(path: Path, summary: Mapping[str, Any]) -> None:
    generated = summary["generated"]
    split = generated["split_floorplans"]
    microphones = generated["microphones_per_static_configuration"]
    storage_gib = float(generated["storage_bytes"]) / (1024**3)
    text = rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\caption{{Overview of the generated FP-RIR {summary['scope']} dataset.}}
\label{{tab:fprir_overview}}
\begin{{tabularx}}{{\columnwidth}}{{@{{}}lX@{{}}}}
\toprule
\textbf{{Property}} & \textbf{{Coverage}} \\
\midrule
Residential floorplans & {generated['completed_floorplans']:,} \\
Data split & Floorplan-disjoint train/validation/test:
{split.get('train', 0):,}/{split.get('validation', 0):,}/{split.get('test', 0):,} \\
Static configurations & {generated['static_configurations']:,} \\
Static RIR channels & {generated['static_rir_channels']:,} \\
Moving-source sequences & {generated['moving_source_trajectories']:,} \\
Moving-source RIR keyframes & {generated['moving_source_rir_keyframes']:,} \\
Microphones per configuration &
{microphones['mean']:.2f} average ({microphones['min']:.0f}--{microphones['max']:.0f}) \\
Sampling configuration &
{generated['sample_rate_hz'] / 1000:g} kHz, {generated['rir_duration_s']:g} s \\
Storage & {storage_gib:.3f} GiB \\
\bottomrule
\end{{tabularx}}
\end{{table}}
"""
    path.write_text(text, encoding="utf-8")


def _write_manifest(
    output_dir: Path,
    jobs: Sequence[DatasetJob],
    *,
    profile: str,
    quality: str,
    fs: int,
    duration_s: float,
    ratios: Sequence[float],
    selected_floorplans: Sequence[int],
    intersection_backend: str,
    configuration_set: str,
    rt_accelerator: str,
    rt_precision: str,
    rt_cuda_device: int,
    nested_tier_sizes: Sequence[int],
    partition_count: int,
    partition_rank: int,
    global_floorplans: int,
) -> dict[str, Any]:
    manifest = {
        "dataset": "FP-RIR",
        "version": DATASET_VERSION,
        "generator_revision": GENERATOR_REVISION,
        "profile": profile,
        "quality": quality,
        "sample_rate_hz": int(fs),
        "rir_duration_s": float(duration_s),
        "split_ratios": {
            "train": float(ratios[0]),
            "validation": float(ratios[1]),
            "test": float(ratios[2]),
        },
        "floorplan_disjoint": True,
        "selected_floorplans": len(selected_floorplans),
        "global_selected_floorplans": int(global_floorplans),
        "planned_configurations": len(jobs),
        "partition": {
            "count": int(partition_count),
            "rank": int(partition_rank),
        },
        "intersection_backend": intersection_backend,
        "configuration_set": configuration_set,
        "rt_accelerator": rt_accelerator,
        "rt_precision": rt_precision,
        "rt_cuda_device": int(rt_cuda_device),
        "nested_tier_sizes": sorted(
            {int(value) for value in nested_tier_sizes if int(value) > 0}
        ),
        "material_profile": dict(MATERIAL_PROFILE),
        "storage": {
            "rir": "gzip-compressed float32 HDF5 shards",
            "index": "JSONL sidecar per shard",
        },
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(
                f"{manifest_path} belongs to a different generation configuration; "
                "choose a new output directory"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (output_dir / "plan.jsonl").open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job.as_dict(), ensure_ascii=True, separators=(",", ":")) + "\n")
    return manifest


def _write_plan_summary(
    path: Path,
    jobs: Sequence[DatasetJob],
    resource_summary: Mapping[str, Any],
) -> dict[str, Any]:
    static = [job for job in jobs if job.kind != "moving_source_mono"]
    motion = [job for job in jobs if job.kind == "moving_source_mono"]
    floorplan_split = {job.floorplan_idx: job.split for job in jobs}
    nominal_motion_frames = [
        max(
            3,
            min(
                65,
                int(
                    math.ceil(
                        float((job.motion or {}).get("distance_m", 0.0))
                        / max(float((job.motion or {}).get("keyframe_spacing_m", 0.25)), 0.05)
                    )
                )
                + 1,
            ),
        )
        for job in motion
    ]
    summary = {
        "dataset": "FP-RIR",
        "version": DATASET_VERSION,
        "generator_revision": GENERATOR_REVISION,
        "status": "planned",
        "resource": dict(resource_summary),
        "plan": {
            "floorplans": len(floorplan_split),
            "split_floorplans": dict(Counter(floorplan_split.values())),
            "configurations": len(jobs),
            "configuration_types": dict(Counter(job.kind for job in jobs)),
            "static_configurations": len(static),
            "static_rir_channels": int(
                sum(
                    int(job.receiver_model.get("count", 1))
                    if job.kind in {"cross_room_circular4", "distributed_mono"}
                    else 1
                    for job in static
                )
            ),
            "moving_source_trajectories": len(motion),
            "nominal_moving_source_keyframes": int(sum(nominal_motion_frames)),
            "microphones_per_static_configuration": _describe(
                [
                    int(job.receiver_model.get("count", 1))
                    if job.kind in {"cross_room_circular4", "distributed_mono"}
                    else 1
                    for job in static
                ]
            ),
            "room_graph_distance": _describe(
                [distance for job in static for distance in job.graph_distances]
            ),
            "euclidean_distance_m": _describe(
                [distance for job in static for distance in job.euclidean_distances_m]
            ),
        },
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def _tier_label(size: int) -> str:
    if int(size) >= 1000 and int(size) % 1000 == 0:
        return f"adapt-{int(size) // 1000}k"
    return f"adapt-{int(size)}"


def _write_nested_tier_indices(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    scan_rows: Sequence[Mapping[str, Any]],
    sizes: Sequence[int],
) -> dict[str, Any]:
    """Write lightweight nested views that all reference the same HDF5 shards."""

    tier_dir = output_dir / "tiers"
    tier_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for requested_size in sorted({int(value) for value in sizes if int(value) > 0}):
        selected = set(_select_floorplans(scan_rows, requested_size))
        tier_records = [
            dict(record)
            for record in records
            if int(record["floorplan_idx"]) in selected
        ]
        label = _tier_label(requested_size)
        index_name = f"{label}.jsonl"
        index_path = tier_dir / index_name
        with index_path.open("w", encoding="utf-8") as handle:
            for record in tier_records:
                handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        static = [record for record in tier_records if record.get("kind") != "moving_source_mono"]
        motion = [record for record in tier_records if record.get("kind") == "moving_source_mono"]
        summaries[label] = {
            "requested_floorplans": requested_size,
            "completed_floorplans": len(
                {int(record["floorplan_idx"]) for record in tier_records}
            ),
            "configurations": len(tier_records),
            "configuration_types": dict(
                Counter(str(record.get("kind")) for record in tier_records)
            ),
            "static_rir_channels": int(
                sum(int(record.get("channel_count", 0)) for record in static)
            ),
            "moving_source_trajectories": len(motion),
            "moving_source_rir_keyframes": int(
                sum(int(record.get("frame_count", 0)) for record in motion)
            ),
            "index": f"tiers/{index_name}",
            "shards": "shared with the parent corpus",
        }
    summary = {
        "nested": True,
        "selection": "deterministic room-count-stratified prefix",
        "tiers": summaries,
    }
    (tier_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_ratios(value: str) -> tuple[float, float, float]:
    parts = tuple(float(part.strip()) for part in str(value).split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts) or not math.isclose(sum(parts), 1.0):
        raise argparse.ArgumentTypeError("split ratios must contain three positive values summing to 1")
    return parts


def _partition_floorplans(
    selected_floorplans: Sequence[int],
    partition_count: int,
    partition_rank: int,
) -> list[int]:
    count = max(1, int(partition_count))
    rank = int(partition_rank)
    if not 0 <= rank < count:
        raise ValueError("partition rank must be in [0, partition count)")
    return list(selected_floorplans[rank::count])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-floorplans", type=int, default=None)
    parser.add_argument("--quality", choices=("preview", "simulation", "fine", "reference"), default=None)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--split-ratios", type=_parse_ratios, default=(0.8, 0.1, 0.1))
    parser.add_argument("--max-distributed-mics", type=int, default=4)
    parser.add_argument("--motion-fraction", type=float, default=None)
    parser.add_argument("--motion-distance-m", type=float, default=1.0)
    parser.add_argument("--motion-spacing-m", type=float, default=0.25)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compression", type=int, choices=range(0, 10), default=4)
    parser.add_argument("--intersection-backend", choices=("auto", "bvh", "linear"), default="auto")
    parser.add_argument(
        "--configuration-set",
        choices=("mixed", "adapt"),
        default="mixed",
        help="adapt omits distributed benchmark configurations from the far-field corpus",
    )
    parser.add_argument("--rt-accelerator", choices=("numba", "cuda", "auto"), default="numba")
    parser.add_argument("--rt-precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--rt-cuda-device", type=int, default=0)
    parser.add_argument(
        "--nested-tier-sizes",
        nargs="*",
        type=int,
        default=(),
        help="write nested JSONL views for these floorplan counts without duplicating HDF5 data",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="scan, split, and sample all metadata without running the RIR solver",
    )
    parser.add_argument(
        "--partition-count",
        type=int,
        default=1,
        help="split the selected FloorPlans into this many deterministic process partitions",
    )
    parser.add_argument(
        "--partition-rank",
        type=int,
        default=0,
        help="zero-based partition generated by this process",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = str(args.profile)
    if args.rt_accelerator == "cuda" and args.rt_precision != "float32":
        raise SystemExit("CUDA FP-RIR generation requires --rt-precision float32")
    quality = str(args.quality or ("preview" if profile == "pilot" else "simulation"))
    duration_s = float(args.duration_s or (1.0 if profile == "pilot" else 2.0))
    maximum = args.max_floorplans
    if maximum is None and profile == "pilot":
        maximum = 15
    motion_fraction = args.motion_fraction
    if motion_fraction is None:
        motion_fraction = 0.2 if profile == "pilot" else 0.1
    motion_fraction = float(np.clip(float(motion_fraction), 0.0, 1.0))
    output_dir = (
        Path(args.output)
        if args.output is not None
        else Path("benchmark-results") / ("fprir-pilot" if profile == "pilot" else "FP-RIR")
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_count = max(1, int(args.partition_count))
    partition_rank = int(args.partition_rank)
    if not 0 <= partition_rank < partition_count:
        raise SystemExit("--partition-rank must be in [0, --partition-count)")

    resource = FloorplanResource()
    scan_rows, resource_summary = _scan_resource(resource, args.split_ratios)
    global_selected_floorplans = _select_floorplans(scan_rows, maximum)
    selected_floorplans = _partition_floorplans(
        global_selected_floorplans,
        partition_count,
        partition_rank,
    )
    jobs: list[DatasetJob] = []
    plan_progress = ProgressBar(len(selected_floorplans), "Plan", unit="scene")
    for index in selected_floorplans:
        split = _split_for_floorplan(index, args.split_ratios)
        include_motion = (
            _stable_int(DATASET_VERSION, "motion", index) / float(2**64)
        ) < motion_fraction
        jobs.extend(
            _build_jobs_for_floorplan(
                resource,
                index,
                split=split,
                max_distributed_mics=max(2, int(args.max_distributed_mics)),
                include_motion=include_motion,
                motion_distance_m=float(args.motion_distance_m),
                motion_spacing_m=float(args.motion_spacing_m),
            )
        )
        plan_progress.update(1, detail=f"floorplan {index}")
    plan_progress.finish()
    if args.configuration_set == "adapt":
        jobs = [
            job
            for job in jobs
            if job.kind in {"same_room_mono", "cross_room_circular4", "moving_source_mono"}
        ]

    _write_manifest(
        output_dir,
        jobs,
        profile=profile,
        quality=quality,
        fs=int(args.fs),
        duration_s=duration_s,
        ratios=args.split_ratios,
        selected_floorplans=selected_floorplans,
        intersection_backend=str(args.intersection_backend),
        configuration_set=str(args.configuration_set),
        rt_accelerator=str(args.rt_accelerator),
        rt_precision=str(args.rt_precision),
        rt_cuda_device=int(args.rt_cuda_device),
        nested_tier_sizes=tuple(int(value) for value in args.nested_tier_sizes),
        partition_count=partition_count,
        partition_rank=partition_rank,
        global_floorplans=len(global_selected_floorplans),
    )
    (output_dir / "resource-statistics.json").write_text(
        json.dumps(resource_summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"FP-RIR {profile}: {len(selected_floorplans):,}/{len(global_selected_floorplans):,} floorplans, "
        f"{len(jobs):,} planned configurations, {quality}, {args.fs} Hz, {duration_s:g} s"
    )
    if args.plan_only:
        plan_summary = _write_plan_summary(
            output_dir / "plan-summary.json",
            jobs,
            resource_summary,
        )
        planned = plan_summary["plan"]
        print(
            "Plan only: "
            f"{planned['static_rir_channels']:,} static RIR channels, "
            f"{planned['moving_source_trajectories']:,} motion trajectories, "
            f"{planned['nominal_moving_source_keyframes']:,} nominal motion keyframes"
        )
        print(f"Output: {output_dir}")
        return
    records, errors = _run_jobs(
        jobs,
        output_dir,
        quality=quality,
        duration_s=duration_s,
        fs=int(args.fs),
        intersection_backend=str(args.intersection_backend),
        rt_accelerator=str(args.rt_accelerator),
        rt_precision=str(args.rt_precision),
        rt_cuda_device=int(args.rt_cuda_device),
        shard_size=int(args.shard_size),
        workers=max(1, int(args.workers)),
        compression=int(args.compression),
    )
    charts = ProgressBar(5, "Summarize", unit="step")
    summary = _summarize(
        resource_summary,
        records,
        errors,
        output_dir,
        selected_floorplans=selected_floorplans,
        fs=int(args.fs),
        duration_s=duration_s,
        quality=quality,
    )
    if args.nested_tier_sizes:
        summary["nested_tiers"] = _write_nested_tier_indices(
            output_dir,
            records,
            scan_rows,
            args.nested_tier_sizes,
        )
    summary_path = output_dir / "fprir-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    charts.update(1, detail="summary JSON")
    svg_path = output_dir / "fprir-statistics.svg"
    _write_statistics_svg(svg_path, scan_rows, records)
    charts.update(1, detail="statistics SVG")
    rendered_png = _render_statistics_asset(
        svg_path,
        output_dir / "fprir-statistics.png",
    )
    charts.update(1, detail="statistics PNG" if rendered_png else "PNG renderer unavailable")
    rendered_pdf = _render_statistics_asset(
        svg_path,
        output_dir / "fprir-statistics.pdf",
    )
    charts.update(1, detail="statistics PDF" if rendered_pdf else "PDF renderer unavailable")
    _write_latex_summary(output_dir / "fprir-overview.tex", summary)
    charts.update(1, detail="LaTeX table")
    charts.finish()

    generated = summary["generated"]
    print(f"Output: {output_dir}")
    print(
        "Completed: "
        f"{generated['configurations']:,} configurations, "
        f"{generated['static_rir_channels']:,} static RIR channels, "
        f"{generated['moving_source_rir_keyframes']:,} motion keyframes, "
        f"{generated['failed_configurations']:,} failures"
    )
    print(f"Statistics: {svg_path}")
    if errors:
        raise SystemExit(
            f"FP-RIR generation completed with {len(errors)} failed configuration(s); "
            f"inspect {output_dir / 'errors.jsonl'}"
        )


if __name__ == "__main__":
    main()
