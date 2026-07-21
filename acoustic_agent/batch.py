from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .engine import SimulationResult, simulate_rir
from .models import Room, SimConfig


@dataclass(frozen=True)
class SimulationPair:
    source: tuple[float, float, float]
    receiver: tuple[float, float, float]
    id: str = ""
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any, index: int = 0) -> "SimulationPair":
        """Normalize a pair object, mapping, or ``(source, receiver)`` tuple."""
        if isinstance(value, cls):
            pair = value
        elif isinstance(value, Mapping):
            pair = cls(
                source=_point(value.get("source"), f"pairs[{index}].source"),
                receiver=_point(value.get("receiver", value.get("mic")), f"pairs[{index}].receiver"),
                id=str(value.get("id", "")),
                seed=int(value["seed"]) if value.get("seed") is not None else None,
                metadata=dict(value.get("metadata") or {}),
            )
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            pair = cls(
                source=_point(value[0], f"pairs[{index}].source"),
                receiver=_point(value[1], f"pairs[{index}].receiver"),
            )
        else:
            raise TypeError(
                f"pairs[{index}] must be a SimulationPair, a source/receiver mapping, "
                "or a (source, receiver) tuple"
            )
        return cls(
            source=_point(pair.source, f"pairs[{index}].source"),
            receiver=_point(pair.receiver, f"pairs[{index}].receiver"),
            id=str(pair.id),
            seed=int(pair.seed) if pair.seed is not None else None,
            metadata=dict(pair.metadata),
        )


@dataclass(frozen=True)
class BatchResult:
    items: tuple[SimulationResult, ...]
    pairs: tuple[SimulationPair, ...]
    metadata: Mapping[str, Any]

    @property
    def rirs(self) -> list[np.ndarray]:
        return [item.rir for item in self.items]

    def save_npz(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"rir_{index:06d}": np.asarray(item.rir, dtype=np.float32) for index, item in enumerate(self.items)}
        np.savez_compressed(
            destination,
            ids=np.asarray([pair.id or str(index) for index, pair in enumerate(self.pairs)], dtype=str),
            sources=np.asarray([pair.source for pair in self.pairs], dtype=np.float32),
            receivers=np.asarray([pair.receiver for pair in self.pairs], dtype=np.float32),
            **arrays,
        )
        return destination


@dataclass(frozen=True)
class ProductionResult:
    """Results from independent Geometry, Floorplan, or Custom scene jobs."""

    items: tuple[Any, ...]
    jobs: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]
    errors: tuple[Mapping[str, Any], ...] = ()

    @property
    def succeeded(self) -> int:
        return len(self.items)

    @property
    def failed(self) -> int:
        return len(self.errors)

    @property
    def rirs(self) -> list[Any]:
        output: list[Any] = []
        for item in self.items:
            if hasattr(item, "rir"):
                output.append(item.rir)
            else:
                output.append(tuple(item.rirs))
        return output

    def save_npz(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        records: list[dict[str, Any]] = []
        for index, (job, item) in enumerate(zip(self.jobs, self.items)):
            job_id = str(job.get("id", index))
            if hasattr(item, "rir"):
                key = f"rir_{index:06d}"
                arrays[key] = np.asarray(item.rir, dtype=np.float32)
                records.append({"id": job_id, "array": key, "dynamic": False})
                continue
            frame_keys = []
            for frame_index, rir in enumerate(item.rirs):
                key = f"rir_{index:06d}_frame_{frame_index:04d}"
                arrays[key] = np.asarray(rir, dtype=np.float32)
                frame_keys.append(key)
            records.append({"id": job_id, "arrays": frame_keys, "dynamic": True})
        manifest = {
            "model": "acoustic_agent_production_v1",
            "metadata": dict(self.metadata),
            "jobs": [dict(job) for job in self.jobs],
            "results": records,
            "errors": [dict(error) for error in self.errors],
        }
        np.savez_compressed(
            destination,
            manifest=np.asarray(json.dumps(_json_value(manifest), ensure_ascii=True)),
            **arrays,
        )
        return destination


def simulate_batch(
    room: Room,
    pairs: Sequence[Any],
    *,
    config: SimConfig | None = None,
    receiver_model: Mapping[str, Any] | None = None,
    source_model: str | Mapping[str, Any] | None = None,
    workers: int = 1,
) -> BatchResult:
    jobs = tuple(_validate_pair(pair, index) for index, pair in enumerate(pairs))
    base_config = config or SimConfig()

    def solve_one(index_pair: tuple[int, SimulationPair]) -> SimulationResult:
        index, pair = index_pair
        item_config = base_config if pair.seed is None else SimConfig(**{**base_config.__dict__, "seed": int(pair.seed)})
        if pair.seed is None and base_config.seed is not None:
            item_config = SimConfig(**{**base_config.__dict__, "seed": int(base_config.seed + index)})
        return simulate_rir(
            room,
            pair.source,
            pair.receiver,
            config=item_config,
            receiver_model=receiver_model,
            source_model=source_model,
        )

    indexed = tuple(enumerate(jobs))
    worker_count = max(1, int(workers))
    if worker_count == 1:
        items = tuple(solve_one(item) for item in indexed)
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="acoustic-agent-batch") as pool:
            items = tuple(pool.map(solve_one, indexed))
    return BatchResult(
        items=items,
        pairs=jobs,
        metadata={
            "model": "acoustic_agent_batch_v1",
            "room_id": room.id,
            "count": len(items),
            "workers": worker_count,
            "sample_rate": int(base_config.fs),
            "receiver_model": dict(receiver_model or {"type": "mono"}),
            "source_model": dict(items[0].source_model) if items else {},
        },
    )


def _validate_pair(pair: SimulationPair, index: int) -> SimulationPair:
    return SimulationPair.from_value(pair, index)


def _point(value: Any, label: str) -> tuple[float, float, float]:
    values = np.asarray(value, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be a finite xyz point")
    return tuple(float(item) for item in values)


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
