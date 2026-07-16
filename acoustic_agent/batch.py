from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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


def simulate_batch(
    room: Room,
    pairs: Sequence[SimulationPair],
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
    if not isinstance(pair, SimulationPair):
        raise TypeError(f"pairs[{index}] must be a SimulationPair")
    for label, point in (("source", pair.source), ("receiver", pair.receiver)):
        values = np.asarray(point, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"pairs[{index}].{label} must be a finite xyz point")
    return pair
