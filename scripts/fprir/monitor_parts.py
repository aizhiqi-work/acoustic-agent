#!/usr/bin/env python3
"""Display aggregate tqdm progress for parallel FP-RIR generator partitions."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from tqdm.auto import tqdm


def _finished_status(part: Path) -> str | None:
    status_path = part / ".status"
    if not status_path.is_file():
        return None
    status = status_path.read_text(encoding="utf-8").strip()
    return status if status.lstrip("-").isdigit() else None


def _line_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_file():
            with path.open("rb") as handle:
                total += sum(1 for line in handle if line.strip())
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--part-count", type=int, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    parts = [args.parts_root / f"part-{rank:03d}" for rank in range(args.part_count)]
    while not all(
        (part / "plan.jsonl").is_file() or _finished_status(part) is not None
        for part in parts
    ):
        time.sleep(max(0.1, args.interval))
    total = _line_count([part / "plan.jsonl" for part in parts])
    with tqdm(total=total, desc="Adapt multi-GPU", unit="config", dynamic_ncols=True) as bar:
        current = 0
        while True:
            completed = _line_count(
                sorted(path for part in parts for path in (part / "shards").glob("*.jsonl"))
            )
            if completed > current:
                bar.update(min(completed, total) - current)
                current = min(completed, total)
            statuses = [status for part in parts if (status := _finished_status(part)) is not None]
            bar.set_postfix(active=args.part_count - len(statuses), finished=len(statuses), refresh=True)
            if len(statuses) == args.part_count:
                if current < total:
                    bar.update(total - current)
                if any(status != "0" for status in statuses):
                    raise SystemExit("one or more FP-RIR partitions failed; inspect the part logs")
                break
            time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    main()
