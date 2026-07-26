#!/usr/bin/env python3
"""Display aggregate progress for concurrently running Dist tiers."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from tqdm.auto import tqdm


def _finished_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    status = path.read_text(encoding="utf-8").strip()
    return status if status.lstrip("-").isdigit() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tiers", nargs="+", required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "localization", "beamforming"),
        default="all",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    stages = (
        ("localization", "beamforming")
        if args.stage == "all"
        else (args.stage,)
    )
    expected = [
        (tier, stage)
        for tier in args.tiers
        for stage in stages
    ]
    completed: set[tuple[str, str]] = set()
    with tqdm(
        total=len(expected),
        desc="Dist multi-GPU",
        unit="stage",
        dynamic_ncols=True,
    ) as bar:
        while True:
            for tier, stage in expected:
                key = (tier, stage)
                summary = args.output_root / f"dist-{tier}" / stage / "summary.json"
                if key not in completed and summary.is_file():
                    completed.add(key)
                    bar.update(1)
                    bar.set_postfix_str(f"{tier}/{stage}", refresh=True)
            statuses = [
                status
                for tier, stage in expected
                if (
                    status := _finished_status(
                        args.output_root / f"dist-{tier}" / f".status-{stage}"
                    )
                )
                is not None
            ]
            if len(statuses) == len(expected):
                if any(status != "0" for status in statuses):
                    raise SystemExit("one or more Dist tiers failed; inspect the logs")
                break
            time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    main()
