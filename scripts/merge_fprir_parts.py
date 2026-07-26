#!/usr/bin/env python3
"""Merge deterministic FP-RIR process partitions without copying HDF5 tensors."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acoustic_agent.floorplan_resource import FloorplanResource
from scripts.generate_fprir import (
    DATASET_VERSION,
    GENERATOR_REVISION,
    MATERIAL_PROFILE,
    ProgressBar,
    _render_statistics_asset,
    _scan_resource,
    _summarize,
    _write_latex_summary,
    _write_nested_tier_indices,
    _write_statistics_svg,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _link_or_copy(source: Path, target: Path) -> str:
    if target.exists():
        if os.path.samefile(source, target):
            return "hardlink"
        raise RuntimeError(f"merge target already exists with different content: {target}")
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def merge_parts(
    output: Path,
    part_dirs: list[Path],
    nested_sizes: list[int],
    physical_gpu_ids: list[str] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    plans: dict[str, dict[str, Any]] = {}
    manifests = []
    link_modes: Counter[str] = Counter()

    for rank, part in enumerate(part_dirs):
        manifest_path = part / "manifest.json"
        summary_path = part / "fprir-summary.json"
        if not manifest_path.is_file() or not summary_path.is_file():
            raise RuntimeError(f"incomplete FP-RIR partition: {part}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        if manifest.get("version") != DATASET_VERSION or manifest.get("generator_revision") != GENERATOR_REVISION:
            raise RuntimeError(f"incompatible generator revision in {manifest_path}")
        for plan in _read_jsonl(part / "plan.jsonl"):
            plans[str(plan["item_id"])] = plan
        errors.extend(_read_jsonl(part / "errors.jsonl"))
        for index_path in sorted((part / "shards").glob("*.jsonl")):
            source_h5 = index_path.with_suffix(".h5")
            if not source_h5.is_file():
                raise RuntimeError(f"missing HDF5 shard for {index_path}")
            merged_name = f"part-{rank:03d}-{source_h5.name}"
            link_modes[_link_or_copy(source_h5, shard_dir / merged_name)] += 1
            merged_records = _read_jsonl(index_path)
            for record in merged_records:
                record["shard"] = merged_name
            records.extend(merged_records)
            merged_index = shard_dir / f"part-{rank:03d}-{index_path.name}"
            merged_index.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
                    for record in merged_records
                ),
                encoding="utf-8",
            )

    item_ids = [str(record["group"]) for record in records]
    if len(item_ids) != len(set(item_ids)):
        raise RuntimeError("partition merge produced duplicate FP-RIR item IDs")
    records.sort(key=lambda record: (int(record["floorplan_idx"]), str(record["group"])))
    plan_rows = sorted(plans.values(), key=lambda row: (int(row["floorplan_idx"]), str(row["item_id"])))
    (output / "plan.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            for row in plan_rows
        ),
        encoding="utf-8",
    )
    (output / "errors.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            for row in errors
        ),
        encoding="utf-8",
    )

    first = manifests[0]
    selected_floorplans = sorted({int(row["floorplan_idx"]) for row in plan_rows})
    manifest = {
        key: value
        for key, value in first.items()
        if key not in {"selected_floorplans", "planned_configurations", "partition", "rt_cuda_device"}
    }
    manifest.update(
        {
            "selected_floorplans": len(selected_floorplans),
            "planned_configurations": len(plan_rows),
            "partition": {
                "count": len(part_dirs),
                "merged": True,
                "storage_links": dict(link_modes),
            },
            "rt_cuda_devices": [0 for _ in part_dirs],
            "physical_gpu_ids": list(physical_gpu_ids or []),
            "nested_tier_sizes": sorted({int(value) for value in nested_sizes if int(value) > 0}),
            "material_profile": dict(MATERIAL_PROFILE),
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    scan_rows, resource_summary = _scan_resource(
        FloorplanResource(),
        (
            float(first["split_ratios"]["train"]),
            float(first["split_ratios"]["validation"]),
            float(first["split_ratios"]["test"]),
        ),
    )
    (output / "resource-statistics.json").write_text(
        json.dumps(resource_summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    charts = ProgressBar(5, "Merge", unit="step")
    summary = _summarize(
        resource_summary,
        records,
        errors,
        output,
        selected_floorplans=selected_floorplans,
        fs=int(first["sample_rate_hz"]),
        duration_s=float(first["rir_duration_s"]),
        quality=str(first["quality"]),
    )
    if nested_sizes:
        summary["nested_tiers"] = _write_nested_tier_indices(
            output,
            records,
            scan_rows,
            nested_sizes,
        )
    (output / "fprir-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    charts.update(1, detail="summary JSON")
    svg_path = output / "fprir-statistics.svg"
    _write_statistics_svg(svg_path, scan_rows, records)
    charts.update(1, detail="statistics SVG")
    png_ok = _render_statistics_asset(svg_path, output / "fprir-statistics.png")
    charts.update(1, detail="statistics PNG" if png_ok else "PNG renderer unavailable")
    pdf_ok = _render_statistics_asset(svg_path, output / "fprir-statistics.pdf")
    charts.update(1, detail="statistics PDF" if pdf_ok else "PDF renderer unavailable")
    _write_latex_summary(output / "fprir-overview.tex", summary)
    charts.update(1, detail="LaTeX table")
    charts.finish()
    if errors:
        raise RuntimeError(f"{len(errors)} FP-RIR configurations failed across partitions")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--physical-gpu-ids", nargs="*", default=())
    parser.add_argument("--nested-tier-sizes", type=int, nargs="*", default=())
    args = parser.parse_args()
    summary = merge_parts(
        args.output.expanduser().resolve(),
        [path.expanduser().resolve() for path in args.parts],
        list(args.nested_tier_sizes),
        list(args.physical_gpu_ids),
    )
    generated = summary["generated"]
    print(
        f"Merged {generated['configurations']:,} configurations and "
        f"{generated['static_rir_channels']:,} static RIR channels into {args.output}"
    )


if __name__ == "__main__":
    main()
