from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acoustic_agent.resplan import DEFAULT_RESPLAN_PATH, ResPlanDataset
from acoustic_agent.resplan_resource import DEFAULT_RESPLAN_RESOURCE, RESOURCE_SCHEMA_VERSION


def _static_record(scene: dict, *, source_index: int) -> dict:
    dataset = scene["dataset"]
    metadata = scene["room"]["metadata"]
    multi_room = metadata["multi_room"]
    return {
        "source_index": int(source_index),
        "sample_id": dataset.get("sample_id"),
        "unit_type": dataset.get("unit_type", "Unknown"),
        "net_area_m2": dataset.get("net_area_m2"),
        "gross_area_m2": dataset.get("gross_area_m2"),
        "meters_per_unit": dataset["meters_per_unit"],
        "scale_source": dataset["scale_source"],
        "wall_depth_m": dataset["wall_depth_m"],
        "height_m": scene["room"]["size"][2],
        "size": scene["room"]["size"][:2],
        "corners": scene["room"]["corners"],
        "rooms": multi_room["rooms"],
        "portals": multi_room["portals"],
        "features": metadata["boundary_features"],
        "surfaces": metadata["surface_segments"],
    }


def build_resource(source: Path, destination: Path) -> None:
    started = time.perf_counter()
    dataset = ResPlanDataset(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE scenes (idx INTEGER PRIMARY KEY, source_idx INTEGER UNIQUE NOT NULL, payload BLOB NOT NULL)"
        )
        metadata = {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "codec": "zlib_json",
            "source_record_count": len(dataset.records),
            "compiled_record_count": len(dataset.eligible_indices),
            "source_stats": dataset.stats(),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value, separators=(",", ":"), sort_keys=True)) for key, value in metadata.items()],
        )
        for resource_index, source_index in enumerate(dataset.eligible_indices):
            scene = dataset.scene(source_index)
            payload = json.dumps(
                _static_record(scene, source_index=source_index),
                separators=(",", ":"),
                ensure_ascii=True,
                sort_keys=True,
            ).encode("utf-8")
            connection.execute(
                "INSERT INTO scenes(idx, source_idx, payload) VALUES (?, ?, ?)",
                (resource_index, int(source_index), zlib.compress(payload, level=9)),
            )
            if resource_index % 500 == 0:
                connection.commit()
                elapsed = time.perf_counter() - started
                print(f"compiled {resource_index + 1}/{len(dataset.eligible_indices)} scenes in {elapsed:.1f}s", flush=True)
        connection.commit()
        connection.execute("CREATE INDEX scenes_source_idx ON scenes(source_idx)")
        connection.execute("VACUUM")
    finally:
        connection.close()
    temporary.replace(destination)
    elapsed = time.perf_counter() - started
    print(
        f"wrote {len(dataset.eligible_indices)} scenes to {destination} "
        f"({destination.stat().st_size / 1024 / 1024:.1f} MB) in {elapsed:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the filtered ResPlan pickle into a random-access resource.")
    parser.add_argument("--source", type=Path, default=DEFAULT_RESPLAN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESPLAN_RESOURCE)
    args = parser.parse_args()
    build_resource(args.source.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
