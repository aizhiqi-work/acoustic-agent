#!/usr/bin/env python3
"""Compile the acoustic-material database into a compact runtime resource."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3
FREQUENCIES = (125, 250, 500, 1000, 2000, 4000)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: Path, output: Path) -> None:
    data = source / "json"
    index = _read(source / "index.json")
    semantic_objects = _read(data / "semantic_objects.json")
    semantic_mappings = _read(data / "semantic_to_material.json")
    material_types = _read(data / "material_types.json")
    materials = _read(data / "materials.json")
    properties = {int(row["material_idx"]): row for row in _read(data / "acoustic_properties.json")}

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE semantic_objects (
                semantic_id TEXT PRIMARY KEY,
                name_zh TEXT NOT NULL,
                name_en TEXT NOT NULL,
                semantic_group TEXT NOT NULL,
                geometry_role TEXT NOT NULL,
                participation_policy TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                selection_enabled INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE material_types (
                material_type_id TEXT PRIMARY KEY,
                name_zh TEXT NOT NULL,
                description_zh TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE semantic_materials (
                semantic_id TEXT NOT NULL,
                material_type_id TEXT NOT NULL,
                priority INTEGER NOT NULL,
                is_default INTEGER NOT NULL,
                condition_code TEXT NOT NULL,
                allowed_classes TEXT NOT NULL,
                selection_note TEXT NOT NULL,
                PRIMARY KEY (semantic_id, material_type_id)
            ) WITHOUT ROWID;
            CREATE TABLE materials (
                material_idx INTEGER PRIMARY KEY,
                material_id TEXT NOT NULL UNIQUE,
                canonical_name TEXT NOT NULL,
                material_type_id TEXT NOT NULL,
                legacy_vlm_type TEXT NOT NULL,
                primary_category TEXT NOT NULL,
                absorption_class TEXT NOT NULL,
                coefficient_kind TEXT NOT NULL,
                a125 REAL NOT NULL,
                a250 REAL NOT NULL,
                a500 REAL NOT NULL,
                a1000 REAL NOT NULL,
                a2000 REAL NOT NULL,
                a4000 REAL NOT NULL,
                alpha_mid REAL NOT NULL,
                alpha_high REAL NOT NULL,
                source_group TEXT NOT NULL,
                confidence REAL NOT NULL,
                quality_status TEXT NOT NULL,
                is_selectable INTEGER NOT NULL
            );
            CREATE INDEX materials_family_class
                ON materials(material_type_id, absorption_class, coefficient_kind, is_selectable);
            """
        )
        metadata = {
            "runtime_schema_version": SCHEMA_VERSION,
            "source_schema_version": index.get("schema_version", "unknown"),
            "database_name": index.get("database_name", "acoustic_materials"),
            "source_index_sha256": _sha256(source / "index.json"),
            "frequencies_hz": list(FREQUENCIES),
            "material_count": len(materials),
            "semantic_count": len(semantic_objects),
            "material_type_count": len(material_types),
            "semantic_mapping_count": len(semantic_mappings),
            "selection_policy": "semantic -> material family -> absorption class -> six-band material",
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ((key, json.dumps(value, ensure_ascii=True, separators=(",", ":"))) for key, value in metadata.items()),
        )
        connection.executemany(
            """INSERT INTO semantic_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                (
                    row["semantic_object_id"], row.get("name_zh", ""), row.get("name_en", ""),
                    row.get("semantic_group", ""), row.get("geometry_role", ""),
                    row.get("participation_policy", ""),
                    json.dumps(row.get("aliases", []), ensure_ascii=True, separators=(",", ":")),
                    int(bool(row.get("selection_enabled", True))),
                )
                for row in semantic_objects
            ),
        )
        connection.executemany(
            "INSERT INTO material_types VALUES (?, ?, ?)",
            (
                (row["material_type_id"], row.get("name_zh", ""), row.get("description_zh", ""))
                for row in material_types
            ),
        )
        connection.executemany(
            "INSERT INTO semantic_materials VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    row["semantic_object_id"], row["material_type_id"], int(row.get("priority", 999)),
                    int(bool(row.get("is_default", False))), row.get("condition_code", ""),
                    "|".join(row.get("allowed_absorption_classes", [])), row.get("selection_note", ""),
                )
                for row in semantic_mappings
            ),
        )

        material_rows = []
        for row in materials:
            prop = properties[int(row["material_idx"])]
            material_rows.append((
                int(row["material_idx"]), row["material_id"], row.get("canonical_name", row["material_id"]),
                row["material_type_id"], row.get("legacy_vlm_type_norm", ""), row.get("primary_category_legacy", ""),
                prop["absorption_class"], prop["coefficient_kind"],
                *(float(prop[f"a{frequency}"]) for frequency in FREQUENCIES),
                float(prop["alpha_mid"]), float(prop["alpha_high"]), row.get("source_group", "unknown"),
                float(row.get("confidence", 0.0)), row.get("data_quality_status", "unknown"),
                int(bool(row.get("is_selectable", True))),
            ))
        connection.executemany(
            "INSERT INTO materials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            material_rows,
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="acoustic_material_db_v3 directory")
    parser.add_argument("output", type=Path, help="runtime SQLite output")
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
