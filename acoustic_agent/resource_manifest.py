from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import h5py


PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
MANIFEST_PATH = RESOURCE_ROOT / "manifest.json"


def load_resource_manifest() -> dict[str, Any]:
    """Load the packaged resource inventory."""

    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_packaged_resources(*, hashes: bool = False) -> dict[str, Any]:
    """Validate all required runtime resources and optionally their SHA-256 hashes."""

    manifest = load_resource_manifest()
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in manifest.get("resources", []):
        path = RESOURCE_ROOT / str(item["path"])
        check = {
            "id": str(item["id"]),
            "kind": str(item["kind"]),
            "path": str(path),
            "ok": True,
        }
        try:
            _verify_file(path, item, hashes=hashes, check=check)
        except Exception as exc:
            check["ok"] = False
            check["error"] = str(exc)
            errors.append(f"{item['id']}: {exc}")
        checks.append(check)
    return {
        "ok": not errors,
        "hashes_checked": bool(hashes),
        "resource_root": str(RESOURCE_ROOT),
        "checks": checks,
        "errors": errors,
    }


def format_resource_report(report: dict[str, Any]) -> str:
    lines = [f"Packaged resources: {report['resource_root']}"]
    for check in report["checks"]:
        marker = "OK" if check["ok"] else "FAIL"
        detail = check.get("summary") or check.get("error", "")
        lines.append(f"[{marker}] {check['id']} ({check['kind']}): {detail}")
    if report["hashes_checked"]:
        lines.append("SHA-256 verification: enabled")
    lines.append("Resource verification passed." if report["ok"] else "Resource verification failed.")
    return "\n".join(lines)


def _verify_file(
    path: Path,
    item: dict[str, Any],
    *,
    hashes: bool,
    check: dict[str, Any],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing packaged file: {path}")
    size = path.stat().st_size
    expected_size = int(item["size_bytes"])
    if size != expected_size:
        raise ValueError(f"size mismatch: expected {expected_size}, got {size}")
    check["size_bytes"] = size

    if hashes:
        digest = _sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: expected {item['sha256']}, got {digest}")
        check["sha256"] = digest

    kind = str(item["kind"])
    if kind == "sofa_hrtf":
        check["summary"] = _verify_sofa(path)
    elif kind == "material_database":
        check["summary"] = _verify_sqlite(path, {"metadata", "materials", "semantic_materials"})
    elif kind == "scene_database":
        check["summary"] = _verify_sqlite(path, {"metadata", "scenes"})
    else:
        check["summary"] = f"{size} bytes"


def _verify_sofa(path: Path) -> str:
    with h5py.File(path, "r") as sofa:
        missing = {"Data.IR", "SourcePosition"}.difference(sofa.keys())
        if missing:
            raise ValueError(f"missing SOFA datasets: {', '.join(sorted(missing))}")
        ir_shape = tuple(int(value) for value in sofa["Data.IR"].shape)
        source_count = int(sofa["SourcePosition"].shape[0])
        database_name = sofa.attrs.get("DatabaseName", path.stem)
        if isinstance(database_name, bytes):
            database_name = database_name.decode("utf-8", errors="replace")
    return f"{database_name}, {source_count} directions, IR shape {ir_shape}"


def _verify_sqlite(path: Path, required_tables: set[str]) -> str:
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        missing = required_tables.difference(tables)
        if missing:
            raise ValueError(f"missing SQLite tables: {', '.join(sorted(missing))}")
        if "scenes" in tables:
            count = int(connection.execute("SELECT COUNT(*) FROM scenes").fetchone()[0])
            return f"{count} compiled scenes"
        count = int(connection.execute("SELECT COUNT(*) FROM materials").fetchone()[0])
        return f"{count} measured material records"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
