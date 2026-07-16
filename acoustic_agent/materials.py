from __future__ import annotations

import json
import random
from threading import Lock
from pathlib import Path
from typing import ClassVar, Iterable, Mapping

from .models import FREQUENCY_BANDS, Material, band_constant


RESOURCE_DIR = Path(__file__).resolve().parent / "resources" / "acoustic_materials"


class MaterialLibrary:
    _load_cache: ClassVar[dict[tuple[type["MaterialLibrary"], str], "MaterialLibrary"]] = {}
    _load_cache_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        records: list[dict],
        *,
        object_candidates: list[dict] | None = None,
        semantic_map: Mapping[str, dict] | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.records = records
        self.object_candidates = object_candidates or []
        self.semantic_map = dict(semantic_map or {})
        self.source_path = source_path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MaterialLibrary":
        root = Path(path) if path is not None else RESOURCE_DIR
        cache_key = (cls, str(root.expanduser().resolve()))
        with cls._load_cache_lock:
            cached = cls._load_cache.get(cache_key)
            if cached is not None:
                return cached

        if root.is_dir():
            records = _read_jsonl(root / "materials.jsonl")
            object_candidates = _read_jsonl(root / "object_material_candidates.jsonl")
            semantic_map = _read_json(root / "semantic_object_map.json")
            library = cls(records, object_candidates=object_candidates, semantic_map=semantic_map, source_path=root)
        else:
            library = cls(_read_jsonl(root), source_path=root if root.exists() else None)

        with cls._load_cache_lock:
            return cls._load_cache.setdefault(cache_key, library)

    def sample(self, semantic: str, *, absorption_level: str | Iterable[str] | None = None, seed: int = 0) -> Material:
        candidates = self._material_candidates(semantic)
        candidates = _filter_absorption(candidates, absorption_level)
        if not candidates:
            return fallback_material(semantic)
        rng = random.Random(seed + _stable_hash(f"material:{semantic}"))
        return _material_from_row(rng.choice(candidates), semantic)

    def sample_object(self, semantic_object: str, *, seed: int = 0) -> Material:
        canonical = _canonical_object(semantic_object, self.semantic_map)
        object_rows = [
            row for row in self.object_candidates
            if str(row.get("semantic_object", "")).lower() == canonical
        ]
        if object_rows:
            rng = random.Random(seed + _stable_hash(f"object:{canonical}"))
            return _material_from_row(rng.choice(object_rows), canonical)
        return self.sample(canonical, seed=seed)

    def _material_candidates(self, semantic: str) -> list[dict]:
        terms = _semantic_terms(semantic, self.semantic_map)
        return [
            row for row in self.records
            if str(row.get("id", "")).lower() in terms
            or str(row.get("material_id", "")).lower() in terms
            or str(row.get("canonical_name", "")).lower() in terms
            or str(row.get("primary_category", "")).lower() in terms
            or str(row.get("material_type_norm", "")).lower() in terms
            or any(str(alias).lower() in terms for alias in row.get("aliases", []))
        ]


def fallback_material(semantic: str) -> Material:
    alpha = {
        "wall": 0.06,
        "floor": 0.10,
        "ceiling": 0.08,
        "door": 0.14,
        "carpet": 0.45,
        "curtain": 0.45,
        "sofa": 0.55,
        "window": 0.08,
    }.get(semantic, 0.20)
    return Material(
        id=f"fallback_{semantic}",
        name=f"fallback {semantic}",
        semantic=semantic,
        absorption=band_constant(alpha),
        scattering=band_constant(0.12),
        transmission_loss_db=_default_transmission_loss(semantic),
        source="fallback",
    )


def _material_from_row(row: Mapping[str, object], semantic: str) -> Material:
    acoustic_model = row.get("acoustic_model") if isinstance(row.get("acoustic_model"), dict) else {}
    absorption = row.get("absorption") or row.get("absorption_coefficients") or acoustic_model.get("absorption")
    scattering = row.get("scattering") or acoustic_model.get("scattering")
    transmission_loss = row.get("transmission_loss_db") or acoustic_model.get("transmission_loss_db")
    material_id = str(row.get("material_id") or row.get("id") or row.get("candidate_material_id") or semantic)
    name = str(row.get("canonical_name") or row.get("name") or row.get("candidate_name") or material_id)
    return Material(
        id=material_id,
        name=name,
        semantic=semantic,
        absorption=_band_table(absorption, 0.2),
        scattering=_band_table(scattering, 0.12),
        transmission_loss_db=(
            _band_table(transmission_loss, 30.0)
            if isinstance(transmission_loss, Mapping)
            else _default_transmission_loss(semantic)
        ),
        source="acoustic_materials",
    )


def _default_transmission_loss(semantic: str) -> dict[str, float]:
    key = semantic.lower()
    base = {
        "wall": (38.0, 42.0, 46.0, 50.0, 52.0, 54.0),
        "floor": (42.0, 46.0, 50.0, 54.0, 56.0, 58.0),
        "ceiling": (35.0, 39.0, 43.0, 47.0, 49.0, 51.0),
        "window": (22.0, 25.0, 28.0, 31.0, 33.0, 35.0),
        "door": (20.0, 23.0, 26.0, 29.0, 31.0, 33.0),
        "curtain": (6.0, 8.0, 10.0, 12.0, 14.0, 16.0),
    }.get(key, (30.0, 33.0, 36.0, 39.0, 41.0, 43.0))
    return {band: float(value) for band, value in zip(FREQUENCY_BANDS, base)}


def _band_table(raw: object, default: float) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return band_constant(default)
    return {band: float(raw.get(band, raw.get(float(band), default))) for band in FREQUENCY_BANDS}


def _canonical_object(semantic: str, semantic_map: Mapping[str, dict]) -> str:
    term = semantic.lower()
    if term in semantic_map:
        return term
    aliases = {
        "walls": "wall",
        "hardwood": "floor",
        "tile": "floor",
        "carpet": "floor",
        "roof": "ceiling",
    }
    for key, meta in semantic_map.items():
        if term in [str(value).lower() for value in meta.get("vlm_aliases", [])]:
            return key
    return aliases.get(term, term)


def _semantic_terms(semantic: str, semantic_map: Mapping[str, dict]) -> set[str]:
    canonical = _canonical_object(semantic, semantic_map)
    terms = {semantic.lower(), canonical}
    if canonical in semantic_map:
        meta = semantic_map[canonical]
        terms.add(str(meta.get("vlm_type_norm", "")).lower())
        terms.update(str(alias).lower() for alias in meta.get("vlm_aliases", []))
    return {term for term in terms if term}


def _filter_absorption(rows: list[dict], level: str | Iterable[str] | None) -> list[dict]:
    if level is None:
        return rows
    levels = {level} if isinstance(level, str) else set(level)
    filtered = [row for row in rows if row.get("absorption_level") in levels]
    return filtered or rows


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _stable_hash(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value))
