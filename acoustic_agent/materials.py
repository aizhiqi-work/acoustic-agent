from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Iterable, Mapping

from .models import FREQUENCY_BANDS, Material, band_constant


RESOURCE_DIR = Path(__file__).resolve().parent / "resources" / "acoustic_materials"
RUNTIME_RESOURCE = RESOURCE_DIR / "acoustic_materials_v3.sqlite3"
ABSORPTION_CLASSES = ("reflective", "semi_reflective", "absorptive", "highly_absorptive")
SURFACE_SEMANTICS = {
    "wall": "wall",
    "floor": "floor",
    "ceiling": "ceiling",
    "door": "door",
    "window": "window_glass",
}
SEMANTIC_INSTANCE_TAGS = {
    "wall": ("wall",),
    "floor": ("floor",),
    "ceiling": ("ceiling",),
    "door": ("door_panel", "window_glass"),
    "window_glass": ("window_glass",),
    "acoustic_treatment": ("acoustic_treatment",),
    "curtain_blind": ("curtain_blind", "soft_furniture"),
    "carpet_rug": ("floor", "curtain_blind"),
    "ceramic_tile_surface": ("floor", "wall"),
    "sofa_couch": ("soft_furniture",),
    "bed_mattress": ("soft_furniture",),
    "chair_seating": ("soft_furniture", "hard_furniture", "human_audience"),
    "table_desk_counter": ("hard_furniture",),
    "cabinet_shelf_wardrobe": ("hard_furniture",),
    "appliance": ("appliance_metal_ceramic", "hard_furniture"),
    "sanitary_fixture": ("appliance_metal_ceramic", "window_glass"),
    "screen_mirror": ("window_glass",),
    "human_person": ("human_audience",),
    "structural_element": ("wall", "hard_furniture", "appliance_metal_ceramic"),
}


class MaterialLibrary:
    """Deterministic semantic material sampler backed by the compact v3 resource."""

    _load_cache: ClassVar[dict[tuple[type["MaterialLibrary"], str], "MaterialLibrary"]] = {}
    _load_cache_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        records: list[dict],
        *,
        object_candidates: list[dict] | None = None,
        semantic_map: Mapping[str, dict] | None = None,
        source_path: Path | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
        semantic_mappings: Mapping[str, list[dict]] | None = None,
        material_types: Mapping[str, dict] | None = None,
    ) -> None:
        self.records = records
        self.object_candidates = object_candidates or []
        self.semantic_map = dict(semantic_map or {})
        self.source_path = source_path
        self.runtime_metadata = dict(runtime_metadata or {})
        self.semantic_mappings = {key: list(value) for key, value in (semantic_mappings or {}).items()}
        self.material_types = dict(material_types or {})
        self._by_id = {str(row.get("material_id") or row.get("id", "")): row for row in records}
        self._by_family_class: dict[tuple[str, str], list[dict]] = {}
        for row in records:
            family = str(row.get("material_type_id") or row.get("material_type_norm", ""))
            level = str(row.get("absorption_class") or row.get("absorption_level", ""))
            if family and level:
                self._by_family_class.setdefault((family, level), []).append(row)
        self._semantic_aliases: dict[str, str] = {}
        for semantic, meta in self.semantic_map.items():
            self._semantic_aliases[_normalized(semantic)] = semantic
            for alias in meta.get("aliases", meta.get("vlm_aliases", [])):
                self._semantic_aliases[_normalized(str(alias))] = semantic
        self._sample_cache: dict[tuple[str, str, str, int, bool], Material] = {}
        self._sample_cache_lock = Lock()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MaterialLibrary":
        root = Path(path) if path is not None else RESOURCE_DIR
        resolved = root.expanduser().resolve()
        cache_key = (cls, str(resolved))
        with cls._load_cache_lock:
            cached = cls._load_cache.get(cache_key)
            if cached is not None:
                return cached

        runtime_path = resolved / RUNTIME_RESOURCE.name if resolved.is_dir() else resolved
        if runtime_path.suffix in {".sqlite", ".sqlite3", ".db"} and runtime_path.is_file():
            library = cls._load_runtime(runtime_path)
        elif resolved.is_dir():
            records = _read_jsonl(resolved / "materials.jsonl")
            object_candidates = _read_jsonl(resolved / "object_material_candidates.jsonl")
            semantic_map = _read_json(resolved / "semantic_object_map.json")
            library = cls(records, object_candidates=object_candidates, semantic_map=semantic_map, source_path=resolved)
        else:
            library = cls(_read_jsonl(resolved), source_path=resolved if resolved.exists() else None)

        with cls._load_cache_lock:
            return cls._load_cache.setdefault(cache_key, library)

    @classmethod
    def _load_runtime(cls, path: Path) -> "MaterialLibrary":
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            metadata = {
                row["key"]: json.loads(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            semantics = {}
            for row in connection.execute("SELECT * FROM semantic_objects ORDER BY semantic_id"):
                semantics[row["semantic_id"]] = {
                    "semantic_object": row["semantic_id"],
                    "name_zh": row["name_zh"],
                    "name_en": row["name_en"],
                    "semantic_group": row["semantic_group"],
                    "geometry_role": row["geometry_role"],
                    "participation_policy": row["participation_policy"],
                    "aliases": json.loads(row["aliases_json"]),
                    "selection_enabled": bool(row["selection_enabled"]),
                }
            material_types = {
                row["material_type_id"]: dict(row)
                for row in connection.execute("SELECT * FROM material_types ORDER BY material_type_id")
            }
            mappings: dict[str, list[dict]] = {}
            for row in connection.execute("SELECT * FROM semantic_materials ORDER BY semantic_id, priority"):
                item = dict(row)
                item["allowed_absorption_classes"] = tuple(filter(None, item.pop("allowed_classes").split("|")))
                item["is_default"] = bool(item["is_default"])
                mappings.setdefault(item["semantic_id"], []).append(item)
            records = [dict(row) for row in connection.execute("SELECT * FROM materials ORDER BY material_idx")]
        finally:
            connection.close()
        return cls(
            records,
            semantic_map=semantics,
            source_path=path,
            runtime_metadata=metadata,
            semantic_mappings=mappings,
            material_types=material_types,
        )

    @property
    def is_v3(self) -> bool:
        return bool(self.runtime_metadata)

    def stats(self) -> dict[str, Any]:
        if self.is_v3:
            return {**self.runtime_metadata, "resource": str(self.source_path), "cached_samples": len(self._sample_cache)}
        return {"material_count": len(self.records), "semantic_count": len(self.semantic_map), "resource": str(self.source_path)}

    def catalog(self) -> list[dict[str, Any]]:
        output = []
        for semantic, meta in self.semantic_map.items():
            mappings = self.semantic_mappings.get(semantic, [])
            available = [
                level for level in ABSORPTION_CLASSES
                if any(self._eligible_family_rows(mapping, level, semantic) for mapping in mappings)
            ]
            default = next((row for row in mappings if row.get("is_default")), mappings[0] if mappings else {})
            output.append({
                "semantic": semantic,
                "name_zh": meta.get("name_zh", ""),
                "name_en": meta.get("name_en", semantic),
                "group": meta.get("semantic_group", ""),
                "available_absorption_classes": available,
                "default_material_type": default.get("material_type_id"),
            })
        return output

    def sample(
        self,
        semantic: str,
        *,
        absorption_level: str | Iterable[str] | None = None,
        seed: int = 0,
    ) -> Material:
        level = absorption_level
        if not isinstance(level, str) and level is not None:
            values = list(level)
            level = values[0] if values else None
        return self.sample_object(semantic, seed=seed, absorption_class=level)

    def sample_object(
        self,
        semantic_object: str,
        *,
        seed: int = 0,
        absorption_class: str | None = None,
        material_type: str | None = None,
    ) -> Material:
        exact = self._by_id.get(str(semantic_object))
        if exact is not None:
            return _material_from_v3_row(exact, str(exact.get("material_type_id", semantic_object))) if self.is_v3 else _material_from_row(exact, semantic_object)
        if self.is_v3:
            canonical = self.canonical_semantic(semantic_object)
            if canonical in self.semantic_mappings:
                return self.sample_semantic(
                    canonical,
                    absorption_class=absorption_class,
                    material_type=material_type,
                    seed=seed,
                )
            family = _material_family_alias(semantic_object)
            if family in self.material_types:
                return self._sample_family(family, absorption_class, semantic_object, seed)
            return fallback_material(canonical)

        canonical = _canonical_object(semantic_object, self.semantic_map)
        object_rows = [
            row for row in self.object_candidates
            if str(row.get("semantic_object", "")).lower() == canonical
        ]
        object_rows = _filter_absorption(object_rows, absorption_class)
        if object_rows:
            rng = random.Random(seed + _stable_hash(f"object:{canonical}"))
            return _material_from_row(rng.choice(object_rows), canonical)
        candidates = _filter_absorption(self._material_candidates(canonical), absorption_class)
        if candidates:
            rng = random.Random(seed + _stable_hash(f"material:{canonical}"))
            return _material_from_row(rng.choice(candidates), canonical)
        return fallback_material(canonical)

    def sample_semantic(
        self,
        semantic: str,
        *,
        absorption_class: str | None = None,
        material_type: str | None = None,
        seed: int = 0,
        allow_equivalent_area: bool = False,
    ) -> Material:
        canonical = self.canonical_semantic(semantic)
        requested = _normalize_absorption(absorption_class)
        family = str(material_type or "")
        key = (canonical, requested, family, int(seed), bool(allow_equivalent_area))
        with self._sample_cache_lock:
            cached = self._sample_cache.get(key)
        if cached is not None:
            return cached
        material = self._sample_semantic_uncached(
            canonical,
            requested,
            family,
            int(seed),
            allow_equivalent_area=bool(allow_equivalent_area),
        )
        with self._sample_cache_lock:
            return self._sample_cache.setdefault(key, material)

    def sample_surface_set(
        self,
        profile: str | Mapping[str, Any] | None = None,
        *,
        seed: int = 0,
        overrides: Mapping[str, str | Mapping[str, Any] | Material] | None = None,
    ) -> dict[str, Material]:
        profile_map: Mapping[str, Any] = profile if isinstance(profile, Mapping) else {}
        global_level = profile if isinstance(profile, str) else None
        override_map = dict(overrides or {})
        output: dict[str, Material] = {}
        for index, (surface, semantic) in enumerate(SURFACE_SEMANTICS.items()):
            item_seed = int(seed) + _stable_hash(f"surface:{surface}") + index
            if surface in override_map:
                output[surface] = self.resolve(override_map[surface], default_semantic=semantic, seed=item_seed)
                continue
            spec = profile_map.get(surface, profile_map.get(semantic, global_level))
            output[surface] = self.resolve(spec, default_semantic=semantic, seed=item_seed)
        return output

    def sample_geometry(self, item: Mapping[str, Any], *, seed: int = 0) -> Material:
        if item.get("material") or item.get("material_id"):
            return self.resolve(
                {"material_id": str(item.get("material") or item.get("material_id"))},
                default_semantic=self.canonical_semantic(str(item.get("semantic", item.get("type", "structural_element")))),
                seed=seed,
            )
        return self.sample_semantic(
            str(item.get("semantic", item.get("type", "structural_element"))),
            absorption_class=str(item.get("absorption_class", item.get("absorption_level", "auto"))),
            material_type=str(item.get("material_type", "")) or None,
            seed=seed,
            allow_equivalent_area=True,
        )

    def resolve(
        self,
        spec: str | Mapping[str, Any] | Material | None,
        *,
        default_semantic: str,
        seed: int = 0,
    ) -> Material:
        if isinstance(spec, Material):
            return spec
        if isinstance(spec, Mapping):
            material_id = spec.get("material_id") or spec.get("material")
            if material_id:
                exact = self._by_id.get(str(material_id))
                if exact is not None and self.is_v3:
                    return _material_from_v3_row(exact, default_semantic)
                return self.sample_object(str(material_id), seed=seed)
            return self.sample_semantic(
                str(spec.get("semantic", default_semantic)),
                absorption_class=spec.get("absorption_class", spec.get("level")),
                material_type=str(spec.get("material_type", "")) or None,
                seed=seed,
            )
        if spec is None or _normalize_absorption(str(spec), strict=False) != "":
            return self.sample_semantic(
                default_semantic,
                absorption_class=None if spec is None else str(spec),
                seed=seed,
            )
        exact = self._by_id.get(str(spec))
        if exact is not None and self.is_v3:
            return _material_from_v3_row(exact, default_semantic)
        return self.sample_object(str(spec), seed=seed)

    def canonical_semantic(self, semantic: str) -> str:
        normalized = _normalized(semantic)
        compatibility = {
            "window": "window_glass",
            "glass": "window_glass",
            "curtain": "curtain_blind",
            "carpet": "carpet_rug",
            "rug": "carpet_rug",
            "sofa": "sofa_couch",
            "bed": "bed_mattress",
            "chair": "chair_seating",
            "table": "table_desk_counter",
            "cabinet": "cabinet_shelf_wardrobe",
            "fridge": "appliance",
            "refrigerator": "appliance",
            "washing_machine": "appliance",
            "person": "human_person",
            "human": "human_person",
            "panel": "acoustic_treatment",
            "plant": "small_objects_ignore",
            "stairs": "small_objects_ignore",
        }
        return self._semantic_aliases.get(normalized, compatibility.get(normalized, normalized))

    def _sample_semantic_uncached(
        self,
        semantic: str,
        requested: str,
        family: str,
        seed: int,
        *,
        allow_equivalent_area: bool,
    ) -> Material:
        mappings = self.semantic_mappings.get(semantic, [])
        if family:
            mappings = [row for row in mappings if row["material_type_id"] == family]
            if not mappings:
                raise ValueError(f"material type {family!r} is not valid for semantic {semantic!r}")
        elif requested == "auto":
            default = next((row for row in mappings if row.get("is_default")), None)
            mappings = [default or mappings[0]] if mappings else []

        resolved = requested
        eligible = self._eligible_mappings(
            semantic,
            mappings,
            None if requested == "auto" else requested,
            allow_equivalent_area=allow_equivalent_area,
        )
        fallback_from: str | None = None
        if not eligible and requested != "auto":
            fallback_from = requested
            for level in _nearest_absorption_classes(requested):
                eligible = self._eligible_mappings(
                    semantic,
                    mappings,
                    level,
                    allow_equivalent_area=allow_equivalent_area,
                )
                if eligible:
                    resolved = level
                    break
        if not eligible:
            return fallback_material(semantic)

        rng = random.Random(seed + _stable_hash(f"semantic:{semantic}:{requested}:{family}"))
        if requested == "auto":
            mapping, rows = eligible[0]
        else:
            weights = [1.0 / max(float(mapping.get("priority", 1)), 1.0) for mapping, _ in eligible]
            mapping, rows = rng.choices(eligible, weights=weights, k=1)[0]
        row = rng.choice(rows)
        material = _material_from_v3_row(row, semantic)
        metadata = {
            **dict(material.metadata),
            "requested_absorption_class": requested,
            "resolved_absorption_class": str(row["absorption_class"]),
            "fallback_from": fallback_from,
            "condition_code": mapping.get("condition_code", ""),
        }
        return Material(
            id=material.id,
            name=material.name,
            semantic=material.semantic,
            absorption=material.absorption,
            scattering=material.scattering,
            transmission_loss_db=material.transmission_loss_db,
            source=material.source,
            metadata=metadata,
        )

    def _sample_family(self, family: str, level: str | None, semantic: str, seed: int) -> Material:
        requested = _normalize_absorption(level)
        levels = ABSORPTION_CLASSES if requested == "auto" else (requested,)
        rows = [row for candidate in levels for row in self._surface_rows(family, candidate)]
        if not rows and requested != "auto":
            rows = [row for candidate in _nearest_absorption_classes(requested) for row in self._surface_rows(family, candidate)]
        if not rows:
            return fallback_material(semantic)
        rng = random.Random(seed + _stable_hash(f"family:{family}:{requested}"))
        return _material_from_v3_row(rng.choice(rows), semantic)

    def _eligible_mappings(
        self,
        semantic: str,
        mappings: list[dict],
        level: str | None,
        *,
        allow_equivalent_area: bool = False,
    ) -> list[tuple[dict, list[dict]]]:
        output = []
        for mapping in mappings:
            if level is None:
                allowed = mapping.get("allowed_absorption_classes", ABSORPTION_CLASSES)
                rows = [
                    row
                    for item_level in allowed
                    for row in self._surface_rows(
                        mapping["material_type_id"], item_level, semantic, allow_equivalent_area=allow_equivalent_area
                    )
                ]
            else:
                rows = self._eligible_family_rows(
                    mapping,
                    level,
                    semantic,
                    allow_equivalent_area=allow_equivalent_area,
                )
            if rows:
                output.append((mapping, rows))
        return output

    def _eligible_family_rows(
        self,
        mapping: Mapping[str, Any],
        level: str,
        semantic: str,
        *,
        allow_equivalent_area: bool = False,
    ) -> list[dict]:
        if level not in mapping.get("allowed_absorption_classes", ABSORPTION_CLASSES):
            return []
        return self._surface_rows(
            str(mapping["material_type_id"]),
            level,
            semantic,
            allow_equivalent_area=allow_equivalent_area,
        )

    def _surface_rows(
        self,
        family: str,
        level: str,
        semantic: str | None = None,
        *,
        allow_equivalent_area: bool = False,
    ) -> list[dict]:
        rows = [
            row for row in self._by_family_class.get((family, level), [])
            if bool(row.get("is_selectable", True))
            and (
                str(row.get("coefficient_kind", "surface_absorption_coefficient")) == "surface_absorption_coefficient"
                or allow_equivalent_area
            )
        ]
        preferred_tags = SEMANTIC_INSTANCE_TAGS.get(str(semantic), ())
        preferred = [row for row in rows if str(row.get("legacy_vlm_type", "")) in preferred_tags]
        return preferred or rows

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
        "carpet_rug": 0.45,
        "curtain": 0.45,
        "curtain_blind": 0.45,
        "sofa": 0.55,
        "sofa_couch": 0.55,
        "window": 0.08,
        "window_glass": 0.08,
    }.get(semantic, 0.20)
    return Material(
        id=f"fallback_{semantic}",
        name=f"fallback {semantic}",
        semantic=semantic,
        absorption=band_constant(alpha),
        scattering=band_constant(0.12),
        transmission_loss_db=_default_transmission_loss(semantic),
        source="fallback",
        metadata={"requested_absorption_class": "auto", "resolved_absorption_class": "fallback"},
    )


def material_summary(material: Material) -> dict[str, Any]:
    return {
        "semantic": material.semantic,
        "material_id": material.id,
        "material_name": material.name,
        "source": material.source,
        "absorption": dict(material.absorption),
        **dict(material.metadata),
    }


def _material_from_v3_row(row: Mapping[str, Any], semantic: str) -> Material:
    absorption = {band: float(row[f"a{band}"]) for band in FREQUENCY_BANDS}
    material_type = str(row.get("material_type_id", ""))
    return Material(
        id=str(row["material_id"]),
        name=str(row.get("canonical_name", row["material_id"])),
        semantic=semantic,
        absorption=absorption,
        scattering=band_constant(0.12),
        transmission_loss_db=_default_transmission_loss(semantic),
        source="acoustic_materials_v3",
        metadata={
            "material_idx": int(row.get("material_idx", 0)),
            "material_type": material_type,
            "absorption_class": str(row.get("absorption_class", "")),
            "coefficient_kind": str(row.get("coefficient_kind", "")),
            "alpha_mid": float(row.get("alpha_mid", 0.0)),
            "alpha_high": float(row.get("alpha_high", 0.0)),
            "source_group": str(row.get("source_group", "")),
            "confidence": float(row.get("confidence", 0.0)),
            "quality_status": str(row.get("quality_status", "")),
        },
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
        metadata={
            "material_type": str(row.get("material_type_norm", row.get("material_family", ""))),
            "absorption_class": str(row.get("absorption_level", "")),
        },
    )


def _default_transmission_loss(semantic: str) -> dict[str, float]:
    key = semantic.lower()
    base = {
        "wall": (38.0, 42.0, 46.0, 50.0, 52.0, 54.0),
        "floor": (42.0, 46.0, 50.0, 54.0, 56.0, 58.0),
        "ceiling": (35.0, 39.0, 43.0, 47.0, 49.0, 51.0),
        "window": (22.0, 25.0, 28.0, 31.0, 33.0, 35.0),
        "window_glass": (22.0, 25.0, 28.0, 31.0, 33.0, 35.0),
        "door": (20.0, 23.0, 26.0, 29.0, 31.0, 33.0),
        "curtain": (6.0, 8.0, 10.0, 12.0, 14.0, 16.0),
        "curtain_blind": (6.0, 8.0, 10.0, 12.0, 14.0, 16.0),
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


def _normalize_absorption(level: str | None, *, strict: bool = True) -> str:
    if level is None:
        return "auto"
    key = _normalized(level)
    aliases = {
        "": "auto",
        "auto": "auto",
        "random": "auto",
        "reflective": "reflective",
        "low": "reflective",
        "semi_reflective": "semi_reflective",
        "semi": "semi_reflective",
        "medium_low": "semi_reflective",
        "absorptive": "absorptive",
        "medium": "absorptive",
        "high": "highly_absorptive",
        "highly_absorptive": "highly_absorptive",
    }
    if key in aliases:
        return aliases[key]
    if strict:
        raise ValueError(f"unknown absorption class: {level!r}; expected auto or one of {ABSORPTION_CLASSES}")
    return ""


def _nearest_absorption_classes(level: str) -> tuple[str, ...]:
    index = ABSORPTION_CLASSES.index(level)
    return tuple(sorted(ABSORPTION_CLASSES, key=lambda item: (abs(ABSORPTION_CLASSES.index(item) - index), ABSORPTION_CLASSES.index(item))))


def _material_family_alias(value: str) -> str:
    key = _normalized(value)
    return {
        "wood": "wood_panel",
        "glass": "glass_reflective",
        "metal": "metal_reflective",
        "plastic": "plastic_vinyl_reflective",
        "fabric": "fabric_textile_absorber",
        "carpet": "carpet_floor_absorber",
    }.get(key, key)


def _normalized(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")
