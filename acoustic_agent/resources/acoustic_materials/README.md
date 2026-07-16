# AudioMaterialDB

This directory is generated from `audiomaterialdb.xlsx`.

## Editable Source

- `audiomaterialdb.xlsx`: the human-editable source workbook.

## Runtime Files

- `materials.jsonl`: canonical material rows from `Normalized_DB`.
- `materials.csv`: compact editable table for quick inspection.
- `object_material_candidates.jsonl`: VLM semantic object to material candidates.
- `object_material_candidates.csv`: compact candidate table.
- `semantic_object_map.json`: VLM aliases, object roles, candidate material types, and sampling policy.
- `aliases.json`: normalized material-name lookup index.
- `taxonomy.json`: compact counts for VLM type, material type, absorption level, and semantic objects.
- `index.json`: database metadata and build statistics.

## Current Build

- Materials: `3741`
- Object candidates: `887`
- Semantic objects: `19`
- Absorption levels: `reflective, semi_reflective, absorptive, highly_absorptive`

Runtime code should use the JSONL/JSON files. Rebuild them after editing the workbook:

```bash
python scripts/build_audio_material_db.py
```

## Hybrid GA Defaults

The workbook currently provides absorption coefficients. Runtime material lookup
and VLM sampling also attach an `acoustic_model` object with engineering
defaults for hybrid geometrical acoustics:

- `scattering`: 6-band roughness-based diffuse split for smooth, medium, or rough surfaces.
- `transmission_loss_db`: 6-band isolation preset for glass, gypsum, wood, masonry, light, door, or open surfaces.
- `transmission`: pressure gain `10 ** (-R / 20)` per band, applied once per
  physical barrier crossed by the direct path.
- `specular_reflection` and `diffuse_reflection`: Steam-style reflected energy
  fractions using `(1 - alpha) * (1 - s)` and `(1 - alpha) * s`.

These defaults are intentionally conservative. They give the current direct,
diffraction, and path-traced RT solver material semantics without requiring
measured scattering or transmission data in the source workbook.
