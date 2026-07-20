# Acoustic Materials DB

The runtime source of truth is `acoustic_materials_v3.sqlite3`, compiled for the
fields used by the simulator from `acoustic_material_db_v3_20260717`.

- 20 VLM semantic object classes
- 16 material families
- 3,741 acoustic material records from five source groups
- four mid-band absorption classes
- six absorption bands: 125, 250, 500, 1000, 2000, and 4000 Hz

Selection follows:

```text
semantic object -> compatible material family -> absorption class -> material instance -> six-band table
```

`MaterialLibrary.load()` caches the parsed runtime index for the process. Sampling is deterministic for a semantic, class, optional material family, and seed. The older JSONL/CSV files remain packaged for backward compatibility and inspection, but are not loaded when the v3 SQLite resource is present.

The VLM-assisted layer maps source material names and descriptions to compatible
simulator semantics. It does not generate acoustic coefficients. Source counts,
URLs, citations, and machine-readable license status are in `sources.json`.
Read `DATA_LICENSE.md` before redistribution: this database contains mixed
upstream terms and is not wholly licensed under Apache-2.0.

Rebuild the runtime resource after changing the source database:

```bash
python scripts/build_acoustic_material_resource.py \
  ../acoustic_material_db_v3_20260717 \
  acoustic_agent/resources/acoustic_materials/acoustic_materials_v3.sqlite3
```

After rebuilding, verify that the source-group counts still match
`sources.json`, then update the top-level resource manifest and run the tests.
