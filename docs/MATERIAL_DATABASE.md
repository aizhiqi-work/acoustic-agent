# Acoustic Materials DB

Acoustic Materials DB turns heterogeneous acoustic coefficient tables into a
reproducible semantic material selector for indoor sound-field simulation. It
contains 3,741 records from five source groups, plus project-authored semantic
mappings and quality metadata.

## Data flow

```text
upstream coefficient tables
  -> source-preserving normalization
  -> VLM-assisted semantic-to-material mapping
  -> deterministic rules and QA review
  -> 20 semantic objects and 16 material families
  -> four absorption classes
  -> six-band runtime SQLite database
```

The VLM is used for semantic organization, not coefficient generation. Each
selected material still resolves to a concrete upstream record with six values
at 125, 250, 500, 1,000, 2,000, and 4,000 Hz.

## Project contribution

The project adds the following layer over the upstream material data:

- a 20-class simulator vocabulary for boundaries, treatments, furniture,
  appliances, people, and structural objects;
- 16 acoustic material families and 64 semantic-to-family compatibility rules;
- VLM-assisted name/description classification followed by deterministic rules;
- a stable selection chain from semantic object to a concrete material record;
- source-group provenance, confidence values, and per-record quality flags;
- deterministic sampling controlled by semantic, absorption class, family, and
  seed; and
- a compact SQLite runtime index cached once per process.

Selection follows:

```text
semantic_object_id
  -> compatible material_type_id
  -> absorption_class
  -> material_idx
  -> six-band acoustic properties
```

## Absorption classes

The classification uses:

```text
alpha_mid = mean(alpha_500, alpha_1000, alpha_2000)
```

| Class | Rule |
| --- | --- |
| `reflective` | `alpha_mid < 0.10` |
| `semi_reflective` | `0.10 <= alpha_mid < 0.30` |
| `absorptive` | `0.30 <= alpha_mid < 0.60` |
| `highly_absorptive` | `alpha_mid >= 0.60` |

These classes are sampling controls, not substitutes for the frequency-dependent
coefficients used by the solver.

## Source distribution

| Source group | Records | Share |
| --- | ---: | ---: |
| PTB | 2,573 | 68.78% |
| ODEON / manufacturer sheets | 981 | 26.22% |
| Pyroomacoustics | 90 | 2.41% |
| Acoustic Supplies | 67 | 1.79% |
| SoundSpaces | 30 | 0.80% |

The source-specific citations and redistribution status are recorded in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) and the packaged
`acoustic_agent/resources/acoustic_materials/sources.json`. The database has
mixed terms and must not be described as wholly Apache-2.0 or wholly open data.

## Runtime packaging

The open-source runtime keeps only `acoustic_materials_v3.sqlite3`,
`sources.json`, `DATA_LICENSE.md`, and this documentation. Duplicate exports and
the offline database-authoring workspace are intentionally excluded. The SQL
resource contains everything required by the semantic material sampler.

Historical rebuild tooling is retained on the
`archive/material-db-rebuild-v3` branch. A replacement runtime database must be
prepared and audited outside the runtime repository, then checked by updating
the size and SHA-256 in `acoustic_agent/resources/manifest.json`, running
`acoustic-agent verify-resources --hashes`, and running the test suite.
