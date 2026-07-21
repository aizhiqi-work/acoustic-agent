# Runtime Resources

Acoustic Agent distributions are resource-complete. The files below are part of
the supported runtime contract and must not be omitted from wheels, source
distributions, archives, or Git LFS checkouts.

## Inventory

| ID | Package path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `cipic_124` | `resources/hrtf/cipic_124.sofa` | 3,602,890 | `c28ff4a874ac889ec0c5885ca524762a70d56984232ff7aadcd9c15d32e1cfb6` |
| `sadie_h12` | `resources/hrtf/sadie_h12.sofa` | 8,753,320 | `44306dad84af6976597ec0ed1d7dbbbc13e0696a41515a9a00ce39bfb6202bd0` |
| `acoustic_materials_v3` | `resources/acoustic_materials/acoustic_materials_v3.sqlite3` | 1,912,832 | `5f905021fad8f9352b6a761bb3ac2e80651957d0a66ae67c5ecce5e0f53bd555` |
| `floorplan_v1` | `resources/floorplan/floorplan_v1.sqlite3` | 63,741,952 | `c6c472c206835dfca69bd25c932ae7175adabe160e5d0b252296d4da5f104e06` |
| `audio_main_voice` | `resources/audio/main_voice.wav` | 1,401,750 | `530e0794868ca5cd0d28d4c371abb2b40a953032e5cb17878739fcc2578b9e4c` |
| `audio_background_speech` | `resources/audio/background_speech.wav` | 1,162,832 | `8d42efd055da764f2e8059867c7869be94e06907450baae6c0761eeda886b1a3` |
| `audio_piano_1` | `resources/audio/piano_1.mp3` | 340,218 | `011e4092d3f422e36688b7da15693709e64ffa9e649b519c3bb7e9406be58c0e` |
| `audio_piano_2` | `resources/audio/piano_2.mp3` | 1,748,577 | `b9a7251a7525a7297dcfe3a6e576e9c0e4146d72a8294eca39821464f94912d3` |
| `audio_pink_noise_bed` | `resources/audio/pink_noise_bed.wav` | 1,764,080 | `661061e89b29924091e879a28da9109285508fb819f440f1d5e5175f2c3bba20` |

The machine-readable copy is `acoustic_agent/resources/manifest.json`.

## Runtime Verification

Fast schema and size checks:

```bash
acoustic-agent verify-resources
```

Full integrity check:

```bash
acoustic-agent verify-resources --hashes
```

The verifier opens each SOFA file with h5py, checks required datasets, opens
each SQLite database in read-only mode, checks required tables, validates audio
container headers, and validates record counts. `--hashes` also reads every
byte and compares SHA-256 values.

## HRTF Resources

`cipic_124.sofa` is the default HRTF. It contains 1,250 source directions and
two-channel HRIRs of 200 samples. `sadie_h12.sofa` contains 2,114 packaged source
directions and two-channel HRIRs of 256 samples.

SOFA global attributes retain database name, organization, contacts, references,
license text, and conversion notes. Do not strip these attributes when replacing
or processing a file.

## Acoustic Materials V3

The runtime SQLite database has schema version 3 and contains:

- 3,741 material and acoustic-property records from five source groups.
- 16 material families.
- 20 semantic object categories.
- 64 semantic-material mappings.
- Six octave bands: 125, 250, 500, 1000, 2000, and 4000 Hz.

The runtime distribution intentionally contains one material data artifact: the
SQLite database. Duplicate CSV, JSON, JSONL, and offline rebuild inputs are not
packaged. Historical rebuild tooling is retained on the
`archive/material-db-rebuild-v3` branch.

The material coefficients have mixed upstream terms. The VLM-assisted mappings,
taxonomy, QA metadata, and runtime selector are project contributions; the VLM
does not generate coefficient values. See [`MATERIAL_DATABASE.md`](MATERIAL_DATABASE.md),
the packaged `acoustic_materials/sources.json`, and
`acoustic_materials/DATA_LICENSE.md` for the complete lineage and release gate.

## Floorplan V1

The runtime database has schema version 1 and stores 15,376 audited scenes from
17,107 source records. Scene payloads use zlib-compressed JSON. The original
dataset index is retained so Web and Python APIs address the same scene.

The resource is adapted from the ResPlan dataset by Mohamed Abouagour and
Eleftherios Garyfallidis, "ResPlan: A Large-Scale Vector-Graph Dataset of 17,000
Residential Floor Plans," arXiv:2508.14006 (2025). See
https://arxiv.org/abs/2508.14006 and
https://www.kaggle.com/datasets/resplan/resplan.

Rebuild from a legacy pickle with:

```bash
python scripts/build_floorplan_resource.py \
  --source /path/to/source.pkl \
  --output acoustic_agent/resources/floorplan/floorplan_v1.sqlite3
```

The raw pickle is not required at runtime. The compiled SQLite database is.
The adapted resource is CC BY-NC-SA 4.0; see `floorplan/DATA_LICENSE.md`.

## Demo Audio

Five source programs are packaged for reproducible listening tests: project
narration, background speech, two piano recordings, and a stationary
pink-noise bed. They are convolved at runtime and are not part of the RIR
itself. See `resources/audio/README.md` and `resources/audio/DATA_LICENSE.md`.

## Git LFS

`.gitattributes` assigns `.sqlite3` and `.pkl` files to Git LFS. The smaller
SOFA resources are tracked directly so their license metadata and default HRTF
remain available in an ordinary source checkout. A valid clone must run
`git lfs pull` before installation or building.

When adding or replacing a runtime resource:

1. Confirm redistribution rights and update `THIRD_PARTY_NOTICES.md`.
2. Prepare and audit the resource outside the runtime repository.
3. Update `manifest.json` size and SHA-256.
4. Run `acoustic-agent verify-resources --hashes`.
5. Run `pytest` and build wheel/sdist.
6. Inspect both archives for all required runtime resources.

## Public Release Gate

The CIPIC and SADIE SOFA files carry license metadata. Acoustic Materials V3
now has source-level attribution and mixed-license documentation. Its ODEON /
manufacturer and Acoustic Supplies subsets still require written redistribution
permission or exclusion from a reviewed public build. Floorplan V1 carries
ResPlan attribution and CC BY-NC-SA 4.0 data terms. These are release
requirements, not reasons to silently omit runtime resources from an advertised
complete build. The demo recordings are currently marked `NOASSERTION`; verify
their redistribution rights before publishing a release artifact.
