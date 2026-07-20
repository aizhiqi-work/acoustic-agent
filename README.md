# Acoustic Agent

Acoustic Agent is a Python indoor sound-field simulation engine and local
WebGL workbench for generating room impulse responses (RIRs). It supports
editable geometric rooms, indexed Floorplan apartments, semantic materials,
directional sources, mono/array/HRTF receivers, and static or moving endpoints.

The repository is self-contained for normal use. Both bundled SOFA HRTF files,
the acoustic-material SQLite database, and the compiled Floorplan SQLite database
are included in source and Python distributions.

> **Project status:** research alpha. The solver is designed for reproducible
> experimentation and dataset generation. Validate results against measurements
> before using them for safety-critical, architectural, or commercial decisions.

## Highlights

- One Python API for Geometry and Floorplan scenes.
- Unified WebGL workbench at `/geometry` and `/floorplan`.
- Direct sound with distance attenuation and air absorption.
- Occlusion, transmission, UTD-style diffraction, and path-traced reflections.
- Six-band energy tracing and a 16-line Hadamard FDN late-reverb model.
- Coupled-room decay and portal routing for open-door cross-room scenes.
- Numba JIT kernels with deterministic seeded simulation and workspace caches.
- Mono, linear array, circular array, and SOFA HRTF receivers.
- Omni, cardioid-family, dipole, and configurable source directivity.
- Reproducible semantic material and furniture sampling.
- Static, approach, and random travel trajectories with per-frame RIRs.

## Bundled Runtime Resources

| Resource | Contents | Installed size |
| --- | --- | ---: |
| `cipic_124.sofa` | Default CIPIC subject 124 HRTF | 3.60 MB |
| `sadie_h12.sofa` | SADIE II subject H12 HRTF | 8.75 MB |
| `acoustic_materials_v3.sqlite3` | 3,741 six-band material records | 1.91 MB |
| `floorplan_v1.sqlite3` | 15,376 audited apartment scenes | 63.74 MB |

The open-source distribution intentionally includes the complete runtime SQLite
resources. A normal installation can therefore run Geometry and Floorplan
simulations without downloading a separate material service or converting the
material data at runtime. The SQLite files use Git LFS; wheels and source
distributions contain the resolved database files.

> **Code and data use different terms.** Apache-2.0 applies to the engine and
> project-authored documentation. Inclusion of a database in this repository is
> not a statement that every upstream record is Apache-2.0 data. Source terms,
> attribution, and change notices remain attached to the bundled resources.

Sizes, SHA-256 values, roles, and license notes are recorded in
`acoustic_agent/resources/manifest.json`. Read
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing a
release. The material architecture and source breakdown are documented in
[`docs/MATERIAL_DATABASE.md`](docs/MATERIAL_DATABASE.md).

## Requirements

- Python 3.10 or newer.
- Git LFS when installing from a Git clone.
- A modern browser with WebGL for the workbench.

The Python runtime dependencies are NumPy, Numba, h5py, Shapely, and NetworkX.
A C/C++ compiler is not required for a normal wheel-based installation.

## Install From Source

```bash
git lfs install
git clone <repository-url>
cd acoustic-agent
git lfs pull

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Verify that installation did not leave Git LFS pointer files in place:

```bash
acoustic-agent verify-resources --hashes
```

For a non-editable local installation:

```bash
python -m pip install .
```

Detailed platform, wheel, offline, and troubleshooting instructions are in
[`docs/INSTALLATION.md`](docs/INSTALLATION.md).

Floorplan data conversion, geometry semantics, attribution, and license terms
are documented in [`docs/FLOORPLAN.md`](docs/FLOORPLAN.md).

## Start The Workbench

```bash
acoustic-agent web
```

Open:

- Geometry: http://127.0.0.1:8765/geometry
- Floorplan: http://127.0.0.1:8765/floorplan

Configuration examples:

```bash
acoustic-agent web --host 127.0.0.1 --port 9000
acoustic-agent web --floorplan-resource /path/to/floorplan_v1.sqlite3
acoustic-agent web --floorplan-dataset /path/to/ResPlan.pkl
```

Startup performs a small Numba warmup. Use `--no-warmup` only when faster server
startup matters more than latency on the first simulation.

The legacy commands remain supported:

```bash
python scripts/run_web.py
python -m acoustic_agent.web_server --port 8765
```

## Geometry Python API

```python
from acoustic_agent import AcousticAgent

room = {
    "shape": "u_shape",
    "size": [6.0, 10.0, 2.8],
    "material_profile": {
        "wall": "auto",
        "floor": "auto",
        "ceiling": "auto",
    },
    "material_seed": 42,
    "opening_width": 0.42,
    "opening_depth": 0.82,
    "opening_offset": 0.5,
}

agent = AcousticAgent(
    room=room,
    source_model={"type": "omni"},
    receiver_model={"type": "mono"},
    quality="simulation",
    duration_s=2.0,
    fs=16000,
)

result = agent.run(
    source=[5.766, 2.587, 1.231],
    receiver=[1.079, 6.252, 1.348],
)

rir = result.rir                 # float32 [channel, sample]
rt60 = result.rt60               # per-band and broadband EDC/solver estimates
paths = result.paths             # direct/diffraction/representative RT paths
metadata = result.metadata       # solver, materials, decay, cache, diagnostics
```

## Floorplan Python API

The shortest form reads the bundled database by `idx`, samples rooms and valid
positions, and builds the complete apartment geometry:

The bundled scenes are derived from **ResPlan**, a 17,000-plan vector-graph
dataset by Mohamed Abouagour and Eleftherios Garyfallidis. Acoustic Agent uses
the name **Floorplan** for its converted geometry and simulation interface.
See the [paper](https://arxiv.org/abs/2508.14006),
[dataset](https://www.kaggle.com/datasets/resplan/resplan), and
[`docs/FLOORPLAN.md`](docs/FLOORPLAN.md) for attribution and conversion details.

```python
from acoustic_agent import AcousticAgent

agent = AcousticAgent.from_floorplan(
    idx=0,
    placement="same_room",       # random / same_room / cross_room
    seed=42,
    material_seed=1451557868,
    material_profile={
        "wall": "auto",
        "floor": "auto",
        "ceiling": "auto",
        "door": "auto",
        "window": "auto",
    },
    source_model={"type": "omni"},
    receiver_model={"type": "mono"},
    quality="simulation",
    duration_s=2.0,
    fs=16000,
)

print(agent.rooms)       # candidate room list
print(agent.placement)   # sampled rooms and positions
rir = agent.run().rir
```

Use `source_room`, `receiver_room`, `source`, or `receiver` only when an
experiment needs explicit placement. The default indexed workflow does not
require room names.

Verified interior doors are represented as open portals with reflective
lintels. Wall-free room connections are full-height portals. Unmatched entry
doors remain closed surfaces, and windows remain glass surfaces. Same-room and
cross-room runs use the same full-apartment model.

## Semantic Furniture And Materials

Acoustic Materials DB is bundled as
`acoustic_agent/resources/acoustic_materials/acoustic_materials_v3.sqlite3`.
It provides 3,741 six-band records from five attributed source groups:

| Source group | Records | Role in the compiled database |
| --- | ---: | --- |
| PTB | 2,573 | Absorption-coefficient database |
| ODEON and identified manufacturers | 981 | Product and material exchange sheets |
| Pyroomacoustics | 90 | Published room-acoustics material presets |
| Acoustic Supplies | 67 | Common building-material coefficient chart |
| SoundSpaces | 30 | Visual-acoustic material presets |

The main project contribution is the VLM-assisted Semantic-to-Material Mapping
layer. Material names and descriptions are organized into 20 simulator-facing
semantic classes, 16 material families, and four absorption classes. The VLM
does not generate or replace acoustic coefficients. At runtime the deterministic
selection chain is:

```text
scene semantic
  -> compatible material family
  -> absorption class
  -> attributed material record
  -> six-band absorption coefficients
```

Source identifiers are retained in the database, while confidence and quality
flags describe the normalization and mapping decisions. `material_seed` makes
the full selection reproducible.

```python
agent = AcousticAgent.from_floorplan(
    idx=12,
    seed=42,
    material_seed=2026,
    acoustic_geometry=[
        {
            "id": "sofa_0",
            "type": "sofa",
            "semantic": "sofa_couch",
            "absorption_class": "highly_absorptive",
            "position": [2.5, 2.0],
            "rotation": 0.0,
            "size": [2.0, 0.9, 0.72],
        }
    ],
)
```

Material classes are `reflective`, `semi_reflective`, `absorptive`, and
`highly_absorptive`. `auto` samples a compatible family and published six-band
record. Door and window segments receive their own sampled materials instead of
inheriting the wall.

The SQL database is provided as a reproducibility and simulation resource, not
as a new blanket license over the upstream records. Acoustic Agent grants only
the rights it owns in the schema, semantic taxonomy, mappings, QA metadata,
builder, and sampler. Users and redistributors remain responsible for following
the source-specific terms recorded in the packaged `DATA_LICENSE.md` and
`sources.json`. Coefficients are reference values whose validity depends on
measurement method, mounting, geometry, and frequency; validate them for the
intended engineering use.

## Motion

```python
motion = agent.sample_motion(
    mode="approach",              # approach / random
    moving="receiver",           # source / receiver
    distance_m=2.0,
    keyframe_spacing_m=0.25,
    seed=42,
)

dynamic = agent.run_dynamic(motion)
rir_frames = dynamic.rirs
```

Floorplan trajectories route through verified openings when they cross rooms.
Each frame is solved against its updated source/receiver position and contains
its own RIR and metrics.

## Quality Presets

| Quality | Rays | Bounces | Intended use |
| --- | ---: | ---: | --- |
| `preview` | 8,192 | 32 | Fast interaction and layout checks |
| `simulation` | 32,768 | 64 | Default dataset and listening work |
| `fine` | 65,536 | 96 | Higher-stability analysis |
| `reference` | 131,072 | 96 | Slow convergence/reference runs |

Cross-room runs can raise the bounce budget adaptively to at least 96 and at
most 128. The exact resolved configuration is reported in result metadata.

RIR length and quality are independent. `duration_s` controls the returned
array length; the quality preset controls ray and bounce counts. Choose a
duration long enough to contain the expected decay.

## Receiver And Source Models

Receiver examples:

```python
{"type": "mono"}
{"type": "hrtf", "orientation_deg": -60}
{"type": "linear", "count": 4, "spacing_m": 0.08, "orientation_deg": 0}
{"type": "circular", "count": 8, "radius_m": 0.12, "orientation_deg": 0}
```

The default HRTF is `cipic_124.sofa`. Select SADIE or a custom SOFA file with
`{"type": "hrtf", "sofa_path": "/path/to/file.sofa"}`.

Source examples:

```python
{"type": "omni"}
{"type": "cardioid", "orientation_deg": 30}
{"type": "dipole", "orientation_deg": 30}
{"type": "focused", "orientation_deg": 30}
{"type": "weighted_dipole", "dipole_weight": 0.65, "dipole_power": 2.0}
```

## HTTP API

Start the workbench, then request a directly usable WAV without base64:

```bash
curl -X POST http://127.0.0.1:8765/api/rir.wav \
  -H 'content-type: application/json' \
  --data '{
    "shape": "rectangle",
    "size": [6, 4, 2.8],
    "source": [1.2, 1.1, 1.5],
    "receiver": [4.7, 2.8, 1.4],
    "quality": "preview"
  }' \
  --output rir.wav
```

Use `POST /api/rir.npy` for the exact float32 `[channel, sample]` array. The
compact `POST /api/v1/simulate` response returns result metadata plus temporary
WAV and NPY download links. Workbench-specific endpoints are documented in
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

The HTTP server is intended for local development. It has no authentication or
TLS. Keep the default `127.0.0.1` binding unless you provide a secure proxy.

## Solver Model

The current pipeline combines:

1. Direct sound, distance attenuation, air absorption, occlusion/transmission,
   and source/receiver directivity.
2. UTD-style diffraction for non-line-of-sight geometry.
3. Path-traced six-band reflection energy with specular and diffuse scattering.
4. Per-band decay estimation and a calibrated 16-line Hadamard FDN tail.
5. Coupled-room energy statistics and deterministic portal paths for Floorplan.
6. Receiver rendering to mono, arrays, FOA diagnostics, or binaural HRTF output.

Displayed RT paths are a representative stratified subset of the same traced
field, not a second visual-only trace. The complete energy field drives the RIR.

## CLI Reference

```bash
acoustic-agent --version
acoustic-agent info
acoustic-agent info --json
acoustic-agent verify-resources
acoustic-agent verify-resources --hashes
acoustic-agent web --help
```

## Development And Release

```bash
python -m pip install -e ".[dev]"
pytest
python -m build
python -m twine check dist/*
```

CI checks out Git LFS resources, validates their schemas, runs the test suite,
and builds both wheel and source distribution. See
[`CONTRIBUTING.md`](CONTRIBUTING.md), [`CHANGELOG.md`](CHANGELOG.md), and
[`docs/RESOURCES.md`](docs/RESOURCES.md).

## Repository Layout

```text
acoustic_agent/              Python package and solver
  resources/                 SOFA, material SQL, Floorplan SQL, manifests
  web/                       Shared Geometry/Floorplan WebGL frontend
docs/                        Installation, configuration, resource documentation
examples/                    Geometry, Floorplan, and motion examples
scripts/                     Web launchers and reproducible resource builders
tests/                       Solver, API, parity, resource, and Web tests
```

## License And Data Terms

Acoustic Agent source code and project-authored documentation are licensed
under Apache-2.0. The bundled SQL databases remain part of the complete
open-source engine distribution so simulations are reproducible and work out of
the box, but bundled data retains separate source-specific terms.

Floorplan V1 is adapted from the ResPlan dataset and distributed under
CC BY-NC-SA 4.0. Its use is limited to noncommercial purposes, attribution and
change notices are required, and adaptations must use the same license. These
conditions apply to the dataset-derived Floorplan resource, not to the
Apache-2.0 engine code. See the packaged
[`floorplan/DATA_LICENSE.md`](acoustic_agent/resources/floorplan/DATA_LICENSE.md).

Acoustic Materials DB combines five attributed source groups with the
project-authored VLM Semantic-to-Material Mapping, taxonomy, normalization, QA,
and deterministic sampler. Because the upstream terms are not uniform, the
compiled database is marked `NOASSERTION`; it is not relicensed wholesale under
Apache-2.0. No warranty is made that a coefficient is suitable for a particular
room, product, mounting condition, safety decision, or commercial specification.

The repository does not grant rights in upstream data beyond the rights granted
by each source. Attribution and this notice do not replace an upstream license
or permission. The authoritative project notices are:

- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): source attribution,
  citations, and change notices;
- [`DATA_LICENSE.md`](acoustic_agent/resources/acoustic_materials/DATA_LICENSE.md):
  material database terms and release policy; and
- [`sources.json`](acoustic_agent/resources/acoustic_materials/sources.json):
  machine-readable source counts, URLs, and status.

This repository records provenance and intended use transparently; it does not
provide legal advice or override third-party terms.
