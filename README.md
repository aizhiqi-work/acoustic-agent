# Acoustic Agent

Acoustic Agent is a Python indoor sound-field simulation engine and local
WebGL workbench for generating room impulse responses (RIRs). It supports
parametric rooms, multi-room residential floor plans, semantic materials and
furniture, directional sources, microphone arrays, SOFA HRTFs, and moving
sources or receivers.

> **Research alpha:** Acoustic Agent is intended for reproducible experiments
> and dataset generation. Validate results against measurements before using
> them for architectural, safety-critical, or commercial decisions.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Web Workbench](#web-workbench)
- [Python API](#python-api)
- [Motion And Batch Production](#motion-and-batch-production)
- [Resources And Data Terms](#resources-and-data-terms)
- [Accuracy Benchmark](#accuracy-benchmark)
- [Documentation](#documentation)
- [Development](#development)

## Features

- One `AcousticAgent.create()` API for Geometry, Floorplan, and Custom scenes.
- Direct sound, distance and air attenuation, occlusion, transmission, and
  UTD-style diffraction.
- Six-band path-traced reflections and parameterized FDN late reverberation.
- Open-portal routing and coupled-room decay for cross-room simulation.
- Mono, linear-array, circular-array, and bundled SOFA HRTF receivers.
- Omni, cardioid, dipole, focused, and weighted-dipole source directivity.
- VLM-assisted semantic-to-material mapping with deterministic sampling.
- Editable acoustic furniture with room-semantic automatic placement.
- Static and multi-keyframe motion simulation.
- Numba JIT kernels, deterministic seeds, and reusable scene caches.

## Installation

Python 3.10 or newer is required. Git clones also require
[Git LFS](https://git-lfs.com/) for the bundled SQLite resources.

```bash
git lfs install
git clone https://github.com/aizhiqi-work/acoustic-agent.git
cd acoustic-agent
git lfs pull

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Verify the runtime resources after installation:

```bash
acoustic-agent verify-resources --hashes
```

See [Installation](docs/INSTALLATION.md) for non-editable, offline, and
platform-specific instructions.

## Web Workbench

```bash
acoustic-agent web
```

Open one of the shared workbench views:

- [Geometry](http://127.0.0.1:8765/geometry): parametric indoor geometry.
- [Floorplan](http://127.0.0.1:8765/floorplan): indexed residential layouts.
- [Custom](http://127.0.0.1:8765/custom): editable or ChatGPT-assisted layouts.

The workbench exposes the same materials, placement, furniture, source,
receiver, motion, solver, RIR, listening, and Python API controls in each mode.
It runs locally and does not require an external model API.

## Python API

Every scene follows the same two-step workflow:

```python
agent = AcousticAgent.create(...)
result = agent.run()
```

### Geometry

```python
from acoustic_agent import AcousticAgent

agent = AcousticAgent.create(
    scene="geometry",
    room={
        "shape": "u_shape",
        "size": [6.0, 10.0, 2.8],
        "material_profile": {
            "wall": "auto",
            "floor": "auto",
            "ceiling": "auto",
        },
        "opening_width": 0.42,
        "opening_depth": 0.82,
        "opening_offset": 0.5,
    },
    source=[5.766, 2.587, 1.231],
    receiver=[1.079, 6.252, 1.348],
    quality="simulation",
    duration_s=2.0,
    fs=16000,
)

result = agent.run()
rir = result.rir  # float32 [channel, sample]
```

### Floorplan

Floorplan scenes are solver-ready conversions of the
[ResPlan](https://arxiv.org/abs/2508.14006) residential-layout dataset. The
engine samples valid rooms and positions from an indexed apartment:

```python
agent = AcousticAgent.create(
    scene="floorplan",
    idx=0,
    placement="cross_room",  # random / same_room / cross_room
    seed=42,
    furnishing="balanced",   # sparse / balanced / compact / None
    quality="simulation",
)

print(agent.rooms)
print(agent.placement)
rir = agent.run().rir
```

Pass `source`, `receiver`, `source_room`, or `receiver_room` only when an
experiment requires explicit placement. Same-room and cross-room runs use the
same complete apartment model; verified open doors are acoustic portals.

### Custom Floorplan

```python
from acoustic_agent import AcousticAgent, FloorplanBuilder

spec = FloorplanBuilder.from_text(
    "12m x 9m, three bedrooms, one living room, one kitchen, two bathrooms",
    seed=42,
)

agent = AcousticAgent.create(
    scene="custom",
    spec=spec,
    source_room="living_0",
    receiver_room="bedroom_2",
    seed=42,
    quality="preview",
)
rir = agent.run().rir
```

The Custom workbench can also prepare a prompt for a floor-plan image or text
description. Paste ChatGPT's JSON result back into the local editor, validate
it, calibrate its scale, and run it through the same multi-room solver.

`result` exposes `rir`, `paths`, `rt60`, and `metadata`. Receiver models,
directivity, materials, furniture, full configuration, and compatibility APIs
are documented in the [Python API guide](docs/API.md).

## Motion And Batch Production

Static and dynamic simulations share `run()`:

```python
dynamic = agent.run(motion={
    "mode": "approach",       # approach / random / through_portal
    "moving": "receiver",    # source / receiver
    "distance_m": 1.5,
    "keyframe_spacing_m": 0.25,
    "seed": 42,
})
rir_frames = dynamic.rirs
```

Use `run_batch()` when geometry and materials stay fixed but endpoint positions
change:

```python
batch = agent.run_batch([
    ([1.0, 1.0, 1.4], [4.0, 2.0, 1.4]),
    {"id": "pair_b", "source": [2.0, 1.0, 1.4], "receiver": [5.0, 3.0, 1.4]},
], workers=4)
batch.save_npz("room_rirs.npz")
```

Use `run_many()` for independent scene, index, placement, or motion jobs:

```python
jobs = [
    {
        "id": f"floorplan_{idx}_{take}",
        "scene": "floorplan",
        "idx": idx,
        "placement": "random",
        "seed": 1000 + idx * 10 + take,
        "quality": "preview",
    }
    for idx in range(100)
    for take in range(4)
]

dataset = AcousticAgent.run_many(jobs, workers=4, on_error="skip")
dataset.save_npz("floorplan_rirs.npz")
```

The NPZ archive stores float32 RIRs plus a structured JSON manifest. See
[batch_production.py](examples/batch_production.py) for a runnable example.

## Quality Presets

| Quality | Rays | Bounces | Typical use |
| --- | ---: | ---: | --- |
| `preview` | 8,192 | 32 | Layout and interaction checks |
| `simulation` | 32,768 | 64 | Default listening and dataset work |
| `fine` | 65,536 | 96 | Higher-stability analysis |
| `reference` | 131,072 | 96 | Slow convergence/reference runs |

Cross-room scenes can raise the bounce budget adaptively to 96-128. RIR length
is controlled separately with `duration_s`.

## Resources And Data Terms

The distribution includes everything required for normal simulation:

| Resource | Contents |
| --- | --- |
| `cipic_124.sofa`, `sadie_h12.sofa` | Bundled HRTF datasets |
| `acoustic_materials_v3.sqlite3` | 3,741 attributed six-band material records |
| `floorplan_v1.sqlite3` | 15,376 audited, solver-ready residential scenes |

Engine code and project-authored documentation use Apache-2.0. Bundled data
retains separate upstream terms:

- Floorplan V1 is adapted from ResPlan and remains CC BY-NC-SA 4.0.
- Acoustic Materials V3 combines several attributed sources with different
  terms and is marked `NOASSERTION`; it is not relicensed as Apache-2.0 data.
- SOFA files retain the licenses recorded by their source datasets.

Read [Third-Party Notices](THIRD_PARTY_NOTICES.md) and the packaged data notices
before redistributing the repository or its databases. Attribution does not
replace an upstream license or permission.

## Accuracy Benchmark

Run the fixed physical-accuracy suite and generate JSON, Markdown, and
self-contained HTML evidence:

```bash
acoustic-agent benchmark --profile quick --output benchmark-results
```

The suite checks direct arrival and attenuation, shoebox RT60, first-order
reflections, FDN isolation, portal coupling, HRTF behavior, dynamic continuity,
and an optional native Steam Audio same-scene comparison. See the
[benchmark guide](docs/BENCHMARKS.md) for profiles and thresholds.

## Documentation

| Guide | Purpose |
| --- | --- |
| [Python API](docs/API.md) | Scenes, models, motion, batching, and compatibility |
| [Configuration](docs/CONFIGURATION.md) | Shapes, materials, quality, and solver controls |
| [Floorplan](docs/FLOORPLAN.md) | ResPlan conversion, portals, citation, and data terms |
| [Custom Floorplan](docs/CUSTOM_FLOORPLAN.md) | Text/image handoff and JSON schema |
| [Material Database](docs/MATERIAL_DATABASE.md) | Semantic mapping, provenance, and sampling |
| [Runtime Resources](docs/RESOURCES.md) | Packaged SQL/SOFA files and verification |
| [Accuracy Benchmark](docs/BENCHMARKS.md) | Fixed scenes, thresholds, reports, and Steam Audio reference |
| [中文说明](docs/README_zh-CN.md) | Chinese project guide |

## Development

```bash
python -m pip install -e ".[dev]"
pytest
python -m build
python -m twine check dist/*
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md),
[CHANGELOG.md](CHANGELOG.md), and [SECURITY.md](SECURITY.md).

## License

Acoustic Agent source code is licensed under [Apache-2.0](LICENSE). This license
does not override the separate terms of bundled HRTF, material, or Floorplan
data. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for authoritative
attribution and change notices.
