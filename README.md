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
- [Audio And Noise](#audio-and-noise)
- [Motion And Batch Production](#motion-and-batch-production)
- [Quality Presets](#quality-presets)
- [CUDA Acceleration](#cuda-acceleration)
- [Performance](#performance)
- [Resources And Data Terms](#resources-and-data-terms)
- [Accuracy Benchmark](#accuracy-benchmark)
- [Research Workflows](#research-workflows)
- [Documentation](#documentation)
- [Development](#development)

## Features

- One `AcousticAgent.create()` API for Geometry, Floorplan, and Custom scenes.
- Selectable `auto` / `linear` / `bvh` intersection backends with cached BVH acceleration for complex scenes.
- Direct sound, distance and air attenuation, occlusion, transmission, and
  UTD-style diffraction.
- Six-band path-traced reflections and parameterized FDN late reverberation.
- Open-portal routing and coupled-room decay for cross-room simulation.
- Mono, linear-array, circular-array, and bundled SOFA HRTF receivers.
- Omni, cardioid, dipole, focused, and weighted-dipole source directivity.
- VLM-assisted semantic-to-material mapping with deterministic sampling.
- Editable acoustic furniture with room-semantic automatic placement.
- Static and multi-keyframe motion simulation.
- Independent multi-source RIRs and speech, music, or noise auralization.
- Selectable Numba FP64/FP32 and single-GPU CUDA FP32 reflection tracing.
- Deterministic seeds and reusable CPU/GPU scene caches.

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

## Audio And Noise

The RIR is independent of signal content: speech, piano, and noise emitted from
the same point reuse the same RIR. A background signal at a different position
needs its own source-to-receiver RIR:

```python
from acoustic_agent import mix_audio_at_snr, render_audio

sources = agent.run_sources({
    "voice": [1.2, 1.1, 1.5],
    "background": [4.8, 1.0, 1.2],
})

voice_wet = render_audio(voice_samples, sources["voice"].rir)
noise_wet = render_audio(noise_samples, sources["background"].rir)
room_mix = mix_audio_at_snr(voice_wet, noise_wet, snr_db=10, normalize=True)
```

Input arrays must use the simulation sample rate; `resample_audio` provides a
dependency-free conversion helper. The Web workbench bundles Project
narration, Background speech, Piano 1, Piano 2, and a Pink-noise bed. It also
generates deterministic white, pink, and brown noise and accepts uploaded
browser-supported audio. Each background source has an independent position
and RIR, including receiver-motion updates.

The SNR control is receiver-domain broadband SNR, not source gain. Both signals
are first propagated through their own RIR; the rendered background is then
scaled to the target RMS ratio. For example, `10 dB` makes the rendered
foreground RMS approximately 10 dB higher than the rendered background RMS.
See the packaged [audio release note](acoustic_agent/resources/audio/DATA_LICENSE.md)
before redistributing the demo recordings.

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
change. Mono production runs omit viewer-only reflection paths and FOA buffers
by default; create the agent with `visualization=True` only when those paths are
needed by a custom viewer:

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

| Quality | Rays | Base bounces | Typical use |
| --- | ---: | ---: | --- |
| `preview` | 8,192 | 32 | Layout and interaction checks |
| `simulation` | 32,768 | 64 | Default listening and dataset work |
| `fine` | 65,536 | 96 | Higher-stability analysis |
| `reference` | 131,072 | 96 | Slow convergence/reference runs |

These values are the requested base budgets. At `simulation` and above, the
solver can increase the effective depth without reducing the ray count:

- Reflective single-room Geometry scenes estimate the depth needed to reach the
  decay fitting floor, capped at 128 bounces by default.
- Cross-room Floorplan and Custom scenes use at least 96 bounces and can rise to
  128 according to portal count and surface survival.
- Preview keeps its explicit 32-bounce budget for responsive editing.

The requested and effective values are recorded in `result.metadata`. RIR
length is independent of quality and is controlled with `duration_s`.

## CUDA Acceleration

The reflection tracer can run on one NVIDIA GPU in FP32. The default remains
Numba FP64 so CPU-only installations and existing experiments keep the same
behavior.

```python
from acoustic_agent import AcousticAgent, SimConfig

config = SimConfig(
    rt_accelerator="cuda",  # numba / cuda / auto
    rt_precision="float32", # Numba: float32 or float64; CUDA: float32
    rt_cuda_device=0,
)

agent = AcousticAgent.create(
    scene="geometry",
    room={"shape": "rectangle", "size": [8.0, 6.0, 2.8]},
    source=[1.0, 1.0, 1.4],
    receiver=[6.5, 4.5, 1.4],
    config=config,
)
result = agent.run()
```

`auto` uses CUDA when the selected device is available and the requested
precision is FP32. It otherwise falls back to Numba. An NVIDIA driver and a
working Numba CUDA target are required:

```bash
python -c "from numba import cuda; print(cuda.is_available())"
```

Warm, single-device measurements below use Python 3.12, Numba 0.65, CUDA 12.4,
the 131,072-ray `reference` preset, and the median of three runs. Times include
the complete RIR call; speedups are relative to 64-thread Numba FP64.

| Geometry workload | Numba FP64 | Numba FP32 | RTX A6000 FP32 | RTX 4090 FP32 |
| --- | ---: | ---: | ---: | ---: |
| Rectangle, 6 surfaces | 201.3 ms | 200.0 ms | 58.5 ms (3.44x) | 58.1 ms (3.47x) |
| Furnished U room, 22 surfaces | 254.6 ms | 247.7 ms | 69.3 ms (3.68x) | 59.6 ms (4.27x) |
| Furnished round room, 98 surfaces | 426.6 ms | 446.8 ms | 126.1 ms (3.38x) | 93.1 ms (4.58x) |

The FloorPlan benchmark uses deterministic cross-room placements. FloorPlan
12513 has 5 rooms; FloorPlan 11282 has 10 rooms.

| FloorPlan workload | Surfaces | Numba FP64 | Numba FP32 | RTX A6000 FP32 | RTX 4090 FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 rooms, empty | 68 | 145.9 ms | 155.3 ms | 84.3 ms (1.73x) | 74.1 ms (1.97x) |
| 5 rooms, 12 furnishings | 83 | 169.8 ms | 166.9 ms | 90.3 ms (1.88x) | 77.6 ms (2.19x) |
| 10 rooms, empty | 145 | 173.9 ms | 175.8 ms | 92.8 ms (1.87x) | 81.8 ms (2.13x) |
| 10 rooms, 27 furnishings | 178 | 200.0 ms | 205.9 ms | 104.2 ms (1.92x) | 92.0 ms (2.17x) |

The CUDA ray-tracing stage alone reached up to 11.40x on RTX A6000 and 13.28x
on RTX 4090 in the formal reference run. Small preview workloads may be
dominated by launch, transfer, and CPU post-processing overhead. See the
[CUDA acceleration guide](docs/CUDA_ACCELERATION.md) for all quality levels,
accuracy deltas, and reproduction commands.

## Performance

Ray intersection can be selected independently of the quality preset:

```python
agent = AcousticAgent.create(
    scene="floorplan",
    idx=0,
    quality="simulation",
    intersection_backend="auto",  # auto / linear / bvh
)
```

`linear` checks every surface and remains the reference traversal. `bvh` uses a
cached bounding-volume hierarchy to reject surfaces that a ray cannot hit, then
runs the same exact primitive intersection test. `auto` keeps scenes with fewer
than 16 surfaces on Linear and selects BVH for larger scenes.

The following Apple M4 measurements use Python 3.12, a 16 kHz 1.2-second Mono
RIR, a warm Numba cache, and the median of five runs. Floorplan idx 0 contains
90 acoustic surfaces.

| Quality | Linear | BVH | BVH speedup |
| --- | ---: | ---: | ---: |
| `preview` | 63 ms | 55 ms | 1.15x |
| `simulation` | 139 ms | 110 ms | 1.26x |
| `fine` | 237 ms | 182 ms | 1.30x |
| `reference` | 437 ms | 332 ms | 1.32x |

Representative additional results:

| Scene | Mode | Time | Notes |
| --- | --- | ---: | --- |
| Small Geometry | `simulation`, Auto -> Linear | 109 ms | Avoids BVH overhead |
| Furnished Floorplan, 110 surfaces | `simulation`, Linear | 261 ms | Reference traversal |
| Furnished Floorplan, 110 surfaces | `simulation`, BVH | 125 ms | 2.08x speedup |

The first run in a new environment also includes Numba compilation and should
not be compared with these steady-state numbers. Runtime varies with CPU,
surface count, materials, adaptive bounce depth, receiver model, and RIR
length. Linear and BVH produced exactly equal hit surfaces, distances, normals,
and final same-room and cross-room RIR samples in the automated tests.

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

Current branch verification:

| Check | Result |
| --- | ---: |
| Unit and integration tests | 149 passed |
| Quick accuracy profile | 9/9 passed |
| Full accuracy profile | 9/9 passed |
| Source distribution and wheel | Passed `twine check` |

Run the longer profile and native Steam Audio comparison with:

```bash
acoustic-agent benchmark \
  --profile full \
  --steam-audio-root /path/to/steam-audio \
  --output benchmark-results/full
```

Both profiles generate machine-readable JSON plus Markdown and self-contained
HTML reports. If the Steam Audio SDK is unavailable, its native comparison is
reported as skipped rather than silently replaced.

## Research Workflows

Research code is kept under [`research/`](research/README.md), separate from
the short API examples and the simulation engine. The first study provides
line-of-sight DOA baselines for HRTF, linear-array, and circular-array
receivers in both Geometry and FloorPlan scenes:

```bash
python -m research.doa.run_los --quick
python -m research.doa.run_los
```

It compares direct-only propagation with LOS room simulation, reports angular
error, handles the linear array's mirror ambiguity explicitly, and writes
reproducible CSV, JSON, Markdown, and NPZ artifacts. See the
[LOS DOA study](research/doa/README.md) for coordinates, estimators, and
interpretation limits.

The [distributed FloorPlan study](research/doa/DISTRIBUTED_FLOORPLAN.md)
compares single microphones, synchronized and asynchronous array nodes, and a
hybrid deployment on disjoint train/test layouts. It evaluates static position,
room identity, cross-room coverage, door-crossing trajectories, and the number
of channels required by a traditional SRP-PHAT/TDOA/Kalman pipeline. Its
stratified protocol scans all 15,376 FloorPlans and validates ten unseen layouts
for every exact room count from 4 to 14, with floor-area-decile coverage:

```bash
python -m research.doa.run_stratified
```

## Documentation

| Guide | Purpose |
| --- | --- |
| [Python API](docs/API.md) | Scenes, models, motion, batching, and compatibility |
| [Configuration](docs/CONFIGURATION.md) | Shapes, materials, quality, and solver controls |
| [CUDA Acceleration](docs/CUDA_ACCELERATION.md) | Single-GPU setup, precision controls, and RTX benchmarks |
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
