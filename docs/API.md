# Python API

The recommended interface has two steps for every scene type:

```python
agent = AcousticAgent.create(...)
result = agent.run()
```

The older constructors remain available for compatibility, but new applications
only need `create`, `run`, `run_sources`, `run_batch`, and `run_many`.

## Geometry

```python
from acoustic_agent import AcousticAgent

agent = AcousticAgent.create(
    scene="geometry",
    room={
        "shape": "rectangle",
        "size": [6.0, 4.0, 2.8],
        "material_profile": {"wall": "auto", "floor": "auto", "ceiling": "auto"},
    },
    source=[1.2, 1.1, 1.5],
    receiver=[4.7, 2.8, 1.4],
    quality="simulation",
)
rir = agent.run().rir
```

## Indexed Floorplan

Let the resource sample valid rooms and positions:

```python
agent = AcousticAgent.create(
    scene="floorplan",
    idx=0,
    placement="cross_room",  # random / same_room / cross_room
    seed=42,
    furnishing="balanced",   # sparse / balanced / compact / None
    quality="simulation",
)
rir = agent.run().rir
```

Pass `source` and `receiver` when exact positions are required. `agent.rooms`
contains available rooms and `agent.placement` records the resolved placement.

## Custom Floorplan

```python
from acoustic_agent import AcousticAgent, FloorplanBuilder

spec = FloorplanBuilder.from_text("10m x 8m, two bedrooms and one living room", seed=42)
agent = AcousticAgent.create(
    scene="custom",
    spec=spec,
    source_room="living_0",
    receiver_room="bedroom_1",
    seed=42,
)
rir = agent.run().rir
```

## Common Options

All scene modes share these options:

| Option | Default | Meaning |
| --- | --- | --- |
| `quality` | `simulation` | `preview`, `simulation`, `fine`, or `reference` |
| `duration_s` | `2.0` | Returned RIR length in seconds |
| `fs` | `16000` | Sample rate |
| `receiver_model` | `mono` | Mono, HRTF, linear, or circular receiver |
| `source_model` | `omni` | Source directivity |
| `material_profile` | scene default | Semantic material choices |
| `material_seed` | deterministic default | Material sampling seed |
| `acoustic_geometry` | empty | Explicit semantic furniture or acoustic objects |
| `visualization` | `False` | Collect representative RT paths for visualization |
| `intersection_backend` | `auto` | `auto`, `linear`, or `bvh` surface lookup |
| `config` | `None` | Full `SimConfig` override |

Convenience aliases are accepted by `create`: `mic` for the receiver position,
`microphone` for `receiver_model`, `directivity` for `source_model`, `objects`
for `acoustic_geometry`, `rir_length` for `duration_s`, and `sample_rate` for
`fs`. Prefer the canonical names in shared libraries and stored job manifests.

## Results

```python
result = agent.run()
result.rir       # float32 [channel, sample]
result.paths     # direct, diffraction, and portal paths in the default fast mode
result.rt60      # broadband and per-band decay estimates
result.metadata  # resolved solver, materials, topology, cache, and diagnostics
```

Mono agents use a headless fast path by default: the RIR and decay metrics are
unchanged, while representative RT paths and the unused FOA/Ambisonic buffer are
not generated. Set `visualization=True` when creating an agent to include RT
paths for a custom viewer. HRTF receivers always retain the FOA reconstruction
needed for binaural rendering. The WebGL workbench enables visualization
automatically.

`intersection_backend="linear"` keeps the exhaustive reference traversal,
while `"bvh"` forces the cached bounding-volume hierarchy. The default
`"auto"` uses Linear for small Geometry scenes and BVH when a scene has at
least 16 surfaces, which normally includes complete Floorplan scenes and
furnished rooms. Both backends call the same exact surface intersection code.

## Multiple Sources And Audio

An RIR describes propagation and does not depend on whether the emitted signal
is speech, music, or noise. Reuse one RIR when only the audio changes at the
same position. Simulate one RIR per source when signals originate at different
positions:

```python
from acoustic_agent import AcousticAgent, mix_audio_at_snr, render_audio

agent = AcousticAgent.create(
    scene="geometry",
    room={"shape": "rectangle", "size": [6.0, 4.0, 2.8]},
    source=[1.2, 1.1, 1.5],
    receiver=[4.7, 2.8, 1.4],
    fs=16000,
)

sources = agent.run_sources({
    "voice": {"position": [1.2, 1.1, 1.5], "source_model": "cardioid"},
    "piano_1": {"position": [4.8, 1.0, 1.2], "source_model": "omni"},
})

# These decoded arrays must use agent.config.fs.
voice_wet = render_audio(voice_samples, sources["voice"].rir)
piano_wet = render_audio(piano_samples, sources["piano_1"].rir)
room_mix = mix_audio_at_snr(voice_wet, piano_wet, snr_db=10.0, normalize=True)
```

`render_audio` downmixes a source recording to one emitting point and preserves
all receiver channels. `mix_audio_at_snr` scales the rendered background to a
receiver-domain broadband RMS target before summation. Optional final
normalization preserves that ratio. Use `resample_audio` when a decoded array
does not match the simulation sample rate.

The Web workbench performs browser-side WAV/MP3 decoding and includes Project
narration, Background speech, two Piano programs, a Pink-noise bed,
deterministic generated noise, and uploaded audio. Enabling its background
source adds a second position and RIR. Changing the program or SNR reuses
existing RIRs; changing source position requires a new simulation.

## Motion

Static and dynamic runs use the same method:

```python
static = agent.run()

dynamic = agent.run(motion={
    "mode": "approach",       # approach / random / through_portal
    "moving": "receiver",    # source / receiver
    "distance_m": 1.5,
    "keyframe_spacing_m": 0.25,
    "seed": 42,
})
rir_frames = dynamic.rirs
```

An already sampled `motion` mapping with `frames` can also be passed to `run`.
The lower-level `sample_motion` and `run_dynamic` methods remain public for
applications that edit keyframes themselves.

## Many Positions In One Scene

Build geometry and materials once, then submit plain coordinate pairs:

```python
pairs = [
    ([1.0, 1.0, 1.4], [4.0, 2.0, 1.4]),
    {"id": "pair_b", "source": [2.0, 1.0, 1.4], "receiver": [5.0, 3.0, 1.4]},
]

batch = agent.run_batch(pairs, workers=4)
batch.save_npz("room_rirs.npz")
```

This is the fastest high-level path when only source and receiver positions
change. Each item is also available through `batch.items` and `batch.rirs`.

## Dataset Production Across Scenes

Each job uses the same keys accepted by `AcousticAgent.create`; add `motion` for
a dynamic job and `id` for a stable dataset key:

```python
jobs = [
    {
        "id": f"scene_{idx}_take_{take}",
        "scene": "floorplan",
        "idx": idx,
        "placement": "random",
        "seed": 1000 + idx * 10 + take,
        "quality": "preview",
        "duration_s": 1.0,
        "fs": 16000,
    }
    for idx in range(100)
    for take in range(4)
]

dataset = AcousticAgent.run_many(jobs, workers=4)
dataset.save_npz("floorplan_rirs.npz")
```

The NPZ contains one float32 array per static RIR or dynamic frame plus a JSON
`manifest` with job IDs, scene settings, and array names. Keep an explicit seed
in every sampled Floorplan/Custom job. Use conservative worker counts because
each solver invocation also executes optimized numerical kernels.

The default `on_error="raise"` stops at the first invalid job. Long unattended
runs can use `on_error="skip"`; inspect `dataset.errors`, `dataset.succeeded`,
and `dataset.failed`. Failed job details are also stored in the NPZ manifest.

## Compatibility

The following APIs remain supported: `AcousticAgent(...)`,
`AcousticAgent.from_floorplan(...)`, `AcousticAgent.from_floorplan_spec(...)`,
`AcousticAgent.from_resplan(...)`, `simulate_rir(...)`, and
`simulate_batch(...)`. They are useful for lower-level integration, but are not
required for the common workflow.
