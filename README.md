# Acoustic Agent

Acoustic Agent is a compact indoor acoustic simulation engine for generating room impulse responses (RIRs) from editable room geometry, material presets, source positions, and receiver models.

## Current Solver

The single-room solver is intentionally unified around three acoustic components:

- Direct sound: distance attenuation, air absorption, optional occlusion/transmission.
- Diffraction: UTD-style edge paths for non-line-of-sight geometry.
- RT field: path-traced reflection energy, RT60 estimation, and reconstructed reflection/reverb IR.

Low-order ISM diagnostics are no longer part of the public single-room pipeline. Reflections are represented by the RT energy field, including directional FOA data for HRTF rendering when available.

## RT Path Display

The Web UI does not draw every traced RT event. The solver still traces the full `rt_num_rays x rt_num_bounces` field, while the displayed `rt_reflection` paths are representative samples from that field.

The current display retention policy is `stratified_order_then_strongest_gain`:

- Use the same rays and bounces as the simulation, not a separate preview setting.
- Build a larger candidate pool of receiver-hit RT paths.
- Bucket candidates by reflection order.
- Keep more low-order paths for readability, but reserve slots for mid- and high-order RT paths.
- Within each order bucket, keep the strongest-gain paths first.

This avoids two bad visualizations: early paths monopolizing the display, and high-order floor/ceiling bounces overwhelming the scene. The metadata reports this as `rt_visual.retention_policy`.

## Receiver Models

- Mono pressure RIR.
- SOFA HRTF binaural render. The bundled default is CIPIC subject 124, matching the source data for Steam Audio's built-in default HRTF; pass `sofa_path` to use another SOFA file.
- Linear microphone array.
- Circular microphone array.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python examples/basic_room.py
python scripts/run_web.py
python scripts/run_resplan_web.py
```

The geometry workbench runs at `http://127.0.0.1:8765`. The ResPlan workbench runs independently at `http://127.0.0.1:8766` and loads `../ResPlan.pkl` by default. Use its plan index, source-room selector, and microphone-room selector to switch between the established same-room solver and the multi-room portal model.

Pass another dataset location when needed:

```bash
python scripts/run_resplan_web.py --dataset /path/to/ResPlan.pkl --port 8766
```

ResPlan coordinates are converted to meters with the audited scale policy below. Same-room scenes retain the original full-height equivalent door/window boundary model. When the source and microphone are in different rooms, the engine builds the whole apartment in global coordinates: verified interior and balcony doors become open `0-2.1 m` portals with reflective lintels, wall-free room connections become full-height portals, entry or unmatched doors remain closed door surfaces, and windows remain closed glass surfaces. Balconies are exposed as source and microphone room candidates. The cross-room solver combines direct visibility or weak wall transmission, deterministic routing through the verified portal graph, and Numba-parallel visual and energy-field tracing through the same openings. The JIT intersection kernel preserves each wall segment's vertical range, including door lintels and window spans. Its energy kernel consumes the same NumPy PCG64 scattering values and sample indices as the reference vectorized solver, preserving the traced echogram while accelerating the computation.

### ResPlan filtering

The ResPlan workbench keeps the source dataset index but navigates only audited single-floor scenes. The audit removes duplicate room polygons, rejects implausible room geometry, and filters plans with stairs, disconnected inner polygons, excessive room counts, or material room overlaps. Metric scale uses a plausible `net_area` first, a gross-area proxy second, and wall depth only as a final fallback.

`GET /api/v1/resplan/stats` returns the current audit counts. `GET /api/v1/resplan/index?idx=2&direction=nearest` resolves a raw dataset index to an eligible index; `direction` also accepts `next`, `previous`, and `random`. Scene metadata includes normalized room connections such as `via_door`, `via_window`, `via_opening`, and outdoor entry or balcony targets. Cross-room propagation through these connections is intentionally not enabled in the same-room solver yet.

## HTTP API

Get an exact float32 RIR as a WAV file with one request. The JSON body accepts the same room, source, receiver, material, and quality fields as the workbench.

```bash
curl -X POST http://127.0.0.1:8765/api/rir.wav \
  -H 'content-type: application/json' \
  --data '{"shape":"rectangle","size":[6,4,2.8],"source":[1.2,1.1,1.5],"receiver":[4.7,2.8,1.4],"quality":"preview"}' \
  --output rir.wav
```

Python users can request `/api/rir.npy` and load the exact `[channel, sample]` float32 array directly with `numpy.load`. `POST /api/v1/simulate` runs once and returns only a compact result summary with `files.wav` and `files.npy` download links. The original `POST /api/simulate` remains available for backward compatibility.

## Python API

The shortest object-oriented form is:

```python
from acoustic_agent import AcousticAgent

agent = AcousticAgent(room=[6, 4, 2.8], quality="simulation")
rir = agent.run(source=[1.2, 1.1, 1.5], receiver=[4.7, 2.8, 1.4]).rir
```

Use the lower-level functions when individual solver settings need to be controlled:

```python
from acoustic_agent import SimConfig, make_room, microphone_array, simulate_rir

room = make_room(
    "rectangle",
    size=(6.0, 4.0, 2.8),
    materials={"wall": "wall", "floor": "carpet", "ceiling": "ceiling"},
)

config = SimConfig(
    fs=16000,
    duration_s=1.2,
    rt_num_rays=32768,
    rt_num_bounces=24,
    rt_duration_s=1.5,
)

result = simulate_rir(
    room,
    source=(1.2, 1.0, 1.5),
    receiver=(4.8, 2.8, 1.4),
    config=config,
    receiver_model=microphone_array("hrtf"),
)

print(result.rir.shape)
print(result.rt60)
print(result.paths[0])
```

## Web Workbench

The Web UI exposes the same core API: `make_room(...)`, `SimConfig(...)`, `microphone_array(...)`, and `simulate_rir(...)`. Editing the room, material, source, receiver, quality, or receiver model sends a local request to the Python engine and redraws direct, diffraction, and RT sample paths.
