# Configuration Reference

## Quality Presets

| Name | Rays | Base bounces | RT field duration |
| --- | ---: | ---: | ---: |
| `preview` | 8,192 | 32 | 2.0 s |
| `simulation` | 32,768 | 64 | 2.0 s |
| `fine` | 65,536 | 96 | 2.0 s |
| `reference` | 131,072 | 96 | 2.0 s |

Floorplan cross-room simulation can increase the resolved bounce count to the
configured `cross_room_min_bounces` and up to `cross_room_max_bounces`.

## `AcousticAgent`

Important constructor arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `room` | `[6, 4, 2.8]` | `Room`, size, or room mapping |
| `shape` | `rectangle` | Geometry generator when `room` is a size |
| `quality` | `simulation` | Quality preset |
| `material_profile` | `None` | Surface absorption classes or `auto` |
| `material_seed` | `0` | Deterministic material sampling seed |
| `fs` | `16000` | Output sample rate in Hz |
| `duration_s` | `2.0` | Returned RIR duration |
| `receiver_model` | `mono` | Mono, HRTF, linear, or circular receiver |
| `source_model` | `omni` | Source directivity mapping |
| `acoustic_geometry` | `None` | Semantic furniture/acoustic objects |
| `config` | `None` | Full `SimConfig` override |

`config` takes precedence over `quality`, `fs`, and `duration_s` when supplied.

## Geometry Rooms

Supported shapes are `rectangle`, `triangle`, `polygon`, `circle`, `l_shape`,
`t_shape`, `trapezoid`, `u_shape`, and `fan_shape`. A size is `[x, y, z]` in
meters. Explicit `corners` are XY points and use `size[2]` as room height.

```python
room = {
    "shape": "rectangle",
    "size": [6.0, 4.0, 2.8],
    "material_profile": {
        "wall": "semi_reflective",
        "floor": "absorptive",
        "ceiling": "auto",
    },
    "material_seed": 42,
}
```

U-shape controls are normalized fractions of the width/depth:

- `opening_width`: opening width fraction.
- `opening_depth`: opening depth fraction.
- `opening_offset`: horizontal opening placement from 0 to 1.

## Floorplan Rooms

```python
agent = AcousticAgent.from_floorplan(
    idx=0,
    placement="cross_room",
    seed=42,
    material_seed=2026,
    room_height_m=2.8,
    position_height_m=1.4,
)
```

`placement` accepts:

- `random`: sample either same-room or connected cross-room placement.
- `same_room`: sample both endpoints in one candidate room.
- `cross_room`: sample endpoints in distinct connected rooms.

The scene index remains the original audited dataset index. Use
`FloorplanResource.resolve_index()` or the Web index endpoint when an unavailable
index must move to the nearest, next, previous, or random eligible scene.

## Material Profiles

Geometry profiles contain `wall`, `floor`, and `ceiling`. Floorplan adds `door`
and `window`. Values accept a concrete material ID, `auto`, or an absorption
class:

- `reflective`
- `semi_reflective`
- `absorptive`
- `highly_absorptive`

The sampler first chooses a semantic-compatible material family, then a published
six-band record. Selection and fallback details are available in
`room.metadata["material_selection"]`.

## Receiver Models

```python
{"type": "mono"}
{"type": "hrtf", "orientation_deg": 0, "interpolation": "bilinear"}
{"type": "linear", "count": 4, "spacing_m": 0.08, "orientation_deg": 0}
{"type": "circular", "count": 8, "radius_m": 0.12, "orientation_deg": 0}
```

HRTF-specific options include `sofa_path`, `interpolation`, `spatial_blend`, and
`loudness_normalization`. The default file is bundled CIPIC subject 124.

## Source Directivity

```python
{"type": "omni"}
{"type": "cardioid", "orientation_deg": 45}
{"type": "dipole", "orientation_deg": 45}
{"type": "focused", "orientation_deg": 45}
{"type": "weighted_dipole", "dipole_weight": 0.65, "dipole_power": 2.0}
```

Orientation is yaw in degrees. Directivity affects direct, diffracted, and
reflected contributions according to their departure direction.

## Motion

`sample_motion()` accepts `approach` and `random`, moving either `source` or
`receiver`. `distance_m` controls travel length. `keyframe_spacing_m` defaults
to 0.25 m; an explicit `keyframes` count overrides spacing-derived sampling.

In Floorplan, trajectories use room and portal connectivity. Cross-room travel is
routed through verified openings rather than interpolated through closed walls.

## Custom Floorplans

`FloorplanBuilder.from_text()` is deterministic and local. It supports metric
dimensions and bedroom, living, kitchen, bathroom, storage, and balcony counts
in compact Chinese or English descriptions. `FloorplanBuilder.validate()` checks
coverage, overlap, opening placement, references, and open-door connectivity.
`AcousticAgent.from_floorplan_spec()` compiles a valid spec into the normal
multi-room solver model.

The Custom JSON schema and zero-key image workflow are documented in
[`CUSTOM_FLOORPLAN.md`](CUSTOM_FLOORPLAN.md).

## Advanced `SimConfig`

Common controls:

| Field | Default | Purpose |
| --- | ---: | --- |
| `late_tail` | `True` | Enable the parametric FDN tail |
| `late_tail_start_s` | `0.08` | Earliest hybrid-tail start |
| `direct_occlusion` | `True` | Test line-of-sight occlusion |
| `direct_transmission` | `True` | Add transmitted direct energy |
| `diffraction_enabled` | `True` | Find diffraction paths |
| `diffraction_order` | `3` | Maximum diffraction order |
| `rt_num_rays` | `32768` | Traced source rays |
| `rt_num_bounces` | `96` | Base reflection depth |
| `rt_receiver_radius_m` | `0.25` | Receiver hit radius |
| `adaptive_cross_room_bounces` | `True` | Raise cross-room reflection depth |
| `cross_room_min_bounces` | `96` | Minimum cross-room depth |
| `cross_room_max_bounces` | `128` | Maximum cross-room depth |
| `seed` | `1729` | Deterministic simulation seed |

Use the quality facade unless a controlled experiment requires individual
fields. Record the complete resolved config with generated data.

## HTTP Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/geometry` | Geometry workbench |
| `GET` | `/floorplan` | Floorplan workbench |
| `GET` | `/custom` | Local custom-floorplan workbench |
| `GET` | `/api/v1/custom/capabilities` | Available local/provider features |
| `GET` | `/api/v1/custom/prompt?mode=image|text` | ChatGPT image/text JSON prompt |
| `GET` | `/api/v1/floorplan/stats` | Compiled-scene audit stats |
| `GET` | `/api/v1/floorplan/index` | Resolve eligible scene index |
| `GET` | `/api/v1/floorplan/scene` | Read an indexed scene |
| `GET` | `/api/v1/materials/semantics` | Material semantic catalog |
| `POST` | `/api/rir.wav` | Direct WAV response |
| `POST` | `/api/rir.npy` | Direct NumPy response |
| `POST` | `/api/v1/simulate` | Compact static simulation response |
| `POST` | `/api/v1/workbench` | Static workbench response |
| `POST` | `/api/v1/dynamic-workbench` | Multi-frame workbench response |
| `POST` | `/api/v1/custom/generate` | Generate and compile a local text scene |
| `POST` | `/api/v1/custom/validate` | Validate an editable Floorplan JSON spec |
| `POST` | `/api/v1/custom/compile` | Compile a valid spec for the workbench |

Legacy `POST /api/simulate` remains available for compatibility. API responses
enable CORS for local integration, but the server has no authentication.
