# Floorplan Scenes

Floorplan is Acoustic Agent's residential-layout simulation mode. It converts
indexed 2D plans into complete multi-room acoustic geometry and uses the same
RIR solver as Geometry mode.

## Data source

The bundled resource is derived from **ResPlan**, a dataset of approximately
17,000 residential floor plans with vector geometry, architectural annotations,
and room-connectivity graphs:

> Mohamed Abouagour and Eleftherios Garyfallidis. "ResPlan: A Large-Scale
> Vector-Graph Dataset of 17,000 Residential Floor Plans." arXiv:2508.14006,
> 2025. https://doi.org/10.48550/arXiv.2508.14006

- Paper: https://arxiv.org/abs/2508.14006
- Dataset: https://www.kaggle.com/datasets/resplan/resplan
- Dataset license: CC BY-NC-SA 4.0

Please cite the upstream dataset when publishing results that use Floorplan V1:

```bibtex
@article{abouagour2025resplan,
  title = {ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans},
  author = {Abouagour, Mohamed and Garyfallidis, Eleftherios},
  journal = {arXiv preprint arXiv:2508.14006},
  year = {2025},
  doi = {10.48550/arXiv.2508.14006},
  url = {https://arxiv.org/abs/2508.14006}
}
```

ResPlan is used only as the upstream dataset name. Acoustic Agent calls the
converted simulation capability and its API **Floorplan**.

## Conversion

The source collection contains 17,107 records. The audit retains 15,376 scenes
and excludes 1,731 records for one or more of these primary reasons:

| Filter reason | Records |
| --- | ---: |
| Stair or multilevel layout | 680 |
| Overlapping rooms | 597 |
| No valid living room | 390 |
| Too many rooms | 170 |
| Disconnected interior geometry | 118 |

Individual records can carry more than one reason. During conversion, Acoustic
Agent also drops invalid/duplicate room polygons, estimates metric scale,
reconstructs boundary features, creates full-height walls and horizontal floor
and ceiling surfaces, and represents verified open doors as acoustic portals.

The resulting SQLite resource stores only solver-ready data. The original
pickle is not loaded during normal use.

## Acoustic model

Each indexed scene contains:

- room polygons, room semantics, and metric area;
- wall, floor, and ceiling surfaces;
- closed window surfaces;
- verified interior doors and wall-free connections represented as portals;
- closed unmatched entrance doors;
- source-room and receiver-room connectivity; and
- valid source/receiver sampling regions.

Same-room and cross-room runs use the same complete scene. Cross-room paths are
routed through verified portals, and coupled-room decay statistics use the room
and opening graph.

## Python API

```python
from acoustic_agent import AcousticAgent

agent = AcousticAgent.from_floorplan(
    idx=0,
    placement="same_room",  # random / same_room / cross_room
    seed=42,
    material_seed=2026,
    receiver_model={"type": "mono"},
    source_model={"type": "omni"},
    furnishing={"mode": "auto", "compactness": "balanced", "seed": 42},
    quality="simulation",
    duration_s=2.0,
    fs=16000,
)

print(agent.rooms)
print(agent.placement)
rir = agent.run().rir
```

`from_resplan()` remains a compatibility alias for v0.1 code. New code should
use `from_floorplan()`.

## Semantic furniture

Floorplan and Custom scenes can generate deterministic, editable furniture from
room semantics:

```python
agent = AcousticAgent.from_floorplan(
    idx=0,
    furnishing={
        "mode": "auto",
        "compactness": "balanced",  # sparse / balanced / compact
        "seed": 42,
    },
)
print(agent.furnishing["summary"])
```

Large objects are aligned to usable wall segments, free-standing objects are
sampled from valid room interiors, and door/portal clearances plus source and
receiver positions are protected. Visual objects use the existing editable
furniture representation and enter the solver through simplified acoustic
proxies. In the Web workbench, **Auto place** replaces only previous automatic
objects; manually added or edited furniture remains untouched.

## Web workbench

```bash
acoustic-agent web
```

Open http://127.0.0.1:8765/floorplan. The legacy `/resplan` URL remains a
compatibility route.

## Data terms

The engine code is Apache-2.0. The converted Floorplan V1 resource remains under
CC BY-NC-SA 4.0 because it is adapted from ResPlan. It requires attribution,
noncommercial use, and ShareAlike distribution of adaptations. See the packaged
`resources/floorplan/DATA_LICENSE.md` for the complete project notice.
