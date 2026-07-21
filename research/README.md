# Research

This directory contains reproducible experiments built on top of the public
Acoustic Agent API. It is intentionally separate from `examples/`:

- `examples/` demonstrates the shortest supported API calls.
- `research/` owns estimators, sweeps, metrics, reports, and cached results.
- `acoustic_agent/` remains the simulation engine and does not depend on a
  particular downstream research algorithm.

Current studies:

- [`doa/`](doa/README.md): line-of-sight direction-of-arrival baselines for
  HRTF, linear-array, and circular-array receivers in Geometry and FloorPlan
  scenes, plus [distributed whole-home localization and tracking](doa/DISTRIBUTED_FLOORPLAN.md).

Generated artifacts are written under `research/results/` and are ignored by
Git.
