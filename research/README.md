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
- [`beamforming/`](beamforming/README.md): single-channel enhancement, local
  four-channel arrays, topology-aware distributed synchronized-mic
  beamforming, WPE dereverberation, WPD, and the complete classical audio
  front-end benchmark, including fixed whole-home single/array hybrid
  deployment with TDOA routing. The completed protocol and flowchart are in
  [`beamforming/WHOLE_HOME_DISTRIBUTED.md`](beamforming/WHOLE_HOME_DISTRIBUTED.md).

Generated artifacts are written under `research/results/` and are ignored by
Git.
