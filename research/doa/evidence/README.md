# Reproducible DOA Evidence

This directory keeps compact result snapshots for code review and citation.
They were generated with the fixed seeds and RIR quality documented by each
experiment. Large RIR caches, observations, spectra, and NPZ files are excluded
from Git and can be regenerated with the commands below.

| Evidence | Command | Contents |
|---|---|---|
| `doa-los/` | `python -m research.doa.run_los` | Geometry/FloorPlan LOS HRTF, linear-array, and circular-array DOA |
| `distributed-floorplan/` | `python -m research.doa.run_distributed` | Array/single/hybrid, synchronization error, and motion comparison |
| `distributed-floorplan-stratified/` | `python -m research.doa.run_stratified` | 15,376-record audit and 110-FloorPlan room-count/area validation |
| `cuda-4090/` | Add `--accelerator cuda --precision float32` | RTX 4090 Preview/Simulation parity, runtime, and quality comparison |

The stratified evidence contains only population, split, recommendation, and
area-summary CSV files. Its complete 8,030-row result table is reproducibly
generated under `research/results/`, which remains ignored to keep clones
small. Reported confidence intervals resample complete FloorPlans.
