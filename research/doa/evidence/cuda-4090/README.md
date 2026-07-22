# RTX 4090 CUDA DOA Experiments

This evidence records the first DOA and distributed-localization runs using
the single-GPU CUDA FP32 reflection tracer. Large NPZ observations and acoustic
measurement caches remain under `research/results/` and are intentionally not
committed.

## Environment

- Date: 2026-07-22
- GPU: one NVIDIA RTX 4090
- CUDA: 12.4
- Python: 3.12.13
- Numba: 0.65
- Accelerator: `cuda`
- Precision: `float32`
- CUDA device inside the isolated process: `0`
- Test suite: 153 passed
- Static-scaling focused tests: 8 passed

The process exposed only the selected RTX 4090 through `CUDA_VISIBLE_DEVICES`.
Every output payload records the accelerator, precision, and logical device.
Distributed measurement-cache keys include those values and use cache version
4, preventing reuse of the earlier CPU observations.

## Commands

```bash
python -m research.doa.run_los \
  --quality preview \
  --accelerator cuda --precision float32 --cuda-device 0 \
  --output-dir research/results/doa-los-cuda-4090

python -m research.doa.run_distributed \
  --quality preview \
  --accelerator cuda --precision float32 --cuda-device 0 \
  --output-dir research/results/distributed-floorplan-cuda-4090

python -m research.doa.run_static_scaling \
  --quality preview \
  --room-counts 4 6 8 10 12 \
  --calibration-per-count 5 --validation-per-count 5 \
  --accelerator cuda --precision float32 --cuda-device 0 \
  --output-dir research/results/static-connected-floorplan-scaling-cuda-4090-5x5
```

The same commands were repeated at `simulation` quality. The stratified
pipeline was validated with `run_stratified --quick` at Preview quality.

## Runtime

Times include process startup, CUDA JIT/cache setup, RIR generation, signal
rendering, localization, and report generation.

| Experiment | Quality | Cases | Elapsed |
|---|---|---:|---:|
| LOS DOA | Preview | 36 estimates | 17.19 s |
| LOS DOA | Simulation | 36 estimates | 16.72 s |
| Distributed FloorPlan | Preview | 35 static targets plus 24 motion frames | 127.22 s |
| Distributed FloorPlan | Simulation | 35 static targets plus 24 motion frames | 141.96 s |
| Stratified quick | Preview | 4 FloorPlans, 46 localization cases | 24.27 s uncached |
| Static connected FloorPlan scaling | Preview | 25 FloorPlans, 2,800 localization cases | 623.9 s uncached plan time |
| Static result filtering and report | Preview | Same 25 FloorPlans | 10.4 s |

The LOS Simulation run was not slower than Preview after warmup despite using
four times as many rays. Distributed runtime rose by 11.6% because the CPU
SRP-PHAT, TDOA grid likelihood, fusion, and tracking stages are unchanged.

## LOS DOA

Both quality levels produced a mean error of 1.4921 degrees and a maximum error
of 10 degrees over all 36 estimates. Moving from Preview to Simulation changed
only two Geometry room-array estimates by one degree in opposite directions.

| Scene | Condition | Receiver | Mean error | Max error |
|---|---|---|---:|---:|
| Geometry | Room | HRTF | 0.33 deg | 1.00 deg |
| Geometry | Room | Linear 4 | 7.00 deg | 10.00 deg |
| Geometry | Room | Circular 8 | 4.67 deg | 7.00 deg |
| FloorPlan | Room | HRTF | 0.67 deg | 1.16 deg |
| FloorPlan | Room | Linear 4 | 2.34 deg | 4.01 deg |
| FloorPlan | Room | Circular 8 | 1.67 deg | 2.84 deg |

Against the committed CPU evidence, 34 of 36 per-case errors were unchanged.
Two HRTF estimates for FloorPlan 0 seed 43 improved; this cannot be attributed
solely to CUDA because the direct-only case also changed.

## Distributed DOA And TDOA

Both qualities selected six topology-aware, globally synchronized single
microphones as the smallest adequate deployment.

| Metric | Preview | Simulation |
|---|---:|---:|
| Static median error | 0.6258 m | 0.6426 m |
| Static P90 error | 1.6885 m | 1.6885 m |
| Static room accuracy | 94.29% | 94.29% |
| Cross-room median error | 1.1052 m | 0.8104 m |
| Cross-room room accuracy | 81.82% | 81.82% |
| Motion raw median error | 0.9969 m | 0.9046 m |
| Motion filtered median error | 0.9433 m | 0.7374 m |
| Motion trend error | 6.2870 deg | 6.3007 deg |
| Motion room accuracy | 95.83% | 91.67% |
| Median absolute portal delay | 0.0 frames | 0.5 frames |

Preview CUDA reproduced all committed CPU deployment medians and room
accuracies. Two array configurations changed P90 by at most 0.1907 m, without
changing the selected deployment. Simulation improved cross-room distance and
motion-position errors, but one additional motion frame received the wrong room
and portal timing shifted. Higher ray count is therefore useful evidence, not a
monotonic accuracy guarantee for the current onset-threshold estimator.

## Stratified Quick Validation

The quick run evaluated four independent FloorPlans in the 4-room and 5-room
strata. It produced 46 localization cases. Four microphones did not meet all
fixed thresholds for the sampled 4-room subset (1.0735 m median, 2.1015 m P90,
100% room accuracy); five microphones passed for the sampled 5-room subset
(0.6419 m median, 1.1759 m P90, 100% room accuracy).

The run also exposed and fixed a report-only bug: quick samples do not always
contain every area bin. Missing bins now render as `0 / n/a` instead of raising
`KeyError` after acoustic simulation completes.

## Static Whole-Home Position And Room

The study removes motion and estimates only global `(x, y)` plus `room_id`. It
admits only fully connected, geometry-valid FloorPlans, with two candidate
installation positions per room. It evaluates five room-count strata: 4, 6,
8, 10, and 12 rooms, with five area-stratified validation FloorPlans in each.

Globally synchronized single microphones use onset TDOA. Four-channel array
nodes use SRP-PHAT DOA plus synchronized inter-array TDOA. The accuracy gate is
median error <= 1 m, P90 <= 2 m, and room accuracy >= 85%. Recommendations
prefer the smallest configuration whose FloorPlan-clustered 95% bootstrap
interval supports all three gates.

For 4, 6, 8, 10, and 12 rooms, the selected single-microphone counts are 5, 8,
6, 8, and 8. The selected 4-channel array-node counts are 4, 5, 7, 7, and 8,
equivalent to 16, 20, 28, 28, and 32 physical microphones. Arrays have lower
median position error, but require substantially more physical microphones.
The 8-room single-mic point estimate passes the gate, but no tested
configuration up to eight singles has 95% interval support for all three
metrics. The other four selected single-mic configurations and all five array
configurations have 95% interval support in this five-plan study.

Detailed recommendations, per-configuration metrics, split membership,
runtime rows, and area sensitivity are under
`static-connected-floorplan-scaling/`. Large measurement caches remain
uncommitted under `research/results/`.
