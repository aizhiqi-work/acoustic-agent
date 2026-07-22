# Static connected-FloorPlan localization scaling study

## Protocol

- Task output: global `(x, y)` and `room_id`; motion is not modeled.
- Only fully connected FloorPlans with valid room geometry are eligible.
- Room-count strata: `4, 6, 8, 10, 12`; validation FloorPlans per stratum: `5`.
- Accuracy gate: median <= 1.0 m, P90 <= 2.0 m, room accuracy >= 85%.
- Each room contributes up to `2` candidate installation positions.
- Singles are globally synchronized and localized with onset TDOA.
- Each array node has `4` microphones; one node is a DOA-only baseline and two or more use DOA plus synchronized inter-array TDOA.
- Reflection tracer: `cuda` / `float32` / device `0`.

## Minimum tested configurations

| Rooms | Evidence | Sensor | Nodes | Physical microphones | Observed gate | 95% CI support | Median | P90 | Room accuracy |
|---:|---|---|---:|---:|---|---|---:|---:|---:|
| 4 | 5-plan | single microphones | 5 | 5 | yes | yes | 0.76 m | 1.13 m | 100.0% |
| 4 | 5-plan | 4-channel arrays | 4 | 16 | yes | yes | 0.41 m | 1.07 m | 95.0% |
| 6 | 5-plan | single microphones | 8 | 8 | yes | yes | 0.46 m | 0.80 m | 100.0% |
| 6 | 5-plan | 4-channel arrays | 5 | 20 | yes | yes | 0.31 m | 0.85 m | 93.3% |
| 8 | 5-plan | single microphones | 6 | 6 | yes | no | 0.68 m | 1.91 m | 92.5% |
| 8 | 5-plan | 4-channel arrays | 7 | 28 | yes | yes | 0.37 m | 0.88 m | 95.0% |
| 10 | 5-plan | single microphones | 8 | 8 | yes | yes | 0.45 m | 1.23 m | 100.0% |
| 10 | 5-plan | 4-channel arrays | 7 | 28 | yes | yes | 0.39 m | 1.31 m | 98.0% |
| 12 | 5-plan | single microphones | 8 | 8 | yes | yes | 0.46 m | 1.57 m | 98.3% |
| 12 | 5-plan | 4-channel arrays | 8 | 32 | yes | yes | 0.37 m | 1.53 m | 93.3% |

## Area sensitivity at the selected room-count configurations

| Sensor | Relative area within room-count stratum | Selected node range | Median | P90 | Room accuracy |
|---|---|---:|---:|---:|---:|
| 4-channel arrays | large | 4-8 | 0.35 m | 1.43 m | 98.5% |
| 4-channel arrays | medium | 4-8 | 0.39 m | 1.17 m | 95.7% |
| 4-channel arrays | small | 4-8 | 0.35 m | 1.23 m | 91.9% |
| single microphones | large | 5-8 | 0.57 m | 1.57 m | 98.5% |
| single microphones | medium | 5-8 | 0.60 m | 1.19 m | 95.7% |
| single microphones | small | 5-8 | 0.55 m | 1.24 m | 98.8% |

## Interpretation limits

Selection first uses the smallest configuration whose 95% cluster-bootstrap interval supports all three accuracy gates. If no tested configuration has that support, the table falls back to the smallest point estimate that passes and marks 95% CI support as `no`; if even the point estimate fails, it reports the best tested configuration. This report uses 5 validation FloorPlans per listed room-count stratum. Results assume one active source, known FloorPlan geometry, known sensor coordinates, open interior portals, and global clock synchronization. Real-device clock drift, sensor placement error, noise, and simulation-to-real transfer require separate validation.
