# FloorPlan Distributed Localization Study

- Training FloorPlans: `[0, 2, 7, 13]`
- Unseen test FloorPlans: `[20, 29, 41, 60]`
- Selected minimax risk quantile: `0.05`
- RIR quality: `preview`
- Placement uses only room polygons, room areas, and portal topology. Test target positions are never used.
- Array nodes use SRP-PHAT DOA. Synchronized single microphones use onset TDOA. Hybrid fusion uses both.

## Static localization

| Configuration | Channels | Median | P90 | Room accuracy | Cross-room-only median | Cross-room room accuracy |
|---|---:|---:|---:|---:|---:|---:|
| array_1x8 | 8 | 5.94 m | 10.52 m | 11.4% | 6.12 m | 3.2% |
| array_2x4_async | 8 | 4.22 m | 10.26 m | 17.1% | 5.72 m | 3.7% |
| array_3x4_async | 12 | 4.20 m | 9.20 m | 20.0% | 4.23 m | 4.3% |
| array_4x4_async | 16 | 4.27 m | 9.26 m | 20.0% | 4.43 m | 0.0% |
| array_2x4_sync | 8 | 1.38 m | 5.19 m | 60.0% | 1.60 m | 55.6% |
| array_3x4_sync | 12 | 0.88 m | 4.80 m | 74.3% | 1.55 m | 65.2% |
| array_4x4_sync | 16 | 0.37 m | 3.50 m | 85.7% | 1.11 m | 73.7% |
| single_3x1 | 3 | 0.96 m | 3.59 m | 65.7% | 1.62 m | 47.8% |
| single_4x1 | 4 | 0.74 m | 3.57 m | 74.3% | 1.88 m | 52.6% |
| single_6x1 | 6 | 0.63 m | 1.69 m | 94.3% | 1.11 m | 81.8% |
| single_6x1_100us | 6 | 0.60 m | 1.82 m | 94.3% | 1.11 m | 81.8% |
| single_6x1_500us | 6 | 0.62 m | 1.87 m | 94.6% | 1.06 m | 84.1% |
| single_8x1 | 8 | 0.41 m | 1.33 m | 100.0% | 0.41 m | 100.0% |
| single_6x1_largest | 6 | 1.15 m | 2.49 m | 74.3% | 2.18 m | 18.2% |
| single_6x1_farthest | 6 | 0.77 m | 2.76 m | 82.9% | 1.44 m | 45.5% |
| hybrid_2x4_4x1 | 12 | 0.54 m | 2.30 m | 88.6% | 0.91 m | 77.8% |

## Selected configuration

The study selector chose **single_6x1** with 6 total channels.
The selector first enforces median <= 1.0 m, P90 <= 2.0 m, and room accuracy >= 85%; if no configuration satisfies all thresholds, it minimizes a fixed accuracy/cost penalty.

## Motion and portal crossing

The selected six-single-microphone deployment was evaluated on 4 unseen door-crossing trajectories (24 frames). A constant-velocity Kalman filter was applied after independent frame localization.

| Raw median | Filtered median | Trend direction error | Room accuracy | Portal transition delay |
|---:|---:|---:|---:|---:|
| 1.00 m | 0.94 m | 6.3 deg | 95.8% | 0.0 frames |

## Interpretation limits

This is a simulation study with known sensor coordinates and synchronized single microphones. The matched FloorPlan and portal graph are available to the estimator. Clock drift, sensor position error, multiple simultaneous speakers, and simulation-to-real mismatch remain separate experiments.
