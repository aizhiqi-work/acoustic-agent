# Stratified FloorPlan Distributed Localization

- Database: **15,376** FloorPlans.
- Main room-count strata: **4-14 rooms**.
- Calibration: **5 FloorPlans per room count**.
- Unseen validation: **10 FloorPlans per room count**.
- Acoustic cases: one source point per room at `preview` RIR quality.
- Validation FloorPlans are selected one per area decile within each room-count stratum.
- Confidence intervals resample entire FloorPlans, not individual source points.

## Database population

| Rooms | Records | Connected | Eligible | Area P10 | Median | Area P90 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 3 (75.0%) | 3 | 8.3 m2 | 9.7 m2 | 13.1 m2 |
| 3 | 14 | 8 (57.1%) | 5 | 9.4 m2 | 13.3 m2 | 15.8 m2 |
| 4 | 108 | 95 (88.0%) | 94 | 18.1 m2 | 37.9 m2 | 47.1 m2 |
| 5 | 664 | 614 (92.5%) | 607 | 31.7 m2 | 42.2 m2 | 52.0 m2 |
| 6 | 3034 | 2889 (95.2%) | 2853 | 45.3 m2 | 57.1 m2 | 77.3 m2 |
| 7 | 3885 | 3644 (93.8%) | 3581 | 51.7 m2 | 71.4 m2 | 90.0 m2 |
| 8 | 2942 | 2704 (91.9%) | 2643 | 63.2 m2 | 84.9 m2 | 107.0 m2 |
| 9 | 2178 | 1959 (89.9%) | 1910 | 74.7 m2 | 100.3 m2 | 130.0 m2 |
| 10 | 1284 | 1111 (86.5%) | 1090 | 86.0 m2 | 113.1 m2 | 146.4 m2 |
| 11 | 588 | 472 (80.3%) | 458 | 97.7 m2 | 123.9 m2 | 167.6 m2 |
| 12 | 375 | 205 (54.7%) | 196 | 104.8 m2 | 134.6 m2 | 175.4 m2 |
| 13 | 197 | 80 (40.6%) | 77 | 112.0 m2 | 147.9 m2 | 193.6 m2 |
| 14 | 103 | 38 (36.9%) | 37 | 128.7 m2 | 168.4 m2 | 219.7 m2 |

Room counts 2 and 3 do not contain enough connected records for a disjoint 5+10 split and are reported in the population table but excluded from the balanced main experiment.

## Minimum synchronized single-microphone deployment

Adequacy requires median error <= 1.0 m, P90 <= 2.0 m, and room accuracy >= 85%.

| Rooms | Recommended mics | Adequate | Median (floorplan-bootstrap 95% CI) | P90 | Room accuracy | Cross-room median |
|---:|---:|:---:|---:|---:|---:|---:|
| 4 | 4 | yes | 0.72 m [0.57, 0.96] | 1.53 m | 100.0% | n/a |
| 5 | 4 | yes | 0.92 m [0.71, 1.13] | 1.41 m | 90.0% | 1.09 m |
| 6 | 4 | yes | 0.61 m [0.51, 0.85] | 1.87 m | 91.7% | 1.04 m |
| 7 | 5 | yes | 0.63 m [0.53, 0.71] | 1.43 m | 98.6% | 0.67 m |
| 8 | 5 | yes | 0.62 m [0.53, 0.66] | 1.88 m | 96.2% | 1.29 m |
| 9 | 6 | yes | 0.61 m [0.44, 0.67] | 1.55 m | 94.4% | 0.73 m |
| 10 | 6 | yes | 0.62 m [0.54, 0.84] | 1.55 m | 98.0% | 0.86 m |
| 11 | 7 | yes | 0.57 m [0.48, 0.67] | 1.50 m | 98.2% | 0.60 m |
| 12 | 7 | yes | 0.52 m [0.36, 0.67] | 1.56 m | 95.0% | 0.67 m |
| 13 | 7 | yes | 0.63 m [0.53, 0.76] | 1.82 m | 86.9% | 0.83 m |
| 14 | 7 | yes | 0.63 m [0.53, 0.79] | 1.78 m | 91.4% | 0.76 m |

## Floor-area sensitivity

Each row below uses the recommended microphone count for its own room-count stratum, so the relative-area comparison does not simply reward smaller homes for having fewer rooms.

| Within-room-count area bin | FloorPlans | Cases | Median | P90 | Room accuracy |
|---|---:|---:|---:|---:|---:|
| small | 43 | 377 | 0.55 m | 1.44 m | 94.4% |
| medium | 32 | 293 | 0.63 m | 1.60 m | 92.8% |
| large | 35 | 320 | 0.74 m | 1.82 m | 95.0% |

| Absolute floor area | FloorPlans | Cases | Median | P90 | Room accuracy |
|---|---:|---:|---:|---:|---:|
| compact_lt60 | 32 | 172 | 0.64 m | 1.51 m | 94.8% |
| medium_60_100 | 25 | 202 | 0.61 m | 1.41 m | 95.0% |
| large_100_150 | 34 | 376 | 0.57 m | 1.62 m | 94.7% |
| very_large_ge150 | 19 | 240 | 0.71 m | 1.92 m | 92.1% |

## Interpretation

The recommendation is the smallest tested topology-aware deployment that satisfies all three fixed validation thresholds. If none does, the table reports the lowest fixed accuracy/cost penalty and marks the row as not adequate. These are simulation conclusions for one active source, known microphone coordinates, known FloorPlan geometry, open interior portals, and globally synchronized microphones.
Across these strata, the observed minimum follows `min(7, max(4, ceil(rooms / 2) + 1))`: four microphones up to six rooms, five for seven to eight, six for nine to ten, and seven for eleven to fourteen rooms. This compact rule is an empirical summary of this validation set, not a universal law.
