# LOS DOA Report

- Sample rate: 16000 Hz
- Speed of sound: 343.0 m/s
- Simulation quality: `preview`
- Linear-array errors are evaluated against the mirror-equivalent half-plane bearing.
- `direct` isolates the direct path; `room` retains reflections and late reverberation while LOS remains open.

| Scene | Condition | Receiver | Cases | Mean error | Max error |
|---|---|---:|---:|---:|---:|
| floorplan | direct | circular8 | 3 | 0.11 deg | 0.16 deg |
| floorplan | direct | hrtf | 3 | 1.34 deg | 2.01 deg |
| floorplan | direct | linear4 | 3 | 0.11 deg | 0.16 deg |
| floorplan | room | circular8 | 3 | 1.67 deg | 2.84 deg |
| floorplan | room | hrtf | 3 | 2.01 deg | 4.01 deg |
| floorplan | room | linear4 | 3 | 2.34 deg | 4.01 deg |
| geometry | direct | circular8 | 3 | 0.00 deg | 0.00 deg |
| geometry | direct | hrtf | 3 | 0.33 deg | 1.00 deg |
| geometry | direct | linear4 | 3 | 0.00 deg | 0.00 deg |
| geometry | room | circular8 | 3 | 4.67 deg | 7.00 deg |
| geometry | room | hrtf | 3 | 0.33 deg | 1.00 deg |
| geometry | room | linear4 | 3 | 7.00 deg | 10.00 deg |

Detailed per-case values are stored in `summary.csv`; RIRs, observations, search grids, and spectra are stored in the NPZ artifacts.
