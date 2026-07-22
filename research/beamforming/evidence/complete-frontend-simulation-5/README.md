# Simulation-Quality Cross-Check

This paired cross-check reruns the first held-out layout in each 4, 6, 8, 10,
and 12-room stratum at `simulation` RIR quality. It contains five FloorPlans,
ten same-room/cross-room cases, and 140 pipeline rows. All other protocol
settings match the 25-layout preview benchmark.

| System | Pipeline | SI-SDR change | PESQ | PESQ change | STOI | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Single | Raw | 0.00 | 1.504 | 0.000 | 0.776 | 0.0000 |
| Local array | MVDR | 9.19 | 2.033 | 0.964 | 0.907 | 0.0049 |
| Local array | **MWF** | **15.57** | **2.578** | **1.510** | **0.914** | **0.0050** |
| Local array | WPE+MWF | 10.02 | 2.071 | 1.002 | 0.851 | 0.0400 |
| Distributed | MVDR | -5.21 | 2.336 | 0.832 | 0.810 | 0.0070 |
| Distributed | **MWF** | **9.63** | **2.788** | **1.283** | **0.943** | **0.0071** |
| Distributed | WPE+MWF | 5.15 | 2.572 | 1.068 | 0.926 | 0.1426 |
| Distributed | WPD | -13.36 | 1.135 | -0.369 | 0.473 | 0.0261 |

The higher-quality RIRs preserve the main benchmark ranking. Local MWF wins
SI-SDR in nine of ten cases. Distributed MWF wins the remaining case and gives
the highest average cross-room PESQ, 3.110. The WPE tradeoff and distributed
WPD failure also reproduce, so they are not preview-quality artifacts.

Files in this directory provide the aggregate, scenario, room-count, and
case-level measurements used for this cross-check.
