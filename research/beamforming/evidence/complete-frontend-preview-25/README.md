# Complete Audio Front-End Result

This directory records the fixed classical front-end benchmark run on
2026-07-23. It contains 25 held-out FloorPlans: five layouts for each exact
room count of 4, 6, 8, 10, and 12. Every layout contributes a same-room and a
cross-room competing-speaker case, producing 50 cases and 700 metric rows.

## Conditions

- Target: `main_voice.wav`.
- Interferer: `background_speech.wav`, independently RIR-rendered.
- Background: independently positioned pink-noise source plus sensor noise.
- RIR quality: `preview`, 1.0 s at 16 kHz.
- Evaluation audio: 2.5 s.
- Runtime host: Apple M4, 16 GB RAM, arm64 macOS, Python 3.12.13, and NumPy
  2.4.6. RTF values are enhancement-only measurements on this host.
- Distributed microphone counts: 5, 7, 8, 8, and 8 for 4, 6, 8, 10, and 12
  rooms, respectively, following the fixed whole-home deployment policy.
- Local array: four-channel circular array with 5 cm radius in the target
  room.
- Steering: SRP-PHAT for the local array and synchronized GCC-PHAT TDOA for
  distributed microphones.

Primary PESQ, STOI, and SI-SDR use a fixed clean target image at each
architecture's calibrated reference microphone. Dry-reference metrics compare
against the original anechoic source and include both residual noise and room
reverberation. Each architecture is level-calibrated over its own microphone
set, so output scores compare complete systems while change scores isolate the
processing contribution.

## Overall Results

| System | Pipeline | Mics | SNR change | SI-SDR change | PESQ | PESQ change | STOI | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | Raw | 1 | 0.00 | 0.00 | 1.501 | 0.000 | 0.776 | 0.0000 |
| Single | Wiener | 1 | 2.58 | 1.20 | 1.463 | -0.038 | 0.756 | 0.0044 |
| Local array | DS | 4 | 1.13 | 0.79 | 1.081 | 0.013 | 0.656 | 0.0025 |
| Local array | MVDR | 4 | 12.32 | 8.88 | 1.835 | 0.767 | 0.902 | 0.0042 |
| Local array | **MWF** | **4** | **18.60** | **15.02** | **2.430** | **1.362** | **0.912** | **0.0043** |
| Local array | WPE+MWF | 4 | 15.98 | 9.80 | 1.975 | 0.907 | 0.852 | 0.0370 |
| Distributed | Weighted DS | 5-8 | 2.25 | -0.79 | 1.580 | 0.078 | 0.786 | 0.0042 |
| Distributed | MVDR | 5-8 | 12.26 | -3.51 | 2.238 | 0.737 | 0.835 | 0.0072 |
| Distributed | **MWF** | **5-8** | **10.79** | **8.40** | **2.497** | **0.996** | **0.927** | **0.0073** |
| Distributed | WPE+MWF | 5-8 | 10.15 | 4.53 | 2.328 | 0.826 | 0.910 | 0.1447 |
| Distributed | WPD | 5-8 | 1.88 | -8.62 | 1.256 | -0.245 | 0.593 | 0.0260 |

Local MWF improves wet-reference SI-SDR in all 50 cases and is the best
SI-SDR pipeline in 46. Distributed MWF improves all 50 cases and wins the
remaining four. For cross-room interferers, distributed MWF reaches PESQ
2.879 and STOI 0.956; local MWF remains stronger in SNR and SI-SDR change.

Across 10,000 deterministic case bootstraps, local MWF's mean SI-SDR change
has a 95% interval of 14.03 to 15.97 dB and its PESQ change interval is 1.20
to 1.54. Distributed MWF's corresponding intervals are 7.37 to 9.42 dB and
0.90 to 1.09. Both pipelines improve PESQ, STOI, SNR, and SI-SDR in all 50
individual cases.

WPE slightly improves the difficult dry-reference diagnostic for some cases,
but its average pure-enhancement metrics are below MWF without WPE. WPD is
stable for the compact array but fails on the large-aperture distributed
network despite TDOA pre-alignment. The recommended defaults are therefore
local MWF and distributed MWF, with WPE exposed as an offline optional stage.

A paired five-layout rerun at `simulation` RIR quality preserves this ranking:
local MWF reaches 15.57 dB SI-SDR improvement and PESQ 2.578, while distributed
MWF reaches 9.63 dB and PESQ 2.788. See
[`../complete-frontend-simulation-5/`](../complete-frontend-simulation-5/README.md).

## Files

- `overall.csv`: one row per architecture and pipeline.
- `scenario-pipeline.csv`: same-room versus cross-room aggregates.
- `room-pipeline.csv`: aggregates for each exact room count.
- `details.csv`: all 700 case-level rows.

Reproduce with:

```bash
python -m research.beamforming.run_frontend_benchmark \
  --output research/results/complete-frontend-preview-25 \
  --room-counts 4 6 8 10 12 \
  --plans-per-room-count 5 \
  --quality preview \
  --duration 2.5 \
  --rir-duration 1.0
```
