# Stratified Distributed Beamforming: Preview 25

This benchmark uses 25 held-out FloorPlans: five each with 4, 6, 8, 10, and 12
rooms. Each layout contributes a same-room and a cross-room interfering-speaker
case. Every case also contains an independently rendered pink-noise source and
low-level independent sensor noise.

All steered algorithms share the same synchronized single microphones and
GCC-PHAT TDOA estimate. Microphone counts are 5, 7, 8, 8, and 8 for the five
room-count strata respectively. RIRs use the `preview` quality preset.

## Overall Results

| Algorithm | Cases | Mean SNR improvement | Mean SI-SDR improvement | Output PESQ | Output STOI | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Best single | 50 | 0.00 dB | 0.00 dB | 1.502 | 0.776 | 0.0000 |
| DS | 50 | -2.55 dB | -4.92 dB | 1.196 | 0.667 | 0.0041 |
| Weighted DS | 50 | +2.25 dB | -0.79 dB | 1.579 | 0.786 | 0.0041 |
| MVDR | 50 | +12.26 dB | -3.50 dB | 2.238 | 0.835 | 0.0071 |
| GEV | 50 | +9.94 dB | -6.45 dB | 2.239 | 0.855 | 0.0108 |
| MWF | 50 | +10.79 dB | +8.41 dB | 2.496 | 0.927 | 0.0071 |

## MWF by Room Count

| Rooms | Mics | Cases | SNR improvement | SI-SDR improvement | Output STOI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5 | 10 | +11.89 dB | +9.24 dB | 0.921 |
| 6 | 7 | 10 | +11.62 dB | +9.46 dB | 0.935 |
| 8 | 8 | 10 | +12.25 dB | +9.45 dB | 0.939 |
| 10 | 8 | 10 | +11.06 dB | +8.78 dB | 0.930 |
| 12 | 8 | 10 | +7.15 dB | +5.10 dB | 0.912 |

## Interpretation

- MWF is the best balanced baseline in this protocol. All 50 cases have
  positive SNR improvement, while mean SI-SDR and STOI also improve.
- MVDR gives the largest mean SNR reduction of interference, but its negative
  SI-SDR shows that it distorts the desired speech more than MWF.
- Equal-weight DS is not robust for a whole-home sparse array. Cross-room
  microphones with weak or multipath-dominated target arrivals can reduce SNR.
- Reliability-weighted DS repairs much of that failure but remains weaker than
  covariance-based methods.
- The 12-room stratum is harder even with eight microphones; MWF SNR gain drops
  to 7.15 dB.
- With the fixed target image at the reference microphone, MWF improves PESQ
  from 1.502 to 2.496. With the original dry source as reference it remains
  near 1.058 because none of these algorithms performs dereverberation.

TDOA localization over the 50 cases has 96% room accuracy, 0.77 m median error,
and 2.29 m P90 error. These results assume sample-synchronous nodes and should
not be generalized to asynchronous hardware without a clock-error sweep.

Primary SI-SDR, PESQ, and STOI use one fixed interference-free target image at
the calibrated reference microphone. Dry-source-reference diagnostics are also
stored in `details.csv`; they measure the combined denoising and dereverberation
task and should not replace the enhancement metrics.

Machine-readable evidence:

- [`details.csv`](details.csv): all 300 case/algorithm rows.
- [`overall.csv`](overall.csv): aggregate by algorithm.
- [`scenario-algorithm.csv`](scenario-algorithm.csv): same-room/cross-room split.
- [`room-algorithm.csv`](room-algorithm.csv): room-count scaling table.
- [`localization.csv`](localization.csv): TDOA localization accuracy.
