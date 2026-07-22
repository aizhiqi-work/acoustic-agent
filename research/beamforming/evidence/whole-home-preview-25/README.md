# Fixed Whole-Home Microphone Evidence

This evidence set evaluates fixed, target-independent microphone deployments
on 25 held-out FloorPlans: five layouts for each of 4, 6, 8, 10, and 12
rooms. Each layout contributes an array-covered target room and an
array-uncovered target room under same-room and cross-room interference, for
100 paired cases and 1,000 strategy rows.

## Protocol

- Distributed single microphones: `5 / 7 / 8 / 8 / 8` for
  `4 / 6 / 8 / 10 / 12` rooms.
- Fixed circular arrays: one four-channel array for 4/6/8-room homes and two
  for 10/12-room homes.
- Sensors are placed before the target room and position are sampled.
- A target-active segment supplies GCC-PHAT TDOA and target covariance; a
  target-silent segment supplies noise covariance.
- Target, interfering speech, and pink-noise gains are calibrated once at an
  oracle target-room reference and reused at every sensor. Architectures are
  not independently normalized to the same input SNR.
- RIR quality is `preview`, audio is 2.5 s at 16 kHz, and enhancement uses
  classical MWF/DS/Wiener processing without a neural model.

## Main Results

| Strategy | Mean devices | Mean channels | SI-SDR change | PESQ | P10 PESQ | STOI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TDOA-routed single, raw | 7.2 | 7.2 | 0.000 dB | 1.056 | 1.044 | 0.562 |
| Distributed singles MWF | 7.2 | 7.2 | +8.517 dB | 1.370 | 1.140 | 0.740 |
| Equal-channel hybrid MWF | 4.2 | 7.2 | +10.117 dB | 1.499 | 1.062 | 0.677 |
| TDOA-routed fixed array MWF | 8.6 | 12.8 | +13.721 dB | 1.652 | 1.048 | 0.620 |
| Coverage hybrid, selected MWF | 8.6 | 12.8 | +11.880 dB | 1.785 | 1.159 | 0.813 |
| Coverage hybrid, all-channel MWF | 8.6 | 12.8 | **+13.046 dB** | **1.907** | **1.186** | **0.841** |
| Oracle target-room array MWF | 1.0 | 4.0 | +15.254 dB | 2.118 | 1.598 | 0.923 |

The oracle array is moved into the true target room and is only an upper
bound. Its device count is therefore not comparable to a fixed whole-home
deployment.

## Paired Findings

Against distributed-singles MWF on the same 100 cases:

- The coverage-preserving all-channel hybrid gains `+0.537 PESQ`
  (FloorPlan-clustered 95% CI `[+0.444, +0.630]`), `+0.101 STOI`
  (`[+0.082, +0.119]`), and `+4.529 dB SI-SDR`
  (`[+3.741, +5.315]`). It wins 91%, 90%, and 96% of cases respectively.
- Selecting only one array plus three singles remains useful, but is less
  stable: `+0.414 PESQ`, `+0.072 STOI`, and `+3.362 dB SI-SDR`, with wins in
  about 63-64% of cases.
- The equal-channel hybrid replaces four spatially separated microphones with
  one compact array. It improves mean PESQ by `0.129` and SI-SDR by `1.600 dB`
  but reduces mean STOI by `0.064`; it wins only 38-53% of individual cases.
  Equal channel count is therefore not equal whole-home coverage.
- A TDOA-routed fixed array performs well when it is in the target room, but
  in array-uncovered rooms its mean input SNR is `-20.571 dB`, PESQ is `1.267`,
  and STOI is `0.324`. A compact array is direction-independent around itself,
  not location-independent across walls.

For array-covered targets, the coverage hybrid reaches `2.422 PESQ` and
`0.938 STOI`. For array-uncovered targets it falls to `1.392 PESQ` and
`0.744 STOI`, close to distributed singles (`1.345`, `0.726`). Remote arrays
add only a small blind-room benefit; the distributed singles preserve the
coverage.

## Localization Finding

Audio GCC-PHAT TDOA identifies the correct room in 76% of the 100 cases, with
`1.075 m` median and `2.128 m` P90 position error. The 24 room errors are
concentrated in small bathrooms being mapped to an adjacent larger room. This
is weaker than earlier onset-TDOA studies and makes localization the next
system bottleneck. A practical follow-up should fuse TDOA with per-room level,
local array DOA, temporal tracking, and clock-error simulation.

## Conclusion

The best tested fixed whole-home design is not “one array instead of four
single microphones.” Keep the topology-optimized synchronized singles for
room coverage and TDOA, then add one array in 4/6/8-room homes or two arrays in
10/12-room homes for local high-quality pickup. Use all available channels for
MWF when compute permits; the measured enhancement RTF is about `0.012`.

These are simulation results with ideal synchronization, oracle target-active
and target-silent calibration, and preview-quality RIRs. They should be
confirmed with clock offset/drift, microphone gain/phase mismatch, real VAD,
longer utterances, and measured-room recordings before making a hardware
claim.

## Reproduce

```bash
python -m research.beamforming.run_whole_home_benchmark \
  --room-counts 4 6 8 10 12 \
  --plans-per-room-count 5 \
  --quality preview \
  --output research/results/whole-home-preview-25
```

The first uncached pass took about 323 seconds on the development machine.
Checkpointed reruns regenerate aggregate tables and confidence intervals in
about one second.
