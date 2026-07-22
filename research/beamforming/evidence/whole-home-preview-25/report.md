# Fixed Whole-Home Microphone Benchmark

- FloorPlans: `25`
- Cases: `100`
- Room counts: `4, 6, 8, 10, 12`
- RIR quality: `preview`
- Invocation elapsed: `1.14 s` (`100` cases resumed from checkpoint)
- Deployment is fixed before target sampling; every source keeps one globally calibrated emission gain.

## Overall

| Strategy | Devices | Channels | SNR in | SNR change | SI-SDR change | PESQ | P10 PESQ | STOI | Dry PESQ | Positive SI-SDR | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coverage_hybrid_all_mwf | 8.6 | 12.8 | -0.171 | 16.206 | 13.046 | 1.907 | 1.186 | 0.841 | 1.130 | 100.0% | 0.0117 |
| coverage_hybrid_selected_mwf | 8.6 | 12.8 | -0.443 | 14.984 | 11.880 | 1.785 | 1.159 | 0.813 | 1.122 | 100.0% | 0.0065 |
| distributed_singles_mwf | 7.2 | 7.2 | -0.236 | 10.775 | 8.517 | 1.370 | 1.140 | 0.740 | 1.079 | 100.0% | 0.0070 |
| equal_channel_hybrid_mwf | 4.2 | 7.2 | -3.798 | 13.149 | 10.117 | 1.499 | 1.062 | 0.677 | 1.126 | 100.0% | 0.0070 |
| oracle_target_room_array_mwf | 1.0 | 4.0 | -0.276 | 19.008 | 15.254 | 2.118 | 1.598 | 0.923 | 1.140 | 100.0% | 0.0043 |
| oracle_target_room_single_raw | 1.0 | 1.0 | -0.421 | 0.000 | 0.000 | 1.054 | 1.047 | 0.592 | 1.027 | 0.0% | 0.0000 |
| tdoa_routed_fixed_array_ds | 8.6 | 12.8 | -10.419 | 0.474 | 0.008 | 1.100 | 1.022 | 0.449 | 1.089 | 54.0% | 0.0025 |
| tdoa_routed_fixed_array_mwf | 8.6 | 12.8 | -10.419 | 18.380 | 13.721 | 1.652 | 1.048 | 0.620 | 1.187 | 100.0% | 0.0043 |
| tdoa_routed_single_raw | 7.2 | 7.2 | -1.650 | 0.000 | 0.000 | 1.056 | 1.044 | 0.562 | 1.033 | 0.0% | 0.0000 |
| tdoa_routed_single_wiener | 7.2 | 7.2 | -1.650 | 2.026 | 1.463 | 1.030 | 1.023 | 0.526 | 1.033 | 95.0% | 0.0040 |

## TDOA Localization

| Target coverage | Cases | Room accuracy | Median error | P90 error |
| --- | ---: | ---: | ---: | ---: |
| all | 100 | 76.0% | 1.075 m | 2.128 m |
| array_covered | 50 | 68.0% | 1.075 m | 1.982 m |
| array_uncovered | 50 | 84.0% | 1.026 m | 2.283 m |

## Paired Against Distributed Singles MWF

Confidence intervals resample complete FloorPlans, preserving the four correlated cases within each layout.

| Strategy | PESQ delta [95% CI] | PESQ wins | STOI delta [95% CI] | STOI wins | SI-SDR delta [95% CI] | SI-SDR wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_target_room_array_mwf | 0.748 [0.637, 0.857] | 92.0% | 0.182 [0.155, 0.209] | 94.0% | 6.736 dB [5.660, 7.734] | 91.0% |
| tdoa_routed_fixed_array_mwf | 0.282 [0.175, 0.386] | 48.0% | -0.121 [-0.154, -0.084] | 46.0% | 5.203 dB [4.133, 6.218] | 84.0% |
| equal_channel_hybrid_mwf | 0.129 [0.023, 0.244] | 38.0% | -0.064 [-0.106, -0.021] | 39.0% | 1.600 dB [0.326, 2.911] | 53.0% |
| coverage_hybrid_selected_mwf | 0.414 [0.319, 0.514] | 64.0% | 0.072 [0.047, 0.096] | 63.0% | 3.362 dB [2.490, 4.196] | 63.0% |
| coverage_hybrid_all_mwf | 0.537 [0.444, 0.630] | 91.0% | 0.101 [0.082, 0.119] | 90.0% | 4.529 dB [3.741, 5.315] | 96.0% |

## Interpretation Rules

- `oracle_target_room_*` moves one reference device into the true target room and is an upper bound, not a fixed whole-home deployment.
- `tdoa_routed_*` selects among devices that were fixed before the target was sampled.
- `equal_channel_hybrid_mwf` spends the same channel budget as distributed singles: one four-channel array plus `N-4` singles.
- `coverage_hybrid_*` keeps all `N` singles and adds one array for 4/6/8-room homes or two arrays for 10/12-room homes.
- Wet-reference PESQ measures enhancement without penalizing the room response. Dry PESQ additionally exposes coloration and reverberation.
- Target-active and target-silent calibration segments are oracle VAD baselines; no clean evaluation waveform is used to estimate weights.
