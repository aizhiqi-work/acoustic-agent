# Complete Audio Front-End Benchmark

- FloorPlans: `25`
- Room counts: `4, 6, 8, 10, 12`
- RIR quality: `preview`
- Elapsed: `146.31 s`

| Architecture | Pipeline | Mics | Input PESQ | PESQ | PESQ change | STOI | STOI change | SNR change | SI-SDR change | Dry PESQ | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distributed_singles | distributed_mvdr | 7 | 1.501 | 2.238 | 0.737 | 0.835 | 0.059 | 12.260 | -3.507 | 1.052 | 0.0072 |
| distributed_singles | distributed_mwf | 7 | 1.501 | 2.497 | 0.996 | 0.927 | 0.151 | 10.790 | 8.403 | 1.058 | 0.0073 |
| distributed_singles | distributed_weighted_ds | 7 | 1.501 | 1.580 | 0.078 | 0.786 | 0.010 | 2.250 | -0.786 | 1.049 | 0.0042 |
| distributed_singles | distributed_wpd | 7 | 1.501 | 1.256 | -0.245 | 0.593 | -0.183 | 1.875 | -8.618 | 1.039 | 0.0260 |
| distributed_singles | distributed_wpe_mvdr | 7 | 1.501 | 2.076 | 0.574 | 0.825 | 0.049 | 11.130 | -3.840 | 1.082 | 0.1446 |
| distributed_singles | distributed_wpe_mwf | 7 | 1.501 | 2.328 | 0.826 | 0.910 | 0.134 | 10.152 | 4.531 | 1.059 | 0.1447 |
| local_array_4ch | local_ds | 4 | 1.067 | 1.081 | 0.013 | 0.656 | 0.098 | 1.129 | 0.787 | 1.027 | 0.0027 |
| local_array_4ch | local_mvdr | 4 | 1.067 | 1.835 | 0.767 | 0.902 | 0.345 | 12.320 | 8.876 | 1.048 | 0.0045 |
| local_array_4ch | local_mwf | 4 | 1.067 | 2.430 | 1.362 | 0.912 | 0.354 | 18.595 | 15.016 | 1.058 | 0.0046 |
| local_array_4ch | local_wpd | 4 | 1.067 | 1.240 | 0.172 | 0.827 | 0.270 | 6.989 | 5.313 | 1.036 | 0.0165 |
| local_array_4ch | local_wpe_mvdr | 4 | 1.067 | 1.628 | 0.561 | 0.848 | 0.291 | 11.927 | 7.066 | 1.054 | 0.0364 |
| local_array_4ch | local_wpe_mwf | 4 | 1.067 | 1.975 | 0.907 | 0.852 | 0.294 | 15.975 | 9.800 | 1.061 | 0.0365 |
| single | single_raw | 1 | 1.501 | 1.501 | 0.000 | 0.776 | 0.000 | 0.000 | 0.000 | 1.042 | 0.0000 |
| single | single_wiener | 1 | 1.501 | 1.463 | -0.038 | 0.756 | -0.020 | 2.581 | 1.201 | 1.037 | 0.0044 |

## Interpretation

- Primary perceptual metrics use each architecture's fixed clean target image at its reference microphone.
- Each architecture is level-calibrated over its own microphone set. Absolute scores compare complete systems; change metrics isolate processing benefit.
- Dry PESQ uses the original anechoic source and therefore measures the harder denoising-plus-dereverberation task.
- WPE is estimated from the observed mixture. Distributed WPE is applied independently at each node before TDOA beamforming.
- RTF covers enhancement only and excludes RIR simulation, localization, and file I/O.

## Recommended Pipelines by Scenario

| Scenario | Pipeline | SNR change | SI-SDR change | PESQ | STOI |
| --- | --- | ---: | ---: | ---: | ---: |
| cross_room | distributed_mwf | 7.953 | 5.843 | 2.879 | 0.956 |
| cross_room | local_mwf | 17.575 | 14.300 | 2.323 | 0.901 |
| same_room | distributed_mwf | 13.627 | 10.963 | 2.115 | 0.898 |
| same_room | local_mwf | 19.616 | 15.733 | 2.537 | 0.923 |
