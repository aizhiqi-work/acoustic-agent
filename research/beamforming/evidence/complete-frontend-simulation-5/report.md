# Complete Audio Front-End Benchmark

- FloorPlans: `5`
- Room counts: `4, 6, 8, 10, 12`
- RIR quality: `simulation`
- Elapsed: `34.83 s`

| Architecture | Pipeline | Mics | Input PESQ | PESQ | PESQ change | STOI | STOI change | SNR change | SI-SDR change | Dry PESQ | RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distributed_singles | distributed_mvdr | 7 | 1.504 | 2.336 | 0.832 | 0.810 | 0.034 | 13.333 | -5.206 | 1.054 | 0.0070 |
| distributed_singles | distributed_mwf | 7 | 1.504 | 2.788 | 1.283 | 0.943 | 0.167 | 12.074 | 9.630 | 1.070 | 0.0071 |
| distributed_singles | distributed_weighted_ds | 7 | 1.504 | 1.548 | 0.044 | 0.777 | 0.001 | 1.096 | -1.235 | 1.071 | 0.0039 |
| distributed_singles | distributed_wpd | 7 | 1.504 | 1.135 | -0.369 | 0.473 | -0.303 | 0.187 | -13.356 | 1.032 | 0.0261 |
| distributed_singles | distributed_wpe_mvdr | 7 | 1.504 | 2.100 | 0.595 | 0.799 | 0.023 | 11.537 | -5.310 | 1.051 | 0.1425 |
| distributed_singles | distributed_wpe_mwf | 7 | 1.504 | 2.572 | 1.068 | 0.926 | 0.151 | 11.269 | 5.149 | 1.073 | 0.1426 |
| local_array_4ch | local_ds | 4 | 1.069 | 1.079 | 0.010 | 0.634 | 0.092 | 0.936 | 0.616 | 1.026 | 0.0027 |
| local_array_4ch | local_mvdr | 4 | 1.069 | 2.033 | 0.964 | 0.907 | 0.365 | 13.008 | 9.194 | 1.045 | 0.0045 |
| local_array_4ch | local_mwf | 4 | 1.069 | 2.578 | 1.510 | 0.914 | 0.372 | 18.963 | 15.569 | 1.057 | 0.0046 |
| local_array_4ch | local_wpd | 4 | 1.069 | 1.241 | 0.173 | 0.806 | 0.264 | 6.582 | 4.961 | 1.034 | 0.0162 |
| local_array_4ch | local_wpe_mvdr | 4 | 1.069 | 1.691 | 0.623 | 0.846 | 0.304 | 11.733 | 7.066 | 1.049 | 0.0383 |
| local_array_4ch | local_wpe_mwf | 4 | 1.069 | 2.071 | 1.002 | 0.851 | 0.309 | 15.873 | 10.019 | 1.059 | 0.0384 |
| single | single_raw | 1 | 1.504 | 1.504 | 0.000 | 0.776 | 0.000 | 0.000 | 0.000 | 1.041 | 0.0000 |
| single | single_wiener | 1 | 1.504 | 1.436 | -0.069 | 0.758 | -0.018 | 2.674 | 1.309 | 1.041 | 0.0041 |

## Interpretation

- Primary perceptual metrics use each architecture's fixed clean target image at its reference microphone.
- Each architecture is level-calibrated over its own microphone set. Absolute scores compare complete systems; change metrics isolate processing benefit.
- Dry PESQ uses the original anechoic source and therefore measures the harder denoising-plus-dereverberation task.
- WPE is estimated from the observed mixture. Distributed WPE is applied independently at each node before TDOA beamforming.
- RTF covers enhancement only and excludes RIR simulation, localization, and file I/O.

## Recommended Pipelines by Scenario

| Scenario | Pipeline | SNR change | SI-SDR change | PESQ | STOI |
| --- | --- | ---: | ---: | ---: | ---: |
| cross_room | distributed_mwf | 9.837 | 7.509 | 3.110 | 0.961 |
| cross_room | local_mwf | 18.192 | 14.884 | 2.464 | 0.898 |
| same_room | distributed_mwf | 14.311 | 11.752 | 2.465 | 0.924 |
| same_room | local_mwf | 19.734 | 16.253 | 2.692 | 0.930 |
