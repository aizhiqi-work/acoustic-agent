# FP-RIR Tiers And CUDA Validation

FP-RIR separates two uses of simulated residential acoustics:

- **Adapt 1K/3K/6K** provides floorplan-disjoint far-field RIRs for downstream
  model adaptation. One maximum corpus can expose nested tier JSONL indices
  without duplicating HDF5 tensors.
- **Dist Quick/Standard/Extended** evaluates synchronized whole-home TDOA
  localization and beamforming. It is an evaluation suite, not training data.

The batch entry points and complete item schema are documented in
[`docs/FP_RIR.md`](../../docs/FP_RIR.md).

## RTX 4090 Quick Validation

Validation used CUDA FP32 tracing on one RTX 4090, with physical GPU selection
mapped to solver device 0. The CUDA energy-field parity suite passed all eight
tests against the Numba FP32 reference.

### Adapt 1K

The complete `simulation`, 16 kHz, 2.0 s tier generated:

| Property | Result |
| --- | ---: |
| Floorplans | 1,000 |
| Train / validation / test | 782 / 106 / 112 |
| Configurations | 2,094 |
| Static RIR channels | 4,996 |
| Moving-source trajectories / keyframes | 95 / 475 |
| Active HDF5 shards | 66 |
| Storage | 609 MiB |
| Sum of per-item generation time | 1,027.99 s |
| Failed configurations | 0 |

An independent integrity pass matched every plan item to one unique HDF5
group and the nested `adapt-1k.jsonl` index. All 2,094 tensors had the declared
shape, finite samples, and nonzero energy.

Mean generation time was 0.206 s for same-room mono, 0.736 s for cross-room
four-channel circular arrays, and 0.909 s for five-keyframe moving-source
sequences. These are one-machine throughput observations, not portable
performance guarantees.

### Dist Quick Localization

Quick uses one calibration and two unseen validation layouts in each 4/6/8/10/12
room stratum, with two target points per room. The run contained 10 validation
floorplans and 1,120 microphone-budget cases.

| Rooms | Selected mics | Median | P90 | Room accuracy | Adequate |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 4 | 4 | 0.58 m | 2.17 m | 100.0% | no |
| 6 | 5 | 0.59 m | 1.58 m | 100.0% | yes |
| 8 | 6 | 0.64 m | 1.91 m | 93.8% | yes |
| 10 | 7 | 0.68 m | 1.83 m | 97.5% | yes |
| 12 | 8 | 0.52 m | 1.81 m | 93.8% | yes |

The four-room miss is only the P90 threshold (`2.0 m`) on two layouts. The
broader 110-layout Standard study remains the statistical microphone-count
reference; Quick checks the pipeline and expected error scale.

### Dist Quick Beamforming

The fixed deployment uses 5/7/8/8/8 synchronized singles for 4/6/8/10/12-room
homes. Quick evaluated five layouts and 20 same-/cross-room interference cases
with the canonical benchmark seed.

| Strategy | SNR change | SI-SDR change | PESQ | STOI |
| --- | ---: | ---: | ---: | ---: |
| Distributed singles MWF | +12.322 dB | +9.714 dB | 1.380 | 0.770 |
| Coverage hybrid, all-channel MWF | +16.629 dB | +13.486 dB | 1.818 | 0.855 |
| Oracle target-room array MWF | +19.530 dB | +15.567 dB | 2.122 | 0.922 |

The corresponding 25-layout results are `+10.775/+8.517 dB`, PESQ `1.370`,
STOI `0.740` for distributed singles; `+16.206/+13.046 dB`, PESQ `1.907`,
STOI `0.841` for the all-channel coverage hybrid; and PESQ `2.118`, STOI
`0.923` for the oracle target-room array. The ranking and effect sizes are
therefore reproduced by Quick.

Quick TDOA routing reached 90.0% room accuracy, 0.489 m median, and 1.398 m P90.
The 25-layout result is 76.0%, 1.075 m, and 2.128 m, so the five-layout Quick
subset is optimistic and must not replace the larger estimate.

## Reproduce

```bash
GPU_ID=0 scripts/fprir/run_adapt_tier.sh 1k
GPU_ID=0 scripts/fprir/run_dist_tier.sh quick
```

Completed generator shards and beamforming checkpoints resume automatically.
Incomplete Adapt shards are rebuilt atomically. Any remaining item failure
causes a nonzero process exit after preserving valid shards and diagnostics.
