# Preview Pipeline Check: FloorPlan 0

This is a deterministic pipeline check, not the final statistical result. It
uses the `preview` RIR preset, FloorPlan index 0, eight synchronized single
microphones, a four-channel circular local array, a 0 dB target/interferer
level, and 18 dB sensor-noise level.

Command:

```bash
python -m research.beamforming.run \
  --study all \
  --quality preview \
  --floorplan-idx 0 \
  --distributed-nodes 8 \
  --subset-counts 2 4 6 8 \
  --duration 2.4 \
  --rir-duration 1.0 \
  --output research/results/beamforming-preview
```

## Selected Results

| System | Condition | SNR improvement | SI-SDR improvement | RTF |
| --- | --- | ---: | ---: | ---: |
| Local array DS, oracle arrival | Same room | +1.11 dB | +0.35 dB | 0.0093 |
| Local array DS, SRP-PHAT DOA | Same room | +1.28 dB | +1.16 dB | 0.0094 |
| Distributed DS, oracle, 8 mics | Same-room interferer | +2.89 dB | +1.58 dB | 0.0222 |
| Distributed DS, estimated, 2 mics | Cross-room interferer | +3.11 dB | -0.78 dB | 0.0049 |
| Distributed DS, oracle, 4 mics, no source-room mic | Cross-room interferer | -0.57 dB | -1.10 dB | 0.0098 |

SRP-PHAT estimated the local target at 36 degrees for a 35-degree truth. The
distributed GCC-PHAT plus FloorPlan grid estimator selected the correct room
with 0.38 m position error.

## Interpretation

- The four-channel local array is already a valid low-risk baseline.
- More distributed microphones are not automatically better. For the
  cross-room interferer, the selected two-mic system beat the eight-mic system.
- Removing every source-room microphone made the cross-room result worse than
  the best synchronized single-mic baseline in this plan.
- Positive SNR improvement with negative SI-SDR improvement means interference
  energy fell while the output moved farther from the fixed dry target. This is
  a warning to report both suppression and speech distortion.
- The estimated steering result may exceed first-arrival steering because DS
  is not an oracle mask: a different delay can trade target coherence for
  stronger interferer cancellation. It does not mean the estimated location
  is more accurate than ground truth.

These observations are hypotheses for the full 4/8/12-room held-out sweep,
not general conclusions from one layout.
