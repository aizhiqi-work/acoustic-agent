# LOS Direction of Arrival

This study evaluates source azimuth recovery from Acoustic Agent output in two
scene families:

- **Geometry**: a controlled rectangular room with known source bearings.
- **FloorPlan**: same-room placements sampled from a converted FloorPlan scene.

Every source and receiver pair has a visible direct path. Two acoustic
conditions separate estimator behavior from room effects:

- `direct`: direct propagation only.
- `room`: direct propagation plus the engine's reflections and late tail.

## Receivers and baselines

| Receiver | Baseline | Observable direction |
|---|---|---|
| HRTF | Interaural phase + level template matching against the bundled SOFA HRTF | Full horizontal circle, subject to generic-HRTF mismatch |
| 4-channel linear array | SRP-PHAT, 4 cm spacing | Half-plane bearing; mirror ambiguity is explicit |
| 8-channel circular array | SRP-PHAT, 5 cm radius | Full horizontal circle |

World azimuth uses `+x = 0 deg`, `+y = 90 deg`, and counter-clockwise-positive
angles. HRTF results also report listener-relative azimuth after subtracting
`orientation_deg`.

## Run

From the repository root:

```bash
python -m research.doa.run_los --quick
python -m research.doa.run_los
```

Run the room condition on one visible NVIDIA GPU in FP32:

```bash
CUDA_VISIBLE_DEVICES=0 python -m research.doa.run_los \
  --accelerator cuda --precision float32 --cuda-device 0 \
  --output-dir research/results/doa-los-cuda-4090
```

The quick command runs one Geometry and one FloorPlan placement. The full
command runs three placements per scene under both acoustic conditions.

Restrict a run or increase solver quality:

```bash
python -m research.doa.run_los \
  --scenes floorplan \
  --conditions room \
  --floorplan-idx 0 \
  --quality simulation
```

Outputs are written to `research/results/doa-los/`:

- `report.md`: aggregate mean and maximum angular errors.
- `summary.csv` / `summary.json`: per-case estimates and metadata.
- `*.npz`: RIR, received probe, scan grid, spatial score, and positions.

The committed compact snapshot is available under
[`evidence/doa-los/`](evidence/doa-los/report.md).

The deterministic broadband probe is used only to create a realistic
multichannel observation. Both estimators use inter-channel phase/level
relationships, so the same pipeline can later be applied to speech or noise.

## Interpretation

The `direct` condition is the coordinate-system and estimator sanity check. A
large error there usually indicates an azimuth convention, microphone geometry,
or channel-order issue. The difference between `direct` and `room` measures the
effect of early reflections and late reverberation while retaining LOS.

The linear array cannot distinguish bearings mirrored across its axis. Its
reported error therefore uses the physically observable equivalent bearing;
the unmodified world truth remains in the result table. HRTF localization is a
matched-dataset upper bound because simulation and estimator use the same SOFA
file. Later studies should add mismatched SOFA subjects and real recordings.

## Distributed whole-home localization

`run_distributed` studies position-independent microphone deployment and
whole-home localization on unseen FloorPlans. The traditional baseline uses:

- topology-aware minimax greedy placement based only on room polygons and
  portal connectivity;
- SRP-PHAT at internally synchronized circular-array nodes;
- onset TDOA across globally synchronized single microphones;
- a FloorPlan grid likelihood that predicts portal-routed arrival direction
  and travel time;
- hybrid DOA/TDOA likelihood fusion.

Run the small or full split:

```bash
python -m research.doa.run_distributed --quick
python -m research.doa.run_distributed
```

The CUDA run must use a separate output directory. Accelerator, precision, and
device are part of the measurement-cache key, so CPU and GPU observations
cannot be mixed silently:

```bash
CUDA_VISIBLE_DEVICES=0 python -m research.doa.run_distributed \
  --accelerator cuda --precision float32 --cuda-device 0 \
  --output-dir research/results/distributed-floorplan-cuda-4090
```

The fixed training split selects the placement risk parameter. Target
positions from the disjoint test FloorPlans never participate in placement or
selection. The report compares one large array, distributed four-channel
arrays, synchronized single microphones, and a hybrid deployment under
explicit channel budgets.

For the larger room-count and floor-area-stratified validation, run:

```bash
python -m research.doa.run_stratified
```

Use the same accelerator flags with `run_stratified`; `--quick` validates two
room-count strata before starting the full 110-FloorPlan study.

This scans all 15,376 compiled FloorPlans, filters disconnected or malformed
room graphs, and evaluates 4- through 14-room homes. Every room-count stratum
uses five disjoint calibration FloorPlans and ten validation FloorPlans sampled
across its complete area distribution. Every room contributes a source point,
and synchronized single-microphone deployments from three nodes through one
node per room are evaluated. The generated report includes room-count and area
breakdowns plus FloorPlan-clustered bootstrap confidence intervals.

The current 110-FloorPlan validation finds a minimum of four microphones for
4-6 rooms, five for 7-8 rooms, six for 9-10 rooms, and seven for 11-14 rooms.
Within each room-count stratum, the median error rises from `0.55 m` for the
small-area group to `0.74 m` for the large-area group. Generated evidence is
written to `research/results/distributed-floorplan-stratified/`.
Compact reports, fixed splits, and aggregate CSV files are committed under
[`evidence/`](evidence/README.md) for review without the large acoustic cache.
