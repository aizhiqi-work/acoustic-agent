# FP-RIR Dataset

FP-RIR is Acoustic Agent's floorplan-based multichannel room impulse response
dataset. The generator uses the bundled, validated residential floorplan
resource and the same public solver API as the Web workbench.

## Protocol

Each floorplan is assigned to exactly one split using a stable hash:

- 80% train
- 10% validation
- 10% test

All source, receiver, material, array, and motion variants of that floorplan
remain in the same split. This prevents geometry and room-connectivity leakage.

The core dataset creates three static configurations per floorplan:

| Configuration | Channels | Placement |
| --- | ---: | --- |
| `same_room_mono` | 1 | Source and microphone in one habitable room |
| `cross_room_circular4` | 4 | Compact 4-microphone circular array across one or more open portals |
| `distributed_mono` | 2-4 | One synchronized mono microphone per selected room |

Cross-room pairs are sampled across the available room-graph distances instead
of always selecting an adjacent room. Distributed microphones include the
source room and connected rooms at increasing graph distance. The optional
`moving_source_mono` companion subset samples a room-constrained source path
with 0.25 m keyframe spacing by default.

Every floorplan uses one deterministic material seed. Wall, floor, ceiling,
door, and window materials are sampled from the semantic acoustic-material
resource and stored with their six-band absorption, scattering, and
transmission parameters.

The core v1 protocol intentionally leaves automatic furniture disabled.
Otherwise, receiver-specific collision avoidance could produce a different
furniture layout for each channel of a distributed configuration. A furnished
extension should generate one layout per floorplan and reuse it unchanged for
every source and receiver.

## Generate

FP-RIR is split into two products with different purposes:

- **Adapt** is a far-field RIR corpus for downstream model adaptation. Its
  published 1K, 3K, and 6K floorplan tiers contain same-room mono, cross-room
  circular-array, and a 10% moving-source subset.
- **Dist** is an evaluation suite for whole-home TDOA localization and
  distributed beamforming. It stores benchmark results rather than duplicating
  a large training corpus.

The batch entry points are:

```bash
# Single GPU. Physical GPU 2 is exposed as solver device 0 inside the script.
GPU_ID=2 scripts/fprir/run_adapt_tier.sh smoke
GPU_ID=2 scripts/fprir/run_adapt_tier.sh 1k
GPU_ID=2 scripts/fprir/run_adapt_tier.sh 3k
GPU_ID=2 scripts/fprir/run_adapt_tier.sh 6k

GPU_ID=2 scripts/fprir/run_dist_tier.sh quick
GPU_ID=2 scripts/fprir/run_dist_tier.sh standard
GPU_ID=2 scripts/fprir/run_dist_tier.sh extended
```

For a server with several GPUs, give the physical device IDs as a
comma-separated list:

```bash
# Adapt deterministically partitions the FloorPlans across GPU processes.
GPU_IDS=2,3,4,5 \
FPRIR_PROCESSES_PER_GPU=4 \
scripts/fprir/run_adapt_multi_gpu.sh 6k

# Quick, Standard, and Extended Dist run concurrently on up to three GPUs.
GPU_IDS=2,3,4,5 scripts/fprir/run_dist_multi_gpu.sh

# Generate both products and every published tier.
GPU_IDS=2,3,4,5 scripts/fprir/run_all_tiers.sh all

# Generate only one product.
GPU_IDS=2,3,4,5 scripts/fprir/run_all_tiers.sh adapt
GPU_IDS=2,3,4,5 scripts/fprir/run_all_tiers.sh dist
```

`run_all_tiers.sh` automatically selects the multi-GPU launchers when
`GPU_IDS` contains more than one device. With one GPU, use `GPU_ID=2`; the
single-device scripts run sequentially.

Do not increase `--workers` inside one CUDA generator. Numba CUDA contexts are
process-local, and concurrent threads on the same device are both slower and
can fail with a device-unavailable error. `FPRIR_PROCESSES_PER_GPU` uses
independent OS processes instead, so GPU tracing can overlap with CPU-side
six-band reconstruction. Start at `2`; on the tested 64-thread, four-RTX-4090
server, `4` processes per GPU was stable and fastest in the pilot. The launcher
automatically divides available Numba CPU threads among all processes.

CUDA remains the recommended backend. A CPU-only fallback is available when
GPUs are busy:

```bash
FPRIR_RT_ACCELERATOR=numba \
FPRIR_RT_PRECISION=float64 \
NUMBA_NUM_THREADS=64 \
scripts/fprir/run_adapt_tier.sh 6k

FPRIR_RT_ACCELERATOR=numba \
FPRIR_RT_PRECISION=float64 \
NUMBA_NUM_THREADS=64 \
scripts/fprir/run_dist_tier.sh quick
```

CPU mode uses the same solver and BVH, but the 64-thread CPU is best reserved
for fallback or validation because it consumes the whole host to approach the
throughput of one 4090.

The Python batch loops use `tqdm`. Every stage displays completed work,
elapsed time, processing rate, and estimated remaining time. Adapt reports
resource scanning, configuration planning, RIR generation, and summary
generation separately. Multi-GPU Adapt displays one aggregate configuration
bar. Dist localization counts completed RIR solves, Dist beamforming counts
target/interference cases, and its multi-GPU launcher displays an aggregate
six-stage bar. Detailed per-process progress is retained in timestamped logs.

Set `FPRIR_OUTPUT_ROOT=/data/fprir` to move all data and logs outside the
repository. Set `PLAN_ONLY=1` for an Adapt metadata-only dry run, or
`DIST_STAGE=localization` / `DIST_STAGE=beamforming` to run one Dist stage.
All CUDA batch scripts use FP32 tracing and retain completed generator shards
or benchmark checkpoints when the same command is resumed. Completed Dist
stages are skipped unless `FORCE=1` is set.

Adapt tiers are nested. A 6K generation writes one set of HDF5 shards plus
`tiers/adapt-1k.jsonl`, `adapt-3k.jsonl`, and `adapt-6k.jsonl`; the smaller
tiers reference the same shards and do not duplicate RIR tensors.

For a short multi-GPU systems check before the 6K run:

```bash
GPU_IDS=2,3 \
FPRIR_PROCESSES_PER_GPU=2 \
FPRIR_MAX_FLOORPLANS=20 \
FPRIR_QUALITY=preview \
FPRIR_DURATION_S=0.25 \
FPRIR_MOTION_FRACTION=0 \
scripts/fprir/run_adapt_multi_gpu.sh 6k
```

Use a fresh `FPRIR_OUTPUT_ROOT` for this check so its manifest cannot be
confused with the final Simulation corpus.

Start with the stratified pilot. It samples floorplans around 4, 6, 8, 10, and
12 rooms and shows progress, generation rate, and ETA:

```bash
python scripts/generate_fprir.py \
  --profile pilot \
  --output benchmark-results/fprir-pilot
```

Inspect the complete full-corpus plan without running the solver:

```bash
python scripts/generate_fprir.py \
  --profile full \
  --plan-only \
  --output benchmark-results/fprir-full-plan
```

Generate the complete dataset:

```bash
python scripts/generate_fprir.py \
  --profile full \
  --output /data/FP-RIR \
  --quality simulation \
  --fs 16000 \
  --duration-s 2.0 \
  --max-distributed-mics 4 \
  --motion-fraction 0.1 \
  --shard-size 32 \
  --workers 1
```

The default single worker avoids oversubscribing Numba's internal parallel
work. A stopped run can be resumed with the same command: a shard is reused
only after both its HDF5 file and JSONL sidecar have been finalized.

With the current 15,376-scene resource and the defaults above, the deterministic
full plan contains:

| Planned property | Count |
| --- | ---: |
| Train / validation / test floorplans | 12,275 / 1,539 / 1,562 |
| Static configurations | 46,120 |
| Static RIR channels | 138,261 |
| Moving-source trajectories | 1,488 |
| Nominal moving-source keyframes | 7,440 |

Four source scenes have no verified open cross-room route, so their
cross-room compact-array and distributed configurations are omitted instead of
inventing connectivity.

## Storage

The output directory contains:

```text
manifest.json
plan.jsonl
resource-statistics.json
fprir-summary.json
fprir-overview.tex
fprir-statistics.svg
fprir-statistics.png       # when ImageMagick is available
fprir-statistics.pdf       # when ImageMagick is available
errors.jsonl
tiers/
  adapt-1k.jsonl
  adapt-3k.jsonl
  adapt-6k.jsonl
  summary.json
shards/
  fprir-00000.h5
  fprir-00000.jsonl
  ...
```

A multi-GPU Adapt output additionally retains resumable process partitions
under `parts/part-000`, `parts/part-001`, and so on. The top-level `shards/`
directory uses hard links to those HDF5 files when the filesystem supports
them, so the merged dataset has one logical index without storing the RIR
tensors twice. The merger falls back to file copies only when hard links are
unavailable.

Dist outputs are benchmark artifacts rather than an RIR training corpus:

```text
dist-quick/
  localization/
  beamforming/
dist-standard/
  localization/
  beamforming/
dist-extended/
  localization/
  beamforming/
logs/
```

Static RIR tensors use shape `[channel, sample]`. Moving-source tensors use
`[keyframe, channel, sample]`. RIRs are stored as gzip-compressed `float32`;
each HDF5 group has the same item ID as its JSONL index record.

Metadata includes:

- floorplan index and disjoint split;
- configuration and receiver type;
- source and microphone coordinates;
- source and receiver room IDs and semantic room types;
- room-graph and Euclidean source-microphone distances;
- material IDs and six-band acoustic coefficients;
- sample rate, duration, quality, and intersection backend;
- broadband and six-band RIR-derived RT60;
- motion trajectory and per-keyframe metrics where applicable.

## Read

```python
import json
from pathlib import Path

import h5py

root = Path("/data/FP-RIR")
record = json.loads(
    (root / "shards" / "fprir-00000.jsonl").read_text().splitlines()[0]
)

with h5py.File(root / "shards" / record["shard"], "r") as shard:
    rir = shard[record["group"]]["rir"][:]

print(rir.shape)
print(record["split"], record["kind"])
print(record["graph_distances"], record["rt60_s"])
```

## Statistics

`fprir-statistics.svg` contains the four distributions used by the FP-RIR
paper section:

1. rooms per floorplan, computed over all 15,376 resource scenes;
2. source-microphone room-graph distance;
3. Euclidean source-microphone distance;
4. broadband RIR-derived RT60.

`fprir-summary.json` is the machine-readable source of all table values.
`fprir-overview.tex` is generated from the same summary so that reported counts
cannot drift from the produced shards.

## Dist Protocol

Dist uses room-count strata 4, 6, 8, 10, and 12. Quick samples one calibration
and two validation layouts per stratum for localization, then one layout per
stratum for whole-home beamforming. Standard uses 5/10 calibration/validation
layouts and five beamforming layouts per stratum. Extended doubles those
layout counts.

The fixed whole-home sensor budget is 5, 7, 8, 8, and 8 microphones for the
4-, 6-, 8-, 10-, and 12-room strata. The benchmark compares synchronized
distributed single microphones and coverage-preserving local arrays under
same-room and cross-room interference. Its reports include room accuracy,
localization error, SNR and SI-SDR improvement, STOI, PESQ when available, and
runtime.
