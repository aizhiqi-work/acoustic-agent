# Single-GPU CUDA Acceleration

Acoustic Agent provides two reflection-tracing backends:

- `numba`: CPU tracing in `float64` or `float32`.
- `cuda`: single-NVIDIA-GPU tracing in `float32`.

The default remains Numba FP64 for portability and reproducibility. CUDA is an
opt-in acceleration path for sufficiently large simulations.

```python
from acoustic_agent import SimConfig

cpu_reference = SimConfig(rt_accelerator="numba", rt_precision="float64")
cpu_fp32 = SimConfig(rt_accelerator="numba", rt_precision="float32")
gpu_fp32 = SimConfig(
    rt_accelerator="cuda",
    rt_precision="float32",
    rt_cuda_device=0,
)
```

`rt_accelerator="auto"` selects CUDA when the requested device is available and
the requested precision is FP32. It falls back to Numba for FP64 or when CUDA is
unavailable. CUDA FP64 is rejected explicitly because RTX 4090 and RTX A6000
have much lower FP64 throughput than FP32 throughput.

The CUDA implementation keeps scene geometry, ray directions, and deterministic
random workspaces in a bounded device cache. Source and receiver positions plus
result buffers are transferred per run. The reflection metadata records the
selected accelerator, precision, device, kernel time, transfer time, and cache
usage.

## Reproduce The Benchmark

Run each process with only the intended GPU visible. The script performs a
warmup before recording three steady-state repetitions.

```bash
NUMBA_NUM_THREADS=64 python scripts/benchmark_accelerators.py \
  --accelerator numba --precision float64 --repeats 3

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_accelerators.py \
  --accelerator cuda --precision float32 --device 0 --repeats 3

NUMBA_NUM_THREADS=64 python scripts/benchmark_accelerators.py \
  --accelerator numba --precision float64 --scene-set floorplan --repeats 3
```

The benchmark disables the late tail, diffraction, visual paths, and ambisonic
rendering so that it measures the reflection accelerator and the shared RIR
post-processing path consistently. `--scene-set floorplan` adds deterministic
cross-room cases from FloorPlan 12513 (5 rooms) and 11282 (10 rooms), both
empty and with `balanced` furnishing.

## July 2026 Results

Environment: Python 3.12.13, NumPy 2.4.4, Numba 0.65, CUDA 12.4, one RTX 4090
or one RTX A6000, and a 64-thread Numba CPU baseline. Times are medians of three
warm runs. `Trace` measures reflection tracing; `E2E` measures the complete RIR
call. Speedups are relative to Numba FP64.

| Room | Quality | Rays | CPU FP64 trace / E2E (ms) | A6000 trace / E2E (ms) | A6000 speedup | 4090 trace / E2E (ms) | 4090 speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rectangle, 6 surfaces | Preview | 8,192 | 15.64 / 58.73 | 3.89 / 49.13 | 4.02x / 1.20x | 5.22 / 62.24 | 2.99x / 0.94x |
| Rectangle, 6 surfaces | Simulation | 32,768 | 50.49 / 90.48 | 5.24 / 51.82 | 9.64x / 1.75x | 4.27 / 51.06 | 11.83x / 1.77x |
| Rectangle, 6 surfaces | Reference | 131,072 | 132.94 / 201.29 | 11.66 / 58.45 | 11.40x / 3.44x | 12.36 / 58.09 | 10.76x / 3.47x |
| Furnished U room, 22 surfaces | Preview | 8,192 | 13.16 / 59.03 | 5.42 / 49.22 | 2.43x / 1.20x | 5.00 / 48.90 | 2.63x / 1.21x |
| Furnished U room, 22 surfaces | Simulation | 32,768 | 50.96 / 93.45 | 8.45 / 52.41 | 6.03x / 1.78x | 6.40 / 50.44 | 7.96x / 1.85x |
| Furnished U room, 22 surfaces | Reference | 131,072 | 182.10 / 254.58 | 23.31 / 69.25 | 7.81x / 3.68x | 13.72 / 59.61 | 13.28x / 4.27x |
| Furnished round room, 98 surfaces | Preview | 8,192 | 21.28 / 68.04 | 11.92 / 57.03 | 1.79x / 1.19x | 9.58 / 54.42 | 2.22x / 1.25x |
| Furnished round room, 98 surfaces | Simulation | 32,768 | 89.20 / 128.22 | 24.00 / 70.13 | 3.72x / 1.83x | 15.23 / 61.09 | 5.86x / 2.10x |
| Furnished round room, 98 surfaces | Reference | 131,072 | 394.27 / 426.62 | 79.35 / 126.11 | 4.97x / 3.38x | 45.34 / 93.06 | 8.70x / 4.58x |

Numba FP32 was close to FP64 on this CPU rather than consistently faster. Its
maximum total-energy difference from FP64 was 0.00195%. Across both GPUs, the
maximum total-energy difference was 0.00554%, the maximum traced RT60 mean
difference was 0.00439%, and the final RIR RT60 values were unchanged.

## FloorPlan Results

The four FloorPlan cases use cross-room source and receiver placement, BVH
traversal, and the same warm-run protocol. Times below are complete RIR calls;
parenthesized values are speedups over Numba FP64.

| FloorPlan | Quality | Rays | Numba FP64 (ms) | Numba FP32 (ms) | A6000 FP32 (ms) | 4090 FP32 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5 rooms, empty, 68 surfaces | Preview | 8,192 | 70.56 | 73.36 | 62.05 (1.14x) | 61.22 (1.15x) |
| 5 rooms, empty, 68 surfaces | Simulation | 32,768 | 98.41 | 98.48 | 63.74 (1.54x) | 61.47 (1.60x) |
| 5 rooms, empty, 68 surfaces | Reference | 131,072 | 145.85 | 155.29 | 84.33 (1.73x) | 74.06 (1.97x) |
| 5 rooms, 12 furnishings, 83 surfaces | Preview | 8,192 | 73.98 | 75.90 | 66.22 (1.12x) | 65.50 (1.13x) |
| 5 rooms, 12 furnishings, 83 surfaces | Simulation | 32,768 | 112.23 | 112.67 | 68.98 (1.63x) | 67.96 (1.65x) |
| 5 rooms, 12 furnishings, 83 surfaces | Reference | 131,072 | 169.78 | 166.89 | 90.33 (1.88x) | 77.57 (2.19x) |
| 10 rooms, empty, 145 surfaces | Preview | 8,192 | 73.37 | 75.30 | 68.72 (1.07x) | 67.98 (1.08x) |
| 10 rooms, empty, 145 surfaces | Simulation | 32,768 | 95.12 | 109.34 | 69.56 (1.37x) | 67.83 (1.40x) |
| 10 rooms, empty, 145 surfaces | Reference | 131,072 | 173.89 | 175.80 | 92.82 (1.87x) | 81.80 (2.13x) |
| 10 rooms, 27 furnishings, 178 surfaces | Preview | 8,192 | 93.04 | 98.60 | 81.70 (1.14x) | 81.76 (1.14x) |
| 10 rooms, 27 furnishings, 178 surfaces | Simulation | 32,768 | 134.26 | 134.37 | 88.51 (1.52x) | 85.79 (1.56x) |
| 10 rooms, 27 furnishings, 178 surfaces | Reference | 131,072 | 199.96 | 205.92 | 104.24 (1.92x) | 91.96 (2.17x) |

Across these FloorPlan cases, the maximum GPU total-energy difference from
Numba FP64 was 0.1442%, the maximum traced RT60 mean difference was 0.5016%,
and the maximum final RIR RT60 difference was 0.014 s. A6000 ray-tracing
speedup ranged up to 4.92x at Simulation and 3.64x at Reference; RTX 4090
reached 6.21x at Simulation and 6.17x at Reference.

Small preview workloads can be dominated by launch, transfer, and CPU
post-processing overhead. Use Numba FP64 when exact CPU continuity matters;
use CUDA FP32 for simulation, reference, batch, and complex-room workloads.
