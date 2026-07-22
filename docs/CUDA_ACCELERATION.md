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
```

The benchmark disables the late tail, diffraction, visual paths, and ambisonic
rendering so that it measures the reflection accelerator and the shared RIR
post-processing path consistently.

## July 2026 Results

Environment: Python 3.12.13, NumPy 2.4.4, Numba 0.65, CUDA 12.4, one RTX 4090
or one RTX A6000, and a 64-thread Numba CPU baseline. Times are medians of three
warm runs. `Trace` measures reflection tracing; `E2E` measures the complete RIR
call. Speedups are relative to Numba FP64.

| Room | Quality | Rays | CPU FP64 trace / E2E (ms) | A6000 trace / E2E (ms) | A6000 speedup | 4090 trace / E2E (ms) | 4090 speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rectangle, 6 surfaces | Preview | 8,192 | 15.64 / 58.73 | 3.89 / 49.13 | 4.02x / 1.20x | 5.22 / 62.24 | 2.99x / 0.94x |
| Rectangle, 6 surfaces | Simulation | 32,768 | 50.49 / 90.48 | 5.24 / 51.82 | 9.64x / 1.75x | 4.27 / 51.06 | 11.83x / 1.77x |
| Rectangle, 6 surfaces | Reference | 98,304 | 144.98 / 187.16 | 11.09 / 59.45 | 13.07x / 3.15x | 8.42 / 57.06 | 17.22x / 3.28x |
| Furnished U room, 22 surfaces | Preview | 8,192 | 13.16 / 59.03 | 5.42 / 49.22 | 2.43x / 1.20x | 5.00 / 48.90 | 2.63x / 1.21x |
| Furnished U room, 22 surfaces | Simulation | 32,768 | 50.96 / 93.45 | 8.45 / 52.41 | 6.03x / 1.78x | 6.40 / 50.44 | 7.96x / 1.85x |
| Furnished U room, 22 surfaces | Reference | 98,304 | 199.27 / 251.90 | 23.42 / 69.24 | 8.51x / 3.64x | 13.40 / 59.11 | 14.87x / 4.26x |
| Furnished round room, 98 surfaces | Preview | 8,192 | 21.28 / 68.04 | 11.92 / 57.03 | 1.79x / 1.19x | 9.58 / 54.42 | 2.22x / 1.25x |
| Furnished round room, 98 surfaces | Simulation | 32,768 | 89.20 / 128.22 | 24.00 / 70.13 | 3.72x / 1.83x | 15.23 / 61.09 | 5.86x / 2.10x |
| Furnished round room, 98 surfaces | Reference | 98,304 | 411.20 / 444.38 | 78.73 / 125.97 | 5.22x / 3.53x | 43.68 / 89.25 | 9.41x / 4.98x |

Numba FP32 was close to FP64 on this CPU rather than consistently faster. Its
maximum total-energy difference from FP64 was 0.00195%. Across both GPUs, the
maximum total-energy difference was 0.00556%, the maximum traced RT60 mean
difference was 0.0105%, and the maximum final RIR RT60 difference was 0.0001 s.

Small preview workloads can be dominated by launch, transfer, and CPU
post-processing overhead. Use Numba FP64 when exact CPU continuity matters;
use CUDA FP32 for simulation, reference, batch, and complex-room workloads.
