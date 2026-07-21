# Acoustic Accuracy Benchmark

The benchmark suite turns the solver's physical assumptions into repeatable,
reviewable evidence. It is separate from performance benchmarking: elapsed time
is recorded, but pass/fail decisions are based on acoustic measurements.

## Run It

```bash
acoustic-agent benchmark --profile quick --output benchmark-results
```

This writes:

- `accuracy-benchmark.json` for automation and regression dashboards;
- `accuracy-benchmark.md` for pull requests and release notes;
- `accuracy-benchmark.html` as a self-contained review report.

The command exits non-zero when a required case fails or errors. During
investigation, `--allow-failures` keeps the report but returns zero. Run one or
more cases with repeated `--case CASE_ID` options.

## Profiles

| Profile | Rays | Bounces | Purpose |
| --- | ---: | ---: | --- |
| `quick` | 8,192 | 32 | Deterministic local and CI regression |
| `full` | 131,072 | 96 | Release evidence and high-budget comparison |

Both profiles use fixed seeds and fixed metric scenes. The full profile is not
intended to run on every commit.

## Checks

| Case id | What it checks |
| --- | --- |
| `direct_arrival` | Path and rendered-peak arrival against `distance / c` |
| `distance_attenuation` | 2 m to 4 m pressure attenuation against 6.0206 dB |
| `shoebox_rt60` | Six-band path-traced RT60 against Sabine and Eyring |
| `early_reflections` | First-order image-source distance, time, order, and pressure law |
| `fdn_isolation` | FDN off/on equality before transition and difference after it |
| `portal_coupling` | Open/closed door path, energy, and decay-signature changes |
| `hrtf_consistency` | ITD, ILD, and energy-normalized binaural loudness |
| `dynamic_continuity` | Adjacent-frame direct delay and direct-energy continuity; raw sample peaks are diagnostic |
| `steam_audio_native` | Native Steam Audio and Acoustic Agent RT60 in the same shoebox |

The report may contain failures. These are evidence of a regression or known
solver limitation, not report-generation failures. In particular, sampled ray
representative gains are not presented as exact image-source impulses; the
early-reflection case labels the analytic amplitude-law and Monte Carlo
diagnostic values separately.

## Native Steam Audio Reference

The external comparison is optional because Steam Audio is not bundled with the
Python package. Point the benchmark at a local SDK or source checkout:

```bash
acoustic-agent benchmark \
  --profile full \
  --steam-audio-root /path/to/steam-audio \
  --output benchmark-results/full
```

The runner compiles the packaged `steam_audio_reference.cpp` into the report
cache and links it against the SDK's `libphonon`. It automatically probes Steam Audio's
public 3-band ABI and experimental 11-band octave ABI. The runner builds the
same 6 m x 4 m x 2.8 m uniform-material shoebox in both engines and compares
matching or nearest represented bands. If the SDK, dynamic library, or a C++
compiler is unavailable, the case is marked `SKIP`, never `PASS`.

Set `STEAM_AUDIO_ROOT` instead of passing the command-line option when useful in
CI. Native reference artifacts remain in the selected output directory and are
not package resources.
