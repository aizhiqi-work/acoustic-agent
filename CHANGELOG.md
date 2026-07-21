# Changelog

All notable changes to Acoustic Agent are documented here. The project follows
semantic versioning after the initial `0.x` research releases.

## Unreleased

## 0.1.1 - 2026-07-22

### Added

- Unified `AcousticAgent.create()` entry point for Geometry, Floorplan, and
  Custom scenes.
- Compact dynamic execution through `agent.run(motion=...)`.
- Plain tuple/mapping inputs for `agent.run_batch()` and mixed-scene dataset
  production through `AcousticAgent.run_many()`.
- NPZ production manifests and optional per-job failure collection.
- Independent multi-source RIR simulation through `agent.run_sources()`.
- NumPy audio rendering, gain-preserving source mixing, and resampling helpers.
- Web auralization with bundled narration, background speech, two piano
  programs, a pink-noise bed, deterministic white/pink/brown noise, uploaded
  audio, and a separately positioned background source.
- Receiver-domain SNR mixing for independently propagated foreground and
  background signals.
- Accuracy benchmark reports for direct arrival, distance attenuation, RT60,
  reflections, FDN isolation, portals, HRTF, motion, and Steam Audio parity.
- Custom floor-plan generation from image-assisted or text-assisted JSON, plus
  semantic furniture auto-placement.
- Cached BVH intersection traversal with an exact linear reference mode.

### Changed

- Web workbench Python examples now use the compact unified API and emit valid
  Python literals for custom scene JSON.
- The background `Level` control is now a target broadband receiver SNR.
- Mono API simulations use a headless path that omits visualization-only RT
  paths and Ambisonic buffers.
- Geometry, Floorplan, and Custom scenes share one workbench and API style.

### Fixed

- Fractional-delay interpolation now preserves energy continuity between
  neighboring RIR samples.
- Dynamic trajectories, portal traversal, late reverberation, and displayed
  acoustic metrics received regression coverage in the benchmark suite.

### Bundled Data

- Added main narration, background speech, two piano programs, and a
  pink-noise bed to wheel and source distributions.

## 0.1.0 - 2026-07-20

### Added

- Geometry and audited Floorplan indoor scene construction.
- Unified Geometry/Floorplan WebGL workbench and local HTTP API.
- Direct, transmission, diffraction, RT energy-field, and FDN RIR pipeline.
- Open-portal cross-room pathing and coupled-room decay model.
- Mono, linear, circular, and bundled SOFA HRTF receivers.
- Directional sources, semantic materials, acoustic furniture, and motion.
- Numba JIT tracing, deterministic caching, and quality presets.
- Resource-complete wheel/sdist packaging with integrity verification.

### Bundled Data

- CIPIC subject 124 and SADIE II H12 SOFA files.
- Acoustic Materials V3 SQLite database and inspection indexes.
- Floorplan V1 SQLite database with 15,376 audited scenes.
