# Changelog

All notable changes to Acoustic Agent are documented here. The project follows
semantic versioning after the initial `0.x` research releases.

## Unreleased

### Added

- Unified `AcousticAgent.create()` entry point for Geometry, Floorplan, and
  Custom scenes.
- Compact dynamic execution through `agent.run(motion=...)`.
- Plain tuple/mapping inputs for `agent.run_batch()` and mixed-scene dataset
  production through `AcousticAgent.run_many()`.
- NPZ production manifests and optional per-job failure collection.

### Changed

- Web workbench Python examples now use the compact unified API and emit valid
  Python literals for custom scene JSON.

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
