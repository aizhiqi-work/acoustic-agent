# Contributing

Thank you for improving Acoustic Agent. Solver changes can affect timing,
energy, decay, and spatial rendering at once, so small, measurable pull requests
are preferred.

## Development Setup

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
acoustic-agent verify-resources --hashes
pytest
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Pull Requests

1. Describe the acoustic behavior being changed and why.
2. Add a focused regression test for changed solver or API behavior.
3. Preserve seeded reproducibility unless the change explicitly revises it.
4. Report accuracy and speed separately for performance work.
5. Run `pytest` and `acoustic-agent verify-resources --hashes`.
6. Avoid committing generated RIRs, logs, Numba caches, or virtual environments.

For solver changes, include at least one comparison of direct-path delay/gain,
reflection timing, per-band decay, or RIR energy. Performance changes must not
alter seeded outputs outside a documented numerical tolerance.

## Large Resources

SQLite and pickle files are managed by Git LFS; the bundled SOFA files are
versioned directly. Do not replace a bundled resource without updating
`acoustic_agent/resources/manifest.json`, its SHA-256, the relevant builder
script, tests, and `THIRD_PARTY_NOTICES.md`.

## Commit Style

Use concise imperative subjects. Conventional prefixes such as `feat:`, `fix:`,
`perf:`, `test:`, and `docs:` are welcome but not required.
