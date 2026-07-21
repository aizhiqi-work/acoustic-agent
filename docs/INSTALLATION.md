# Installation

## Supported Environment

Acoustic Agent supports CPython 3.10 and newer on macOS, Linux, and Windows.
Python 3.12 is used for the primary local development environment. A browser
with WebGL is required only for the visual workbench.

Runtime dependencies:

- NumPy for numeric arrays and signal processing.
- Numba for JIT-compiled tracing and FDN kernels.
- h5py for SOFA HRTF files.
- Shapely and NetworkX for Floorplan geometry, portals, and motion routing.

## Clone With Binary Resources

SOFA and SQLite resources are intentionally part of the project. The SQLite
databases are stored with Git LFS; the two SOFA files are versioned directly.

```bash
git lfs install
git clone <repository-url>
cd acoustic-agent
git lfs pull
```

Check that the large files are present:

```bash
git lfs ls-files
```

The LFS list must include the material and Floorplan SQLite databases. The two
SOFA files must exist under `acoustic_agent/resources/hrtf/` as regular files.

## Editable Development Install

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
acoustic-agent verify-resources --hashes
pytest
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
acoustic-agent verify-resources --hashes
pytest
```

## Regular Local Install

```bash
python -m pip install .
acoustic-agent info
acoustic-agent verify-resources --hashes
```

The generated wheel contains the frontend, both SOFA files, both SQLite
databases, and the material indexes. Runtime access does not depend on the
original source checkout.

## Build Distributions

```bash
python -m pip install -e ".[dev]"
rm -rf build dist
python -m build
python -m twine check dist/*
```

Before publishing, verify the files in the wheel:

```bash
python -m zipfile -l dist/acoustic_agent-0.1.1-py3-none-any.whl
```

The wheel is large because it is deliberately resource-complete. Do not create
a smaller release by silently dropping SOFA or SQLite files. If a future thin
distribution is needed, publish it under a distinct package name or explicit
optional-resource contract.

## Install A Built Wheel

```bash
python -m pip install dist/acoustic_agent-0.1.1-py3-none-any.whl
acoustic-agent verify-resources --hashes
acoustic-agent web
```

## Offline Installation

On a networked machine with the same platform and Python version:

```bash
mkdir wheelhouse
python -m pip download --dest wheelhouse .
python -m build
cp dist/*.whl wheelhouse/
```

Move `wheelhouse` to the offline machine, then run:

```bash
python -m pip install --no-index --find-links wheelhouse acoustic-agent
```

## Numba Startup And Cache

The first process to execute a new kernel signature compiles it. The Web command
performs a tiny warmup so normal interaction does not pay all compilation cost.
Numba writes `.nbc` and `.nbi` cache files beside Python bytecode; these are
ignored by Git.

Changing Python, NumPy, Numba, CPU architecture, or relevant source code can
invalidate the cache. This is normal and does not change solver accuracy.

## Troubleshooting

### `file signature not found` for a SOFA file

The checkout probably contains a Git LFS pointer instead of HDF5 content:

```bash
git lfs pull
acoustic-agent verify-resources --hashes
```

### `file is not a database`

The SQLite file is probably also an LFS pointer. Run `git lfs pull` and verify
again.

### The first simulation is slow

Allow the default Web warmup to complete. CLI and Python processes also pay JIT
compilation once per compatible cache. Benchmark the second and later runs when
measuring steady-state solver speed.

### Port 8765 is already in use

```bash
acoustic-agent web --port 8766
```

### The Web UI should not be public

The built-in server is a local development server without TLS or authentication.
Keep `--host 127.0.0.1`, or place it behind a secured production service.
