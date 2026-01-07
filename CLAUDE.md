# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`asterisms-py` uses Polars and PyTorch to search astronomical star catalogs (Tycho-2) for interesting asterisms - geometric patterns formed by stars. The project processes ~2.4M stars to find triangular and square patterns based on spatial and magnitude criteria.

## Development Workflow

This project uses **nbdev** - Python modules are auto-generated from Jupyter notebooks. The source of truth is the notebooks in `nbs/`, not the Python files in `asterisms_py/`.

### Key Commands

**Install dependencies:**
```bash
uv sync
# or with dev dependencies:
uv sync --all-extras
```

**Run notebooks:**
```bash
./runnotebook.sh
# or manually (accessible from network):
uv run jupyter notebook --ip=0.0.0.0
```

**Export notebooks to Python modules:**
```bash
uv run nbdev_export
```

**Work with notebooks in order:**
1. `nbs/00_tycho2_dat_to_parquet.ipynb` - Convert Tycho-2 catalog to Parquet
2. `nbs/01_polars+pytorch.ipynb` - Core processing logic (scoring, region processing)
3. `nbs/02_processing_results.ipynb` - Result visualization and constellation mapping

### Package Management

Uses uv for fast, reliable Python package management. Dependencies are specified in `pyproject.toml` using standard PEP 621 format.

**PyTorch Installation (AMD Strix Halo on NixOS):**

✅ **GPU Support Available!** AMD provides gfx1151-specific PyTorch wheels with ROCm 7.10.0.

```bash
nix-shell shell.nix
# Then inside the shell:
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio
```

**Verification:**
```bash
uv run python test_gpu.py
# Expected: ROCm available: True, Device: Radeon 8060S Graphics
```

**Benchmark Results (AMD Strix Halo gfx1151):**
- CPU (16-core): 2436 GFLOPS (2000×2000 matmul)
- GPU (40 CU): 2371 GFLOPS
- Note: For this APU, performance varies by workload. GPU excels at highly parallel operations.

See `README_ROCM.md` for detailed setup and troubleshooting.

## Architecture

### Data Pipeline

**Input:** Tycho-2 star catalog (`support/tyc2.dat` → `support/tyc2.parquet`)
- 2.4M+ stars with RA, Dec, magnitudes (BT, VT)
- Computes visual magnitude: `V = VT - 0.090*(BT-VT)`

**Processing:** Grid-based region search
- Sky divided into 4° grid (361 RA points × declination range)
- Each region analyzed for geometric patterns using PyTorch
- Results stored in parquet files (`result_triangle2.parquet`)

**Output:** Scored asterism candidates with visualizations

### Key Modules (Auto-generated from notebooks)

**`asterisms_py/tycho2_ingest.py`** (from `00_tycho2_dat_to_parquet.ipynb`)
- `read_tycho2()` - Parse Tycho-2 DAT format to Polars DataFrame

**`asterisms_py/tycho2_main.py`** (from `01_polars+pytorch.ipynb`)
- Coordinate transformations: RA/Dec → Cartesian (distance from magnitude)
- Scoring functions: `mass_score_triangle_torch()`, `measure_squareness()`
- Region queries: `get_region()`, `stars_for_center_and_radius()`
- Grid utilities: `get_grid_points()`, `get_grid_point_by_idx()`

### Coordinate System

- Uses J2000 epoch
- RA in degrees (0-360), Dec in degrees (-90 to +90)
- Distance estimation from apparent magnitude assuming absolute magnitude M=0
- Cartesian conversion for geometric calculations

### GPU Acceleration

**Status: ✅ Working** (AMD Strix Halo gfx1151 with ROCm 7.10.0)

Automatically use GPU when available:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scores, points = mass_score_triangle_torch(tensor, device=device)
```

Or explicitly:
```python
scores, points = mass_score_triangle_torch(tensor, device='cuda')  # Use GPU
```

**ROCm provides CUDA API compatibility** - use `device='cuda'` for AMD GPUs.

**Important:** Do not use `device='mps'` (Apple Metal) - this will fail on Linux/NixOS.

## Important Paths

- `support/tyc2.dat` - Raw Tycho-2 catalog (525MB)
- `support/tyc2.parquet` - Processed catalog (49MB)
- `nbs/` - Source notebooks (edit these, not Python files)
- `asterisms_py/` - Auto-generated Python modules (do not edit directly)
- `star-charts/` - Star chart generation utilities
- `de421.bsp`, `hip_main.dat`, `constellationship.fab` - Ephemeris and constellation data for Skyfield

## Editing Code

**CRITICAL RULE: NEVER EDIT EXPORTED PYTHON FILES**

- ✅ **DO:** Edit notebooks in `nbs/` - these are the source of truth
- ❌ **DON'T:** Edit Python files in `asterisms_py/` - these are auto-generated
- Cells marked with `#| export` are exported to Python modules
- After editing notebooks, run `nbdev_export` to regenerate Python files
- If you need to fix code, find it in the notebook and edit it there

## Dependencies

Core stack:
- **polars** - DataFrame operations on star catalogs
- **torch** - Geometric scoring and tensor operations
- **pyarrow** - Parquet file I/O
- **skyfield** - Astronomical coordinate transformations, constellation mapping
- **matplotlib** - Star chart visualization
- **starplot** - Star chart rendering
- **scipy** - Scientific computing utilities

Dev dependencies:
- **nbdev** - Notebook-driven development
- **jupyter** / **jupyterlab-vim** - Interactive development
