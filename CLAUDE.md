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
# or manually:
uv run jupyter notebook
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

⚠️ **GPU Limitation:** The Strix Halo's gfx1102 architecture isn't supported by PyTorch ROCm 6.2 wheels yet. Use CPU-only for now.

```bash
nix-shell shell.nix
# Then inside the shell:
rm -rf .venv
uv venv --python python3.12
uv sync --all-extras
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Verification:**
```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

The shell.nix provides Python 3.12 and all C++ libraries needed by PyTorch wheels. When GPU support becomes available, switch to:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2  # Future
```

See `README_ROCM.md` for detailed GPU status and building from source.

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

Code supports GPU acceleration. Current deployment uses AMD Strix Halo with ROCm:
```python
scores, points = mass_score_triangle_torch(tensor, device='cuda')  # ROCm provides CUDA compatibility
```

Historical note: Code originally used `device='mps'` for Apple Silicon.

## Important Paths

- `support/tyc2.dat` - Raw Tycho-2 catalog (525MB)
- `support/tyc2.parquet` - Processed catalog (49MB)
- `nbs/` - Source notebooks (edit these, not Python files)
- `asterisms_py/` - Auto-generated Python modules (do not edit directly)
- `star-charts/` - Star chart generation utilities
- `de421.bsp`, `hip_main.dat`, `constellationship.fab` - Ephemeris and constellation data for Skyfield

## Editing Code

**CRITICAL:** Edit notebooks in `nbs/`, not Python files. Cells marked with `#| export` are exported to modules. After editing notebooks, run `nbdev_export` to regenerate Python files.

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
