# HIP Triangle Kernel for AMD GPUs

Pure HIP kernel implementation for massively parallel triangle generation and scoring.

## Why HIP?

- **Eliminates `torch.combinations` overhead** - generates triangles directly in GPU kernel
- **Fully parallel** - each GPU thread computes one triangle independently
- **AMD native** - uses ROCm/HIP, not CUDA emulation
- **Zero CPU involvement** - entire computation on GPU

## Architecture

```
GPU Kernel Flow:
1. Thread ID → (i, j, k) combinatorial unranking (O(1) per thread)
2. Fetch 3 star positions from global memory
3. Compute triangle side lengths
4. Normalize and score
5. Write to output arrays
```

For 100 stars → 161,700 triangles processed in **parallel** across GPU cores.

## Build Instructions

**NixOS with ROCm:**

```bash
nix-shell  # Enter the development environment
cd hip_triangle
./build.sh  # Compiles triangle_kernel.hip and hip_binding.cpp
```

The build script:
- Detects PyTorch paths automatically (include headers and libraries)
- Compiles the HIP kernel for gfx1151 (AMD Strix Halo)
- Links against PyTorch and ROCm libraries
- Produces `triangle_hip.so` Python extension module

**Standard Linux:**
```bash
cd hip_triangle
python setup.py install  # Uses PyTorch's CUDAExtension
```

## Requirements

- ROCm 5.0+
- PyTorch with ROCm support
- hipcc compiler (comes with ROCm)

## Usage

**In Python (inside nix-shell or with proper LD_LIBRARY_PATH):**

```bash
# Set library path to find libtorch.so at runtime
export LD_LIBRARY_PATH=/path/to/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
```

```python
import sys
sys.path.insert(0, 'hip_triangle')
from hip_triangle import score_triangles_hip, HIP_AVAILABLE

if HIP_AVAILABLE:
    # stars_tensor: [n_stars, 3] on GPU
    scores, points = score_triangles_hip(stars_tensor)
    # scores: [n_triangles] float32 on GPU
    # points: [n_triangles, 3, 3] - the 3 vertices of each triangle

print(f"Generated {len(scores)} triangles")
```

**Note:** The HIP kernel runs on GPU and returns GPU tensors. No CPU transfer needed!

## Integration with Existing Code

Replace this:
```python
scores, points = mass_score_triangle_torch(stars_tensor, device='cuda')
```

With this:
```python
scores, points = score_triangles_hip(stars_tensor)
```

Same interface, pure GPU implementation.

## Performance

Expected speedup: **10-100x** over `torch.combinations` depending on star count.

- Small regions (50 stars): ~10x
- Large regions (200 stars): ~100x

The more stars, the bigger the win, since combinatorial generation overhead dominates.

## Troubleshooting

**Build fails:**
```bash
# Check ROCm version
rocm-smi --showproductname

# Verify hipcc is in PATH
which hipcc

# Verbose build
python setup.py install --verbose
```

**Runtime error "HIP kernel not compiled":**
```bash
cd hip_triangle
python setup.py install
# Then restart Jupyter kernel
```
