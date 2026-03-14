# HIP Kernel Usage in Jupyter Notebook

## Build the HIP Kernel

First, build the kernel in nix-shell:

```bash
nix-shell
cd hip_triangle
./build.sh
```

Expected output:
```
✓ Build successful: triangle_hip.so
```

## Use in Jupyter Notebook

The notebook (01_polars+pytorch.ipynb) now includes HIP kernel support automatically.

### Automatic Setup (cell: hip_kernel_setup_001)

The notebook automatically:
1. Sets `LD_LIBRARY_PATH` to find libtorch.so
2. Adds `hip_triangle/` to Python path
3. Imports `score_triangles_hip` if available
4. Shows status message

### Benchmark Comparison

Run the benchmark cell (`benchmark_approaches`) to compare:

1. **Original** - Sequential per-region processing
2. **Vectorized** - Batched filtering + torch.combinations
3. **CUDA Streams** - 4 parallel streams + torch.combinations
4. **HIP Kernel** - Pure GPU triangle generation (NO torch.combinations)

Expected results:
```
PERFORMANCE SUMMARY
======================================================================
original       :   0.69s  (speedup: 1.00x)
vectorized     :   0.49s  (speedup: 1.41x)
streams        :   0.14s  (speedup: 5.08x)
hip_kernel     :   0.0Xs  (speedup: XX.XXx)  <- Expected 10-100x
======================================================================
```

## How HIP Kernel Works

**torch.combinations approach:**
```python
# Generate combinations on GPU (overhead!)
idx_combinations = torch.combinations(torch.arange(n), r=3)
# Then process triangles
p1, p2, p3 = tensor[idx_combinations[:, 0]], ...
```

**HIP kernel approach:**
```python
# Direct GPU kernel - generates triangles in parallel
scores, points = score_triangles_hip(stars_tensor)
# Each GPU thread computes one triangle independently
# Uses combinatorial unranking: thread_id → (i, j, k)
```

## Performance Characteristics

| Approach | Triangle Generation | Scoring | Python Overhead |
|----------|-------------------|---------|-----------------|
| torch.combinations | GPU (slow) | GPU | High (per region) |
| CUDA Streams | GPU (slow) | GPU | Medium (async) |
| HIP Kernel | **GPU (fast)** | GPU | Low (batched) |

**Key advantage:** HIP kernel eliminates the expensive `torch.combinations` call entirely, generating triangle indices directly in the kernel using O(1) math.

## Troubleshooting

### "HIP kernel not available"
Build it:
```bash
nix-shell
cd hip_triangle && ./build.sh
```

### "ImportError: libtorch.so: cannot open shared object"
Run Jupyter from nix-shell:
```bash
nix-shell
./runnotebook.sh
```

### "HIP kernel import failed"
Check that `triangle_hip.so` exists:
```bash
ls -lh hip_triangle/triangle_hip.so
```

Should show: `249K` file

## Integration with Production Code

To use HIP kernel in your processing pipeline:

```python
# At the top of your processing function
if HIP_AVAILABLE:
    scores, points = score_triangles_hip(stars_tensor)
else:
    # Fallback to torch.combinations
    scores, points = mass_score_triangle_torch(stars_tensor, device='cuda')
```

The HIP kernel has the same interface as `mass_score_triangle_torch`, so it's a drop-in replacement.
