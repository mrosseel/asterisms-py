# ROCm Setup for AMD Strix Halo on NixOS

## ✅ GPU Support Working!

**GPU Architecture:** gfx1151 (AMD Strix Halo - Radeon 8060S Graphics)
**PyTorch Support:** ✅ AMD provides gfx1151-specific wheels with ROCm 7.10.0
**Status:** Fully functional GPU acceleration

## Quick Start (GPU Enabled)

1. **Enter the development shell:**
```bash
nix-shell shell.nix
```

2. **Create Python 3.12 venv and install dependencies:**
```bash
rm -rf .venv  # Remove old venv if needed
uv venv --python python3.12
uv sync --all-extras
```

3. **Install PyTorch (CPU-only for now):**
```bash
# CPU version (works reliably)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# OR ROCm version (GPU detected but kernels fail on gfx1102)
# uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

4. **Verify installation:**
```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'Device: cuda' if torch.cuda.is_available() else 'cpu')"
```

5. **Run notebooks:**
```bash
uv run jupyter notebook
```

## What the shell.nix Provides

- **ROCm 6.4 libraries**: hipBLAS, rocBLAS, rocFFT, hipFFT, rocRAND, RCCL
- **APU optimization**: HSA_ENABLE_SDMA=0 to prevent artifacts
- **Library paths**: Properly configured LD_LIBRARY_PATH for PyTorch wheels
- **Standard libraries**: C++ stdlib and other dependencies for binary wheels

**Note:** HSA_OVERRIDE_GFX_VERSION is no longer needed - modern PyTorch wheels have native gfx1151 support!

## Troubleshooting

### "ROCm not available" or cuda.is_available() returns False

1. **Check GPU is recognized:**
```bash
rocm-smi
```

2. **Verify HSA override:**
```bash
echo $HSA_OVERRIDE_GFX_VERSION
# Should show: 11.0.2
```

3. **Check permissions:**
Ensure your user is in the `video` and `render` groups (should be configured in NixOS config).

### ImportError: libstdc++.so.6 not found

Make sure you're running from inside `nix-shell shell.nix`. The shell provides all necessary libraries.

### GPU VRAM Management (CRITICAL for Performance)

The AMD Strix Halo APU allocates VRAM dynamically. Default is only **512MB**, which severely limits GPU utilization.

**Check current VRAM:**
```bash
gpu-vram-info
```

**Increase VRAM allocation** (requires GPU restart - will kill Jupyter):
```bash
# Quick presets
gpu-vram-presets  # Interactive menu

# Or direct allocation
gpu-vram-set 8192   # 8GB - good for medium workloads
gpu-vram-set 16384  # 16GB - recommended for full catalog processing
gpu-vram-set 65536  # 64GB - maximum for large-scale parallel processing
```

**VRAM Requirements for Asterism Processing:**
- Star catalog: ~28 MB (loaded once to GPU)
- Per region (worst case): ~3.5 GB
- **Recommended:** 16GB minimum, 64GB for optimal performance
- With 96GB VRAM: Can process ~28 regions in parallel

### GPU Optimization Features

The asterism processing code includes GPU-specific optimizations:

1. **Catalog Preloading**: Entire star catalog loaded to VRAM once (~28MB)
2. **GPU Filtering**: Star filtering done on GPU using boolean masks (much faster than CPU)
3. **Persistent GPU Tensors**: Intermediate results kept on GPU between operations
4. **Minimal CPU-GPU Transfers**: Only final results transferred back to CPU

**Enable GPU optimization in notebooks:**
```python
# Automatically uses GPU if available
process_regions(grid_points, start=0, use_gpu=True)
```

**Expected performance improvement:**
- Without optimization: 6% GPU utilization, 90%+ VRAM full (512MB)
- With optimization: 60-90% GPU utilization, improved throughput

### Performance Notes

The AMD Strix Halo (gfx1151) has unique characteristics as an APU:
- **Integrated GPU**: Shares memory with CPU, reducing transfer overhead
- **40 RDNA 3.5 compute units**: Good for parallel workloads
- **16-core CPU**: Very powerful, competitive with GPU for some tasks

**Benchmark Results:**
- Matrix multiplication (2000×2000): CPU ~2436 GFLOPS, GPU ~2371 GFLOPS
- GPU advantage increases with highly parallel operations (e.g., many triangles)
- Best for workloads with thousands of concurrent operations
- **Real workload**: Processing 2.4M stars, 2912 regions, ~4M triangles per region average

## System-Wide ROCm (Optional)

If you want ROCm available system-wide instead of just in this shell, add to your NixOS configuration:

```nix
# In ~/nixos-config/machines/nixtop/config.nix or similar

{
  # ROCm support
  hardware.graphics = {
    enable = true;
    extraPackages = with pkgs.rocmPackages; [
      clr
      rocm-runtime
      rocm-smi
      rocblas
      hipblas
      rocfft
      hipfft
      rccl
    ];
  };

  # Environment variables for ROCm
  environment.variables = {
    HSA_OVERRIDE_GFX_VERSION = "11.0.2";  # AMD Strix Halo
    ROC_ENABLE_PRE_VEGA = "1";
  };
}
```

Then rebuild your system:
```bash
sudo nixos-rebuild switch
```

## Performance Notes

- The AMD Strix Halo has 40 RDNA 3.5 compute units
- Memory allocation is configured in `configuration.nix` via TTM pages_limit
- Current setting: 64GB VRAM allocation (16777216 pages)
- ROCm provides CUDA API compatibility, so use `device='cuda'` in PyTorch code

## References

- [Strix Halo Wiki - AI Capabilities](https://strixhalo.wiki/AI/AI_Capabilities_Overview)
- [NixOS ROCm Documentation](https://nixos.wiki/wiki/AMD_GPU)
- [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
