# ROCm Setup for AMD Strix Halo on NixOS

## Quick Start

1. **Enter the development shell:**
```bash
nix-shell shell.nix
```

2. **Create Python 3.12 venv and install dependencies:**
```bash
rm -rf .venv  # Remove old Python 3.13 venv if exists
uv venv --python python3.12
uv sync --all-extras
```

3. **Install PyTorch with ROCm:**
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

4. **Verify GPU support:**
```bash
uv run python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

5. **Run notebooks:**
```bash
uv run jupyter notebook
```

## What the shell.nix Provides

- **ROCm 6.4 libraries**: hipBLAS, rocBLAS, rocFFT, hipFFT, rocRAND, RCCL
- **GFX version override**: Set to 11.0.2 for AMD Strix Halo (RDNA 3.5)
- **Library paths**: Properly configured LD_LIBRARY_PATH for PyTorch wheels
- **Standard libraries**: C++ stdlib and other dependencies for binary wheels

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

### PyTorch sees wrong architecture

The AMD Strix Halo uses gfx1103 (RDNA 3.5), but PyTorch ROCm wheels target gfx1100. The `HSA_OVERRIDE_GFX_VERSION=11.0.2` environment variable tells ROCm to use gfx1100 compatibility mode.

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
