{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  name = "asterisms-py-rocm";

  buildInputs = with pkgs; [
    # Python 3.12 (PyTorch doesn't support 3.13 yet)
    python312
    uv

    # ROCm packages for AMD GPU support
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.hipfft
    rocmPackages.hipsolver
    rocmPackages.hipsparse
    rocmPackages.rccl
    rocmPackages.rocblas
    rocmPackages.rocfft
    rocmPackages.rocm-smi
    rocmPackages.rocrand

    # Standard libraries needed by PyTorch binary wheels
    stdenv.cc.cc.lib
    zlib
    zstd

    # For Jupyter/development
    pkg-config
  ];

  shellHook = ''
    # Add ROCm to PATH
    export PATH="${pkgs.rocmPackages.clr}/bin:$PATH"
    export HSA_OVERRIDE_GFX_VERSION=11.5.1  # AMD Strix Halo (gfx1151)

    # Library paths for dynamic linking
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.zstd
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-runtime
      pkgs.rocmPackages.rocblas
      pkgs.rocmPackages.hipblas
      pkgs.rocmPackages.rocfft
      pkgs.rocmPackages.hipfft
    ]}:$LD_LIBRARY_PATH"

    echo "🚀 asterisms-py environment for AMD Strix Halo"
    echo "Python: $(python --version)"
    echo "GPU: gfx1151 (AMD Strix Halo - Radeon 8060S) - ROCm available but PyTorch kernels missing"
    echo ""
    echo "⚠️  PyTorch GPU support unavailable (gfx1151 not in ROCm 6.2 wheels)"
    echo ""
    echo "Install PyTorch CPU-only:"
    echo "  uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
    echo ""
    echo "To run notebooks:"
    echo "  uv run jupyter notebook --ip=0.0.0.0"
    echo "  or: ./runnotebook.sh"
  '';
}
