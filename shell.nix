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
    export HSA_OVERRIDE_GFX_VERSION=11.0.2  # AMD Strix Halo (RDNA 3.5)

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

    echo "🚀 asterisms-py environment with ROCm for AMD Strix Halo"
    echo "Python: $(python --version)"
    echo "ROCm: $(rocm-smi --version 2>/dev/null || echo 'ROCm tools available')"
    echo ""
    echo "To install PyTorch with ROCm:"
    echo "  uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
    echo ""
    echo "To run notebooks:"
    echo "  uv run jupyter notebook"
  '';
}
