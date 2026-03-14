{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  name = "asterisms-py-rocm";

  buildInputs = with pkgs; [
    # Python 3.12 (PyTorch doesn't support 3.13 yet)
    python312
    uv

    # ROCm packages for AMD GPU support
    rocmPackages.clr
    rocmPackages.hipcc          # HIP compiler for custom kernels
    rocmPackages.rocm-cmake     # CMake support for ROCm
    rocmPackages.rocm-device-libs  # Device libraries for compilation
    rocmPackages.rocprim        # ROCm primitives (required by rocThrust)
    rocmPackages.rocthrust      # Thrust library for HIP (required by PyTorch headers)
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
    # Add ROCm tools to PATH
    export PATH="${pkgs.rocmPackages.clr}/bin:${pkgs.rocmPackages.hipcc}/bin:$PATH"

    # Set ROCM_PATH for build scripts
    export ROCM_PATH="${pkgs.rocmPackages.clr}"
    export HIP_PATH="${pkgs.rocmPackages.clr}"
    export ROCPRIM_PATH="${pkgs.rocmPackages.rocprim}"
    export ROCTHRUST_PATH="${pkgs.rocmPackages.rocthrust}"
    export HIPSPARSE_PATH="${pkgs.rocmPackages.hipsparse}"
    export HIPBLAS_PATH="${pkgs.rocmPackages.hipblas}"

    # APU-specific setting to prevent artifacts
    export HSA_ENABLE_SDMA=0

    # NOTE: HSA_OVERRIDE_GFX_VERSION no longer needed - modern PyTorch has native gfx1151 support
    # (Previously: export HSA_OVERRIDE_GFX_VERSION=11.5.1)

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
      pkgs.rocmPackages.rocm-device-libs
    ]}:$LD_LIBRARY_PATH"

    echo "🚀 asterisms-py environment for AMD Strix Halo"
    echo "Python: $(python --version)"
    echo "GPU: gfx1151 (AMD Strix Halo - Radeon 8060S)"
    echo "ROCm: $(hipcc --version | head -1 || echo 'hipcc not found')"
    echo ""
    echo "💡 PyTorch GPU support available for gfx1151!"
    echo ""
    echo "Install PyTorch with ROCm support:"
    echo "  uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio"
    echo ""
    echo "🔥 HIP kernel compilation enabled"
    echo "  Build custom kernel: cd hip_triangle && ./build.sh"
    echo ""
    echo "To run notebooks:"
    echo "  uv run jupyter notebook --ip=0.0.0.0"
    echo "  or: ./runnotebook.sh"
  '';
}
