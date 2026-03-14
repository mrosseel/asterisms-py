#!/usr/bin/env bash
# Manual build script for HIP kernel on NixOS with ROCm

set -e

echo "Building HIP triangle kernel..."

# Check if we're in nix-shell
if [ -z "$ROCM_PATH" ]; then
    echo "Error: Not in nix-shell environment"
    echo "Run: nix-shell"
    echo "Then: cd hip_triangle && ./build.sh"
    exit 1
fi

# Find hipcc
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found in PATH"
    echo "Make sure you're in nix-shell"
    exit 1
fi

# Get PyTorch paths (use uv to access venv with torch)
# Use tail -1 to extract only the last line (the actual path), filtering out ROCm SDK warnings
PYTHON_INCLUDE=$(uv run python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>&1 | tail -1)
TORCH_PATH=$(uv run python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>&1 | tail -1)
TORCH_INCLUDE="$TORCH_PATH/include"
TORCH_LIB="$TORCH_PATH/lib"

# Verify paths exist
if [ ! -d "$TORCH_INCLUDE" ]; then
    echo "Error: PyTorch include directory not found at $TORCH_INCLUDE"
    exit 1
fi

if [ ! -d "$TORCH_LIB" ]; then
    echo "Error: PyTorch lib directory not found at $TORCH_LIB"
    exit 1
fi

echo "ROCm path: $ROCM_PATH"
echo "Python include: $PYTHON_INCLUDE"
echo "PyTorch path: $TORCH_PATH"
echo "PyTorch include: $TORCH_INCLUDE"
echo "PyTorch lib: $TORCH_LIB"

# Compile HIP kernel to object file
echo "Compiling triangle_kernel.hip..."
hipcc \
    -c triangle_kernel.hip \
    -o triangle_kernel.o \
    -O3 \
    --offload-arch=gfx1151 \
    -fPIC \
    -std=c++17

# Compile C++ binding
echo "Compiling hip_binding.cpp..."
hipcc \
    -c hip_binding.cpp \
    -o hip_binding.o \
    -I"$PYTHON_INCLUDE" \
    -I"$TORCH_INCLUDE" \
    -I"$ROCM_PATH/include" \
    -I"$ROCPRIM_PATH/include" \
    -I"$ROCTHRUST_PATH/include" \
    -I"$HIPSPARSE_PATH/include" \
    -I"$HIPBLAS_PATH/include" \
    -O3 \
    -fPIC \
    -std=c++17 \
    -D__HIP_PLATFORM_AMD__ \
    -DTORCH_EXTENSION_NAME=triangle_hip

# Link into shared library
echo "Linking triangle_hip.so..."
hipcc \
    triangle_kernel.o \
    hip_binding.o \
    -o triangle_hip.so \
    -shared \
    -L"$TORCH_LIB" \
    -ltorch \
    -ltorch_python \
    -lc10 \
    -lamdhip64

echo "✓ Build successful: triangle_hip.so"
echo ""
echo "To use in Python:"
echo "  import sys"
echo "  sys.path.insert(0, '/home/mike/dev/amateur_astro/py-asterisms/hip_triangle')"
echo "  from hip_triangle import score_triangles_hip, HIP_AVAILABLE"
