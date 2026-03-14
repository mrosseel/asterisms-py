#!/usr/bin/env bash
# Build script for HIP square kernel on NixOS with ROCm

set -e

echo "Building HIP square kernel..."

if [ -z "$ROCM_PATH" ]; then
    echo "Error: Not in nix-shell environment"
    echo "Run: nix-shell"
    echo "Then: cd hip_square && ./build.sh"
    exit 1
fi

if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found in PATH"
    exit 1
fi

PYTHON_INCLUDE=$(uv run python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>&1 | tail -1)
TORCH_PATH=$(uv run python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>&1 | tail -1)
TORCH_INCLUDE="$TORCH_PATH/include"
TORCH_LIB="$TORCH_PATH/lib"

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

echo "Compiling square_kernel.hip..."
hipcc \
    -c square_kernel.hip \
    -o square_kernel.o \
    -O3 \
    --offload-arch=gfx1151 \
    -fPIC \
    -std=c++17

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
    -DTORCH_EXTENSION_NAME=square_hip

echo "Linking square_hip.so..."
hipcc \
    square_kernel.o \
    hip_binding.o \
    -o square_hip.so \
    -shared \
    -L"$TORCH_LIB" \
    -ltorch \
    -ltorch_python \
    -lc10 \
    -lamdhip64

echo "✓ Build successful: square_hip.so"
