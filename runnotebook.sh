#!/bin/sh
# Check if running inside nix-shell
if [ -z "$IN_NIX_SHELL" ]; then
    echo "⚠️  Starting nix-shell environment..."
    exec nix-shell shell.nix --run "$0"
fi

export PYTHONPATH=.:$PYTHONPATH

# Add torch libs to LD_LIBRARY_PATH so HIP kernel .so can link at dlopen time
TORCH_LIB=$(uv run python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
fi

# Install PyTorch from AMD's gfx1151 index if not present
if ! uv run python -c "import torch" 2>/dev/null; then
    echo "🔧 Installing PyTorch with ROCm support for gfx1151..."
    uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision torchaudio
fi

# Build HIP kernel if not compiled
if [ ! -f hip_triangle/triangle_hip.so ]; then
    echo "🔧 Building HIP triangle kernel..."
    (cd hip_triangle && ./build.sh)
fi

uv run --all-extras jupyter notebook --ip=0.0.0.0 &
