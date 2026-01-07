#!/bin/sh
# Check if running inside nix-shell
if [ -z "$IN_NIX_SHELL" ]; then
    echo "⚠️  Starting nix-shell environment..."
    exec nix-shell shell.nix --run "$0"
fi

export PYTHONPATH=.:$PYTHONPATH
uv run jupyter notebook --ip=0.0.0.0 &
