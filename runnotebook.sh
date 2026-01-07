#!/bin/sh
export PYTHONPATH=.:$PYTHONPATH
uv run jupyter notebook --ip=0.0.0.0 &
