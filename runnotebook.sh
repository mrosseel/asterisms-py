#!/bin/sh
export PYTHONPATH=.:$PYTHONPATH
uv run jupyter notebook &
