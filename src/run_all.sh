#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
for stage in calibrate text steer perception; do
  if [[ "$stage" == "calibrate" && -f results/directions.npz ]]; then
    continue
  fi
  .venv/bin/python src/run_model.py --stage "$stage" > "results/${stage}.log" 2>&1
done
