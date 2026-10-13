#!/bin/bash
# Full-parameter Math->Law (Qwen3-1.7B), one GPU, sequentially. Every step is resumable.
# Approximate cost on one 141 GB GPU: Math start ~17 min/seed; each Law run ~12 min; each evaluation 4-6 min per split.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m sfa.run_full_param --set selection_dev       # seed 17: development candidates + endpoint/EMA grids (dev)
python -m sfa.run_full_param --set main_test           # seeds 17/42/123: reported configurations (dev + test)
python -m sfa.run_full_param --set replication_test    # seeds 7/2026: the three R_mid configurations (dev + test)
python -m sfa.analysis.full_param                      # results/tables/full_param_*.md
