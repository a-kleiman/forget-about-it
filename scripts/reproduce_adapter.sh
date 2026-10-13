#!/bin/bash
# Adapter-level continual learning for one model size, one GPU, sequentially (resumable).
#   bash scripts/reproduce_adapter.sh Qwen/Qwen3-14B            # default methods, order 1, seeds 17/42/123
#   ARMS="sfa_be25 sfa_be75 look_b25 look_b75" SEEDS=17 SPLITS=dev bash scripts/reproduce_adapter.sh Qwen/Qwen3-14B   # coefficient candidates
#   ORDER=4 bash scripts/reproduce_adapter.sh Qwen/Qwen3-14B    # reversed task order
# Approximate cost per stream (4 stages + dev/test evaluation) on one 141 GB GPU: 1.7B ~25 min, 8B ~40 min, 14B ~1 h, 32B ~2 h.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SFA_MODEL="${1:-Qwen/Qwen3-1.7B}"
ORDER="${ORDER:-1}"; SEEDS="${SEEDS:-17 42 123}"; ARMS="${ARMS:-seqlora replay sfa lookahead aim}"; SPLITS="${SPLITS:-dev test}"
CKPT=""; [ "$SFA_MODEL" = "Qwen/Qwen3-32B" ] && CKPT="--grad_ckpt"
for seed in $SEEDS; do
  for arm in $ARMS; do
    python -m sfa.run_adapter_stream --order "$ORDER" --seed "$seed" --arm "$arm" --splits $SPLITS $CKPT
  done
done
python -m sfa.analysis.adapter
