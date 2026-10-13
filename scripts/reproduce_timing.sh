#!/bin/bash
# Controlled single-stage training-loop timing (SFA vs replay vs AIMMerging) on one GPU. Needs the model's sequential-LoRA
# order-1 seed-17 stage-1 adapter (first stage of `python -m sfa.run_adapter_stream --order 1 --seed 17 --arm seqlora`).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SFA_MODEL="${1:-Qwen/Qwen3-14B}"
python -m sfa.timing
python -m sfa.analysis.timing
