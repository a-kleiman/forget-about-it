#!/bin/bash
# Template for running any single command of this repository as a one-GPU cluster job. Fill in the placeholders.
#   sbatch scripts/slurm_template.sh python -m sfa.run_adapter_stream --order 1 --seed 17 --arm sfa
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# export SFA_MODEL=Qwen/Qwen3-14B      # adapter experiments: model size
"$@"
