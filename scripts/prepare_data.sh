#!/bin/bash
# Download, process and tokenize all data, and build every batch schedule (CPU only; about 30-60 min with 16+ cores).
# All data are public and downloaded here at pinned revisions (Hugging Face Hub and one GitHub repository).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m sfa.data.prepare_math_law                                   # Math (MetaMathQA / GSM8K) + Law (CaseHOLD / ToS / Overruling)
python -m sfa.data.prepare_cl                                         # 4-task continual-learning benchmark (pinned upstream commit)
python -m sfa.data.tokenize --tokenize math law cl_dbpedia cl_amazon cl_yahoo cl_agnews
python -m sfa.data.tokenize --schedule math law --seeds 17 42 123 7 2026   # full-parameter 10M-token schedules
python -m sfa.data.adapter_schedules --orders 1 4 --seeds 17 42 123        # adapter-level released schedules + memory buffers
