# Forget About It: Mitigating Forgetting Without Replay or Auxiliary Objectives

Code for the paper

> **Forget About It: Mitigating Forgetting Without Replay or Auxiliary Objectives**<br>
> Anat Kleiman, Gintare Karolina Dziugaite, Jonathan Frankle, Sham Kakade, Mansheej Paul<br>
> Preprint, 2026. [arXiv link to be added]

The paper studies simple in-training interpolation as a forgetting-mitigation baseline: **Sequential Fine-tuning with
Averaging (SFA)**, introduced in the paper, and replay-free **Lookahead**. This repository reproduces the controlled
multi-seed Qwen3 experiments:

1. **Full-parameter Math → Law** (Qwen3-1.7B; Section 5.1, Figure 1 and Appendix B): SFA vs. Lookahead vs. fine-tuning (FT)
   with development-selected hyperparameters, plus end-of-training interpolation and EMA baselines; five seeds for the main
   comparison.
2. **Adapter-level continual learning** (Qwen3-1.7B / 8B / 14B / 32B, one shared LoRA adapter, DBpedia → Amazon → Yahoo
   Answers → AG News; Section 5.3, Figure 3 and Appendices A and C): SFA vs. sequential LoRA, replay, AIMMerging (our port,
   see **Third-party code and data**) and Lookahead, coefficient selection, SFA with replay (SFA-R), a reversed task order,
   and a controlled single-stage training-time benchmark (Figure 3b).

The other experiments reported in the paper (Llama 2-7B Math → Law, full-parameter four-task replay comparison, and the
model-family and vision experiments in Appendices F–H) are not part of this release.

**SFA** periodically interpolates the working weights toward a checkpoint that is held fixed for the whole current task
(the weights at the start of the task), and then continues training from the interpolated weights:

```
anchor a <- weights at the start of the task
for update t = 1..T:
    w <- optimizer_step(w)
    if t is a merge step (every k = round(pT) updates, and at T):   w <- (1 - beta) * w + beta * a
```

With `--beta_end b`, the per-merge coefficient is `beta = 1 - (1 - b)^(1/K)` for K merges, so the gradient-free contraction
toward the anchor over the task is `1 - b`. **Lookahead** uses the same merge steps but a *moving* slow copy
(`s <- (1 - beta) w + beta s;  w <- s`). Adam moments are kept across merges; the optimizer restarts at each task.

## Contents

```
sfa/
  formatting.py            paths (SFA_HOME), model selection (SFA_MODEL), Qwen3 chat formatting (thinking disabled)
  data/                    prepare_math_law.py, prepare_cl.py, tokenize.py, adapter_schedules.py
  train_full.py            full-parameter training: ft / sfa / lookahead (+ eval-only EMA shadows)
  train_lora.py            adapter training, one stage: seqlora / replay / sfa / lookahead / aim / sfa_replay
  aim.py                   port of the released AIMMerging controller (merge timing, fusion, memory buffer)
  export_lora.py           merge an adapter into dense weights for evaluation
  evaluate.py              GSM8K (generation), Law (multiple-choice scoring), continual-learning tasks (exact match)
  run_full_param.py        driver for the Math -> Law experiments (resumable)
  run_adapter_stream.py    driver for one 4-stage adapter stream (resumable)
  timing.py                single-stage training-loop timing benchmark
  analysis/                full_param.py, adapter.py, timing.py: the result tables
configs/                   models.json (pinned model / dataset revisions), full_param.json, adapter.json
scripts/                   prepare_data.sh, reproduce_full_param.sh, reproduce_adapter.sh, reproduce_timing.sh, slurm_template.sh
```

The repository contains code and configuration only: no datasets, checkpoints or results. All data are public and are
downloaded by the preparation scripts (see **1. Data**).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate          # Python 3.12
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
export PYTHONPATH=$PWD                                      # or run everything from the repository root
```

Every command below runs on one GPU and can be launched directly or as a batch job on your cluster:
`scripts/slurm_template.sh` is a generic Slurm wrapper; replace `<PARTITION>`, `<ACCOUNT>` and the resource lines with your
cluster's settings (or use your own scheduler).

Everything is written under `SFA_HOME` (default: the repository root): `data/`, `runs/` (checkpoints, schedules, logs),
`results/` (evaluation summaries, per-example predictions, tables). Models and datasets are downloaded from the Hugging Face Hub
at the revisions pinned in `configs/models.json`; none of them is gated.

**Hardware.** Every job uses a single GPU; our runs used one 141 GB GPU (NVIDIA H200) per job. Peak memory: full-parameter 1.7B ≈ 95 GB; adapter runs
≈ 96 GB (8B), ≈ 118 GB (14B); 32B uses activation checkpointing (`--grad_ckpt`) and a 2,048-token microbatch cap. The
microbatch caps in `configs/models.json` only bound memory: the loss is a token mean over the full 32-sequence batch, so they do
not change the objective. A full reproduction stores a few TB of checkpoints (each full-parameter checkpoint is ≈ 6.9 GB).

## 1. Data

```bash
bash scripts/prepare_data.sh        # CPU only, ~30-60 min; downloads everything below
```

All datasets are public and are retrieved automatically at pinned versions (`configs/models.json`); nothing has to be
gathered by hand.

| task | public source | split used |
|---|---|---|
| Math (train) | MetaMathQA, Hugging Face `meta-math/MetaMathQA` | rows derived from a dev question, exact matches of a GSM8K test question, and rows sharing a 13-gram with GSM8K test are removed |
| Math (dev / test) | GSM8K, Hugging Face `openai/gsm8k` (main) | dev = 500 seeded GSM8K-train questions; test = GSM8K test (1,319) |
| Law: CaseHOLD | LexGLUE `case_hold`, Hugging Face `coastalcph/lex_glue` | official train / validation (dev) / test |
| Law: ToS | UNFAIR-ToS sentences (Lippi et al., 2019), Hugging Face `LawInformedAI/claudette_tos` | first 75% train, rest test; dev = label-stratified 5% of train |
| Law: Overruling | Overruling (Zheng et al., 2021), Hugging Face `LawInformedAI/overruling` | first 75% train, rest test; dev = label-stratified 5% of train |
| Continual learning | DBpedia, Amazon, Yahoo, AG News (1,000 training examples per task), as released in the public `Recurrent_KIF` GitHub repository | released train / dev / test files (sha256-checked); dev excludes examples that also appear in test |

Each Law example is a multiple-choice record with a fixed task instruction (`sfa/data/prepare_math_law.py`); the Law score is the
unweighted mean of the three task accuracies. The preparation then tokenizes everything with the Qwen3 tokenizer and builds all
batch schedules (identical across methods for a given seed).

## 2. Full-parameter Math → Law (Qwen3-1.7B)

```bash
bash scripts/reproduce_full_param.sh
```

which runs

| step | command | what | approx. cost (one 141 GB GPU) |
|---|---|---|---|
| selection | `python -m sfa.run_full_param --set selection_dev` | seed 17, development split: SFA β_end ∈ {.25,.5,.75}, Lookahead β ∈ {.25,.5,.75} (both merging at updates 185/370/555/740), FT lr ∈ {1e-5, 5e-6, 2.5e-6}, plus endpoint-interpolation and EMA grids | ~4 GPU-h |
| main | `python -m sfa.run_full_param --set main_test` | seeds 17/42/123: the selected configurations and baselines, dev + test | ~8 GPU-h |
| replication | `python -m sfa.run_full_param --set replication_test` | seeds 7/2026: SFA β_end .5, Lookahead β .5, FT lr 5e-6 | ~3 GPU-h |
| tables | `python -m sfa.analysis.full_param` | `results/tables/full_param_{selection,test,paired}.md` | CPU |

Each seed first trains a Math start (MetaMathQA, 10M tokens), which is shared by all Law runs of that seed as initialization
and as the SFA anchor / Lookahead slow weights; all Law runs of a seed see identical examples in identical order (10M tokens).
**Selection rule** (`configs/full_param.json`): `L_ref` = development Law accuracy of FT at lr 1e-5 (seed 17); for each method,
among candidates with development Law ≥ `L_ref − 0.01` (R_mid; also 0.005 / 0.02), pick the highest development GSM8K accuracy.
`python -m sfa.analysis.full_param` applies the rule; the `main_test` and `replication_test` sets in `configs/full_param.json` list
the configurations that were run on the remaining seeds and on test.

## 3. Adapter-level continual learning

```bash
bash scripts/reproduce_adapter.sh Qwen/Qwen3-14B                       # 5 methods x 3 seeds, order 1 (~15 GPU-h)
ARMS="sfa_be25 sfa_be75 look_b25 look_b75" SEEDS=17 SPLITS=dev bash scripts/reproduce_adapter.sh Qwen/Qwen3-14B
SEEDS="42 123" ARMS="look_b75" bash scripts/reproduce_adapter.sh Qwen/Qwen3-14B   # replicate the selected coefficient
ORDER=4 bash scripts/reproduce_adapter.sh Qwen/Qwen3-14B               # reversed task order
bash scripts/reproduce_adapter.sh Qwen/Qwen3-1.7B                      # likewise Qwen3-8B, Qwen3-32B
ARMS=sfa_replay bash scripts/reproduce_adapter.sh Qwen/Qwen3-1.7B      # SFA-R
python -m sfa.analysis.adapter                                         # results/tables/adapter_*.md
```

One stream = 4 stages (DBpedia → Amazon → Yahoo → AG News for order 1), each 310 updates of the released AIMMerging schedule
(1,000 examples per task, 10 epochs, batch 32), one LoRA adapter (r 8, α 32, q/v projections) carried across stages, AdamW
lr 3e-4. After every stage the adapter is merged into dense weights and all tasks seen so far are evaluated (dev and test).
Replay and AIMMerging use the released 2% memory buffer (20 / 40 / 60 examples). Approximate cost per stream on one 141 GB GPU:
1.7B ≈ 25 min, 8B ≈ 40 min, 14B ≈ 1 h, 32B ≈ 2 h. Method arguments: `configs/adapter.json`.

**Coefficient selection** (`configs/adapter.json`): for each method, among β ∈ {.25, .5, .75} with seed-17 development
acquisition ≥ that of sequential LoRA − 0.01, pick the highest development accuracy on earlier tasks
(`python -m sfa.analysis.adapter`); the selected coefficients are then replicated unchanged on seeds 42 and 123.

**Timing** (`bash scripts/reproduce_timing.sh Qwen/Qwen3-14B`): one stage (Amazon, seed 17), one GPU, two repetitions in
reversed order, identical start adapter, schedule and microbatch cap for all methods; `python -m sfa.analysis.timing` reports the
mean training-loop time per method and its ratio to SFA. This measures one training stage only, not end-to-end runtime.

## Reproducibility notes

* Seed 17 is the development seed: all selections use its development split only; other seeds and test splits are evaluated
  only with frozen configurations.
* Data preparation is deterministic (fixed seeds, pinned revisions, sha256-checked downloads): it rebuilds the exact files used
  for the reported runs. Schedules are identical across methods for a given seed.
* Training uses fp32 master weights, bf16 autocast, TF32 off. GPU kernels are not guaranteed to be bitwise deterministic across
  hardware or library versions, and vLLM greedy decoding can differ slightly with batch composition, so repeated runs can
  differ by small amounts.
* `python -m sfa.evaluate --limit N` and `--max_steps N` (training) run quick smoke tests.

## Third-party code and data

* **AIMMerging.** `sfa/aim.py` (with the `aim` and memory-buffer paths of `sfa/train_lora.py` and
  `sfa/data/adapter_schedules.py`) is our reimplementation of the released AIMMerging code
  ([WoodScene/AimMerging](https://github.com/WoodScene/AimMerging), commit `90733c3`) for use as a baseline under our shared
  adapter protocol; each rule cites the released file and line it reproduces. AIMMerging is the work of Feng et al. (EMNLP 2025);
  please cite their paper if you use this baseline. Results obtained with this port are not results of the original
  implementation.
* **Continual-learning data.** The DBpedia, Amazon, Yahoo Answers and AG News files are downloaded at run time from the public
  [WoodScene/Recurrent_KIF](https://github.com/WoodScene/Recurrent_KIF) repository (Feng et al., ACL 2025) at a pinned commit;
  they are not redistributed here. The underlying datasets are those of Zhang et al. (2015).
* **Models and other datasets** are downloaded from the Hugging Face Hub at the revisions pinned in `configs/models.json`
  (Qwen3: Apache-2.0; MetaMathQA and GSM8K: MIT; LexGLUE: CC-BY-4.0) and are not redistributed here. Each is subject to its
  own license and terms.

## Citation

```bibtex
@article{kleiman2026forget,
  title   = {Forget About It: Mitigating Forgetting Without Replay or Auxiliary Objectives},
  author  = {Kleiman, Anat and Dziugaite, Gintare Karolina and Frankle, Jonathan and Kakade, Sham and Paul, Mansheej},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). Third-party models, datasets and the AIMMerging
method remain under their own terms (see **Third-party code and data**).
