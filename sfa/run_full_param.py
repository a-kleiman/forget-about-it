"""Run the full-parameter Math->Law experiments for one seed (resumable; completed runs/evaluations are skipped).

  python -m sfa.run_full_param --set selection_dev                 # seed-17 development candidates (dev only)
  python -m sfa.run_full_param --set main_test                     # seeds 17/42/123, reported configurations (dev + test)
  python -m sfa.run_full_param --set replication_test              # seeds 7/2026, the three R_mid configurations
  python -m sfa.run_full_param --seed 42 --law_runs sfamatch_e50_p25 --splits dev test   # any subset
Each seed first trains the Math start (math_ft_s<seed>), shared by all Law runs of that seed (initialization and SFA anchor /
Lookahead slow weights); all Law runs of a seed use the same Law schedule (identical examples in identical order).
Run definitions: configs/full_param.json.
"""
import argparse, json, subprocess, sys
from pathlib import Path

from sfa.formatting import ROOT

CFG = json.load(open(Path(__file__).resolve().parents[1] / "configs/full_param.json"))


def done_eval(tag, split):
    f = ROOT / f"results/evals/{tag}__{split}.json"
    return f.exists() and json.load(open(f)).get("complete", False)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", "-m"] + cmd, check=True)


def evaluate(model, tag, splits, tasks=("math", "law")):
    for sp in splits:
        if not done_eval(tag, sp):
            run(["sfa.evaluate", "--model", model, "--tag", tag, "--split", sp, "--tasks", *tasks])


def one_seed(seed, law_runs, eval_only, splits):
    for t in ("math", "law"):
        if not (ROOT / f"runs/schedules/{t}_s{seed}_b10M.npz").exists():
            run(["sfa.data.tokenize", "--schedule", t, "--seeds", str(seed)])
    ms = CFG["math_start"]; mid = ms["run_id"].format(seed=seed)
    if not (ROOT / f"runs/{mid}/summary.json").exists():
        run(["sfa.train_full", "--run_id", mid, "--seed", str(seed)] + [x.format(seed=seed) for x in ms["args"]])
    evaluate(f"runs/{mid}/final", mid, splits)
    for name in law_runs:
        rid = f"m2l_s{seed}_{name}"
        if not (ROOT / f"runs/{rid}/summary.json").exists():
            run(["sfa.train_full", "--run_id", rid, "--task", "law", "--init", f"runs/{mid}/final",
                 "--schedule", f"runs/schedules/law_s{seed}_b10M", "--seed", str(seed), *CFG["law_runs"][name]["args"],
                 "--heldout_tasks", "math", "law"])
        evaluate(f"runs/{rid}/final", rid, splits)
    for name in eval_only:
        if "ft" not in law_runs and not (ROOT / f"runs/m2l_s{seed}_ft/summary.json").exists():
            raise SystemExit(f"{name} needs the FT lr 1e-5 run (law run 'ft') for seed {seed}")
        evaluate(CFG["eval_only"][name]["source"].format(seed=seed), f"m2l_s{seed}_{name}", splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(CFG["sets"]))
    ap.add_argument("--seed", type=int, nargs="*")
    ap.add_argument("--law_runs", nargs="*")
    ap.add_argument("--eval_only", nargs="*")
    ap.add_argument("--splits", nargs="+")
    a = ap.parse_args()
    st = CFG["sets"][a.set] if a.set else {"seeds": [], "splits": ["dev"], "law_runs": [], "eval_only": []}
    seeds = a.seed or st["seeds"]
    law_runs = a.law_runs if a.law_runs is not None else st["law_runs"]
    eval_only = a.eval_only if a.eval_only is not None else st["eval_only"]
    splits = a.splits or st["splits"]
    assert seeds, "give --set or --seed"
    for s in seeds:
        one_seed(s, law_runs, eval_only, splits)


if __name__ == "__main__":
    main()
