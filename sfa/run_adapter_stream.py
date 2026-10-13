"""Run one adapter-level continual-learning stream: 4 stages of sfa.train_lora chained through one persistent adapter,
exporting merged dense weights after every stage and evaluating ALL tasks seen so far (dev and test). Resumable: completed
stages (runs/<id>/summary.json) and completed evaluations (results/evals/<tag>__<split>.json) are skipped.

  SFA_MODEL=Qwen/Qwen3-14B python -m sfa.run_adapter_stream --order 1 --seed 17 --arm sfa
  SFA_MODEL=Qwen/Qwen3-32B python -m sfa.run_adapter_stream --order 1 --seed 17 --arm aim --grad_ckpt
Arms: configs/adapter.json["arms"]. Run ids: <prefix>_o<order>_s<seed>_<arm>_st<stage> with <prefix> from configs/models.json.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

from sfa.formatting import MODEL_ID, MODELS, ROOT

CFG = json.load(open(Path(__file__).resolve().parents[1] / "configs/adapter.json"))


def done_eval(tag, split):
    f = ROOT / f"results/evals/{tag}__{split}.json"
    return f.exists() and json.load(open(f)).get("complete", False)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", "-m"] + cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, required=True, choices=[int(k) for k in CFG["orders"]])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=sorted(CFG["arms"]))
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--grad_ckpt", action="store_true", help="activation checkpointing (used for Qwen3-32B)")
    ap.add_argument("--keep_dense", action="store_true", help="keep merged dense weights after evaluation (large)")
    a = ap.parse_args()
    m = MODELS["models"][MODEL_ID]
    prefix, mb = m["run_prefix"], m["adapter_microbatch_tokens"]
    if m.get("activation_checkpointing") and not a.grad_ckpt:
        print(f"note: the reported {MODEL_ID} runs used --grad_ckpt", flush=True)
    order = CFG["orders"][str(a.order)]
    init, seen = "none", []
    for i, task in enumerate(order):
        sched = f"runs/schedules/lora_o{a.order}_s{a.seed}_st{i}_{task}_ep10"
        rid = f"{prefix}_o{a.order}_s{a.seed}_{a.arm}_st{i}"
        if not (ROOT / f"runs/{rid}/summary.json").exists():
            run(["sfa.train_lora", "--run_id", rid, "--schedule", sched, "--init_adapter", init, "--seed", str(a.seed),
                 *CFG["arms"][a.arm]["args"], "--max_tokens_mb", str(mb)] + (["--grad_ckpt"] if a.grad_ckpt else []))
        seen.append(f"cl_{task}")
        dense = ROOT / f"runs/{rid}/dense"
        for split in a.splits:
            if not done_eval(rid, split):
                if not dense.exists():
                    run(["sfa.export_lora", "--adapter", str(ROOT / f"runs/{rid}/adapter"), "--out", str(dense)])
                run(["sfa.evaluate", "--model", str(dense), "--tag", rid, "--split", split, "--tasks", *seen])
        if not a.keep_dense and all(done_eval(rid, s) for s in a.splits):
            shutil.rmtree(dense, ignore_errors=True)
        init = str(ROOT / f"runs/{rid}/adapter")
    print("stream done", flush=True)


if __name__ == "__main__":
    main()
