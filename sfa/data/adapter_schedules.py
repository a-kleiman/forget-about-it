"""Adapter-level continual-learning streams: the released AIMMerging current-task schedule + the released memory buffer.

Current-task schedule (released Standard-CL script: per-device batch 8 x gradient accumulation 4 over 1,000 examples, 10 epochs):
  HF Trainer gives max_steps = 10 * ((1000 // 8) // 4) = 310 updates; its sample counter is not reset per epoch, so every update
  holds exactly 32 examples, updates straddle epoch boundaries, and the last 80 examples of epoch 10 are never seen.
  Equivalent construction used here: concatenate 10 per-epoch permutations (numpy rng [seed, order, stage, epoch, 31337]), cut them
  into consecutive 32-example batches and keep the first 310. Identical for every method (matched current-task exposure).
Memory buffer (sfa.aim.build_memory): the released load_memory_buffer (2% per previous task, class-stratified, random.seed(42)),
  seed-independent: 0 / 20 / 40 / 60 examples at stages 1-4.

  python -m sfa.data.adapter_schedules --orders 1 4 --seeds 17 42 123
"""
import argparse, hashlib, json

import numpy as np

from sfa.formatting import ROOT
from sfa.data.prepare_cl import ORDERS, SRC_MUL
from sfa.aim import build_memory

N_EPOCHS, BATCH, MICRO, GA = 10, 32, 8, 4


def released_batches(n, seed, order_id, stage, epochs=N_EPOCHS):
    perms = [np.random.default_rng([seed, order_id, stage, e, 31337]).permutation(n) for e in range(epochs)]
    steps_per_epoch = (n // MICRO) // GA          # 125 // 4 = 31  (n % MICRO == 0 for all 4 tasks)
    assert n % MICRO == 0
    stream = np.concatenate(perms)
    T = epochs * steps_per_epoch
    return [stream[i * BATCH:(i + 1) * BATCH] for i in range(T)]


def build(order_id, seed):
    order = ORDERS[order_id]
    d = ROOT / "runs/schedules"; d.mkdir(parents=True, exist_ok=True)
    names = []
    for si, task in enumerate(order):
        z = np.load(ROOT / f"data/tokenized/cl_{task}/train.npz")
        lens = np.diff(z["off"]); npr = z["npr"]
        bl = released_batches(len(lens), seed, order_id, si)
        T = len(bl)
        B = np.full((T, BATCH), -1, dtype=np.int64)
        for k, b in enumerate(bl):
            B[k, :len(b)] = si * SRC_MUL + b
        tok = int(sum(lens[b].sum() for b in bl)); sup = int(sum((lens[b] - npr[b]).sum() for b in bl))
        mem = build_memory(ROOT, order, si)
        tag = f"lora_o{order_id}_s{seed}_st{si}_{task}_ep10"
        np.savez(d / f"{tag}.npz", batches=B, memory=np.array([m["src"] * SRC_MUL + m["idx"] for m in mem], dtype=np.int64))
        mem_len = [int(np.diff(np.load(ROOT / f"data/tokenized/cl_{m['task']}/train.npz")["off"])[m["idx"]]) for m in mem]
        stats = {"order": order, "order_id": order_id, "stage": si, "task": task, "seed": seed, "mode": "released_ga4", "T": T,
                 "global_batch": BATCH, "epochs": N_EPOCHS, "examples_seen": int(sum(len(b) for b in bl)),
                 "unique_examples_seen": int(len(np.unique(np.concatenate(bl)))), "last_batch_size": int(len(bl[-1])),
                 "input_tokens": tok, "supervised_tokens": sup, "sources": [f"cl_{x}" for x in order],
                 "memory": mem, "memory_size": len(mem), "memory_input_tokens": int(sum(mem_len)),
                 "memory_task_counts": {t: sum(1 for m in mem if m["task"] == t) for t in order[:si]}}
        stats["sha256"] = hashlib.sha256(open(d / f"{tag}.npz", "rb").read()).hexdigest()
        json.dump(stats, open(d / f"{tag}.json", "w"), indent=1)
        names.append(f"runs/schedules/{tag}")
        print(tag, {k: stats[k] for k in ["T", "examples_seen", "unique_examples_seen", "input_tokens", "memory_size", "memory_task_counts"]})
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 123])
    a = ap.parse_args()
    for o in a.orders:
        for s in a.seeds:
            build(o, s)
