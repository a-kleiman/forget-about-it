"""Tokenize processed training sets and build deterministic full-parameter batch schedules.

data/tokenized/<task>/train.npz : ids (int32 flat), off (int64), npr (prompt lengths), uid index (+ train_stats.json)
runs/schedules/<task>_s<seed>_b<budget>M.npz : batches [T, 32] of example indices, filled from seeded permutations until the
    cumulative non-padding input tokens reach the budget (+ stats json with the schedule sha256)

  python -m sfa.data.tokenize --tokenize math law                       # full-parameter tasks
  python -m sfa.data.tokenize --tokenize cl_dbpedia cl_amazon cl_yahoo cl_agnews
  python -m sfa.data.tokenize --schedule math law --seeds 17 42 123     # 10M-token schedules
Tokenization uses the Qwen3-1.7B tokenizer (shared by all Qwen3 sizes used here).
"""
import argparse, hashlib, json, os
from multiprocessing import Pool

import numpy as np

from sfa.formatting import MAX_LEN, MODELS, ROOT, encode_pair, train_pair

TOKENIZER = MODELS["default_model"]
_tok = None


def _init(rev):
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=rev)


def _enc(args):
    task, r = args
    u, a = train_pair(task, r)
    out = _tok and encode_pair(_tok, u, a)
    return out


def tokenize_task(task, rev, nproc=int(os.environ.get("SFA_TOKENIZE_PROCS", 16))):
    src = ROOT / f"data/processed/{task}/train.jsonl"
    rows = [json.loads(l) for l in open(src)]
    with Pool(nproc, initializer=_init, initargs=(rev,)) as pool:
        enc = pool.map(_enc, [(task, r) for r in rows], chunksize=256)
    keep, ids, off, npr, stats = [], [], [0], [], {"n_rows": len(rows), "boundary_mismatch": 0, "over_max_len": 0}
    for r, e in zip(rows, enc):
        if e is None:
            stats["boundary_mismatch"] += 1; continue
        t, p = e
        if len(t) > MAX_LEN:
            stats["over_max_len"] += 1; continue
        keep.append(r["uid"]); ids.extend(t); off.append(off[-1] + len(t)); npr.append(p)
    lens = np.diff(off)
    stats.update(n_kept=len(keep), mean_len=float(lens.mean()), mean_sup=float((lens - np.array(npr)).mean()),
                 p99_len=float(np.percentile(lens, 99)))
    d = ROOT / f"data/tokenized/{task}"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "train.npz", ids=np.array(ids, dtype=np.int32), off=np.array(off, dtype=np.int64),
             npr=np.array(npr, dtype=np.int32), uid=np.array(keep))
    stats["sha256"] = hashlib.sha256(open(d / "train.npz", "rb").read()).hexdigest()
    stats["source_sha256"] = hashlib.sha256(open(src, "rb").read()).hexdigest()
    stats["tokenizer_revision"] = rev
    json.dump(stats, open(d / "train_stats.json", "w"), indent=1)
    print(task, stats)


def build_schedule(task, seed, budget, batch=32):
    z = np.load(ROOT / f"data/tokenized/{task}/train.npz")
    lens = np.diff(z["off"]); sup = lens - z["npr"]
    task_salt = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([seed, task_salt])
    batches, tok, stok, epoch, buf = [], 0, 0, 0, []
    while tok < budget:
        perm = rng.permutation(len(lens)); epoch += 1
        for i in perm:
            buf.append(i)
            if len(buf) == batch:
                b = np.array(buf); buf = []
                batches.append(b); tok += int(lens[b].sum()); stok += int(sup[b].sum())
                if tok >= budget:
                    break
    B = np.stack(batches)
    d = ROOT / "runs/schedules"; d.mkdir(parents=True, exist_ok=True)
    name = f"{task}_s{seed}_b{budget // 10**6}M"
    stats = {"task": task, "seed": seed, "budget_input_tokens": budget, "global_batch": batch, "T": len(B),
             "input_tokens": tok, "supervised_tokens": stok, "epochs_started": epoch,
             "unique_examples": int(len(np.unique(B))), "n_pool": int(len(lens))}
    np.savez(d / f"{name}.npz", batches=B)
    stats["sha256"] = hashlib.sha256(open(d / f"{name}.npz", "rb").read()).hexdigest()
    json.dump(stats, open(d / f"{name}.json", "w"), indent=1)
    print(name, stats)
    return name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenize", nargs="*", default=[])
    ap.add_argument("--schedule", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[17])
    ap.add_argument("--budget", type=int, default=10_000_000)
    a = ap.parse_args()
    rev = MODELS["models"][TOKENIZER]["revision"]
    for t in a.tokenize:
        tokenize_task(t, rev)
    for t in a.schedule:
        for s in a.seeds:
            build_schedule(t, s, a.budget)
