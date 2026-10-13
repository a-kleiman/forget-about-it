"""Single-GPU full-parameter SFT for one task with FT / SFA / Lookahead / eval-only EMA shadows.

One continuous process per task: optimizer moments are retained across within-task merges;
optimizer + LR schedule start fresh at the task boundary (i.e. at the start of this process).

SFA (beta = ANCHOR coefficient, anchor a = immutable copy of the task-start weights):
    k = max(1, floor(p*T + 0.5)); merge_steps = sorted({k,2k,...<=T} U {T})
    after optimizer.step() at each listed update u:  w <- (1-beta) w + beta a
Lookahead: same events, slow s (init = task start):  s <- (1-beta) w + beta s ; w <- s
EMA (eval only): shadow <- d*shadow + (1-d)*w every update; never copied into the learner.
Precision: fp32 master weights, bf16 autocast; AdamW(0.9, 0.999, eps 1e-8, no weight decay), clip 1.0, linear warmup
(ceil(warmup * T) updates) then cosine decay; loss = token-mean over the supervised (assistant) tokens of each 32-sequence batch.

  python -m sfa.train_full --run_id <id> --task law --init runs/math_ft_s17/final --schedule runs/schedules/law_s17_b10M \
      --seed 17 --method sfa --lr 1e-5 --p 0.25 --beta_end 0.5 --heldout_tasks math law
"""
import argparse, json, math, socket, subprocess, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sfa.formatting import MODEL_ID, MODELS, ROOT, model_revision


def merge_schedule(p, T):
    k = max(1, math.floor(p * T + 0.5))
    return sorted(set(range(k, T + 1, k)) | {T})


def matched_beta(beta_end, K):
    return 1.0 - (1.0 - beta_end) ** (1.0 / K)


def lr_lambda_factory(T, warmup_frac):
    W = max(1, math.ceil(warmup_frac * T))

    def f(step):  # step = number of completed updates; value used for update step+1
        if step < W:
            return (step + 1) / W
        prog = (step - W) / max(1, T - W)  # 0 at first post-warmup update; last update gets lr > 0
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f, W


@torch.no_grad()
def interp_(params, others, beta):
    """params <- (1-beta)*params + beta*others, computed in fp32, copied to storage dtype."""
    for p, a in zip(params, others):
        if p.dtype == torch.float32:
            p.mul_(1 - beta).add_(a, alpha=beta)
        else:
            p.copy_(p.float().mul_(1 - beta).add_(a.float(), alpha=beta))


def make_microbatches(lens, max_tokens):
    """Group global-batch example positions (sorted by length) so padded tokens <= max_tokens."""
    order = np.argsort(-lens, kind="stable")
    mbs, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lens[cur[0]] > max_tokens:
            mbs.append(cur); cur = []
        cur.append(int(i))
    if cur:
        mbs.append(cur)
    return mbs


class Data:
    """Single task, or multiple sources (CL replay): global id = src * 10_000_000 + idx."""
    def __init__(self, task, sources=None):
        self.src = []
        for t in (sources or [task]):
            z = np.load(ROOT / f"data/tokenized/{t}/train.npz")
            self.src.append((z["ids"], z["off"], z["npr"]))

    def example(self, g):
        s, i = divmod(int(g), 10_000_000)
        ids, off, npr = self.src[s]
        return ids[off[i]:off[i + 1]], int(npr[i])


def collate(examples, pad_id, device):
    L = max(len(e[0]) for e in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    for j, (t, npr) in enumerate(examples):
        t = torch.as_tensor(t, dtype=torch.long)
        ids[j, :len(t)] = t; att[j, :len(t)] = 1; lab[j, npr:len(t)] = t[npr:]
    return ids.to(device, non_blocking=True), att.to(device, non_blocking=True), lab.to(device, non_blocking=True)


def sum_loss(model, ids, att, lab):
    """Sum of next-token CE over supervised positions; logits only where supervised."""
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    m = tgt != -100
    hs = h[:, :-1][m]
    logits = model.lm_head(hs).float()
    return F.cross_entropy(logits, tgt[m], reduction="sum"), int(m.sum())


@torch.no_grad()
def heldout_loss(model, exs, pad_id, device, max_tokens):
    model.eval()
    tot, n = 0.0, 0
    lens = np.array([len(e[0]) for e in exs])
    for mb in make_microbatches(lens, max_tokens):
        ids, att, lab = collate([exs[i] for i in mb], pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, c = sum_loss(model, ids, att, lab)
        tot += l.item(); n += c
    model.train()
    return tot / max(1, n)


def load_heldout(task, tok, n=128):
    """Teacher-forced held-out examples from the task's DEV split (for curves only)."""
    from sfa.formatting import encode_pair, law_user
    import re
    if task == "law":
        rows = []
        for s in ["casehold", "tos", "overruling"]:
            rs = [json.loads(l) for l in open(ROOT / f"data/processed/law/{s}_dev.jsonl")]
            rows += rs[: n // 3]
        pairs = [(law_user(r), r["choices"][r["gold"]]) for r in rows]
    elif task == "math":
        from datasets import load_dataset  # worked GSM8K-train solutions of the dev questions
        rows = [json.loads(l) for l in open(ROOT / "data/processed/math/dev.jsonl")][:n]
        g = load_dataset("openai/gsm8k", "main", revision=MODELS["datasets"]["openai/gsm8k"], split="train")
        pairs = []
        for r in rows:
            sol = g[int(r["uid"].split("_")[-1])]["answer"]
            body, ans = sol.split("####")
            body = re.sub(r"<<[^>]*>>", "", body).strip()
            pairs.append((r["question"], f"{body}\nThe answer is: {ans.strip()}"))
    else:
        raise ValueError(task)
    out = []
    for u, a in pairs:
        e = encode_pair(tok, u, a)
        if e is not None:
            out.append((np.array(e[0]), e[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--init", required=True, help="HF model dir or 'base'")
    ap.add_argument("--schedule", required=True, help="runs/schedules/<name>")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", default="ft", choices=["ft", "sfa", "lookahead"])
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None, help="anchor coefficient per event")
    ap.add_argument("--beta_end", type=float, default=None, help="matched strength: beta = 1-(1-beta_end)^(1/K)")
    ap.add_argument("--ema", type=float, nargs="*", default=[])
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max_tokens_mb", type=int, default=16384)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke tests only: truncate T (schedule/merges use truncated T)")
    ap.add_argument("--heldout_tasks", nargs="*", default=[])
    ap.add_argument("--n_heldout_evals", type=int, default=10)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--save", default="final", choices=["final", "none"])
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = ROOT / "runs" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.jsonl", "w")  # no resume: a (re)started run always starts from scratch
    def log(d):
        d["time"] = time.time(); logf.write(json.dumps(d) + "\n"); logf.flush(); print(json.dumps(d), flush=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision())
    init = MODEL_ID if a.init == "base" else a.init
    kw = {"revision": model_revision()} if a.init == "base" else {}
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(init, dtype=torch.float32, attn_implementation="sdpa", **kw).to(dev)
    model.config.use_cache = False
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad and p.is_floating_point()]  # dedups tied weights
    tied = model.config.tie_word_embeddings and model.lm_head.weight is model.model.embed_tokens.weight

    B = np.load(ROOT / f"{a.schedule}.npz")["batches"]
    sched_stats = json.load(open(ROOT / f"{a.schedule}.json"))
    T = len(B) if a.max_steps is None else min(a.max_steps, len(B))
    data = Data(a.task, sched_stats.get("sources"))

    merges, beta = [], None
    if a.method in ("sfa", "lookahead"):
        merges = merge_schedule(a.p, T)
        beta = a.beta if a.beta_end is None else matched_beta(a.beta_end, len(merges))
    merge_set = set(merges)
    need_anchor = a.method in ("sfa", "lookahead")
    anchor = [p.detach().clone() for p in params] if need_anchor else None  # immutable (SFA) or slow (Lookahead)
    shadows = {d: [p.detach().clone() for p in params] for d in a.ema}

    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
    fn, W = lr_lambda_factory(T, a.warmup)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, fn)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")

    held = {t: load_heldout(t, tok) for t in a.heldout_tasks if t in ("math", "law")}
    eval_at = set(np.unique(np.linspace(0, T, a.n_heldout_evals + 1).round().astype(int)).tolist()) if held else set()
    def do_held(u):
        if u in eval_at:
            torch.cuda.synchronize(); t0 = time.time()
            r = {f"heldout_{t}": heldout_loss(model, e, pad_id, dev, a.max_tokens_mb) for t, e in held.items()}
            torch.cuda.synchronize(); r["heldout_eval_sec"] = time.time() - t0
            log({"update": u, **r})

    persist_bytes = {"anchor_or_slow": sum(p.numel() * p.element_size() for p in anchor) if anchor else 0,
                     "ema_shadows_each": sum(p.numel() * p.element_size() for p in params) if shadows else 0,
                     "n_ema_shadows": len(shadows)}
    cfg = {**vars(a), "T": T, "T_schedule": len(B), "merge_steps": merges, "K": len(merges), "beta_effective": beta,
           "warmup_steps": W, "tied_embeddings": bool(tied), "n_param_tensors": len(params),
           "n_params": sum(p.numel() for p in params), "model_id": MODEL_ID, "model_revision": model_revision(),
           "schedule_stats": sched_stats, "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "persist_bytes": persist_bytes,
           "load_sec": time.time() - t_load,
           "code_commit": subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                         capture_output=True, text=True).stdout.strip()}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    log({"event": "start", "T": T, "K": len(merges), "beta": beta, "merges_head": merges[:5], "merges_tail": merges[-3:]})

    do_held(0)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t_train0 = time.time(); merge_sec = 0.0; held_sec = 0.0; n_merges_done = 0; skipped = []
    ema_ev = []  # (start, end) cuda events, timed without per-step sync
    tok_in = tok_sup = 0
    win_loss = win_tok = 0.0
    for step in range(T):
        u = step + 1
        bidx = B[step]
        exs = [data.example(int(i)) for i in bidx]
        lens = np.array([len(e[0]) for e in exs])
        n_sup = int(sum(len(e[0]) - e[1] for e in exs))
        assert n_sup > 0
        tok_in += int(lens.sum()); tok_sup += n_sup
        loss_acc = torch.zeros((), device=dev)
        for mb in make_microbatches(lens, a.max_tokens_mb):
            ids, att, lab = collate([exs[i] for i in mb], pad_id, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l, _ = sum_loss(model, ids, att, lab)
            (l / n_sup).backward()
            loss_acc += l.detach()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            skipped.append(u); opt.zero_grad(set_to_none=True); sch.step()
            log({"update": u, "skipped": True}); continue
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if shadows:
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
            with torch.no_grad():
                for d, sh in shadows.items():
                    torch._foreach_lerp_(sh, params, 1 - d)  # sh = d*sh + (1-d)*w
            e1.record(); ema_ev.append((e0, e1))
        if u in merge_set:
            torch.cuda.synchronize(); t0 = time.time()
            if a.method == "sfa":
                interp_(params, anchor, beta)
            else:  # lookahead
                with torch.no_grad():
                    for s, p in zip(anchor, params):
                        s.mul_(beta).add_(p, alpha=1 - beta); p.copy_(s)
            torch.cuda.synchronize(); merge_sec += time.time() - t0; n_merges_done += 1
            log({"update": u, "merge": n_merges_done, "beta": beta})
        win_loss += loss_acc.item(); win_tok += n_sup  # one sync per step for logging; negligible vs step
        if u % 10 == 0 or u == T:
            log({"update": u, "loss": win_loss / win_tok, "lr": sch.get_last_lr()[0], "grad_norm": float(gn),
                 "tok_in": tok_in, "tok_sup": tok_sup, "elapsed": time.time() - t_train0})
            win_loss = win_tok = 0.0
        if eval_at and u in eval_at:
            th = time.time(); do_held(u); held_sec += time.time() - th
    torch.cuda.synchronize(); train_sec = time.time() - t_train0 - held_sec
    ema_sec = sum(e0.elapsed_time(e1) for e0, e1 in ema_ev) / 1000.0

    summary = {"run_id": a.run_id, "status": "completed", "T": T, "updates_applied": T - len(skipped), "skipped": skipped,
               "merges_applied": n_merges_done, "K": len(merges), "input_tokens": tok_in, "supervised_tokens": tok_sup,
               "train_sec_excl_heldout": train_sec, "merge_sec": merge_sec, "ema_update_sec": ema_sec, "heldout_sec": held_sec,
               "tokens_per_sec": tok_in / train_sec, "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30, "persist_bytes": persist_bytes, "n_gpus": 1}
    if a.save == "final":
        t0 = time.time()
        model.save_pretrained(out / "final", safe_serialization=True); tok.save_pretrained(out / "final")
        for d, sh in shadows.items():
            with torch.no_grad():
                bak = [p.detach().clone() for p in params]
                for p, s in zip(params, sh): p.copy_(s)
                model.save_pretrained(out / f"ema{d}", safe_serialization=True); tok.save_pretrained(out / f"ema{d}")
                for p, b in zip(params, bak): p.copy_(b)
        summary["save_sec"] = time.time() - t0
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log({"event": "done", **summary})


if __name__ == "__main__":
    main()
