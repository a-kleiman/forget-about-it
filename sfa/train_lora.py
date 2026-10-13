"""One stage of the adapter-level (shared-LoRA) continual-learning comparison, single GPU (configs/adapter.json).

Methods:
  seqlora     sequential LoRA
  replay      LoRA + replay: each update = 32 current-task examples + 8 memory examples drawn uniformly WITH replacement
              from the stage's released memory buffer; token-mean loss over the combined 40 sequences
  sfa         adapter-level SFA: at merge events A <- (1-b)A + b*A0, B <- (1-b)B + b*B0 (A0/B0 = adapter at stage start)
  lookahead   adapter-level Lookahead: slow s (init = stage-start adapter): s <- b*s + (1-b)*w ; w <- s
  aim         AIMMerging, port of the released controller (sfa/aim.py); importance EMAs kept (cost accounting only)
  sfa_replay  SFA-R: the replay batches of `replay` combined with the merges of `sfa`

Shared by all methods: ONE persistent LoRA adapter (r=8, alpha=32, dropout=0.05, q_proj/v_proj) carried across stages;
frozen fp32 base, fp32 adapter master weights, bf16 autocast; released current-task schedule (sfa/data/adapter_schedules.py);
AdamW(0.9,0.999,1e-8,wd 0) lr 3e-4, HF linear schedule with 50 warmup steps (lambda(0)=0, as HF), clip 1.0, fresh
optimizer + scheduler per stage; token-mean loss over each global batch; FINAL weights passed on and evaluated.
Model: SFA_MODEL (default Qwen/Qwen3-1.7B; see configs/models.json). --grad_ckpt (activation checkpointing; recomputation only)
was used for Qwen3-32B.
"""
import argparse, hashlib, json, socket, subprocess, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sfa.formatting import MODEL_ID, ROOT, encode_pair, model_revision
from sfa.train_full import make_microbatches, collate, merge_schedule, matched_beta, interp_, Data
from sfa.aim import AimController, ImportanceEMA, fuse_, GA_REF

LORA = dict(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")


def hf_linear_lambda(T, warmup):
    """transformers.get_linear_schedule_with_warmup (HF 4.28.1 optimization.py): value used for the (s+1)-th update."""
    def f(s):
        if s < warmup:
            return float(s) / float(max(1, warmup))
        return max(0.0, float(T - s) / float(max(1, T - warmup)))
    return f


def lora_named(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def adapter_digest(model):
    h = hashlib.sha256()
    for n, p in lora_named(model):
        h.update(n.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def sum_loss(model, ids, att, lab):
    """Sum of next-token CE over supervised positions (PEFT model; LoRA layers are injected in place)."""
    inner = model.get_base_model()
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:].to(h.device)          # no-op on one GPU; needed if the base is split over GPUs
    m = tgt != -100
    logits = inner.lm_head(h[:, :-1][m]).float()
    return F.cross_entropy(logits, tgt[m].to(logits.device), reduction="sum"), int(m.sum())


class Stepper:
    """Forward/backward helper: token-mean over the whole example set, optional extra scale (released /GA)."""
    def __init__(self, model, pad_id, device, max_tokens, amp_dtype):
        self.model, self.pad_id, self.dev, self.max_tokens, self.amp = model, pad_id, device, max_tokens, amp_dtype

    def _ctx(self):
        return torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp is not None)

    def backward(self, exs, scale=1.0):
        lens = np.array([len(e[0]) for e in exs]); n_sup = int(sum(len(e[0]) - e[1] for e in exs))
        tot = torch.zeros((), device=self.dev)
        for mb in make_microbatches(lens, self.max_tokens):
            ids, att, lab = collate([exs[i] for i in mb], self.pad_id, self.dev)
            with self._ctx():
                l, _ = sum_loss(self.model, ids, att, lab)
            (l * (scale / n_sup)).backward()
            tot += l.detach().to(tot.device)   # no-op on one GPU
        return tot, n_sup, int(lens.sum())

    @torch.no_grad()
    def loss(self, exs, scale=1.0):
        lens = np.array([len(e[0]) for e in exs]); n_sup = int(sum(len(e[0]) - e[1] for e in exs))
        tot = 0.0
        for mb in make_microbatches(lens, self.max_tokens):
            ids, att, lab = collate([exs[i] for i in mb], self.pad_id, self.dev)
            with self._ctx():
                l, _ = sum_loss(self.model, ids, att, lab)
            tot += l.item()
        return tot / n_sup * scale, n_sup, int(lens.sum())


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def run_stage(model, data, batches, memory, method, *, seed, stage, lr=3e-4, warmup=50, clip=1.0, p=None, beta=None,
              beta_end=None, replay_k=8, aim_cfg=None, pad_id=0, device=torch.device("cpu"), max_tokens=16384,
              amp=torch.bfloat16, log=print, dev_sets=None, dev_every=40, fused=None, return_opt=False):
    """Train one stage in place. batches: list of arrays of global ids (current task). memory: list of global ids.
    Returns summary dict (accounting)."""
    model.train()
    named = lora_named(model)
    params = [p_ for _, p_ in named]
    T = len(batches)
    st = Stepper(model, pad_id, device, max_tokens, amp)
    if fused is None:
        fused = device.type == "cuda"
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=fused)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, hf_linear_lambda(T, warmup))
    rng_rep = np.random.default_rng([seed, stage, 7919])     # replay-arm draws
    rng_mon = np.random.default_rng([seed, stage, 4243])     # AIM monitor batches (RandomSampler, fresh iterator)
    rng_out = np.random.default_rng([seed, stage, 5501])     # AIM outer (replay) batches

    use_replay = method in ("replay", "sfa_replay")
    merge_kind = {"sfa": "sfa", "sfa_replay": "sfa", "lookahead": "lookahead"}.get(method)
    merges, b_eff = [], None
    if merge_kind:
        merges = merge_schedule(p, T)
        b_eff = beta if beta_end is None else matched_beta(beta_end, len(merges))
    merge_set = set(merges)
    anchor = [x.detach().clone() for x in params] if merge_kind else None
    aim_on = method == "aim" and len(memory) > 0           # released: every AIM branch needs train_dataset_memory (stage 1 = plain)
    ctrl = AimController(**(aim_cfg or {})) if aim_on else None
    imp = ImportanceEMA() if aim_on else None
    theta_t = {n: x.detach().clone() for n, x in named} if aim_on else None   # trainer.py:2244
    outer_bs = 32                                          # train_batch_size_outer (run_train_StandardCL.sh:25)

    acc = {k: 0 for k in ["cur_tok_in", "cur_tok_sup", "replay_grad_tok_in", "replay_grad_tok_sup", "monitor_tok_in",
                          "monitor_tok_sup", "cur_seqs", "replay_grad_seqs", "monitor_seqs", "inner_steps",
                          "replay_steps", "merges"]}
    tm = {k: 0.0 for k in ["monitor_sec", "replay_sec", "merge_sec", "importance_sec", "dev_loss_sec"]}
    aux_peak = {"theta_t_and_k": 0, "importance": 0}
    merge_steps, skipped, dev_curve = [], [], []
    aim_counts = {"adaptive_adjustments": 0, "cold_start_merges": 0}; aim_merges = []
    stage_start_digest = adapter_digest(model)
    log({"event": "stage_start", "adapter_sha256": stage_start_digest, "T": T, "method": method, "memory_size": len(memory),
         "merge_steps_planned": merges, "beta_effective": b_eff})

    def T_(): sync(device); return time.time()

    def dev_eval(u):
        if not dev_sets or (u % dev_every):
            return
        t0 = T_(); model.eval()
        r = {t: st.loss(e)[0] for t, e in dev_sets.items()}
        model.train(); tm["dev_loss_sec"] += T_() - t0
        r["mean"] = float(np.mean(list(r.values()))); dev_curve.append({"update": u, **r})
        log({"update": u, "dev_loss": r})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_loop = T_()
    win_l = win_n = 0.0
    for step in range(T):
        u = step + 1
        cur = [int(g) for g in batches[step] if g >= 0]
        exs = [data.example(g) for g in cur]
        nrep = 0
        if use_replay and memory:
            pick = rng_rep.integers(len(memory), size=replay_k)
            rep = [data.example(int(memory[i])) for i in pick]
            nrep = len(rep)
            rl = sum(len(e[0]) for e in rep); rs = sum(len(e[0]) - e[1] for e in rep)
            acc["replay_grad_tok_in"] += rl; acc["replay_grad_tok_sup"] += rs; acc["replay_grad_seqs"] += nrep
            exs = exs + rep
        l, n_sup, n_tok = st.backward(exs)
        cur_e = exs[:len(cur)]
        acc["cur_tok_in"] += sum(len(e[0]) for e in cur_e); acc["cur_tok_sup"] += sum(len(e[0]) - e[1] for e in cur_e)
        acc["cur_seqs"] += len(cur_e)
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        if not torch.isfinite(gn):
            skipped.append(u); opt.zero_grad(set_to_none=True); sch.step(); log({"update": u, "skipped": True}); continue
        opt.step(); sch.step(); acc["inner_steps"] += 1
        if aim_on:                                                        # trainer.py:2423-2424
            t0 = T_(); imp.update("inner", named); tm["importance_sec"] += T_() - t0
        opt.zero_grad(set_to_none=True)                                   # 2427 (global_step += 1 at 2429 == u)
        if aim_on:
            t0 = T_()                                                     # 2437-2457 monitor (train mode, dropout on)
            mi = rng_mon.permutation(len(memory))[:min(outer_bs, len(memory))]
            hl, ms, mt = st.loss([data.example(int(memory[i])) for i in mi], scale=1.0 / GA_REF)
            acc["monitor_tok_in"] += mt; acc["monitor_tok_sup"] += ms; acc["monitor_seqs"] += len(mi)
            tm["monitor_sec"] += T_() - t0
            do_merge, rec = ctrl.observe(u, hl)
            aim_counts[rec["phase"]] = aim_counts.get(rec["phase"], 0) + 1
            log({"aim_monitor": rec})
            if do_merge:
                t0 = T_()
                theta_k = {n: x.detach().clone() for n, x in named}        # 2524
                d_in_abs = float(sum((theta_k[n] - theta_t[n]).abs().sum().item() for n in theta_k))   # 2531, 2681
                aux_peak["theta_t_and_k"] = max(aux_peak["theta_t_and_k"],
                                                2 * sum(x.numel() * x.element_size() for x in theta_k.values()))
                tm["merge_sec"] += T_() - t0
                n_outer = ctrl.outer_iterations
                t0 = T_()
                for _ in range(n_outer):                                  # 2536-2630
                    oi = rng_out.permutation(len(memory))[:min(outer_bs, len(memory))]
                    oe = [data.example(int(memory[i])) for i in oi]
                    _, os_, ot = st.backward(oe, scale=1.0 / GA_REF)        # training_step: loss / GA (3438-3440)
                    acc["replay_grad_tok_in"] += ot; acc["replay_grad_tok_sup"] += os_; acc["replay_grad_seqs"] += len(oi)
                    torch.nn.utils.clip_grad_norm_(params, clip)           # 2584-2587
                    opt.step()                                            # 2623: SAME inner optimizer, no scheduler step
                    acc["replay_steps"] += 1
                    ti = T_(); imp.update("outer", named); tm["importance_sec"] += T_() - ti   # 2626
                    opt.zero_grad(set_to_none=True)
                tm["replay_sec"] += T_() - t0
                t0 = T_()
                w_in, w_out = ctrl.fusion_weights()                       # 1921-1938 (flag, S before update)
                fuse_(named, theta_t, theta_k, w_in, w_out)               # 2638, 2649
                theta_t = {n: x.detach().clone() for n, x in named}        # 2656
                aux_peak["importance"] = max(aux_peak["importance"], imp.nbytes())
                imp.empty("inner")                                        # 2658-2659 (empty_inner_score_flag=1)
                flag_at, S_at = ctrl.flag, ctrl.S
                info = ctrl.after_merge(u, d_in_abs)                      # 2681-2735
                del theta_k
                tm["merge_sec"] += T_() - t0
                acc["merges"] += 1; merge_steps.append(u)
                aim_counts["trigger_" + str(rec.get("trigger"))] = aim_counts.get("trigger_" + str(rec.get("trigger")), 0) + 1
                aim_counts["adaptive_adjustments" if info["adapted"] else "cold_start_merges"] += 1
                aim_merges.append({"update": u, "trigger": rec.get("trigger"), "merge_flag": flag_at, "S_before": S_at,
                                   "S_after": info["S_next"], "outer_iterations": n_outer, "lambda": info["lambda"],
                                   "adapted": info["adapted"], "w_in": w_in, "w_out": w_out})
                log({"aim_merge": {"update": u, "trigger": rec.get("trigger"), "merge_flag": flag_at, "S_sched": S_at,
                                   "outer_iterations": n_outer, "w_in": w_in, "w_out": w_out, **info,
                                   "lr": opt.param_groups[0]["lr"]}})
        if u in merge_set:
            t0 = T_()
            if merge_kind == "sfa":
                interp_(params, anchor, b_eff)
            else:
                with torch.no_grad():
                    for s_, x in zip(anchor, params):
                        s_.mul_(b_eff).add_(x, alpha=1 - b_eff); x.copy_(s_)
            tm["merge_sec"] += T_() - t0
            acc["merges"] += 1; merge_steps.append(u)
            log({"update": u, "merge": len(merge_steps), "beta": b_eff})
        win_l += l.item(); win_n += n_sup
        if u % 10 == 0 or u == T:
            log({"update": u, "loss": win_l / max(1, win_n), "lr_next": sch.get_last_lr()[0], "grad_norm": float(gn),
                 "elapsed": time.time() - t_loop, **{k: acc[k] for k in ("cur_tok_in", "replay_grad_tok_in", "monitor_tok_in")}})
            win_l = win_n = 0.0
        dev_eval(u)
    loop_sec = T_() - t_loop - tm["dev_loss_sec"]
    trainable = sum(x.numel() for x in params)
    summ = {"T": T, "updates_applied": T - len(skipped), "skipped": skipped, "merge_steps": merge_steps, **acc,
            "optimizer_steps_total": acc["inner_steps"] + acc["replay_steps"],
            "train_loop_sec_excl_devloss": loop_sec, **tm,
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None,
            "aux_state_bytes": {"trainable_params": trainable, "adapter_fp32": trainable * 4, "adam_state": trainable * 8,
                                "anchor_or_slow": trainable * 4 if anchor is not None else 0,
                                "theta_t_and_k_peak": aux_peak["theta_t_and_k"] if aim_on else 0,
                                "theta_t_persistent": trainable * 4 if aim_on else 0,
                                "importance_ema_peak": aux_peak["importance"] if aim_on else 0,
                                "memory_buffer_examples": len(memory) if method in ("replay", "aim", "sfa_replay") else 0},
            "stage_start_adapter_sha256": stage_start_digest, "final_adapter_sha256": adapter_digest(model),
            "beta_effective": b_eff, "dev_curve": dev_curve}
    if dev_curve:  # released load_best_model_at_end would pick the lowest dev loss among multiples of 40 (NOT used)
        best = min(dev_curve, key=lambda r: r["mean"])
        summ["released_selection_would_pick_update"] = best["update"]
    sup_tot = acc["cur_tok_sup"] + acc["replay_grad_tok_sup"]
    summ["replay_sup_token_share"] = acc["replay_grad_tok_sup"] / sup_tot if sup_tot else 0.0  # = replay's effective loss weight (replay arm)
    if aim_on:
        summ["aim_counts"] = aim_counts      # monitor phases (accumulate/threshold_set/forgetting_signal/forced_signal/monitor), triggers, adaptations
        summ["aim_merges"] = aim_merges
        summ["aim_final_state"] = {"S": ctrl.S, "outer_iterations": ctrl.outer_iterations, "lambda_list": ctrl.lam_list}
    return (summ, opt, sch) if return_opt else summ


def load_dev_sets(tok, tasks, n=25):
    """Teacher-forced cleaned-dev subsets (25/seen task; numpy rng 42 permutation as HF datasets shuffle(seed=42)).
    Used only to LOG what the released best-by-dev-loss selection would pick; never used to select."""
    out = {}
    for t in tasks:
        rows = [json.loads(l) for l in open(ROOT / f"data/processed/{t}/dev.jsonl")]
        idx = np.random.default_rng(42).permutation(len(rows))[:min(n, len(rows))]
        exs = []
        for i in idx:
            e = encode_pair(tok, rows[i]["prompt"], rows[i]["response"])
            if e is not None:
                exs.append((np.array(e[0]), e[1]))
        out[t] = exs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--schedule", required=True, help="runs/schedules/lora_o<order>_s<seed>_st<stage>_<task>_ep10")
    ap.add_argument("--init_adapter", default="none", help="adapter dir of the previous stage, or 'none' (stage 1)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", required=True, choices=["seqlora", "replay", "sfa", "lookahead", "aim", "sfa_replay"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--p", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--beta_end", type=float, default=None)
    ap.add_argument("--replay_k", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max_tokens_mb", type=int, default=16384)
    ap.add_argument("--no_dev_loss_log", action="store_true")
    ap.add_argument("--export_dense", default="yes", choices=["yes", "no"])
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="activation checkpointing (non-reentrant, RNG-preserving); used for every method at Qwen3-32B")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    import peft
    out = ROOT / "runs" / a.run_id; out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.jsonl", "w")
    def log(d):
        d["time"] = time.time(); logf.write(json.dumps(d, default=float) + "\n"); logf.flush()
        if "aim_monitor" not in d:
            print(json.dumps(d, default=float), flush=True)

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision())
    t_load = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=model_revision(), dtype=torch.float32,
                                                attn_implementation="sdpa").to(dev)
    base.config.use_cache = False
    if a.grad_ckpt:   # recomputation only: same math/precision; dropout masks replayed via preserved RNG state
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
    if a.init_adapter == "none":
        torch.manual_seed(a.seed)                    # adapter init (A kaiming-uniform, B = 0) identical across arms per seed
        model = get_peft_model(base, LoraConfig(**LORA))
    else:
        model = PeftModel.from_pretrained(base, a.init_adapter, is_trainable=True)
        assert all(p.dtype == torch.float32 for _, p in lora_named(model))
    torch.manual_seed(a.seed * 1000 + 1)             # dropout RNG
    model.to(dev)
    load_sec = time.time() - t_load
    B = np.load(ROOT / f"{a.schedule}.npz")
    sched = json.load(open(ROOT / f"{a.schedule}.json"))
    batches = list(B["batches"]); memory = [int(g) for g in B["memory"]]
    if a.max_steps:
        batches = batches[:a.max_steps]
    data = Data(None, sched["sources"])
    stage = sched["stage"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    dev_sets = None if a.no_dev_loss_log else load_dev_sets(tok, sched["sources"][:stage + 1])
    in_digest = adapter_digest(model)
    if a.init_adapter != "none":
        prev = json.load(open(Path(a.init_adapter).parent / "summary.json"))
        assert prev["final_adapter_sha256"] == in_digest, "stage chaining: loaded adapter != previous stage final adapter"
    code_dir = str(Path(__file__).resolve().parents[1])
    cfg = {**vars(a), "model_id": MODEL_ID, "trainable_params": sum(p.numel() for _, p in lora_named(model)), "lora": LORA,
           "peft": peft.__version__, "stage": stage, "task": sched["task"], "T": len(batches),
           "T_schedule": sched["T"], "schedule_sha256": sched["sha256"], "memory_size": len(memory),
           "model_revision": model_revision(), "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "load_sec": load_sec,
           "init_adapter_sha256": in_digest, "precision": "frozen fp32 base, fp32 LoRA master weights, bf16 autocast",
           "execution_change": "activation checkpointing (non-reentrant, RNG-preserving)" if a.grad_ckpt else None,
           "code_commit": subprocess.run(["git", "-C", code_dir, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    summ = run_stage(model, data, batches, memory, a.method, seed=a.seed, stage=stage, lr=a.lr, warmup=a.warmup, p=a.p,
                     beta=a.beta, beta_end=a.beta_end, replay_k=a.replay_k, pad_id=pad_id, device=dev,
                     max_tokens=a.max_tokens_mb, log=log, dev_sets=dev_sets)
    summ.update({"run_id": a.run_id, "status": "completed", "load_sec": load_sec, "n_gpus": 1})
    t0 = time.time()
    model.save_pretrained(out / "adapter")
    if a.export_dense == "yes":                     # merged dense W0 + (alpha/r) B A in fp32 for sfa.evaluate (vLLM)
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "dense", safe_serialization=True); tok.save_pretrained(out / "dense")
    summ["save_export_sec"] = time.time() - t0
    json.dump(summ, open(out / "summary.json", "w"), indent=1, default=float)
    log({"event": "done", **{k: v for k, v in summ.items() if k != "dev_curve"}})


if __name__ == "__main__":
    main()
