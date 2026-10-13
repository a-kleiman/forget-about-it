"""Evaluation: GSM8K (greedy generation, answer extraction), Law multiple choice (length-normalized candidate scoring),
continual-learning tasks cl_<task> (greedy generation, case-insensitive exact match).

Model spec: a HF dir, 'base', or 'interp:<W>:<A>:<beta>' = (1-beta)*W + beta*A in fp32 (beta = anchor coefficient;
used for the end-of-training interpolation baseline). Inference in bf16 with vLLM (generation) / transformers (scoring).
Writes results/predictions/<split>/<tag>/<task>.jsonl and results/evals/<tag>__<split>.json ("complete": true when finished).

  python -m sfa.evaluate --model runs/<run_id>/final --tag <run_id> --split dev --tasks math law
  python -m sfa.evaluate --model runs/<run_id>/dense --tag <run_id> --split test --tasks cl_dbpedia cl_amazon
"""
import argparse, json, re, shutil, tempfile, time
from pathlib import Path

import numpy as np
import torch

from sfa.formatting import MAX_LEN, MODEL_ID, ROOT, model_revision, encode_pair, gsm_user, law_user, prompt_ids

GEN_MAX_NEW = 1024


# ---------------------------------------------------------------- model materialization
def materialize(spec, pins):
    """Return (path, cleanup_dir). Interp specs are computed in fp32 and stored in bf16 for inference."""
    if spec == "base":
        from huggingface_hub import snapshot_download
        return snapshot_download(MODEL_ID, revision=model_revision(pins)), None
    if not spec.startswith("interp:"):
        return spec, None
    _, w, an, beta = spec.split(":")
    beta = float(beta)
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    def load_dir(d):
        if d == "base":
            from huggingface_hub import snapshot_download
            d = snapshot_download(MODEL_ID, revision=model_revision(pins))
        sd = {}
        for f in sorted(Path(d).glob("*.safetensors")):
            sd.update(load_file(str(f)))
        return d, sd
    wd, sw = load_dir(w); ad, sa = load_dir(an)
    if "lm_head.weight" in sw and "lm_head.weight" not in sa and "model.embed_tokens.weight" in sa:
        sa["lm_head.weight"] = sa["model.embed_tokens.weight"]
    out = {}
    for k, v in sw.items():
        if v.is_floating_point():
            out[k] = ((1 - beta) * v.float() + beta * sa[k].float()).to(torch.bfloat16).contiguous()
        else:
            out[k] = v
    (ROOT / "cache/tmp").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "cache/tmp", prefix="interp_"))
    save_file(out, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    for f in Path(wd).iterdir():
        if f.suffix in (".json", ".txt", ".jinja") and "index" not in f.name:
            shutil.copy(f, tmp / f.name)
    return str(tmp), tmp


# ---------------------------------------------------------------- law
def law_eval(path, tok, splits, max_tokens=32768):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    res = {}
    for name, rows in splits.items():
        items = []
        for ri, r in enumerate(rows):
            u = law_user(r)
            for ci, c in enumerate(r["choices"]):
                e = encode_pair(tok, u, c)
                ids, npr = e  # no truncation: scored at full length (model context 40k)
                items.append((ri, ci, ids, npr))
        order = sorted(range(len(items)), key=lambda i: -len(items[i][2]))
        scores = {}
        i = 0
        with torch.no_grad():
            while i < len(order):
                L = len(items[order[i]][2]); n = max(1, max_tokens // L)
                chunk = [items[j] for j in order[i:i + n]]; i += n
                ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros_like(ids)
                for b, (_, _, t, _) in enumerate(chunk):
                    ids[b, :len(t)] = torch.tensor(t); att[b, :len(t)] = 1
                ids, att = ids.cuda(), att.cuda()
                h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
                for b, (ri, ci, t, npr) in enumerate(chunk):
                    hs = h[b, npr - 1:len(t) - 1]
                    lp = torch.log_softmax(model.lm_head(hs).float(), -1)
                    tgt = torch.tensor(t[npr:], device=lp.device)
                    tl = lp.gather(1, tgt[:, None]).squeeze(1)
                    scores[(ri, ci)] = (tl.mean().item(), tl.sum().item(), len(tgt))
        preds = []
        for ri, r in enumerate(rows):
            mean = [scores[(ri, c)][0] for c in range(len(r["choices"]))]
            ssum = [scores[(ri, c)][1] for c in range(len(r["choices"]))]
            p = int(np.argmax(mean))
            preds.append({"uid": r["uid"], "gold": r["gold"], "pred": p, "correct": int(p == r["gold"]),
                          "pred_sum": int(np.argmax(ssum)), "mean_logp": mean})
        acc = float(np.mean([x["correct"] for x in preds]))
        ncls = max(len(r["choices"]) for r in rows)
        per_cls = {c: float(np.mean([x["correct"] for x in preds if x["gold"] == c])) for c in range(ncls)
                   if any(x["gold"] == c for x in preds)}
        res[name] = {"acc": acc, "acc_sum_unnormalized": float(np.mean([x["pred_sum"] == x["gold"] for x in preds])),
                     "macro_recall": float(np.mean(list(per_cls.values()))), "per_class_recall": per_cls,
                     "class_balance": {c: int(sum(x["gold"] == c for x in preds)) for c in per_cls}, "n": len(rows),
                     "_preds": preds}
    del model; torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- math
NUM = r"-?[\d,]*\.?\d+"


def to_num(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        m = re.findall(NUM, s)
        return float(m[-1].replace(",", "")) if m else None


def extract_math(text):
    m = re.findall(r"The answer is:?\s*\$?\s*(" + NUM + ")", text)
    if m:
        return to_num(m[-1]), "answer_is"
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        v = to_num(m[-1])
        if v is not None:
            return v, "boxed"
    m = re.findall(NUM, text)
    return (to_num(m[-1]), "last_number") if m else (None, "none")


# ---------------------------------------------------------------- generation
def generate(path, tok, prompts, max_new=GEN_MAX_NEW):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=path, tokenizer=path, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=MAX_LEN + GEN_MAX_NEW,
              seed=0, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new, seed=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    res = [(o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids)) for o in outs]
    del llm
    return res


def load_rows(rel):
    return [json.loads(l) for l in open(ROOT / "data/processed" / rel)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--tasks", nargs="+", default=["math", "law"])
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: first N examples per task")
    a = ap.parse_args()
    pins = None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision(pins))
    t0 = time.time()
    path, tmp = materialize(a.model, pins)
    t_mat = time.time() - t0
    pdir = ROOT / "results/predictions" / a.split / a.tag; pdir.mkdir(parents=True, exist_ok=True)
    metrics = {"tag": a.tag, "model": a.model, "split": a.split, "limit": a.limit,
               "gen_max_new_tokens": GEN_MAX_NEW, "decoding": "greedy", "chat_template": "enable_thinking=False",
               "inference_dtype": "bfloat16", "materialize_sec": t_mat, "tasks": {}}
    lim = lambda rows: rows[: a.limit] if a.limit else rows
    od = ROOT / "results/evals"; od.mkdir(parents=True, exist_ok=True)
    mpath = od / f"{a.tag}__{a.split}{'_lim' + str(a.limit) if a.limit else ''}.json"
    def save():
        metrics["total_sec"] = time.time() - t0
        json.dump(metrics, open(mpath, "w"), indent=1)
    try:
        if "law" in a.tasks:
            t1 = time.time()
            sp = {s: lim(load_rows(f"law/{s}_{a.split}.jsonl")) for s in ["casehold", "tos", "overruling"]}
            res = law_eval(path, tok, sp)
            for s, r in res.items():
                with open(pdir / f"law_{s}.jsonl", "w") as f:
                    for x in r.pop("_preds"):
                        f.write(json.dumps(x) + "\n")
                metrics["tasks"][f"law_{s}"] = r
            metrics["tasks"]["law_mean"] = {"acc": float(np.mean([res[s]["acc"] for s in res]))}
            metrics["law_sec"] = time.time() - t1
            save()
        gen_jobs = []
        if "math" in a.tasks:
            rows = lim(load_rows(f"math/{a.split}.jsonl")); gen_jobs.append(("math", rows, [gsm_user(r) for r in rows]))
        for t in [t for t in a.tasks if t.startswith("cl_")]:
            rows = lim(load_rows(f"{t}/{a.split}.jsonl")); gen_jobs.append((t, rows, [r["prompt"] for r in rows]))
        if gen_jobs:
            t1 = time.time()
            allp = [prompt_ids(tok, u) for _, _, us in gen_jobs for u in us]
            only_cl = all(t.startswith("cl_") for t, _, _ in gen_jobs)
            outs = generate(path, tok, allp, max_new=32 if only_cl else GEN_MAX_NEW)
            metrics["gen_max_new_tokens"] = 32 if only_cl else GEN_MAX_NEW
            metrics["gen_sec"] = time.time() - t1
            k = 0
            for task, rows, _ in gen_jobs:
                o = outs[k:k + len(rows)]; k += len(rows)
                preds = []
                if task == "math":
                    for r, (txt, fin, nt) in zip(rows, o):
                        v, how = extract_math(txt)
                        ok = v is not None and abs(v - float(r["answer"])) < 1e-6
                        preds.append({"uid": r["uid"], "gold": r["answer"], "pred": v, "how": how, "correct": int(ok),
                                      "finish": fin, "n_tokens": nt, "output": txt})
                else:
                    for r, (txt, fin, nt) in zip(rows, o):
                        ok = txt.strip().lower() == r["label"].strip().lower()
                        preds.append({"uid": r["uid"], "gold": r["label"], "pred": txt.strip()[:100], "correct": int(ok), "finish": fin, "n_tokens": nt})
                with open(pdir / f"{task}.jsonl", "w") as f:
                    for x in preds:
                        f.write(json.dumps(x) + "\n")
                metrics["tasks"][task] = {"acc": float(np.mean([x["correct"] for x in preds])), "n": len(preds),
                                          "truncated_frac": float(np.mean([x["finish"] == "length" for x in preds]))}
                if task == "math":
                    metrics["tasks"][task]["mean_output_tokens"] = float(np.mean([x["n_tokens"] for x in preds]))
                    metrics["tasks"][task]["extraction"] = {h: sum(x["how"] == h for x in preds) for h in ["answer_is", "boxed", "last_number", "none"]}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    metrics["complete"] = True
    save()
    print(json.dumps({k: v for k, v in metrics["tasks"].items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
