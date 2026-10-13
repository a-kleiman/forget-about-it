"""Controlled single-stage training-loop timing: SFA vs replay vs AIMMerging on ONE GPU, sequentially, two repetitions with the
method order reversed in the second repetition.

Identical across methods: seed 17, order 1, stage 2 (Amazon; the first stage where replay and AIMMerging are active), the same
start adapter (sequential-LoRA stage-1 adapter of seed 17, so run that stream's first stage first), fp32 base + bf16 autocast, the
model's microbatch cap, default method settings, development-loss logging OFF. Reported quantity: the trainer's
training-loop time (summary.json "train_loop_sec_excl_devloss"), plus whole-process wall time. This measures one stage's training
loop only: not complete-stream or end-to-end runtime.

  SFA_MODEL=Qwen/Qwen3-14B python -m sfa.timing          # writes results/timing/<prefix>_timing.jsonl and prints a summary
"""
import json, shutil, subprocess, sys, time

import numpy as np

from sfa.formatting import MODEL_ID, MODELS, ROOT

ARMS = {"sfa": ["--method", "sfa", "--p", "0.25", "--beta_end", "0.5"], "replay": ["--method", "replay", "--replay_k", "8"],
        "aim": ["--method", "aim"]}
SEQUENCE = [(1, "sfa"), (1, "replay"), (1, "aim"), (2, "aim"), (2, "replay"), (2, "sfa")]
KEEP = ["train_loop_sec_excl_devloss", "monitor_sec", "replay_sec", "merge_sec", "importance_sec", "cur_tok_in", "replay_grad_tok_in",
        "monitor_tok_in", "inner_steps", "replay_steps", "merges", "optimizer_steps_total", "peak_alloc_gb"]


def main():
    m = MODELS["models"][MODEL_ID]; prefix = m["run_prefix"]
    init = ROOT / f"runs/{prefix}_o1_s17_seqlora_st0/adapter"
    assert init.exists(), f"run the sequential-LoRA stream's first stage first: {init}"
    out = ROOT / f"results/timing/{prefix}_timing.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for rep, arm in SEQUENCE:
        rid = f"{prefix}_timing_r{rep}_{arm}_st1"
        shutil.rmtree(ROOT / f"runs/{rid}", ignore_errors=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-u", "-m", "sfa.train_lora", "--run_id", rid, "--schedule", "runs/schedules/lora_o1_s17_st1_amazon_ep10",
                        "--init_adapter", str(init), "--seed", "17", *ARMS[arm], "--max_tokens_mb", str(m["adapter_microbatch_tokens"]),
                        "--no_dev_loss_log"] + (["--grad_ckpt"] if m.get("activation_checkpointing") else []), check=True)
        s = json.load(open(ROOT / f"runs/{rid}/summary.json"))
        r = {"rep": rep, "arm": arm, "run_id": rid, "whole_process_sec": time.time() - t0, **{k: s.get(k) for k in KEEP}}
        recs.append(r)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
    loop = {a: float(np.mean([r["train_loop_sec_excl_devloss"] for r in recs if r["arm"] == a])) for a in ARMS}
    print(json.dumps({"model": MODEL_ID, "mean_training_loop_sec": loop,
                      "relative_to_sfa": {a: loop[a] / loop["sfa"] for a in ARMS}}, indent=1))


if __name__ == "__main__":
    main()
