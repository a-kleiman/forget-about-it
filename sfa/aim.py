"""AIMMerging (Feng et al., EMNLP 2025), faithful port of the RELEASED code (WoodScene/AimMerging @ 90733c38e571,
https://github.com/WoodScene/AimMerging). Reimplemented for use as a baseline; please cite the original paper.

Every rule cites the released file:line (paths relative to the AimMerging repo). Pure-python / torch helpers with no
training-loop state.

Released Standard-CL settings (scripts/run_train_StandardCL.sh): inner_step=S_init=8, outer_step=8 (ratio 1),
cold_start_step=3, max_step_len=128, min_step_len=2, threshold_factor=2.0, adaptive_merge_flag=True,
empty_inner_score_flag=1 (empty_outer_score_flag default 0), batch 32 = micro 8 x GA 4, train_batch_size_outer=32.
"""
import json, random
from collections import Counter
from pathlib import Path

import torch

GA_REF = 4  # released gradient_accumulation_steps = batch_size // micro_batch_size = 32 // 8 (finetune_ours.py:189)


class AimController:
    """Merge controller state machine: trainer.py:2476-2497 (threshold / forgetting signal), 2513-2514 (trigger),
    1921-1938 (v2 fusion weights), 2681-2735 (learning signal, adaptive interval, resets)."""

    def __init__(self, S_init=8, outer_init=8, cold_start_step=3, S_min=2, S_max=128, threshold_factor=2.0,
                 adaptive=True):
        self.S = S_init                                   # self._inner_iterations (trainer.py:785)
        self.outer_iterations = outer_init                # trainer.py:576
        self.ratio = outer_init / S_init                  # inner_outer_ratio (trainer.py:789)
        self.outer_begin = S_init                         # self._outer_begin = self._inner_iterations (trainer.py:787)
        self.cold, self.S_min, self.S_max = cold_start_step, S_min, S_max
        self.factor, self.adaptive = threshold_factor, adaptive
        self.step_cal, self.hist, self.threshold, self.flag = 0, [], 0.0, 0   # trainer.py:2249-2252
        self.lam_list = []                                # ipt_score.delta_in_abs_change_list
        self.last_event = None

    def observe(self, global_step, history_loss):
        """Call once per optimizer step AFTER global_step += 1 with that step's monitor loss.
        Returns (merge_now: bool, record: dict)."""
        rec = {"update": global_step, "history_loss": history_loss, "S": self.S, "outer_begin": self.outer_begin}
        w = int(self.S * 2 / 3)
        if self.step_cal < w:                                             # 2477-2480
            self.hist.append(history_loss); self.step_cal += 1; rec["phase"] = "accumulate"
        elif self.step_cal == w:                                          # 2481-2486
            self.hist.append(history_loss)
            self.threshold = sum(self.hist) / len(self.hist) * self.factor
            self.step_cal += 1; rec["phase"] = "threshold_set"
        else:                                                             # 2487-2497
            if history_loss >= self.threshold:
                self.flag += 1; rec["phase"] = "forgetting_signal"
            elif global_step >= self.outer_begin + self.S:
                self.flag += 1; rec["phase"] = "forced_signal"
            else:
                rec["phase"] = "monitor"
        rec.update(threshold=self.threshold, merge_flag=self.flag)
        merge = (global_step >= self.outer_begin and self.flag >= 1) or self.flag >= 3   # 2514
        if merge:
            rec["trigger"] = "early_flag>=3" if (global_step < self.outer_begin and self.flag >= 3) else (
                "forced_2S" if rec["phase"] == "forced_signal" else "scheduled_with_signal")
        return merge, rec

    def fusion_weights(self):
        """update_parameters_with_task_vectors_version2 (trainer.py:1921-1938), called with (merge_flag, S)."""
        if self.flag > 1:
            return (0.5, 0.5) if self.S <= 70 else (0.4, 0.6)
        return (0.6, 0.4) if self.S <= 70 else (0.5, 0.5)

    def after_merge(self, global_step, delta_in_abs_sum):
        """trainer.py:2681-2735. delta_in_abs_sum = sum_n |delta_in[n]|.sum() over all lora_ tensors."""
        S_used = self.S
        lam = delta_in_abs_sum / self.S                                   # 2681
        adapted, change_factor = False, None
        if self.adaptive:                                                 # 2686
            if len(self.lam_list) <= self.cold:                           # 2692 (cold start)
                pass
            else:
                prev = self.lam_list[-1]                                  # 2696
                if lam >= prev:                                           # 2702-2707
                    change_factor = lam / prev
                    self.S = max(self.S_min, int(self.S / change_factor))
                    if self.S >= 64:
                        self.S = int(self.S / 2)
                else:                                                     # 2708-2713
                    change_factor = (prev / lam) + 1 if self.S < 64 else (prev / lam) + 0.5
                    self.S = min(self.S_max, int(self.S * change_factor))
                self.outer_iterations = int(self.S * self.ratio)          # 2718
                adapted = True
        self.lam_list.append(lam)                                         # 2722
        self.outer_begin = global_step + self.S                           # 2726
        self.step_cal, self.hist, self.threshold, self.flag = 0, [], 0.0, 0   # 2732-2735
        return {"lambda": lam, "S_used": S_used, "S_next": self.S, "outer_iterations_next": self.outer_iterations,
                "adapted": adapted, "change_factor": change_factor, "cold_start": not adapted}


class ImportanceEMA:
    """RankAllocator(taylor='param_first', beta1=beta2=0.85) from utils/lora_importance_bilevel_adaptive.py:85-143.
    KEPT ONLY for faithful cost accounting: its output is consumed solely by update_parameters_with_task_vectors (v1,
    trainer.py:1828-1829), whose call is commented out (2647); the active v2 merge (2649) takes no masks."""

    def __init__(self, beta1=0.85, beta2=0.85):
        self.b1, self.b2 = beta1, beta2
        self.ipt, self.avg, self.unc = {"inner": {}, "outer": {}}, {"inner": {}, "outer": {}}, {"inner": {}, "outer": {}}

    @torch.no_grad()
    def update(self, which, named_params):
        ipt, avg, unc = self.ipt[which], self.avg[which], self.unc[which]
        for n, p in named_params:
            if "lora_" not in n:
                continue
            if torch.isnan(p.grad).any():                                 # :89-94 / :120-124 (break on NaN)
                break
            if n not in ipt:
                ipt[n] = torch.zeros_like(p); avg[n] = torch.zeros_like(p); unc[n] = torch.zeros_like(p)
            ipt[n] = (p * p.grad).abs().detach()
            avg[n] = self.b1 * avg[n] + (1 - self.b1) * ipt[n]
            unc[n] = self.b2 * unc[n] + (1 - self.b2) * (ipt[n] - avg[n]).abs()

    def empty(self, which):
        self.ipt[which], self.avg[which], self.unc[which] = {}, {}, {}

    def nbytes(self):
        return sum(t.numel() * t.element_size() for d in (self.ipt, self.avg, self.unc) for w in d.values() for t in w.values())


@torch.no_grad()
def fuse_(named_params, theta_t, theta_k, w_in, w_out):
    """theta <- theta_t + w_in*(theta_k - theta_t) + w_out*(theta - theta_k)   (trainer.py:2531, 2638, 1941-1947).
    Computed in fp32 (adapter params are fp32 master weights)."""
    for n, p in named_params:
        if n in theta_t:
            d_in = theta_k[n] - theta_t[n]
            d_out = p.detach() - theta_k[n]
            p.copy_(theta_t[n] + (w_in * d_in + w_out * d_out))


def released_memory_indices(rows, ratio=2, seed=42, task_order=None):
    """utils/load_data.py:84-148 for ONE call. rows: list of per-task row lists (file order), tasks 0..task_id-1.
    Returns list of (task_pos, row_index) in the released concatenation order. random.seed(42) at every call (:96)."""
    random.seed(seed)
    out = []
    for i, task_data in enumerate(rows):
        num_samples = int(len(task_data) * ratio / 100)                   # :115
        labels = [r["output"] for r in task_data]
        label_counts = Counter(labels)                                    # :120
        num_classes = len(label_counts)
        samples_per_class = max(1, num_samples // num_classes)            # :122
        sampled = []
        for label in label_counts:                                        # :126-128
            label_indices = [idx for idx, value in enumerate(labels) if value == label]
            sampled.extend(random.sample(label_indices, min(samples_per_class, len(label_indices))))
        if len(sampled) < num_samples:                                    # :131-133
            remaining = list(set(range(len(task_data))) - set(sampled))
            sampled.extend(random.sample(remaining, num_samples - len(sampled)))
        out += [(i, j) for j in sampled]
    return out


def build_memory(root, order, stage):
    """Memory buffer for stage `stage` (0-based) of `order` (list of task names): released buffer over tasks
    0..stage-1, mapped from the released file rows (our data/raw/cl_standard/train/<task>.json has identical order and
    content to AimMerging's data_longsequence/train/<task>_T5.json) to our tokenized indices via uid."""
    import numpy as np
    if stage == 0:
        return []
    rows = [json.load(open(Path(root) / f"data/raw/cl_standard/train/{t}.json")) for t in order[:stage]]
    picks = released_memory_indices(rows)
    out = []
    for ti, j in picks:
        t = order[ti]
        uid = f"cl_{t}_train_{rows[ti][j]['id']}"
        z = np.load(Path(root) / f"data/tokenized/cl_{t}/train.npz")
        pos = int(np.nonzero(z["uid"] == uid)[0][0])
        out.append({"task": t, "src": ti, "row": j, "uid": uid, "idx": pos, "label": rows[ti][j]["output"]})
    return out
