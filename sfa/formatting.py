"""Paths, model selection and chat formatting (Qwen3 chat template with enable_thinking=False), shared by training and inference.

All data, runs and results live under SFA_HOME (default: the repository root):
  data/      raw + processed + tokenized datasets       runs/      checkpoints, schedules, training logs
  results/   evaluation summaries and per-example predictions     cache/     temporary files
"""
import json, os
from pathlib import Path

ROOT = Path(os.environ.get("SFA_HOME", Path(__file__).resolve().parents[1]))
MODELS = json.load(open(Path(__file__).resolve().parents[1] / "configs/models.json"))

# Model selection: SFA_MODEL (a key of configs/models.json["models"]); default Qwen/Qwen3-1.7B. The revision is always pinned.
MODEL_ID = os.environ.get("SFA_MODEL", MODELS["default_model"])
if MODEL_ID not in MODELS["models"]:
    raise RuntimeError(f"unknown SFA_MODEL={MODEL_ID}; choose one of {sorted(MODELS['models'])}")
MAX_LEN = 4096

LAW_INSTR = "\nAnswer with the exact text of the correct option."


def model_revision(_pins=None):
    """Pinned Hugging Face revision of the selected model."""
    return MODELS["models"][MODEL_ID]["revision"]


def law_user(r):
    opts = "".join(f"\n - {c}" for c in r["choices"])
    return f"{r['query']}\nOptions:{opts}{LAW_INSTR}"


def gsm_user(r):
    return r["question"]


def train_pair(task, r):
    """(user, assistant) for a training record."""
    if task == "law":
        return law_user(r), r["choices"][r["gold"]]
    return r["prompt"], r["response"]


def prompt_ids(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=True, return_dict=False)


def encode_pair(tok, user, assistant):
    """Return (ids, n_prompt) with loss on assistant content + <|im_end|>; None if boundary mismatch.

    The full conversation is rendered by the chat template itself and the prompt part is
    verified to be an exact token prefix, so no BOS/EOS/template tokens are duplicated.
    """
    p = prompt_ids(tok, user)
    full = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                                   enable_thinking=False, tokenize=True, return_dict=False)
    if full[:len(p)] != p:
        return None
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    # drop the template's trailing newline after the final <|im_end|>
    while full and full[-1] != im_end:
        full = full[:-1]
    if len(full) <= len(p):
        return None
    return full, len(p)
