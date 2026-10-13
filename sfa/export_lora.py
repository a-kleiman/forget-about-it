"""Export merged dense weights W0 + (alpha/r) B A (fp32) from a saved adapter, for evaluation with vLLM.
Same code path as the export at the end of sfa.train_lora (peft merge_and_unload in fp32).

  python -m sfa.export_lora --adapter runs/<run_id>/adapter --out runs/<run_id>/dense
"""
import argparse
import torch
from sfa.formatting import MODEL_ID, model_revision


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision())
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=model_revision(), dtype=torch.float32)
    m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    m.save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
