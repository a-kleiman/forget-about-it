"""Summarize the single-stage timing benchmark (sfa.timing): mean training-loop time per method and the ratio to SFA.

  python -m sfa.analysis.timing [files...]     # default: results/timing/*.jsonl
"""
import glob, json, os, sys

import numpy as np

from sfa.formatting import ROOT


def main():
    files = sys.argv[1:] or sorted(glob.glob(str(ROOT / "results/timing/*.jsonl")))
    for f in files:
        rs = [json.loads(l) for l in open(f)]
        arms = sorted({r["arm"] for r in rs}, key=["sfa", "replay", "aim"].index)
        loop = {a: [r["train_loop_sec_excl_devloss"] for r in rs if r["arm"] == a] for a in arms}
        whole = {a: [r["whole_process_sec"] for r in rs if r["arm"] == a] for a in arms}
        print(f"## {os.path.basename(f)}")
        print("| method | training loop (s), per repetition | mean | ratio to SFA | whole process (s) | peak GPU memory (GB) |")
        print("|---|---|---|---|---|---|")
        for a in arms:
            pk = max(r.get("peak_alloc_gb") or 0 for r in rs if r["arm"] == a)
            print(f"| {a} | {', '.join(f'{x:.1f}' for x in loop[a])} | {np.mean(loop[a]):.1f} | {np.mean(loop[a]) / np.mean(loop['sfa']):.3f} | "
                  f"{np.mean(whole[a]):.1f} | {pk:.1f} |")
        print()


if __name__ == "__main__":
    main()
