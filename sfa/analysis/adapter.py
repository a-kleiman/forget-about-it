"""Tables for the adapter-level continual-learning experiments, computed from results/evals/*.json.

  python -m sfa.analysis.adapter        # writes results/tables/adapter_{streams,paired,selection}.{md,csv}

Per stream (model prefix, order, seed, arm), A[j][i] = accuracy on task i after stage j:
  final average = mean_i A[3][i];  acquisition = mean_j A[j][j];  earlier-task accuracy = mean_{i<3} A[3][i];
  forgetting = mean_{i<3} (max_{i<=j<3} A[j][i] - A[3][i]).
Selection: the retention-constrained coefficient rule of configs/adapter.json on seed-17 DEV (where candidates exist).
"""
import csv, json
from pathlib import Path

import numpy as np

from sfa.formatting import MODELS, ROOT

CFG = json.load(open(Path(__file__).resolve().parents[2] / "configs/adapter.json"))
OUT = ROOT / "results/tables"
SEEDS = CFG["seeds"]
PREFIXES = [(v["run_prefix"], k) for k, v in MODELS["models"].items()]


def stream(prefix, order, seed, arm, split):
    tasks = CFG["orders"][str(order)]; N = len(tasks); A = np.full((N, N), np.nan)
    for j in range(N):
        f = ROOT / f"results/evals/{prefix}_o{order}_s{seed}_{arm}_st{j}__{split}.json"
        if not f.exists(): return None
        m = json.load(open(f))
        if not m.get("complete"): return None
        for i in range(j + 1): A[j, i] = m["tasks"][f"cl_{tasks[i]}"]["acc"]
    return {"final_avg": float(A[N - 1].mean()), "acquisition": float(np.mean(np.diag(A))),
            "earlier_task": float(np.mean(A[N - 1, :N - 1])),
            "forgetting": float(np.mean([np.max(A[i:N - 1, i]) - A[N - 1, i] for i in range(N - 1)]))}


def ms(v, signed=False):
    v = [100 * x for x in v]
    if not v: return "—"
    f = "+.1f" if signed else ".1f"
    return f"{np.mean(v):{f}}" + (f" ± {np.std(v, ddof=1):.1f}" if len(v) > 1 else "") + " [" + ", ".join(f"{x:{f}}" for x in v) + "]"


def streams():
    L = ["| model | order | arm | final avg | acquisition | earlier-task | forgetting |", "|---|---|---|---|---|---|---|"]
    rows = []
    for prefix, model in PREFIXES:
        for order in CFG["orders"]:
            for arm in CFG["arms"]:
                v = [stream(prefix, order, s, arm, "test") for s in SEEDS]
                got = [(s, x) for s, x in zip(SEEDS, v) if x]
                if not got: continue
                for s, x in got: rows.append({"model": model, "order": order, "arm": arm, "seed": s, **x})
                xs = [x for _, x in got]
                L.append(f"| {model} | {order} | {CFG['arms'][arm]['label']} | " + " | ".join(ms([x[k] for x in xs]) for k in
                         ("final_avg", "acquisition", "earlier_task", "forgetting")) + (" |" if len(got) == 3 else f" (seeds {[s for s, _ in got]}) |"))
    return L, rows


def paired():
    comps = [("sfa", "seqlora"), ("sfa", "replay"), ("sfa", "aim"), ("sfa", "lookahead"), ("look_b75", "sfa"),
             ("sfa_replay", "aim"), ("sfa_replay", "replay"), ("sfa_replay", "look_b75")]
    L = ["| model | order | comparison (final avg, points) | earlier-task | acquisition |", "|---|---|---|---|---|"]
    rows = []
    for prefix, model in PREFIXES:
        for order in CFG["orders"]:
            for a, b in comps:
                d = {k: [] for k in ("final_avg", "earlier_task", "acquisition")}
                for s in SEEDS:
                    x, y = stream(prefix, order, s, a, "test"), stream(prefix, order, s, b, "test")
                    if x and y:
                        for k in d: d[k].append(x[k] - y[k])
                        rows.append({"model": model, "order": order, "comparison": f"{a} - {b}", "seed": s, **{f"d_{k}": x[k] - y[k] for k in d}})
                if len(d["final_avg"]) == len(SEEDS):
                    L.append(f"| {model} | {order} | {a} − {b}: {ms(d['final_avg'], True)} | {ms(d['earlier_task'], True)} | {ms(d['acquisition'], True)} |")
    return L, rows


def selection():
    R = CFG["coefficient_selection_rule"]; tols = R["requirements"]; dseed = CFG["development_seed"]
    L, rows = [], []
    for prefix, model in PREFIXES:
        ref = stream(prefix, 1, dseed, "seqlora", "dev")
        grids = CFG["coefficient_grids"]
        if ref is None or not all(stream(prefix, 1, dseed, arm, "dev") for g in grids.values() for arm in g.values()):
            continue
        A_ref = ref["acquisition"]
        L += ["", f"### {model} (A_ref = sequential-LoRA dev acquisition = {A_ref:.4f})", "",
              "| method | coefficient | dev acquisition | dev earlier-task | " + " | ".join(tols) + " |", "|---|---|---|---|" + "---|" * len(tols)]
        picks = {}
        for meth, grid in grids.items():
            D = {c: stream(prefix, 1, dseed, arm, "dev") for c, arm in grid.items()}
            for c, d in D.items():
                L.append(f"| {meth} | {c} | {d['acquisition']:.4f} | {d['earlier_task']:.4f} | " +
                         " | ".join("✓" if d["acquisition"] >= A_ref - t else "✗" for t in tols.values()) + " |")
                rows.append({"model": model, "method": meth, "coefficient": c, **{f"dev_{k}": v for k, v in d.items()}, "A_ref": A_ref})
            for rn, t in tols.items():
                ok = {c: d for c, d in D.items() if d["acquisition"] >= A_ref - t}
                if not ok: picks[(meth, rn)] = "INFEASIBLE"; continue
                best = max(d["earlier_task"] for d in ok.values())
                tied = [c for c, d in ok.items() if d["earlier_task"] >= best - 0.002]
                picks[(meth, rn)] = "0.5" if "0.5" in tied else max(tied, key=float)
        L += ["", "| requirement | " + " | ".join(grids) + " |", "|---|" + "---|" * len(grids)]
        for rn in tols:
            L.append(f"| {rn} | " + " | ".join(picks[(m, rn)] for m in grids) + " |")
    return L, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("adapter_streams", streams), ("adapter_paired", paired), ("adapter_selection", selection)):
        lines, rows = fn()
        (OUT / f"{name}.md").write_text("\n".join([f"# {name} (test unless noted; accuracy in points)", ""] + lines) + "\n")
        if rows:
            with open(OUT / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
        print("\n".join([f"## {name}"] + lines), "\n")


if __name__ == "__main__":
    main()
