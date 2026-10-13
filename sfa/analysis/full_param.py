"""Tables for the full-parameter Math->Law experiments, computed from results/evals/*.json.

  python -m sfa.analysis.full_param            # writes results/tables/full_param_{selection,test,paired}.{md,csv} (+ figure)

1. selection: the retention-constrained development rule (configs/full_param.json["selection_rule"]) on seed-17 dev.
2. test: GSM8K and Law (unweighted mean of CaseHOLD/ToS/Overruling) test accuracy per configuration, mean +- sample SD over seeds.
3. paired: per-seed paired differences on both axes for the R_mid comparison (SFA beta_end .5, Lookahead beta .5, FT lr 5e-6),
   reported separately for seeds 17/42/123, for the additional seeds 7/2026, and pooled. Descriptive only (no significance tests).
"""
import csv, json
from pathlib import Path

import numpy as np

from sfa.formatting import ROOT

CFG = json.load(open(Path(__file__).resolve().parents[2] / "configs/full_param.json"))
OUT = ROOT / "results/tables"
SUB = ("law_casehold", "law_tos", "law_overruling")


def ev(tag, split):
    f = ROOT / f"results/evals/{tag}__{split}.json"
    if not f.exists(): return None
    m = json.load(open(f))
    if not m.get("complete"): return None
    t = m["tasks"]
    return {"gsm8k": t["math"]["acc"], "gsm8k_n": t["math"]["n"], "law": t["law_mean"]["acc"], **{k: t[k]["acc"] for k in SUB}}


def ms(v, scale=100.0, signed=False):
    v = [scale * x for x in v if x is not None]
    if not v: return "—"
    f = "+.2f" if signed else ".2f"
    return f"{np.mean(v):{f}}" + (f" ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else "") + " [" + ", ".join(f"{x:{f}}" for x in v) + "]"


def selection():
    R = CFG["selection_rule"]; tols = R["requirements"]
    ref = ev("m2l_s17_ft", "dev")
    if ref is None: return ["(selection: seed-17 FT lr 1e-5 dev results missing)"], []
    L_ref = ref["law"]
    L = [f"L_ref (FT lr 1e-5, seed-17 dev Law) = {L_ref:.5f}", "",
         "| method | coefficient | GSM8K dev | Law dev | Law − L_ref | " + " | ".join(f"{k} (−{v})" for k, v in tols.items()) + " |",
         "|---|---|---|---|---|" + "---|" * len(tols)]
    rows, picks = [], {}
    for meth, cands in R["candidates"].items():
        D = {lab: ev(f"m2l_s17_{name}", "dev") for lab, name in cands.items()}
        orig = next(iter(cands))
        for lab, name in cands.items():
            d = D[lab]
            if d is None: L.append(f"| {meth} | {lab} | missing | | | |"); continue
            mg = d["law"] - L_ref
            L.append(f"| {meth} | {lab} | {d['gsm8k']:.3f} | {d['law']:.5f} | {mg:+.5f} | " +
                     " | ".join(("✓" if mg >= -t else "✗") + f" ({mg + t:+.5f})" for t in tols.values()) + " |")
            rows.append({"method": meth, "coefficient": lab, "run": name, "gsm8k_dev": d["gsm8k"], "law_dev": d["law"], "margin_vs_ref": mg})
        for rn, tol in tols.items():
            ok = {lab: d for lab, d in D.items() if d is not None and d["law"] >= L_ref - tol}
            if not ok: picks[(meth, rn)] = "INFEASIBLE"; continue
            best = max(d["gsm8k"] for d in ok.values())
            tied = [lab for lab, d in ok.items() if d["gsm8k"] >= best - 0.002 - 1e-12]
            pick = orig if orig in tied else max(tied, key=lambda l: (ok[l]["law"], float(l)))
            picks[(meth, rn)] = pick
    L += ["", "| requirement | " + " | ".join(R["candidates"]) + " |", "|---|" + "---|" * len(R["candidates"])]
    for rn in tols:
        L.append(f"| {rn} | " + " | ".join(str(picks.get((m, rn))) for m in R["candidates"]) + " |")
    return L, rows


def test_table():
    names = [("math_start", "Math-FT start")] + [(n, CFG["law_runs"][n]["label"]) for n in CFG["sets"]["main_test"]["law_runs"]] + \
            [("ft_ema0.999", "EMA 0.999 (eval-only)"), ("endpoint_b0.5", "Endpoint interpolation beta 0.5")]
    seeds = CFG["sets"]["main_test"]["seeds"] + CFG["sets"]["replication_test"]["seeds"]
    L = ["| configuration | seeds | GSM8K test | Law test | CaseHOLD | ToS | Overruling |", "|---|---|---|---|---|---|---|"]
    rows = []
    for n, lab in names:
        tag = (lambda s: f"math_ft_s{s}") if n == "math_start" else (lambda s, n=n: f"m2l_s{s}_{n}")
        v = {s: ev(tag(s), "test") for s in seeds}; v = {s: x for s, x in v.items() if x}
        if not v: continue
        for s, x in v.items(): rows.append({"configuration": lab, "seed": s, **x})
        L.append(f"| {lab} | {','.join(map(str, v))} | {ms([x['gsm8k'] for x in v.values()])} | {ms([x['law'] for x in v.values()])} | "
                 + " | ".join(ms([x[k] for x in v.values()]) for k in SUB) + " |")
    return L, rows


def paired():
    pairs = [("sfamatch_e50_p25", "look_b50_p25", "SFA − Lookahead"), ("sfamatch_e50_p25", "ft_lr5e-6", "SFA − FT (lr 5e-6)"),
             ("look_b50_p25", "ft_lr5e-6", "Lookahead − FT (lr 5e-6)")]
    groups = [("seeds 17, 42, 123", CFG["sets"]["main_test"]["seeds"]), ("additional seeds 7, 2026", CFG["sets"]["replication_test"]["seeds"]),
              ("all five seeds", CFG["sets"]["main_test"]["seeds"] + CFG["sets"]["replication_test"]["seeds"])]
    L, rows = [], []
    for gname, seeds in groups:
        L += ["", f"### {gname} (points; mean ± sample SD [per seed])", "", "| comparison | GSM8K (old task) | Law (new task) |", "|---|---|---|"]
        for a, b, lab in pairs:
            dE, dL = [], []
            for s in seeds:
                x, y = ev(f"m2l_s{s}_{a}", "test"), ev(f"m2l_s{s}_{b}", "test")
                if x and y:
                    dE.append(x["gsm8k"] - y["gsm8k"]); dL.append(x["law"] - y["law"])
                    rows.append({"group": gname, "comparison": lab, "seed": s, "d_gsm8k": x["gsm8k"] - y["gsm8k"], "d_law": x["law"] - y["law"]})
            L.append(f"| {lab} | {ms(dE, signed=True)} | {ms(dL, signed=True)} |")
    return L, rows


def figure():
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    pts = [("sfamatch_e50_p25", "SFA β_end=.5", "#2a78d6", "o", True), ("sfamatch_e25_p25", "SFA β_end=.25", "#2a78d6", "o", False),
           ("look_b50_p25", "Lookahead β=.5", "#eb6834", "s", True), ("ft_lr5e-6", "FT lr 5e-6", "#1baf7a", "^", True),
           ("ft", "FT lr 1e-5", "#1baf7a", "^", False), ("endpoint_b0.5", "endpoint β=.5", "#6f6e69", "D", True),
           ("ft_ema0.999", "EMA .999", "#6f6e69", "P", True)]
    seeds = CFG["sets"]["main_test"]["seeds"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=200)
    start = [ev(f"math_ft_s{s}", "test") for s in seeds]
    if all(start): ax.axhline(np.mean([x["gsm8k"] for x in start]), color="#6f6e69", lw=1, ls="--", label="Math-FT start (GSM8K)")
    for n, lab, c, mk, filled in pts:
        v = [ev(f"m2l_s{s}_{n}", "test") for s in seeds]
        if not all(v): continue
        g, l = np.array([x["gsm8k"] for x in v]), np.array([x["law"] for x in v])
        ax.errorbar(l.mean(), g.mean(), xerr=l.std(ddof=1), yerr=g.std(ddof=1), fmt="none", ecolor=c, elinewidth=1.2)
        ax.plot(l.mean(), g.mean(), marker=mk, ms=8, mec=c, mfc=c if filled else "white", mew=2, ls="none", label=lab)
    ax.set_xlabel("Law test accuracy (new task)"); ax.set_ylabel("GSM8K test accuracy (old task)")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.grid(True, color="#e6e5e0", lw=0.6); ax.legend(fontsize=7, frameon=False); fig.tight_layout()
    fig.savefig(OUT / "full_param_tradeoff_test.pdf"); fig.savefig(OUT / "full_param_tradeoff_test.png")


def write(name, lines, rows):
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")
    if rows:
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("full_param_selection", selection), ("full_param_test", test_table), ("full_param_paired", paired)):
        lines, rows = fn(); write(name, [f"# {name}", ""] + lines, rows); print("\n".join([f"## {name}"] + lines), "\n")
    figure()


if __name__ == "__main__":
    main()
