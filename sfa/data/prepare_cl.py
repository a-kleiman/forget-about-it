"""Standard continual-learning benchmark (DBpedia, Amazon, Yahoo, AG News) for the adapter-level experiments.

Data: the publicly released Recurrent_KIF files (data_longsequence/{train,dev,test}/<task>.json, Alpaca-style
instruction/input/output; 1,000 training examples per task), downloaded at a pinned commit and checked against sha256.
Dev = released dev minus any example whose text also appears in test (the released dev and test overlap).
Outputs data/processed/cl_<task>/{train,dev,test}.jsonl and data/processed/cl_manifest.json.

  python -m sfa.data.prepare_cl            # download (if missing) + verify + process
"""
import hashlib, json, urllib.request

from sfa.formatting import ROOT

TASKS = ["dbpedia", "amazon", "yahoo", "agnews"]
ORDERS = {1: ["dbpedia", "amazon", "yahoo", "agnews"],
          4: ["agnews", "yahoo", "amazon", "dbpedia"]}  # order 4 = exact reverse of order 1 (robustness check)
SRC_MUL = 10_000_000   # global example id = position of the task in the order * SRC_MUL + row index

UPSTREAM = "https://raw.githubusercontent.com/WoodScene/Recurrent_KIF/adc7f1e562add730f52319b24f40ad917dea6a7b/data_longsequence"
SHA256 = {
    "dev/agnews.json": "72a30011ca96ca4aacf1a64a95b6aaa88acf61c28b940b2d8944b7921ec66467",
    "dev/amazon.json": "e7b7246388d51e22671528dbc300a72846d5eff4c32eedbad6b86ed25ff22c14",
    "dev/dbpedia.json": "d653cc31ac64e74ba4cbd7a357e7bbc52bb2586c3121290dd267fbcb11a0a56e",
    "dev/yahoo.json": "9f1bbe02fd42df2a4f1950cf894dae638c0d87cc93e686a4b3672775310833fa",
    "test/agnews.json": "035cac222331ed08166d945cd3a91fcdb27debbd98d745b08e324a74093de000",
    "test/amazon.json": "6300a45b8bf8c6fe235daa93042f8749c004432865d8f6284d1b3cb6346e287d",
    "test/dbpedia.json": "5c17b22e2bea002ab8884773515a4c1cd7511d538cf7a6003384ce00c24e83b3",
    "test/yahoo.json": "ac26220118d12c8f4d18815cae4b4cf1f3dc85c6e4fb51161fcc27e7773e1dac",
    "train/agnews.json": "583845cf3be385c419910edf91b1630e525bec3fa812ced0a9917788570d6608",
    "train/amazon.json": "9db982c16a0e49ee99c371d1e95f258feba0130fb1463ebb63a779c73cecac92",
    "train/dbpedia.json": "c3bc4387f0b2d43f744af1cae92a441cb98fcd1515ce089988ed9d2370e14dd0",
    "train/yahoo.json": "ca0fd17026027d2a5053a119455d491e54f9dede9654a9088aa9902887ab7f84",
}


def download():
    raw = ROOT / "data/raw/cl_standard"
    for rel, want in SHA256.items():
        p = raw / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{UPSTREAM}/{rel}", p)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert got == want, f"{rel}: sha256 {got} != {want}"
    print("continual-learning source files verified")


def prep():
    raw = ROOT / "data/raw/cl_standard"
    man = {}
    for t in TASKS:
        tr, dv, te = (json.load(open(raw / s / f"{t}.json")) for s in ("train", "dev", "test"))
        key = lambda r: (r["instruction"] + r["input"]).strip()
        tek = {key(r) for r in te}
        n_dev = len(dv)
        dv = [r for r in dv if key(r) not in tek]
        trk = {key(r) for r in tr}
        conv = lambda rs, split: [{"uid": f"cl_{t}_{split}_{r['id']}", "prompt": r["instruction"] + r["input"], "response": r["output"],
                                   "label": r["output"]} for r in rs]
        out = ROOT / f"data/processed/cl_{t}"; out.mkdir(parents=True, exist_ok=True)
        for split, rs in [("train", tr), ("dev", dv), ("test", te)]:
            with open(out / f"{split}.jsonl", "w") as f:
                for r in conv(rs, split):
                    f.write(json.dumps(r) + "\n")
        man[t] = {"train": len(tr), "dev_released": n_dev, "dev_clean": len(dv), "test": len(te),
                  "train_in_test": len(trk & tek), "labels": sorted({r["output"] for r in te})}
    json.dump(man, open(ROOT / "data/processed/cl_manifest.json", "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    download()
    prep()
