"""Build the Math and Law train/dev/test files (full-parameter Math->Law experiments).

Outputs data/processed/math/{train,dev,test}.jsonl, data/processed/law/train.jsonl, data/processed/law/<subtask>_{dev,test}.jsonl
and data/processed/manifest.json (dataset revisions, counts, sha256s, contamination checks). Every record has a stable `uid`.

Sources (revisions pinned in configs/models.json):
  Math : MetaMathQA (train) and GSM8K. Dev = 500 seeded GSM8K-train questions; every MetaMathQA row derived from a dev question,
         exact-matching a GSM8K test question, or sharing a 13-gram with GSM8K test is removed from training. Test = GSM8K test (1319).
  Law  : three public multiple-choice tasks, each record {query, choices, gold} with a fixed task instruction prepended to the text:
         CaseHOLD   LexGLUE case_hold (coastalcph/lex_glue): official train / validation / test.
         ToS        sentence-level fair/unfair clauses from online Terms of Service (LawInformedAI/claudette_tos, 9,414 rows).
         Overruling overruling / not-overruling sentences (LawInformedAI/overruling, 2,400 rows).
         For ToS and Overruling the first 75% of rows (dataset order) are train and the rest test; test queries end with
         the "Options: ... Answer:" block of the task's evaluation format, which reproduces the reported test sets exactly.
         Dev: CaseHOLD validation; for ToS and Overruling a label-stratified 5% of train grouped by normalized query.
         Train rows whose normalized query appears in dev or test are removed.
All downloads happen here; the repository itself contains no data.
"""
import ast, collections, hashlib, json, random, re

from sfa.formatting import MODELS, ROOT

OUT = ROOT / "data/processed"
SEED = 20261002  # split seed (fixed, independent of training seeds)
CASEHOLD_PREFIX = ("Given an excerpt from a court decision, containing a reference to a particular case, select the correct holding "
                   "statement from the following options. Excerpt: ")
TOS_PREFIX = ("Given contractual terms, determine if they are fair or if they are unfair. Article 3 of the Directive 93/13 on Unfair Terms "
              "in Consumer Contracts defines an unfair contractual term as follows. A contractual term is unfair if: (1) it has not been "
              "individually negotiated; and (2) contrary to the requirement of good faith, it causes a significant imbalance in the parties "
              "rights and obligations, to the detriment of the consumer. Sentence: ")
OVERRULING_PREFIX = ("Given a sentence, determine if it is not overruling a prior decision, or if it is. In law, an overruling sentence is a "
                     "statement that nullifies a previous case decision as a precedent, by a constitutionally valid statute or a decision by "
                     "the same or higher ranking court which establishes a different rule on the point of law involved. Sentence: ")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"path": str(path.relative_to(ROOT)), "n": len(rows), "sha256": sha(path)}


def ngrams(text, n=13):
    w = norm(text).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def stratified_holdout(rows, frac, key_fn, label_fn, rng):
    """Group by normalized query so duplicates fall on one side; stratify by label."""
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[key_fn(r)].append(i)
    by_label = collections.defaultdict(list)
    for k, idx in groups.items():
        by_label[label_fn(rows[idx[0]])].append(k)
    dev_keys = set()
    for lab in sorted(by_label):
        ks = sorted(by_label[lab])
        rng.shuffle(ks)
        dev_keys.update(ks[: max(1, int(round(frac * len(ks))))])
    dev = [r for r in rows if key_fn(r) in dev_keys]
    train = [r for r in rows if key_fn(r) not in dev_keys]
    return train, dev


def casehold_split(split, rev):
    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", "case_hold", split=split, revision=rev)
    return [{"query": CASEHOLD_PREFIX + r["context"], "choices": list(r["endings"]), "gold": int(r["label"])} for r in ds]


def binary_split(rid, rev, text_col, prefix, choices):
    """First 75% of rows -> train, rest -> test (dataset order); test queries carry the evaluation-file suffix."""
    from datasets import load_dataset
    ds = load_dataset(rid, revision=rev, split="train")
    n_train = int(0.75 * len(ds))
    suffix = "\nOptions:" + "".join(f"\n - {c}" for c in choices) + "\nAnswer: "
    rec = lambda i, suf: {"query": prefix + str(ds[i][text_col]) + suf, "choices": list(choices), "gold": int(ds[i]["label"])}
    return {"train": [rec(i, "") for i in range(n_train)], "test": [rec(i, suffix) for i in range(n_train, len(ds))]}


def main():
    from datasets import load_dataset
    rev = MODELS["datasets"]
    man = {"split_seed": SEED, "dataset_revisions": rev, "files": {}, "checks": {}}
    rng = random.Random(SEED)

    # ---------------- Math ----------------
    mm = load_dataset("meta-math/MetaMathQA", revision=rev["meta-math/MetaMathQA"], split="train")
    gsm_tr = load_dataset("openai/gsm8k", "main", revision=rev["openai/gsm8k"], split="train")
    gsm_te = load_dataset("openai/gsm8k", "main", revision=rev["openai/gsm8k"], split="test")
    gold = lambda a: a.split("####")[-1].strip().replace(",", "")
    test = [{"uid": f"gsm8k_test_{i}", "question": r["question"], "answer": gold(r["answer"])}
            for i, r in enumerate(gsm_te)]
    tr_idx = list(range(len(gsm_tr)))
    rng.shuffle(tr_idx)
    dev_idx = sorted(tr_idx[:500])
    dev = [{"uid": f"gsm8k_train_{i}", "question": gsm_tr[i]["question"], "answer": gold(gsm_tr[i]["answer"])}
           for i in dev_idx]
    dev_norm = {norm(r["question"]) for r in dev}
    test_norm = {norm(r["question"]) for r in test}
    test_grams = set().union(*(ngrams(r["question"]) for r in test))
    dev_grams = set().union(*(ngrams(r["question"]) for r in dev))
    train, drop = [], collections.Counter()
    for i, r in enumerate(mm):
        oq, q = norm(r["original_question"]), norm(r["query"])
        if oq in dev_norm or q in dev_norm:
            drop["origin_is_dev_question"] += 1; continue
        if oq in test_norm or q in test_norm:
            drop["exact_match_gsm8k_test"] += 1; continue
        g = ngrams(r["query"])
        if g & test_grams:
            drop["13gram_overlap_gsm8k_test"] += 1; continue
        if r["type"].startswith("GSM") and (g & dev_grams):
            drop["13gram_overlap_dev"] += 1; continue
        train.append({"uid": f"metamathqa_{i}", "prompt": r["query"], "response": r["response"], "type": r["type"]})
    man["checks"]["math"] = {"metamathqa_total": len(mm), "dropped": dict(drop),
                             "dev_source": "500 GSM8K-train questions (seeded); all MetaMathQA rows whose original_question is a dev question removed from training",
                             "test": "GSM8K main test (1319)"}
    man["files"]["math/train"] = write(OUT / "math/train.jsonl", train)
    man["files"]["math/dev"] = write(OUT / "math/dev.jsonl", dev)
    man["files"]["math/test"] = write(OUT / "math/test.jsonl", test)

    # ---------------- Law ----------------
    def fix(r, sub, i, split):
        ch = r["choices"] if isinstance(r["choices"], list) else ast.literal_eval(r["choices"])
        return {"uid": f"{sub}_{split}_{i}", "subtask": sub, "query": r["query"], "choices": list(ch), "gold": int(r["gold"])}
    raw = {"casehold": {s: casehold_split(s, rev["coastalcph/lex_glue"]) for s in ("train", "test", "validation")},
           "tos": binary_split("LawInformedAI/claudette_tos", rev["LawInformedAI/claudette_tos"], "text", TOS_PREFIX, ["fair", "unfair"]),
           "overruling": binary_split("LawInformedAI/overruling", rev["LawInformedAI/overruling"], "sentence1", OVERRULING_PREFIX,
                                      ["not overruling", "overruling"])}
    law = {}
    for sub, dev_f in [("casehold", "validation"), ("tos", None), ("overruling", None)]:
        tr = [fix(r, sub, i, "train") for i, r in enumerate(raw[sub]["train"])]
        te = [fix(r, sub, i, "test") for i, r in enumerate(raw[sub]["test"])]
        if dev_f:
            dv = [fix(r, sub, i, "dev") for i, r in enumerate(raw[sub][dev_f])]
            src = "official validation split"
        else:
            tr, dv = stratified_holdout(tr, 0.05, lambda r: norm(r["query"]), lambda r: r["gold"], rng)
            src = "label-stratified 5% of train, grouped by normalized query"
        dvn, ten = {norm(r["query"]) for r in dv}, {norm(r["query"]) for r in te}
        n0 = len(tr)
        tr = [r for r in tr if norm(r["query"]) not in dvn and norm(r["query"]) not in ten]
        dv = [r for r in dv if norm(r["query"]) not in ten]
        law[sub] = (tr, dv, te)
        man["checks"][f"law_{sub}"] = {"dev_source": src, "train_removed_overlap_dev_or_test": n0 - len(tr),
                                       "label_counts": {s: dict(collections.Counter(r["gold"] for r in x)) for s, x in
                                                        [("train", tr), ("dev", dv), ("test", te)]},
                                       "n_choices": sorted({len(r["choices"]) for r in tr + dv + te})}
        man["files"][f"law/{sub}_dev"] = write(OUT / f"law/{sub}_dev.jsonl", dv)
        man["files"][f"law/{sub}_test"] = write(OUT / f"law/{sub}_test.jsonl", te)
    man["files"]["law/train"] = write(OUT / "law/train.jsonl", [r for s in law for r in law[s][0]])

    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    print(json.dumps(man["checks"], indent=1))
    print(json.dumps({k: v["n"] for k, v in man["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
