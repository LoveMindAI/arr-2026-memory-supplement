#!/usr/bin/env python3
"""Same decomposition as style_ceiling.py, on blind style-stripped codes (analysis/coded/)."""
import json, os, glob
import numpy as np
from collections import defaultdict
import style_ceiling as sc

def main():
    rows = sc.load()
    C = {json.load(open(f))["i"]: json.load(open(f))["code"] for f in glob.glob(os.path.join(sc.OUT, "coded", "*.json"))}
    assert len(C) == len(rows)
    out = {}
    E = sc.embed([C[i]["neutral"] for i in range(len(rows))]); S = E @ E.T
    out["neutral"] = dict(decomp=sc.decompose(rows, S, "neutral paraphrase"), ident=sc.identify(rows, S, "neutral paraphrase"))
    print("\nCategory agreement rate by pair type (same code = 1)")
    out["cats"] = {}
    for field in ("kind", "focus", "reason_type", "specificity", "valence"):
        lab = [str(C[i].get(field, "")).split(" ")[0].strip().lower() for i in range(len(rows))]
        S2 = np.array([[1.0 if a == b else 0.0 for b in lab] for a in lab])
        print(f"-- {field}")
        out["cats"][field] = sc.decompose(rows, S2, field)
    # combined categorical profile: one-hot of kind+focus+reason_type+specificity, cosine
    fields = ("kind", "focus", "reason_type", "specificity")
    vocab = sorted({(f, str(C[i].get(f, "")).split(" ")[0].lower()) for i in range(len(rows)) for f in fields})
    ix = {v: k for k, v in enumerate(vocab)}
    M = np.zeros((len(rows), len(vocab)))
    for i in range(len(rows)):
        for f in fields:
            M[i, ix[(f, str(C[i].get(f, "")).split(" ")[0].lower())]] = 1
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    S3 = M @ M.T
    out["profile_ident"] = sc.identify(rows, S3, "category profile (kind+focus+reason+specificity)")
    from collections import Counter
    for f in fields:
        print(f, Counter(str(C[i].get(f, "")).split(" ")[0].lower() for i in range(len(rows))).most_common(10))
    json.dump(out, open(os.path.join(sc.OUT, "style_ceiling_coded.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
