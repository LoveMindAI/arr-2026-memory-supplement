#!/usr/bin/env python3
"""AI readers on the same measures agent2 used for the human six-trait wave: per-domain Spearman between reader domain scores (mean of
the four facets the reader returned) and the canonical HEXACO domain means, and mean absolute gap vs a leave-one-out population-mean
baseline. On the 19 random wave-2 profiles and on all lcw150 narrative cards. Readers: DeepSeek, Luna (cards/reads_api), writer lcw150.
-> analysis/ai_readers_rank.json"""
import csv, glob, json, os
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TR = "<LOCAL>/Documents/Master_Prompt_Workshop/LSI_StoriesOfYourLife_Canonical_PARSEL_Data/PARSEL_GroundTruth/V6_Canonical_290_Private_Human_Data/canonical_scale_scores_290.tsv"
T = {r["globalPID"]: r for r in csv.DictReader(open(TR), delimiter="\t")}
DOM = {"HH": "H_", "E": "E_", "EX": "X_", "A": "A_", "C": "C_", "OP": "O_"}
sel = json.load(open(os.path.join(ROOT, "prolific", "six_trait_wave2_20261009", "SELECTION_PRIVATE.json")))
rand = [r["pid"] for r in sel["records"]]   # the 19 uniformly sampled profiles (forced C01, C05 are listed separately)
out = {"n_random": len(rand)}
R = {}
for f in glob.glob(os.path.join(ROOT, "cards", "reads_api", "*", "*.json")):
    x = json.load(open(f)); rd = os.path.basename(os.path.dirname(f))
    if x["writer"] != "lcw150": continue
    R.setdefault(rd, {})[x["pid"]] = {d: float(np.mean([v for k, v in x["facets"].items() if k.startswith(p)])) for d, p in DOM.items()}
for rd, P in R.items():
    for name, pids in (("random19", [p for p in rand if p in P]), ("all", [p for p in P if p in T])):
        rho = {}; gap = []; base = []
        for d in DOM:
            y = [float(T[p][f"HEXACO_{d}_mean"]) for p in pids]; x = [P[p][d] for p in pids]
            rho[d] = round(float(spearmanr(x, y).correlation), 2)
            allv = [float(T[q][f"HEXACO_{d}_mean"]) for q in T if T[q].get(f"HEXACO_{d}_mean")]
            for p, xi, yi in zip(pids, x, y):
                gap.append(abs(xi - yi)); base.append(abs((sum(allv) - yi) / (len(allv) - 1) - yi))
        out[f"{rd}|{name}"] = {"n": len(pids), "rho": rho, "mean_rho": round(float(np.mean(list(rho.values()))), 2), "gap": round(float(np.mean(gap)), 2), "pop_mean_gap": round(float(np.mean(base)), 2)}
json.dump(out, open(os.path.join(ROOT, "analysis", "ai_readers_rank.json"), "w"), indent=1)
print(json.dumps(out, indent=0))
