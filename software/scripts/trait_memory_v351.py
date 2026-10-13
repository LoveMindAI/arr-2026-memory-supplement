#!/usr/bin/env python3
"""Traits vs person-level memory style, recomputed on codebook v3.5.1 consensus codes (replaces trait_memory.py part A, which used
the v1 single-coder codes; part B, embeddings -> traits, uses no codes and stands as is).
Person features (people with >= 3 memories and scores): share of each yes/no code (analysis/srm_glmm/long.csv, consensus codes;
options chosen in 5-95% of reports), means of the numeric codes (details, intensity, appraisals), and PARSEL's own ratings of the
moment (timing slider, pleasure, arousal, dominance) plus report lengths. Spearman with each trait (HEXACO domains, SIAS, trust,
PPTS; PPTS scored so higher = more psychopathic traits), Benjamini-Hochberg FDR over all tests. -> analysis/trait_memory_v351.json"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_ceiling as sc
from trait_memory import TRAITS, SCORES, bh
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    L = list(csv.DictReader(open(os.path.join(ROOT, "analysis", "srm_glmm", "long.csv"))))
    outcomes = [k for k in L[0] if k.startswith(("b__", "n__"))]
    rows = sc.load(); S = {r["globalPID"]: r for r in csv.DictReader(open(SCORES), delimiter="\t")}
    byp = defaultdict(list); rat = defaultdict(list)
    for r in L: byp[r["rem"]].append(r)
    for r in rows: rat[r["p"]].append(r)
    people = [p for p in byp if len(byp[p]) >= 3 and p in S]
    feats = {}
    for p in people:
        v = {}
        for k in outcomes:
            vals = [float(r[k]) for r in byp[p] if r[k] != ""]
            v[k[3:]] = float(np.mean(vals)) if vals else np.nan
        for k, nm in (("timing", "rating_timing"), ("a", "rating_arousal"), ("d", "rating_dominance"), ("pl", "rating_pleasure")):
            vals = [x[k] for x in rat[p] if not np.isnan(x[k])]; v[nm] = float(np.mean(vals)) if vals else np.nan
        v["memory_words"] = float(np.mean([len(x["mem"].split()) for x in rat[p]])); v["reason_words"] = float(np.mean([len(x["why"].split()) for x in rat[p]]))
        feats[p] = v
    fnames = [f for f in feats[people[0]] if np.nanstd([feats[p][f] for p in people]) > 0]
    tests = []
    for t in TRAITS:
        y = np.array([float(S[p][t]) if S[p][t] not in ("", "nan") else np.nan for p in people])
        for f in fnames:
            x = np.array([feats[p][f] for p in people]); m = ~np.isnan(x) & ~np.isnan(y)
            rho, pv = spearmanr(x[m], y[m]); tests.append((t, f, float(rho), float(pv), int(m.sum())))
    q = bh([x[3] for x in tests])
    sig = sorted([(x, float(qq)) for x, qq in zip(tests, q) if qq < 0.05], key=lambda z: z[1])
    print(f"people {len(people)}; {len(tests)} feature x trait tests; q < .05 after BH: {len(sig)}")
    for (t, f, rho, pv, n), qq in sig: print(f"  {t:38s} {f:42s} rho={rho:+.2f} q={qq:.3f}")
    code_sig = [s for s in sig if not s[0][1].startswith(("rating_", "memory_words", "reason_words"))]
    print("survivors that are codebook features:", len(code_sig))
    json.dump(dict(n_people=len(people), n_tests=len(tests), survivors_q05=[dict(trait=t, feature=f, rho=rho, q=qq, n=n) for (t, f, rho, pv, n), qq in sig],
                   all=[dict(trait=t, feature=f, rho=rho, p=pv, q=float(qq), n=n) for (t, f, rho, pv, n), qq in zip(tests, q)]),
              open(os.path.join(ROOT, "analysis", "trait_memory_v351.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
