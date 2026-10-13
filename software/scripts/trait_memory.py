#!/usr/bin/env python3
"""Do traits show up in what/how people remember? (human data only; exploratory)
Person-level memory features (conv memories; people with >= 3) vs canonical scale scores (290):
  A) feature-trait Spearman correlations, Benjamini-Hochberg FDR across all tests
  B) ridge regression from person-mean memory embeddings (mem + why) to each trait, 10-fold CV, permutation null (200)
Features come from the blind style-stripped codes (analysis/coded/, Gemini 3.8 Flash coder; fine for human-only analysis)
plus ratings (timing slider, pleasure/arousal/dominance) and lengths."""
import csv, glob, json, os
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import spearmanr
import style_ceiling as sc

SCORES = "<LOCAL>/Documents/Master_Prompt_Workshop/LSI_StoriesOfYourLife_Canonical_PARSEL_Data/PARSEL_GroundTruth/V6_Canonical_290_Private_Human_Data/canonical_scale_scores_290.tsv"
TRAITS = ["HEXACO_HH_mean", "HEXACO_E_mean", "HEXACO_EX_mean", "HEXACO_A_mean", "HEXACO_C_mean", "HEXACO_OP_mean", "SIAS_mean",
          "Trust_Propensity_mean", "Trust_Benevolence_mean", "Trust_Ability_mean", "Trust_Integrity_mean",
          "PPTS_Affective_Responsiveness_mean", "PPTS_Cognitive_Responsiveness_mean", "PPTS_Interpersonal_Manipulation_mean", "PPTS_Egocentricity_mean"]


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def main():
    rows = sc.load()
    C = {json.load(open(f))["i"]: json.load(open(f))["code"] for f in glob.glob(os.path.join(sc.OUT, "coded", "*.json"))}
    S = {r["globalPID"]: r for r in csv.DictReader(open(SCORES), delimiter="\t")}
    byp = defaultdict(list)
    for i, r in enumerate(rows):
        byp[r["p"]].append(i)
    people = [p for p, ix in byp.items() if len(ix) >= 3 and p in S]
    print("people with >=3 memories and scores:", len(people))
    cats = {f: sorted({str(C[i].get(f, "")).split(" ")[0].lower() for i in C}) for f in ("kind", "focus", "reason_type", "specificity", "valence")}
    feats = {}
    for p in people:
        ix = byp[p]; v = {}
        for f, labs in cats.items():
            cnt = Counter(str(C[i].get(f, "")).split(" ")[0].lower() for i in ix)
            for l in labs:
                v[f"{f}={l}"] = cnt[l] / len(ix)
        for k in ("timing", "a", "d", "pl"):
            vals = [rows[i][k] for i in ix if not np.isnan(rows[i][k])]
            v[{"a": "arousal", "d": "dominance", "pl": "pleasure"}.get(k, k)] = float(np.mean(vals)) if vals else np.nan
        v["mem_words"] = float(np.mean([len(rows[i]["mem"].split()) for i in ix]))
        v["why_words"] = float(np.mean([len(rows[i]["why"].split()) for i in ix]))
        feats[p] = v
    fnames = [f for f in next(iter(feats.values())) if np.nanstd([feats[p][f] for p in people]) > 0 and
              np.mean([feats[p][f] > 0 for p in people]) > 0.05]
    tests = []
    for t in TRAITS:
        y = np.array([float(S[p][t]) for p in people])
        for f in fnames:
            x = np.array([feats[p][f] for p in people]); m = ~np.isnan(x) & ~np.isnan(y)
            rho, pv = spearmanr(x[m], y[m]); tests.append((t, f, rho, pv, int(m.sum())))
    q = bh([x[3] for x in tests])
    sig = sorted([(x, qq) for x, qq in zip(tests, q) if qq < 0.10], key=lambda z: z[1])
    print(f"\nA) {len(tests)} feature x trait tests; q<.10 after BH:")
    for (t, f, rho, pv, n), qq in sig:
        print(f"  {t:40s} {f:32s} rho={rho:+.2f} p={pv:.4f} q={qq:.3f} n={n}")
    if not sig:
        print("  none")
    top = sorted(tests, key=lambda x: x[3])[:12]
    print("\n  strongest 12 by p (uncorrected):")
    for t, f, rho, pv, n in top:
        print(f"  {t:40s} {f:32s} rho={rho:+.2f} p={pv:.4f}")
    # B) embedding ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    Em, Ew = np.load(os.path.join(sc.OUT, "emb_mem.npy")), np.load(os.path.join(sc.OUT, "emb_why.npy"))
    X = np.array([np.concatenate([Em[byp[p]].mean(0), Ew[byp[p]].mean(0)]) for p in people])
    rng = np.random.default_rng(0)
    print("\nB) predict trait from person-mean memory embeddings (mem+why), 10-fold CV ridge; r(pred, actual), permutation p (200)")
    outB = {}
    for t in TRAITS:
        y = np.array([float(S[p][t]) for p in people])
        mdl = RidgeCV(alphas=np.logspace(-1, 4, 20))
        kf = KFold(10, shuffle=True, random_state=0)
        r = np.corrcoef(cross_val_predict(mdl, X, y, cv=kf), y)[0, 1]
        null = [np.corrcoef(cross_val_predict(mdl, X, yy, cv=kf), yy)[0, 1] for yy in (rng.permutation(y) for _ in range(200))]
        pv = (1 + sum(n >= r for n in null)) / 201
        outB[t] = dict(r=float(r), p_perm=float(pv))
        print(f"  {t:40s} r={r:+.3f}  p_perm={pv:.3f}")
    json.dump(dict(n_people=len(people), A=[dict(trait=t, feature=f, rho=rho, p=pv, q=float(qq), n=n) for (t, f, rho, pv, n), qq in zip(tests, q)],
                   B=outB), open(os.path.join(sc.OUT, "trait_memory.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
