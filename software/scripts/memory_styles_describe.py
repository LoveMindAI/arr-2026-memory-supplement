#!/usr/bin/env python3
"""Follow-up to memory_styles.py: what the profiles are made of. PCA on the person profiles (loadings of the first three components,
variance explained, against the shuffle null), k=2 and k=3 cluster means in raw units, k=2 again WITHOUT the two word-count features
(is it just length?), and cluster/PC relations with traits, social anxiety, age, sex, education (BH-corrected).
-> analysis/memory_styles_describe.json"""
import json, os, sys, csv, collections
import numpy as np
src = open("memory_styles.py").read().split("# (2) person profiles")[0].replace("B = 500", "B = 100")
exec(src)
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, kruskal
def profiles(labels, Xr):
    ps = collections.defaultdict(list)
    for i, p in enumerate(labels): ps[p].append(i)
    keep = sorted(p for p, ix in ps.items() if len(ix) >= 3)
    M = np.array([np.nanmean(Xr[ps[p]], axis=0) for p in keep]); col = np.nanmean(M, axis=0); M = np.where(np.isnan(M), col, M)
    sd = M.std(axis=0); sd[sd == 0] = 1
    return keep, M, (M - M.mean(axis=0)) / sd
keep, M, Z = profiles(person, X)
res = {}
pca = PCA(5).fit(Z); ev = pca.explained_variance_ratio_
nev = np.array([PCA(5).fit(profiles(p, X)[2]).explained_variance_ratio_ for p in nulls[:100]])
res["pca_var"] = [round(float(x), 3) for x in ev]; res["pca_var_null95"] = [round(float(x), 3) for x in np.percentile(nev, 95, axis=0)]
res["loadings"] = {}
for c in range(3):
    L = sorted(zip(NAMES, pca.components_[c]), key=lambda t: -abs(t[1]))[:9]
    res["loadings"][f"PC{c+1}"] = [(n, round(float(v), 2)) for n, v in L]
S = pca.transform(Z)[:, :3]
def show(labels, tag):
    d = {}
    for k in sorted(set(labels)):
        m = M[labels == k].mean(axis=0); d[int(k)] = {"n": int((labels == k).sum()), **{n: round(float(v), 2) for n, v in zip(NAMES, m)}}
    res[tag] = d
for k in (2, 3):
    show(KMeans(k, n_init=20, random_state=0).fit(Z).labels_, f"k{k}_means")
nw = [i for i, n in enumerate(NAMES) if not n.startswith("words_")]
km2 = KMeans(2, n_init=20, random_state=0).fit(Z[:, nw]); show(km2.labels_, "k2_nowords_means")
res["k2_nowords_vs_k2_ari"] = round(float(adjusted_rand_score(km2.labels_, KMeans(2, n_init=20, random_state=0).fit(Z).labels_)), 3)
res["k2_nowords_silhouette"] = round(float(silhouette_score(Z[:, nw], km2.labels_)), 3)
# traits
TR = "<LOCAL>/Documents/Master_Prompt_Workshop/LSI_StoriesOfYourLife_Canonical_PARSEL_Data/PARSEL_GroundTruth/V6_Canonical_290_Private_Human_Data/canonical_scale_scores_290.tsv"
T = {r["globalPID"]: r for r in csv.DictReader(open(TR), delimiter="\t")}
DEM = {r["PID"] if "PID" in r else None: r for r in []}
glob_of = {r["selfPID"]: r.get("globalPID") for r in rows.values()}
VARS = ["HEXACO_HH_mean", "HEXACO_E_mean", "HEXACO_EX_mean", "HEXACO_A_mean", "HEXACO_C_mean", "HEXACO_OP_mean", "SIAS_mean"]
tests = []
for v in VARS:
    x = []
    for p in keep:
        t = T.get(glob_of.get(p) or p)
        try: x.append(float(t[v]))
        except Exception: x.append(np.nan)
    x = np.array(x); ok = ~np.isnan(x)
    for c in range(3):
        r = spearmanr(x[ok], S[ok, c]); tests.append([v, f"PC{c+1}", round(float(r.correlation), 3), float(r.pvalue), int(ok.sum())])
    g = km2.labels_
    kw = kruskal(x[ok & (g == 0)], x[ok & (g == 1)]); tests.append([v, "k2_nowords", None, float(kw.pvalue), int(ok.sum())])
ps_ = np.array([t[3] for t in tests]); o = np.argsort(ps_); q = np.empty(len(ps_)); prev = 1.0; m = len(ps_)
for rank, i in reversed(list(enumerate(o, 1))): prev = min(prev, ps_[i] * m / rank); q[i] = prev
res["traits"] = [t[:3] + [round(t[3], 4), round(float(q[i]), 4), t[4]] for i, t in enumerate(tests)]
json.dump(res, open(os.path.join(ROOT, "analysis", "memory_styles_describe.json"), "w"), indent=1)
print("PCA var", res["pca_var"], "null95", res["pca_var_null95"])
for c, L in res["loadings"].items(): print(c, L)
for tag in ("k2_means", "k2_nowords_means"):
    print(tag); [print("  ", k, {n: v for n, v in d.items() if n in ("n", "words_moment", "words_why", "details", "scope:one_moment", "scope:stretch", "scope:not_an_event", "what:partner_fact_or_quality", "what:did_together", "commentary", "feeling_named", "shared:nobody", "new_fact", "matters:connection")}) for k, d in res[tag].items()]
print("k2 without words vs with: ARI", res["k2_nowords_vs_k2_ari"], "silhouette", res["k2_nowords_silhouette"])
print("traits q<.10:", [t for t in res["traits"] if t[4] < .10]); print("smallest q", min(t[4] for t in res["traits"]))
