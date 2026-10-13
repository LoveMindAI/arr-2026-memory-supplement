#!/usr/bin/env python3
"""Memory styles (A1, Oct 10 evening: "We cluster the real humans by sort of memory types... something we should have done the
first time around anyway"). No API calls. Run with PYTHONHASHSEED=0.

Data: the adjudicated consensus codes of every real report (analysis/code_v351/human/FINAL.jsonl, codebook v3.5.1; TIE -> missing),
plus word counts of the moment and the reason.

(1) What does a person carry from call to call? For each report feature, the share of its variance that lies between people
    (one-way ICC on person means), against a null that shuffles reports among the people of the same session (keeping each person's
    number of reports and each session's set of reports). Reported for every feature, with the null's 95th percentile.
(2) Are there memory TYPES? People with >= 3 reports get a profile (mean of each feature over their reports, standardized).
    k-means for k = 2..6 (and a Gaussian mixture for comparison); silhouette against the same within-session shuffle null;
    stability = mean adjusted Rand index between the clustering of the full data and of profiles rebuilt from bootstrap-resampled
    reports within each person. Everything tried is reported, not only the best k.
(3) The chosen k is described in plain terms (feature means by cluster) and related to HEXACO, social anxiety, age, sex and
    education (Kruskal-Wallis / chi-square; Benjamini-Hochberg).
-> analysis/memory_styles.json"""
import scope_merge as SM, sys as _sys
import collections, csv, json, os, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import kruskal, chi2_contingency
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rows = {r["perspectiveID"]: r for r in P.rows()}
F = {}
for l in open(os.path.join(ROOT, "analysis", "code_v351", "human", "FINAL.jsonl")):
    j = json.loads(l); F[j["key"].split("|")[0]] = SM.codes(j["codes"])

def b(x): return np.nan if x is None else float(x)
def feats(c, row):
    f = {}
    ext = c.get("extent"); sh = c.get("shared_by"); pk = c.get("personal_kinds") or []; rem = set(c.get("remembered") or [])
    for v in SM.OPTIONS:
        f[f"scope:{v.lower()}"] = np.nan if ext in (None, "TIE") else float(ext == v)
    for v in ("NOBODY", "PARTNER", "BOTH"):
        f[f"shared:{v.lower()}"] = np.nan if sh in (None, "TIE") else float(sh == v)
    f["disclosed_anything"] = np.nan if not pk else float(pk != ["nothing"])
    f["disclosed_deep"] = np.nan if not pk else float(bool(set(pk) & {"info_revealing", "thoughts_intimate", "feelings_mild", "feelings_deep"}))
    ni = c.get("new_info"); f["new_fact"] = np.nan if ni in (None, "TIE") else float(ni == "YES")
    for v in ("did_together", "partner_said_or_did", "partner_fact_or_quality", "reporter_thought", "situation", "reporter_said_or_did"):
        f[f"what:{v}"] = float(v in rem) if rem else np.nan
    fe = c.get("feelings") or []; f["feeling_named"] = np.nan if not fe else float(fe != ["none"])
    it = c.get("intensity"); f["intense"] = float(int(it) >= 3) if str(it).isdigit() else np.nan
    mf = c.get("matters_for") or []; f["matters:connection"] = np.nan if not mf else float("connection" in mf)
    bm = c.get("bigger_meaning"); f["bigger_meaning"] = np.nan if bm in (None, "TIE") else float(bm == "YES")
    d = c.get("details"); f["details"] = float(d) if str(d).isdigit() else np.nan
    mt = c.get("moment_text"); f["commentary"] = np.nan if mt in (None, "TIE") else float(mt in ("THOUGHTS", "BOTH"))
    tp = set(c.get("topic") or [])
    for v in ("the_study_or_call", "partner_impression", "feelings_mood", "work_education_service", "place_travel"):
        f[f"topic:{v}"] = float(v in tp) if tp else np.nan
    rs = c.get("reason") or []
    if rs:
        firsts = [r["about"][0].strip("[]") for r in rs if r.get("about")]
        for v in ("ME", "PARTNER", "US", "CALL"):
            f[f"why:{v.lower()}"] = float(np.mean([x == v for x in firsts])) if firsts else np.nan
        dets = [set(x.strip("[]") for x in r.get("details", [])) for r in rs]
        for v in ("IN_COMMON", "UNEXPECTED"):
            f[f"why:{v.lower()}"] = float(np.mean([v in s for s in dets]))
    f["words_moment"] = float(np.log1p(len(str(row["conv_SP_Mem_descr"]).split())))
    f["words_why"] = float(np.log1p(len(str(row["conv_SP_Mem_reason"]).split())))
    return f

recs = [(pid, rows[pid]["selfPID"], rows[pid]["batchID"], feats(c, rows[pid])) for pid, c in F.items() if pid in rows]
NAMES = sorted(recs[0][3])
X = np.array([[r[3].get(n, np.nan) for n in NAMES] for r in recs]); person = np.array([r[1] for r in recs]); batch = np.array([r[2] for r in recs])
rng = np.random.default_rng(2026)

def icc(x, g):
    ok = ~np.isnan(x); x, g = x[ok], g[ok]
    groups = collections.defaultdict(list)
    for v, p in zip(x, g): groups[p].append(v)
    gs = [np.array(v) for v in groups.values() if len(v) >= 2]
    if len(gs) < 10: return np.nan
    k = np.mean([len(v) for v in gs]); grand = np.mean(np.concatenate(gs))
    msb = sum(len(v) * (v.mean() - grand) ** 2 for v in gs) / (len(gs) - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in gs) / (sum(len(v) for v in gs) - len(gs))
    return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else np.nan

def shuffle_within_session(lab):
    out = lab.copy()
    for bt in np.unique(batch):
        idx = np.where(batch == bt)[0]; out[idx] = lab[rng.permutation(idx)]
    return out

out = {"n_reports": len(recs), "n_people": len(set(person)), "features": NAMES, "carry": {}}
B = 500
nulls = [shuffle_within_session(person) for _ in range(B)]
for j, n in enumerate(NAMES):
    obs = icc(X[:, j], person); nl = np.array([icc(X[:, j], p) for p in nulls])
    out["carry"][n] = dict(icc=round(obs, 3), null_mean=round(float(np.nanmean(nl)), 3), null_95=round(float(np.nanpercentile(nl, 95)), 3),
                           p=round(float((np.sum(nl >= obs) + 1) / (B + 1)), 4), base_rate=round(float(np.nanmean(X[:, j])), 3))

# (2) person profiles
def profiles(labels, Xr):
    ps = collections.defaultdict(list)
    for i, p in enumerate(labels): ps[p].append(i)
    keep = sorted(p for p, ix in ps.items() if len(ix) >= 3)
    M = np.array([np.nanmean(Xr[ps[p]], axis=0) for p in keep])
    col = np.nanmean(M, axis=0); M = np.where(np.isnan(M), col, M)
    sd = M.std(axis=0); sd[sd == 0] = 1
    return keep, (M - M.mean(axis=0)) / sd, ps

keep, Z, ps = profiles(person, X)
out["n_profiled"] = len(keep); out["k"] = {}
null_Z = [profiles(p, X)[1] for p in nulls[:100]]
for k in range(2, 7):
    km = KMeans(k, n_init=20, random_state=0).fit(Z); sil = silhouette_score(Z, km.labels_)
    nsil = [silhouette_score(z, KMeans(k, n_init=10, random_state=0).fit(z).labels_) for z in null_Z]
    aris = []
    for bsi in range(50):
        Xb_idx = []; lab = []
        for p in keep:
            ix = ps[p]; pick = rng.choice(ix, len(ix)); Xb_idx += list(pick); lab += [p] * len(ix)
        _, Zb, _ = profiles(np.array(lab), X[np.array(Xb_idx)])
        aris.append(adjusted_rand_score(km.labels_, KMeans(k, n_init=10, random_state=0).fit(Zb).labels_))
    gm = GaussianMixture(k, covariance_type="diag", n_init=5, random_state=0).fit(Z)
    out["k"][k] = dict(silhouette=round(float(sil), 3), null_silhouette_mean=round(float(np.mean(nsil)), 3), null_silhouette_95=round(float(np.percentile(nsil, 95)), 3),
                       stability_ari=round(float(np.mean(aris)), 3), sizes=sorted(np.bincount(km.labels_).tolist(), reverse=True),
                       gmm_bic=round(float(gm.bic(Z)), 1), gmm_agree_with_kmeans_ari=round(float(adjusted_rand_score(km.labels_, gm.predict(Z))), 3))
json.dump(out, open(os.path.join(ROOT, "analysis", "memory_styles.json"), "w"), indent=1)
print("reports", out["n_reports"], "people", out["n_people"], "profiled", out["n_profiled"])
print("\n(1) carried from call to call (ICC vs within-session shuffle):")
for n, v in sorted(out["carry"].items(), key=lambda kv: -(kv[1]["icc"] if kv[1]["icc"] == kv[1]["icc"] else -9)):
    print(f"  {n:32s} icc {v['icc']:+.3f}  null95 {v['null_95']:+.3f}  p {v['p']:.3f}  base {v['base_rate']:.2f}")
print("\n(2) types:")
for k, v in out["k"].items(): print(" ", k, v)
