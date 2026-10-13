#!/usr/bin/env python3
"""Does history help most where a memory resembles the person's other memories?
typicality_i = mean cosine(memory i, the person's other memories) - mean cosine(memory i, other people's memories in the batch),
on mem and why embeddings (all-mpnet-base-v2, from style_ceiling.py; same row order as the data file).
gain_i = history - none on the 10 cheap models (mean of three judges, averaged over models). Spearman across memories; bootstrap over people."""
import json, os, sys, collections
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P, figs_models as FM
ROOT = FM.ROOT
rows = P.rows(); E = {f: np.load(os.path.join(ROOT, "analysis", f"emb_{f}.npy")) for f in ("mem", "why")}
idx = {r["perspectiveID"]: i for i, r in enumerate(rows)}
typ = {}
for i, r in enumerate(rows):
    own = [j for j, x in enumerate(rows) if x["selfPID"] == r["selfPID"] and j != i]
    oth = [j for j, x in enumerate(rows) if x["batchID"] == r["batchID"] and x["selfPID"] not in (r["selfPID"], r["otherPID"])]
    if own and oth:
        typ[r["perspectiveID"]] = {f: float((E[f][own] @ E[f][i]).mean() - (E[f][oth] @ E[f][i]).mean()) for f in E}
S, _ = FM.load("main_text_cheap")
cheap = sorted({k[0] for k in S})
res = {}
for m in ("similarity", "same_focus", "same_reason", "event_match"):
    g = {}
    for pid in typ:
        d = []
        for mdl in cheap:
            x, y = S.get((mdl, "text", "none", pid)), S.get((mdl, "text", "history", pid))
            if x and y:
                u, v = FM.avg2(x, m), FM.avg2(y, m)
                if u is not None and v is not None:
                    d.append(v - u)
        if len(d) >= 8:
            g[pid] = np.mean(d)
    for f in ("mem", "why"):
        ks = sorted(g); a = np.array([typ[k][f] for k in ks]); b = np.array([g[k] for k in ks])
        rho = spearmanr(a, b).correlation
        people = collections.defaultdict(list)
        for n, k in enumerate(ks): people[rows[idx[k]]["selfPID"]].append(n)
        P_ = list(people); rng = np.random.default_rng(0); bs = []
        for _ in range(1000):
            take = np.concatenate([people[P_[t]] for t in rng.integers(0, len(P_), len(P_))])
            bs.append(spearmanr(a[take], b[take]).correlation)
        q = np.array_split(np.argsort(a), 4)
        res[f"{m}|{f}"] = dict(n=len(ks), rho=float(rho), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                               gain_by_typicality_quartile=[float(b[ix].mean()) for ix in q])
        print(f"{m:12s} {f}: rho {rho:+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  gain by quartile " + " ".join(f"{b[ix].mean():+.3f}" for ix in q))
json.dump(res, open(os.path.join(ROOT, "analysis", "history_mechanism.json"), "w"), indent=1)
