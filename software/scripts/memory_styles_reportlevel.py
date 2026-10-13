#!/usr/bin/env python3
"""Report-level version of the type-match test (fixes memory_styles_models.py's donor_type_match, which is invalid: in the *_wrong
conditions each of a target's calls has its own donor, so a person-level profile mixes donors). No API calls. PYTHONHASHSEED=0.
Each single report (people's and every model prediction) is projected onto the human first principal axis (PC1, the axis that separates
"the stretch we shared" from "what they told me"; fitted on person profiles, standardized on people). For each condition, per model:
  own:   Spearman between the model's report PC1 and the TARGET person's profile PC1 (computed from their other reports only, so the
         target report never informs the yardstick);
  donor: in *_wrong conditions, Spearman with the DONOR's profile PC1 (the person whose history/card the model was given, per report).
People's own report-vs-other-reports correlation is the ceiling. Bootstrap over target people. -> analysis/memory_styles_reportlevel.json"""
import collections, json, os, sys
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "memory_styles_models.py")).read()
exec(src.split("rng = np.random.default_rng(2026)")[0])   # people features, NAMES, mu, sd, pca, CONDS, coded(), featvec(), rows
import prompts as PP
rng = np.random.default_rng(2026)
def proj(v):
    x = np.array([v.get(n, np.nan) for n in NAMES]); x = np.where(np.isnan(x), mu, x); return float(pca.transform(((x - mu) / sd)[None])[0, 0])
by_person = collections.defaultdict(list)
for pid, sp, bt, v in P_: by_person[sp].append((pid, v))
def person_pc1(sp, exclude=None):
    vs = [v for pid, v in by_person.get(sp, []) if pid != exclude]
    if len(vs) < 2: return None
    m = {n: np.nanmean([v.get(n, np.nan) for v in vs]) for n in NAMES}; return proj(m)
def boot(a, b, who, B=1000):
    a, b, who = np.array(a), np.array(b), np.array(who); people = np.unique(who); idx = {p: np.where(who == p)[0] for p in people}
    r = spearmanr(a, b).correlation; bs = []
    for _ in range(B):
        ii = np.concatenate([idx[p] for p in rng.choice(people, len(people))]); bs.append(spearmanr(a[ii], b[ii]).correlation)
    return [round(float(r), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(people)]
res = {}
a, b, w = [], [], []
for pid, sp, bt, v in P_:
    y = person_pc1(sp, exclude=pid)
    if y is not None: a.append(proj(v)); b.append(y); w.append(sp)
res["people_ceiling"] = boot(a, b, w)
donor_cache = {}
def donor_sp(pid, cond):
    k = (pid, cond)
    if k not in donor_cache:
        row = rows[pid]
        try:
            if cond.endswith("history_wrong"): donor_cache[k] = PP.donor(row, need_hist=len(PP.history(row)))[0]["selfPID"]
            elif cond.endswith("card_thin_wrong"): donor_cache[k] = PP.donor(row, rung="thin")[0]["selfPID"]
            elif cond.endswith("card_narr_wrong"): donor_cache[k] = PP.donor(row, rung="narr")[0]["selfPID"]
            else: donor_cache[k] = None
        except Exception: donor_cache[k] = None
    return donor_cache[k]
for cond, (cdirs, runs) in CONDS.items():
    C = coded(cdirs); text = {}
    for run in runs:
        for l in open(os.path.join(ROOT, "runs", run, "preds.jsonl")):
            j = json.loads(l)
            if j.get("ok"): text[(j["perspectiveID"], j["model"].split("/")[1], j["info"])] = j["pred"]
    res[cond] = {}
    for m in M5:
        own = ([], [], []); don = ([], [], [])
        for k, cc in C.items():
            pid, mdl, info = k.split("|")
            if mdl != m or pid not in rows or len(cc) != 2: continue
            pr = text.get((pid, mdl, info)) or {}
            x = proj(featvec(cc, {"conv_SP_Mem_descr": pr.get("moment", ""), "conv_SP_Mem_reason": pr.get("reason", "")}))
            sp = rows[pid]["selfPID"]; y = person_pc1(sp, exclude=pid)
            if y is not None: own[0].append(x); own[1].append(y); own[2].append(sp)
            if cond.endswith("_wrong"):
                d = donor_sp(pid, cond); yd = person_pc1(d) if d else None
                if yd is not None: don[0].append(x); don[1].append(yd); don[2].append(sp)
        r = {"own": boot(*own) if len(own[0]) > 30 else None}
        if don[0]: r["donor"] = boot(*don)
        res[cond][m] = r
    o = [r["own"][0] for r in res[cond].values() if r["own"]]; d = [r["donor"][0] for r in res[cond].values() if r.get("donor")]
    print(f"{cond:22s} own rho mean {np.mean(o):+.3f} (models: {' '.join(f'{x:+.2f}' for x in o)})" + (f" | donor rho mean {np.mean(d):+.3f} ({' '.join(f'{x:+.2f}' for x in d)})" if d else ""), flush=True)
print("people ceiling (one report vs the person's other reports):", res["people_ceiling"])
json.dump(res, open(os.path.join(ROOT, "analysis", "memory_styles_reportlevel.json"), "w"), indent=1)
