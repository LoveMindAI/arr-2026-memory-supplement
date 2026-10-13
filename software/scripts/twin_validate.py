#!/usr/bin/env python3
"""Does the twin remember like people? (Oct 10 night). No API calls. PYTHONHASHSEED=0.
Same instrument on both sides: the paper's codebook v3.5.1, coded by Haiku 5.5 + Luna, averaged (real: analysis/code_v351/human;
twin: analysis/code_v351/twin). Features as in memory_styles.py.
(1) Every feature: real mean vs twin mean, and the gap in plain units (proportion points, or SD for continuous); also the gap for the
    twin's own TEMPLATES' reports (the real people whose coded specs the confederates followed), which separates "the confederate
    missed its spec" from "the sample of templates differs".
(2) Memory types: twins (>= 3 reports) assigned to the human k=2 types; share vs people's.
(3) Carry: ICC of key features within twin person, vs people's.
-> analysis/twin_validate.json"""
import collections, json, os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "memory_styles_models.py")).read()
exec(src.split("rng = np.random.default_rng(2026)")[0])   # people features P_, NAMES, mu, sd, km, TYPE, coded(), featvec(), icc, rows
ROOT = os.path.join(HERE, "..")
PEOPLE = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "twin", "people.json")))}
SET = os.environ.get("TWIN_SET", "twin"); MDIR = "memories" if SET == "twin" else "memories_" + SET.split("_", 1)[1]
T = coded(SET); TW = []
for k, cc in T.items():
    if len(cc) != 2: continue
    call, tid = k.split("_")
    m = json.load(open(os.path.join(ROOT, "twin", "v2", MDIR, f"{call}_{tid}.json")))
    TW.append((k, tid, call, featvec(cc, {"conv_SP_Mem_descr": m["answers"]["moment"], "conv_SP_Mem_reason": m["answers"].get("reason", "")})))
templates = {PEOPLE[t]["template"] for _, t, _, _ in TW}
REAL = [r for r in P_ if r[1] in templates]
def mean(recs, n): return float(np.nanmean([r[3].get(n, np.nan) for r in recs]))
res = {"n_twin_reports": len(TW), "n_twins": len({t for _, t, _, _ in TW}), "features": {}}
sd_all = {n: float(np.nanstd([r[3].get(n, np.nan) for r in P_])) for n in NAMES}
for n in NAMES:
    a, t, b = mean(P_, n), mean(TW, n), mean(REAL, n)
    cont = n.startswith("words_") or n == "details"
    gap = (t - a) / sd_all[n] if cont else (t - a)
    res["features"][n] = dict(real=round(a, 3), twin=round(t, 3), templates=round(b, 3), gap=round(gap, 3), unit="SD" if cont else "points")
# types
g = collections.defaultdict(list)
for k, tid, call, v in TW: g[tid].append(v)
prof = {t: np.array([np.nanmean([v.get(n, np.nan) for v in vs]) for n in NAMES]) for t, vs in g.items() if len(vs) >= 3}
if prof:
    Mz = np.array([np.where(np.isnan(x), mu, x) for x in prof.values()]); lab = km.predict((Mz - mu) / sd)
    res["types"] = {"twins_profiled": len(prof), "twin_share": {t: round(float(np.mean([TYPE[int(x)] == t for x in lab])), 3) for t in TYPE.values()},
                    "people_share": {t: round(float(np.mean([TYPE[int(x)] == t for x in km.labels_])), 3) for t in TYPE.values()}}
    tm = {t: TYPE[int(x)] for t, x in zip(prof, lab)}
    ht = {p: TYPE[int(l)] for p, l in zip(hk, km.labels_)}
    both = [t for t in tm if PEOPLE[t]["template"] in ht]
    res["types"]["twin_matches_template_type"] = dict(acc=round(float(np.mean([tm[t] == ht[PEOPLE[t]["template"]] for t in both])), 3), n=len(both))
# carry
pers = np.array([t for _, t, _, _ in TW]); res["carry"] = {}
for n in ("words_moment", "words_why", "details", "scope:one_moment", "what:did_together", "new_fact", "shared:nobody"):
    x = np.array([v.get(n, np.nan) for _, _, _, v in TW]); xr = np.array([r[3].get(n, np.nan) for r in P_]); pr = np.array([r[1] for r in P_])
    res["carry"][n] = dict(twin=round(icc(x, pers), 3), people=round(icc(xr, pr), 3))
json.dump(res, open(os.path.join(ROOT, "analysis", f"twin_validate_{SET}.json"), "w"), indent=1)
print("twin reports", res["n_twin_reports"], "twins", res["n_twins"])
print(f"{'feature':30s} {'real':>6s} {'twin':>6s} {'tmpl':>6s}  gap")
for n, v in sorted(res["features"].items(), key=lambda kv: -abs(kv[1]["gap"])):
    print(f"{n:30s} {v['real']:6.2f} {v['twin']:6.2f} {v['templates']:6.2f}  {v['gap']:+.2f} {v['unit']}")
print("types", res.get("types")); print("carry", res["carry"])
