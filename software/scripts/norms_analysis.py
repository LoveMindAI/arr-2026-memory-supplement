#!/usr/bin/env python3
"""Norms-note experiment, test half only (A1, Oct 9). Paired within report and model: norms vs no-history baseline.
(1) Population: total variation distance between people's answer distribution and the models' (codebook v3.5.1, Haiku and Luna separately),
    none vs norms; (2) person: coder-matched correspondence per question; (3) blind judges' scores. Bootstraps over people.
-> analysis/norms_result.json"""
import collections, json, os
import numpy as np
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
import figs_models as FM
test = set(json.load(open(os.path.join(ROOT, "analysis", "norms_test_pids.json"))))
out = {"population": {}, "person": {}, "judges": {}}
def shares(vals):
    c = collections.Counter(); n = 0
    for v in vals:
        if v is None: continue
        n += 1
        for x in (v if isinstance(v, frozenset) else [v]): c[str(x)] += 1
    return {o: c[o] / n for o in c}
def tvd(a, b): return 0.5 * sum(abs(a.get(o, 0) - b.get(o, 0)) for o in set(a) | set(b))
QS = ["details", "feelings", "intensity", "extent", "new_info", "moment_text", "shared_by", "matters_for", "remembered", "topic", "bigger_meaning", "reason.first", "personal_kinds"]
for coder in CODERS:
    H = {p: v for p, v in load("human", coder).items() if p in test}
    N = {k: v for k, v in load("main_text_cheap__none", coder).items() if k.split("|")[0] in test}
    M = load("norms_note_v2__norms", coder)
    out["population"][coder] = {q: {"tvd_none": round(tvd(shares(h[q] for h in H.values()), shares(v[q] for v in N.values())), 3),
                                     "tvd_norms": round(tvd(shares(h[q] for h in H.values()), shares(v[q] for v in M.values())), 3)} for q in QS}
    nmap = {(k.split("|")[0], k.split("|")[1]): v for k, v in N.items()}; out["person"][coder] = {}
    for k in KEYS:
        dv, dp, a_, b_ = [], [], [], []
        for key, pm in M.items():
            pid, model = key.split("|")[0], key.split("|")[1]; pn = nmap.get((pid, model))
            if pid in H and pn is not None:
                sa, sb = score(k, H[pid][k], pm[k]), score(k, H[pid][k], pn[k])
                if sa is not None and sb is not None: dv.append(sa - sb); dp.append(pid.split("+")[0]); a_.append(sa); b_.append(sb)
        out["person"][coder][k] = dict(norms=round(float(np.mean(a_)), 3) if a_ else None, none=round(float(np.mean(b_)), 3) if b_ else None, diff=cboot(dv, dp) if dv else None)
S = {}
for r in ["main_text_cheap", "norms_note_v2"]:
    s, _ = FM.load(r); S.update(s)
rng2 = np.random.default_rng(1)
for m in ["similarity", "same_focus", "same_reason", "event_match", "topic_match"]:
    d, who = [], []
    for (model, mod, info, pid), v in S.items():
        if info != "norms" or pid not in test: continue
        b = S.get((model, "text", "none", pid))
        if b:
            x, y = FM.avg2(v, m), FM.avg2(b, m)
            if x is not None and y is not None: d.append(x - y); who.append(pid)
    if d:
        lo, hi = FM.cboot(np.array(d), who, rng2); out["judges"][m] = dict(diff=round(float(np.mean(d)), 3), lo=round(lo, 3), hi=round(hi, 3), n=len(d))
json.dump(out, open(os.path.join(ROOT, "analysis", "norms_result.json"), "w"), indent=1)
for c, d in out["population"].items():
    print("==", c, "population gap (TVD, lower = closer to people), none -> norms")
    for q, v in d.items(): print(f"   {q:16s} {v['tvd_none']:.2f} -> {v['tvd_norms']:.2f}")
for c, d in out["person"].items():
    up = [k for k, v in d.items() if v["diff"] and v["diff"][1] > 0]; dn = [k for k, v in d.items() if v["diff"] and v["diff"][2] < 0]
    print("==", c, "person-level: better with note on", len(up), up, "| worse on", len(dn), dn)
print("== judges (norms minus none):", out["judges"])
