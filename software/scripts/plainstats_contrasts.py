#!/usr/bin/env python3
"""Plain-statistics check, step 4: paired contrasts on the BEATS score (same report, same model where both are models), person-clustered
bootstrap. -> analysis/plainstats/contrasts.json"""
import collections, json, os
import numpy as np
src = open("plainstats.py").read().split("RUNGS = [")[0]; exec(src)
def beats(meas, o, pid, excl=()):
    s = score(meas, o, pid)
    if s is None or (isinstance(s, float) and np.isnan(s)): return None
    c = [score(meas, o, m) for m in mates(pid, excl)]; c = [x for x in c if x is not None and not np.isnan(x)]
    return float(np.mean([1.0 if s > x else .5 if s == x else 0.0 for x in c])) - .5 if c else None
def model_scores(meas, cond, pid): return {o["m"]: beats(meas, o, pid) for o in PRED[f"model: {cond}"].get(pid, [])}
PAIRS = [("history", "history_wrong"), ("history", "none"), ("history_prior", "none"), ("card_narr", "none"), ("card_thin", "none"), ("norms_history", "norms")]
out = {}
for meas in ["form (all ten)", "subject: content words"] + [f"subject: {n}" for n in EMB]:
    out[meas] = {}
    for a, b in PAIRS:
        d, w = [], []
        for pid in common:
            A, Bm = model_scores(meas, a, pid), model_scores(meas, b, pid)
            for m in set(A) & set(Bm):
                if A[m] is not None and Bm[m] is not None: d.append(A[m] - Bm[m]); w.append(H[pid]["selfPID"])
        out[meas][f"{a} minus {b}"] = cboot(d, w)
    d, w = [], []
    for pid in common:
        pp = f"{H[pid]['otherPID']}+{H[pid]['selfPID']}"
        if pp not in H: continue
        pb = beats(meas, human_obj(pp), pid, (pp,)); ms = [v for v in model_scores(meas, "none", pid).values() if v is not None]
        if pb is not None and ms: d.append(float(np.mean(ms)) - pb); w.append(H[pid]["selfPID"])
    out[meas]["blind model minus partner"] = cboot(d, w)
    print(meas); [print(f"   {k:28s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] n={v[3]}") for k, v in out[meas].items() if v]
json.dump(out, open(os.path.join(D, "contrasts.json"), "w"), indent=1)
