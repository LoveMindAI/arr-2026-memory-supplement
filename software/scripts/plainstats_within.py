#!/usr/bin/env python3
"""Plain-statistics check, step 5 (A1, Oct 11: "is there a way to further eliminate the 'own wording' noise?"). WITHIN-PERSON subject test:
compare a prediction (or the partner's memory) with the person's memory of THIS call against the SAME person's memories of their OTHER calls.
Same author on both sides, so the person's own wording, length habits and recurring topics cancel; only what is specific to this call counts.
Score = share of the person's other memories the predictor is closer to this call's memory than to, minus .5 (0 = nothing call-specific).
Paired contrasts as plainstats_contrasts. -> analysis/plainstats/within.json"""
import collections, json, os
import numpy as np
src = open("plainstats.py").read().split("RUNGS = [")[0]; exec(src)
def within(meas, o, pid):
    s = score(meas, o, pid)
    if s is None or (isinstance(s, float) and np.isnan(s)): return None
    oth = [x for x in by_person[H[pid]["selfPID"]] if x != pid]
    c = [score(meas, o, x) for x in oth]; c = [x for x in c if x is not None and not np.isnan(x)]
    return float(np.mean([1.0 if s > x else .5 if s == x else 0.0 for x in c])) - .5 if c else None
pids = [p for p in H if len(by_person[H[p]["selfPID"]]) >= 2]
out = {"n_reports": len(pids)}
for meas in ["subject: content words"] + [f"subject: {n}" for n in EMB] + ["form (all ten)"]:
    R = {}
    def run(name, get):
        v, w = [], []
        for pid in pids:
            for o in get(pid):
                s = within(meas, o, pid)
                if s is not None: v.append(s); w.append(H[pid]["selfPID"])
        R[name] = cboot(v, w)
    run("partner's memory", lambda pid: [human_obj(f"{H[pid]['otherPID']}+{H[pid]['selfPID']}")] if f"{H[pid]['otherPID']}+{H[pid]['selfPID']}" in H else [])
    for cond in ("none", "history", "history_prior", "history_wrong", "card_narr", "card_thin", "norms", "norms_history"):
        run(f"model: {cond}", lambda pid, c=cond: PRED[f"model: {c}"].get(pid, []))
    for a, b in (("history", "history_wrong"), ("history", "none")):
        d, w = [], []
        for pid in pids:
            A = {o["m"]: within(meas, o, pid) for o in PRED[f"model: {a}"].get(pid, [])}; B = {o["m"]: within(meas, o, pid) for o in PRED[f"model: {b}"].get(pid, [])}
            for m in set(A) & set(B):
                if A[m] is not None and B[m] is not None: d.append(A[m] - B[m]); w.append(H[pid]["selfPID"])
        R[f"{a} minus {b} (paired)"] = cboot(d, w)
    out[meas] = R
    print(meas); [print(f"   {k:32s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] n={v[3]}") for k, v in R.items() if v]
json.dump(out, open(os.path.join(D, "within.json"), "w"), indent=1)
