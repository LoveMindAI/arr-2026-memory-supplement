#!/usr/bin/env python3
# Ties in typical() follow the neutral rule in tie_rule.py (order-free).
"""Character stability (A1, Oct 9 night): is an AI character as consistent with itself from call to call as a person is?
Mirror of the ladder's "person's own earlier reports" rung, applied to model outputs. For each model and condition
(no profile / scores card / narrative profile), the target is the model's coded memory of call X for person p, the predictor is the
typical answer (mode / majority labels / median, as in ladder_v351.typical) of THE SAME MODEL, SAME CONDITION, on p's EARLIER calls.
Score = person-specific fit: closeness to the target minus mean closeness to the same model's (same condition) memories of other
dyads in the same batch. People are scored the same way on the same calls (their own earlier reports predicting this one).
Each coder separately; person-clustered 95% bootstrap. No API calls; reads analysis/code_v351/.
-> analysis/character_stability.json"""
import collections, json, os
import numpy as np
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)

import tie_rule
def typical(vals, k): return tie_rule.typical(vals, k, ORD)   # neutral tie rule (Oct 12)
score = tie_rule.wrap_score(score)

SETS = {"none": "main_text_cheap__none", "card_thin": "cards_text_cheap__card_thin", "card_narr": "cards_narr_cheap__card_narr"}
out = {}
for coder in CODERS:
    H = load("human", coder)
    M = {}  # (cond, model) -> {pid: codes}
    for cond, sd in SETS.items():
        for key, v in load(sd, coder).items():
            pid, model = key.split("|")[:2]
            M.setdefault((cond, model), {})[pid] = v
    def fit(src, k):
        """src: {pid: codes}. Returns (excess values, people) over pids with >=1 earlier call in src."""
        by_person = collections.defaultdict(list); by_batch = collections.defaultdict(list)
        for pid in src: by_person[rows[pid]["selfPID"]].append(pid); by_batch[rows[pid]["batchID"]].append(pid)
        vals, ppl = [], []
        for pid, cv in src.items():
            t = cv.get(k)
            if t is None: continue
            er = [o for o in by_person[rows[pid]["selfPID"]] if int(rows[o]["roundIdx"]) < int(rows[pid]["roundIdx"])]
            pred = typical([src[o].get(k) for o in er], k)
            if pred is None: continue
            s = score(k, t, pred)
            if s is None: continue
            oth = [o for o in by_batch[rows[pid]["batchID"]] if rows[o]["dyadID"] != rows[pid]["dyadID"] and rows[o]["selfPID"] != rows[pid]["selfPID"]]
            c = [score(k, src[o].get(k), pred) for o in oth]; c = [x for x in c if x is not None]
            if c: vals.append(s - float(np.mean(c))); ppl.append(rows[pid]["selfPID"])
        return vals, ppl
    R = out[coder] = {}
    models = sorted({m for (_, m) in M})
    for k in KEYS:
        R[k] = {}
        # people, restricted to calls every condition covers (same calls as the models)
        common = set(H)
        for key_ in M: common &= set(M[key_])
        hv, hp = fit({p: H[p] for p in common}, k)
        R[k]["people"] = cboot(hv, hp) if len(hv) > 20 else None
        for cond in SETS:
            allv, allp, per = [], [], {}
            for m in models:
                src_ = {p: M[(cond, m)][p] for p in common}
                v, pp = fit(src_, k)
                if len(v) > 20: per[m] = round(float(np.mean(v)), 3)
                allv += v; allp += pp
            R[k][cond] = dict(pooled=cboot(allv, allp) if len(allv) > 20 else None, per_model=per)
json.dump(out, open(os.path.join(ROOT, "analysis", "character_stability.json"), "w"), indent=1)
for coder in CODERS:
    print("==", coder)
    for k, d in out[coder].items():
        f = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]" if x else "   -   "
        print(f"  {k:24s} people {f(d['people'])} | none {f(d['none']['pooled'])} | thin {f(d['card_thin']['pooled'])} | narr {f(d['card_narr']['pooled'])}")
