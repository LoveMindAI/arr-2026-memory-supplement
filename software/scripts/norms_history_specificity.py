#!/usr/bin/env python3
"""Is the norms-note gain person-specific? For each prediction: score vs the person's own memory MINUS mean score vs other people's
memories in the same batch (other calls). Compare that excess, none vs norms, paired within report and model; test half only.
-> analysis/norms_history_specificity.json"""
import collections, json, os
import numpy as np
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
test = set(json.load(open(os.path.join(ROOT, "analysis", "norms_test_pids.json"))))
out = {}
for coder in CODERS:
    H = load("human", coder); by_batch = collections.defaultdict(list)
    for pid in H: by_batch[rows[pid]["batchID"]].append(pid)
    N = load("main_text_cheap__history", coder); M = load("norms_history_v2__norms_history", coder)
    nmap = {(k.split("|")[0], k.split("|")[1]): v for k, v in N.items()}; out[coder] = {}
    for k in KEYS:
        dv, dp, en, em = [], [], [], []
        for key, pm in M.items():
            pid, model = key.split("|")[0], key.split("|")[1]; pn = nmap.get((pid, model))
            if pid not in H or pid not in test or pn is None: continue
            others = [o for o in by_batch[rows[pid]["batchID"]] if rows[o]["dyadID"] != rows[pid]["dyadID"] and rows[o]["selfPID"] != rows[pid]["selfPID"]]  # Oct 9: exclude the rememberer's own other memories (A1 OK)
            def excess(pred):
                s = score(k, H[pid][k], pred[k]); c = [score(k, H[o][k], pred[k]) for o in others]; c = [x for x in c if x is not None]
                return None if s is None or not c else s - float(np.mean(c))
            a, b = excess(pm), excess(pn)
            if a is not None and b is not None: dv.append(a - b); dp.append(pid.split("+")[0]); en.append(b); em.append(a)
        out[coder][k] = dict(excess_none=round(float(np.mean(en)), 3), excess_norms=round(float(np.mean(em)), 3), change=cboot(dv, dp)) if dv else None
json.dump(out, open(os.path.join(ROOT, "analysis", "norms_history_specificity.json"), "w"), indent=1)
for c, d in out.items():
    print("==", c, "person-specific excess over batch-mates: history -> norms+history [change, CI]")
    for k, v in d.items():
        if v: print(f"   {k:24s} {v['excess_none']:+.3f} -> {v['excess_norms']:+.3f}   {v['change'][0]:+.3f} [{v['change'][1]:+.3f},{v['change'][2]:+.3f}]")
