#!/usr/bin/env python3
# Ties follow tie_rule.py, as in ladder_v351.
"""People predicting (Oct 12): two of us each predicted 12 strangers' memories from the transcript (6 calls shared, 6 each).
Scored exactly like the ladder: per codebook question, closeness to this person's report minus mean closeness to other people's
reports from the same batch (other calls, not the person's own) = EXCESS (person-specific fit). Coded by the same two AI coders.
For comparison, the models' no-history and history predictions on the SAME 18 reports. Not restricted to reports with earlier
reports (the ladder's common set), since the 18 calls were drawn without that filter. -> analysis/people_predict_v351.json"""
import collections, json, os
import numpy as np
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
import tie_rule
score = tie_rule.wrap_score(score)
SETS = {"people": ["human_predict__none"], "model: no history": ["main_text_cheap__none"], "model: history": ["main_text_cheap__history"]}
out = {}
for coder in CODERS:
    H = load("human", coder); by_batch = collections.defaultdict(list)
    for pid in H: by_batch[rows[pid]["batchID"]].append(pid)
    preds = {}
    for name, sets in SETS.items():
        d = collections.defaultdict(list)
        for sd in sets:
            for key, v in load(sd, coder).items(): d[key.split("|")[0]].append((key.split("|")[1], v))
        preds[name] = d
    pids = sorted(preds["people"])
    out[coder] = {"n_reports": len(pids), "q": {}}
    for k in KEYS:
        R = {}
        for name in SETS:
            ex, who, raw = [], [], []
            for pid in pids:
                h = H.get(pid, {}).get(k)
                if h is None: continue
                for src_, p in preds[name].get(pid, []):
                    s = score(k, h, p[k])
                    if s is None: continue
                    oth = [o for o in by_batch[rows[pid]["batchID"]] if rows[o]["dyadID"] != rows[pid]["dyadID"] and rows[o]["selfPID"] != rows[pid]["selfPID"]]
                    c = [score(k, H[o][k], p[k]) for o in oth]; c = [x for x in c if x is not None]
                    if c: ex.append(s - float(np.mean(c))); who.append(pid.split("+")[0]); raw.append(s)
            R[name] = dict(n=len(ex), raw=round(float(np.mean(raw)), 3) if raw else None, excess=cboot(ex, who) if len(ex) >= 10 else None)
        out[coder]["q"][k] = R
json.dump(out, open(os.path.join(ROOT, "analysis", "people_predict_v351.json"), "w"), indent=1)
for k in ["topic", "extent", "remembered", "new_info", "details", "moment_text", "reason.first", "personal_kinds", "feelings"]:
    line = f"{k:16s}"
    for name in SETS:
        es = [out[c]["q"][k][name]["excess"] for c in CODERS]; es = [e for e in es if e]
        if es: line += f" | {name[:12]:12s} {np.mean([e[0] for e in es]):+.3f} [{min(e[1] for e in es):+.2f},{max(e[2] for e in es):+.2f}] n={out[CODERS[0]]['q'][k][name]['n']}"
    print(line)
