#!/usr/bin/env python3
"""Item 2 (A1, Oct 9): does history help where memory is stable within a person?
Stability per question = mean closeness between two reports by the SAME person (different calls, different partners) minus mean closeness
between reports by DIFFERENT people in the same batch (different calls) -- a rememberer effect on the codebook scale.
History gains per question from analysis/correspondence_v351.json (L3: history minus wrong-person history = person-specific; history minus none).
Spearman across the 18 questions, per coder; plus the ladder's 'person's own earlier reports' excess. -> analysis/stability_vs_history.json"""
import collections, json, os
import numpy as np
from scipy.stats import spearmanr
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
C = json.load(open(os.path.join(ROOT, "analysis", "correspondence_v351.json")))["coders"]
LAD = json.load(open(os.path.join(ROOT, "analysis", "ladder_v351.json")))
out = {}
for coder in CODERS:
    H = load("human", coder); by_batch = collections.defaultdict(list)
    for pid in H: by_batch[rows[pid]["batchID"]].append(pid)
    stab = {}
    for k in KEYS:
        same, diff = [], []
        for b, ps in by_batch.items():
            for i, a in enumerate(ps):
                for c in ps[i + 1:]:
                    ra, rc = rows[a], rows[c]
                    if ra["dyadID"] == rc["dyadID"] or ra["otherPID"] == rc["otherPID"] or ra["otherPID"] == rc["selfPID"] or rc["otherPID"] == ra["selfPID"]: continue
                    s = score(k, H[a][k], H[c][k])
                    if s is None: continue
                    (same if ra["selfPID"] == rc["selfPID"] else diff).append(s)
        stab[k] = float(np.mean(same) - np.mean(diff))
    g_wrong = {k: C[coder]["L3"]["history_minus_wrong"][k][0] for k in KEYS if C[coder]["L3"]["history_minus_wrong"].get(k)}
    g_none = {k: C[coder]["L3"]["history_minus_none"][k][0] for k in KEYS if C[coder]["L3"]["history_minus_none"].get(k)}
    own = {k: LAD[coder]["all"][k]["person's own earlier reports"]["excess"][0] for k in KEYS if LAD[coder]["all"].get(k, {}).get("person's own earlier reports", {}).get("excess")}
    ks = [k for k in KEYS if k in g_wrong and k in g_none and k in own]
    r1 = spearmanr([stab[k] for k in ks], [g_wrong[k] for k in ks]); r2 = spearmanr([stab[k] for k in ks], [g_none[k] for k in ks]); r3 = spearmanr([stab[k] for k in ks], [own[k] for k in ks])
    out[coder] = dict(stability={k: round(stab[k], 3) for k in ks}, rho_stability_vs_history_minus_wrong=[round(r1.correlation, 2), round(r1.pvalue, 3)],
                      rho_stability_vs_history_minus_none=[round(r2.correlation, 2), round(r2.pvalue, 3)], rho_stability_vs_own_earlier_excess=[round(r3.correlation, 2), round(r3.pvalue, 3)], n_questions=len(ks))
    print(coder, "stability:", {k: round(stab[k], 3) for k in sorted(ks, key=lambda z: -stab[z])})
    print("   rho with history-minus-wrong", out[coder]["rho_stability_vs_history_minus_wrong"], "| history-minus-none", out[coder]["rho_stability_vs_history_minus_none"], "| own-earlier excess", out[coder]["rho_stability_vs_own_earlier_excess"])
json.dump(out, open(os.path.join(ROOT, "analysis", "stability_vs_history.json"), "w"), indent=1)
