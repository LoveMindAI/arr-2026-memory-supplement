#!/usr/bin/env python3
"""What is the models' late-chosen content? (A1, Oct 9.) For model no-history predictions located in the last fifth of the call vs earlier,
and for people's memories located in the last fifth vs earlier: codebook profile of those memories (Haiku and Luna codes, averaged).
Reports the answers most over-represented in late choices. -> analysis/late_content.json"""
import collections, json, os
import numpy as np
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
def relpos(p):
    if not os.path.exists(p): return None
    L = json.load(open(p))
    if L.get("kind") != "in_conversation" or not L.get("turns"): return None
    t = L["turns"]; t = json.loads(t) if isinstance(t, str) else t; n = int(L.get("n_turns") or 0)
    return ((t[0] + t[-1]) / 2) / max(1, n - 1)
QS = ["topic", "remembered", "personal_kinds", "matters_for", "reason.first", "feelings", "extent"]
out = {}
for who in ("models", "people"):
    share = {q: {"late": collections.Counter(), "early": collections.Counter(), "n_late": 0, "n_early": 0} for q in QS}
    for coder in CODERS:
        codes = load("main_text_cheap__none" if who == "models" else "human", coder)
        for key, c in codes.items():
            pid = key.split("|")[0]
            p = relpos(os.path.join(ROOT, "analysis", "located_preds", key.split("|")[1], pid + ".json")) if who == "models" else relpos(os.path.join(ROOT, "analysis", "located", pid + ".json"))
            if p is None: continue
            g = "late" if p >= 0.8 else "early"
            for q in QS:
                v = c.get(q)
                if v is None: continue
                share[q]["n_" + g] += 1
                for x in (v if isinstance(v, frozenset) else [v]): share[q][g][str(x)] += 1
    out[who] = {}
    for q in QS:
        s = share[q]; opts = set(s["late"]) | set(s["early"])
        diff = {o: round(s["late"][o] / max(1, s["n_late"]) - s["early"][o] / max(1, s["n_early"]), 3) for o in opts}
        out[who][q] = dict(late_n=s["n_late"] // len(CODERS), top=sorted(diff.items(), key=lambda x: -abs(x[1]))[:3])
json.dump(out, open(os.path.join(ROOT, "analysis", "late_content.json"), "w"), indent=1)
for who in out:
    print("==", who, "(late = last fifth; share in late minus share earlier)")
    for q, v in out[who].items(): print(f"   {q:15s} n_late={v['late_n']:5d}  " + ", ".join(f"{o} {d:+.2f}" for o, d in v["top"]))
