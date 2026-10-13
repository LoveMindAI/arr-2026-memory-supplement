#!/usr/bin/env python3
"""Human prediction mini-study (A1, Oct 8): A1, A3, A4 and A5 each read 12 call transcripts from the rememberer's seat and write what
that person remembered most clearly and why: 6 calls shared, 6 unique each (30 calls). The stimulus is the models' own 'none' prompt
(scripts/prompts.py build(..., 'text', 'none')) up to the questions, and the two questions are PARSEL's, verbatim. So human predictions are
scored exactly like model predictions (same judges, same v3.5.1 coders). Excludes every call containing a report either coder has seen (B001-B100).
-> human_coding/PREDICT_A1.html, ../lasting-impressions-coding/PREDICT_A3.html; key analysis/human_predict_key.json (LOCAL ONLY)."""
import json, os, random
import prompts as P
ROOT = ".."
seen = set(json.load(open(os.path.join(ROOT, "analysis", "A3_item_key.json"))).values())
rows = [r for r in P.rows() if r["dyadID"] != "PID_n+PID_n"]
seen_dyads = {r["dyadID"] for r in rows if r["perspectiveID"] in seen}
by_dyad = {}
for r in rows:
    if r["dyadID"] not in seen_dyads: by_dyad.setdefault(r["dyadID"], []).append(r)
rng = random.Random(1812); dyads = sorted(by_dyad); rng.shuffle(dyads)
pick, people = [], set()
for d in dyads:                                     # one report per call, and no person twice
    r = rng.choice(by_dyad[d])
    if r["selfPID"] in people: continue
    pick.append(r); people.add(r["selfPID"])
    if len(pick) == 36: break
shared, a1_u, a3_u, a4_u, a5_u, a2_u = pick[:6], pick[6:12], pick[12:18], pick[18:24], pick[24:30], pick[30:36]   # A4, A5 (Oct 8), A2 (Oct 9)
def stim(r):
    t = P.render_text(P.build(r, "text", "none")).split("===== USER =====", 1)[1]
    return t.split("Now please answer")[0].strip()
key = {}
def items(rs, coder):
    out = []
    for r in rs:
        pid = f"P{len(key) + 1:02d}"; key[pid] = dict(perspectiveID=r["perspectiveID"], coder=coder, shared=r in shared)
        out.append(dict(id=pid, stim=stim(r)))
    return out
a1 = items(shared, "shared_all4") + items(a1_u, "A1")
shared_items = [dict(id=pid, stim=stim(next(r for r in shared if r["perspectiveID"] == k["perspectiveID"]))) for pid, k in key.items() if k["shared"]]
a3 = shared_items + items(a3_u, "A3")
a4 = [dict(x) for x in shared_items] + items(a4_u, "A4"); a5 = [dict(x) for x in shared_items] + items(a5_u, "A5")
a2 = [dict(x) for x in shared_items] + items(a2_u, "A2")
random.Random(5).shuffle(a1); random.Random(6).shuffle(a3); random.Random(7).shuffle(a4); random.Random(8).shuffle(a5); random.Random(9).shuffle(a2)
json.dump(key, open(os.path.join(ROOT, "analysis", "human_predict_key.json"), "w"), indent=1)
TPL = open("predict_template.html").read()
for name, its, path in [("A1", a1, os.path.join(ROOT, "human_coding", "PREDICT_A1.html")),
                        ("A3", a3, os.path.join(ROOT, "..", "lasting-impressions-coding", "PREDICT_A3.html")),
                        ("A4", a4, os.path.join(ROOT, "..", "lasting-impressions-coding", "PREDICT_A4.html")),
                        ("A5", a5, os.path.join(ROOT, "..", "lasting-impressions-coding", "PREDICT_A5.html")),
                        ("A2", a2, os.path.join(ROOT, "..", "lasting-impressions-coding-A2", "PREDICT_A2.html"))]:
    D = dict(key=f"PREDICT_{name.upper()}_v1", title=name, q1=P.Q_DESCR, q2=P.Q_REASON, items=its)
    open(path, "w").write(TPL.replace("/*DATA*/null", json.dumps(D, ensure_ascii=False)))
print("calls", len(pick), "A1", len(a1), "A3", len(a3), "A4", len(a4), "A5", len(a5), "shared ids", sorted(p for p, k in key.items() if k["shared"]))
