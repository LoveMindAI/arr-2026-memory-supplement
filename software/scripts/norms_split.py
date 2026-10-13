#!/usr/bin/env python3
"""Population-norm note (A1, Oct 9): split rememberers in half (seed 909). From the FIRST half only, compare people's memory reports with the
five main models' no-history predictions, as coded by Haiku and Luna (codebook v3.5.1; averaged over the two coders), and print the share of
each answer. These numbers feed the plain-language note; the experiment is then run and scored on the SECOND half only.
-> analysis/norms_split.json"""
import collections, json, os, random
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
people = sorted({r["selfPID"] for r in rows.values()}); random.Random(909).shuffle(people)
half = len(people) // 2; A, Bh = set(people[:half]), set(people[half:])
out = {"train_people": sorted(A), "test_people": sorted(Bh), "shares": {}}
def shares(vals):
    c = collections.Counter(); n = 0
    for v in vals:
        if v is None: continue
        n += 1
        for x in (v if isinstance(v, frozenset) else [v]): c[str(x)] += 1
    return {o: c[o] / n for o in c}, n
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for coder in CODERS:
    H = load("human", coder); M = load("main_text_cheap__none", coder)
    Ht = {p: v for p, v in H.items() if rows[p]["selfPID"] in A}; Mt = {k: v for k, v in M.items() if rows[k.split("|")[0]]["selfPID"] in A}
    for q in ["details", "feelings", "intensity", "extent", "new_info", "moment_text", "shared_by", "matters_for", "remembered", "topic", "bigger_meaning", "reason.first"]:
        hs, hn = shares(h[q] for h in Ht.values()); ms, mn = shares(m[q] for m in Mt.values())
        for o in set(hs) | set(ms): agg[q][o].append((hs.get(o, 0), ms.get(o, 0)))
        agg[q]["_n"].append((hn, mn))
for q, d in agg.items():
    out["shares"][q] = {o: [round(sum(x[0] for x in v) / len(v), 3), round(sum(x[1] for x in v) / len(v), 3)] for o, v in d.items()}
json.dump(out, open(os.path.join(ROOT, "analysis", "norms_split.json"), "w"), indent=1)
print("train people", len(A), "test people", len(Bh))
for q, d in out["shares"].items():
    n = d.pop("_n"); print(f"== {q}  (reports: people {n[0]:.0f}, model {n[1]:.0f})")
    for o, (h, m) in sorted(d.items(), key=lambda x: -abs(x[1][1] - x[1][0]))[:5]: print(f"   {o:28s} people {h:.2f}  models {m:.2f}")
