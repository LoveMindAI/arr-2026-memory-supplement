#!/usr/bin/env python3
"""Do pairs who clicked remember the same moment? (A1, Oct 9 night)
Shared memory per call = the three judges' comparison of the partners' two memories (ref_partner items already judged for the main runs:
event_match = same specific moment, similarity 0-1, topic_match), averaged over judges (GPT-6 Luna, DeepSeek v4.1 Flash, Claude Haiku 5.5).
Connection per call, from PARSEL's post-call items (both partners' ratings averaged): rapport ("The interaction had good rapport"),
sync ("We were in sync/in tune"), liking of the partner, similarity in values, and willingness to meet again (0/1).
Two tests: (1) across calls, Spearman between shared memory and the pair's mean connection, call-clustered bootstrap is the unit;
(2) within person: for each rememberer, center their rating of each partner on their own mean, and ask whether calls they rated
above their own average have more shared memory than calls they rated below (person-clustered bootstrap). Test 2 removes
'some people rate everyone high'. No API calls. -> analysis/shared_memory_rapport.json"""
import collections, glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rows = {r["perspectiveID"]: r for r in P.rows()}
J = {"openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"}   # Oct 10: three judges
S = collections.defaultdict(dict)  # pid -> judge -> scores
for f in [os.path.join(ROOT, "runs", "main_text_cheap", "judged.jsonl")]:   # one fixed reference run (Oct 12): the main run's partner-reference judgments, all three judges
    for l in open(f):
        x = json.loads(l)
        if x.get("kind") == "ref_partner" and x.get("scores") and x["judge"] in J:
            S[x["id"].split("|")[0]][x["judge"]] = x["scores"]
MET = ["event_match", "similarity", "topic_match"]
def num(v):
    try:
        v = float(v); return v if np.isfinite(v) else None
    except Exception: return None
CONN = {"rapport": "conv_CC_rapp1", "sync": "conv_CC_sync1", "liking": "conv_PP_like", "similar": "conv_PP_sim", "meet_again": "conv_PP_Will"}
# one record per call (dyad): shared memory = mean over both perspectives' judged comparisons (they compare the same two memories)
calls = collections.defaultdict(list)
for pid, js in S.items():
    if pid not in rows or len(js) < len(J): continue
    calls[rows[pid]["dyadID"]].append(pid)
recs = []
for d, pids in calls.items():
    sm = {m: float(np.mean([S[p][j][m] for p in pids for j in S[p]])) for m in MET}
    r = rows[pids[0]]; partner = rows.get(f"{r['otherPID']}+{r['selfPID']}")
    if not partner: continue
    c = {}
    for k, col in CONN.items():
        a, b = num(r[col]), num(partner[col]); c[k] = None if a is None or b is None else (a + b) / 2
    recs.append(dict(dyad=d, batch=r["batchID"], persons=(r["selfPID"], partner["selfPID"]), shared=sm, conn=c, pids=[r["perspectiveID"], partner["perspectiveID"]]))
rng = np.random.default_rng(2026)
out = {"n_calls": len(recs), "across_calls": {}, "within_person": {}}
for m in MET:
    for k in CONN:
        v = [(x["shared"][m], x["conn"][k]) for x in recs if x["conn"][k] is not None]
        a, b = np.array(v).T; rho = spearmanr(a, b).correlation
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(a), len(a)); bs.append(spearmanr(a[i], b[i]).correlation)
        out["across_calls"][f"{m}~{k}"] = [round(float(rho), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3), len(a)]
# within person: each perspective's own rating of the call, centered on that person's mean; shared memory of that call
by_person = collections.defaultdict(list)
dyad_shared = {x["dyad"]: x["shared"] for x in recs}
for pid, r in rows.items():
    if r["dyadID"] in dyad_shared: by_person[r["selfPID"]].append(pid)
for m in MET:
    for k, col in CONN.items():
        diffs = []
        for person, pids in by_person.items():
            v = [(num(rows[p][col]), dyad_shared[rows[p]["dyadID"]][m]) for p in pids if num(rows[p][col]) is not None]
            if len(v) < 2: continue
            mu = np.mean([x for x, _ in v]); hi = [s for x, s in v if x > mu]; lo = [s for x, s in v if x < mu]
            if hi and lo: diffs.append(np.mean(hi) - np.mean(lo))
        diffs = np.array(diffs)
        bs = [rng.choice(diffs, len(diffs)).mean() for _ in range(2000)]
        out["within_person"][f"{m}~{k}"] = [round(float(diffs.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(diffs)]
out["base_rates"] = {m: round(float(np.mean([x["shared"][m] for x in recs])), 3) for m in MET}
json.dump(out, open(os.path.join(ROOT, "analysis", "shared_memory_rapport.json"), "w"), indent=1)
print("calls", len(recs), "base rates", out["base_rates"])
for sec in ("across_calls", "within_person"):
    print("==", sec)
    for k, v in out[sec].items(): print(f"  {k:28s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] n={v[3]}")
