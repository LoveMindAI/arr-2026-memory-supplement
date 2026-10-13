#!/usr/bin/env python3
"""Compare the three locators on the 58 vocal-looking disagreements, Whisper transcript vs Scribe transcript (which marks laughter and fillers).
Agreement on kind; shared turns (within one transcript); position by time vs the person's own timing answer. -> analysis/scribe_recheck.json"""
import collections, glob, json, os
import numpy as np
A = "../analysis"; P = json.load(open(os.path.join(A, "scribe_recheck_pids.json")))
def load(d): return {json.load(open(f))["perspectiveID"]: json.load(open(f)) for f in glob.glob(os.path.join(A, d, "PID_*.json"))}
W = {l: load(d) for l, d in (("sonnet", "located"), ("gemini", "located_g3f"), ("luna", "located_luna"))}
S = {l: load(d) for l, d in (("sonnet", "located_scribe_sonnet"), ("gemini", "located_scribe_g3f"), ("luna", "located_scribe_luna"))}
def rng_(j):
    t = j.get("turns") or []
    return set(range(min(t), max(t) + 1)) if (j.get("kind") == "in_conversation" and t) else None
RW = {l: load(d) for l, d in (("sonnet", "located_whisper_rerun_sonnet"), ("gemini", "located_whisper_rerun_g3f"), ("luna", "located_whisper_rerun_luna"))}
C = json.load(open(os.path.join(A, "scribe_control_pids.json")))
out = {}
for name, D, P in (("vocal: whisper", W, P), ("vocal: whisper rerun (retest)", RW, P), ("vocal: scribe", S, P), ("non-vocal control: whisper", W, C), ("non-vocal control: scribe", S, C)):
    kinds = collections.Counter(tuple(D[l][p].get("kind") for l in D) for p in P)
    same_kind = np.mean([len({D[l][p].get("kind") for l in D}) == 1 for p in P])
    agree = np.mean([(len({D[l][p].get("kind") for l in D}) == 1) and (D["sonnet"][p].get("kind") != "in_conversation" or bool(set.intersection(*[rng_(D[l][p]) or set() for l in D]))) for p in P])
    placed = collections.Counter(sum(D[l][p].get("kind") == "in_conversation" for l in D) for p in P)
    gap = [abs(D[l][p]["located_pos"] - D[l][p]["human_timing"]) for l in D for p in P if D[l][p].get("kind") == "in_conversation" and D[l][p].get("located_pos") is not None and D[l][p].get("human_timing") is not None]
    out[name] = dict(n=len(P), all_three_same_kind=round(float(same_kind), 3), all_three_agree_fully=round(float(agree), 3), locators_placing_it=dict(sorted(placed.items())),
                     median_gap_to_own_timing=round(float(np.median(gap)), 1) if gap else None, top_kind_patterns=[[list(k), v] for k, v in kinds.most_common(5)])
    print(name, out[name])
json.dump(out, open(os.path.join(A, "scribe_recheck.json"), "w"), indent=1)
