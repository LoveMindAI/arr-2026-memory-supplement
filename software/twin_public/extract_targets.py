#!/usr/bin/env python3
"""The ONLY place the public stand-in touches PARSEL (A1, Oct 11). Reads PARSEL and writes population-level NUMBERS to
targets_from_parsel.json: memory word lengths, call shape (turns, words, duration, group sizes, calls), and the two memory types'
shares and defining profiles. No text, row, ID, score or code is written. Generators read only the JSON."""
import collections, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import prompts as P

q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (5, 10, 25, 50, 75, 90, 95)}
rows = P.rows()
# memory lengths, with the person-level split of log length (between vs within person)
L = collections.defaultdict(list)
for r in rows:
    m, w = len(str(r["conv_SP_Mem_descr"]).split()), len(str(r["conv_SP_Mem_reason"]).split())
    L[r["globalPID"]].append((m, w))
def lengths(i):
    allv = np.array([np.log1p(x[i]) for v in L.values() for x in v])
    pm = {p: np.mean([np.log1p(x[i]) for x in v]) for p, v in L.items() if len(v) >= 2}
    within = np.mean([np.var([np.log1p(x[i]) for x in v]) for v in L.values() if len(v) >= 2])
    raw = [x[i] for v in L.values() for x in v]
    return dict(log1p_mean=float(allv.mean()), log1p_sd=float(allv.std()), between_person_sd=float(np.std(list(pm.values()))),
                within_person_sd=float(np.sqrt(within)), raw_quantiles=q(raw))
targets = dict(memory_moment_words=lengths(0), memory_reason_words=lengths(1))
# call shape from the Scribe transcripts
T, W, D, WPT = [], [], [], []
for f in glob.glob(os.path.join(ROOT, "transcripts_v3_scribe", "*.json")):
    j = json.load(open(f)); tt = [t for t in j.get("turns", []) if not t.get("bc")]
    if len(tt) < 5: continue
    T.append(len(tt)); W.append(sum(len(t["text"].split()) for t in tt)); WPT += [len(t["text"].split()) for t in tt]
    D.append(tt[-1]["end"] - tt[0]["start"])
targets["call_shape"] = dict(n_calls_measured=len(T), turns_per_call=q(T), words_per_call=q(W), words_per_turn=q(WPT), duration_seconds=q(D))
# group sizes and number of calls
size = collections.Counter(); people = collections.defaultdict(set); calls = set()
for r in rows:
    people[r["batchID"]].add(r["globalPID"]); calls.add(tuple(sorted((str(r["conv_selfLPID"]), str(r["conv_otherLPID"])))) + (r["batchID"],))
targets["groups"] = dict(n_groups=len(people), size_mix=dict(collections.Counter(len(v) for v in people.values())), n_calls=len(calls))
# the two memory types (person-level clusters, k = 2)
d = json.load(open(os.path.join(ROOT, "analysis", "memory_styles_describe.json")))["k2_means"]
DEF = ["scope:one_moment", "scope:stretch", "what:did_together", "what:partner_said_or_did", "new_fact"]
n = {k: d[k]["n"] for k in d}; tot = sum(n.values())
stretch = max(d, key=lambda k: d[k]["what:did_together"]); told = [k for k in d if k != stretch][0]
targets["memory_types"] = dict(
    note="person-level types; profile = share of a type's memories with each defining feature (codebook 3.5.1, people's own reports)",
    shared_stretch=dict(share=n[stretch] / tot, profile={f: d[stretch][f] for f in DEF}),
    what_they_were_told=dict(share=n[told] / tot, profile={f: d[told][f] for f in DEF}))
json.dump(targets, open(os.path.join(HERE, "targets_from_parsel.json"), "w"), indent=1)
print(json.dumps(targets, indent=1))
