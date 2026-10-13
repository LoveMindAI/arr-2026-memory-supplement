#!/usr/bin/env python3
"""Position-swap results. For each call x model, in the CONTROL (order unchanged) and SWAP (person's remembered stretch moved to the end)
transcripts: does the model's located moment overlap (a) the person's remembered stretch, (b) the stretch that model chose on the ORIGINAL
transcript (mapped through each turn's source index)? Plus the relative position of the model's choice, and the blind judges' scores
against the person's memory (swap minus control, paired). Person-clustered bootstrap. -> analysis/position_swap_result.json"""
import collections, json, os
import numpy as np
import figs_models as FM
ROOT = ".."; D = os.path.join(ROOT, "analysis", "position_swap"); rng = np.random.default_rng(7)
def recs(pid, v): return json.load(open(os.path.join(D, f"{pid}__{v}.json")))
def boot(vals, who):
    g = collections.defaultdict(list)
    for v, w in zip(vals, who): g[w].append(v)
    G = list(g.values()); bs = [np.mean([x for i in rng.integers(0, len(G), len(G)) for x in G[i]]) for _ in range(2000)]
    return [round(float(np.mean(vals)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(vals)]
rows = collections.defaultdict(dict)
for f in os.listdir(os.path.join(D, "located")):
    L = json.load(open(os.path.join(D, "located", f))); rows[(L["pid"], L["model"])][L["version"]] = L
out = {"by_version": {}, "judges_swap_minus_control": {}}
M = collections.defaultdict(list); who = collections.defaultdict(list)
for (pid, model), v in rows.items():
    if "swap" not in v or "control" not in v: continue
    orig = os.path.join(ROOT, "analysis", "located_preds", model.split("/")[1], pid + ".json")
    if not os.path.exists(orig): continue
    O = json.load(open(orig)); ot = O["turns"]; ot = json.loads(ot) if isinstance(ot, str) else ot; ot = set(ot or [])
    for ver in ("control", "swap"):
        L = v[ver]; T = recs(pid, ver)["turns"]; chosen = set(L["turns"]) if L.get("kind") == "in_conversation" else set()
        person = set(L["moved"]); before = {k for k, t in enumerate(T) if isinstance(t.get("src"), int) and t["src"] in ot}
        M[f"{ver}|overlaps person's stretch"].append(float(bool(chosen & person))); M[f"{ver}|overlaps the model's original choice"].append(float(bool(chosen & before)))
        if chosen: M[f"{ver}|relative position of choice"].append(float(np.mean(sorted(chosen))) / max(1, len(T) - 1))
        for k in (f"{ver}|overlaps person's stretch", f"{ver}|overlaps the model's original choice"): who[k].append(pid.split("+")[0])
        if chosen: who[f"{ver}|relative position of choice"].append(pid.split("+")[0])
for k in sorted(M): out["by_version"][k] = boot(M[k], who[k]); print(f"{k:50s} {out['by_version'][k]}")
S, _ = FM.load("position_swap")
for m in ("similarity", "same_focus", "event_match", "topic_match", "same_reason"):
    d, w = [], []
    for (model, mod, info, pid), j in S.items():
        if info != "swap": continue
        c = S.get((model, mod, "control", pid))
        if c:
            a, b = FM.avg2(j, m), FM.avg2(c, m)
            if a is not None and b is not None: d.append(a - b); w.append(pid.split("+")[0])
    if d: out["judges_swap_minus_control"][m] = boot(d, w); print("judges swap - control", m, out["judges_swap_minus_control"][m])
json.dump(out, open(os.path.join(ROOT, "analysis", "position_swap_result.json"), "w"), indent=1)
