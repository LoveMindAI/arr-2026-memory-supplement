#!/usr/bin/env python3
"""Time course around the remembered stretch (lit room's check against Tsfasman et al. ICMI 2022, whose speaker activity peaked just
BEFORE the remembered interval, p. 102): for each located memory, turns before the stretch (offsets -3..-1), inside it (first, middle,
last), and after it (+1..+3). Per position: the speaker's own loudness (own mic, eGeMAPS, centered within speaker and call) and the
speaking partner's look-away rate (both gaze networks agree, 15 deg), each averaged within rememberer, then mean over rememberers with a
person bootstrap. Length-adjusted: each value minus the same speaker's mean for turns of the same length bin in that call. Descriptive. -> analysis/timecourse_v2.json"""
import collections, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "multimodal", "scripts"))
import prompts as P, gaze_memory_v2 as G2
rows = {r["perspectiveID"]: r for r in P.rows()}
GI = len(G2.INSTR); both = G2.INSTR.index("both_15")
acc = {m: collections.defaultdict(lambda: collections.defaultdict(list)) for m in ("loud", "partner_away", "self_away")}
for f in sorted(x for x in os.listdir(os.path.join(ROOT, "analysis", "located")) if x.startswith("PID_") and x.endswith(".json")):
    L = json.load(open(os.path.join(ROOT, "analysis", "located", f)))
    if L.get("kind") != "in_conversation" or not L.get("turns"): continue
    pid = L["perspectiveID"]; loc = L["turns"]; loc = sorted(set(json.loads(loc) if isinstance(loc, str) else loc))
    r = rows[pid]; turns = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]
    vf = os.path.join(ROOT, "analysis", "voice_v2", f"{pid}.npz")
    if not os.path.exists(vf): continue
    z = np.load(vf, allow_pickle=True); X = z["X"]; names = [str(n) for n in z["names"]]; li = names.index("loud")
    TF = G2.turn_feats(pid, turns)
    a, b = loc[0], loc[-1]
    # length adjustment: subtract this call's mean for the same speaker and the same turn-length bin (as gaze/voice_memory match on length)
    lb = [G2.lenbin(t["text"]) for t in turns]
    def adj(vals, k, spk_only=None):
        same = [j for j in range(min(len(turns), len(vals))) if lb[j] == lb[k] and turns[j]["spk"] == turns[k]["spk"] and np.isfinite(vals[j])]
        return vals[k] - np.mean([vals[j] for j in same]) if same else np.nan
    loudv = [float(X[j][li]) if j < len(X) else np.nan for j in range(len(turns))]
    pav = [float(TF[j][GI + both]) if TF is not None else np.nan for j in range(len(turns))]
    sav = [float(TF[j][both]) if TF is not None else np.nan for j in range(len(turns))]
    pos = {}
    for k in range(len(turns)):
        if a - 3 <= k < a: pos[k] = f"before{k - a}"
        elif k > b and k <= b + 3: pos[k] = f"after+{k - b}"
        elif a <= k <= b: pos[k] = "inside_first" if k == a else ("inside_last" if k == b else "inside_middle")
    for k, lab in pos.items():
        if np.isfinite(loudv[k]):
            v = adj(loudv, k)
            if np.isfinite(v): acc["loud"][lab][r["selfPID"]].append(float(v))
        if TF is not None and turns[k]["spk"] != r["selfPID"] and np.isfinite(pav[k]):
            v = adj(pav, k)
            if np.isfinite(v): acc["partner_away"][lab][r["selfPID"]].append(float(v))
        if TF is not None and np.isfinite(sav[k]):
            v = adj(sav, k)
            if np.isfinite(v): acc["self_away"][lab][r["selfPID"]].append(float(v))
ORDER = ["before-3", "before-2", "before-1", "inside_first", "inside_middle", "inside_last", "after+1", "after+2", "after+3"]
rng = np.random.default_rng(0); out = {}
for m, D in acc.items():
    out[m] = {}
    for lab in ORDER:
        per = np.array([np.mean(v) for v in D[lab].values()]) if D[lab] else np.array([])
        if len(per) < 20: continue
        bs = [np.mean(rng.choice(per, len(per))) for _ in range(2000)]
        out[m][lab] = dict(mean=round(float(per.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], n_people=len(per))
json.dump(out, open(os.path.join(ROOT, "analysis", "timecourse_v2_lenadj.json"), "w"), indent=1)
for m in out:
    print(m); [print(f"  {lab:14s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] n={v['n_people']}") for lab, v in out[m].items()]
