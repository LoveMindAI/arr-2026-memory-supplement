#!/usr/bin/env python3
"""Per-turn gaze features from the v2 instruments, aligned exactly like scripts/gaze_memory.py aligns the old ones (same located-file
order, same turns), for memorability.py's GAZE=1 family. Columns = gaze_memory_v2.INSTR for self then partner (away rates per turn,
centered within the call). Writes analysis/gaze_turn_features_v2.npy (all instruments) and gaze_turn_features_v2_nets.npy
(UniGaze 15, PureGaze 15, both-15 and MediaPipe head, self and partner)."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "multimodal", "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import gaze_memory_v2 as G2, prompts as P
rows = {r["perspectiveID"]: r for r in P.rows()}
feats = []; n = len(G2.INSTR) * 2
for f in sorted(x for x in os.listdir(os.path.join(ROOT, "analysis", "located")) if x.startswith("PID_") and x.endswith(".json")):
    L = json.load(open(os.path.join(ROOT, "analysis", "located", f)))
    if L.get("kind") != "in_conversation": continue
    pid = L["perspectiveID"]; loc = L["turns"]; loc = set(json.loads(loc) if isinstance(loc, str) else (loc or []))
    if not loc: continue
    turns = json.load(open(os.path.join(P.TX, rows[pid]["dyadID"] + ".json")))["turns"]
    TF = G2.turn_feats(pid, turns)
    for k in range(len(turns)):
        feats.append(TF[k] if TF is not None else [np.nan] * n)
F = np.array(feats, float); old = np.load(os.path.join(ROOT, "analysis", "gaze_turn_features.npy"))
assert F.shape[0] == old.shape[0], (F.shape, old.shape)
names = [f"self_{i}" for i in G2.INSTR] + [f"partner_{i}" for i in G2.INSTR]
keep = [names.index(f"{s}_{i}") for s in ("self", "partner") for i in ("ug_15", "pg_15", "both_15", "mp_head")]
np.save(os.path.join(ROOT, "analysis", "gaze_turn_features_v2.npy"), F); np.save(os.path.join(ROOT, "analysis", "gaze_turn_features_v2_nets.npy"), F[:, keep])
print("units", F.shape, "with gaze", int((~np.all(np.isnan(F), 1)).sum()), "| old", old.shape, "| nets cols", [names[k] for k in keep])
