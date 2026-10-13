#!/usr/bin/env python3
"""Per-turn voice features from the OWN-MIC run (analysis/voice_v2), aligned exactly like scripts/voice_memory.py aligns the old
ones (same located-file order, same turns), for memorability.py's VOICE=1 family.
Writes analysis/voice_turn_features_v2.npy (the same 11 eGeMAPS features as before, now own-mic) and
analysis/voice_turn_features_v2emo.npy (those 11 + audEERING arousal, dominance, valence)."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import prompts as P
rows = {r["perspectiveID"]: r for r in P.rows()}
OLD = ["pitch", "pitch_var", "loud", "loud_var", "voiced_rate", "pause_len", "jitter", "shimmer", "hnr", "slope", "listener_loud"]
EMO = OLD + ["arousal", "dominance", "valence"]
A, B = [], []
for f in sorted(x for x in os.listdir(os.path.join(ROOT, "analysis", "located")) if x.startswith("PID_") and x.endswith(".json")):
    L = json.load(open(os.path.join(ROOT, "analysis", "located", f)))
    if L.get("kind") != "in_conversation": continue
    pid = L["perspectiveID"]; loc = L["turns"]; loc = set(json.loads(loc) if isinstance(loc, str) else (loc or []))
    if not loc: continue
    r = rows[pid]; turns = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]
    vf = os.path.join(ROOT, "analysis", "voice_v2", f"{pid}.npz")
    X = names = None
    if os.path.exists(vf):
        z = np.load(vf, allow_pickle=True); X = z["X"]; names = [str(n) for n in z["names"]]
    for k, t in enumerate(turns):
        if X is not None and k < len(X):
            A.append([X[k][names.index(n)] for n in OLD]); B.append([X[k][names.index(n)] for n in EMO])
        else:
            A.append([np.nan] * len(OLD)); B.append([np.nan] * len(EMO))
A, B = np.array(A, float), np.array(B, float)
old = np.load(os.path.join(ROOT, "analysis", "voice_turn_features.npy"))
assert old.shape[0] == A.shape[0], (old.shape, A.shape)          # same unit order and count as the old file
np.save(os.path.join(ROOT, "analysis", "voice_turn_features_v2.npy"), A); np.save(os.path.join(ROOT, "analysis", "voice_turn_features_v2emo.npy"), B)
print("units", A.shape, "with voice", int((~np.all(np.isnan(A), 1)).sum()), "| old file", old.shape)
