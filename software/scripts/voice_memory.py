#!/usr/bin/env python3
"""Voice and memory, at the person level: the voice counterpart of gaze_memory.py.
Reads analysis/voice/<pid>.npz (voice_extract.py: per-turn eGeMAPS features on the speaker's own track, centered on that
speaker's own mean in the call, plus the listener's loudness). Writes analysis/voice_turn_features.npy aligned to
memorability.py's unit order (for VOICE=1), and tests, within each located report, remembered turns minus non-remembered
turns in the same fifth of the call and the same length bin, separately for turns spoken by the rememberer and by the
partner. Mean over rememberers, bootstrap over rememberers. -> analysis/voice_memory.json"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V = os.path.join(ROOT, "analysis", "voice")
rows = {r["perspectiveID"]: r for r in P.rows()}
NAMES = ["pitch", "pitch_var", "loud", "loud_var", "voiced_rate", "pause_len", "jitter", "shimmer", "hnr", "slope", "listener_loud"]
lenbin = lambda t: (lambda n: 0 if n <= 5 else 1 if n <= 15 else 2 if n <= 30 else 3)(len((t or "").split()))


def main():
    units, feats = [], []
    for f in sorted(x for x in os.listdir(os.path.join(ROOT, "analysis", "located")) if x.startswith("PID_") and x.endswith(".json")):
        L = json.load(open(os.path.join(ROOT, "analysis", "located", f)))
        if L.get("kind") != "in_conversation": continue
        pid = L["perspectiveID"]; loc = L["turns"]; loc = set(json.loads(loc) if isinstance(loc, str) else (loc or []))
        if not loc: continue
        r = rows[pid]; turns = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]; end = max(t["start"] for t in turns) + 5
        vf = os.path.join(V, f"{pid}.npz"); X = np.load(vf)["X"] if os.path.exists(vf) else None
        for k, t in enumerate(turns):
            units.append((pid, k, int(k in loc), r["selfPID"], min(4, int(5 * t["start"] / end)), int(t["spk"] == r["selfPID"]), lenbin(t["text"])))
            feats.append(X[k] if X is not None and k < len(X) else [np.nan] * len(NAMES))
    F = np.array(feats, float); np.save(os.path.join(ROOT, "analysis", "voice_turn_features.npy"), F)
    have = ~np.all(np.isnan(F), 1); print("units", len(units), "with voice", int(have.sum()))
    by = collections.defaultdict(list)
    for i, u in enumerate(units): by[u[0]].append(i)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for pid, ii in by.items():
        for role, rv in (("rememberer_speaking", 1), ("partner_speaking", 0)):
            ins = [i for i in ii if units[i][2] == 1 and have[i] and units[i][5] == rv]
            if not ins: continue
            cells = {(units[i][4], units[i][6]) for i in ins}
            out = [i for i in ii if units[i][2] == 0 and have[i] and units[i][5] == rv and (units[i][4], units[i][6]) in cells]
            if not out: continue
            for j, nm in enumerate(NAMES):
                per[units[ii[0]][3]][f"{role}:{nm}"].append(np.nanmean(F[ins, j]) - np.nanmean(F[out, j]))
    rng = np.random.default_rng(20261009); R = {"features": {}}; people = sorted(per)
    if os.environ.get("SUBSET") == "train": people = [q for q in people if q in set(json.load(open(os.path.join(ROOT, "analysis", "norms_split.json")))["train_people"])]
    sd = {nm: float(np.nanstd(F[:, j])) for j, nm in enumerate(NAMES)}
    for role in ("rememberer_speaking", "partner_speaking"):
        for nm in NAMES:
            key = f"{role}:{nm}"; vals = [np.nanmean(per[p][key]) for p in people if per[p].get(key)]
            if not vals: continue
            est = float(np.nanmean(vals)); boots = [np.nanmean([vals[k] for k in rng.choice(len(vals), len(vals))]) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            R["features"][key] = {"inside_minus_outside": round(est, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "in_sd_units": round(est / sd[nm], 3), "n_people": len(vals)}
            flag = "*" if lo > 0 or hi < 0 else " "
            print(f"{flag} {key:32s} {est:+.4f} [{lo:+.4f}, {hi:+.4f}]  ({est / sd[nm]:+.3f} SD)  n={len(vals)}")
    json.dump(R, open(os.path.join(ROOT, "analysis", "voice_memory" + ("_train" if os.environ.get("SUBSET") == "train" else "") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
