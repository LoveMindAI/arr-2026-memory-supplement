#!/usr/bin/env python3
"""Voice v2 analysis (own-mic tracks for both speakers; voice_extract_v2.py). Same design as voice_memory.py: within each
located report, remembered turns minus non-remembered turns in the same fifth of the call and the same length bin, separately
for turns spoken by the rememberer and by the partner; mean over rememberers, bootstrap over rememberers (2,000, seed 20261009).
Adds: Praat prosody, audEERING arousal/dominance/valence, emotion2vec+ scores; a sensitivity run without remote-copy tracks;
old-vs-new comparison with analysis/voice_memory.json; instrument agreement (eGeMAPS vs Praat; audEERING vs emotion2vec);
and whether the rememberer's vocal A/D/V in the remembered stretch tracks their own affect-button ratings for that memory
(within-person, both sides centered on the person's mean over their reports). -> analysis/voice_memory_v2.json"""
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V = os.path.join(ROOT, "analysis", "voice_v2")
rows = {r["perspectiveID"]: r for r in P.rows()}
lenbin = lambda t: (lambda n: 0 if n <= 5 else 1 if n <= 15 else 2 if n <= 30 else 3)(len((t or "").split()))
rng0 = 20261009


def boot(vals, rng, B=2000):
    vals = np.array(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) < 3: return np.nan, [np.nan, np.nan], len(vals)
    bs = [np.mean(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(B)]
    return float(vals.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(vals)


def load_units():
    units, feats, names, kinds = [], [], None, {}
    for f in sorted(glob.glob(os.path.join(ROOT, "analysis", "located", "PID_*.json"))):
        L = json.load(open(f))
        if L.get("kind") != "in_conversation": continue
        pid = L["perspectiveID"]; loc = L["turns"]; loc = set(json.loads(loc) if isinstance(loc, str) else (loc or []))
        if not loc: continue
        vf = os.path.join(V, f"{pid}.npz")
        if not os.path.exists(vf): continue
        z = np.load(vf, allow_pickle=True); X = z["X"]; names = list(z["names"]); kinds[pid] = (str(z["self_kind"]), str(z["partner_kind"]))
        r = rows[pid]; turns = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]; end = max(t["start"] for t in turns) + 5
        for k, t in enumerate(turns):
            units.append((pid, k, int(k in loc), r["selfPID"], min(4, int(5 * t["start"] / end)), int(t["spk"] == r["selfPID"]), lenbin(t["text"])))
            feats.append(X[k] if k < len(X) else [np.nan] * X.shape[1])
    return units, np.array(feats, float), names, kinds


def contrast(units, F, names, keep_pid=lambda p: True):
    have = ~np.all(np.isnan(F), 1)
    by = collections.defaultdict(list)
    for i, u in enumerate(units):
        if keep_pid(u[0]): by[u[0]].append(i)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for pid, ii in by.items():
        for role, rv in (("rememberer_speaking", 1), ("partner_speaking", 0)):
            ins = [i for i in ii if units[i][2] == 1 and have[i] and units[i][5] == rv]
            if not ins: continue
            cells = {(units[i][4], units[i][6]) for i in ins}
            out = [i for i in ii if units[i][2] == 0 and have[i] and units[i][5] == rv and (units[i][4], units[i][6]) in cells]
            if not out: continue
            for j, nm in enumerate(names):
                a, b = np.nanmean(F[ins, j]), np.nanmean(F[out, j])
                if np.isfinite(a) and np.isfinite(b): per[units[ii[0]][3]][f"{role}:{nm}"].append(a - b)
    rng = np.random.default_rng(rng0); R = {}; people = sorted(per)
    sd = {nm: float(np.nanstd(F[:, j])) for j, nm in enumerate(names)}
    for role in ("rememberer_speaking", "partner_speaking"):
        for nm in names:
            key = f"{role}:{nm}"; vals = [np.nanmean(per[p][key]) for p in people if per[p].get(key)]
            est, ci, n = boot(vals, rng)
            if n < 3: continue
            R[key] = {"inside_minus_outside": round(est, 4), "ci95": [round(ci[0], 4), round(ci[1], 4)], "in_sd_units": round(est / sd[nm], 3) if sd[nm] else None,
                      "n_people": n, "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    return R


def affect_button(units, F, names):
    """Rememberer's vocal A/D/V over their own remembered turns (already centered on their call mean) vs their ratings."""
    col = {n: names.index(n) for n in ("arousal", "dominance", "valence")}
    vocal = collections.defaultdict(list)
    for i, u in enumerate(units):
        if u[2] == 1 and u[5] == 1: vocal[u[0]].append(F[i])
    recs = []
    for pid, L in vocal.items():
        r = rows[pid]
        try:
            rt = {"a": float(r["conv_SP_Mem_a"]), "d": float(r["conv_SP_Mem_d"]), "p": float(r["conv_SP_Mem_p"])}
        except (TypeError, ValueError):
            continue
        v = np.nanmean(np.array(L), 0)
        recs.append(dict(person=r["selfPID"], va=v[col["arousal"]], vd=v[col["dominance"]], vv=v[col["valence"]], **rt))
    by = collections.defaultdict(list)
    for x in recs: by[x["person"]].append(x)
    pairs = {"arousal~button_arousal": ("va", "a"), "dominance~button_dominance": ("vd", "d"), "valence~button_pleasure": ("vv", "p"),
             "arousal~button_pleasure": ("va", "p"), "valence~button_arousal": ("vv", "a"), "dominance~button_pleasure": ("vd", "p")}
    rng = np.random.default_rng(rng0); out = {"n_reports": len(recs)}
    people = [p for p, L in by.items() if len(L) >= 2]
    for name, (x, y) in pairs.items():
        def cent(p):
            L = [q for q in by[p] if np.isfinite(q[x]) and np.isfinite(q[y])]
            if len(L) < 2: return [], []
            xs = np.array([q[x] for q in L]); ys = np.array([q[y] for q in L]); return list(xs - xs.mean()), list(ys - ys.mean())
        C = {p: cent(p) for p in people}; C = {p: v for p, v in C.items() if v[0]}
        def r_of(ps):
            xs = sum((C[p][0] for p in ps), []); ys = sum((C[p][1] for p in ps), [])
            return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 3 and np.std(xs) > 0 and np.std(ys) > 0 else np.nan
        ps = sorted(C)
        if len(ps) < 5:
            out[name] = {"within_person_r": None, "n_people": len(ps)}; continue
        est = r_of(ps)
        bs = [r_of([ps[k] for k in rng.integers(0, len(ps), len(ps))]) for _ in range(1000)]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out[name] = {"within_person_r": round(est, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "n_people": len(ps),
                     "n_reports": int(sum(len(C[p][0]) for p in ps))}
    return out


def agreement():
    """Turn-level agreement on the raw (uncentered) dyad files, and within speaker (centered on each speaker's call mean)."""
    X, S = [], []
    for f in sorted(glob.glob(os.path.join(V, "dyad", "*__cpu.npz"))):
        d = os.path.basename(f).split("__")[0]; fa, fb = os.path.join(V, "dyad", f"{d}__adv.npz"), os.path.join(V, "dyad", f"{d}__e2v.npz")
        z = np.load(f); nt = len(z["spk"])
        YA = np.load(fa)["Y"] if os.path.exists(fa) else np.full((nt, 3), np.nan)
        YB = np.load(fb)["Y"] if os.path.exists(fb) else np.full((nt, 9), np.nan)
        X.append(np.concatenate([z["X"], YA, YB], 1)); S += [f"{d}|{s}" for s in z["spk"]]
    X = np.concatenate(X); S = np.array(S)
    Xc = X.copy()
    for s in np.unique(S):
        m = S == s; Xc[m] = X[m] - np.nanmean(X[m], 0)
    n = {k: i for i, k in enumerate(["pitch", "pitch_var", "loud", "loud_var", "voiced_rate", "pause_len", "jitter", "shimmer", "hnr", "slope",
                                      "praat_f0_st", "praat_f0_sd_st", "praat_int_db", "praat_int_sd_db", "listener_loud",
                                      "arousal", "dominance", "valence", "angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"])}
    pos_neg = X[:, n["happy"]] - X[:, n["angry"]] - X[:, n["sad"]] - X[:, n["disgusted"]] - X[:, n["fearful"]]
    pos_neg_c = Xc[:, n["happy"]] - Xc[:, n["angry"]] - Xc[:, n["sad"]] - Xc[:, n["disgusted"]] - Xc[:, n["fearful"]]
    def r(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return (round(float(np.corrcoef(a[m], b[m])[0, 1]), 3), int(m.sum())) if m.sum() > 10 else (None, int(m.sum()))
    A = {}
    for lab, i, j in (("eGeMAPS pitch vs Praat F0 (semitones)", "pitch", "praat_f0_st"), ("eGeMAPS pitch variability (CV) vs Praat F0 SD", "pitch_var", "praat_f0_sd_st"),
                      ("eGeMAPS loudness vs Praat intensity (dB)", "loud", "praat_int_db"), ("eGeMAPS loudness variability vs Praat intensity SD", "loud_var", "praat_int_sd_db")):
        A[lab] = {"raw_r_n": r(X[:, n[i]], X[:, n[j]]), "within_speaker_r_n": r(Xc[:, n[i]], Xc[:, n[j]])}
    A["audEERING valence vs emotion2vec (happy - angry - sad - disgusted - fearful)"] = {"raw_r_n": r(X[:, n["valence"]], pos_neg), "within_speaker_r_n": r(Xc[:, n["valence"]], pos_neg_c)}
    A["audEERING arousal vs emotion2vec neutral"] = {"raw_r_n": r(X[:, n["arousal"]], X[:, n["neutral"]]), "within_speaker_r_n": r(Xc[:, n["arousal"]], Xc[:, n["neutral"]])}
    A["audEERING arousal vs eGeMAPS loudness"] = {"raw_r_n": r(X[:, n["arousal"]], X[:, n["loud"]]), "within_speaker_r_n": r(Xc[:, n["arousal"]], Xc[:, n["loud"]])}
    # per-speaker pitch agreement distribution (flags octave-error speakers)
    per = []
    for s in np.unique(S):
        m = (S == s) & np.isfinite(X[:, n["pitch"]]) & np.isfinite(X[:, n["praat_f0_st"]])
        if m.sum() >= 8: per.append(np.corrcoef(X[m, n["pitch"]], X[m, n["praat_f0_st"]])[0, 1])
    per = np.array(per)
    A["per_speaker_pitch_r"] = {"median": round(float(np.nanmedian(per)), 3), "share_below_0.5": round(float(np.mean(per < .5)), 3), "n_speakers": int(len(per))}
    A["n_turns"] = int(len(X))
    return A


def main():
    units, F, names, kinds = load_units()
    print("units", len(units), "perspectives", len(kinds), "features", len(names))
    e2v = [i for i, nm in enumerate(names) if nm.startswith("e2v_")]
    # e2v positive-minus-negative index as an extra column
    ix = {nm: i for i, nm in enumerate(names)}
    pn = F[:, ix["e2v_happy"]] - F[:, ix["e2v_angry"]] - F[:, ix["e2v_sad"]] - F[:, ix["e2v_disgusted"]] - F[:, ix["e2v_fearful"]]
    F2 = np.concatenate([F, pn[:, None]], 1); names2 = names + ["e2v_pos_minus_neg"]
    R = {"design": __doc__, "n_units": len(units), "n_perspectives": len(kinds), "names": names2}
    R["contrast_all"] = contrast(units, F2, names2)
    R["contrast_own_machine_only"] = contrast(units, F2, names2, keep_pid=lambda p: kinds[p] == ("self_local", "self_local"))
    R["n_perspectives_own_machine_only"] = sum(1 for p in kinds if kinds[p] == ("self_local", "self_local"))
    old = json.load(open(os.path.join(ROOT, "analysis", "voice_memory.json")))["features"]
    R["old_vs_new"] = {k: {"old_received_stream": old[k], "new_own_mic": R["contrast_all"].get(k)} for k in old}
    R["affect_button"] = affect_button(units, F2, names2)
    R["agreement"] = agreement()
    R["n_dyads_with_emotion"] = {"audeering": len(glob.glob(os.path.join(V, "dyad", "*__adv.npz"))), "emotion2vec": len(glob.glob(os.path.join(V, "dyad", "*__e2v.npz")))}
    json.dump(R, open(os.path.join(ROOT, "analysis", "voice_memory_v2.json"), "w"), indent=1)
    for k, v in R["contrast_all"].items():
        print(("*" if v["excludes_zero"] else " "), f"{k:40s} {v['inside_minus_outside']:+.4f} {v['ci95']} ({v['in_sd_units']} SD) n={v['n_people']}")
    print(json.dumps(R["affect_button"], indent=0)); print(json.dumps(R["agreement"], indent=0))


if __name__ == "__main__":
    main()
