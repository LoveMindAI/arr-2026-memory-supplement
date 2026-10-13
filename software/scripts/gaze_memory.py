#!/usr/bin/env python3
"""Gaze aversion and memory, at the person level (extends Tsfasman et al. ICMI 2022 / dissertation ch. 4, where gaze presence
and maxGaze were LOWER in highly memorable moments, read as gaze aversion during encoding).

Signals: analysis/gaze/<perspectiveID>.npz from gaze_extract.py (5 fps; MediaPipe face landmarks; per person: face found,
head yaw/pitch, eye-look blendshapes). On a video call we can't tell "looking at the partner" from "looking at the screen",
so the measure is LOOKING AWAY FROM THE SCREEN, per frame:
  away = face not found, or |yaw - person's call median| > 15 deg, or |pitch - median| > 15 deg,
         or max(eyeLookOut*, eyeLookUp*, eyeLookDown*) - person's call median > 0.3   (eye direction without head turn)
Thresholds are fixed here before looking at memory labels.
Per turn (from the transcript turn start to the next turn's start): the rememberer's away rate and the partner's, plus
mean |yaw|, eye-look-out, blink. Every feature is centered within the person's call (their own baseline).

A. Descriptive (her RQ1a, person level): within each located report, the rememberer's away rate inside the remembered stretch
   minus their away rate on turns outside it in the same fifth of the call (position-matched). Same for the partner.
   Mean difference with a person-clustered bootstrap (resampling rememberers).
B. Predictive: writes analysis/gaze_turn_features.npy aligned to memorability.py's unit order, for its optional "gaze" family
   (memorability.py reads it when GAZE=1).
-> analysis/gaze_memory.json"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
G = os.path.join(ROOT, "analysis", "gaze")
LOOKS = ["eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight"]
rows = {r["perspectiveID"]: r for r in P.rows()}
def lenbin(t):
    n = len((t or "").split()); return 0 if n <= 5 else 1 if n <= 15 else 2 if n <= 30 else 3


def away_series(z, side):
    f = z[f"{side}__found"]; yaw = z[f"{side}__yaw"]; pit = z[f"{side}__pitch"]
    look = np.nanmax(np.vstack([z[f"{side}__{k}"] for k in LOOKS]), 0)
    dy = np.abs(yaw - np.nanmedian(yaw)); dp = np.abs(pit - np.nanmedian(pit)); dl = look - np.nanmedian(look)
    away = (f == 0) | (np.nan_to_num(dy) > 15) | (np.nan_to_num(dp) > 15) | (np.nan_to_num(dl) > 0.3)
    return away.astype(float), np.abs(yaw - np.nanmedian(yaw)), np.nanmean(np.vstack([z[f"{side}__eyeLookOutLeft"], z[f"{side}__eyeLookOutRight"]]), 0), \
        np.nanmean(np.vstack([z[f"{side}__eyeBlinkLeft"], z[f"{side}__eyeBlinkRight"]]), 0)


def turn_feats(pid):
    """List over turns of [self_away, self_absyaw, self_lookout, self_blink, partner_away, ...] centered within the call; or None."""
    f = os.path.join(G, f"{pid}.npz")
    if not os.path.exists(f): return None
    z = np.load(f); t = z["t"]; r = rows[pid]
    turns = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]
    S = {side: away_series(z, side) for side in ("self", "partner")}
    out = []
    for k, tu in enumerate(turns):
        a, b = tu["start"], (turns[k + 1]["start"] if k + 1 < len(turns) else tu["start"] + 5)
        m = (t >= a) & (t < max(b, a + 0.4))
        row = []
        for side in ("self", "partner"):
            row += [float(np.nanmean(x[m])) if m.any() else np.nan for x in S[side]]
        out.append(row)
    A = np.array(out, float)
    return A - np.nanmean(A, 0)          # each person's own baseline within this call


def main():
    units, feats, by_call = [], [], {}
    for f in sorted(x for x in os.listdir(os.path.join(ROOT, "analysis", "located")) if x.startswith("PID_") and x.endswith(".json")):
        L = json.load(open(os.path.join(ROOT, "analysis", "located", f)))
        if L.get("kind") != "in_conversation": continue
        pid = L["perspectiveID"]; loc = L["turns"]; loc = set(json.loads(loc) if isinstance(loc, str) else (loc or []))
        if not loc: continue
        r = rows[pid]; turns = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]
        TF = turn_feats(pid); end = max(tu["start"] for tu in turns) + 5
        for k, tu in enumerate(turns):
            units.append((pid, k, int(k in loc), r["selfPID"], min(4, int(5 * tu["start"] / end)), int(tu["spk"] == r["selfPID"]), lenbin(tu["text"])))
            feats.append(TF[k] if TF is not None else [np.nan] * 8)
    F = np.array(feats, float)
    np.save(os.path.join(ROOT, "analysis", "gaze_turn_features.npy"), F)
    have = ~np.isnan(F[:, 0])
    print("units", len(units), "with gaze", int(have.sum()), "reports with gaze", len({u[0] for u, h in zip(units, have) if h}))
    # A. position-matched inside-vs-outside difference per report, per side
    names = ["self_away", "self_absyaw", "self_lookout", "self_blink", "partner_away", "partner_absyaw", "partner_lookout", "partner_blink"]
    per = collections.defaultdict(lambda: collections.defaultdict(list))   # rememberer -> feature -> list of report diffs
    by = collections.defaultdict(list)
    for i, u in enumerate(units): by[u[0]].append(i)
    STRATA = {"all": lambda i: True, "rememberer_speaking": lambda i: units[i][5] == 1, "partner_speaking": lambda i: units[i][5] == 0,
              "partner_speaking_lenmatched": lambda i: units[i][5] == 0, "rememberer_speaking_lenmatched": lambda i: units[i][5] == 1}
    for pid, ii in by.items():
        for sname, sf in STRATA.items():
            ins = [i for i in ii if units[i][2] == 1 and have[i] and sf(i)]
            if not ins: continue
            fifths = {units[i][4] for i in ins}
            if sname.endswith("lenmatched"):
                cells = {(units[i][4], units[i][6]) for i in ins}
                out = [i for i in ii if units[i][2] == 0 and have[i] and (units[i][4], units[i][6]) in cells and sf(i)]
            else:
                out = [i for i in ii if units[i][2] == 0 and have[i] and units[i][4] in fifths and sf(i)]
            if not out: continue
            for j, nm in enumerate(names):
                per[units[ii[0]][3]][f"{sname}:{nm}"].append(np.nanmean(F[ins, j]) - np.nanmean(F[out, j]))
    rng = np.random.default_rng(20261009); people = sorted(per)
    if os.environ.get("SUBSET") == "train":
        people = [q for q in people if q in set(json.load(open(os.path.join(ROOT, "analysis", "norms_split.json")))["train_people"])]
    R = {"n_rememberers": len(people), "n_reports": sum(len(per[p].get("all:self_away", [])) for p in people), "features": {}}
    # raw (uncentered) away rates, to read the differences against
    raw = {}
    for side in ("self", "partner"):
        rates = []
        for pid in by:
            f = os.path.join(G, f"{pid}.npz")
            if os.path.exists(f): rates.append(float(np.nanmean(away_series(np.load(f), side)[0])))
        raw[side] = round(float(np.mean(rates)), 4)
    R["mean_away_rate"] = raw; print("mean away rate", raw)
    for nm in [f"{s_}:{n_}" for s_ in ("all", "rememberer_speaking", "partner_speaking", "rememberer_speaking_lenmatched", "partner_speaking_lenmatched") for n_ in names]:
        vals = [np.nanmean(per[p][nm]) for p in people if per[p].get(nm)]
        est = float(np.nanmean(vals)); boots = []
        for _ in range(2000):
            s = rng.choice(len(vals), len(vals)); boots.append(np.nanmean([vals[k] for k in s]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        R["features"][nm] = {"inside_minus_outside": round(est, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_people": len(vals)}
        print(f"{nm:34s} inside-outside {est:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={len(vals)}")
    R["definition"] = __doc__.split("Signals:")[1].split("A. Descriptive")[0].strip()
    json.dump(R, open(os.path.join(ROOT, "analysis", "gaze_memory" + ("_train" if os.environ.get("SUBSET") == "train" else "") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
