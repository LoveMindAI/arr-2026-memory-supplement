#!/usr/bin/env python3
"""Writer-room check (Oct 10): is the inside-vs-outside voice contrast "remembered stretches are louder/steadier", or mostly "talk turns
quieter right after"? scripts/voice_memory_v2.contrast() run unchanged, with chosen outside turns EXCLUDED (neither inside nor outside):
  after3   the three turns right after the stretch
  before3  the three turns right before it
  both3    both
Plus, per speaker role, the length-adjusted level of the first / middle / last inside turns (same speaker, same length bin, same call).
-> analysis/voice_contrast_exclusions.json"""
import collections, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import voice_memory_v2 as VM
units, F, names, kinds = VM.load_units()
span = collections.defaultdict(list)
for u in units:
    if u[2] == 1: span[u[0]].append(u[1])
lo = {p: min(v) for p, v in span.items()}; hi = {p: max(v) for p, v in span.items()}
FEATS = ["loud", "loud_var", "voiced_rate", "praat_int_db", "praat_int_sd_db", "arousal", "dominance"]
idx = [names.index(f) for f in FEATS]; Fs = F[:, idx]
out = {}
for tag, rule in {"none": lambda u: False,
                  "after3": lambda u: hi[u[0]] < u[1] <= hi[u[0]] + 3,
                  "before3": lambda u: lo[u[0]] - 3 <= u[1] < lo[u[0]],
                  "both3": lambda u: (hi[u[0]] < u[1] <= hi[u[0]] + 3) or (lo[u[0]] - 3 <= u[1] < lo[u[0]])}.items():
    U = [(u[0], u[1], (-1 if (u[2] == 0 and rule(u)) else u[2])) + tuple(u[3:]) for u in units]
    out[tag] = VM.contrast(U, Fs, FEATS)
# inside first/middle/last, length-adjusted within speaker x call x length bin, by role
by = collections.defaultdict(list)
for i, u in enumerate(units): by[u[0]].append(i)
lev = collections.defaultdict(lambda: collections.defaultdict(list))
for pid, ii in by.items():
    for j, f in enumerate(FEATS):
        col = Fs[:, j]
        for i in ii:
            u = units[i]
            if u[2] != 1 or not np.isfinite(col[i]): continue
            pos = "first" if u[1] == lo[pid] else ("last" if u[1] == hi[pid] else "middle")
            role = "rememberer_speaking" if u[5] == 1 else "partner_speaking"
            same = [k for k in ii if units[k][5] == u[5] and units[k][6] == u[6] and np.isfinite(col[k])]
            if len(same) > 1: lev[f"{role}:{f}:{pos}"][u[3]].append(col[i] - np.mean(col[same]))
rng = np.random.default_rng(0); out["inside_levels_lenadj"] = {}
for k, D in lev.items():
    per = np.array([np.mean(v) for v in D.values()])
    if len(per) < 20: continue
    bs = [np.mean(rng.choice(per, len(per))) for _ in range(2000)]
    sd = float(np.nanstd(Fs[:, FEATS.index(k.split(":")[1])]))
    out["inside_levels_lenadj"][k] = dict(mean=round(float(per.mean()), 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], in_sd=round(float(per.mean()) / sd, 3), n_people=len(per))
json.dump(out, open(os.path.join(ROOT, "analysis", "voice_contrast_exclusions.json"), "w"), indent=1)
for tag in ("none", "after3", "before3", "both3"):
    print("==", tag)
    for k, v in out[tag].items():
        if k.split(":")[1] in ("loud", "praat_int_db", "loud_var", "praat_int_sd_db"): print(f"  {k:40s} {v['in_sd_units']:+.3f} SD {v['ci95']} n={v['n_people']} {'*' if v['excludes_zero'] else ''}")
print("== inside levels (length-adjusted)")
for k, v in sorted(out["inside_levels_lenadj"].items()):
    if k.split(":")[1] in ("loud", "praat_int_db"): print(f"  {k:45s} {v['in_sd']:+.3f} SD [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] n={v['n_people']}")
