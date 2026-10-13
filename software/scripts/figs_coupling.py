#!/usr/bin/env python3
"""Trait-to-behavior coupling figure + numbers for the text, from analysis/trait_steering.json.
Left: correlation of each HEXACO domain with each memory rating, in people vs in models given the written profile (median over models).
Right: cross-validated R^2 of each rating from the six domains: people, models without a profile, with scores, with the written profile."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 8})
d = json.load(open(os.path.join(ROOT, "analysis", "trait_steering.json")))
models = sorted({k.split("|")[0] for k in d["models"]})
OUT = ["pleasure", "arousal", "dominance", "getting_along", "unit"]
LAB = {"pleasure": "Pleasure", "arousal": "Arousal", "dominance": "Dominance", "getting_along": "Got along", "unit": "Felt a unit"}
med_r = lambda info, o: np.median([d["models"][f"{m}|{info}"][o]["r"] for m in models], 0)
med_r2 = lambda info, o: float(np.median([d["models"][f"{m}|{info}"][o]["cv_r2"] for m in models]))
H = np.array([d["human"][o]["r"] for o in OUT]); M = np.array([med_r("card_narr", o) for o in OUT])
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.15), gridspec_kw={"width_ratios": [1, 1, 1.25]})
for a, X, t in ((ax[0], H, "People"), (ax[1], M, "Models + written profile")):
    im = a.imshow(X, cmap="RdBu_r", vmin=-.55, vmax=.55)
    a.set_xticks(range(6)); a.set_xticklabels(list("HEXACO")); a.set_yticks(range(len(OUT)))
    a.set_yticklabels([LAB[o] for o in OUT] if a is ax[0] else []); a.set_title(t, fontsize=8)
    for i in range(len(OUT)):
        for j in range(6):
            a.text(j, i, (f"{X[i, j]:.2f}".replace("0.", ".") if abs(X[i, j]) >= .005 else ".00"), ha="center", va="center", fontsize=5.5, color="k" if abs(X[i, j]) < .35 else "w")
conds = [("People", None), ("No profile", "none"), ("Scores", "card_thin"), ("Narrative", "card_narr")]
w = .2; y = np.arange(len(OUT))
for k, (lab, info) in enumerate(conds):
    v = [d["human"][o]["cv_r2"] if info is None else med_r2(info, o) for o in OUT]
    ax[2].barh(y + (1.5 - k) * w, v, height=w, label=lab, color=["#d98e32", "#bbbbbb", "#7fb3ba", "#1b6e7a"][k])
ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_xlabel("Cross-validated $R^2$ from the six traits"); ax[2].axvline(0, color="#888", lw=.5)
ax[2].invert_yaxis(); ax[2].legend(frameon=False, fontsize=6, loc="lower right"); ax[2].spines[["top", "right"]].set_visible(False)
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(os.path.join(ROOT, "paper", "figs", "fig_coupling.pdf"))
for o in OUT:
    print(f"{o:14s} R2 people {d['human'][o]['cv_r2']:+.3f} none {med_r2('none',o):+.3f} thin {med_r2('card_thin',o):+.3f} narr {med_r2('card_narr',o):+.3f} person {med_r2('card_person',o):+.3f}")
print("max |r| people", np.abs(H).max().round(2), " models+narr", np.abs(M).max().round(2))
