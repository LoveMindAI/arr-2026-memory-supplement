#!/usr/bin/env python3
"""Compact Figure 4 (one column): how much of each rating of the remembered moment the six HEXACO domains predict
(cross-validated R^2, folds by person), in people (their own traits) and in models with no card, the scores card, or the
narrative card (median over ten models; traits are those on the card). -> paper_v2/figs/fig_coupling_compact.pdf"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
T = json.load(open(os.path.join(ROOT, "analysis", "trait_steering.json")))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7.3})
R = [("pleasure", "pleasure"), ("arousal", "arousal"), ("dominance", "dominance"), ("getting_along", "getting along")]
G = [("people", "people (own traits)", "#1f7a72"), ("none", "LLM, no profile", "#cfc6e3"), ("card_thin", "LLM, score profile", "#a68bd1"), ("card_narr", "LLM, written profile", "#6b4fa0")]
fig, ax = plt.subplots(figsize=(3.15, 1.4))
x = np.arange(len(R)); w = 0.2
for j, (g, lab, col) in enumerate(G):
    vals = []
    for r, _ in R:
        if g == "people": v = T["human"][r]["cv_r2"]
        else: v = float(np.median([m[r]["cv_r2"] for k, m in T["models"].items() if k.endswith("|" + g) and r in m]))
        vals.append(max(v, 0))
    ax.bar(x + (j - 1.5) * w, vals, w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in R]); ax.set_ylabel("variance explained\nby traits ($R^2$)")
ax.set_ylim(0, 0.5); ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left", handlelength=1.2, columnspacing=0.8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
out = os.path.join(ROOT, "paper_v2", "figs", "fig_coupling_compact.pdf"); fig.savefig(out, bbox_inches="tight"); print(out)
