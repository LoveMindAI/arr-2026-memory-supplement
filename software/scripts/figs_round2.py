#!/usr/bin/env python3
"""Steering vs fit, one dot per kind of person information (5 models, alone; from analysis/round2.json).
x: trait-to-behavior coupling (median CV R^2 of the model's dominance, pleasure and arousal ratings from the traits on the card, averaged).
y: change in agreement with the person's real memory vs no information (same focus, mean over models).
Also writes paper/tab_round2.tex."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
d = json.load(open(os.path.join(ROOT, "analysis", "round2.json")))
NAME = {"card_thin": "Self-report scores", "card_narr": "Narrative card", "card_impthin": "Impression scores", "card_impnarr": "Impression card",
        "card_facts": "Facts, listed", "card_factnarr": "Facts, narrated", "card_weave": "Card, facts woven in"}
COL = {"card_thin": "#7fb3ba", "card_narr": "#1b6e7a", "card_impthin": "#e0b07a", "card_impnarr": "#d98e32", "card_facts": "#aaaaaa",
       "card_factnarr": "#666666", "card_weave": "#7a3b69"}
fig, ax = plt.subplots(figsize=(3.2, 2.5))
rows = []
for k, nm in NAME.items():
    if k not in d["fidelity"]:
        continue
    c = d["coupling"][k]; f = d["fidelity"][k]["same_focus"]; s = d["fidelity"][k]["similarity"]
    if f["mean"] is None or f["models"] < 5 or any(c.get(o) is None for o in ("dominance", "pleasure", "arousal")):
        print("skip (incomplete):", nm); continue
    x = float(np.mean([c[o] for o in ("dominance", "pleasure", "arousal")]))
    rows.append((nm, x, f, s))
    ax.scatter(x, f["mean"], s=28, color=COL[k], zorder=3)
    ax.annotate(nm, (x, f["mean"]), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
ax.axhline(0, color="#888", lw=.6); ax.set_xlabel("Trait-to-behavior coupling ($R^2$)"); ax.set_ylabel("Change in same focus vs. nothing")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "paper", "figs", "fig_round2.pdf"))
L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{lccc}", "\\toprule",
     "Description & Same focus & Similarity & Coupling\\\\", "\\midrule"]
for nm, x, f, s in rows:
    L.append(f"{nm} & {f['mean']:+.3f} ({f['up']}/{f['down']}) & {s['mean']:+.3f} ({s['up']}/{s['down']}) & {x:.2f}\\\\")
L += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Each kind of information about the person, given alone to five models on all memories. Same focus and similarity: paired change against no information (mean of three judges, averaged over models; in parentheses, models whose 95\\% interval lies above/below zero). Coupling: how well the traits on the card predict the model's ratings of the remembered moment (cross-validated $R^2$, mean of dominance, pleasure and arousal; median over models).}",
      "\\label{tab:round2}", "\\end{table}"]
open(os.path.join(ROOT, "paper", "tab_round2.tex"), "w").write("\n".join(L))
for r in rows:
    print(f"{r[0]:28s} coupling {r[1]:.3f}  focus {r[2]['mean']:+.3f} ({r[2]['up']}/{r[2]['down']})  sim {r[3]['mean']:+.3f} ({r[3]['up']}/{r[3]['down']})")
