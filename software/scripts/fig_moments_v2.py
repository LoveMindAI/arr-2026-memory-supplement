#!/usr/bin/env python3
"""Figure 3. (a) Where in the call the remembered turns fall: people vs each of five models (share of remembered/chosen turns by fifth).
(b) What predicts which turns people remember: each feature family alone (logistic regression, balanced accuracy, 100 group-held-out splits),
including gaze and voice. -> paper_v2/figs/fig_moments.pdf"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
C = {"person": "#1f7a72", "model": "#6b4fa0", "mm": "#9a9a9a"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7.5})
Mm = A("memorability_models.json"); M = A("memorability.json"); Gz = A("memorability_gaze.json"); Vc = A("memorability_voice.json")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 1.6), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55})
x = np.arange(1, 6)
for i, (m, v) in enumerate(Mm["models"].items()):
    a1.plot(x, v["share_by_fifth_model"], "-", color=C["model"], alpha=0.55, lw=1, label="LLMs (5)" if i == 0 else None)
a1.plot(x, next(iter(Mm["models"].values()))["share_by_fifth_people"], "-o", color=C["person"], lw=1.8, ms=3.5, label="people")
a1.set_xticks(x); a1.set_xticklabels(["first", "2nd", "3rd", "4th", "last"]); a1.set_xlabel("position in the conversation (fifths)"); a1.set_ylabel("share of reported turns")
a1.legend(frameon=False, fontsize=6.8, loc="upper left"); a1.set_title("(a) Position of reported turns in the conversation", fontsize=7.8, loc="left")
for s in ("top", "right"): a1.spines[s].set_visible(False)
fam = {k: v["alone"] for k, v in M["ablation"]["LR"]["families"].items()}
fam["gaze (video)"] = Gz["ablation"]["LR"]["families"]["gaze"]["alone"]; fam["voice (audio)"] = Vc["ablation"]["LR"]["families"]["voice"]["alone"]
items = sorted(fam.items(), key=lambda kv: kv[1])
cols = [C["mm"] if k in ("gaze (video)", "voice (audio)") else C["person"] for k, _ in items]
a2.barh(range(len(items)), [v - 0.5 for _, v in items], left=0.5, color=cols)
a2.set_yticks(range(len(items))); a2.set_yticklabels([k for k, _ in items]); a2.axvline(0.5, color="#555", lw=0.6)
a2.axvline(M["ablation"]["LR"]["all_families"], color=C["person"], lw=0.8, ls="--")
a2.text(M["ablation"]["LR"]["all_families"] + 0.002, len(items) - 0.6, "all text features", fontsize=6.5, color=C["person"])
a2.set_xlim(0.49, 0.62); a2.set_xlabel("balanced accuracy, each family alone (chance .50)")
a2.set_title("(b) Balanced accuracy by feature family", fontsize=7.8, loc="left")
for s in ("top", "right"): a2.spines[s].set_visible(False)
out = os.path.join(ROOT, "paper_v2", "figs", "fig_moments.pdf"); fig.savefig(out, bbox_inches="tight"); print(out)
