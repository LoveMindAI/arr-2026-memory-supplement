#!/usr/bin/env python3
"""Figures for the human-side results. Reads analysis/*.json; writes paper/figs/*.pdf. Numbers come only from the analysis files."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
OUT = os.path.join(ROOT, "paper", "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
REM, PAR = "#1b6e7a", "#d98e32"


def fig_srm():
    v1 = A("srm_v1.json"); dr = A("srm_memory.json")
    rows = [("Specific moment (vs. general)", "specific"), ("Reason: self-perception", "self_perception"), ("Reason: facts about others", "fact_about_others"),
            ("Reason: shared experience", "shared_experience"), ("Integrative meaning", "integrative")]
    lab, rem, par, rem_e, par_e = [], [], [], [], []
    for name, k in rows:
        a = [v1[c][k]["rememberer"] for c in v1]; b = [v1[c][k]["partner"] for c in v1]
        lab.append(name); rem.append(np.mean(a)); par.append(np.mean(b)); rem_e.append((max(a) - min(a)) / 2); par_e.append((max(b) - min(b)) / 2)
    for name, k in [("Timing in the call", "timing"), ("Felt pleasure", "pleasure"), ("Felt arousal", "arousal")]:
        lab.append(name); rem.append(dr[k]["rememberer"]); par.append(dr[k]["partner"]); rem_e.append(0); par_e.append(0)
    y = np.arange(len(lab))[::-1]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.barh(y + 0.18, rem, 0.36, xerr=rem_e, color=REM, label="Rememberer", error_kw=dict(lw=0.6, capsize=1.5))
    ax.barh(y - 0.18, par, 0.36, xerr=par_e, color=PAR, label="Partner", error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_yticks(y); ax.set_yticklabels(lab); ax.set_xlabel("Share of variance (SRM)")
    ax.axhline(len(lab) - 5 - 0.5, color="#999", lw=0.5, ls=":")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=2, fontsize=8); ax.set_xlim(0, 0.45)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_srm.pdf")); plt.close(fig)


def fig_ident():
    sc = A("style_ceiling.json"); cc = A("style_ceiling_coded.json"); xc = A("style_cross_context.json")
    bars = [("Raw memory text", sc["mem"]["ident"]), ("Raw reason text", sc["why"]["ident"]), ("Style-stripped paraphrase", cc["neutral"]["ident"]),
            ("Codebook v1 profile", A("ident_v1.json")["claude-sonnet-5.5"]), ("Task call, from social calls", xc["mem"]["ident"])]
    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    y = np.arange(len(bars))[::-1]
    ax.barh(y, [b[1]["top1"] for b in bars], 0.6, color=REM)
    for yy, b in zip(y, bars):
        ax.plot([b[1]["chance"]] * 2, [yy - 0.35, yy + 0.35], color="k", lw=1)
        ax.text(b[1]["top1"] + 0.01, yy, f"{b[1]['top1']:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bars]); ax.set_xlim(0, 0.65)
    ax.set_xlabel("Top-1 identification (bar = chance)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ident.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_srm(); fig_ident(); print("ok", os.listdir(OUT))
