#!/usr/bin/env python3
"""Figure 2 (two panels, full width). (a) Social relations GLMM: latent variance share of rememberer vs partner for selected codes.
(b) The ladder on person-specific fit, by codebook question: partner's memory, the person's own earlier memories, model without and
with history (mean of the two AI coders; whiskers span the two coders' 95% intervals). -> paper_v2/figs/fig_person.pdf"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C = {"person": "#1f7a72", "partner": "#b7791f", "model": "#6b4fa0", "model_h": "#a68bd1"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7.5})
S = json.load(open(os.path.join(ROOT, "analysis", "srm_glmm", "srm_glmm.json")))
L = json.load(open(os.path.join(ROOT, "analysis", "ladder_v351.json")))
A = [("b__extent__ONE_MOMENT", "one moment"), ("b__extent__LONGER", "longer part of the conversation"), ("b__remembered__PARTNER_SAID_OR_DID", "what the partner said or did"),
     ("b__remembered__DID_TOGETHER", "something they did together"), ("b__new_info__YES", "a new fact learned"), ("b__moment_text__WHAT_HAPPENED", "plain report of what happened"),
     ("b__shared_by__PARTNER", "the partner shared something"), ("b__reason_first__PARTNER", "reason: the partner"), ("b__reason_first__ME", "reason: me"),
     ("b__topic__PARTNER_IMPRESSION", "topic: how the partner came across"), ("b__topic__WORK_EDUCATION_SERVICE", "topic: work or study"),
     ("b__topic__HOBBIES_INTERESTS_MEDIA", "topic: hobbies and media"), ("b__topic__PLANS_EVENTS", "topic: plans and events")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.0), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.75})
y = np.arange(len(A))[::-1]
rem = [S[k]["glmm"]["rememberer"] for k, _ in A]; par = [S[k]["glmm"]["partner"] for k, _ in A]
a1.barh(y + 0.19, rem, 0.38, color=C["person"], label="rememberer"); a1.barh(y - 0.19, par, 0.38, color=C["partner"], label="partner")
a1.set_yticks(y); a1.set_yticklabels([l for _, l in A]); a1.set_xlabel("share of variance (latent scale)"); a1.set_xlim(0, 0.5)
a1.axhline(3.5, color="#999", lw=0.5, ls=":"); a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.3, -0.17), ncol=2, fontsize=6.8)
a1.set_title("(a) Variance share by annotated answer", fontsize=7.8, loc="left")
for s in ("top", "right"): a1.spines[s].set_visible(False)
Q = [("topic", "topic"), ("extent", "how much of the conversation"), ("remembered", "whose words or acts"), ("new_info", "new fact"), ("details", "details"),
     ("moment_text", "report or commentary"), ("reason.first", "kind of reason"), ("personal_kinds", "what was disclosed"), ("feelings", "feelings")]
R = [("partner's memory", "partner's report", C["partner"], "o"), ("person's own earlier reports", "own earlier reports", C["person"], "s"),
     ("model: no history", "LLM, no history", C["model"], "^"), ("model: history", "LLM, with history", C["model_h"], "v")]
yq = np.arange(len(Q))[::-1]
for j, (key, lab, col, mk) in enumerate(R):
    off = (j - 1.5) * 0.17
    for i, (q, _) in enumerate(Q):
        es = [L[c]["all"][q][key]["excess"] for c in L if q in L[c]["all"] and L[c]["all"][q].get(key, {}).get("excess")]
        if not es: continue
        m = np.mean([e[0] for e in es]); lo = min(e[1] for e in es); hi = max(e[2] for e in es)
        a2.plot([lo, hi], [yq[i] + off] * 2, color=col, lw=0.8); a2.plot(m, yq[i] + off, mk, color=col, ms=3.6, label=lab if i == 0 else None)
a2.axvline(0, color="#555", lw=0.6); a2.set_yticks(yq); a2.set_yticklabels([l for _, l in Q]); a2.set_xlabel("person-specific fit")
a2.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2); a2.set_title("(b) Person-specific fit by question", fontsize=7.8, loc="left")
for s in ("top", "right"): a2.spines[s].set_visible(False)
out = os.path.join(ROOT, "paper_v2", "figs", "fig_person.pdf"); fig.savefig(out, bbox_inches="tight"); print(out)
