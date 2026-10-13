#!/usr/bin/env python3
"""Appendix tables for the personality profiles (paper_v2/sections/app_parts/tab_cards_*.tex). Every number is read from:
analysis/card_fidelity.json (facet recovery by reader x writer), analysis/trait_steering.json (trait-to-rating coupling per model),
analysis/round2.json (other descriptions: change vs no information, judges; coupling), analysis/card_content_coupling.json (per code)."""
import json, os
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper_v2", "sections", "app_parts"); os.makedirs(OUT, exist_ok=True)
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
f2 = lambda x: ".00" if round(x, 2) == 0 else (("$-$" if x < 0 else "") + f"{abs(x):.2f}".lstrip("0") if abs(x) < 1 else f"{x:.2f}")
s2 = lambda x: ".00" if round(x, 2) == 0 else ("+" if x >= 0 else "$-$") + f"{abs(x):.2f}".lstrip("0")
NAMES = {"deepseek-v4.1-flash": "DeepSeek v4.1 Flash", "gemini-3.8-flash": "Gemini 3.8 Flash", "gemma-4-31b-it": "Gemma 4 31B", "glm-5.3-flashx": "GLM-5.3 FlashX",
         "kimi-k2-0905": "Kimi K2 (0905)", "mimo-v2.5": "MiMo v2.5", "mimo-v2.5-pro": "MiMo v2.5 Pro", "mimo-v2.6-flash": "MiMo v2.6 Flash",
         "muse-glimmer-30b": "Muse Glimmer 30B", "qwen3.8-omni-flash": "Qwen3.8 Omni Flash"}
s3 = lambda x: ("+" if x >= 0 else "$-$") + f"{abs(x):.3f}".lstrip("0")

# 1. facet recovery
F = A("card_fidelity.json")["cells"]
RD = [("deepseek", "DeepSeek"), ("luna", "Luna"), ("readerB", "Paired reader")]
WR = [("lcw150", "Our writer"), ("lcw150_personpass_person", "Our writer + facts"), ("claude-opus-5.5", "Opus 5.5"), ("claude-sonnet-5.5", "Sonnet 5.5")]
L = [r"\begin{table}[h]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrccc@{}}", r"\toprule",
     r"Writer & Profiles & " + " & ".join(n for _, n in RD) + r" \\", r"\midrule"]
for w, wn in WR:
    ns = {F[f"{r}|{w}"]["n"] for r, _ in RD}
    L.append(f"{wn} & {max(ns):,} & " + " & ".join(f"{F[f'{r}|{w}']['pooled_ccc']:.2f}".lstrip("0") for r, _ in RD) + r" \\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Facet recovery: Lin's concordance between the 24 facet scores a reader recovers from a profile and the person's own scores, pooled over people and facets. Readers: DeepSeek v4.1 Flash, GPT-6 Luna, and the reader trained alongside our writer. Our writer + facts: the written profile with facts from the person's other calls woven in. Claude Opus and Sonnet 5.5 wrote profiles for 50 seeded people.}",
      r"\label{tab:card_fidelity}", r"\end{table}"]
open(os.path.join(OUT, "tab_cards_fidelity.tex"), "w").write("\n".join(L) + "\n")

# 2. trait-to-rating coupling, every model (narrative card), plus medians per condition and people
T = A("trait_steering.json")
R = [("pleasure", "Pleasure"), ("arousal", "Arousal"), ("dominance", "Control"), ("getting_along", "Along")]
models = sorted({k.split("|")[0] for k in T["models"]})
L = [r"\begin{table}[h]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     "Source & " + " & ".join(n for _, n in R) + r" \\", r"\midrule",
     "People & " + " & ".join(f2(T["human"][r]["cv_r2"]) for r, _ in R) + r" \\"]
for cond, cn in (("none", "No profile (median)"), ("card_thin", "Score profile (median)"), ("card_narr", "Written profile (median)")):
    L.append(cn + " & " + " & ".join(f2(float(np.median([T["models"][f"{m}|{cond}"][r]["cv_r2"] for m in models if f"{m}|{cond}" in T["models"]]))) for r, _ in R) + r" \\")
L.append(r"\midrule \multicolumn{5}{@{}l}{\emph{Written profile, each model}} \\")
for m in models:
    d = T["models"][f"{m}|card_narr"]
    L.append(NAMES.get(m, m) + " & " + " & ".join(f2(d[r]["cv_r2"]) for r, _ in R) + r" \\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{How much of each rating of the remembered moment the six HEXACO domains predict (cross-validated $R^2$, ridge regression, five folds grouped by person). People: from their own traits. Models: from the traits in the profile (with no profile, those of the person whose seat the model took); median over ten models, then each model with the written profile. Along: whether the moment helped the pair get along.}",
      r"\label{tab:card_coupling}", r"\end{table}"]
open(os.path.join(OUT, "tab_cards_coupling.tex"), "w").write("\n".join(L) + "\n")

# 3. other descriptions (round 2, five models)
R2 = A("round2.json")
CN = [("card_thin", "Scores"), ("card_narr", "Written profile"), ("card_impthin", "Impression scores"), ("card_impnarr", "Impression profile"),
      ("card_facts", "Facts list"), ("card_factnarr", "Facts narrative"), ("card_weave", "Woven profile")]
L = [r"\begin{table}[h]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r"Description & Whose & Lower & Control & Pleasure \\", r"\midrule"]
for c, n in CN:
    fo = R2["fidelity"][c]["same_focus"]; cp = R2["coupling"][c]
    L.append(f"{n} & {s3(fo['mean'])} & {fo['down']} of {fo['models']} & {f2(cp['dominance'])} & {f2(cp['pleasure'])}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Other descriptions of the person, five models (DeepSeek v4.1 Flash, Gemini 3.8 Flash, Gemma 4 31B, Qwen3.8 Omni Flash, MiMo v2.6 Flash) on all reports. Scores and written profile: the self-report descriptions of the main text. Whose: change against no information in the judges' agreement that the prediction centres on the same person (mean of three judges and five models); Lower: models whose interval lies below zero. Control, pleasure: $R^2$, how much of the model's ratings the person's traits predict (for impression descriptions, the impression traits; cross-validated, pooled over the five models). Facts carry no traits, so their $R^2$ is a null.}",
      r"\label{tab:card_other}", r"\end{table}"]
open(os.path.join(OUT, "tab_cards_other.tex"), "w").write("\n".join(L) + "\n")

# 4. content coupling: every code whose AUC gain exceeds .05 in any card condition under either coder
C = A("card_content_coupling.json"); coders = list(C)
conds = [("people", "People"), ("no_card", "No profile"), ("scores_card", "Score"), ("narrative_card", "Written")]
codes = sorted({k for cd in coders for cond in ("scores_card", "narrative_card") for k, v in C[cd][cond].items() if v.get("auc_gain") is not None and v["auc_gain"] > .05})
def g(cd, cond, k):
    v = C[cd][cond].get(k); return "--" if not v or v.get("auc_gain") is None else s2(v["auc_gain"])
L = [r"\begin{table*}[h]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l" + "cccc" * len(coders) + r"@{}}", r"\toprule",
     " & " + " & ".join(r"\multicolumn{4}{c}{" + ("Coded by Claude Haiku 5.5" if "haiku" in cd else "Coded by GPT-6 Luna") + "}" for cd in coders) + r" \\",
     "Code & " + " & ".join(n for cd in coders for _, n in conds) + r" \\", r"\midrule"]
for k in codes:
    L.append(k.replace("_", " ").replace("=", ": ").lower() + " & " + " & ".join(g(cd, c, k) for cd in coders for c, _ in conds) + r" \\")
L += [r"\midrule", "Codes with gain $>$ .05 & " + " & ".join(
    str(sum(1 for v in C[cd][c].values() if v.get("auc_gain") is not None and v["auc_gain"] > .05)) + " of " + str(sum(1 for v in C[cd][c].values() if v.get("auc_gain") is not None))
    for cd in coders for c, _ in conds) + r" \\", r"\bottomrule", r"\end{tabular}",
      r"\caption{How much the six HEXACO domains on the profile improve prediction of each yes-or-no code in the model's memory (cross-validated AUC with traits and model identity, minus AUC with model identity alone; five folds grouped by person). People: their own traits and memories. Rows: every code with a gain above .05 under a score or written profile for either coder; codes with a 5--95\% rate only.}",
      r"\label{tab:card_content}", r"\end{table*}"]
open(os.path.join(OUT, "tab_cards_content.tex"), "w").write("\n".join(L) + "\n")
print("wrote", OUT, "content codes:", len(codes))
