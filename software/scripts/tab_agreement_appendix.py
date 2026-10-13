#!/usr/bin/env python3
"""Appendix table: per scored answer, agreement between the human annotators, between each LLM annotator and the humans, and between LLM annotators
(Krippendorff's alpha, 95% item-bootstrap intervals), with each question's status under the paper's rule (scripts/human_agreement.py).
Reads analysis/code_v351/humans/HUMAN_AGREEMENT.json (scope in three classes) and humans_scope4/ (the original four classes, one line).
-> overleaf_git/sections/app_parts/tab_agreement.tex"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = json.load(open(os.path.join(ROOT, "analysis", "code_v351", "humans", "HUMAN_AGREEMENT.json")))
R4 = json.load(open(os.path.join(ROOT, "analysis", "code_v351", "humans_scope4", "HUMAN_AGREEMENT.json")))
LAB = {"topic": "What was the moment about?", "remembered": "What is remembered?", "new_info": "Is a new fact remembered?",
       "extent": "How much of the call?", "details": "How many details?", "moment_text": "Report or commentary?",
       "shared_by": "Who shared something personal?", "personal_kinds": "What did they share?", "feelings": "Which feelings?",
       "intensity": "How intense?", "appraisal.pleasant": "Appraisal: pleasant", "appraisal.unpleasant": "Appraisal: unpleasant",
       "appraisal.own_behavior": "Appraisal: caused by the reporter", "appraisal.other_behavior": "Appraisal: caused by someone else",
       "reason.first": "Reason: first choice", "reason.memo": "Reason: original labels", "matters_for": "What does it matter for?",
       "bigger_meaning": "Something bigger?"}
AIS = [("claude-haiku-5.5", "Haiku"), ("gpt-6-luna", "Luna"), ("deepseek-v4.1-flash", "DeepSeek")]
f = lambda s: "--" if not s else f"{s['alpha']['est']:.2f}".replace("0.", ".").replace("-.", "$-$.")
fi = lambda s: "--" if not s else (f"{s['alpha']['est']:.2f} [{s['alpha']['ci'][0]:.2f}, {s['alpha']['ci'][1]:.2f}]").replace("0.", ".").replace("-.", "$-$.")
ST = {"trusted": "trusted", "descriptive": "descriptive", "not interpreted": "not interpreted"}
L = [r"\begin{table*}[h]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.7pt}", r"\begin{tabular}{@{}lccccc l@{}}", r"\toprule",
     r"Scored answer & People & Haiku--people & Luna--people & DeepSeek--people & LLM--LLM & Status \\", r"\midrule"]
for k, lab in LAB.items():
    r = R["slots"].get(k)
    if not r: continue
    sv = r.get("status", {})   # answer-level status (paper's rule): trusted only if every LLM annotator is trusted
    if not r.get("humans_above_chance"): st = "not interpreted"
    elif all(v == "trusted" for v in sv.values()): st = "included"
    else: st = "descriptive (" + ", ".join(n for m, n in AIS if sv.get(m) != "trusted") + ")"
    L.append(f"{lab} & {fi(r['HH'])} & " + " & ".join(f(r["AI_H"].get(m)) for m, _ in AIS) + f" & {f(r['AI_AI'])} & {st}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Agreement on the " + str(R["n_items"]) + r" reports annotated by people (Krippendorff's $\alpha$; people: the " + {2: "two", 3: "three", 4: "four", 5: "five"}.get(len(R["humans"]), str(len(R["humans"]))) + r" human annotators, with a 95\% interval over reports; LLM--people: each LLM annotator's mean pairwise $\alpha$ with the human annotators; LLM--LLM: among the three LLM annotators). A question is included when people agree above chance and each LLM annotator agrees with people at least as well as the lower end of the people's interval.}",
      r"\label{tab:agreement}", r"\end{table*}"]
open(os.environ.get("OUT") or os.path.join(ROOT, "overleaf_git", "sections", "app_parts", "tab_agreement.tex"), "w").write("\n".join(L) + "\n")
e4 = R4["slots"]["extent"]["HH"]["alpha"]; e3 = R["slots"]["extent"]["HH"]["alpha"]
print("written; humans:", R["humans"], "items", R["n_items"], "| scope 3-way", e3, "4-way", e4)
