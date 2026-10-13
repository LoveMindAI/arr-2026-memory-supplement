#!/usr/bin/env python3
"""Appendix (app:srm): method paragraph + table of every social-relations outcome, read from analysis/srm_glmm/srm_glmm.json
(model fits from scripts/srm_glmm.R) and analysis/srm_glmm/outcomes.json (base rates / means).
-> paper_v2/sections/app_parts/app_srm.tex"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = json.load(open(os.path.join(ROOT, "analysis", "srm_glmm", "srm_glmm.json")))
O = json.load(open(os.path.join(ROOT, "analysis", "srm_glmm", "outcomes.json")))
LAB = {
    "topic__WORK_EDUCATION_SERVICE": "Topic: work, education or service", "topic__PLACE_TRAVEL": "Topic: places or travel",
    "topic__FAMILY_RELATIONSHIPS_PETS": "Topic: family, relationships or pets", "topic__HOBBIES_INTERESTS_MEDIA": "Topic: hobbies, interests, media",
    "topic__PLANS_EVENTS": "Topic: plans, goals or recent events", "topic__FEELINGS_MOOD": "Topic: feelings or mood of the call",
    "topic__THE_STUDY_OR_CALL": "Topic: the study, call or technology", "topic__PARTNER_IMPRESSION": "Topic: how the partner came across",
    "topic__OTHER": "Topic: something else",
    "remembered__PARTNER_SAID_OR_DID": "Remembered: something the partner said or did", "remembered__PARTNER_FACT_OR_QUALITY": "Remembered: a fact or quality of the partner",
    "remembered__REPORTER_SAID_OR_DID": "Remembered: something the reporter said or did", "remembered__DID_TOGETHER": "Remembered: something they did together",
    "remembered__REPORTER_THOUGHT": "Remembered: the reporter's own thought or feeling", "remembered__SITUATION": "Remembered: the situation",
    "new_info__YES": "A specific new fact is remembered", "new_info__NO": "About information, but nothing new", "new_info__NOT_ABOUT_INFORMATION": "Not about information",
    "extent__ONE_MOMENT": "Scope: one moment", "extent__LONGER": "Scope: a longer part, or the whole call", "extent__STRETCH": "Scope: a stretch of the call",
    "extent__WHOLE_CALL_OR_PATTERN": "Scope: the whole call or a pattern", "extent__NOT_AN_EVENT": "Scope: not an event",
    "details": "Number of details (0 to 3 or more)",
    "moment_text__WHAT_HAPPENED": "Report of what happened", "moment_text__THOUGHTS": "Thoughts or feelings only", "moment_text__BOTH": "Report and commentary",
    "shared_by__PARTNER": "Partner shared something personal", "shared_by__BOTH": "Both shared something personal", "shared_by__NOBODY": "Nobody shared anything personal",
    "personal_kinds__INFO_GENERAL": "Shared: surface facts", "personal_kinds__INFO_REVEALING": "Shared: revealing information",
    "personal_kinds__THOUGHTS_EVENTS_PLANS": "Shared: thoughts about events or plans", "personal_kinds__NOTHING": "Shared: nothing personal",
    "feelings__JOY": "Feeling named: joy", "feelings__SURPRISE": "Feeling named: surprise", "feelings__OTHER": "Feeling named: another feeling",
    "feelings__NONE": "No feeling shown", "intensity": "Intensity of the feeling (1 to 5)",
    "appraisal_pleasant": "Appraisal: pleasant (1 to 5)", "appraisal_unpleasant": "Appraisal: unpleasant (1 to 5)",
    "appraisal_own_behavior": "Appraisal: caused by the reporter (1 to 5)", "appraisal_other_behavior": "Appraisal: caused by someone else (1 to 5)",
    "reason_first__PARTNER": "Reason mainly about: the partner", "reason_first__ME": "Reason mainly about: the reporter",
    "reason_first__US": "Reason mainly about: the two of them", "reason_first__CALL": "Reason mainly about: the call itself",
    "memo__fact_about_others.social_facts": "Reason, original scheme: social facts about the partner", "memo__fact_about_others.unexpected_info": "Reason, original scheme: unexpected information",
    "memo__meta_behaviour_of_other.behaviour": "Reason, original scheme: the partner's behavior", "memo__self_perception.annotator_feelings": "Reason, original scheme: the reporter's own feelings",
    "memo__self_perception.annotator_stories": "Reason, original scheme: the reporter's own stories", "memo__shared_experience.shared_story": "Reason, original scheme: a shared story",
    "matters_for__CONNECTION": "Matters for: connection with the partner", "matters_for__NONE": "Matters for: none of these",
    "bigger_meaning__YES": "Connects to something bigger",
}
def p(x):
    if x is None: return "--"
    return "$<$.001" if x < .001 else f"{x:.3f}".lstrip("0")
def s(x): return "--" if x is None else (f"{x:.2f}"[1:] if x >= 0 else "$-$" + f"{-x:.2f}"[1:])
DAG, AST = "$^\\dagger$", "$^\\ast$"
rows, missing = [], []
for key, v in D.items():
    name = key.split("__", 1)[1]
    lab = LAB.get(name);
    if lab is None: missing.append(name); lab = name.replace("_", " ")
    if v["kind"] == "binary":
        g, l = v["glmm"], v["lpm"]; unstable = g.get("regularised", False)
        rate = f"{O[key]['rate']:.2f}".lstrip("0")
        rem, par = ("--", "--") if unstable else (s(g["rememberer"]), s(g["partner"]))
        rows.append(f"{lab}{DAG if unstable else ''} & {rate} & {rem} & {par} & {p(g['p_rem'])} & {p(g['p_par'])} & {s(l['rememberer'])} & {s(l['partner'])} \\\\")
    else:
        m = v["lmm"]
        rows.append(f"{lab}{AST} & {O[key]['mean']:.2f} & {s(m['rememberer'])} & {s(m['partner'])} & {p(m['p_rem'])} & {p(m['p_par'])} & -- & -- \\\\")
assert not missing, missing
n_b = sum(v["kind"] == "binary" for v in D.values()); n_n = len(D) - n_b; n_u = sum(v.get("glmm", {}).get("regularised", False) for v in D.values())
nreports = max(v["n"] for v in D.values())
W = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
intro = (f"For each coded outcome we fit a model with crossed random intercepts for the rememberer, the partner and the group (session), using lme4 \\citep{{bates2015lme4}}: "
         f"a logistic mixed model for the {n_b} yes-or-no codes and a linear mixed model for the {W[n_n]} numeric ones (up to {nreports:,} reports each). "
         "For the logistic models we report variance shares on the latent scale, with the residual fixed at $\\pi^2/3$ \\citep{nakagawa2017coefficient}. "
         "Each $p$ is a likelihood-ratio test that drops one variance, halved because the variance sits on the boundary of its range. "
         "When a variance ran above 10 on the logit scale, a sign of separation in a rare code, we refit with blme and a weakly informative gamma(2, 0.5) prior on the standard deviations, from the family proposed by \\citet{chung2013nondegenerate} (their default takes the rate toward zero). "
         f"For the {W[n_u]} codes that separate even with the prior, we report only the linear-probability shares (LPM columns, given for every binary code). "
         "We model every binary code present in 5\\% to 95\\% of reports, except the \\emph{no} answer to the something-bigger question, which is the complement of its \\emph{yes}.\n")
tab = ["\\begin{table*}[t]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
       " & & \\multicolumn{2}{c}{Share of variance} & \\multicolumn{2}{c}{$p$} & \\multicolumn{2}{c}{LPM share} \\\\",
       "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
       "Outcome & Rate & Rememberer & Partner & Rememberer & Partner & Rememberer & Partner \\\\", "\\midrule"] + rows + [
       "\\bottomrule", "\\end{tabular}",
       "\\caption{Social relations models for every coded outcome. Rate: share of reports with the code (for numeric outcomes$^\\ast$, the mean; these use a linear mixed model, so shares are of total variance). "
       "$^\\dagger$Separates even with the prior; LPM shares only.}",
       "\\label{tab:srm_all}", "\\end{table*}"]
out = os.path.join(ROOT, "paper_v2", "sections", "app_parts"); os.makedirs(out, exist_ok=True)
open(os.path.join(out, "app_srm.tex"), "w").write("% Generated by scripts/tab_srm_appendix.py from analysis/srm_glmm/srm_glmm.json and outcomes.json. Do not edit by hand.\n" + intro + "\n" + "\n".join(tab) + "\n")
print(len(rows), "rows;", n_b, "binary,", n_n, "numeric,", n_u, "unstable")
