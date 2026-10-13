#!/usr/bin/env python3
"""Who is predictable? (A1, Oct 9 night). Person-level version of the question-level stability-vs-history result.
(1) Consistency per person: for each of their reports with an earlier call, the person-specific fit of their own earlier reports
    (typical answer; same scoring and comparison set as ladder_v351: other people's memories in the batch), averaged over the FORM
    questions (scope, what is remembered, new fact, details, report or commentary, who shared) and both coders, then over reports.
(2) History gain per person: on the same reports, person-specific fit of a model given the RIGHT history minus the same model given
    the history of someone from another session who never met the partner, same number of memories (r3c *_history_wrong; prompts.donor), five models, form questions, both coders. Using the
    right-minus-wrong difference on the same target cancels how unusual that target memory is, which would otherwise couple (1) and (2).
(3) What marks a consistent rememberer: Spearman of consistency with HEXACO domains, social anxiety, trust, PPTS and the length of
    the person's reports; Benjamini-Hochberg across these tests.
No API calls. Run with PYTHONHASHSEED=0. -> analysis/who_is_predictable.json"""
import collections, csv, json, os
import numpy as np
from scipy.stats import spearmanr
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)

import tie_rule
def typical(vals, k): return tie_rule.typical(vals, k, ORD)   # neutral tie rule (Oct 12)
score = tie_rule.wrap_score(score)

FORM = ["extent", "remembered", "new_info", "details", "moment_text", "shared_by"]
MODELS = {"deepseek-v4.1-flash": "deepseek", "gemini-3.8-flash": "gemini", "gemma-4-31b-it": "gemma", "mimo-v2.6-flash": "mimo", "qwen3.8-omni-flash": "qwen3.8"}
cons = collections.defaultdict(list); gain = collections.defaultdict(list)
for coder in CODERS:
    H = load("human", coder)
    right = load("main_text_cheap__history", coder)
    wrong = {}
    for m, short in MODELS.items(): wrong.update(load(f"r3c_{short}__history_wrong", coder))
    by_person = collections.defaultdict(list); by_batch = collections.defaultdict(list)
    for pid in H: by_person[rows[pid]["selfPID"]].append(pid); by_batch[rows[pid]["batchID"]].append(pid)
    def excess(pid, k, pred):
        h = H[pid].get(k); s = score(k, h, pred) if h is not None and pred is not None else None
        if s is None: return None
        oth = [o for o in by_batch[rows[pid]["batchID"]] if rows[o]["dyadID"] != rows[pid]["dyadID"] and rows[o]["selfPID"] != rows[pid]["selfPID"]]
        c = [score(k, H[o].get(k), pred) for o in oth]; c = [x for x in c if x is not None]
        return s - float(np.mean(c)) if c else None
    for pid in H:
        person = rows[pid]["selfPID"]
        er = [o for o in by_person[person] if int(rows[o]["roundIdx"]) < int(rows[pid]["roundIdx"])]
        if not er: continue
        e = [excess(pid, k, typical([H[o].get(k) for o in er], k)) for k in FORM]; e = [x for x in e if x is not None]
        if e: cons[person].append(float(np.mean(e)))
        g = []
        for m in MODELS:
            kr, kw = f"{pid}|{m}|history", f"{pid}|{m}|history_wrong"
            if kr in right and kw in wrong:
                d = [(excess(pid, k, right[kr].get(k)), excess(pid, k, wrong[kw].get(k))) for k in FORM]
                d = [a - b for a, b in d if a is not None and b is not None]
                if d: g.append(float(np.mean(d)))
        if g: gain[person].append(float(np.mean(g)))
C = {p: float(np.mean(v)) for p, v in cons.items()}; G = {p: float(np.mean(v)) for p, v in gain.items()}
both = sorted(set(C) & set(G))
rho = spearmanr([C[p] for p in both], [G[p] for p in both])
rng = np.random.default_rng(2026); a = np.array([C[p] for p in both]); b = np.array([G[p] for p in both])
bs = [spearmanr(a[i], b[i]).correlation for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
out = {"n_people": len(both), "rho_consistency_vs_history_gain": [round(float(rho.correlation), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), round(float(rho.pvalue), 4)],
       "consistency_mean": round(float(a.mean()), 3), "gain_mean": round(float(b.mean()), 3)}
# split by consistency tercile: mean history gain
q = np.quantile(a, [1/3, 2/3]); out["gain_by_consistency_tercile"] = [round(float(b[a <= q[0]].mean()), 3), round(float(b[(a > q[0]) & (a <= q[1])].mean()), 3), round(float(b[a > q[1]].mean()), 3)]
# traits
TR = "<LOCAL>/Documents/Master_Prompt_Workshop/LSI_StoriesOfYourLife_Canonical_PARSEL_Data/PARSEL_GroundTruth/V6_Canonical_290_Private_Human_Data/canonical_scale_scores_290.tsv"
T = {r["globalPID"]: r for r in csv.DictReader(open(TR), delimiter="\t")}
VARS = ["HEXACO_HH_mean", "HEXACO_E_mean", "HEXACO_EX_mean", "HEXACO_A_mean", "HEXACO_C_mean", "HEXACO_OP_mean", "SIAS_mean",
        "Trust_Benevolence_mean", "Trust_Propensity_mean", "Trust_Ability_mean", "Trust_Integrity_mean", "PPTS_total_mean"]
glob_of = {r["selfPID"]: r.get("globalPID") for r in rows.values()}
words = collections.defaultdict(list)
for r in rows.values(): words[r["selfPID"]].append(len(str(r["conv_SP_Mem_descr"]).split()))
tests = []
for v in VARS + ["report_words"]:
    xs, ys = [], []
    for p in C:
        if v == "report_words": x = float(np.median(words[p]))
        else:
            t = T.get(glob_of.get(p) or p)
            try: x = float(t[v]) if t else None
            except Exception: x = None
        if x is None or not np.isfinite(x): continue
        xs.append(x); ys.append(C[p])
    r_ = spearmanr(xs, ys); tests.append([v, round(float(r_.correlation), 3), float(r_.pvalue), len(xs)])
ps = np.array([t[2] for t in tests]); o = np.argsort(ps); q_ = np.empty(len(ps)); m = len(ps); prev = 1.0
for rank, i in reversed(list(enumerate(o, 1))): prev = min(prev, ps[i] * m / rank); q_[i] = prev
out["consistency_traits"] = [[t[0], t[1], round(t[2], 4), round(float(q_[i]), 4), t[3]] for i, t in enumerate(tests)]
json.dump(out, open(os.path.join(ROOT, "analysis", "who_is_predictable.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "consistency_traits"}))
for t in out["consistency_traits"]: print(f"  {t[0]:34s} rho {t[1]:+.3f} p {t[2]:.4f} q {t[3]:.3f} n {t[4]}")
