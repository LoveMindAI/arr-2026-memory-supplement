#!/usr/bin/env python3
"""Card ladder: what each kind of person information adds to memory prediction.
Conditions: none, card_thin, card_narr, card_person (cards alone) and history, card_*_history (cards on top of history).
For each model: paired contrast of every card condition against its base (none for cards alone, history for card+history), mean of three judges.
Writes analysis/card_ladder.json, paper/tab_cards.tex and paper/figs/fig_cards.pdf. Runs whatever conditions exist on disk."""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figs_models as FM
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = FM.ROOT
RUNS = ["main_text_cheap", "main_text_pro", "cards_text_cheap", "cards_text_pro", "cards_narr_cheap", "cards_narr_pro", "cards_person_cheap", "cards_person_pro"]
RUNGS_ALL = [("card_thin", "Scores"), ("card_narr", "Narrative card"), ("card_person", "Person-specific card")]
# Oct 7: person-specific cards turned out not to be woven (IDEAS_MAP §3); they stay in the JSON but not in the paper table.
RUNGS = RUNGS_ALL if os.environ.get("ALL_RUNGS") else RUNGS_ALL[:2]
MET = ["similarity", "same_focus", "same_reason", "event_match"]


def main():
    S = {}
    for r in RUNS:
        s, _ = FM.load(r); S.update(s)
    rng = np.random.default_rng(0)
    models = sorted({k[0] for k in S}, key=lambda m: (m in ("gemini-3.1-pro-preview", "gpt-5.6-sol", "mimo-v2.6-pro-ultraspeed"), m))
    res = {}
    for mdl in models:
        res[mdl] = {}
        for rung, _ in RUNGS:
            for base, cond in (("none", rung), ("history", rung + "_history")):
                if not any(k[0] == mdl and k[2] == cond for k in S):
                    continue
                res[mdl][cond] = {m: FM.paired(S, mdl, "text", base, cond, m, rng) for m in MET}
                res[mdl][cond]["level"] = {m: FM.level(S, mdl, "text", cond, m)[0] for m in MET}
    # summary: how many models gain / lose (CI excludes 0) per condition and metric, and the mean contrast over models
    summ = {}
    for rung, _ in RUNGS:
        for cond in (rung, rung + "_history"):
            for m in MET:
                cs = [res[x][cond][m] for x in res if cond in res[x] and res[x][cond][m]]
                if cs:
                    summ[f"{cond}|{m}"] = dict(models=len(cs), up=sum(c["lo"] > 0 for c in cs), down=sum(c["hi"] < 0 for c in cs),
                                               mean=float(np.mean([c["mean"] for c in cs])))
    json.dump(dict(per_model=res, summary=summ), open(os.path.join(ROOT, "analysis", "card_ladder.json"), "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:34s} models {v['models']:2d}  up {v['up']:2d}  down {v['down']:2d}  mean {v['mean']:+.3f}")
    # table: one row per condition, mean contrast over models and up/down counts, for focus and similarity
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{lcccc}", "\\toprule",
         "& \\multicolumn{2}{c}{Same focus} & \\multicolumn{2}{c}{Similarity}\\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Added information & $\\Delta$ & up/down & $\\Delta$ & up/down\\\\", "\\midrule"]
    for suffix, head in (("", "\\emph{vs.\\ nothing}"), ("_history", "\\emph{on top of history}")):
        rows = [(rung + suffix, name) for rung, name in RUNGS if f"{rung + suffix}|same_focus" in summ]
        if not rows:
            continue
        L.append(f"{head} & & & & \\\\")
        for cond, name in rows:
            f, s = summ[f"{cond}|same_focus"], summ[f"{cond}|similarity"]
            L.append(f"\\quad {name} & {f['mean']:+.3f} & {f['up']}/{f['down']} & {s['mean']:+.3f} & {s['up']}/{s['down']}\\\\")
    n = max((v["models"] for v in summ.values()), default=0)
    L += ["\\bottomrule", "\\end{tabular}",
          f"\\caption{{What a personality description adds. $\\Delta$: paired change in the mean of three judges, averaged over {n} models; up/down: models whose 95\\% bootstrap interval (resampling people) lies above/below zero. Scores: the person's HEXACO-60 domain and facet scores; the narrative card is written from them by our fine-tuned writer; higher is closer to the person's own memory.}}",
          "\\label{tab:cards}", "\\end{table}"]
    open(os.path.join(ROOT, "paper", "tab_cards.tex"), "w").write("\n".join(L))
    print("ok")


if __name__ == "__main__":
    main()
