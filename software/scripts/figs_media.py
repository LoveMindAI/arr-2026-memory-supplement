#!/usr/bin/env python3
"""Modality comparison on the 300-memory sample: does hearing (and seeing) the call change predictions?
Six omni models x {text, audio, text_audio, text_audio_video} x {none, history}. Text-only predictions come from the main text runs.
Paired contrast of each media condition against text with the same person information; mean of three judges.
Writes analysis/modality.json, paper/tab_media.tex, paper/figs/fig_media.pdf."""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figs_models as FM
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = FM.ROOT
MODS = [("audio", "Audio instead"), ("text_audio", "+ audio"), ("text_audio_video", "+ audio, video")]
MET = ["similarity", "same_focus", "same_reason", "event_match"]


def paired_mod(S, model, info, mod, m, rng, pids):
    d = []; who = []
    for pid in pids:
        x, y = S.get((model, "text", info, pid)), S.get((model, mod, info, pid))
        if x and y:
            u, v = FM.avg2(x, m), FM.avg2(y, m)
            if u is not None and v is not None:
                d.append(v - u); who.append(pid)
    d = np.array(d)
    if len(d) < 20:
        return None
    lo, hi = FM.cboot(d, who, rng)   # Oct 8: bootstrap over people
    return dict(mean=float(d.mean()), lo=lo, hi=hi, n=int(len(d)))


def main():
    S = {}
    for r in ["main_text_cheap", "main_text_pro", "main_media_cheap", "main_media_pro"]:
        s, _ = FM.load(r); S.update(s)
    pids = set(json.load(open(os.path.join(ROOT, "runs", "MAIN_SETS.json")))["pro300"])
    rng = np.random.default_rng(0)
    models = sorted({k[0] for k in S if k[1] != "text"})
    res = {}
    for mdl in models:
        res[mdl] = {}
        for info in ("none", "history"):
            for mod, _ in MODS:
                res[mdl][f"{mod}|{info}"] = {m: paired_mod(S, mdl, info, mod, m, rng, pids) for m in MET}
    summ = {}
    for mod, _ in MODS:
        for info in ("none", "history"):
            for m in MET:
                cs = [res[x][f"{mod}|{info}"][m] for x in res if res[x][f"{mod}|{info}"][m]]
                if cs:
                    summ[f"{mod}|{info}|{m}"] = dict(models=len(cs), up=sum(c["lo"] > 0 for c in cs), down=sum(c["hi"] < 0 for c in cs),
                                                     mean=float(np.mean([c["mean"] for c in cs])), n=[c["n"] for c in cs])
    json.dump(dict(per_model=res, summary=summ), open(os.path.join(ROOT, "analysis", "modality.json"), "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:40s} models {v['models']}  up {v['up']}  down {v['down']}  mean {v['mean']:+.3f}  n {min(v['n'])}-{max(v['n'])}")
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{lcccc}", "\\toprule",
         "& \\multicolumn{2}{c}{Same focus} & \\multicolumn{2}{c}{Similarity}\\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "vs.\\ transcript & $\\Delta$ & up/down & $\\Delta$ & up/down\\\\", "\\midrule"]
    nm = 0
    for info, head in (("none", "\\emph{nothing about the person}"), ("history", "\\emph{with history}")):
        L.append(f"\\multicolumn{{5}}{{l}}{{{head}}}\\\\")
        for mod, name in MODS:
            f, s = summ.get(f"{mod}|{info}|same_focus"), summ.get(f"{mod}|{info}|similarity")
            if f and s:
                nm = max(nm, f["models"])
                L.append(f"\\quad {name} & {f['mean']:+.3f} & {f['up']}/{f['down']} & {s['mean']:+.3f} & {s['up']}/{s['down']}\\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          f"\\caption{{Hearing and seeing the call, against reading its transcript (300-memory sample, {nm} models). $\\Delta$: paired change in the mean of three judges, averaged over models; up/down: models whose 95\\% bootstrap interval (resampling people) lies above/below zero.}}",
          "\\label{tab:media}", "\\end{table}"]
    open(os.path.join(ROOT, "paper", "tab_media.tex"), "w").write("\n".join(L))
    print("ok")


if __name__ == "__main__":
    main()
