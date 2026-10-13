#!/usr/bin/env python3
"""Appendix table: does hearing or seeing the call change predictions? Per omni model, on the 300-memory media sample, each media
condition minus the transcript-only prediction for the same memory, with no information about the person; mean of three judges,
95% bootstrap interval (analysis/modality.json, written by scripts/figs_media.py). -> paper_v2/sections/app_parts/tab_media.tex"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = json.load(open(os.path.join(ROOT, "analysis", "modality.json")))
NAME = {"gemini-3.1-pro-preview": "Gemini 3.1 Pro (preview)", "gemini-3.8-flash": "Gemini 3.8 Flash", "mimo-v2.5": "MiMo v2.5",
        "mimo-v2.6-flash": "MiMo v2.6 Flash", "mimo-v2.6-pro-ultraspeed": "MiMo v2.6 Pro", "qwen3.8-omni-flash": "Qwen3.8 Omni Flash"}
MODS = [("audio", "Audio instead"), ("text_audio", "+ audio"), ("text_audio_video", "+ audio, video")]
s3 = lambda x: ("$-$" if x < 0 else "+") + f"{abs(x):.3f}"
def cell(c):
    s = f"{s3(c['mean'])} [{s3(c['lo'])}, {s3(c['hi'])}]"
    return f"\\textbf{{{s}}}" if (c["lo"] > 0 or c["hi"] < 0) else s
rows = []
for metric, ml in (("similarity", "Similarity"), ("same_focus", "Whose moment")):
    rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{ml}}}}} \\\\")
    for m, nm in NAME.items():
        pm = D["per_model"][m]
        rows.append(f"{nm} & " + " & ".join(cell(pm[f"{mod}|none"][metric]) for mod, _ in MODS) + " \\\\")
    sm = [D["summary"][f"{mod}|none|{metric}"] for mod, _ in MODS]
    rows.append("Mean over models (better / worse) & " + " & ".join(f"{s3(x['mean'])} ({x['up']} / {x['down']})" for x in sm) + " \\\\")
    rows.append("\\midrule")
n = sorted({c["n"] for m in D["per_model"].values() for k, v in m.items() if k.endswith("|none") for c in v.values()})
t = ["\\begin{table*}[h]", "\\centering\\footnotesize", "\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Model & " + " & ".join(l for _, l in MODS) + " \\\\", "\\midrule"] + rows[:-1] + [
     "\\bottomrule", "\\end{tabular}",
     f"\\caption{{Hearing and seeing the call. For each model, its prediction from the audio alone, the transcript plus audio, or the transcript plus audio and video, minus its prediction from the transcript alone, for the same memory with no information about the person; judged similarity to the person's memory (0--1) and agreement on whose moment it is. Mean of three judges with 95\\% bootstrap intervals (bold: interval excludes zero); {n[0]}--{n[-1]} memories per cell. ``Better / worse'' counts models whose interval lies above or below zero.}}",
     "\\label{tab:media}", "\\end{table*}"]
open(os.path.join(ROOT, "paper_v2", "sections", "app_parts", "tab_media.tex"), "w").write("\n".join(t) + "\n")
print("wrote tab_media.tex", n)
