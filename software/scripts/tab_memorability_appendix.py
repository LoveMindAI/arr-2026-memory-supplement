#!/usr/bin/env python3
"""Appendix tables for the memorability classifier and the gaze/voice contrasts (Lasting Impressions, paper_v2).
(1) Feature-family ablation: each family alone, and the drop in balanced accuracy when it is removed, for logistic regression (100
    batch-grouped splits) and random forest (20 splits); the nine text families from analysis/memorability.json, gaze and voice from
    memorability_gaze.json / memorability_voice.json (runs that add that one family to the nine).
(2) Gaze and voice inside-vs-outside contrasts by speaker role: matched on position (same fifth of the call) and, where available,
    on turn length; full sample and the random half of people (norms_split train). From gaze_memory(_train).json, voice_memory(_train).json.
-> paper_v2/sections/app_parts/tab_mem_ablation.tex, tab_gaze.tex, tab_voice.tex"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
OUT = os.path.join(ROOT, "paper_v2", "sections", "app_parts")
M, G, V = A("memorability.json"), A("memorability_gaze.json"), A("memorability_voice.json")
LAB = {"position": "Position in the call", "speaker": "Who is speaking", "length & timing": "Length and timing", "lexical": "Words used",
       "semantic novelty": "How new the topic is", "sentiment": "Sentiment", "dialogue structure": "Dialogue structure",
       "content": "Content (sentence embedding)", "rememberer traits": "Rememberer's personality scores", "gaze": "Gaze (webcam)", "voice": "Voice (own track)"}
f3 = lambda x: f"{x:.3f}"; s3 = lambda x: ("$-$" if x < 0 else "+") + f"{abs(x):.3f}"
rows = []
for fam in M["families"] + ["gaze", "voice"]:
    src = G if fam == "gaze" else V if fam == "voice" else M
    lr, rf = src["ablation"]["LR"]["families"][fam], src["ablation"]["RF"]["families"][fam]
    rows.append(f"{LAB[fam]} & {f3(lr['alone'])} & {f3(rf['alone'])} & {s3(lr['drop_without'])} & {s3(rf['drop_without'])} \\\\")
t = ["\\begin{table}[h]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
     " & \\multicolumn{2}{c}{Alone} & \\multicolumn{2}{c}{Drop when removed} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
     "Feature family & LR & RF & LR & RF \\\\", "\\midrule"] + rows[:9] + ["\\midrule"] + rows[9:] + [
     "\\midrule", f"All nine families (LR 100, RF 20 splits) & \\multicolumn{{2}}{{c}}{{{f3(M['ablation']['LR']['all_families'])} \\,/\\, {f3(M['ablation']['RF']['all_families'])}}} & & \\\\",
     f"Dummy (base rate) & \\multicolumn{{2}}{{c}}{{{f3(M['models']['Dummy']['bal_acc'])}}} & & \\\\",
     "\\bottomrule", "\\end{tabular}",
     f"\\caption{{Which kinds of features find remembered turns. Balanced accuracy of each feature family on its own, and the loss when it is removed from all the others, for logistic regression (LR, 100 splits) and random forest (RF, 20 splits) holding out whole groups; {M['units']:,} turns in {M['reports']:,} located memories, {M['positive_rate']*100:.0f}\\% of turns in a remembered stretch. Gaze and voice rows come from runs that add that family to the nine text families.}}",
     "\\label{tab:mem_ablation}", "\\end{table}"]
open(os.path.join(OUT, "tab_mem_ablation.tex"), "w").write("\n".join(t) + "\n")

def ci(d):
    if d is None: return "---"
    v, (lo, hi) = d["inside_minus_outside"], d["ci95"]; sig = lo > 0 or hi < 0
    s = f"{s3(v)} [{s3(lo)}, {s3(hi)}]"
    return f"\\textbf{{{s}}}" if sig else s
GA, GT = A("gaze_memory.json")["features"], A("gaze_memory_train.json")["features"]
grow = []
for role, rl in (("rememberer_speaking", "Rememberer speaking"), ("partner_speaking", "Partner speaking")):
    for who, wl in (("self", "rememberer"), ("partner", "partner")):
        for match, ml in (("", "position"), ("_lenmatched", "position + length")):
            k = f"{role}{match}:{who}_away"
            grow.append(f"{rl} & {wl} & {ml} & {ci(GA.get(k))} & {ci(GT.get(k))} \\\\")
gt = ["\\begin{table*}[h]", "\\centering\\footnotesize", "\\begin{tabular}{@{}lllcc@{}}", "\\toprule",
      "Turns & Whose gaze & Matched on & Full sample & Random half of people \\\\", "\\midrule"] + grow + ["\\bottomrule", "\\end{tabular}",
      f"\\caption{{Looking away from the screen inside the remembered stretch minus outside it, in turns of the same fifth of the call (and, where marked, the same length bin), by who is speaking; proportion of frames, mean over rememberers with 95\\% bootstrap intervals over rememberers (bold: interval excludes zero). Base rates of looking away: rememberer {A('gaze_memory.json')['mean_away_rate']['self']:.3f}, partner {A('gaze_memory.json')['mean_away_rate']['partner']:.3f}.}}",
      "\\label{tab:gaze}", "\\end{table*}"]
open(os.path.join(OUT, "tab_gaze.tex"), "w").write("\n".join(gt) + "\n")

VA, VT = A("voice_memory.json")["features"], A("voice_memory_train.json")["features"]
VL = {"pitch": "Pitch level", "pitch_var": "Pitch variability", "loud": "Loudness", "loud_var": "Loudness variability", "voiced_rate": "Voiced segments per second",
      "pause_len": "Pause length", "jitter": "Jitter", "shimmer": "Shimmer", "hnr": "Harmonics-to-noise ratio", "slope": "Spectral slope (0--500 Hz)",
      "listener_loud": "Listener's loudness"}
vrow = []
for role, rl in (("rememberer_speaking", "Rememberer"), ("partner_speaking", "Partner")):
    for k, kl in VL.items():
        d, dt = VA.get(f"{role}:{k}"), VT.get(f"{role}:{k}")
        sd = f"{d['in_sd_units']:+.2f}".replace("-", "$-$") if d else "---"
        vrow.append(f"{rl} & {kl} & {ci(d)} & {sd} & {ci(dt)} \\\\")
    vrow.append("\\midrule")
vt = ["\\begin{table*}[h]", "\\centering\\footnotesize", "\\begin{tabular}{@{}llccc@{}}", "\\toprule",
      "Speaker & Feature & Full sample & In SD & Random half of people \\\\", "\\midrule"] + vrow[:-1] + ["\\bottomrule", "\\end{tabular}",
      "\\caption{Voice inside the remembered stretch minus outside it, on the speaker's own track, in turns of the same fifth of the call and the same length bin; each feature centered on the speaker's own mean in the call. Mean over rememberers with 95\\% bootstrap intervals (bold: interval excludes zero); ``In SD'' expresses the full-sample difference in standard deviations of that feature.}",
      "\\label{tab:voice}", "\\end{table*}"]
open(os.path.join(OUT, "tab_voice.tex"), "w").write("\n".join(vt) + "\n")
print("wrote tab_mem_ablation.tex tab_gaze.tex tab_voice.tex")
