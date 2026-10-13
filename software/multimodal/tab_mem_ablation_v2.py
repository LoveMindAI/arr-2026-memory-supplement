#!/usr/bin/env python3
"""tab_mem_ablation with the v2 instruments: gaze = UniGaze + PureGaze (ETH-XGaze) + head pose on each speaker's own full-resolution
camera (analysis/memorability_gaze_v2nets.json); voice = own-microphone eGeMAPS (analysis/memorability_voice_ownmic.json), plus a row with
the audEERING arousal/dominance/valence added (memorability_voice_ownmic_emo.json). Same code as scripts/tab_memorability_appendix.py.
-> paper_v2/sections/app_parts/tab_mem_ablation_v2.tex"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
OUT = os.path.join(ROOT, "paper_v2", "sections", "app_parts")
M, G, V, VE = A("memorability.json"), A("memorability_gaze_v2nets.json"), A("memorability_voice_ownmic.json"), A("memorability_voice_ownmic_emo.json")
LAB = {"position": "Position in the call", "speaker": "Who is speaking", "length & timing": "Length and timing", "lexical": "Words used",
       "semantic novelty": "How new the topic is", "sentiment": "Sentiment", "dialogue structure": "Dialogue structure",
       "content": "Content (sentence embedding)", "rememberer traits": "Rememberer's personality scores",
       "gaze": "Gaze (two networks and head pose)", "voice": "Voice (own microphone)", "voice_emo": "Voice with vocal emotion"}
f3 = lambda x: f"{x:.3f}"; s3 = lambda x: ("$-$" if x < 0 else "+") + f"{abs(x):.3f}"
rows = []
for fam in M["families"] + ["gaze", "voice", "voice_emo"]:
    src, key = (G, "gaze") if fam == "gaze" else (V, "voice") if fam == "voice" else (VE, "voice") if fam == "voice_emo" else (M, fam)
    lr, rf = src["ablation"]["LR"]["families"][key], src["ablation"]["RF"]["families"][key]
    rows.append(f"{LAB[fam]} & {f3(lr['alone'])} & {f3(rf['alone'])} & {s3(lr['drop_without'])} & {s3(rf['drop_without'])} \\\\")
t = ["\\begin{table}[h]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
     " & \\multicolumn{2}{c}{Alone} & \\multicolumn{2}{c}{Drop when removed} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
     "Feature family & LR & RF & LR & RF \\\\", "\\midrule"] + rows[:9] + ["\\midrule"] + rows[9:] + [
     "\\midrule", f"All nine families (LR 100, RF 20 splits) & \\multicolumn{{2}}{{c}}{{{f3(M['ablation']['LR']['all_families'])} \\,/\\, {f3(M['ablation']['RF']['all_families'])}}} & & \\\\",
     f"Dummy (base rate) & \\multicolumn{{2}}{{c}}{{{f3(M['models']['Dummy']['bal_acc'])}}} & & \\\\",
     "\\bottomrule", "\\end{tabular}",
     f"\\caption{{Which kinds of features find remembered turns. Balanced accuracy of each feature family on its own, and the loss when it is removed from all the others, for logistic regression (LR, 100 splits) and random forest (RF, 20 splits) holding out whole groups; {M['units']:,} turns in {M['reports']:,} located memories, {M['positive_rate']*100:.0f}\\% of turns in a remembered stretch. Gaze and voice rows come from runs that add that family to the nine text families; gaze is measured on each speaker's own full-resolution camera with two gaze networks and head pose, voice on each speaker's own microphone.}}",
     "\\label{tab:mem_ablation}", "\\end{table}"]
open(os.path.join(OUT, "tab_mem_ablation_v2.tex"), "w").write("\n".join(t) + "\n"); print("wrote tab_mem_ablation_v2.tex")
