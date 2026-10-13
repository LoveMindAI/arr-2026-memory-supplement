#!/usr/bin/env python3
"""Appendix gaze table from the v2 instruments (analysis/gaze_memory_v2.json): looking away from the screen inside the remembered
stretch minus outside, by who is speaking and whose eyes, per instrument at the primary 15-degree threshold; position-matched and
position x length-matched. -> paper_v2/sections/app_parts/tab_gaze_v2.tex (label tab:gaze)."""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = json.load(open(os.path.join(ROOT, "analysis", "gaze_memory_v2.json"))); F = D["features"]
INS = [("mp", "MediaPipe (head and eyes)"), ("mp_head", "MediaPipe (head pose only)"), ("ug_15", "UniGaze"), ("pg_15", "PureGaze (ETH-XGaze)"), ("both_15", "Both networks agree")]
def ci(d):
    if not d: return "---"
    s = f"{100*d['inside_minus_outside']:+.1f} [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}]".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if (d["ci95"][0] > 0 or d["ci95"][1] < 0) else s
rows = []
for role, rl in (("partner_speaking", "Partner speaking"), ("rememberer_speaking", "Rememberer speaking")):
    for who, wl in (("partner", "Partner's eyes"), ("self", "Rememberer's eyes")):
        for k, kl in INS:
            a, b = F.get(f"{role}:{who}_{k}"), F.get(f"{role}_lenmatched:{who}_{k}")
            if a or b: rows.append(f"{rl} & {wl} & {kl} & {ci(a)} & {ci(b)} \\\\")
        rows.append("\\addlinespace")
    rows.append("\\midrule")
t = ["\\begin{table*}[h]", "\\centering\\scriptsize", "\\begin{tabular}{@{}lllcc@{}}", "\\toprule",
     "Who is speaking & Whose eyes & Instrument & Matched on position & Matched on position and length \\\\", "\\midrule"] + rows[:-2] + ["\\bottomrule", "\\end{tabular}",
     f"\\caption{{Looking away from the screen inside the remembered stretch minus outside it, in percentage points of frames, measured on each speaker's own full-resolution camera at 3 frames per second. A frame counts as away if no face is found or the measure departs from the person's own median in the call by more than 15 degrees (eye-look score by more than 0.3 for MediaPipe); each person's away rate is centered on their own mean in the call. Mean over rememberers with 95\\% bootstrap intervals (bold: interval excludes zero). Mean away rates differ by instrument (MediaPipe {100*D['mean_away_rate']['mp']:.0f}\\%, UniGaze {100*D['mean_away_rate']['ug_15']:.0f}\\%, PureGaze {100*D['mean_away_rate']['pg_15']:.0f}\\%, both {100*D['mean_away_rate']['both_15']:.0f}\\%).}}",
     "\\label{tab:gaze}", "\\end{table*}"]
out = os.path.join(ROOT, "paper_v2", "sections", "app_parts", "tab_gaze_v2.tex"); open(out, "w").write("\n".join(t) + "\n"); print("wrote", out, len(rows))
