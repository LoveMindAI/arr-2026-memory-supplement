#!/usr/bin/env python3
"""Appendix voice table from the OWN-MIC run (analysis/voice_memory_v2.json): inside-minus-outside by speaker role, for the
eGeMAPS features (as before), Praat's pitch and intensity (second instrument) and the two emotion models; full sample and the
subset where both speakers recorded on their own machines. -> paper_v2/sections/app_parts/tab_voice_v2.tex (label tab:voice)."""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V = json.load(open(os.path.join(ROOT, "analysis", "voice_memory_v2.json"))); VA, VO = V["contrast_all"], V["contrast_own_machine_only"]
def ci(d):
    if not d: return "---"
    s = f"{d['inside_minus_outside']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if (d["ci95"][0] > 0 or d["ci95"][1] < 0) else s
GROUPS = [("openSMILE eGeMAPS", {"pitch": "Pitch level", "pitch_var": "Pitch variability", "loud": "Loudness", "loud_var": "Loudness variability",
                                 "voiced_rate": "Voiced segments per second", "pause_len": "Pause length", "jitter": "Jitter", "shimmer": "Shimmer",
                                 "hnr": "Harmonics-to-noise ratio", "slope": "Spectral slope (0--500 Hz)", "listener_loud": "Listener's loudness"}),
          ("Praat", {"praat_f0_st": "Pitch (F0, semitones)", "praat_f0_sd_st": "Pitch variability (F0 SD)", "praat_int_db": "Intensity (dB)", "praat_int_sd_db": "Intensity variability"}),
          ("audEERING (dimensional)", {"arousal": "Arousal", "dominance": "Dominance", "valence": "Valence"}),
          ("emotion2vec+", {"e2v_happy": "Happy", "e2v_neutral": "Neutral", "e2v_pos_minus_neg": "Positive minus negative"})]
rows = []
for role, rl in (("rememberer_speaking", "Rememberer"), ("partner_speaking", "Partner")):
    for g, feats in GROUPS:
        for k, kl in feats.items():
            d, do = VA.get(f"{role}:{k}"), VO.get(f"{role}:{k}")
            sd = f"{d['in_sd_units']:+.2f}".replace("-", "$-$") if d else "---"
            rows.append(f"{rl} & {g} & {kl} & {ci(d)} & {sd} & {ci(do)} \\\\")
    rows.append("\\midrule")
t = ["\\begin{table*}[h]", "\\centering\\scriptsize", "\\begin{tabular}{@{}lllccc@{}}", "\\toprule",
     "Speaker & Instrument & Feature & Full sample & In SD & Both on own machines \\\\", "\\midrule"] + rows[:-1] + ["\\bottomrule", "\\end{tabular}",
     f"\\caption{{Voice inside the remembered stretch minus outside it, measured on each speaker's own microphone, in turns of the same fifth of the call and the same length bin; each feature centered on the speaker's own mean in the call. Mean over rememberers ({VA['rememberer_speaking:loud']['n_people']} with the rememberer speaking, {VA['partner_speaking:loud']['n_people']} with the partner speaking) with 95\\% bootstrap intervals (bold: interval excludes zero); ``In SD'' expresses the full-sample difference in standard deviations of that feature. The last column repeats the contrast on the {V['n_perspectives_own_machine_only']} memories whose two speakers both recorded on their own machines.}}",
     "\\label{tab:voice}", "\\end{table*}"]
out = os.path.join(ROOT, "paper_v2", "sections", "app_parts", "tab_voice_v2.tex"); open(out, "w").write("\n".join(t) + "\n"); print("wrote", out, len(rows), "rows")
