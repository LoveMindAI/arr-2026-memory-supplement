#!/usr/bin/env python3
"""Appendix tables: the predicting models (paper_v2/sections/app_parts/tab_models.tex) and fine-tuning / preference training
(tab_dpo.tex). Sources: analysis/model_results.json (reports scored per model), runs/*/preds.jsonl (which models ran which condition),
analysis/modality.json (media models), analysis/dpo_results.json and analysis/dpo_onpol_results.json (paired changes, lengths),
runs/memft1_eval via finetune/score_memft.py (SFT, recomputed here by importing its loader)."""
import collections, json, os, subprocess, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper_v2", "sections", "app_parts"); os.makedirs(OUT, exist_ok=True)
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
NAMES = {"deepseek-v4.1-flash": "DeepSeek v4.1 Flash", "gemini-3.8-flash": "Gemini 3.8 Flash", "gemma-4-31b-it": "Gemma 4 31B", "glm-5.3-flashx": "GLM-5.3 FlashX",
         "kimi-k2-0905": "Kimi K2 (0905)", "mimo-v2.5": "MiMo v2.5", "mimo-v2.5-pro": "MiMo v2.5 Pro", "mimo-v2.6-flash": "MiMo v2.6 Flash",
         "muse-glimmer-30b": "Muse Glimmer 30B", "qwen3.8-omni-flash": "Qwen3.8 Omni Flash", "gemini-3.1-pro-preview": "Gemini 3.1 Pro (preview)",
         "gpt-5.6-sol": "GPT-5.6 Sol", "mimo-v2.6-pro-ultraspeed": "MiMo v2.6 Pro UltraSpeed"}

def ran(run):
    s = collections.defaultdict(set)
    p = os.path.join(ROOT, "runs", run, "preds.jsonl")
    for l in open(p):
        x = json.loads(l)
        if x.get("ok"): s[x["model"]].add(x["info"])
    return s

prov = {}
info = collections.defaultdict(set)
for run in ("main_text_cheap", "main_text_pro", "cards_text_cheap", "cards_narr_cheap", "cards_text_pro", "cards_narr_pro", "norms_note_v2",
            "r3c_deepseek", "r3c_gemini", "r3c_gemma", "r3c_mimo", "r3c_qwen3.8", "position_swap"):
    for m, inf in ran(run).items():
        short = m.split("/")[-1]; prov[short] = m.split("/")[0]; info[short] |= inf | ({"swap"} if run == "position_swap" else set())
MR = {m["model"]: m["n"] for m in A("model_results.json")["text"]}
MED = set(A("modality.json")["per_model"])
L = [r"\begin{table*}[h]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrccccc@{}}", r"\toprule",
     r"Model & Provider & Reports & Profiles & Controls & Note & Audio, video & Position swap \\", r"\midrule"]
yes = lambda b: r"\textbullet" if b else ""
for m in sorted(MR, key=lambda k: (-MR[k], NAMES.get(k, k))):
    i = info[m]
    L.append(f"{NAMES.get(m, m)} & {prov.get(m, '')} & {MR[m]:,} & {yes('card_narr' in i)} & {yes('history_wrong' in i)} & {yes('norms' in i)} & {yes(m in MED)} & {yes('swap' in i)}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{The thirteen predicting models. Reports: memories scored with no information and with history (all 1,147, or a seeded 300-report sample stratified by group). Profiles: scores and written profiles on every report with questionnaire scores. Controls: a stranger's history or profile (someone from another session who never met the partner, with as many memories), and earlier calls only. Note: the note on how people answer, on the held-out half. Audio, video: the 300-report sample, as audio alone, transcript and audio, and transcript, audio and video. Position swap: 52 calls.}",
      r"\label{tab:models}", r"\end{table*}"]
open(os.path.join(OUT, "tab_models.tex"), "w").write("\n".join(L) + "\n")

# fine-tuning and preference training
s3 = lambda x: ".000" if round(x, 3) == 0 else ("+" if x >= 0 else "$-$") + f"{abs(x):.3f}".lstrip("0")
ci = lambda v: f"{s3(v['mean'])} [{s3(v['lo'])}, {s3(v['hi'])}]"
LRSUF = r"\,e$-$5"
D = A("dpo_results.json"); O = A("dpo_onpol_results.json")
L = [r"\begin{table*}[h]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lllcccr@{}}", r"\toprule",
     r"Run & lr & Against & Whose moment & Exact moment & Similarity & Words \\", r"\midrule",
     r"\multicolumn{7}{@{}l}{\emph{Round 1: the person's real answer preferred over a human one}} \\"]
for tag, base, bn in (("dpo_history_lr2e-05", "base_history", "untuned"), ("dpo_history_lr5e-05", "base_history", "untuned"),
                      ("dpo_card_lr2e-05", "base_card", "untuned"), ("dpo_card_lr5e-05", "base_card", "untuned")):
    lab = ("History" if "history" in tag else "Profile") + " & " + tag.split("lr")[1].replace("e-05", LRSUF)
    P = D["paired"]
    L.append(f"{lab} & {bn} ({D['stats'][base]['moment_words_median']:.0f} w.) & {ci(P[tag + '|test|same_focus'])} & {ci(P[tag + '|test|event_match'])} & {ci(P[tag + '|test|similarity'])} & {D['stats'][tag]['moment_words_median']:.0f}" + r" \\")
L.append(r"\midrule \multicolumn{7}{@{}l}{\emph{Round 2: start from the round-1 model (history, lr 5\,e$-$5); its own guesses as counterexamples}} \\")
P = O["paired"]
for arm, an in (("onpol", "Guesses rejected"), ("mixed", "Guesses + human"), ("self", "Own right guess")):
    for lr in ("2e-05", "5e-05"):
        tag = f"dpo_{arm}_lr{lr}"
        lrs = lr.replace("e-05", LRSUF)
        for vs, vn in (("vs_base_history", "untuned"), ("vs_r1_history", "round 1")):
            L.append(f"{an} & {lrs} & {vn} & {ci(P[f'{tag}|{vs}|test|same_focus'])} & {ci(P[f'{tag}|{vs}|test|event_match'])} & {ci(P[f'{tag}|{vs}|test|similarity'])} & {O['stats'][tag]['moment_words_median']:.0f}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\caption{Preference training (DPO) on Gemma 4 31B, test people only (219 reports; nobody in training). Changes in the three judges' scores against the comparison model, with 95\% person-clustered intervals. Run: History and Profile are the first-round conditions; the second-round arms are described in the text. Against: the untuned model (with its median length in words, w.) or the first-round model. Words: median length of the predicted moment (people: 16).}",
      r"\label{tab:dpo}", r"\end{table*}"]
open(os.path.join(OUT, "tab_dpo.tex"), "w").write("\n".join(L) + "\n")
print("wrote tab_models.tex, tab_dpo.tex")
