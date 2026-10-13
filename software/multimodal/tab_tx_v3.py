#!/usr/bin/env python3
"""Consolidated transcript ablation table: all 13 predicting models, without and with history, Whisper (v2) vs Scribe (v3)
transcripts, judged by the same three judges (Luna, DeepSeek with GLM on DeepSeek's own predictions, Haiku). Per model x condition:
mean over judges of similarity, same focus (whose moment) and exact moment for each transcript, and the paired difference with a
95% interval from a bootstrap over rememberers. -> analysis/tx_v3_all.json, paper_v2/sections/app_parts/tab_tx_v3.tex"""
import collections, json, os
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JUDGES = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]
MEAS = [("similarity", "Similarity"), ("same_focus", "Whose moment"), ("event_match", "Exact moment")]
GROUPS = [("main_text_cheap", "tx_v3_cheap", ["deepseek/deepseek-v4.1-flash", "google/gemini-3.8-flash", "google/gemma-4-31b-it", "xiaomi/mimo-v2.6-flash", "qwen/qwen3.8-omni-flash"]),
          ("main_text_cheap", "tx_v3_cheap2", ["z-ai/glm-5.3-flashx", "moonshotai/kimi-k2-0905", "xiaomi/mimo-v2.5", "xiaomi/mimo-v2.5-pro", "meta/muse-glimmer-30b"]),
          ("main_text_pro", "tx_v3_pro", ["openai/gpt-5.6-sol", "google/gemini-3.1-pro-preview", "xiaomi/mimo-v2.6-pro-ultraspeed"])]
NICE = {"deepseek/deepseek-v4.1-flash": "DeepSeek v4.1 Flash", "google/gemini-3.8-flash": "Gemini 3.8 Flash", "google/gemma-4-31b-it": "Gemma 4 31B",
        "xiaomi/mimo-v2.6-flash": "MiMo v2.6 Flash", "qwen/qwen3.8-omni-flash": "Qwen3.8 Omni Flash", "z-ai/glm-5.3-flashx": "GLM-5.3 FlashX",
        "moonshotai/kimi-k2-0905": "Kimi K2 (0905)", "xiaomi/mimo-v2.5": "MiMo v2.5", "xiaomi/mimo-v2.5-pro": "MiMo v2.5 Pro",
        "meta/muse-glimmer-30b": "Muse Glimmer 30B", "openai/gpt-5.6-sol": "GPT-5.6 Sol", "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro (preview)",
        "xiaomi/mimo-v2.6-pro-ultraspeed": "MiMo v2.6 Pro UltraSpeed"}


def load(run, models):
    S = collections.defaultdict(dict)
    p = os.path.join(ROOT, "runs", run, "judged.jsonl")
    if not os.path.exists(p): return S
    for l in open(p):
        j = json.loads(l)
        if j.get("kind") != "model" or not j.get("scores"): continue
        pid, model, mod, info = j["id"].split("|")[:4]
        if model in models and mod == "text": S[(pid, model, info)][j["judge"]] = j["scores"]   # last non-empty line wins
    return S


def boot(d, people, B=2000, seed=0):
    rng = np.random.default_rng(seed); g = collections.defaultdict(list)
    for v, p in zip(d, people): g[p].append(v)
    G = list(g.values()); s = np.array([sum(x) for x in G]); n = np.array([len(x) for x in G])
    bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(G), len(G)) for _ in range(B))]
    return [round(float(np.mean(d)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), len(d), len(G)]


out = {"judges": JUDGES, "rows": {}, "pooled": {}}
pool = collections.defaultdict(lambda: ([], []))
for r2, r3, models in GROUPS:
    v2, v3 = load(r2, set(models)), load(r3, set(models))
    for m in models:
        for info in ("none", "history"):
            row = {}
            for meas, _ in MEAS:
                d, ppl, a, b = [], [], [], []
                for (pid, model, inf), sc3 in v3.items():
                    if model != m or inf != info: continue
                    sc2 = v2.get((pid, model, inf), {})
                    for jd in JUDGES:
                        x, y = sc2.get(jd, {}).get(meas), sc3.get(jd, {}).get(meas)
                        if x is None or y is None: continue
                        d.append(float(y) - float(x)); a.append(float(x)); b.append(float(y)); ppl.append(pid.split("+")[0])
                if d:
                    row[meas] = dict(v2=round(float(np.mean(a)), 4), v3=round(float(np.mean(b)), 4), diff=boot(d, ppl))
                    pool[(info, meas)][0].extend(d); pool[(info, meas)][1].extend(ppl)
            if row: out["rows"][f"{m}|{info}"] = row
for (info, meas), (d, ppl) in pool.items(): out["pooled"][f"{info}|{meas}"] = boot(d, ppl)
json.dump(out, open(os.path.join(ROOT, "analysis", "tx_v3_all.json"), "w"), indent=1)

def cell(r):
    if not r: return "--- & ---"
    lo, hi = r["diff"][1], r["diff"][2]; s = f"{r['diff'][0]:+.3f}".replace("-", "$-$")
    s = f"\\textbf{{{s}}}" if (lo > 0 or hi < 0) else s
    return f"{r['v2']:.2f}$\\to${r['v3']:.2f} & {s}"
lines = []
for info, il in (("none", "No information about the person"), ("history", "With the person's earlier memories")):
    lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{il}}}}} \\\\")
    for _, _, models in GROUPS:
        for m in models:
            r = out["rows"].get(f"{m}|{info}", {})
            lines.append(f"{NICE[m]} & " + " & ".join(cell(r.get(meas)) for meas, _ in MEAS) + " \\\\")
    pl = [out["pooled"].get(f"{info}|{meas}") for meas, _ in MEAS]
    lines.append("\\textit{All models} & " + " & ".join((f"& {p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}]".replace("-", "$-$") if p else "& ---") for p in pl) + " \\\\")
    lines.append("\\midrule")
t = ["\\begin{table*}[h]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
     " & \\multicolumn{2}{c}{" + MEAS[0][1] + "} & \\multicolumn{2}{c}{" + MEAS[1][1] + "} & \\multicolumn{2}{c}{" + MEAS[2][1] + "} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}", "Model & Whisper$\\to$Scribe & Change & Whisper$\\to$Scribe & Change & Whisper$\\to$Scribe & Change \\\\", "\\midrule"] + lines[:-1] + [
     "\\bottomrule", "\\end{tabular}",
     "\\caption{The transcriber as a variable. Each model predicted every memory twice, once from the Whisper transcript and once from the ElevenLabs Scribe transcript of the same audio, with the same prompts and settings; three judges scored both against the person's report. Cells give the mean over judges for each transcript and the paired change (Scribe minus Whisper), bold where its 95\\% interval (bootstrap over rememberers) excludes zero; the last row of each panel pools all models with its interval.}",
     "\\label{tab:tx_ablation}", "\\end{table*}"]
open(os.path.join(ROOT, "paper_v2", "sections", "app_parts", "tab_tx_v3.tex"), "w").write("\n".join(t) + "\n")
for k, v in sorted(out["pooled"].items()): print(k, v)
print("rows", len(out["rows"]))
