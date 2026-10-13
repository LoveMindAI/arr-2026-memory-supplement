#!/usr/bin/env python3
"""Transcript-fidelity test: same five models, same people, same prompts and settings; Whisper transcripts (runs/main_text_cheap)
vs Scribe transcripts (runs/tx_v3_cheap). Per judge, model and condition: paired difference (v3 minus v2) in similarity, same focus
(whose moment) and exact-moment match, with 95% intervals from a bootstrap over people (the rememberer). Pooled over judges and
models too. Also the partner reference (the partner's own memory, judged against the person's), which does not depend on the
transcript, as the human yardstick. -> analysis/tx_v3_compare.json"""
import collections, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIVE = ["deepseek/deepseek-v4.1-flash", "google/gemini-3.8-flash", "google/gemma-4-31b-it", "xiaomi/mimo-v2.6-flash", "qwen/qwen3.8-omni-flash"]
SETS = {"five": ("main_text_cheap", "tx_v3_cheap", FIVE),
        "cheap2": ("main_text_cheap", "tx_v3_cheap2", ["z-ai/glm-5.3-flashx", "moonshotai/kimi-k2-0905", "xiaomi/mimo-v2.5", "xiaomi/mimo-v2.5-pro", "meta/muse-glimmer-30b"]),
        "pro": ("main_text_pro", "tx_v3_pro", ["openai/gpt-5.6-sol", "google/gemini-3.1-pro-preview", "xiaomi/mimo-v2.6-pro-ultraspeed"])}
WHICH = os.environ.get("SET", "five"); RUN2, RUN3, FIVE = SETS[WHICH]
JUDGES = sys.argv[1:] or ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]
MEAS = ["similarity", "same_focus", "event_match"]


def load(run):
    S = collections.defaultdict(dict)        # (pid, model, info) -> {judge: scores}
    for l in open(os.path.join(ROOT, "runs", run, "judged.jsonl")):
        j = json.loads(l)
        if j.get("kind") != "model" or not j.get("scores"):
            continue
        pid, model, mod, info = j["id"].split("|")[:4]
        if model in FIVE and mod == "text":
            S[(pid, model, info)][j["judge"]] = j["scores"]
    return S


def boot(d, people, B=2000, seed=0):
    rng = np.random.default_rng(seed); g = collections.defaultdict(list)
    for v, p in zip(d, people): g[p].append(v)
    G = list(g.values()); s = np.array([sum(x) for x in G]); n = np.array([len(x) for x in G])
    bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(G), len(G)) for _ in range(B))]
    return [round(float(np.mean(d)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), len(d), len(G)]


v2, v3 = load(RUN2), load(RUN3)
out = {"judges": JUDGES, "cells": {}, "pooled": {}, "levels": {}}
pool = collections.defaultdict(lambda: ([], []))
for info in ("none", "history"):
    for m in FIVE:
        for jd in JUDGES:
            for meas in MEAS:
                d, ppl, a, b = [], [], [], []
                for (pid, model, inf), sc3 in v3.items():
                    if model != m or inf != info or jd not in sc3:
                        continue
                    sc2 = v2.get((pid, model, inf), {}).get(jd)
                    if sc2 is None or sc2.get(meas) is None or sc3[jd].get(meas) is None:
                        continue
                    d.append(float(sc3[jd][meas]) - float(sc2[meas])); a.append(float(sc2[meas])); b.append(float(sc3[jd][meas]))
                    ppl.append(pid.split("+")[0])
                if d:
                    out["cells"][f"{info}|{m.split('/')[1]}|{jd.split('/')[1]}|{meas}"] = dict(v2=round(float(np.mean(a)), 4), v3=round(float(np.mean(b)), 4), diff=boot(d, ppl))
                    pool[(info, meas)][0].extend(d); pool[(info, meas)][1].extend(ppl)
for (info, meas), (d, ppl) in pool.items():
    out["pooled"][f"{info}|{meas}"] = boot(d, ppl)
out["note"] = "diff = v3 (Scribe transcript) minus v2 (Whisper transcript), paired on person x model x condition x judge; [mean, lo, hi, n pairs, n people]; pooled = all five models and listed judges"
json.dump(out, open(os.path.join(ROOT, "analysis", f"tx_v3_compare_{WHICH}.json"), "w"), indent=1)
for k, v in sorted(out["pooled"].items()):
    print(k, v)
