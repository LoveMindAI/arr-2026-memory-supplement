#!/usr/bin/env python3
"""Code model-predicted memories with the SAME codebook prompt as the human memories (code_v1.PROMPT; GPT-6-Luna, one memory per request).
Scope (Oct 7, cost/time): 5 models x the 300-memory sample x 10 conditions (none, history, card_thin, card_narr, card_narr_history,
card_facts, card_factnarr, card_weave, card_impthin, card_impnarr). Output analysis/code_preds/<key>__gpt-6-luna.json."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import code_v1 as C
ROOT = C.ROOT; C.OUT = os.path.join(ROOT, "analysis", "code_preds"); os.makedirs(C.OUT, exist_ok=True)
M5 = {"deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"}
CONDS = {"none", "history", "card_thin", "card_narr", "card_narr_history", "card_facts", "card_factnarr", "card_weave", "card_impthin", "card_impnarr"}
pro = set(json.load(open(os.path.join(ROOT, "runs", "MAIN_SETS.json")))["pro300"])
jobs = []
for run in ["main_text_cheap", "cards_text_cheap", "cards_narr_cheap", "r2_facts_impthin", "r2_weave_factnarr", "r2_weave_factnarr_impnarr"]:
    for l in open(os.path.join(ROOT, "runs", run, "preds.jsonl")):
        j = json.loads(l); m = j["model"].split("/")[1]
        if j.get("ok") and m in M5 and j["info"] in CONDS and j["perspectiveID"] in pro and j["modality"] == "text":
            pr = j["pred"]; jobs.append((f"{j['perspectiveID']}|{m}|{j['info']}", str(pr.get("moment", "")).strip(), str(pr.get("reason", "")).strip(), "openai/gpt-6-luna"))
jobs = list({k[0]: k for k in jobs}.values())
print("jobs", len(jobs), flush=True)
tot = 0.0; bad = 0
with ThreadPoolExecutor(64) as ex:
    for c, ok in ex.map(C.job, jobs):
        tot += c; bad += not ok
print("coded", len(jobs), "failed", bad, "cost $", round(tot, 2))
