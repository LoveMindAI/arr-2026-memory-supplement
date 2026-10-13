#!/usr/bin/env python3
"""Plain-statistics check (A1, Oct 11: "anything else we can do ... with just the data rather than LLMs that would strengthen the
objective measurements"). Step 1: collect every text the paper's prediction table scores, plus the human reports, into one file.
Same reports as the codebook table (the analyzed 1,147, analysis/code_v351/human/FINAL.jsonl keys); same prediction runs as
ladder_v351.py / correspondence sets; all models present in each run (M5 flag marks the five with controls). -> analysis/plainstats/texts.json"""
import json, os, collections
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "analysis", "plainstats")
M5 = {"deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"}
keep = [json.loads(l)["key"].split("|")[0] for l in open(os.path.join(ROOT, "analysis", "code_v351", "human", "FINAL.jsonl"))]
R = {r["perspectiveID"]: r for r in P.rows()}
human = {pid: dict(moment=R[pid]["conv_SP_Mem_descr"].strip(), reason=R[pid]["conv_SP_Mem_reason"].strip(), selfPID=R[pid]["selfPID"], otherPID=R[pid]["otherPID"],
                   dyadID=R[pid]["dyadID"], batchID=R[pid]["batchID"], round=int(R[pid]["roundIdx"])) for pid in keep if pid in R}
RUNS = [("main_text_cheap", ("none", "history")), ("norms_note_v2", ("norms",)), ("norms_history_v2", ("norms_history",)),
        ("cards_text_cheap", ("card_thin",)), ("cards_narr_cheap", ("card_narr",))] + \
       [(f"r3c_{m}", ("history_prior", "history_wrong", "card_thin_wrong", "card_narr_wrong")) for m in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")]
preds = []; seen = set()
for run, infos in RUNS:
    for l in open(os.path.join(ROOT, "runs", run, "preds.jsonl")):
        j = json.loads(l)
        if not j.get("ok") or j["info"] not in infos or j.get("modality", "text") != "text" or j["perspectiveID"] not in human: continue
        m = j["model"].split("/")[1]; k = (j["perspectiveID"], m, j["info"])
        if k in seen: continue
        seen.add(k); p = j.get("pred") or {}
        preds.append(dict(pid=j["perspectiveID"], model=m, cond=j["info"], m5=m in M5, moment=str(p.get("moment", "")).strip(), reason=str(p.get("reason", "")).strip()))
json.dump(dict(human=human, preds=preds), open(os.path.join(OUT, "texts.json"), "w"))
print("human", len(human), "preds", len(preds)); print(collections.Counter((p["cond"], p["m5"]) for p in preds))
