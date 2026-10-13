#!/usr/bin/env python3
"""Item 3 (A1, Oct 9): locate the five main models' predicted moments (no history) in the same transcripts, with the SAME locator
prompt and model as the people's memories (locate_moments.py; the locator is blind to who wrote the memory). Lets the memorability
classifier ask whether models pick moments for the same reasons people do. Cap $40. -> analysis/located_preds/<model>/<pid>.json"""
import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
import orc, prompts as P, locate_moments as LM
INFO = os.environ.get("INFO", "none")   # Oct 11 (A1 OK): INFO=history locates the history-condition guesses -> analysis/located_preds_history/
SUB = ("located_preds" if INFO == "none" else f"located_preds_{INFO}") + os.environ.get("LOC_TAG", "")   # LOC_TAG: '_g3f', '_luna' for the second and third locators
ROOT = ".."; CAP = float(os.environ.get("CAP", 40)); spent = [0.0]; lock = threading.Lock()
M5 = {"deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"}
rows = {r["perspectiveID"]: r for r in P.rows()}
preds = []
SRC = [os.path.join(ROOT, "runs", "main_text_cheap", "preds.jsonl")] if INFO in ("none", "history") else [os.path.join(ROOT, "runs", f"r3c_{m}", "preds.jsonl") for m in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")]
for l in (x for f in SRC for x in open(f)):
    j = json.loads(l)
    if j.get("ok") and j["info"] == INFO and j["model"].split("/")[1] in M5 and j.get("modality", "text") == "text" and j["perspectiveID"] != "PID_n+PID_n":
        preds.append(j)
def job(j):
    m = j["model"].split("/")[1]; d = os.path.join(ROOT, "analysis", SUB, m); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, j["perspectiveID"] + ".json")
    if os.path.exists(path): return
    with lock:
        if spent[0] > CAP: return
    r = rows[j["perspectiveID"]]; tx, turns = LM.numbered(r["dyadID"], r["selfPID"])
    msg = [{"role": "user", "content": LM.PROMPT.format(tx=tx, mem=str(j["pred"].get("moment", "")), why=str(j["pred"].get("reason", "")))}]
    c = orc.call(LM.MODEL, msg, os.path.join(ROOT, "analysis", SUB, "calls.jsonl"), tag=f"{m}|{j['perspectiveID']}", max_tokens=3000, temperature=0,
                 response_format={"type": "json_object"}, reasoning={"effort": "low"})
    with lock: spent[0] += orc.cost(c)
    t = c.get("text") or ""
    try: o = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception: return
    idx = [i for i in (o.get("turns") or []) if isinstance(i, int) and 0 <= i < len(turns)]
    json.dump(dict(perspectiveID=j["perspectiveID"], model=m, ok=True, kind=o.get("kind"), turns=idx, confidence=o.get("confidence"), n_turns=len(turns),
                   pred_timing=j["pred"].get("timing"), note=o.get("note")), open(path, "w"))
with ThreadPoolExecutor(int(os.environ.get('CONC', 24))) as ex: list(ex.map(job, preds))
print("predictions", len(preds), "spent $", round(spent[0], 2))
