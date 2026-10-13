#!/usr/bin/env python3
"""Lay Prolific coders (original_memory cohort, 12 synthetic AI-written memories, one lay coder each) vs our three AI coders on the
same items (analysis/code_v351/prolific_s2). Lay answers are mapped to codebook codes by exact option label (codebook v3.5.1 wording);
Q13 (reason sub-details) is skipped. Agreement per question = the codebook's own closeness score (correspondence_v351.score),
lay vs each AI coder, against AI vs AI on the same items. Descriptive only: 12 items. -> analysis/lay_vs_ai_s2.json"""
import collections, json, os
import numpy as np
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
d = json.load(open(os.path.join(ROOT, "prolific", "annotation_handoff_20261009", "ANNOTATIONS_DEIDENTIFIED_PRIVATE.json")))
QID = {"Q1": "topic", "Q2": "remembered", "Q3": "new_info", "Q4": "extent", "Q5": "details", "Q6": "moment_text", "Q7": "shared_by",
       "Q8": "personal_kinds", "Q9": "feelings", "Q10": "intensity", "Q12": "reason", "Q14": "matters_for", "Q15": "bigger_meaning"}
QS = {q["id"]: q for q in Q}
def code_for(qid, label):
    opts = QS[qid].get("options", [])
    for o in opts:
        if o["label"].strip().lower() == label.strip().lower() or label.strip().lower().startswith(o["label"].strip().lower()): return o["code"]
    raise KeyError((qid, label))
APP = {q["id"]: q for q in Q}["appraisal"]
SCALE = {o["label"].lower(): o["code"] for o in APP["scale"]}
miss = []
lay = {}
for r in d["reader_item_rows"]:
    if r["cohort"] != "original_memory": continue
    a = r["annotations"]; c = {}
    for qn, qid in QID.items():
        labs = a[qn]["labels"]
        try:
            codes = [code_for(qid, l) for l in labs]
        except KeyError as e:
            miss.append(str(e)); continue
        if qid == "reason": c["reason"] = {"about": codes[:1], "details": []}
        elif QS[qid]["type"] == "multi": c[qid] = codes
        else: c[qid] = str(codes[0]) if isinstance(codes[0], int) else codes[0]
    c["appraisal"] = {s: str(SCALE[a[f"Q11_{i+1}"]["labels"][0].lower()]) for i, s in enumerate(["pleasant", "unpleasant", "own_behavior", "other_behavior"])}
    lay[r["item_id"]] = norm(c)
AI = {cd: {k.split("|")[0]: v for k, v in load("prolific_s2", cd).items()} for cd in CODERS + ["deepseek-v4.1-flash"]}
keys = ["topic", "remembered", "new_info", "extent", "details", "moment_text", "shared_by", "personal_kinds", "feelings", "intensity",
        "appraisal.pleasant", "appraisal.unpleasant", "appraisal.own_behavior", "appraisal.other_behavior", "reason.first", "matters_for", "bigger_meaning"]
out = {"n_items": len(lay), "unmapped_labels": miss, "per_question": {}}
for k in keys:
    la, aa = [], []
    for it, lc in lay.items():
        for cd in AI:
            if it in AI[cd]:
                s = score(k, lc.get(k), AI[cd][it].get(k)); 
                if s is not None: la.append(s)
        cds = [cd for cd in AI if it in AI[cd]]
        for i in range(len(cds)):
            for j in range(i + 1, len(cds)):
                s = score(k, AI[cds[i]][it].get(k), AI[cds[j]][it].get(k))
                if s is not None: aa.append(s)
    out["per_question"][k] = {"lay_vs_ai": round(float(np.mean(la)), 3) if la else None, "ai_vs_ai": round(float(np.mean(aa)), 3) if aa else None, "n": len(la)}
out["mean_lay_vs_ai"] = round(float(np.mean([v["lay_vs_ai"] for v in out["per_question"].values() if v["lay_vs_ai"] is not None])), 3)
out["mean_ai_vs_ai"] = round(float(np.mean([v["ai_vs_ai"] for v in out["per_question"].values() if v["ai_vs_ai"] is not None])), 3)
json.dump(out, open(os.path.join(ROOT, "analysis", "lay_vs_ai_s2.json"), "w"), indent=1)
print("items", out["n_items"], "unmapped", miss[:5])
for k, v in out["per_question"].items(): print(f"  {k:26s} lay-AI {v['lay_vs_ai']}  AI-AI {v['ai_vs_ai']}  n {v['n']}")
print("mean", out["mean_lay_vs_ai"], out["mean_ai_vs_ai"])
