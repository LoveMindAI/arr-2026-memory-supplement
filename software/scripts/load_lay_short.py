#!/usr/bin/env python3
"""Lay coders, short-report wave (agent2's handoff prolific/v6_short/annotation_handoff_private): convert each reader's Prolific answers
into codebook v3.5.1 codes, in the same file format load_human_coders.py writes (<key>__human-<reader>.json), and copy the three AI
coders' files for the same items (analysis/code_v351/prolific_short). Then run human_agreement.py on the folder.
Mapping is by exact option label (the form's labels are the codebook's); "Skip this question" -> missing; Q12 = main reason
(reason.about[0]); Q13 "[Parent] sub" -> reason.details (sub codes), parents appended to reason.about after the main one.
No answers are invented; missing readers stay missing. -> analysis/code_v351/lay_short/"""
import json, os, shutil, sys, importlib, glob
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "codebook"))
SPEC = importlib.import_module("spec_v351_FROZEN").SPEC
Q = {q["id"]: q for p in SPEC["parts"] for q in p["questions"]}
H = os.path.join(ROOT, "prolific", "v6_short", "annotation_handoff_private", "ANNOTATIONS_DEIDENTIFIED_PRIVATE.json")
OUT = os.path.join(ROOT, "analysis", "code_v351", "lay_short"); os.makedirs(OUT, exist_ok=True)
low = lambda s: s.strip().lower()
def opt(qid, label):
    for o in Q[qid].get("options", []) + Q[qid].get("scale", []):
        if low(o["label"]) == low(label) or low(label).startswith(low(o["label"])): return o["code"]
    raise KeyError((qid, label))
QID = {"Q1": "topic", "Q2": "remembered", "Q3": "new_info", "Q4": "extent", "Q5": "details", "Q6": "moment_text", "Q7": "shared_by",
       "Q8": "personal_kinds", "Q9": "feelings", "Q10": "intensity", "Q14": "matters_for", "Q15": "bigger_meaning"}
APP = ["pleasant", "unpleasant", "own_behavior", "other_behavior"]
REASON = Q["reason"]["options"]
def sub_code(label):
    p, s = label[1:].split("] ", 1)
    for o in REASON:
        if low(o["label"]) == low(p):
            for su in o.get("sub", []):
                if low(su["label"]) == low(s): return o["code"], su["code"]
    raise KeyError(label)
rows = json.load(open(H))["reader_item_rows"]; n = 0; skipped = 0
for r in rows:
    a = r["annotations"]; c = {}
    for qn, qid in QID.items():
        labs = [l for l in (a.get(qn, {}).get("labels") or []) if low(l) != "skip this question"]
        if not labs: skipped += 1; continue
        codes = [opt(qid, l) for l in labs]
        c[qid] = codes if Q[qid]["type"] == "multi" else codes[0]
    c["appraisal"] = {}
    for i, s in enumerate(APP):
        labs = [l for l in (a.get(f"Q11_{i+1}", {}).get("labels") or []) if low(l) != "skip this question"]
        if labs: c["appraisal"][s] = opt("appraisal", labs[0])
    main = [l for l in (a.get("Q12", {}).get("labels") or []) if low(l) != "skip this question"]
    det = [sub_code(l) for l in (a.get("Q13", {}).get("labels") or []) if l.startswith("[")]
    if main:
        about = [opt("reason", main[0])] + [p for p, _ in det if p != opt("reason", main[0])]
        c["reason"] = {"about": list(dict.fromkeys(about)), "details": [s for _, s in det]}
    json.dump({"key": r["item_id"], "coder": f"human-{r['reader_code']}", "codes": c, "ok": True, "item": r["item_id"], "source": "prolific v6_short"},
              open(os.path.join(OUT, f"{r['item_id']}__human-{r['reader_code']}.json"), "w"), indent=1); n += 1
items = {r["item_id"] for r in rows}
for f in glob.glob(os.path.join(ROOT, "analysis", "code_v351", "prolific_short", "*__*.json")):
    j = json.load(open(f))
    if os.path.basename(f).split("__")[0] in items and j.get("ok") and isinstance(j.get("codes"), dict): shutil.copy(f, OUT)   # failed AI calls (DeepSeek S25, S28, S45) stay missing
print(f"{n} lay reader-item files for {len(items)} items; skipped answers {skipped}; AI files copied -> {OUT}")
