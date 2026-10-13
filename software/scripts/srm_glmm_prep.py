#!/usr/bin/env python3
"""Long table for the social-relations analysis on codebook v3.5.1 consensus codes (analysis/code_v351/human/FINAL.jsonl).
One row per report: key, batch, rememberer, partner, then outcomes:
  binary  b__<question>__<option>  (single-choice one-hot; multi-select membership; reason first choice; MeMo labels rebuilt)
  numeric n__details, n__intensity, n__appraisal_<statement>  (median of the coders; Can't tell = missing)
Multi-select and nominal codes in FINAL are already consensus; appraisal and reason are per-coder lists, combined here:
appraisal = median of the numeric answers; reason first choice = the answer most coders gave first (ties -> missing);
MeMo label = present if most coders' details rebuild to it. Options chosen in <5% or >95% of reports are dropped (too rare to model).
Writes analysis/srm_glmm/long.csv and outcomes.json."""
import collections, csv, json, os, sys, importlib
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "codebook")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scope_merge as SM
SPEC = SM.spec(importlib.import_module("spec_v351_FROZEN").SPEC)
import prompts as P
Q = {q["id"]: q for p in SPEC["parts"] if p.get("models") is not False for q in p["questions"]}
MEMO = {s["code"]: s.get("memo") for q in Q.values() if q["type"] == "nested_ranked" for o in q["options"] for s in o.get("sub", [])}
up = lambda x: str(x).strip().upper()
OUT = os.path.join(ROOT, "analysis", "srm_glmm"); os.makedirs(OUT, exist_ok=True)

rows = {r["perspectiveID"]: r for r in P.rows()}
recs = []
for line in open(os.path.join(ROOT, "analysis", "code_v351", "human", "FINAL.jsonl")):
    j = json.loads(line); c = SM.codes(j["codes"]); r = rows.get(j["key"])
    if not r: continue
    d = {"key": j["key"], "batch": r["batchID"], "rem": r["selfPID"], "par": r["otherPID"]}
    for qid, q in Q.items():
        v = c.get(qid)
        if isinstance(v, str) and up(v) == "TIE": v = None      # coders split with no majority after escalation
        if isinstance(v, list): v = [x for x in v if up(x) != "TIE"] or None
        if q["type"] == "single":
            v = v[0] if isinstance(v, list) and len(v) == 1 else v
            ordered = all(isinstance(o["code"], int) for o in q["options"] if o["code"] != "CANT_TELL")
            if ordered:
                d[f"n__{qid}"] = "" if v is None or up(v) == "CANT_TELL" else float(v)
            else:
                for o in q["options"]:
                    d[f"b__{qid}__{o['code']}"] = "" if v is None else int(up(v) == up(o["code"]))
        elif q["type"] == "multi":
            s = {up(x) for x in (v or [])}
            for o in q["options"]: d[f"b__{qid}__{o['code']}"] = "" if not s else int(up(o["code"]) in s)
        elif q["type"] == "statements":
            for st in q["statements"]:
                nums = [float(x[st["code"]]) for x in (v or []) if isinstance(x, dict) and str(x.get(st["code"])).isdigit()]
                d[f"n__appraisal_{st['code']}"] = float(np.median(nums)) if nums else ""
        elif q["type"] == "nested_ranked":
            lst = [x for x in (v or []) if isinstance(x, dict) and x.get("about")]
            firsts = collections.Counter(up(x["about"][0]) for x in lst)
            top = firsts.most_common(2); first = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None
            for o in q["options"]: d[f"b__reason_first__{o['code']}"] = "" if not lst or first is None else int(first == up(o["code"]))
            memo = collections.Counter(m for x in lst for m in {MEMO.get(up(s)) for s in (x.get("details") or [])} if m)
            labels = sorted({m for m in MEMO.values() if m})
            for m in labels: d[f"b__memo__{m}"] = "" if not lst else int(memo[m] > len(lst) / 2)
    recs.append(d)
cols = list(recs[0].keys())
for d in recs[1:]:
    for k in d:
        if k not in cols: cols.append(k)
keep, info = [], {}
for k in cols:
    if not k.startswith(("b__", "n__")): keep.append(k); continue
    vals = [d.get(k) for d in recs if d.get(k) not in ("", None)]
    if k.startswith("b__"):
        p = float(np.mean(vals)) if vals else 0
        if not (0.05 <= p <= 0.95): continue
        info[k] = {"kind": "binary", "rate": round(p, 3), "n": len(vals)}
    else:
        info[k] = {"kind": "numeric", "mean": round(float(np.mean(vals)), 3), "n": len(vals)}
    keep.append(k)
with open(os.path.join(OUT, "long.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore"); w.writeheader()
    for d in recs: w.writerow({k: d.get(k, "") for k in keep})
json.dump(info, open(os.path.join(OUT, "outcomes.json"), "w"), indent=1)
print(len(recs), "reports;", sum(v["kind"] == "binary" for v in info.values()), "binary and", sum(v["kind"] == "numeric" for v in info.values()), "numeric outcomes")
