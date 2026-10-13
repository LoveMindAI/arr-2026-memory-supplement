#!/usr/bin/env python3
"""Summary of analysis/ladder_tx_v3.json: for each coder and condition, the Scribe rung against the matching Whisper rung, over the
ladder's questions: mean of (Scribe - Whisper) person-specific excess fit, max |diff|, number of questions with |diff| > .01, and the
worst question [question, Whisper, Scribe]. Scope 'all' (key coder|condition) and the held-out half (key coder|condition|held-out half;
the paper reads the norms note there). -> analysis/ladder_tx_v3_summary.json"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
L = json.load(open(os.path.join(ROOT, "analysis", "ladder_tx_v3.json")))
PAIRS = {"none": "no history", "history": "history", "history_prior": "earlier-calls history", "norms": "norms note", "norms_history": "norms + history"}
out = {}
for coder, D in L.items():
    for scope, tag in (("all", ""), ("held_out_half", "|held-out half")):
        for cond, rung in PAIRS.items():
            w, s = f"model: {rung}", f"model (Scribe transcript): {rung}"
            rows = []
            for q, R in D[scope].items():
                a, b = (R.get(w) or {}).get("excess"), (R.get(s) or {}).get("excess")
                if a and b: rows.append((q, a[0], b[0], b[0] - a[0]))
            if not rows: continue
            wq = max(rows, key=lambda r: abs(r[3]))
            out[f"{coder}|{cond}{tag}"] = dict(n_questions=len(rows), mean_diff=round(sum(r[3] for r in rows) / len(rows), 4),
                                               max_abs=round(abs(wq[3]), 3), n_over_01=sum(abs(r[3]) > .01 for r in rows), worst=[wq[0], wq[1], wq[2]])
json.dump(out, open(os.path.join(ROOT, "analysis", "ladder_tx_v3_summary.json"), "w"), indent=1)
for k, v in out.items(): print(f"{k:45s} n={v['n_questions']:2d} mean {v['mean_diff']:+.4f} max {v['max_abs']:.3f} over.01 {v['n_over_01']:2d} worst {v['worst']}")
