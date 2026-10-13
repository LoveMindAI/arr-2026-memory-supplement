#!/usr/bin/env python3
"""Free stand-in for the stranger-history control on span (Oct 11). A generic 'human-shaped' shift from seeing any history cannot make a
guess's span follow THIS person's habit. Test: across (report, model), regress the span of the model's guess on the person's own mean span
over their OTHER memories (exactly the memories the history prompt shows), controlling for the blind guess's span on the same call (the call's
own pull). History guesses should load on the person's habit; blind guesses should not. Standardized slopes, person-clustered bootstrap.
-> analysis/plainstats/span_person_test.json"""
import collections, glob, json, os
import numpy as np
ROOT = ".."; rng = np.random.default_rng(7)
H = json.load(open(os.path.join(ROOT, "analysis", "plainstats", "texts.json")))["human"]
def span(j): return len(j["turns"]) / j["n_turns"] if j and j.get("kind") == "in_conversation" and j.get("turns") and j.get("n_turns") else None
L = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "analysis", "located", "PID_*.json"))}
by_person = collections.defaultdict(list)
for p, h in H.items(): by_person[h["selfPID"]].append(p)
M5 = ["deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"]
LP = {c: {m: {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "analysis", sub, m, "PID_*.json"))} for m in M5}
      for c, sub in (("none", "located_preds"), ("history", "located_preds_history"))}
rows = []
for pid, h in H.items():
    own = [span(L.get(o)) for o in by_person[h["selfPID"]] if o != pid]; own = [x for x in own if x is not None]
    if not own: continue
    habit = float(np.mean(own))
    for m in M5:
        b, y = span(LP["none"][m].get(pid)), span(LP["history"][m].get(pid))
        if b is not None and y is not None: rows.append((h["selfPID"], habit, b, y, span(L.get(pid))))
people = np.array([r[0] for r in rows]); X = np.array([r[1:] for r in rows], float)   # habit, blind span, history span, person's actual span
z = lambda v: (v - v.mean()) / v.std()
def slopes(idx):
    hab, bl, hi = z(X[idx, 0]), z(X[idx, 1]), z(X[idx, 2])
    A = np.c_[np.ones(len(idx)), hab, bl]; bh = np.linalg.lstsq(A, hi, rcond=None)[0][1]      # history guess ~ habit + blind (call)
    bb = np.linalg.lstsq(np.c_[np.ones(len(idx)), hab], bl, rcond=None)[0][1]                 # blind guess ~ habit (placebo)
    return bh, bb, bh - bb
ppl = np.unique(people); ix = {p: np.where(people == p)[0] for p in ppl}
est = slopes(np.arange(len(X))); bs = []
for _ in range(2000):
    bs.append(slopes(np.concatenate([ix[p] for p in rng.choice(ppl, len(ppl))])))
bs = np.array(bs); ci = lambda k: [round(float(np.percentile(bs[:, k], 2.5)), 3), round(float(np.percentile(bs[:, k], 97.5)), 3)]
out = dict(n_pairs=len(X), n_people=len(ppl),
           history_guess_on_habit_controlling_call=[round(float(est[0]), 3)] + ci(0),
           blind_guess_on_habit=[round(float(est[1]), 3)] + ci(1),
           difference=[round(float(est[2]), 3)] + ci(2),
           person_actual_span_on_habit=round(float(np.corrcoef(X[~np.isnan(X[:, 3]), 0], X[~np.isnan(X[:, 3]), 3])[0, 1]), 3))
json.dump(out, open(os.path.join(ROOT, "analysis", "plainstats", "span_person_test.json"), "w"), indent=1); print(json.dumps(out, indent=1))
