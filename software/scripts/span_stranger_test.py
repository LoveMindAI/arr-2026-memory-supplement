#!/usr/bin/env python3
"""The literal stranger control for span (A1, Oct 11: "for $28, let's just do it"). Placements of the five models' stranger-history guesses
(analysis/located_preds_history_wrong, Sonnet locator, same prompt). Donor = the stranger whose memories the prompt showed, rebuilt with the
same seeded rule (prompts.donor, need_hist = number of the person's other memories; first n of the donor's memories by round).
(1) beats-session-mates on span: stranger history vs history vs none (paired). (2) regression: span of each guess on the PERSON's habit and on
the DONOR's habit (mean span of the memories shown), with the blind guess for the same call held fixed. -> analysis/plainstats/span_stranger_test.json"""
import collections, glob, json, os
import numpy as np
import prompts as P
ROOT = ".."; rng = np.random.default_rng(11); A = os.path.join(ROOT, "analysis")
rows = {r["perspectiveID"]: r for r in P.rows()}
def span(j): return len(j["turns"]) / j["n_turns"] if j and j.get("kind") == "in_conversation" and j.get("turns") and j.get("n_turns") else None
L = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(A, "located", "PID_*.json"))}
H = json.load(open(os.path.join(A, "plainstats", "texts.json")))["human"]
by_person = collections.defaultdict(list)
for p, h in H.items(): by_person[h["selfPID"]].append(p)
M5 = ["deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"]
LP = {c: {m: {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(A, sub, m, "PID_*.json"))} for m in M5}
      for c, sub in (("none", "located_preds"), ("history", "located_preds_history"), ("history_wrong", "located_preds_history_wrong"))}
def habit(pids):
    v = [span(L.get(o)) for o in pids]; v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None
recs = []
for pid, h in H.items():
    r = rows[pid]; n = len(P.history(r))
    if n == 0: continue
    don = sorted(P.donor(r, need_hist=n), key=lambda x: int(x["roundIdx"]))[:n]
    ph = habit([o for o in by_person[h["selfPID"]] if o != pid]); dh = habit([x["perspectiveID"] for x in don])
    if ph is None or dh is None: continue
    for m in M5:
        b, y, w = span(LP["none"][m].get(pid)), span(LP["history"][m].get(pid)), span(LP["history_wrong"][m].get(pid))
        if b is not None: recs.append((h["selfPID"], ph, dh, b, y, w))
X = np.array([r[1:] for r in recs], float); people = np.array([r[0] for r in recs])
z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
def fit(idx, ycol):
    ok = idx[~np.isnan(X[idx, ycol])]
    D = np.c_[np.ones(len(ok)), z(X[ok, 0]), z(X[ok, 1]), z(X[ok, 2])]
    return np.linalg.lstsq(D, z(X[ok, ycol]), rcond=None)[0][1:3]   # slopes on person's habit, donor's habit
ppl = np.unique(people); ix = {p: np.where(people == p)[0] for p in ppl}; allidx = np.arange(len(X))
out = {"n_rows": len(X), "n_people": len(ppl)}
for name, col in (("history guess", 3), ("stranger-history guess", 4)):
    est = fit(allidx, col); bs = np.array([fit(np.concatenate([ix[p] for p in rng.choice(ppl, len(ppl))]), col) for _ in range(2000)])
    out[name] = {"on person's habit": [round(float(est[0]), 3)] + [round(float(x), 3) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                 "on stranger's habit": [round(float(est[1]), 3)] + [round(float(x), 3) for x in np.percentile(bs[:, 1], [2.5, 97.5])]}
json.dump(out, open(os.path.join(A, "plainstats", "span_stranger_test.json"), "w"), indent=1); print(json.dumps(out, indent=1))
