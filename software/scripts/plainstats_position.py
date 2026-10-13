#!/usr/bin/env python3
"""Position check (A1, Oct 11: "where in a call the memories happen (or if it's not a specific moment)").
(1) Locator vs the person: the locator's position (midpoint of located turns as % of the call) against the person's OWN answer to "When during
    your conversation did this moment occur? (0-100)". Human-grounded check of the locator.
(2) Self-reported timing, no locator and no judge: closeness = -|timing difference|; person-specific "beats session-mates" score as plainstats.py,
    rungs: partner's own timing answer, the person's earlier answers (median), models' timing answers by condition (five models with controls).
(3) Located position and span: person vs partner, vs own earlier, vs the five models' located no-history predictions (located_preds).
    Span = share of the call's turns covered; 'general' = the locator found no specific moment.
-> analysis/plainstats/position.json"""
import collections, glob, json, os
import numpy as np
from scipy.stats import spearmanr
src = open("plainstats.py").read().split("Hf = {")[0]; exec(src)   # H, rng, helpers
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); LOC = os.path.join(ROOT, "analysis", "located")
def cboot(v, w, B=2000):
    v = np.array(v, float); w = np.array(w); ok = ~np.isnan(v); v, w = v[ok], w[ok]
    if len(v) < 20: return None
    ppl = np.unique(w); ix = {p: np.where(w == p)[0] for p in ppl}; s = np.array([v[ix[p]].sum() for p in ppl]); c = np.array([len(ix[p]) for p in ppl])
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(ppl), len(ppl)) for _ in range(B))]
    return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(len(v)), int(len(ppl))]
L = {}
for f in glob.glob(os.path.join(LOC, "PID_*.json")):
    j = json.load(open(f)); L[j["perspectiveID"]] = j
def span(j): return len(j["turns"]) / j["n_turns"] if j and j.get("kind") == "in_conversation" and j.get("turns") and j.get("n_turns") else None
out = {}
# (1) locator vs self-report
pairs = [(j["located_pos"], j["human_timing"], j.get("confidence")) for p, j in L.items() if p in H and j.get("kind") == "in_conversation" and j.get("located_pos") is not None and j.get("human_timing") is not None]
r = spearmanr([a for a, b, c in pairs], [b for a, b, c in pairs])
out["locator_vs_self_report"] = dict(spearman=round(float(r.correlation), 3), n=len(pairs),
    by_confidence={c: (round(float(spearmanr([a for a, b, cc in pairs if cc == c], [b for a, b, cc in pairs if cc == c]).correlation), 3), sum(cc == c for *_, cc in pairs)) for c in ("high", "medium", "low")},
    median_abs_diff=round(float(np.median([abs(a - b) for a, b, c in pairs])), 1))
out["locator_kinds"] = dict(collections.Counter(L[p].get("kind") for p in H if p in L))
# shared machinery
by_person = collections.defaultdict(list); by_batch = collections.defaultdict(list)
for pid, h in H.items(): by_person[h["selfPID"]].append(pid); by_batch[h["batchID"]].append(pid)
earlier = {pid: [o for o in by_person[H[pid]["selfPID"]] if H[o]["round"] < H[pid]["round"]] for pid in H}
def mates(pid, excl=()): return [o for o in by_batch[H[pid]["batchID"]] if H[o]["dyadID"] != H[pid]["dyadID"] and H[o]["selfPID"] != H[pid]["selfPID"] and o not in excl]
def beats(val, pid, target, excl=()):
    t = target(pid)
    if val is None or t is None: return None
    s = -abs(val - t); c = [target(m) for m in mates(pid, excl)]; c = [-abs(val - x) for x in c if x is not None]
    return float(np.mean([1.0 if s > x else .5 if s == x else 0.0 for x in c])) - .5 if c else None
partner = lambda pid: f"{H[pid]['otherPID']}+{H[pid]['selfPID']}"
# (2) self-reported timing
import prompts as P
TIME = {r["perspectiveID"]: float(L[r["perspectiveID"]]["human_timing"]) for r in P.rows() if r["perspectiveID"] in L and L[r["perspectiveID"]].get("human_timing") is not None}
tgt = lambda pid: TIME.get(pid)
M5 = {"deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"}
MT = collections.defaultdict(lambda: collections.defaultdict(dict))
RUNS = [("main_text_cheap", ("none", "history")), ("cards_text_cheap", ("card_thin",)), ("cards_narr_cheap", ("card_narr",))] + [(f"r3c_{m}", ("history_prior", "history_wrong")) for m in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")]
for run, infos in RUNS:
    for l in open(os.path.join(ROOT, "runs", run, "preds.jsonl")):
        j = json.loads(l); m = j["model"].split("/")[1]
        if j.get("ok") and j["info"] in infos and m in M5 and j.get("modality", "text") == "text" and j["perspectiveID"] in H:
            try: MT[j["info"]][j["perspectiveID"]][m] = float((j.get("pred") or {}).get("timing"))
            except Exception: pass
common = [p for p in H if earlier[p] and p in TIME]
T2 = {}
def rung(name, get):
    v, w = [], []
    for pid in common:
        for val, ex in get(pid):
            b = beats(val, pid, tgt, ex)
            if b is not None: v.append(b); w.append(H[pid]["selfPID"])
    T2[name] = cboot(v, w)
rung("partner's own timing answer", lambda pid: [(TIME.get(partner(pid)), (partner(pid),))] if partner(pid) in TIME else [])
rung("person's earlier answers (median)", lambda pid: [(float(np.median([TIME[o] for o in earlier[pid] if o in TIME])), ())] if any(o in TIME for o in earlier[pid]) else [])
for c in ("none", "history", "history_prior", "history_wrong", "card_thin", "card_narr"):
    rung(f"model: {c}", lambda pid, c=c: [(v, ()) for v in MT[c].get(pid, {}).values()])
for a, b in (("history", "history_wrong"), ("history", "none")):
    d, w = [], []
    for pid in common:
        A, B = MT[a].get(pid, {}), MT[b].get(pid, {})
        for m in set(A) & set(B):
            x, y = beats(A[m], pid, tgt), beats(B[m], pid, tgt)
            if x is not None and y is not None: d.append(x - y); w.append(H[pid]["selfPID"])
    T2[f"{a} minus {b} (paired)"] = cboot(d, w)
out["self_reported_timing"] = T2
# (3) located position and span
LP = collections.defaultdict(dict); LPH = collections.defaultdict(dict); LPW = collections.defaultdict(dict)   # Oct 11: + history-condition placements (located_preds_history)
for m in M5:
    for f in glob.glob(os.path.join(ROOT, "analysis", "located_preds", m, "PID_*.json")):
        j = json.load(open(f)); LP[m][os.path.basename(f)[:-5]] = j
    for f in glob.glob(os.path.join(ROOT, "analysis", "located_preds_history", m, "PID_*.json")):
        j = json.load(open(f)); LPH[m][os.path.basename(f)[:-5]] = j
    for f in glob.glob(os.path.join(ROOT, "analysis", "located_preds_history_wrong", m, "PID_*.json")):
        j = json.load(open(f)); LPW[m][os.path.basename(f)[:-5]] = j
pos = lambda j: 100 * (float(np.mean(j["turns"])) + .5) / j["n_turns"] if j and j.get("kind") == "in_conversation" and j.get("turns") and j.get("n_turns") else None   # turn-index midpoint, same rule for people and models
T3 = {}
for meas, fn in (("located position", pos), ("span (share of call)", span)):
    tg = lambda pid, fn=fn: fn(L.get(pid))
    R = {}
    def rr(name, get):
        v, w = [], []
        for pid in [p for p in H if earlier[p]]:
            for val, ex in get(pid):
                b = beats(val, pid, tg, ex)
                if b is not None: v.append(b); w.append(H[pid]["selfPID"])
        R[name] = cboot(v, w)
    rr("partner's memory", lambda pid: [(fn(L.get(partner(pid))), (partner(pid),))])
    rr("person's earlier memories (mean)", lambda pid: [(float(np.mean(x)), ())] if (x := [fn(L.get(o)) for o in earlier[pid] if fn(L.get(o)) is not None]) else [])
    rr("model: none (located)", lambda pid: [(fn(LP[m].get(pid)), ()) for m in M5 if pid in LP[m]])
    rr("model: history (located)", lambda pid: [(fn(LPH[m].get(pid)), ()) for m in M5 if pid in LPH[m]])
    d, w = [], []
    for pid in [p for p in H if earlier[p]]:
        for m in M5:
            if pid in LP[m] and pid in LPH[m]:
                a, b = beats(fn(LPH[m][pid]), pid, tg), beats(fn(LP[m][pid]), pid, tg)
                if a is not None and b is not None: d.append(a - b); w.append(H[pid]["selfPID"])
    R["history minus none (paired, located)"] = cboot(d, w)
    rr("model: stranger's history (located)", lambda pid: [(fn(LPW[m].get(pid)), ()) for m in M5 if pid in LPW[m]])
    d, w = [], []
    for pid in [p for p in H if earlier[p]]:
        for m in M5:
            if pid in LPW[m] and pid in LPH[m]:
                a, b = beats(fn(LPH[m][pid]), pid, tg), beats(fn(LPW[m][pid]), pid, tg)
                if a is not None and b is not None: d.append(a - b); w.append(H[pid]["selfPID"])
    R["history minus stranger's history (paired, located)"] = cboot(d, w)
    T3[meas] = R
gen = lambda j: None if not j or j.get("kind") not in ("in_conversation", "general") else float(j["kind"] == "general")
out["located"] = T3
out["general_share"] = dict(people=round(float(np.mean([gen(L.get(p)) for p in H if gen(L.get(p)) is not None])), 3),
                            models_none=round(float(np.mean([gen(LP[m][p]) for m in M5 for p in LP[m] if gen(LP[m][p]) is not None])), 3))
json.dump(out, open(os.path.join(D, "position.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("locator_vs_self_report", "locator_kinds", "general_share")}, indent=0))
for sec in ("self_reported_timing",): print(sec); [print(f"   {k:36s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] n={v[3]}") for k, v in out[sec].items() if v]
for meas, R in T3.items(): print(meas); [print(f"   {k:36s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] n={v[3]}") for k, v in R.items() if v]
