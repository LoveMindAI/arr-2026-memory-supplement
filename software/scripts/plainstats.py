#!/usr/bin/env python3
"""Plain-statistics check, step 3: the paper's ladder (ladder_v351.py) with NO LLM in the measurement.
FORM: ten counted features of a report (lengths, we / he-she / I pronoun shares, reported speech, quotation, concrete names and
numbers, loose typing, position words), z-scored on people; closeness = minus the mean absolute z difference (and per feature).
SUBJECT: content-word overlap (Jaccard on lowercased, stop-worded, lightly stemmed words of the moment), and cosine similarity of the
moment's embedding for every embedder found in analysis/plainstats/emb_*.npz.
Ladder rungs as ladder_v351: partner's memory, the person's own earlier reports (mean), models by condition (five models with controls,
plus all models for none/history). Common set: reports with at least one earlier report by the same person. EXCESS = closeness to this
person's report minus mean closeness to batch-mates' reports from other calls (never the person's own); BEATS = share of those batch-mates
the prediction is closer to this person than to, minus .5 (scale-free: word overlap and cosine levels shift with text length); person-clustered bootstrap.
-> analysis/plainstats/plainstats.json"""
import collections, glob, json, os, re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = os.path.join(ROOT, "analysis", "plainstats")
T = json.load(open(os.path.join(D, "texts.json"))); H = T["human"]; rng = np.random.default_rng(2026)
W = lambda t: re.findall(r"[A-Za-z']+|\d+", t)
SETS = dict(WE={"we", "us", "our", "ours", "ourselves", "we're", "we'd", "we've", "let's", "we'll"},
            THIRD={"he", "she", "him", "her", "his", "hers", "himself", "herself", "he's", "she's", "he'd", "she'd", "he'll", "she'll"},
            I={"i", "me", "my", "mine", "myself", "i'm", "i'd", "i've", "i'll"},
            SPEECH={"said", "says", "say", "saying", "told", "tell", "tells", "telling", "asked", "ask", "asks", "asking", "mentioned", "mention",
                    "mentions", "mentioning", "explained", "replied", "admitted", "joked", "commented", "answered"},
            POS={"start", "started", "beginning", "begin", "end", "ended", "ending", "middle", "minute", "minutes", "halfway", "towards", "toward", "final", "last", "first", "later", "early"})
FEATS = ["len_moment", "len_reason", "we_share", "third_share", "i_share", "speech_rate", "quote", "concrete_rate", "loose_typing", "position_word"]
def feats(m, r):
    w = W(m); lw = [x.lower() for x in w]; n = max(1, len(w))
    we, th, i = (sum(x in SETS[k] for x in lw) for k in ("WE", "THIRD", "I")); pr = we + th + i
    caps = sum(1 for k, x in enumerate(w) if x[:1].isupper() and x != "I" and not (k == 0 or re.search(r"[.!?]\s*$", m[:m.find(x)] if x in m else "")))
    nums = sum(1 for x in w if x.isdigit())
    loose = float(bool(m) and (m[0].islower() or not re.search(r"[.!?\"')]\s*$", m) or re.search(r"\bi\b", m) is not None))
    return [np.log1p(len(w)), np.log1p(len(W(r))), we / pr if pr else np.nan, th / pr if pr else np.nan, i / pr if pr else np.nan,
            sum(x in SETS["SPEECH"] for x in lw) / n, float(bool(re.search(r"[\"“”‘]|'[A-Z]", m))), (caps + nums) / n, loose, float(any(x in SETS["POS"] for x in lw))]
Hf = {pid: np.array(feats(h["moment"], h["reason"]), float) for pid, h in H.items()}
M = np.array(list(Hf.values())); mu = np.nanmean(M, 0); sd = np.nanstd(M, 0); sd[sd == 0] = 1
Z = lambda v: (np.where(np.isnan(v), mu, v) - mu) / sd
Hz = {pid: Z(v) for pid, v in Hf.items()}
STOP = set(ENGLISH_STOP_WORDS) | {"just", "really", "like", "got", "bit", "think", "thought", "thing", "things", "one", "lot", "quite", "said", "told", "asked", "talked", "talking", "remember", "moment", "call", "conversation"}
def cw(t): return frozenset(x[:-1] if len(x) > 4 and x.endswith("s") else x for x in (y.lower() for y in W(t)) if x not in STOP and len(x) > 2 and not x.isdigit())
Hc = {pid: cw(h["moment"]) for pid, h in H.items()}
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else np.nan
EMB = {}
for f in sorted(glob.glob(os.path.join(D, "emb_*.npz"))):
    z = np.load(f, allow_pickle=True); keys = list(z["keys"]); E = z["E"].astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9; E = E.astype(np.float16)
    EMB[os.path.basename(f)[4:-4]] = dict(zip(keys, E))
by_person = collections.defaultdict(list); by_batch = collections.defaultdict(list)
for pid, h in H.items(): by_person[h["selfPID"]].append(pid); by_batch[h["batchID"]].append(pid)
earlier = {pid: [o for o in by_person[H[pid]["selfPID"]] if H[o]["round"] < H[pid]["round"]] for pid in H}
common = [pid for pid in H if earlier[pid]]
def mates(pid, exclude=()): return [o for o in by_batch[H[pid]["batchID"]] if H[o]["dyadID"] != H[pid]["dyadID"] and H[o]["selfPID"] != H[pid]["selfPID"] and o not in exclude]
# predictors: name -> pid -> list of objects {z, cw, emb-key}
PRED = collections.defaultdict(lambda: collections.defaultdict(list))
for k, p in enumerate(T["preds"]):
    obj = dict(z=Z(np.array(feats(p["moment"], p["reason"]), float)), cw=cw(p["moment"]), ek=f"P{k}", m=p["model"])
    if p["m5"]: PRED[f"model: {p['cond']}"][p["pid"]].append(obj)
    if p["cond"] in ("none", "history"): PRED[f"all models: {p['cond']}"][p["pid"]].append(obj)
def human_obj(pid): return dict(z=Hz[pid], cw=Hc[pid], ek=f"H:{pid}")
def earlier_obj(pid):
    e = earlier[pid]; return dict(z=np.mean([Hz[o] for o in e], 0), cw=frozenset().union(*[Hc[o] for o in e]), ek=[f"H:{o}" for o in e])
def emb(name, ek):
    E = EMB[name]
    if isinstance(ek, list):
        v = np.mean([E[x] for x in ek if x in E], 0) if any(x in E for x in ek) else None
        return None if v is None else v / (np.linalg.norm(v) + 1e-9)
    return E.get(ek)
MEASURES = ["form (all ten)"] + [f"form: {f}" for f in FEATS] + ["subject: content words"] + [f"subject: {n}" for n in EMB]
def score(meas, pred, pid):
    if meas == "form (all ten)": return -float(np.mean(np.abs(pred["z"] - Hz[pid])))
    if meas.startswith("form: "): i = FEATS.index(meas[6:]); return -float(abs(pred["z"][i] - Hz[pid][i]))
    if meas == "subject: content words": return jac(pred["cw"], Hc[pid])
    a, b = emb(meas[9:], pred["ek"]), emb(meas[9:], f"H:{pid}")
    return None if a is None or b is None else float(a @ b)
def cboot(v, w, B=2000):
    v = np.array(v, float); w = np.array(w); ok = ~np.isnan(v); v, w = v[ok], w[ok]
    if len(v) < 20: return None
    ppl = np.unique(w); ix = {p: np.where(w == p)[0] for p in ppl}; s = np.array([v[ix[p]].sum() for p in ppl]); c = np.array([len(ix[p]) for p in ppl])
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(ppl), len(ppl)) for _ in range(B))]
    return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(len(v)), int(len(ppl))]
RUNGS = ["partner's memory", "person's own earlier reports"] + sorted(PRED)
out = {"n_reports": len(common), "features": FEATS, "embedders": list(EMB), "results": {}}
for meas in MEASURES:
    res = {}
    for rung in RUNGS:
        ex, who, bt = [], [], []
        for pid in common:
            person = H[pid]["selfPID"]
            if rung == "partner's memory":
                pp = f"{H[pid]['otherPID']}+{H[pid]['selfPID']}"
                objs, excl = ([human_obj(pp)], (pp,)) if pp in H else ([], ())
            elif rung == "person's own earlier reports": objs, excl = [earlier_obj(pid)], ()
            else: objs, excl = PRED[rung].get(pid, []), ()
            for o in objs:
                s = score(meas, o, pid)
                if s is None or (isinstance(s, float) and np.isnan(s)): continue
                c = [score(meas, o, m) for m in mates(pid, excl)]; c = [x for x in c if x is not None and not np.isnan(x)]
                if c: ex.append(s - float(np.mean(c))); who.append(person); bt.append(float(np.mean([1.0 if s > x else .5 if s == x else 0.0 for x in c])) - .5)
        res[rung] = dict(excess=cboot(ex, who), beats=cboot(bt, who))
    out["results"][meas] = res
    for kind in ("excess", "beats"):
        print(f"{meas[:24]:24s} {kind:6s} " + " | ".join(f"{r.replace('model: ', 'M:').replace('all models: ', 'ALL:')[:15]} {v[kind][0]:+.3f}{'*' if v[kind][1] > 0 or v[kind][2] < 0 else ' '}" for r, v in res.items() if v[kind]))
json.dump(out, open(os.path.join(D, "plainstats.json"), "w"), indent=1)
