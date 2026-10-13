#!/usr/bin/env python3
"""Correspondence between people's memories and model predictions, codebook v3.5.1 (codebook/SCORING_PARADIGM_v351.md, A1 OK Oct 8).
Coder-matched: a person's memory and a model's prediction are compared using the SAME coder's codes (Haiku with Haiku, Luna with Luna).
Per question, never pooled. Scores per pair: nominal 1/0; ordinal 1-|a-b|/range; multi-select Jaccard; appraisal 1-|a-b|/4 (Can't tell = missing);
reason: first choice same (1/0) and Jaccard on rebuilt MeMo labels.
Anchors: CHANCE = the person's memory vs every other person's memory in the same batch (other calls); CEILING = vs the partner's memory of the same call.
Levels (A2): L1 population (answer distributions), L2 conversation (no-history prediction vs chance and ceiling), L3 person (right vs wrong-person info).
Bootstraps resample people (the rememberer). -> analysis/correspondence_v351.json"""
import collections, glob, importlib, json, os, sys
import numpy as np
import prompts as P
ROOT = ".."; sys.path.insert(0, os.path.join(ROOT, "codebook"))
import scope_merge as SM
SPEC = SM.spec(importlib.import_module("spec_v351_FROZEN").SPEC)   # scope in three classes (scope_merge.py)
Q = [q for p in SPEC["parts"] if p.get("models") is not False for q in p["questions"]]
CODERS = ["claude-haiku-5.5", "gpt-6-luna"]
MEMO = {s["code"]: s.get("memo") for q in Q if q["type"] == "nested_ranked" for o in q["options"] for s in o.get("sub", [])}
up = lambda x: str(x).strip().upper() if x is not None else None
LAB = {(q["id"], up(o["label"])): o["code"] for q in Q for o in q.get("options", []) + q.get("scale", [])}
CODES = {q["id"]: {up(o["code"]) for o in q.get("options", [])} for q in Q}
D = os.path.join(ROOT, "analysis", "code_v351")
rng = np.random.default_rng(2026)


def fix(qid, v):
    if isinstance(v, str) and (qid, up(v)) in LAB: return LAB[(qid, up(v))]
    return v


def norm(codes):
    """-> dict of comparable values per question; None = missing / can't tell."""
    out = {}
    for q in Q:
        v = codes.get(q["id"]); t = q["type"]; qid = q["id"]
        if t == "single":
            v = v[0] if isinstance(v, list) and len(v) == 1 else v; v = up(fix(qid, v))
            if qid == "extent": v = SM.value(v)
            if qid in ("details", "intensity"):
                try: out[qid] = int(v)
                except Exception: out[qid] = None
            else: out[qid] = v if v in CODES[qid] else None
        elif t == "multi":
            v = [v] if isinstance(v, str) else (v or [])
            s = {up(fix(qid, x)) for x in v} & CODES[qid]
            if os.environ.get("COMPLY_MAX3") == "1" and qid in ("topic", "remembered", "feelings") and len(s) > 3: s = set()   # sensitivity (Oct 12): over the codebook's limit of three -> missing
            out[qid] = frozenset(s) if s else None
        elif t == "statements":
            for s in q["statements"]:
                x = fix(qid, (v or {}).get(s["code"]) if isinstance(v, dict) else None)
                try: out[f"appraisal.{s['code']}"] = int(x)
                except Exception: out[f"appraisal.{s['code']}"] = None
        elif t == "nested_ranked":
            if isinstance(v, dict) and v.get("about"):
                ab = [up(x) for x in v["about"]]; dd = {up(x) for x in (v.get("details") or [])}
                out["reason.first"] = ab[0]
                mm = {MEMO.get(x) or "other" for x in dd} | ({"none"} if "NO_REASON" in ab else set()) | ({"other"} if "OTHER" in ab and not dd else set())
                out["reason.memo"] = frozenset(mm) if mm else None
            else: out["reason.first"] = out["reason.memo"] = None
    return out


ORD = {"details": 3, "intensity": 4}
def score(k, a, b):
    if a is None or b is None: return None
    if k in ORD: return 1 - abs(a - b) / ORD[k]
    if k.startswith("appraisal."): return 1 - abs(a - b) / 4
    if isinstance(a, frozenset): return len(a & b) / len(a | b) if (a | b) else 1.0
    return float(a == b)


def load(setdir, coder):
    out = {}
    for f in glob.glob(os.path.join(D, setdir, f"*__{coder}.json")):
        j = json.load(open(f))
        if j.get("ok") and isinstance(j.get("codes"), dict): out[j["key"]] = norm(j["codes"])
    return out


def cboot(vals, people, B=2000):
    g = collections.defaultdict(list)
    for v, p in zip(vals, people): g[p].append(v)
    G = list(g.values()); s = np.array([sum(x) for x in G]); n = np.array([len(x) for x in G])
    bs = [s[k].sum() / n[k].sum() for k in (rng.integers(0, len(G), len(G)) for _ in range(B))]
    return [round(float(np.mean(vals)), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(G)]


rows = {r["perspectiveID"]: r for r in P.rows()}
KEYS = sorted({k for q in Q for k in ([f"appraisal.{s['code']}" for s in q["statements"]] if q["type"] == "statements" else ["reason.first", "reason.memo"] if q["type"] == "nested_ranked" else [q["id"]])})
COND = {"none": "main_text_cheap__none", "history": "main_text_cheap__history", "card_thin": "cards_text_cheap__card_thin", "card_narr": "cards_narr_cheap__card_narr"}
R3C = {"deepseek-v4.1-flash": "deepseek", "gemini-3.8-flash": "gemini", "gemma-4-31b-it": "gemma", "mimo-v2.6-flash": "mimo", "qwen3.8-omni-flash": "qwen3.8"}
res = {"keys": KEYS, "coders": {}}
for coder in CODERS:
    H = load("human", coder); C = {}
    for cond, sd in COND.items(): C[cond] = load(sd, coder)
    for cond in ("history_wrong", "history_prior", "card_thin_wrong", "card_narr_wrong"):
        C[cond] = {}
        for m, short in R3C.items(): C[cond].update(load(f"r3c_{short}__{cond}", coder))
    by_batch = collections.defaultdict(list)
    for pid in H: by_batch[rows[pid]["batchID"]].append(pid)
    R = res["coders"][coder] = {"L2": {}, "L3": {}, "L1": {}}
    # anchors
    chance, ceil = collections.defaultdict(dict), collections.defaultdict(dict)
    for pid, h in H.items():
        r = rows[pid]; partner = f"{r['otherPID']}+{r['selfPID']}" if f"{r['otherPID']}+{r['selfPID']}" in H else None
        others = [o for o in by_batch[r["batchID"]] if rows[o]["dyadID"] != r["dyadID"]]
        for k in KEYS:
            sc = [score(k, h[k], H[o][k]) for o in others]; sc = [x for x in sc if x is not None]
            if sc: chance[k][pid] = float(np.mean(sc))
            if partner: s = score(k, h[k], H[partner][k]); ceil[k][pid] = s if s is not None else None
    # L2: no-history predictions
    for k in KEYS:
        mv, mp = [], []
        for key, pr in C["none"].items():
            pid = key.split("|")[0]
            if pid in H:
                s = score(k, H[pid][k], pr[k])
                if s is not None: mv.append(s); mp.append(pid.split("+")[0])
        cv = [(v, p.split("+")[0]) for p, v in chance[k].items()]; pv = [(v, p.split("+")[0]) for p, v in ceil[k].items() if v is not None]
        R["L2"][k] = {"model_none": cboot(mv, mp), "chance": cboot(*zip(*cv)) if cv else None, "partner_ceiling": cboot(*zip(*pv)) if pv else None}
    # L3: paired within report and model: right minus comparison
    for name, (a, b) in {"history_minus_wrong": ("history", "history_wrong"), "prior_minus_none": ("history_prior", "none"), "history_minus_none": ("history", "none"),
                         "cardthin_minus_wrong": ("card_thin", "card_thin_wrong"), "cardnarr_minus_wrong": ("card_narr", "card_narr_wrong")}.items():
        R["L3"][name] = {}
        bmap = {(k.split("|")[0], k.split("|")[1]): v for k, v in C[b].items()}
        for k in KEYS:
            dv, dp = [], []
            for key, pa in C[a].items():
                pid, model = key.split("|")[0], key.split("|")[1]; pb = bmap.get((pid, model))
                if pid in H and pb is not None:
                    sa, sb = score(k, H[pid][k], pa[k]), score(k, H[pid][k], pb[k])
                    if sa is not None and sb is not None: dv.append(sa - sb); dp.append(pid.split("+")[0])
            R["L3"][name][k] = cboot(dv, dp) if dv else None
    # L1: answer distributions, humans vs no-history predictions (single and multi questions)
    for q in Q:
        if q["type"] not in ("single", "multi"): continue
        def share(vals):
            c = collections.Counter(); n = 0
            for v in vals:
                if v is None: continue
                n += 1
                for x in (v if isinstance(v, frozenset) else [v]): c[str(x)] += 1
            return {o: round(c[o] / n, 3) for o in sorted(c)} if n else {}
        hs, ps = share(h[q["id"]] for h in H.values()), share(p[q["id"]] for p in C["none"].values())
        R["L1"][q["id"]] = {"human": hs, "models_none": ps, "biggest_gaps": sorted(((round(ps.get(o, 0) - hs.get(o, 0), 3), o) for o in set(hs) | set(ps)), key=lambda x: -abs(x[0]))[:3]}
    print(coder, "humans", len(H), {c: len(v) for c, v in C.items()}, flush=True)
json.dump(res, open(os.path.join(ROOT, "analysis", "correspondence_v351.json"), "w"), indent=1)
print("written")
