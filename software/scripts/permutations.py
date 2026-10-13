#!/usr/bin/env python3
"""Follow-up analyses (A1, Oct 7: "please go about doing all of this"). Writes analysis/permutations.json.
A) card x history: does the person's history wash out a card's trait coupling? (10 cheap models; card_thin/_history, card_narr/_history)
B) dose-response: across models (and model x condition in round 2), does stronger coupling go with a bigger loss of fit (same focus vs none)?
C) human structure in predictions: social relations split (rememberer vs partner) of the models' own ratings of the moment, by condition,
   next to the same split for people (5 models; none / history / card_narr; all memories).
D) whose memory is this, for predicted text: author identification from a person's other PREDICTED memories (embeddings), by condition.
E) self-other mismatch: do people whose self-report differs from how they came across (4-model impressions) lose more fit from a
   self-report card than from an impression card?"""
import collections, json, os, sys, warnings
import numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import figs_models as FM, trait_steering as T, prompts as P
ROOT = FM.ROOT
CHEAP = ["deepseek-v4.1-flash", "glm-5.3-flashx", "kimi-k2-0905", "mimo-v2.5-pro", "mimo-v2.6-flash", "qwen3.8-omni-flash", "gemini-3.8-flash",
         "mimo-v2.5", "gemma-4-31b-it", "muse-glimmer-30b"]
M5 = ["deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"]
RUNS = ["main_text_cheap", "cards_text_cheap", "cards_narr_cheap", "r2_facts_impthin", "r2_weave_factnarr", "r2_weave_factnarr_impnarr"]
OUTS = ["pleasure", "arousal", "dominance"]
res = {}


def load_preds():
    D = collections.defaultdict(dict)
    for r in RUNS:
        for l in open(os.path.join(ROOT, "runs", r, "preds.jsonl")):
            j = json.loads(l)
            if j.get("ok") and j["modality"] == "text":
                D[(j["model"].split("/")[1], j["info"])][j["perspectiveID"]] = j["pred"]
    return D


D = load_preds()
F = {k: {pid: T.feats(p.get("moment", ""), p) for pid, p in v.items()} for k, v in D.items()}
S = {}
for r in RUNS:
    s, _ = FM.load(r); S.update(s)
rng = np.random.default_rng(0)

# A) card x history
A = {}
for card in ("card_thin", "card_narr"):
    for info in (card, card + "_history"):
        v = collections.defaultdict(list)
        for m in CHEAP:
            t = T.r2_table(F.get((m, info), {}))
            for o in OUTS:
                if o in t: v[o].append(t[o]["cv_r2"])
        A[info] = {o: float(np.median(x)) for o, x in v.items()}
print("A) coupling (median over 10 models):", {k: {o: round(x, 3) for o, x in v.items()} for k, v in A.items()})
res["card_x_history"] = A

# B) dose-response
def coupling(m, info, trmap=None):
    bk = T.TR
    if trmap is not None: T.TR = trmap
    t = T.r2_table(F.get((m, info), {})); T.TR = bk
    v = [t[o]["cv_r2"] for o in OUTS if o in t]
    return float(np.mean(v)) if v else None
imp = json.load(open(os.path.join(ROOT, "cards", "inputs", "impression_facets.json")))
sys.path.insert(0, os.path.join(ROOT, "cards")); import score_fidelity as SF
TR_IMP = {p: np.array([np.mean([v[i] for i in SF.IDS if i[0] == d]) for d in "HEXACO"]) for p, v in imp.items()}
pts = []
for m in CHEAP:
    c = coupling(m, "card_narr"); f = FM.paired(S, m, "text", "none", "card_narr", "same_focus", rng)
    if c is not None and f: pts.append((m, c, f["mean"]))
rho = spearmanr([p[1] for p in pts], [p[2] for p in pts])
res["dose_models_narr"] = dict(points=pts, rho=float(rho.correlation), p=float(rho.pvalue))
print(f"B1) across 10 models, narrative card: coupling vs same-focus change rho {rho.correlation:+.2f} (p {rho.pvalue:.2f})")
pts2 = []
for m in M5:
    for info in ("card_thin", "card_narr", "card_impthin", "card_impnarr", "card_facts", "card_factnarr", "card_weave"):
        c = coupling(m, info, TR_IMP if "imp" in info else None); f = FM.paired(S, m, "text", "none", info, "same_focus", rng)
        if c is not None and f: pts2.append((m, info, c, f["mean"]))
rho2 = spearmanr([p[2] for p in pts2], [p[3] for p in pts2])
res["dose_model_x_condition"] = dict(points=pts2, rho=float(rho2.correlation), p=float(rho2.pvalue))
print(f"B2) 5 models x 7 descriptions: coupling vs same-focus change rho {rho2.correlation:+.2f} (p {rho2.pvalue:.3f}, n {len(pts2)})")

# C) SRM of model ratings by condition
import pandas as pd, statsmodels.formula.api as smf
rows = {r["perspectiveID"]: r for r in P.rows()}
def srm(vals):
    df = pd.DataFrame([dict(y=v, pid=rows[k]["selfPID"], partner=rows[k]["otherPID"], batch=rows[k]["batchID"]) for k, v in vals.items() if v is not None])
    if len(df) < 200 or df.y.std() == 0: return None
    try:
        mm = smf.mixedlm("y ~ 1", df, groups="batch", vc_formula={"pid": "0 + C(pid)", "partner": "0 + C(partner)"}).fit(reml=True)
        vc = dict(zip(mm.model.exog_vc.names, mm.vcomp)); tot = vc["pid"] + vc["partner"] + mm.scale
        return dict(rememberer=float(vc["pid"] / tot), partner=float(vc["partner"] / tot))
    except Exception:
        return None
SRMRES = {}  # not 'C': patsy's C() in the formula would see a global named C
H = {"pleasure": "conv_SP_Mem_p", "arousal": "conv_SP_Mem_a", "dominance": "conv_SP_Mem_d"}
for o, col in H.items():
    SRMRES[f"human|{o}"] = srm({k: (float(r[col]) if r[col].strip() not in ("", "nan") else None) for k, r in rows.items()})
for info in ("none", "history", "card_narr"):
    for o in OUTS:
        v = [srm({k: f.get(o) for k, f in F.get((m, info), {}).items()}) for m in M5]
        v = [x for x in v if x]
        if v: SRMRES[f"{info}|{o}"] = dict(rememberer=float(np.median([x["rememberer"] for x in v])), partner=float(np.median([x["partner"] for x in v])), models=len(v))
res["srm_ratings"] = SRMRES
for k, v in SRMRES.items():
    if v: print(f"C) {k:22s} rememberer {v['rememberer']:.2f} partner {v['partner']:.2f}")

# E) self-other mismatch
mis = {}
for p in imp:
    if p in SF.TRUE:
        a = np.array([imp[p][i] for i in SF.IDS]); t = np.array([SF.TRUE[p][SF.MAP[i]] for i in SF.IDS])
        mis[p] = float(np.mean(np.abs(a - t)))
def person_change(info):
    acc = collections.defaultdict(list)
    for m in M5:
        for k in rows:
            x, y = S.get((m, "text", "none", k)), S.get((m, "text", info, k))
            if x and y:
                u, v = FM.avg2(x, "same_focus"), FM.avg2(y, "same_focus")
                if u is not None and v is not None: acc[rows[k]["selfPID"]].append(v - u)
    return {p: float(np.mean(v)) for p, v in acc.items()}
E = {}
for info in ("card_narr", "card_thin", "card_impnarr", "card_impthin"):
    ch = person_change(info); ps = [p for p in ch if p in mis]
    r_ = spearmanr([mis[p] for p in ps], [ch[p] for p in ps]); E[info] = dict(n=len(ps), rho=float(r_.correlation), p=float(r_.pvalue))
    print(f"E) mismatch vs same-focus change, {info:13s}: rho {r_.correlation:+.3f} (p {r_.pvalue:.3f}, n {len(ps)})")
res["self_other_mismatch"] = E
json.dump(res, open(os.path.join(ROOT, "analysis", "permutations.json"), "w"), indent=1)
