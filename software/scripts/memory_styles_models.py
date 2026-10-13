#!/usr/bin/env python3
"""Do models remember in the human memory types? (A1, Oct 10: "can we cluster the LLMs, too? Particularly the blind versions (no history),
the personalized ones (just personality is fine), and the ones who were given the 'humans did it this way' prompt?")
No API calls. PYTHONHASHSEED=0.

Same features as memory_styles.py, but every source (people AND models) is coded by the same two AI coders (Haiku 5.5, Luna),
averaged, so the instrument is identical on both sides. A model "person" = one model writing as one target person in one condition,
over that person's calls (>= 3 reports). Conditions: none (main_text_cheap), card_narr (narrative personality card), norms (the
approved "how people actually answered" note; held-out half only). Five models: DeepSeek v4.1 Flash, Gemini 3.8 Flash, Gemma 4 31B,
MiMo v2.6 Flash, Qwen3.8 Omni Flash.
(a) Human k=2 types are fitted on people (standardized on people); every model profile is assigned to the nearest human centroid.
    Share of each type per model x condition, against people's own split.
(b) Position on the human first two principal axes: mean and spread (SD) of model profiles vs people's.
(c) Carry: ICC of each feature within target person for each condition (does the model give one person a consistent style?) vs the
    within-session shuffle, and against people's own carry.
(d) Type match: is the model's type for a person that person's own human type? Accuracy vs the chance rate implied by the margins.
(e) Models clustered on their own (k = 2..4, silhouette), pooled across models within condition.
-> analysis/memory_styles_models.json"""
import collections, glob, json, os, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "memory_styles.py")).read()
exec(src.split("recs = [")[0])          # rows, feats(), helpers
exec("def icc" + src.split("def icc")[1].split("def shuffle_within_session")[0])
ROOT = os.path.join(HERE, ".."); D = os.path.join(ROOT, "analysis", "code_v351")
CODERS = ["claude-haiku-5.5", "gpt-6-luna"]
M5 = ["deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "mimo-v2.6-flash", "qwen3.8-omni-flash"]
R3 = {"deepseek": "deepseek-v4.1-flash", "gemini": "gemini-3.8-flash", "gemma": "gemma-4-31b-it", "mimo": "mimo-v2.6-flash", "qwen3.8": "qwen3.8-omni-flash"}
# Oct 10 (A1: "run everything so we just really process our data every which way"): every coded condition, both transcripts.
CONDS = {"none": (["main_text_cheap__none"], ["main_text_cheap"]), "history": (["main_text_cheap__history"], ["main_text_cheap"]),
         "card_thin": (["cards_text_cheap__card_thin"], ["cards_text_cheap"]), "card_narr": (["cards_narr_cheap__card_narr"], ["cards_narr_cheap"]),
         "norms": (["norms_note_v2__norms"], ["norms_note_v2"]), "norms_history": (["norms_history_v2__norms_history"], ["norms_history_v2"]),
         "history_prior": ([f"r3c_{t}__history_prior" for t in R3], [f"r3c_{t}" for t in R3]),
         "history_wrong": ([f"r3c_{t}__history_wrong" for t in R3], [f"r3c_{t}" for t in R3]),
         "card_thin_wrong": ([f"r3c_{t}__card_thin_wrong" for t in R3], [f"r3c_{t}" for t in R3]),
         "card_narr_wrong": ([f"r3c_{t}__card_narr_wrong" for t in R3], [f"r3c_{t}" for t in R3]),
         "scribe:none": (["tx_v3_cheap__none"], ["tx_v3_cheap"]), "scribe:history": (["tx_v3_cheap__history"], ["tx_v3_cheap"]),
         "scribe:norms": (["tx_v3_norms_note__norms"], ["tx_v3_norms_note"]), "scribe:norms_history": (["tx_v3_norms_history__norms_history"], ["tx_v3_norms_history"]),
         "scribe:history_prior": (["tx_v3_r3c__history_prior"], ["tx_v3_r3c"])}
SINGLE = {"extent", "shared_by", "new_info", "details", "moment_text", "intensity", "bigger_meaning"}

def norm(c):
    o = {}
    for k, v in c.items():
        if k == "reason":
            vv = [v] if isinstance(v, dict) else (v or [])
            flat = lambda xs: [str(y) for x in (xs or []) for y in (x if isinstance(x, list) else [x])]
            o[k] = [dict(about=flat(r.get("about")), details=flat(r.get("details"))) for r in vv if isinstance(r, dict)]
        elif k == "appraisal": o[k] = v
        elif isinstance(v, list): o[k] = [str(x).lower() for x in v]
        elif isinstance(v, str) and k in SINGLE and not v.isdigit(): o[k] = v.upper()
        else: o[k] = v
    return o

def coded(setdirs):
    by = collections.defaultdict(dict)
    for f in [g for sd in ([setdirs] if isinstance(setdirs, str) else setdirs) for g in glob.glob(os.path.join(D, sd, "*__*.json"))]:
        j = json.load(open(f)); cod = os.path.basename(f).rsplit("__", 1)[1][:-5]
        if cod in CODERS and j.get("ok") and isinstance(j.get("codes"), dict): by[j["key"]][cod] = norm(j["codes"])
    return by

def featvec(codes_by_coder, row_like):
    vs = [feats(SM.codes(c), row_like) for c in codes_by_coder.values()]
    return {n: float(np.nanmean([v.get(n, np.nan) for v in vs])) if any(not np.isnan(v.get(n, np.nan)) for v in vs) else np.nan for n in vs[0]}

# people, same instrument
H = coded("human")
P_ = []
for k, cc in H.items():
    pid = k.split("|")[0]
    if pid in rows and len(cc) == 2: P_.append((pid, rows[pid]["selfPID"], rows[pid]["batchID"], featvec(cc, rows[pid])))
NAMES = sorted(P_[0][3])
def profile_table(recs, min_n=3):
    g = collections.defaultdict(list)
    for r in recs: g[r[1]].append(r[3])
    keep = sorted(p for p, v in g.items() if len(v) >= min_n)
    return keep, np.array([[np.nanmean([v.get(n, np.nan) for v in g[p]]) for n in NAMES] for p in keep])
hk, HM = profile_table(P_)
mu = np.nanmean(HM, axis=0); HM = np.where(np.isnan(HM), mu, HM); sd = HM.std(axis=0); sd[sd == 0] = 1
Z = (HM - mu) / sd
km = KMeans(2, n_init=20, random_state=0).fit(Z); pca = PCA(2).fit(Z); HP = pca.transform(Z)
# name the types by the human means: the type with more 'did_together' is "the stretch we shared"
i_tog = NAMES.index("what:did_together"); tog = [HM[km.labels_ == t, i_tog].mean() for t in (0, 1)]
TYPE = {int(np.argmax(tog)): "stretch_we_shared", int(np.argmin(tog)): "what_they_told_me"}
htype = {p: TYPE[int(l)] for p, l in zip(hk, km.labels_)}
res = {"people": {"n": len(hk), "type_share": {t: round(float(np.mean([v == t for v in htype.values()])), 3) for t in TYPE.values()},
                  "pc_mean": [round(float(x), 2) for x in HP.mean(0)], "pc_sd": [round(float(x), 2) for x in HP.std(0)]}, "models": {}}
rng = np.random.default_rng(2026)
def carry(recs, names=("words_moment", "details", "scope:one_moment", "what:did_together", "new_fact", "shared:nobody")):
    person = np.array([r[1] for r in recs]); batch = np.array([r[2] for r in recs]); out = {}
    nulls = []
    for _ in range(200):
        o = person.copy()
        for bt in np.unique(batch):
            idx = np.where(batch == bt)[0]; o[idx] = person[rng.permutation(idx)]
        nulls.append(o)
    for n in names:
        x = np.array([r[3].get(n, np.nan) for r in recs]); obs = icc(x, person); nl = [icc(x, p) for p in nulls]
        out[n] = [round(obs, 3), round(float(np.nanpercentile(nl, 95)), 3)]
    return out
res["people"]["carry"] = carry(P_)
import prompts as PP
def donor_pid(row, cond):
    try:
        if cond.endswith("history_wrong"):
            return PP.donor(row, need_hist=len(PP.history(row)))[0]["selfPID"]
        if cond.endswith("card_thin_wrong"): return PP.donor(row, rung="thin")[0]["selfPID"]
        if cond.endswith("card_narr_wrong"): return PP.donor(row, rung="narr")[0]["selfPID"]
    except Exception:
        return None
for cond, (cdirs, runs) in CONDS.items():
    C = coded(cdirs); text = {}
    for run in runs:
        for l in open(os.path.join(ROOT, "runs", run, "preds.jsonl")):
            j = json.loads(l)
            if j.get("ok"): text[(j["perspectiveID"], j["model"].split("/")[1], j["info"])] = j["pred"]
    print("..", cond, "coded keys", len(C), flush=True)
    for m in M5 + ["ALL5"]:
        recs = []
        for k, cc in C.items():
            pid, mdl, info = k.split("|")
            if (m != "ALL5" and mdl != m) or mdl not in M5 or pid not in rows or len(cc) != 2: continue
            pr = text.get((pid, mdl, info)) or {}
            fake = {"conv_SP_Mem_descr": pr.get("moment", ""), "conv_SP_Mem_reason": pr.get("reason", "")}
            who = rows[pid]["selfPID"] if m != "ALL5" else mdl + "::" + rows[pid]["selfPID"]
            recs.append((pid, who, rows[pid]["batchID"], featvec(cc, fake)))
        if not recs: continue
        mk, MM = profile_table(recs); MM = np.where(np.isnan(MM), mu, MM); MZ = (MM - mu) / sd
        lab = km.predict(MZ); MP = pca.transform(MZ)
        r = {"n_profiles": len(mk), "type_share": {t: round(float(np.mean([TYPE[int(x)] == t for x in lab])), 3) for t in TYPE.values()},
             "pc_mean": [round(float(x), 2) for x in MP.mean(0)], "pc_sd": [round(float(x), 2) for x in MP.std(0)],
             "words_moment_median": round(float(np.expm1(np.median(MM[:, NAMES.index("words_moment")]))), 1)}
        if m != "ALL5":
            r["carry"] = carry(recs)
            t_m = {p: TYPE[int(x)] for p, x in zip(mk, lab)}; both = [p for p in mk if p in htype]
            acc = float(np.mean([t_m[p] == htype[p] for p in both]))
            pm = np.mean([t_m[p] == "stretch_we_shared" for p in both]); ph = np.mean([htype[p] == "stretch_we_shared" for p in both])
            r["type_match"] = dict(acc=round(acc, 3), chance=round(float(pm * ph + (1 - pm) * (1 - ph)), 3), n=len(both))
            if cond.endswith("_wrong"):
                dmap = {}
                for pid in rows:
                    sp = rows[pid]["selfPID"]
                    if sp in t_m and sp not in dmap:
                        dmap[sp] = donor_pid(rows[pid], cond)
                dd = [p for p in mk if dmap.get(p) in htype]
                if dd:
                    pdn = np.mean([htype[dmap[p]] == "stretch_we_shared" for p in dd]); pmm = np.mean([t_m[p] == "stretch_we_shared" for p in dd])
                    r["donor_type_match"] = dict(acc=round(float(np.mean([t_m[p] == htype[dmap[p]] for p in dd])), 3),
                                                 chance=round(float(pmm * pdn + (1 - pmm) * (1 - pdn)), 3), n=len(dd))
        else:
            sil = {}
            for k in (2, 3, 4):
                kk = KMeans(k, n_init=10, random_state=0).fit(MZ); sil[k] = round(float(silhouette_score(MZ, kk.labels_)), 3)
            r["own_clusters_silhouette"] = sil
            lab_by_model = collections.Counter((p.split("::")[0], TYPE[int(x)]) for p, x in zip(mk, lab))
            r["by_model_type"] = {f"{a}|{b}": n for (a, b), n in lab_by_model.items()}
        res["models"][f"{cond}|{m}"] = r
json.dump(res, open(os.path.join(ROOT, "analysis", "memory_styles_models.json"), "w"), indent=1)
print("PEOPLE", res["people"])
pc_ = res["people"]; pcar = np.mean([v[0] for v in pc_["carry"].values()])
print(f"\nSUMMARY (five models pooled): cond | share 'stretch we shared' (people {pc_['type_share']['stretch_we_shared']:.3f}) | PC1 mean, sd (people 0, {pc_['pc_sd'][0]:.2f}) | median words | mean carry ICC (people {pcar:.3f}) | own-type acc vs chance | donor-type acc vs chance")
for cond in CONDS:
    a = res["models"].get(f"{cond}|ALL5")
    if not a: continue
    per = [res["models"][f"{cond}|{m}"] for m in M5 if f"{cond}|{m}" in res["models"]]
    car = np.mean([np.mean([v[0] for v in r["carry"].values()]) for r in per]); acc = np.mean([r["type_match"]["acc"] for r in per]); ch = np.mean([r["type_match"]["chance"] for r in per])
    dn = [r["donor_type_match"] for r in per if "donor_type_match" in r]
    print(f"  {cond:22s} {a['type_share']['stretch_we_shared']:.3f} | {a['pc_mean'][0]:+.2f} {a['pc_sd'][0]:.2f} | {a['words_moment_median']:5.1f} | {car:+.3f} | {acc:.3f} vs {ch:.3f} | " + (f"{np.mean([d['acc'] for d in dn]):.3f} vs {np.mean([d['chance'] for d in dn]):.3f}" if dn else ""))
for k, r in res["models"].items():
    print(k, {x: r[x] for x in ("n_profiles", "type_share", "pc_mean", "pc_sd", "words_moment_median") if x in r}, r.get("type_match", ""))
for k, r in res["models"].items():
    if "carry" in r: print("carry", k, r["carry"])
