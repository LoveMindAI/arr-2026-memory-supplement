#!/usr/bin/env python3
"""Analysis of disclosure depth in the calls (disclosure_depth.py output; Haiku 5.5 + Luna). No API calls. PYTHONHASHSEED=0.
Unit: one person in one call. Per coder: counts of surface (level 2) and deep (level 3) disclosures; the two coders are averaged.
(0) Coder agreement: Spearman of per-person-call deep and surface counts between coders; agreement on "any deep".
(1) How much and how deep: per person-call means; share of calls where someone goes deep.
(2) Whose is it? ICC of a person's depth across their calls vs the within-session shuffle (as memory_styles.py);
    partner effect: my depth in this call vs my partner's mean depth in their OTHER calls (Spearman, bootstrap over calls).
(3) Reciprocity: correlation of the two partners' depth in the same call, against random same-session pairings.
(4) Who goes deep: person mean depth vs six HEXACO domains, social anxiety, age, female, education, not UK (standardized OLS, BH).
(5) Does it reach memory? (a) depth in the call vs the memory codes (shared_by not NOBODY; personal_kinds deep) of BOTH partners' reports;
    (b) vs partners remembering the same moment (shared_memory_rapport judges); (c) vs the post-call connection items; (d) memory type (k=2).
-> analysis/disclosure_depth.json"""
import collections, csv, glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
rows = {r["perspectiveID"]: r for r in P.rows()}
LV = {"INFO_GENERAL": 2, "THOUGHTS_EVENTS_PLANS": 2, "FEELINGS_MILD": 2, "INFO_REVEALING": 3, "THOUGHTS_INTIMATE": 3, "FEELINGS_DEEP": 3}
V2 = os.environ.get("SCHEME") == "v2"   # Oct 10: scheme v2 (A1's tightened rule), three coders
CODERS = ["claude-haiku-5.5", "gpt-6-luna", "gemini-3-flash-preview"] if V2 else ["claude-haiku-5.5", "gpt-6-luna"]
D = collections.defaultdict(dict)   # (dyad, pid) -> coder -> (surface, deep)
calls = collections.defaultdict(set)
for f in glob.glob(os.path.join(ROOT, "analysis", "disclosure_depth_v2" if V2 else "disclosure_depth", "*__*.json")):
    j = json.load(open(f))
    if not j["ok"]: continue
    c = j["coder"].split("/")[1]
    if c not in CODERS: continue
    for pid in j["speakers"].values():
        if pid: D[(j["dyad"], pid)][c] = [0, 0]
    for x in j["disclosures"]:
        if x.get("pid") and (j["dyad"], x["pid"]) in D:
            D[(j["dyad"], x["pid"])][c][(1 if x["depth"] == "DEEP" else 0) if V2 else LV[x["category"]] - 2] += 1
    calls[j["dyad"]].add(c)
both = {k: v for k, v in D.items() if all(c in v for c in CODERS)}
res = {"n_person_calls": len(both), "n_calls": len({k[0] for k in both})}
s1 = [both[k][CODERS[0]] for k in both]; s2 = [both[k][CODERS[1]] for k in both]
res["coder_agreement"] = {"deep_spearman": round(float(spearmanr([a[1] for a in s1], [b[1] for b in s2]).correlation), 3),
                          "surface_spearman": round(float(spearmanr([a[0] for a in s1], [b[0] for b in s2]).correlation), 3),
                          "any_deep_agree": round(float(np.mean([(a[1] > 0) == (b[1] > 0) for a, b in zip(s1, s2)])), 3),
                          "mean_deep_by_coder": [round(float(np.mean([a[1] for a in s1])), 2), round(float(np.mean([b[1] for b in s2])), 2)],
                          "mean_surface_by_coder": [round(float(np.mean([a[0] for a in s1])), 2), round(float(np.mean([b[0] for b in s2])), 2)]}
USE = [os.environ["ONLY_CODER"]] if os.environ.get("ONLY_CODER") else CODERS   # per-coder rerun (two-instrument check)
deep = {k: float(np.mean([v[c][1] for c in USE])) for k, v in both.items()}
surf = {k: float(np.mean([v[c][0] for c in USE])) for k, v in both.items()}
anydeep = {k: float(np.mean([v[c][1] > 0 for c in USE])) for k, v in both.items()}
by_call = collections.defaultdict(list)
for (d, p) in both: by_call[d].append(p)
res["describe"] = {"deep_per_person_call": round(float(np.mean(list(deep.values()))), 2), "surface_per_person_call": round(float(np.mean(list(surf.values()))), 2),
                   "share_person_calls_any_deep_both_coders": round(float(np.mean([v == 1 for v in anydeep.values()])), 3),
                   "share_calls_someone_deep_both_coders": round(float(np.mean([any(anydeep[(d, p)] == 1 for p in ps) for d, ps in by_call.items()])), 3)}
# (2) carry
batch_of = {}; self_rows = {}
for r in rows.values():
    batch_of[(r["dyadID"], r["selfPID"])] = r["batchID"]; self_rows[(r["dyadID"], r["selfPID"])] = r
keys = [k for k in both if k in batch_of]
person = np.array([k[1] for k in keys]); batch = np.array([batch_of[k] for k in keys])
rng = np.random.default_rng(2026)
def icc(x, g):
    ok = ~np.isnan(x); x, g = x[ok], g[ok]
    groups = collections.defaultdict(list)
    for v, p in zip(x, g): groups[p].append(v)
    gs = [np.array(v) for v in groups.values() if len(v) >= 2]
    k = np.mean([len(v) for v in gs]); grand = np.mean(np.concatenate(gs))
    msb = sum(len(v) * (v.mean() - grand) ** 2 for v in gs) / (len(gs) - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in gs) / (sum(len(v) for v in gs) - len(gs))
    return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else np.nan
def shuffle_within_session(lab):
    out = lab.copy()
    for bt in np.unique(batch):
        idx = np.where(batch == bt)[0]; out[idx] = lab[rng.permutation(idx)]
    return out
nulls = [shuffle_within_session(person) for _ in range(500)]
res["carry"] = {}
for name, M in (("deep", deep), ("surface", surf), ("any_deep", anydeep)):
    x = np.array([M[k] for k in keys]); obs = icc(x, person); nl = np.array([icc(x, p) for p in nulls])
    res["carry"][name] = dict(icc=round(obs, 3), null_95=round(float(np.nanpercentile(nl, 95)), 3), p=round(float((np.sum(nl >= obs) + 1) / 501), 4))
# partner effect: my depth here vs partner's mean depth in their other calls
pm = collections.defaultdict(list)
for (d, p), v in deep.items(): pm[p].append((d, v))
xs, ys, dd = [], [], []
for (d, p) in keys:
    others = [q for q in by_call[d] if q != p]
    if not others: continue
    q = others[0]; o = [v for dd_, v in pm[q] if dd_ != d]
    if o: xs.append(np.mean(o)); ys.append(deep[(d, p)]); dd.append(d)
r = spearmanr(xs, ys).correlation; xs, ys = np.array(xs), np.array(ys)
bs = [spearmanr(xs[i], ys[i]).correlation for i in (rng.integers(0, len(xs), len(xs)) for _ in range(2000))]
own_x = []
for (d, p) in keys:
    o = [v for dd_, v in pm[p] if dd_ != d]
    if o: own_x.append((np.mean(o), deep[(d, p)]))
ro = spearmanr([a for a, _ in own_x], [b for _, b in own_x]).correlation
res["whose"] = {"own_other_calls_rho": round(float(ro), 3), "partner_other_calls_rho": round(float(r), 3),
                "partner_ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": len(xs)}
# (3) reciprocity
pairs = [(deep[(d, ps[0])], deep[(d, ps[1])]) for d, ps in by_call.items() if len(ps) == 2]
obs = spearmanr([a for a, _ in pairs], [b for _, b in pairs]).correlation
bcall = collections.defaultdict(list)
for d, ps in by_call.items():
    if len(ps) == 2 and (d, ps[0]) in batch_of: bcall[batch_of[(d, ps[0])]].append(d)
nl = []
for _ in range(500):
    a_, b_ = [], []
    for bt, ds in bcall.items():
        perm = rng.permutation(len(ds))
        for i, j in enumerate(perm):
            a_.append(deep[(ds[i], by_call[ds[i]][0])]); b_.append(deep[(ds[j], by_call[ds[j]][1])])
    nl.append(spearmanr(a_, b_).correlation)
res["reciprocity"] = {"rho_same_call": round(float(obs), 3), "null_mean": round(float(np.mean(nl)), 3), "null_95": round(float(np.percentile(nl, 95)), 3), "n_calls": len(pairs)}
# (4) who goes deep
INTAKE = os.path.expanduser("<LOCAL_FILE> Story/cleaned_curated_Intake_Session_new.csv")
intake = {r["globalPID"]: r for r in csv.DictReader(open(INTAKE, encoding="utf-8-sig"))}
FAC = json.load(open(os.path.join(ROOT, "cards", "inputs", "facets.json")))
DOM = {"H": ["Sincerity", "Fairness", "Greed Avoidance", "Modesty"], "E": ["Fearfulness", "Anxiety", "Dependence", "Sentimentality"],
       "X": ["Social Self-Esteem", "Social Boldness", "Sociability", "Liveliness"], "A": ["Forgiveness", "Gentleness", "Flexibility", "Patience"],
       "C": ["Organization", "Diligence", "Perfectionism", "Prudence"], "O": ["Aesthetic Appreciation", "Inquisitiveness", "Creativity", "Unconventionality"]}
EDU = {"Less than high school": 0, "High school graduate": 1, "College": 2, "Bachelor's degree": 3, "Master's degree": 4, "Professional degree (i.e. MD or JD)": 5, "Doctorate": 5}
UK = {"United Kingdom", "England", "Scotland", "Wales", "Northern Ireland"}
CANON = {r["globalPID"]: float(r["SIAS_mean"]) for r in csv.DictReader(open(os.path.expanduser(
    "<LOCAL_FILE>")), delimiter="\t") if r["SIAS_mean"].strip()}
glob_of = {r["selfPID"]: (r.get("globalPID") or r["selfPID"]) for r in rows.values()}
X, Yd, Ys = [], [], []
for p in sorted(pm):
    g = glob_of.get(p, p)
    if g not in FAC or g not in intake: continue
    it = intake[g]
    try: age = float(it["Age_part"])
    except Exception: continue
    x = [np.mean([FAC[g][f] for f in DOM[d]]) for d in "HEXACO"] + [CANON.get(g, np.nan), age, float(it["Sex_part"] == "Female"), EDU.get(it["Edu_part"], np.nan), float(it["Nat_part"] not in UK)]
    if any(np.isnan(x)): continue
    X.append(x); Yd.append(np.mean([v for _, v in pm[p]])); Ys.append(np.mean([surf[(d, p)] for d, _ in pm[p]]))
X = np.array(X); names = list("HEXACO") + ["social_anxiety", "age", "female", "education", "not_UK"]
Xs = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / X.std(0)])
from scipy import stats
tests = []
for yname, y in (("deep", np.array(Yd)), ("surface", np.array(Ys))):
    yz = (y - y.mean()) / y.std(); beta, *_ = np.linalg.lstsq(Xs, yz, rcond=None)
    resid = yz - Xs @ beta; dof = len(yz) - Xs.shape[1]; s2 = resid @ resid / dof; se = np.sqrt(np.diag(s2 * np.linalg.inv(Xs.T @ Xs)))
    for i, n in enumerate(names, 1):
        t = beta[i] / se[i]; tests.append([yname, n, round(float(beta[i]), 3), float(2 * stats.t.sf(abs(t), dof))])
ps_ = np.array([t[3] for t in tests]); o = np.argsort(ps_); q = np.empty(len(ps_)); prev = 1.0; m = len(ps_)
for rank, i in reversed(list(enumerate(o, 1))): prev = min(prev, ps_[i] * m / rank); q[i] = prev
res["who"] = {"n_people": len(X), "tests": [t[:3] + [round(t[3], 4), round(float(q[i]), 4)] for i, t in enumerate(tests)]}
# (5) memory links
FIN = {}
for l in open(os.path.join(ROOT, "analysis", "code_v351", "human", "FINAL.jsonl")):
    j = json.loads(l); FIN[j["key"].split("|")[0]] = j["codes"]
def mem(pid):
    c = FIN.get(pid)
    if not c: return None
    sh = c.get("shared_by"); pk = set(c.get("personal_kinds") or [])
    return dict(shared=None if sh in (None, "TIE") else float(sh != "NOBODY"), mem_deep=float(bool(pk & {"info_revealing", "thoughts_intimate", "feelings_mild", "feelings_deep"})) if pk else None)
link = collections.defaultdict(lambda: ([], []))
for pid, r in rows.items():
    k = (r["dyadID"], r["selfPID"]); m = mem(pid)
    if k not in deep or not m: continue
    call_deep = sum(deep[(r["dyadID"], q)] for q in by_call[r["dyadID"]])
    for name, v in m.items():
        if v is not None: link[name][0].append(call_deep); link[name][1].append(v)
res["memory_link"] = {n: dict(rho=round(float(spearmanr(a, b).correlation), 3), p=round(float(spearmanr(a, b).pvalue), 5), n=len(a)) for n, (a, b) in link.items()}
try:
    sm = json.load(open(os.path.join(ROOT, "analysis", "shared_memory_rapport.json")))
except Exception:
    sm = None
CONN = {"rapport": "conv_CC_rapp1", "sync": "conv_CC_sync1", "liking": "conv_PP_like", "similar": "conv_PP_sim", "meet_again": "conv_PP_Will"}
def num(v):
    try: v = float(v); return v if np.isfinite(v) else None
    except Exception: return None
cl = {}
for name, col in CONN.items():
    a, b = [], []
    for pid, r in rows.items():
        k = (r["dyadID"], r["selfPID"]); v = num(r[col])
        if k in deep and v is not None:
            a.append(sum(deep[(r["dyadID"], q)] for q in by_call[r["dyadID"]])); b.append(v)
    cc = spearmanr(a, b); cl[name] = dict(rho=round(float(cc.correlation), 3), p=round(float(cc.pvalue), 5), n=len(a))
res["connection_link"] = cl
json.dump(res, open(os.path.join(ROOT, "analysis", "disclosure_depth" + ("_v2" if V2 else "") + ("_" + os.environ["ONLY_CODER"] if os.environ.get("ONLY_CODER") else "") + ".json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "who"}, indent=1))
print("who q<.10:", [t for t in res["who"]["tests"] if t[4] < .10], "min q", min(t[4] for t in res["who"]["tests"]))
