#!/usr/bin/env python3
"""Who discloses what (A1, Oct 7: "what gets disclosed, at what rate, by whom, based on what kind of predictable collection of data").
Unit: one person in one call (facts extracted from that call, all kinds, before dropping). Outcomes per call: number of facts; facts of each
kind; trait-adjacent facts; whether they said their name; session-feeling facts.
1) Social relations split of per-call disclosure: discloser vs listener (partner) variance, crossed random effects within batch (as srm_v1).
2) Person-level predictors (mean per call): six HEXACO domains, social anxiety (SIAS), age, sex, education, from outside the UK. Standardized
   OLS; FDR (Benjamini-Hochberg) across all outcome x predictor tests.
3) Does disclosure help models read personality? Per-person impression accuracy (profile r between the 4-reader mean impression and
   self-report over 24 facets) against facts per call and trait-adjacent facts per call (Spearman), with total words spoken as a control.
Writes analysis/disclosure.json."""
import csv, glob, json, os, sys, collections, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INTAKE = os.path.expanduser("<LOCAL_FILE> Story/cleaned_curated_Intake_Session_new.csv")
KINDS = ["home_place", "work_money", "education", "family_relationships", "pets", "interests_activities", "health_body", "identity",
         "plans_daily_life", "opinions_beliefs", "study_participation_life", "study_procedure", "other"]


def per_call():
    rows = collections.defaultdict(lambda: collections.Counter())
    for f in glob.glob(os.path.join(ROOT, "facts", "coded_v2", "items", "*.json")):
        x = json.load(open(f)); k = (x["pid"], x["dyad"]); c = rows[k]
        c["n"] += 1; c["kind:" + x.get("kind", "other")] += 1; c["trait_adjacent"] += int(x.get("trait_adjacent", 0))
        c["said_name"] = max(c["said_name"], int(x.get("drop") == "name_only")); c["session_feeling"] += int(x.get("drop") == "session_feeling")
    # include calls with zero facts for people who had the call
    out = []
    for r in P.rows():
        if r["dyadID"] == "PID_n+PID_n":
            continue
        c = rows.get((r["selfPID"], r["dyadID"]), collections.Counter())
        out.append(dict(pid=r["selfPID"], partner=r["otherPID"], dyad=r["dyadID"], batch=r["batchID"], **{k: float(v) for k, v in c.items()}))
    return out


def srm(data, y):
    import pandas as pd, statsmodels.formula.api as smf
    df = pd.DataFrame([dict(y=d.get(y, 0.0), pid=d["pid"], partner=d["partner"], batch=d["batch"]) for d in data])
    if df.y.std() == 0:
        return None
    m = smf.mixedlm("y ~ 1", df, groups="batch", vc_formula={"pid": "0 + C(pid)", "partner": "0 + C(partner)"}).fit(reml=True)
    vc = dict(zip(m.model.exog_vc.names, m.vcomp)); tot = vc["pid"] + vc["partner"] + m.scale
    return dict(discloser=float(vc["pid"] / tot), listener=float(vc["partner"] / tot))


def bh(ps):
    ps = np.array(ps); o = np.argsort(ps); q = np.empty_like(ps); m = len(ps); prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = m - rank; prev = min(prev, ps[i] * m / k); q[i] = prev
    return q


def main():
    from scipy import stats
    data = per_call(); res = {}
    outs = ["n", "trait_adjacent", "said_name", "session_feeling"] + ["kind:" + k for k in KINDS]
    res["srm"] = {y: srm(data, y) for y in outs}
    for y in outs:
        if res["srm"][y]:
            print(f"SRM {y:28s} discloser {res['srm'][y]['discloser']:.2f}  listener {res['srm'][y]['listener']:.2f}")
    # person level
    per = collections.defaultdict(list)
    for d in data:
        per[d["pid"]].append(d)
    F = json.load(open(os.path.join(ROOT, "cards", "inputs", "facets.json")))
    DOM = {"H": ["Sincerity", "Fairness", "Greed Avoidance", "Modesty"], "E": ["Fearfulness", "Anxiety", "Dependence", "Sentimentality"],
           "X": ["Social Self-Esteem", "Social Boldness", "Sociability", "Liveliness"], "A": ["Forgiveness", "Gentleness", "Flexibility", "Patience"],
           "C": ["Organization", "Diligence", "Perfectionism", "Prudence"], "O": ["Aesthetic Appreciation", "Inquisitiveness", "Creativity", "Unconventionality"]}
    intake = {r["globalPID"]: r for r in csv.DictReader(open(INTAKE, encoding="utf-8-sig"))}
    EDU = {"Less than high school": 0, "High school graduate": 1, "College": 2, "Bachelor's degree": 3, "Master's degree": 4,
           "Professional degree (i.e. MD or JD)": 5, "Doctorate": 5}
    UK = {"United Kingdom", "England", "Scotland", "Wales", "Northern Ireland"}
    # SIAS from the canonical scored file (reverse-keyed items handled there), not raw item means
    CANON = {r["globalPID"]: float(r["SIAS_mean"]) for r in csv.DictReader(open(os.path.expanduser(
        "<LOCAL_FILE>")), delimiter="\t") if r["SIAS_mean"].strip()}
    X, Y, pids = [], collections.defaultdict(list), []
    for p, ds in per.items():
        if p not in F or p not in intake:
            continue
        it = intake[p]
        try:
            age = float(it["Age_part"])
        except Exception:
            continue
        x = [np.mean([F[p][f] for f in DOM[d]]) for d in "HEXACO"] + [CANON.get(p, np.nan), age, float(it["Sex_part"] == "Female"),
                                                                     EDU.get(it["Edu_part"], np.nan), float(it["Nat_part"] not in UK)]
        if any(np.isnan(x)):
            continue
        X.append(x); pids.append(p)
        for y in outs:
            Y[y].append(np.mean([d.get(y, 0.0) for d in ds]))
    X = np.array(X); names = list("HEXACO") + ["social_anxiety", "age", "female", "education", "not_UK"]
    Xs = (X - X.mean(0)) / X.std(0); Xs = np.column_stack([np.ones(len(Xs)), Xs])
    tests = []
    for y in outs:
        v = np.array(Y[y])
        if v.std() == 0:
            continue
        ys = (v - v.mean()) / v.std(); b, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        resid = ys - Xs @ b; s2 = resid @ resid / (len(ys) - Xs.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(Xs.T @ Xs)))
        for j, nm in enumerate(names):
            t = b[j + 1] / se[j + 1]; p = 2 * stats.t.sf(abs(t), len(ys) - Xs.shape[1])
            tests.append(dict(outcome=y, predictor=nm, beta=float(b[j + 1]), p=float(p)))
    q = bh([t["p"] for t in tests])
    for t, qq in zip(tests, q):
        t["q"] = float(qq)
    res["person_level"] = dict(n=len(pids), tests=tests)
    print(f"\nperson-level predictors (n={len(pids)}), FDR q<.05:")
    for t in sorted(tests, key=lambda t: t["q"]):
        if t["q"] < .05:
            print(f"  {t['outcome']:28s} {t['predictor']:15s} beta {t['beta']:+.2f}  q {t['q']:.3f}")
    # 3) disclosure and impression accuracy
    imp = json.load(open(os.path.join(ROOT, "cards", "inputs", "impression_facets.json")))
    sys.path.insert(0, os.path.join(ROOT, "cards")); import score_fidelity as SF
    acc, fpc, tpc, words = [], [], [], []
    for p in pids:
        if p not in imp:
            continue
        a = np.array([imp[p][i] for i in SF.IDS]); t = np.array([SF.TRUE[p][SF.MAP[i]] for i in SF.IDS])
        acc.append(np.corrcoef(a, t)[0, 1]); ds = per[p]
        fpc.append(np.mean([d.get("n", 0) for d in ds])); tpc.append(np.mean([d.get("trait_adjacent", 0) for d in ds]))
        w = 0
        for d in ds:
            j = json.load(open(os.path.join(P.TX, d["dyad"] + ".json")))
            w += sum(len(tt["text"].split()) for tt in j["turns"] if tt["spk"] == p)
        words.append(w / len(ds))
    def partial(x, y, z):
        rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return stats.pearsonr(ex, ey)
    res["impression_accuracy"] = dict(n=len(acc), mean_profile_r=float(np.mean(acc)),
        facts_per_call=dict(zip(("rho", "p"), map(float, stats.spearmanr(fpc, acc)))), trait_adjacent_per_call=dict(zip(("rho", "p"), map(float, stats.spearmanr(tpc, acc)))),
        words_per_call=dict(zip(("rho", "p"), map(float, stats.spearmanr(words, acc)))),
        trait_adjacent_partial_words=dict(zip(("r", "p"), map(float, partial(tpc, acc, words)))), facts_partial_words=dict(zip(("r", "p"), map(float, partial(fpc, acc, words)))))
    print("\nimpression accuracy:", json.dumps(res["impression_accuracy"], indent=0))
    json.dump(res, open(os.path.join(ROOT, "analysis", "disclosure.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
