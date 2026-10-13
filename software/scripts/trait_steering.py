#!/usr/bin/env python3
"""A1's lens (Oct 7): not "does the card bring the model closer to the person" but "does assigning a personality move the model off its
baseline, and does the movement follow the assigned traits?"
Outputs scored without a judge: pleasure, arousal, dominance, timing, getting_along, minimize_conflict, unit (the survey items),
plus words in the moment, first-person rate (I/me/my per word) and partner-reference rate.
For each source (humans; each cheap model under none / card_thin / card_narr / card_person), predict each output from the person's six
HEXACO domain means with ridge regression, cross-validated by person (5 folds). Report CV R^2 per output, and the shift off the
no-card baseline (card - none, paired by memory, in SD units of the baseline). No judge involved; numbers come straight from outputs."""
import json, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FAC = json.load(open(os.path.join(ROOT, "cards", "inputs", "facets.json")))
DOM = {"H": ["Sincerity", "Fairness", "Greed Avoidance", "Modesty"], "E": ["Fearfulness", "Anxiety", "Dependence", "Sentimentality"],
       "X": ["Social Self-Esteem", "Social Boldness", "Sociability", "Liveliness"], "A": ["Forgiveness", "Gentleness", "Flexibility", "Patience"],
       "C": ["Organization", "Diligence", "Perfectionism", "Prudence"], "O": ["Aesthetic Appreciation", "Inquisitiveness", "Creativity", "Unconventionality"]}
TR = {p: np.array([np.mean([f[x] for x in DOM[d]]) for d in "HEXACO"]) for p, f in FAC.items()}
OUT = ["pleasure", "arousal", "dominance", "timing", "getting_along", "minimize_conflict", "unit", "words", "self_rate", "partner_rate"]
SELF = re.compile(r"\b(i|me|my|mine|myself|i'm|i've|i'd)\b", re.I)
OTHER = re.compile(r"\b(he|she|they|him|her|them|his|hers|their|partner|other person)\b", re.I)


def num(x):
    try:
        v = float(x); return v if np.isfinite(v) else None
    except Exception:
        return None


def feats(moment, d):
    w = str(moment).split(); n = max(1, len(w))
    f = {k: num(d.get(k)) for k in OUT[:7]}
    f.update(words=float(len(w)), self_rate=len(SELF.findall(str(moment))) / n, partner_rate=len(OTHER.findall(str(moment))) / n)
    return f


def human():
    out = {}
    for r in P.rows():
        d = dict(pleasure=r["conv_SP_Mem_p"], arousal=r["conv_SP_Mem_a"], dominance=r["conv_SP_Mem_d"], timing=r["conv_SP_Mem"],
                 getting_along=r["conv_TP_Mem_sc2"], minimize_conflict=r["conv_TP_Mem_sc3"], unit=r["conv_TP_Mem_ent1"])
        out[r["perspectiveID"]] = feats(r["conv_SP_Mem_descr"], d)
    return out


def model_preds():
    M = collections.defaultdict(dict)   # (model, info) -> pid -> feats
    for run in ["main_text_cheap", "cards_text_cheap", "cards_narr_cheap", "cards_person_cheap"]:
        for l in open(os.path.join(ROOT, "runs", run, "preds.jsonl")):
            j = json.loads(l)
            if j.get("ok") and j["info"] in ("none", "card_thin", "card_narr", "card_person"):
                M[(j["model"].split("/")[1], j["info"])][j["perspectiveID"]] = feats(j["pred"].get("moment", ""), j["pred"])
    return M


def cv_r2(X, y, groups, lam=1.0, k=5):
    ug = np.array(sorted(set(groups))); rng = np.random.default_rng(0); rng.shuffle(ug)
    fold = {g: i % k for i, g in enumerate(ug)}; f = np.array([fold[g] for g in groups]); pred = np.zeros_like(y)
    for i in range(k):
        tr, te = f != i, f == i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9; Xt = (X[tr] - mu) / sd; ym = y[tr].mean()
        b = np.linalg.solve(Xt.T @ Xt + lam * len(Xt) * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
        pred[te] = ym + ((X[te] - mu) / sd) @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def r2_table(data):
    res = {}
    for o in OUT:
        ks = [k for k, v in data.items() if k.split("+")[0] in TR and v.get(o) is not None]
        if len(ks) < 100:
            continue
        X = np.array([TR[k.split("+")[0]] for k in ks]); y = np.array([data[k][o] for k in ks]); g = [k.split("+")[0] for k in ks]
        if y.std() == 0:
            continue
        res[o] = dict(n=len(ks), cv_r2=float(cv_r2(X, y, g)),
                      r=[float(np.corrcoef(X[:, j], y)[0, 1]) for j in range(6)])
    return res


def main():
    H = human(); M = model_preds(); res = {"human": r2_table(H), "models": {}, "shift": {}}
    models = sorted({m for m, _ in M})
    for m in models:
        for info in ("none", "card_thin", "card_narr", "card_person"):
            if (m, info) in M:
                res["models"][f"{m}|{info}"] = r2_table(M[(m, info)])
        base = M.get((m, "none"), {})
        for info in ("card_thin", "card_narr", "card_person"):
            c = M.get((m, info), {}); sh = {}
            for o in OUT:
                ks = [k for k in base if k in c and base[k].get(o) is not None and c[k].get(o) is not None]
                if len(ks) < 100:
                    continue
                b = np.array([base[k][o] for k in ks]); x = np.array([c[k][o] for k in ks]); sd = b.std() or 1.0
                sh[o] = dict(mean_shift_sd=float((x - b).mean() / sd), mean_abs_shift_sd=float(np.abs(x - b).mean() / sd))
            res["shift"][f"{m}|{info}"] = sh
    json.dump(res, open(os.path.join(ROOT, "analysis", "trait_steering.json"), "w"), indent=1)
    # summary: CV R^2 by output, humans vs median over models per condition
    print(f"{'output':18s} {'human':>7s} " + " ".join(f"{c:>12s}" for c in ("none", "card_thin", "card_narr", "card_person")))
    for o in OUT:
        hv = res["human"].get(o, {}).get("cv_r2")
        row = []
        for info in ("none", "card_thin", "card_narr", "card_person"):
            v = [res["models"][f"{m}|{info}"][o]["cv_r2"] for m in models if f"{m}|{info}" in res["models"] and o in res["models"][f"{m}|{info}"]]
            row.append(f"{np.median(v):+.3f}({len(v)})" if v else "--")
        print(f"{o:18s} {('%+.3f' % hv) if hv is not None else '--':>7s} " + " ".join(f"{x:>12s}" for x in row))
    print("\nmean |shift| off the no-card baseline (SD units), median over models")
    for o in OUT:
        row = []
        for info in ("card_thin", "card_narr", "card_person"):
            v = [res["shift"][f"{m}|{info}"][o]["mean_abs_shift_sd"] for m in models if o in res["shift"].get(f"{m}|{info}", {})]
            row.append(f"{np.median(v):.2f}" if v else "--")
        print(f"{o:18s} " + " ".join(f"{x:>8s}" for x in row))


if __name__ == "__main__":
    main()
