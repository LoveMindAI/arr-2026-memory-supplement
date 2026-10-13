#!/usr/bin/env python3
"""Card fidelity: how well each reader recovers the person's true 24 HEXACO facets from a card (Altruism ignored).
Metrics per (reader, writer): pooled CCC over person x facet; mean over facets of the across-people Pearson r; same at domain level (6 domains).
Paired writer contrasts on per-person profile CCC (24 facets), bootstrap over people. Writes analysis/card_fidelity.json."""
import json, os, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MAP = {"H_Sincerity": "Sincerity", "H_Fairness": "Fairness", "H_GreedAvoid": "Greed Avoidance", "H_Modesty": "Modesty", "E_Fearful": "Fearfulness",
       "E_Anxiety": "Anxiety", "E_Dependence": "Dependence", "E_Sentimentality": "Sentimentality", "X_SocialSelfEsteem": "Social Self-Esteem",
       "X_SocialBoldness": "Social Boldness", "X_Sociability": "Sociability", "X_Liveliness": "Liveliness", "A_Forgiveness": "Forgiveness",
       "A_Gentleness": "Gentleness", "A_Flexibility": "Flexibility", "A_Patience": "Patience", "C_Organization": "Organization", "C_Diligence": "Diligence",
       "C_Perfectionism": "Perfectionism", "C_Prudence": "Prudence", "O_Aesthetic": "Aesthetic Appreciation", "O_Inquisitive": "Inquisitiveness",
       "O_Creativity": "Creativity", "O_Unconventional": "Unconventionality"}
IDS = list(MAP)
TRUE = json.load(open(os.path.join(HERE, "inputs", "facets.json")))


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


def load():
    R = collections.defaultdict(dict)   # (reader, writer) -> {key: (pid, pred vec)}
    for f in glob.glob(os.path.join(HERE, "reads_api", "*", "*.json")):
        d = json.load(open(f)); rd = os.path.basename(os.path.dirname(f))
        R[(rd, d["writer"])][d["card_id"]] = (d["pid"], np.array([d["facets"][i] for i in IDS]))
    rb = os.path.join(HERE, "run_cards1", "cards_out", "reads_readerB.jsonl")
    if os.path.exists(rb):
        for l in open(rb):
            d = json.loads(l)
            try:
                t = d["read"]; fac = json.loads(t[t.find("{"): t.rfind("}") + 1])["facets"]
            except Exception:
                continue
            if not all(i in fac and isinstance(fac[i], (int, float)) and 1 <= fac[i] <= 5 for i in IDS):
                continue
            kind = d["kind"].split("/")[-1]
            w = "lcw150_personpass_person" if kind == "lcw150_personpass" else kind
            pid = d["key"].split("+")[0]
            R[("readerB", w)][f"{w}__{d['key']}"] = (pid, np.array([fac[i] for i in IDS]))
    return R


def summarize(items):
    P = np.array([v for _, v in items]); T = np.array([[TRUE[p][MAP[i]] for i in IDS] for p, _ in items])
    dom = lambda M: M.reshape(len(M), 6, 4).mean(2) if False else np.stack([M[:, [j for j, i in enumerate(IDS) if i[0] == d]].mean(1) for d in "HEXACO"], 1)
    fr = np.mean([np.corrcoef(P[:, j], T[:, j])[0, 1] for j in range(24)])
    dP, dT = dom(P), dom(T)
    dr = np.mean([np.corrcoef(dP[:, j], dT[:, j])[0, 1] for j in range(6)])
    return dict(n=len(items), pooled_ccc=float(ccc(P.ravel(), T.ravel())), facet_r=float(fr), domain_r=float(dr),
                domain_r_each={d: float(np.corrcoef(dP[:, j], dT[:, j])[0, 1]) for j, d in enumerate("HEXACO")})


def per_person(items):  # profile CCC per person (mean over that person's cards)
    acc = collections.defaultdict(list)
    for p, v in items:
        acc[p].append(ccc(v, [TRUE[p][MAP[i]] for i in IDS]))
    return {p: float(np.mean(v)) for p, v in acc.items()}


def main():
    R = load(); res = {"cells": {}, "contrasts": {}}
    for (rd, w), d in sorted(R.items()):
        res["cells"][f"{rd}|{w}"] = summarize(list(d.values()))
        print(rd, w, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["cells"][f"{rd}|{w}"].items() if k != "domain_r_each"})
    rng = np.random.default_rng(0)
    readers = sorted({rd for rd, _ in R}); writers = sorted({w for _, w in R})
    for rd in readers:
        pp = {w: per_person(list(R[(rd, w)].values())) for w in writers if (rd, w) in R}
        for a in pp:
            for b in pp:
                if a < b:
                    common = sorted(set(pp[a]) & set(pp[b]))
                    if len(common) < 20:
                        continue
                    dlt = np.array([pp[a][p] - pp[b][p] for p in common])
                    bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
                    res["contrasts"][f"{rd}|{a}-{b}"] = dict(n=len(common), mean=float(dlt.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
                    print("  ", rd, a, "-", b, res["contrasts"][f"{rd}|{a}-{b}"])
    json.dump(res, open(os.path.join(HERE, "..", "analysis", "card_fidelity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
