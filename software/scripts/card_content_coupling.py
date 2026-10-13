#!/usr/bin/env python3
"""Do the traits on a personality card steer WHAT a model remembers, or only how the moment feels?
For every yes/no content code (codebook v3.5.1 options for topic, remembered, new_info, extent, moment_text, shared_by,
personal_kinds, feelings, reason first choice, matters_for, bigger_meaning; rate 5-95%), we ask how well the six HEXACO
domain scores of the person on the card predict the code, cross-validated (logistic regression, 5 folds grouped by person,
model identity as a covariate in both the base and the trait model). Score = AUC(traits + model) - AUC(model only).
Conditions (same coder on every side): no card (the null: traits are not in the prompt), the scores card, the narrative card,
and people's own memories with their own traits (human reference). Coders: Haiku and Luna separately.
-> analysis/card_content_coupling.json"""
import collections, csv, glob, json, os, sys, importlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "codebook")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scope_merge as SM
SPEC = SM.spec(importlib.import_module("spec_v351_FROZEN").SPEC)
import prompts as P
Q = {q["id"]: q for p in SPEC["parts"] if p.get("models") is not False for q in p["questions"]}
up = lambda x: str(x).strip().upper()
TR = "<LOCAL>/Documents/Master_Prompt_Workshop/LSI_StoriesOfYourLife_Canonical_PARSEL_Data/PARSEL_GroundTruth/V6_Canonical_290_Private_Human_Data/canonical_scale_scores_290.tsv"
DOM = ["HEXACO_HH_mean", "HEXACO_E_mean", "HEXACO_EX_mean", "HEXACO_A_mean", "HEXACO_C_mean", "HEXACO_OP_mean"]
T = {r["globalPID"]: [float(r[k]) for k in DOM] for r in csv.DictReader(open(TR), delimiter="\t") if all(r.get(k) not in ("", None, "nan") for k in DOM)}
rows = {r["perspectiveID"]: r for r in P.rows()}


def binaries(c):
    out = {}
    for qid, q in Q.items():
        v = c.get(qid)
        if q["type"] == "single" and not all(isinstance(o["code"], int) for o in q["options"] if o["code"] != "CANT_TELL"):
            v = v[0] if isinstance(v, list) and len(v) == 1 else v
            if v is None: continue
            for o in q["options"]: out[f"{qid}={o['code']}"] = int(up(v) == up(o["code"]))
        elif q["type"] == "multi":
            v = [v] if isinstance(v, str) else v
            if not v: continue
            s = {up(x) for x in v}
            for o in q["options"]: out[f"{qid}={o['code']}"] = int(up(o["code"]) in s)
        elif q["type"] == "nested_ranked" and isinstance(v, dict) and v.get("about"):
            for o in q["options"]: out[f"reason_first={o['code']}"] = int(up(v["about"][0]) == up(o["code"]))
    return out


def load(dirname, coder):
    recs = []
    for f in glob.glob(os.path.join(ROOT, "analysis", "code_v351", dirname, f"*__{coder}.json")):
        j = json.load(open(f))
        if not j.get("ok"): continue
        parts = j["key"].split("|"); pid = parts[0]; model = parts[1] if len(parts) > 1 else "human"
        r = rows.get(pid); t = T.get(r["selfPID"]) if r else None
        if t is None: continue
        recs.append((r["selfPID"], model, t, binaries(SM.codes(j["codes"]))))
    return recs


def score(recs):
    keys = sorted({k for r in recs for k in r[3]}); models = sorted({r[1] for r in recs}); out = {}
    for k in keys:
        sub = [r for r in recs if k in r[3]]
        y = np.array([r[3][k] for r in sub]); rate = y.mean()
        if not (.05 <= rate <= .95) or len(sub) < 200: continue
        M = np.array([[r[1] == m for m in models] for r in sub], float); X = np.array([r[2] for r in sub]); X = (X - X.mean(0)) / X.std(0)
        g = np.array([r[0] for r in sub]); pb, pt = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            for Z, p in ((M, pb), (np.hstack([M, X]), pt)):
                if len(np.unique(y[tr])) < 2: p[te] = rate; continue
                p[te] = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        base = roc_auc_score(y, pb) if len(models) > 1 else 0.5
        out[k] = {"rate": round(float(rate), 3), "n": len(y), "auc_gain": round(float(roc_auc_score(y, pt) - base), 4)}
    return out


def main():
    R = {}
    for coder in ["claude-haiku-5.5", "gpt-6-luna"]:
        R[coder] = {}
        for name, d in [("people", "human"), ("no_card", "main_text_cheap__none"), ("scores_card", "cards_text_cheap__card_thin"), ("narrative_card", "cards_narr_cheap__card_narr")]:
            recs = load(d, coder); s = score(recs); R[coder][name] = s
            g = [v["auc_gain"] for v in s.values()]
            print(f"{coder:18s} {name:15s} n={len(recs):6d} outcomes={len(g):3d} median AUC gain {np.median(g):+.3f}  max {max(g):+.3f}  n>.05 {sum(x > .05 for x in g)}", flush=True)
    json.dump(R, open(os.path.join(ROOT, "analysis", "card_content_coupling.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
