#!/usr/bin/env python3
"""Conventional-ML baselines with feature-family ablations (A2's review §7-8; her dissertation ch. 4 protocol: Logistic Regression,
Random Forest, NuSVC, MLP vs a Dummy classifier, balanced accuracy, 100 resampled splits, feature ablation, paired tests vs dummy).
Splits are grouped by BATCH (so no person, and no partner, is in both train and test), 20% of batches held out.
Outcomes: binary features of each person's memory report, majority of the three v3.5.1 model coders (Haiku, Luna, DeepSeek).
Feature families (from the conversation and the people, never from the report): own traits, partner traits, context (round, group size),
talk structure (turns, words, shares, questions, laughter), content (sentence embeddings of each speaker's words).
LLM comparison: the same outcomes read off the five main models' predicted memories (no history / with history), coder-matched
(Haiku codes on both sides, then Luna). -> analysis/ml_baselines.json"""
import collections, csv, json, os, re, sys
import numpy as np
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import style_ceiling as sc
N_SPLITS = int(os.environ.get("N_SPLITS", 100))
TR = "<LOCAL>/Documents/Master_Prompt_Workshop/LSI_StoriesOfYourLife_Canonical_PARSEL_Data/PARSEL_GroundTruth/V6_Canonical_290_Private_Human_Data/canonical_scale_scores_290.tsv"
TRAITS = ["HEXACO_HH_mean", "HEXACO_E_mean", "HEXACO_EX_mean", "HEXACO_A_mean", "HEXACO_C_mean", "HEXACO_OP_mean", "SIAS_mean",
          "Trust_Propensity_mean", "Trust_Benevolence_mean", "Trust_Ability_mean", "Trust_Integrity_mean",
          "PPTS_Affective_Responsiveness_mean", "PPTS_Cognitive_Responsiveness_mean", "PPTS_Interpersonal_Manipulation_mean", "PPTS_Egocentricity_mean"]
T = {r["globalPID"]: r for r in csv.DictReader(open(TR), delimiter="\t")}
gid = {r["selfPID"]: r["globalPID"] for r in rows.values()}
def traits(pid):
    t = T.get(gid.get(pid)); return [float(t[k]) if t and t.get(k) not in (None, "", "nan") else np.nan for k in TRAITS]
H = {c: load("human", c) for c in CODERS + ["deepseek-v4.1-flash"]}
items = sorted(k for k in H["claude-haiku-5.5"] if all(k in H[c] for c in H))
OUT = {"extent=one moment": lambda n: None if n["extent"] is None else n["extent"] == "ONE_MOMENT",
       "about the partner": lambda n: None if n["remembered"] is None else bool(n["remembered"] & {"PARTNER_SAID_OR_DID", "PARTNER_FACT_OR_QUALITY"}),
       "a feeling shown": lambda n: None if n["feelings"] is None else n["feelings"] != frozenset({"NONE"}),
       "new fact learned": lambda n: None if n["new_info"] is None else n["new_info"] == "YES",
       "reason mainly the partner": lambda n: None if n["reason.first"] is None else n["reason.first"] == "PARTNER",
       "matters for connection": lambda n: None if n["matters_for"] is None else "CONNECTION" in n["matters_for"],
       "plain report of what happened": lambda n: None if n["moment_text"] is None else n["moment_text"] == "WHAT_HAPPENED",
       "3+ details": lambda n: None if n["details"] is None else n["details"] >= 3}
def majority(k, f):
    v = [f(H[c][k]) for c in H]; v = [x for x in v if x is not None]
    return None if len(v) < 2 else (sum(v) * 2 > len(v)) if sum(v) * 2 != len(v) else None
# features
def talk(r):
    j = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json"))); me = r["selfPID"]
    t_me = [t["text"] for t in j["turns"] if t["spk"] == me]; t_p = [t["text"] for t in j["turns"] if t["spk"] != me]
    w = lambda ts: sum(len(x.split()) for x in ts); laugh = lambda ts: sum(len(re.findall(r"\b(ha+h*a*|haha\w*|lol|laugh\w*)\b", x.lower())) for x in ts)
    tot_w = max(1, w(t_me) + w(t_p))
    return [len(j["turns"]), w(t_me) + w(t_p), w(t_me) / tot_w, len(t_me) / max(1, len(j["turns"])), sum(x.count("?") for x in t_me), sum(x.count("?") for x in t_p),
            laugh(t_me), laugh(t_p), np.mean([len(x.split()) for x in t_me]) if t_me else 0, np.mean([len(x.split()) for x in t_p]) if t_p else 0], " ".join(t_me), " ".join(t_p)
gsize = collections.Counter(); 
for r in rows.values(): gsize[r["batchID"]] += 0
people_in_batch = collections.defaultdict(set)
for r in rows.values(): people_in_batch[r["batchID"]].add(r["selfPID"])
F = {"own traits": [], "partner traits": [], "context": [], "talk structure": [], "content": []}; me_txt, p_txt = [], []
for k in items:
    r = rows[k]; ts, a, b = talk(r)
    F["own traits"].append(traits(r["selfPID"])); F["partner traits"].append(traits(r["otherPID"]))
    F["context"].append([int(r["roundIdx"]), len(people_in_batch[r["batchID"]])]); F["talk structure"].append(ts); me_txt.append(a or "(silence)"); p_txt.append(b or "(silence)")
E1, E2 = sc.embed(me_txt), sc.embed(p_txt); F["content"] = np.hstack([E1, E2])
for f in F: F[f] = np.nan_to_num(np.asarray(F[f], float), nan=np.nanmean(np.asarray(F[f], float)))
groups = np.array([rows[k]["batchID"] for k in items])
MODELS = {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)),
          "RF": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=5, n_jobs=-1, random_state=0),
          "NuSVC": lambda: make_pipeline(StandardScaler(), NuSVC(nu=0.3, class_weight="balanced")),
          "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=0)),
          "Dummy": lambda: DummyClassifier(strategy="stratified", random_state=0)}
def evaluate(X, y, mk, splits):
    s = []
    for tr, te in splits:
        m = mk(); m.fit(X[tr], y[tr]); s.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return np.array(s)
res = {"n_items": len(items), "outcomes": {}}
by_person = collections.defaultdict(list)
for k in items: by_person[rows[k]["selfPID"]].append(k)
for name, f in OUT.items():
    y_all = [majority(k, f) for k in items]; keep = np.array([v is not None for v in y_all]); y = np.array([bool(v) for v, kk in zip(y_all, keep) if kk])
    lab = dict(zip(items, y_all)); base = float(np.mean([v for v in y_all if v is not None]))
    prior = []   # own EARLIER memories only (prospective; mirrors the models' earlier-calls-only history): rate and count
    for k in items:
        r = rows[k]; e = [lab[o] for o in by_person[r["selfPID"]] if int(rows[o]["roundIdx"]) < int(r["roundIdx"]) and lab[o] is not None]
        prior.append([float(np.mean(e)) if e else base, len(e)])
    F["own earlier memories"] = np.asarray(prior, float)
    Xf = {fam: F[fam][keep] for fam in F}; g = groups[keep]; X = np.hstack([Xf[fam] for fam in F])
    splits = list(GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=0).split(X, y, g))
    R = {"n": int(keep.sum()), "positive_rate": round(float(y.mean()), 3), "models": {}, "ablation_LR": {}}
    sc_ = {}
    for mn, mk in MODELS.items():
        sc_[mn] = evaluate(X, y, mk, splits); R["models"][mn] = dict(bal_acc=round(float(sc_[mn].mean()), 3), sd=round(float(sc_[mn].std()), 3))
    for mn in MODELS:
        if mn != "Dummy": d = sc_[mn] - sc_["Dummy"]; R["models"][mn]["beats_dummy_share_of_splits"] = round(float((d > 0).mean()), 3)
    for fam in F:   # leave one family out, LR
        Xa = np.hstack([Xf[f2] for f2 in F if f2 != fam]); s = evaluate(Xa, y, MODELS["LR"], splits)
        R["ablation_LR"][fam] = dict(bal_acc=round(float(s.mean()), 3), drop=round(float(sc_["LR"].mean() - s.mean()), 3))
    R["single_family_LR"] = {}
    for fam in F:   # each family alone, LR
        s1 = evaluate(Xf[fam], y, MODELS["LR"], splits); R["single_family_LR"][fam] = round(float(s1.mean()), 3)
    res["outcomes"][name] = R
    print(f"{name:30s} n={R['n']} pos={R['positive_rate']}  " + "  ".join(f"{m} {v['bal_acc']:.3f}" for m, v in R["models"].items()) +
          "  | drop w/o: " + ", ".join(f"{f2} {v['drop']:+.3f}" for f2, v in R["ablation_LR"].items()), flush=True)
# LLM predictions read the same way (coder-matched), all items
for cond, sd in (("none", "main_text_cheap__none"), ("history", "main_text_cheap__history"), ("history_prior", None)):
    for coder in CODERS:
        if sd: Pm = load(sd, coder)
        else:
            Pm = {}
            for short in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8"): Pm.update(load(f"r3c_{short}__history_prior", coder))
        Hc = H[coder]
        for name, f in OUT.items():
            yt, yp = [], []
            for key, pr in Pm.items():
                pid = key.split("|")[0]
                if pid in Hc:
                    a, b = f(Hc[pid]), f(pr)
                    if a is not None and b is not None: yt.append(a); yp.append(b)
            res["outcomes"][name].setdefault("llm", {})[f"{cond}|{coder}"] = round(float(balanced_accuracy_score(yt, yp)), 3)
for name, R in res["outcomes"].items(): print(f"{name:30s} LLM bal acc: {R['llm']}")
json.dump(res, open(os.path.join(ROOT, "analysis", "ml_baselines.json"), "w"), indent=1)
