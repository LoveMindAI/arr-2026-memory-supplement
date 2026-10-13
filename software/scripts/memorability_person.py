#!/usr/bin/env python3
"""Memorability prediction (A2, Oct 9: one memory-related target, replicating MeMo, with feature ablation).
Unit = one turn of a call, seen from one rememberer. Label = 1 if the turn is in the stretch that rememberer reported (LLM locator,
analysis/located/; checked by A1 on 48 reports: 41 right, 3 partly right, 0 wrong). Only reports the locator placed in the conversation.
Protocol (Tsfasman dissertation ch. 4): LR, RF, NuSVC, MLP vs Dummy; balanced accuracy; 100 resampled splits (grouped by batch, 20% of
batches held out); feature-family ablation (leave one out, and each alone). Also hit@1: is the call's top-scored turn inside the stretch.
Families (her review §8): position, speaker, length & timing, lexical, semantic novelty, sentiment (VADER), dialogue structure, content
(sentence embedding, PCA 16), the rememberer's traits. -> analysis/memorability.json"""
import collections, csv, json, os, re, sys
import numpy as np
import prompts as P
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
ROOT = ".."; N_SPLITS = int(os.environ.get("N_SPLITS", 100)); N_ABL_RF = int(os.environ.get("N_ABL_RF", 20)); rng = np.random.default_rng(0)
rows = {r["perspectiveID"]: r for r in P.rows()}
TR = "<LOCAL>/Documents/Master_Prompt_Workshop/LSI_StoriesOfYourLife_Canonical_PARSEL_Data/PARSEL_GroundTruth/V6_Canonical_290_Private_Human_Data/canonical_scale_scores_290.tsv"
TRAITS = ["HEXACO_HH_mean", "HEXACO_E_mean", "HEXACO_EX_mean", "HEXACO_A_mean", "HEXACO_C_mean", "HEXACO_OP_mean", "SIAS_mean",
          "Trust_Propensity_mean", "Trust_Benevolence_mean", "Trust_Ability_mean", "Trust_Integrity_mean",
          "PPTS_Affective_Responsiveness_mean", "PPTS_Cognitive_Responsiveness_mean", "PPTS_Interpersonal_Manipulation_mean", "PPTS_Egocentricity_mean"]
T = {r["globalPID"]: r for r in csv.DictReader(open(TR), delimiter="\t")}
def traits(pid):
    t = T.get(pid); return [float(t[k]) if t and t.get(k) not in (None, "", "nan") else np.nan for k in TRAITS]
VS = SentimentIntensityAnalyzer()
LAUGH = re.compile(r"\b(ha+h*a*|haha\w*|lol|laugh\w*)\b"); W = lambda s: re.findall(r"[A-Za-z']+", s)
# ---- units
units, texts = [], []
for f in sorted(x for x in os.listdir(os.path.join(ROOT, "analysis", "located")) if x.startswith("PID_") and x.endswith(".json")):
    L = json.load(open(os.path.join(ROOT, "analysis", "located", f)))
    if L.get("kind") != "in_conversation": continue
    pid = L["perspectiveID"]; r = rows[pid]; loc = L["turns"]; loc = set(json.loads(loc) if isinstance(loc, str) else (loc or []))
    if os.environ.get("LOC_FILL") == "1" and loc: loc = set(range(min(loc), max(loc) + 1))   # sensitivity (Oct 12): fill each listed span first-to-last
    if not loc: continue
    turns = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]; n = len(turns); end = max(t["start"] for t in turns) + 5
    for k, t in enumerate(turns):
        units.append(dict(pid=pid, batch=r["batchID"], k=k, n=n, y=int(k in loc), self=int(t["spk"] == r["selfPID"]), start=t["start"], end=end,
                          prev=turns[k - 1] if k else None, nxt=turns[k + 1] if k + 1 < n else None, text=t["text"], gid=r["globalPID"]))
        texts.append(t["text"] or "(silence)")
print("units", len(units), "positives", sum(u["y"] for u in units), "reports", len({u["pid"] for u in units}), flush=True)
EMB = os.path.join(ROOT, "analysis", "turn_emb_memorability.npy")
if os.path.exists(EMB): E = np.load(EMB)
else:
    import style_ceiling as sc; E = sc.embed(texts); np.save(EMB, E)
# ---- features
fam = collections.defaultdict(list); idx_by_call = collections.defaultdict(list)
for i, u in enumerate(units): idx_by_call[u["pid"]].append(i)
for pid, ii in idx_by_call.items():
    cen = E[ii].mean(0); cen /= np.linalg.norm(cen) + 1e-9
    for j, i in enumerate(ii):
        u = units[i]; t = u["text"] or ""; w = W(t); lw = [x.lower() for x in w]
        nxt_start = u["nxt"]["start"] if u["nxt"] else u["end"]; dur = max(0.5, nxt_start - u["start"])
        prev = u["prev"]; prev_q = int(bool(prev) and "?" in (prev["text"] or "") and prev["spk"] != (u.get("spk")))
        prevE = E[ii[max(0, j - 3):j]].mean(0) if j else None
        nov = 1 - float(E[i] @ (prevE / (np.linalg.norm(prevE) + 1e-9))) if prevE is not None else 0.0
        s = VS.polarity_scores(t)
        nxt = u["nxt"]; nxt_w = len(W(nxt["text"] or "")) if nxt else 0
        fam["position"].append([u["start"] / u["end"], u["start"], j / max(1, u["n"] - 1), int(u["start"] / u["end"] < .1), int(u["start"] / u["end"] > .9)])
        fam["speaker"].append([u["self"]])
        fam["length & timing"].append([len(w), dur, len(w) / dur, (u["start"] - prev["start"]) if prev else 0.0])
        fam["lexical"].append([t.count("?"), len(LAUGH.findall(t.lower())), sum(x in ("i", "i'm", "my", "me", "myself") for x in lw), sum(x in ("you", "your", "you're") for x in lw),
                               sum(x in ("we", "us", "our", "both") for x in lw), len(re.findall(r"\d", t)), sum(1 for x in w[1:] if x[:1].isupper() and x not in ("I", "I'm")), t.count("!")])
        fam["semantic novelty"].append([nov, 1 - float(E[i] @ cen)])
        fam["sentiment"].append([s["compound"], s["pos"], s["neg"]])
        fam["dialogue structure"].append([prev_q, int(len(w) <= 2), nxt_w, len(LAUGH.findall((nxt["text"] or "").lower())) if nxt else 0])
        fam["rememberer traits"].append(traits(u["gid"]))
order = [i for ii in idx_by_call.values() for i in ii]
y = np.array([units[i]["y"] for i in order]); groups = np.array([units[i]["batch"] for i in order]); calls = np.array([units[i]["pid"] for i in order])
F = {}
for f, v in fam.items():
    a = np.asarray(v, float); F[f] = np.where(np.isnan(a), np.nanmean(a, 0), a)
F["content"] = PCA(16, random_state=0).fit_transform(E[order])        # unsupervised, fit on all turns (no labels used)
FAMS = list(F)

# ---- Item 4 (A1, Oct 9): the person's own EARLIER remembered stretches (prospective; earlier rounds only, located in the conversation)
LOC = {}
for f in os.listdir(os.path.join(ROOT, "analysis", "located")):
    if not (f.startswith("PID_") and f.endswith(".json")): continue
    L = json.load(open(os.path.join(ROOT, "analysis", "located", f)))
    if L.get("kind") != "in_conversation" or not L.get("turns"): continue
    t = L["turns"]; t = json.loads(t) if isinstance(t, str) else t; n = int(L.get("n_turns") or 0)
    if n: LOC[L["perspectiveID"]] = (((t[0] + t[-1]) / 2) / max(1, n - 1), len(t) / n)
by_person = collections.defaultdict(list)
for pid in LOC: by_person[rows[pid]["selfPID"]].append(pid)
gpos, gspan = np.mean([v[0] for v in LOC.values()]), np.mean([v[1] for v in LOC.values()])
pers = []
for i in order:
    u = units[i]; r = rows[u["pid"]]
    e = [LOC[o] for o in by_person[r["selfPID"]] if int(rows[o]["roundIdx"]) < int(r["roundIdx"])]
    mp = float(np.mean([x[0] for x in e])) if e else gpos; ms = float(np.mean([x[1] for x in e])) if e else gspan
    rp = u["k"] / max(1, u["n"] - 1)
    pers.append([abs(rp - mp), mp, ms, len(e)])
F["person's earlier memories"] = np.asarray(pers, float)
MODELS = {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
          "RF": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_leaf=10, n_jobs=-1, random_state=0),
          "NuSVC": lambda: make_pipeline(StandardScaler(), NuSVC(nu=0.3, class_weight="balanced")),
          "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, random_state=0)),
          "Dummy": lambda: DummyClassifier(strategy="stratified", random_state=0)}
BAL = {"RF": None, "MLP": None, "NuSVC": 4000}   # balanced training sample: all positives (NuSVC: up to 4000) + the same number of negatives
def fit_eval(X, mk, tr, te, name, rank=False):
    trn = tr
    if name in BAL:
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        if BAL[name]: pos = rng.choice(pos, min(BAL[name], len(pos)), replace=False)
        trn = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
    m = mk(); m.fit(X[trn], y[trn]); pr = m.predict(X[te]); ba = balanced_accuracy_score(y[te], pr)
    h = None
    if rank:
        sc_ = m.predict_proba(X[te])[:, 1]; hits, ch = [], []
        for c in np.unique(calls[te]):
            ix = np.where(calls[te] == c)[0]; hits.append(y[te][ix[np.argmax(sc_[ix])]]); ch.append(y[te][ix].mean())
        h = (float(np.mean(hits)), float(np.mean(ch)))
    return ba, h
base_fams = [f for f in F if f != "person's earlier memories"]
Xb = np.hstack([F[f] for f in base_fams]); Xp = np.hstack([F[f] for f in F]); Xo = F["person's earlier memories"]
splits = list(GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=0).split(Xb, y, groups))
res = {"units": int(len(y)), "with_earlier": int((Xo[:, 3] > 0).sum()), "compare": {}}
for mname, nsp in (("LR", N_SPLITS), ("RF", N_ABL_RF)):
    sp = splits[:nsp]; R = {}
    for tag, X in (("conversation features", Xb), ("+ person's earlier memories", Xp), ("person's earlier memories alone", Xo)):
        o = [fit_eval(X, MODELS[mname], tr, te, mname, rank=True) for tr, te in sp]
        R[tag] = dict(bal_acc=round(float(np.mean([a[0] for a in o])), 3), hit_at_1=round(float(np.mean([a[1][0] for a in o])), 3), chance=round(float(np.mean([a[1][1] for a in o])), 3))
    d = [fit_eval(Xp, MODELS[mname], tr, te, mname)[0] - fit_eval(Xb, MODELS[mname], tr, te, mname)[0] for tr, te in sp]
    R["gain_share_of_splits"] = round(float((np.array(d) > 0).mean()), 3); res["compare"][mname] = R
    print(mname, json.dumps(R), flush=True)
json.dump(res, open(os.path.join(ROOT, "analysis", "memorability_person" + ("_filled" if os.environ.get("LOC_FILL") == "1" else "") + ".json"), "w"), indent=1)
print("written")
