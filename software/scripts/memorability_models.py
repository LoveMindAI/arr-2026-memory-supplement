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

# ---- Item 3 (A1, Oct 9): where do the MODELS' predicted moments land, and for the same reasons?
M5 = ["deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "mimo-v2.6-flash", "qwen3.8-omni-flash"]
def located(d, pid):
    p = os.path.join(d, pid + ".json")
    if not os.path.exists(p): return None
    L = json.load(open(p))
    if L.get("kind") != "in_conversation" or not L.get("turns"): return None
    t = L["turns"]; return set(json.loads(t) if isinstance(t, str) else t)
pos_of = F["position"][:, 2]                    # relative turn index within the call
res = {"models": {}}
for m in M5:
    d = os.path.join(ROOT, "analysis", "located_preds", m); ym = np.full(len(order), -1)
    for n_, i in enumerate(order):
        L = located(d, units[i]["pid"])
        if L is not None: ym[n_] = int(units[i]["k"] in L)
    have = ym >= 0
    calls_h = np.unique(calls[have])
    # overlap with the person's own remembered stretch (Jaccard over turns, per report)
    jac = []
    for c in calls_h:
        ix = np.where((calls == c) & have)[0]; a, b = set(ix[y[ix] == 1]), set(ix[ym[ix] == 1])
        if a or b: jac.append(len(a & b) / len(a | b))
    # where in the call (fifths), model vs people, on the same reports
    bins = np.linspace(0, 1, 6); hp = np.histogram(pos_of[have & (y == 1)], bins)[0]; hm = np.histogram(pos_of[have & (ym == 1)], bins)[0]
    # does the people-trained memorability model predict the model's choice? (train on people's labels, test on model labels, grouped splits)
    X = np.hstack([F[f] for f in FAMS]); sp = list(GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=0).split(X, y, groups))
    ba_h, ba_m = [], []
    for tr, te in sp:
        te = te[have[te]]; mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X[tr], y[tr]); pr = mdl.predict(X[te])
        ba_h.append(balanced_accuracy_score(y[te], pr)); ba_m.append(balanced_accuracy_score(ym[te], pr))
    # which features predict the MODEL's choice, each family alone (LR, 20 splits), vs people's
    alone = {}
    for f in FAMS:
        a1, a2 = [], []
        for tr, te in sp:
            trm, tem = tr[have[tr]], te[have[te]]
            a1.append(balanced_accuracy_score(ym[tem], make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(F[f][trm], ym[trm]).predict(F[f][tem])))
            a2.append(balanced_accuracy_score(y[tem], make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(F[f][trm], y[trm]).predict(F[f][tem])))
        alone[f] = dict(model_choice=round(float(np.mean(a1)), 3), people=round(float(np.mean(a2)), 3))
    res["models"][m] = dict(reports=int(len(calls_h)), jaccard_with_person=round(float(np.mean(jac)), 3), share_by_fifth_people=[round(x / hp.sum(), 3) for x in hp],
                            share_by_fifth_model=[round(x / hm.sum(), 3) for x in hm], people_trained_model_on_people=round(float(np.mean(ba_h)), 3),
                            people_trained_model_on_model_choice=round(float(np.mean(ba_m)), 3), family_alone=alone)
    print(m, json.dumps({k: v for k, v in res["models"][m].items() if k != "family_alone"}), flush=True)
# partner's own remembered stretch as the human reference for overlap
jp = []
for c in np.unique(calls):
    r = rows[c]; pp = f"{r['otherPID']}+{r['selfPID']}"; L = located(os.path.join(ROOT, "analysis", "located"), pp)
    if L is None: continue
    ix = np.where(calls == c)[0]; a = {units[order[i]]["k"] for i in ix if y[i] == 1}
    if a or L: jp.append(len(a & L) / len(a | L))
res["partner_jaccard_with_person"] = round(float(np.mean(jp)), 3); print("partner Jaccard", res["partner_jaccard_with_person"])
json.dump(res, open(os.path.join(ROOT, "analysis", "memorability_models" + ("_filled" if os.environ.get("LOC_FILL") == "1" else "") + ".json"), "w"), indent=1)
print("written")
