#!/usr/bin/env python3
"""Do three locators from three labs place people's memories alike? (A1, Oct 11.) Sonnet 5.5 (the paper's), Gemini 3 Flash, GPT-6 Luna,
same prompt. Turns are read as a consecutive range first..last (the prompt asks for consecutive turns; Gemini and Luna often return only
the end points). Per report: kind agreement; for reports all three place in the conversation, turn overlap (Jaccard), any overlap,
midpoint distance as a share of the call; each locator's position against the person's own timing answer; each against A1's checks.
Same for model guesses (none / history / history_wrong) where all three exist. PROMPT=v2 compares the revised prompt.
-> analysis/locator_agreement[_v2].json"""
import collections, glob, itertools, json, os, sys
import numpy as np
from scipy.stats import spearmanr
ROOT = ".."; A = os.path.join(ROOT, "analysis"); V2 = os.environ.get("PROMPT") == "v2"
def load(d):
    out = {}
    for f in glob.glob(os.path.join(A, d, "*", "PID_*.json")) + glob.glob(os.path.join(A, d, "PID_*.json")):
        j = json.load(open(f)); k = (j.get("model") if "/" not in str(j.get("model", "/")) else "", j["perspectiveID"]); out[k] = j
    return out
def rng_(j):
    t = j.get("turns") or []
    return set(range(min(t), max(t) + 1)) if (j.get("kind") == "in_conversation" and t) else None
def mid(j):
    r = rng_(j); return None if r is None else (np.mean(sorted(r)) + .5) / j["n_turns"]
H = {"sonnet": "located_v2_sonnet" if V2 else "located", "gemini": "located_v2_g3f" if V2 else "located_g3f", "luna": "located_v2_luna" if V2 else "located_luna"}
SETS = {"people": H} if V2 else {"people": H, "model guess, none": {"sonnet": "located_preds", "gemini": "located_preds_g3f", "luna": "located_preds_luna"},
        "model guess, history": {"sonnet": "located_preds_history", "gemini": "located_preds_history_g3f", "luna": "located_preds_history_luna"},
        "model guess, stranger's history": {"sonnet": "located_preds_history_wrong", "gemini": "located_preds_history_wrong_g3f", "luna": "located_preds_history_wrong_luna"}}
TIME = {j["perspectiveID"]: j.get("human_timing") for j in load("located").values()}
out = {}
for name, dirs in SETS.items():
    D = {loc: load(d) for loc, d in dirs.items()}
    keys = set.intersection(*(set(v) for v in D.values()))
    kinds = {loc: collections.Counter(D[loc][k].get("kind") for k in keys) for loc in D}
    allk = np.mean([len({D[l][k].get("kind") for l in D}) == 1 for k in keys]) if keys else None
    pair = {}
    for a, b in itertools.combinations(D, 2):
        ka = [D[a][k].get("kind") == D[b][k].get("kind") for k in keys]
        both = [k for k in keys if rng_(D[a][k]) and rng_(D[b][k])]
        jac = [len(rng_(D[a][k]) & rng_(D[b][k])) / len(rng_(D[a][k]) | rng_(D[b][k])) for k in both]
        anyo = [bool(rng_(D[a][k]) & rng_(D[b][k])) for k in both]
        md = [abs(mid(D[a][k]) - mid(D[b][k])) for k in both]
        pair[f"{a}-{b}"] = dict(kind_agree=round(float(np.mean(ka)), 3), n=len(keys), both_placed=len(both), overlap_any=round(float(np.mean(anyo)), 3),
                                jaccard=round(float(np.mean(jac)), 3), median_mid_gap=round(float(np.median(md)), 3))
    tv = {}
    if name == "people":
        for loc in D:
            x = [(mid(D[loc][k]), TIME.get(k[1])) for k in keys if mid(D[loc][k]) is not None and TIME.get(k[1]) is not None]
            tv[loc] = dict(spearman=round(float(spearmanr(*zip(*x)).correlation), 3), n=len(x))
        all3 = [k for k in keys if all(rng_(D[l][k]) for l in D)]
        agree3 = [k for k in all3 if set.intersection(*(rng_(D[l][k]) for l in D))]
        tv["all three place it and share a turn"] = dict(share_of_all3=round(len(agree3) / len(all3), 3), n_all3=len(all3))
        x = [(mid(D["sonnet"][k]), TIME.get(k[1])) for k in agree3 if TIME.get(k[1]) is not None]; y = [(mid(D["sonnet"][k]), TIME.get(k[1])) for k in all3 if k not in agree3 and TIME.get(k[1]) is not None]
        tv["timing rho when all three share a turn / when they don't"] = [round(float(spearmanr(*zip(*x)).correlation), 3), len(x), round(float(spearmanr(*zip(*y)).correlation), 3) if len(y) > 5 else None, len(y)]
    out[name] = dict(n=len(keys), kinds={l: dict(v) for l, v in kinds.items()}, all_three_same_kind=None if allk is None else round(float(allk), 3), pairs=pair, vs_self_timing=tv)
    print("==", name, "n", len(keys), "all-3 same kind", out[name]["all_three_same_kind"]); [print("  ", p, v) for p, v in pair.items()]; [print("   timing", l, v) for l, v in tv.items()]
# against A1's checks (people, v1 prompt): did the other locators also hit the turns A1 accepted?
if not V2:
    D = {loc: load(d) for loc, d in H.items()}; res = collections.defaultdict(collections.Counter)
    for a, k in (("human_coding/answers/LOCATOR_A1_answers_20261010.json", "analysis/locator_a1_packet_key.json"), ("human_coding/answers/LOCATOR_A1_2_answers_20261011.json", "analysis/locator_a1_packet2_key.json")):
        ans = json.load(open(os.path.join(ROOT, a)))["answers"]; K = json.load(open(os.path.join(ROOT, k)))
        for i, v in K.items():
            if ans.get(i) != "yes" or v["kind"] != "in_conversation": continue
            ref = set(v["turns"]) if v["turns"] else None
            for loc in ("gemini", "luna"):
                j = D[loc].get(("", v["pid"])); r = rng_(j) if j else None
                res[loc]["overlaps A1-accepted turns" if (r and ref and r & ref) else ("other kind" if j and j.get("kind") != "in_conversation" else "elsewhere")] += 1
    out["vs_a1_accepted"] = {l: dict(c) for l, c in res.items()}; print("vs A1-accepted:", out["vs_a1_accepted"])
json.dump(out, open(os.path.join(A, "locator_agreement" + ("_v2" if V2 else "") + ".json"), "w"), indent=1)
