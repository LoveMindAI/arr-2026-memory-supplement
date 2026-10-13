#!/usr/bin/env python3
"""Lay human readers of the lcw150 narrative cards (Prolific, agent2's handoff prolific/annotation_handoff_20261009), against the
person's true HEXACO scores, beside the AI readers (cards/reads_api, DeepSeek and Luna) on the same people.
24-facet readers: original_detailed + detailed_expansion (F items). Six-domain readers: six_trait_bridge (C items). The original broad
cohort is apparatus-affected (form defect) and excluded, per the handoff. Lin's CCC per reader (across facets or domains) and pooled.
Local, private inputs only. -> analysis/human_card_readers.json"""
import ast, collections, glob, json, os
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
d = json.load(open(os.path.join(ROOT, "prolific", "annotation_handoff_20261009", "ANNOTATIONS_DEIDENTIFIED_PRIVATE.json")))
K = json.load(open(os.path.join(ROOT, "prolific", "v2", "KEY_study1_NEVER_UPLOAD.json")))
DOMF = [("HH", ["Sincerity", "Fairness", "Greed_Avoidance", "Modesty"]), ("E", ["Fearfulness", "Anxiety", "Dependence", "Sentimentality"]),
        ("EX", ["Social_Self_Esteem", "Social_Boldness", "Sociability", "Liveliness"]), ("A", ["Forgiveness", "Gentleness", "Flexibility", "Patience"]),
        ("C", ["Organization", "Diligence", "Perfectionism", "Prudence"]), ("OP", ["Aesthetic_Appreciation", "Inquisitiveness", "Creativity", "Unconventionality"])]
FK = [f"HEXACO_{a}_{f}" for a, fs in DOMF for f in fs]; DK = [f"HEXACO_{a}_mean" for a, _ in DOMF]
sc = lambda c: (lambda s: ast.literal_eval(s) if isinstance(s, str) else s)(K[c]["scores"])
def ccc(x, y):
    x, y = np.array(x, float), np.array(y, float); return float(2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))
def val(a):
    try: v = float(a["raw_values"][0]); return v if 1 <= v <= 5 else None
    except Exception: return None
out = {"facet_readers": [], "domain_readers": []}; PF = ([], []); PD = ([], [])
for r in d["reader_item_rows"]:
    if r["cohort"] in ("detailed_expansion", "original_detailed"):
        c = "C" + r["item_id"][1:]; s = sc(c)
        pr = [(val(r["annotations"].get(f"facet_{i+1:02d}", {"raw_values": [None]})), s[k]) for i, k in enumerate(FK)]; pr = [p for p in pr if p[0] is not None]
        a, b = zip(*pr); PF[0].extend(a); PF[1].extend(b); out["facet_readers"].append([c, r["reader_code"], round(ccc(a, b), 3), len(a)])
    elif r["cohort"] == "six_trait_bridge":
        s = sc(r["item_id"]); pr = [(val(r["annotations"][f"domain_{i+1:02d}"]), s[k]) for i, k in enumerate(DK)]; pr = [p for p in pr if p[0] is not None]
        a, b = zip(*pr); PD[0].extend(a); PD[1].extend(b); out["domain_readers"].append([r["item_id"], r["reader_code"], round(ccc(a, b), 3), len(a)])
out["facet_pooled_ccc"] = round(ccc(*PF), 3); out["facet_median_reader_ccc"] = round(float(np.median([x[2] for x in out["facet_readers"]])), 3)
out["domain_pooled_ccc"] = round(ccc(*PD), 3)
out["profiles_facet"] = sorted({x[0] for x in out["facet_readers"]}); out["profiles_domain"] = sorted({x[0] for x in out["domain_readers"]})
# AI readers on the same people (lcw150 cards)
want = {K[c]["selfPID"]: c for c in out["profiles_facet"]}
IDS = None
src = open(os.path.join(ROOT, "cards", "score_fidelity.py")).read(); g = {"__file__": os.path.join(ROOT, "cards", "score_fidelity.py")}; exec(src.split("def load")[0], g); IDS = g["IDS"]
ai = collections.defaultdict(lambda: ([], []))
for f in glob.glob(os.path.join(ROOT, "cards", "reads_api", "*", "*.json")):
    x = json.load(open(f)); rd = os.path.basename(os.path.dirname(f))
    if x["writer"] != "lcw150" or x["pid"] not in want: continue
    s = sc(want[x["pid"]]); ai[rd][0].extend([x["facets"][i] for i in IDS]); ai[rd][1].extend([s[k] for k in FK])
out["ai_pooled_ccc_same_people"] = {rd: round(ccc(*v), 3) for rd, v in ai.items()}
json.dump(out, open(os.path.join(ROOT, "analysis", "human_card_readers.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if "readers" not in k}))
