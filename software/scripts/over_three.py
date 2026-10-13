#!/usr/bin/env python3
"""How often a coder chose more than the codebook's three answers (topic, remembered, feelings), after normalization
(correspondence_v351.norm), over every coded file of the two AI coders used in comparisons. -> analysis/over_three.json"""
import glob, json, os, collections
os.environ.pop("COMPLY_MAX3", None)
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
out = {}
for coder in CODERS:
    n = collections.Counter(); c = collections.Counter()
    for f in glob.glob(os.path.join(D, "*", f"*__{coder}.json")):
        j = json.load(open(f))
        if not (j.get("ok") and isinstance(j.get("codes"), dict)): continue
        v = norm(j["codes"])
        for q in ("topic", "remembered", "feelings"):
            if v.get(q) is not None: n[q] += 1; c[q] += len(v[q]) > 3
    out[coder] = {q: [c[q], n[q], round(c[q] / n[q], 4)] for q in n}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(ROOT, "analysis", "over_three.json"), "w"), indent=1)
