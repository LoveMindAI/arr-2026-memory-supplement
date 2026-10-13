#!/usr/bin/env python3
"""What kind of memory does each condition make a model predict? Codebook (same prompt, GPT-6-Luna) on predicted memories
(5 models x 300-memory sample x 10 conditions) vs people's own memories (Luna codes, same 300 memories).
Per condition: share 'specific' (one exchange vs a whole topic), share 'integrative' meaning, and the share of memories citing each top-level
reason (self_perception, fact_about_others, shared_experience, meta_behaviour_of_other, ...). Also, per memory, whether the prediction's
structure and top-level reasons MATCH the person's own (agreement with the real memory, at the level of kind). analysis/coded_preds.json."""
import collections, glob, json, os
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
pro = set(json.load(open(os.path.join(ROOT, "runs", "MAIN_SETS.json")))["pro300"])
top = lambda rs: {r.split(".")[0] for r in (rs or []) if r and r != "none"}
H = {}
for f in glob.glob(os.path.join(ROOT, "analysis", "code_v1", "*__gpt-6-luna.json")):
    d = json.load(open(f))
    if d["key"] in pro:
        H[d["key"]] = d["code"]
Pc = collections.defaultdict(dict)
for f in glob.glob(os.path.join(ROOT, "analysis", "code_preds", "*.json")):
    d = json.load(open(f)); pid, m, info = d["key"].split("|"); Pc[info][(pid, m)] = d["code"]
REASONS = sorted({r for c in list(H.values()) + [c for v in Pc.values() for c in v.values()] for r in top(c.get("reasons"))})


def profile(codes):
    n = len(codes)
    out = dict(n=n, specific=float(np.mean([c.get("structure") == "specific" for c in codes])),
               integrative=float(np.mean([c.get("meaning") == "integrative" for c in codes])))
    for r in REASONS:
        out["reason:" + r] = float(np.mean([r in top(c.get("reasons")) for c in codes]))
    return out


res = {"human": profile(list(H.values()))}
for info in ["none", "history", "card_thin", "card_narr", "card_narr_history", "card_facts", "card_factnarr", "card_weave", "card_impthin", "card_impnarr"]:
    if info not in Pc:
        continue
    res[info] = profile(list(Pc[info].values()))
    # agreement with the person's own memory, kind-level
    st = [Pc[info][k].get("structure") == H[k[0]].get("structure") for k in Pc[info] if k[0] in H]
    jac = [len(top(Pc[info][k].get("reasons")) & top(H[k[0]].get("reasons"))) / max(1, len(top(Pc[info][k].get("reasons")) | top(H[k[0]].get("reasons"))))
           for k in Pc[info] if k[0] in H]
    res[info].update(structure_match=float(np.mean(st)), reason_jaccard=float(np.mean(jac)))
cols = ["n", "specific", "integrative", "structure_match", "reason_jaccard"] + ["reason:" + r for r in REASONS]
print("condition".ljust(20) + " ".join(c.replace("reason:", "")[:11].rjust(11) for c in cols))
for k, v in res.items():
    print(k.ljust(20) + " ".join((f"{v[c]:.2f}" if isinstance(v.get(c), float) else str(v.get(c, "--"))).rjust(11) for c in cols))
json.dump(res, open(os.path.join(ROOT, "analysis", "coded_preds.json"), "w"), indent=1)
