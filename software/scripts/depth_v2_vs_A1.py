#!/usr/bin/env python3
"""Scheme v2 coders vs A1's packet labels (in-sample: v2's rule was written from these labels). A1's candidates are matched to each
coder's v2 disclosures for the same speaker by word overlap (>= half of the shorter quote's words); unmatched = the coder says 'not a
disclosure'. Also the old v1 labels on the same candidates, for comparison. -> printed + analysis/depth_v2_vs_A1.json"""
import json, os, re, collections, numpy as np
from sklearn.metrics import cohen_kappa_score
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A = json.load(open(os.path.join(ROOT, "human_coding", "answers", "DEPTH_A1_answers_20261010.json")))["answers"]
K = json.load(open(os.path.join(ROOT, "analysis", "depth_packet_key.json")))
C = ["claude-haiku-5.5", "gpt-6-luna", "gemini-3-flash-preview"]; BL = {"no": 0, "surface": 1, "deep": 2}
LV1 = {"INFO_GENERAL": 1, "THOUGHTS_EVENTS_PLANS": 1, "FEELINGS_MILD": 1, "INFO_REVEALING": 2, "THOUGHTS_INTIMATE": 2, "FEELINGS_DEEP": 2}
W = lambda s: set(re.findall(r"[a-z']+", s.lower()))
rows = []; pc = []
for iid, k in K.items():
    v2 = {}
    for c in C:
        f = os.path.join(ROOT, "analysis", "disclosure_depth_v2", f"{k['dyad']}__{c}.json")
        j = json.load(open(f)) if os.path.exists(f) else None
        v2[c] = [x for x in j["disclosures"] if x["speaker"] == k["speaker"]] if j and j["ok"] else None
    for cid, cand in k["cands"].items():
        b = A[iid]["c"].get(cid, {}).get("level")
        if b is None: continue
        w = W(cand["quote"]); r = dict(why=k["why"], A1=BL[b])
        for c in C:
            r["v1:" + c] = max([LV1[cat] for co, cat in cand["by"] if co == c], default=0)
            if v2[c] is None: r["v2:" + c] = None; continue
            hits = [x for x in v2[c] if (lambda u: u and w and len(u & w) >= 0.5 * min(len(u), len(w)))(W(x["quote"]))]
            r["v2:" + c] = max([2 if x["depth"] == "DEEP" else 1 for x in hits], default=0)
        rows.append(r)
    a1_deep = sum(1 for cid in k["cands"] if A[iid]["c"].get(cid, {}).get("level") == "deep")
    pc.append(dict(iid=iid, why=k["why"], A1=a1_deep, **{c: (sum(x["depth"] == "DEEP" for x in v2[c]) if v2[c] is not None else None) for c in C}))
out = {}
for sub in ("all", "random"):
    R = [r for r in rows if sub == "all" or r["why"] == sub]; out[sub] = {}
    print(f"\n== {sub} (n={len(R)})")
    for ver in ("v1", "v2"):
        for c in C + ["median"]:
            if c == "median":
                RR = [r for r in R if all(r[f"{ver}:{x}"] is not None for x in C)]; b = np.array([r["A1"] for r in RR])
                v = np.array([sorted(r[f"{ver}:{x}"] for x in C)[1] for r in RR])
            else:
                RR = [r for r in R if r[f"{ver}:{c}"] is not None]; b = np.array([r["A1"] for r in RR]); v = np.array([r[f"{ver}:{c}"] for r in RR])
            d = dict(k3=round(cohen_kappa_score(b, v), 3), kw=round(cohen_kappa_score(b, v, weights="quadratic"), 3), kdeep=round(cohen_kappa_score(b == 2, v == 2), 3),
                     kno=round(cohen_kappa_score(b == 0, v == 0), 3), coder_deep=round(float(np.mean(v == 2)), 3), a1_deep=round(float(np.mean(b == 2)), 3), n=len(RR))
            out[sub][f"{ver}:{c}"] = d
            print(f"  {ver} {c:24s} 3-way {d['k3']:+.2f}  weighted {d['kw']:+.2f}  deep-vs-not {d['kdeep']:+.2f}  disclosure-or-not {d['kno']:+.2f}  coder deep {d['coder_deep']:.2f} (A1 {d['a1_deep']:.2f})")
R = [p for p in pc if p["why"] == "random"]
print("\nrandom 10, deep per person-call (v2 counts ALL its disclosures, not only A1's candidates): A1", np.mean([p["A1"] for p in R]),
      {c: np.mean([p[c] for p in R if p[c] is not None]) for c in C})
out["per_person_call"] = pc
json.dump(out, open(os.path.join(ROOT, "analysis", "depth_v2_vs_A1.json"), "w"), indent=1)
