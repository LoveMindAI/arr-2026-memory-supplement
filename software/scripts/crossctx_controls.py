#!/usr/bin/env python3
"""A2's point (Oct 9, A1: do it without her): the cross-context author identification (a collaboration-call memory -> its author,
from that person's social-call memories; paper: 40% vs 23% chance on raw text) needs style and identity-leakage controls.
Same identification rule as style_cross_context.py, on:
  raw text (replication) | neutral one-sentence paraphrases (same prompt and model as analysis/coded/, code_memories.py) |
  v3.5.1 codebook profiles, Haiku and Luna separately, WITH topic and WITHOUT topic (no topic = no content facts like 'I'm a nurse'
  that could identify a person without any memory tendency).
Generation (cached, resumable): analysis/coded_coll/ (paraphrases), analysis/code_v351/coll/ (codebook). -> analysis/crossctx_controls.json"""
import csv, glob, json, os, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.argv = sys.argv[:1] + ["--set", "x"] if len(sys.argv) == 1 else sys.argv
import orc, code_memories as CM, style_ceiling as sc
import code_v35 as C35
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)   # norm(), load(), Q, CODERS
LS = os.path.expanduser("<LOCAL_FILE> Story")
coll = [dict(p=r["selfPID"], t=r["otherPID"], b=r["batchID"], mem=r["coll_SP_Mem_descr"].strip(), why=r["coll_SP_Mem_reason"].strip())
        for r in csv.DictReader(open(LS + "/cleaned_curated_imp_coll_coord_data_combined_ffinal.csv", encoding="utf-8-sig"))]
conv = sc.load()
PC = os.path.join(ROOT, "analysis", "coded_coll"); os.makedirs(PC, exist_ok=True)
def para(i):
    p = os.path.join(PC, f"{i:04d}.json")
    if os.path.exists(p): return 0.0
    r = coll[i]; msgs = [{"role": "user", "content": CM.PROMPT.format(mem=r["mem"] or "(blank)", why=r["why"] or "(blank)")}]
    rec = orc.call(CM.MODEL, msgs, os.path.join(PC, "calls.jsonl"), tag=str(i), max_tokens=2000, temperature=0, response_format={"type": "json_object"}, reasoning={"effort": "low"})
    t = rec.get("text") or ""
    try: code = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception: code = None
    if code: json.dump(dict(i=i, ok=True, code=code, model=CM.MODEL), open(p, "w"))
    return orc.cost(rec)
OC = os.path.join(ROOT, "analysis", "code_v351", "coll"); os.makedirs(OC, exist_ok=True)
ckey = lambda r: f"COLL|{r['p']}+{r['t']}"
if "--generate" in sys.argv:
    with ThreadPoolExecutor(32) as ex: c1 = sum(ex.map(para, range(len(coll))))
    with ThreadPoolExecutor(8) as ex: c1 += sum(ex.map(para, range(len(coll))))
    jobs = [(OC, ckey(r), r["mem"] or "(blank)", r["why"] or "(blank)", c) for r in coll for c in C35.CODERS if c.split("/")[1] in CODERS]
    with ThreadPoolExecutor(64) as ex: c2 = sum(ex.map(C35.job, jobs))
    with ThreadPoolExecutor(8) as ex: c2 += sum(ex.map(C35.job, jobs))
    print("paraphrase $", round(c1, 3), "codebook $", round(c2, 3)); sys.exit()

def ident(sim, label):
    """sim(i_conv, j_coll) -> similarity. Rule copied from style_cross_context.py."""
    conv_by_p = defaultdict(list)
    for i, x in enumerate(conv): conv_by_p[x["p"]].append(i)
    hits, ch, who = [], [], []
    for j, y in enumerate(coll):
        cands = {x["p"] for x in conv if x["b"] == y["b"]} - {y["t"]}; sc_ = {}
        for c in cands:
            ii = [i for i in conv_by_p[c] if conv[i]["t"] != y["t"] and conv[i]["p"] != y["t"]]
            v = [sim(i, j) for i in ii]; v = [x for x in v if x is not None]
            if len(v) >= 2: sc_[c] = float(np.mean(v))
        if y["p"] not in sc_ or len(sc_) < 3: continue
        best = max(sc_, key=sc_.get); hits.append(best == y["p"]); ch.append(1 / len(sc_)); who.append(y["p"])
    d = np.array(hits, float) - np.array(ch); g = defaultdict(list)
    for v, w in zip(d, who): g[w].append(v)
    G = list(g.values()); rng = np.random.default_rng(3); bs = []
    for _ in range(2000):
        k = rng.integers(0, len(G), len(G)); bs.append(np.mean([x for i in k for x in G[i]]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{label:44s} n={len(hits)} top-1 {np.mean(hits):.3f} vs chance {np.mean(ch):.3f}  above chance {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] people={len(G)}")
    return dict(n=len(hits), people=len(G), top1=round(float(np.mean(hits)), 3), chance=round(float(np.mean(ch)), 3), above=round(float(d.mean()), 3), lo=round(float(lo), 3), hi=round(float(hi), 3))

out = {}
E = sc.embed([r["mem"] or "(blank)" for r in conv + coll]); S = E @ E.T; nc = len(conv)
out["raw_text"] = ident(lambda i, j: S[i, nc + j], "raw text (replication)")
Pv = {json.load(open(f))["i"]: json.load(open(f))["code"].get("neutral") for f in glob.glob(os.path.join(ROOT, "analysis", "coded", "*.json"))}
Pc = {json.load(open(f))["i"]: json.load(open(f))["code"].get("neutral") for f in glob.glob(os.path.join(PC, "*.json"))}
ci = [i for i in range(len(conv)) if Pv.get(i)]; cj = [j for j in range(len(coll)) if Pc.get(j)]
E2 = sc.embed([Pv[i] for i in ci] + [Pc[j] for j in cj]); ix = {i: k for k, i in enumerate(ci)}; jx = {j: len(ci) + k for k, j in enumerate(cj)}
S2 = E2 @ E2.T
out["neutral_paraphrase"] = ident(lambda i, j: S2[ix[i], jx[j]] if i in ix and j in jx else None, "neutral paraphrase (style stripped)")
for coder in CODERS:
    Hc = load("human", coder); Cc = load("coll", coder)
    for with_topic in (True, False):
        keys = [k for k in KEYS if with_topic or k != "topic"]
        def vec(n):
            if n is None: return None
            v = {}
            for k in keys:
                x = n.get(k)
                if x is None: continue
                for a in (x if isinstance(x, frozenset) else [x]): v[(k, str(a))] = 1.0
            return v
        V1 = [vec(Hc.get(f"{x['p']}+{x['t']}")) for x in conv]; V2 = [vec(Cc.get(ckey(y))) for y in coll]
        def cos(a, b):
            if not a or not b: return None
            num = sum(1 for k in a if k in b); return num / (len(a) ** .5 * len(b) ** .5)
        tag = f"codebook profile, {coder}, {'with' if with_topic else 'WITHOUT'} topic"
        out[tag] = ident(lambda i, j: cos(V1[i], V2[j]), tag)
json.dump(out, open(os.path.join(ROOT, "analysis", "crossctx_controls.json"), "w"), indent=1)
