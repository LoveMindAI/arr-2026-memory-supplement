#!/usr/bin/env python3
"""Does memory style carry across conversation types? PARSEL: every Feud (task) pair also had a get-to-know (conv) call.
Cross-context pairs (one conv memory, one coll memory), same batch:
  Px  same rememberer, different partners      -> style across contexts
  PSx same rememberer, same partner            -> relationship across contexts
  Tx  different rememberers, same partner
  Rx  reciprocal (A on B in one context, B on A in the other)
  Nx  no shared person
Identification: for a coll memory of A, rank batch rememberers by mean similarity of their CONV memories (excluding conv memories
about A's Feud partner); top-1 vs chance. Raw text embeddings (all-mpnet-base-v2); codebook-free."""
import csv, os, random, json
from collections import defaultdict
import numpy as np
import style_ceiling as sc

LS = os.path.expanduser("<LOCAL_FILE> Story")
rng = random.Random(5)

def load_coll():
    out = []
    for r in csv.DictReader(open(LS + "/cleaned_curated_imp_coll_coord_data_combined_ffinal.csv", encoding="utf-8-sig")):
        out.append(dict(p=r["selfPID"], t=r["otherPID"], b=r["batchID"], mem=r["coll_SP_Mem_descr"].strip(), why=r["coll_SP_Mem_reason"].strip(), ctx="coll"))
    return out

def main():
    conv = sc.load()
    for r in conv: r["ctx"] = "conv"
    coll = load_coll()
    allr = conv + coll
    res = {}
    for field in ("mem", "why"):
        E = sc.embed([r[field] or "(blank)" for r in allr]); S = E @ E.T
        nc = len(conv)
        types = defaultdict(list)
        for j in range(nc, len(allr)):
            y = allr[j]
            for i in range(nc):
                x = allr[i]
                if x["b"] != y["b"]: continue
                if x["p"] == y["p"]:
                    k = "PSx" if x["t"] == y["t"] else "Px"
                elif x["t"] == y["t"]:
                    k = "Tx"
                elif x["p"] == y["t"] and x["t"] == y["p"]:
                    k = "Rx"
                elif x["p"] == y["t"] or x["t"] == y["p"]:
                    k = "Lx"
                else:
                    k = "Nx"
                types[k].append(S[i, j])
        N = np.mean(types["Nx"])
        print(f"\n[{field}] cross-context mean cosine (conv memory vs Feud memory)")
        out = {}
        for k in ("Px", "PSx", "Tx", "Rx", "Lx", "Nx"):
            v = types[k]; out[k] = dict(n=len(v), mean=float(np.mean(v)), minus_N=float(np.mean(v) - N))
            print(f"  {k:4s} {np.mean(v):.3f}  (n={len(v)})  diff vs Nx {np.mean(v)-N:+.3f}")
        # identification
        conv_by_p = defaultdict(list)
        for i in range(nc): conv_by_p[allr[i]["p"]].append(i)
        hits, ch = [], []
        for j in range(nc, len(allr)):
            y = allr[j]
            cands = {allr[i]["p"] for i in range(nc) if allr[i]["b"] == y["b"]} - {y["t"]}
            sc_ = {}
            for c in cands:
                ii = [i for i in conv_by_p[c] if allr[i]["t"] != y["t"] and allr[i]["p"] != y["t"]]
                if len(ii) >= 2: sc_[c] = np.mean([S[i, j] for i in ii])
            if y["p"] not in sc_ or len(sc_) < 3: continue
            best = max(sc_, key=sc_.get); hits.append(best == y["p"]); ch.append(1 / len(sc_))
        print(f"[{field}] identify Feud-memory author from their CONV memories: n={len(hits)} top-1 {np.mean(hits):.3f} vs chance {np.mean(ch):.3f}")
        out["ident"] = dict(n=len(hits), top1=float(np.mean(hits)), chance=float(np.mean(ch)))
        res[field] = out
    json.dump(res, open(os.path.join(sc.OUT, "style_cross_context.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
