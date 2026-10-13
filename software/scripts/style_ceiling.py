#!/usr/bin/env python3
"""Is there a person-specific part of what people remember? (human ceiling, no LLM calls)

PARSEL is a round-robin: within a batch, each person talks with several others and reports one memory per conversation.
That gives a social-relations-style decomposition of memory similarity:
  P  same rememberer, different partners          -> rememberer ("perceiver") style
  T  same partner, different rememberers           -> partner ("target") memorability
  R  reciprocal: A about B and B about A           -> the shared conversation
  N  no shared person, same batch                  -> baseline (same study context)
  X  different batch                               -> lower baseline
Also: identification test (whose memory is this? pick among same-batch rememberers by similarity to their OTHER memories)
and ICCs for the numeric memory ratings by rememberer vs by partner.
Exploratory. Embeddings: sentence-transformers/all-mpnet-base-v2 (local).
"""
import csv, json, os, random, sys
from collections import defaultdict
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
BASE = os.path.expanduser("<LOCAL_FILE> Story/Conv Data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(7)


def load():
    rows = list(csv.DictReader(open(BASE + "/cleaned_curated_conv_data_combined_ffinal.csv", encoding="utf-8-sig")))
    out = []
    for r in rows:
        f = lambda k: float(r[k]) if r[k].strip() not in ("", "nan") else np.nan
        out.append(dict(p=r["selfPID"], t=r["otherPID"], b=r["batchID"], mem=r["conv_SP_Mem_descr"].strip(),
                        why=r["conv_SP_Mem_reason"].strip(), timing=f("conv_SP_Mem"), a=f("conv_SP_Mem_a"), d=f("conv_SP_Mem_d"),
                        pl=f("conv_SP_Mem_p")))
    return out


def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    e = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(e)


def pair_type(x, y):
    if x["b"] != y["b"]:
        return "X"
    if x["p"] == y["p"]:
        return "P" if x["t"] != y["t"] else "dup"
    if x["t"] == y["t"]:
        return "T"
    if x["p"] == y["t"] and x["t"] == y["p"]:
        return "R"
    if x["p"] == y["t"] or x["t"] == y["p"]:
        return "L"  # linked: one's rememberer is the other's partner (different conversations)
    return "N"


def decompose(rows, S, label):
    n = len(rows)
    by = defaultdict(list)
    idx_by_batch = defaultdict(list)
    for i, r in enumerate(rows):
        idx_by_batch[r["b"]].append(i)
    for b, ix in idx_by_batch.items():
        for a in range(len(ix)):
            for c in range(a + 1, len(ix)):
                i, j = ix[a], ix[c]
                by[pair_type(rows[i], rows[j])].append((i, j))
    # cross-batch sample
    allix = list(range(n))
    xs = []
    while len(xs) < 20000:
        i, j = rng.sample(allix, 2)
        if rows[i]["b"] != rows[j]["b"]:
            xs.append((i, j))
    by["X"] = xs
    res = {k: dict(n=len(v), mean=float(np.mean([S[i, j] for i, j in v]))) for k, v in by.items() if v}
    # person-clustered bootstrap for P-N, T-N, R-N: resample batches
    batches = list(idx_by_batch)
    pairs_by_batch = defaultdict(lambda: defaultdict(list))
    for k in ("P", "T", "R", "N", "L"):
        for i, j in by.get(k, []):
            pairs_by_batch[rows[i]["b"]][k].append(S[i, j])
    boots = defaultdict(list)
    for _ in range(2000):
        bs = [rng.choice(batches) for _ in batches]
        agg = defaultdict(list)
        for b in bs:
            for k, v in pairs_by_batch[b].items():
                agg[k].extend(v)
        if agg["N"]:
            for k in ("P", "T", "R", "L"):
                if agg[k]:
                    boots[k].append(np.mean(agg[k]) - np.mean(agg["N"]))
    for k, v in boots.items():
        res[k]["minus_N"] = float(res[k]["mean"] - res["N"]["mean"])
        res[k]["ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    print(f"\n[{label}] mean cosine by pair type (n pairs)")
    for k in ("P", "T", "R", "L", "N", "X"):
        if k in res:
            r = res[k]
            extra = f"  diff vs N {r['minus_N']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" if "minus_N" in r else ""
            print(f"  {k}  {r['mean']:.3f}  (n={r['n']}){extra}")
    return res


def identify(rows, S, label, min_mem=3):
    """For each memory m of rememberer A (A has >= min_mem memories): candidates = rememberers in the same batch with
    >= min_mem memories. Score each candidate by mean sim of m to the candidate's memories (excluding m itself and excluding
    any memory about the same partner as m, for every candidate). Accuracy of top-1 = A, vs chance 1/k."""
    by_p = defaultdict(list)
    for i, r in enumerate(rows):
        by_p[r["p"]].append(i)
    hits, chance, mrr = [], [], []
    for i, r in enumerate(rows):
        if len(by_p[r["p"]]) < min_mem:
            continue
        cands = sorted({x["p"] for x in rows if x["b"] == r["b"] and len(by_p[x["p"]]) >= min_mem})
        if len(cands) < 3:
            continue
        sc = {}
        for c in cands:
            js = [j for j in by_p[c] if j != i and rows[j]["t"] != r["t"] and rows[j]["p"] != r["t"]]
            # exclude the partner's own memories too (c == r["t"] would have reciprocal content)
            if c == r["t"] or not js:
                continue
            sc[c] = np.mean([S[i, j] for j in js])
        if r["p"] not in sc or len(sc) < 3:
            continue
        order = sorted(sc, key=lambda c: -sc[c])
        rank = order.index(r["p"]) + 1
        hits.append(rank == 1); chance.append(1 / len(sc)); mrr.append(1 / rank)
    acc, ch = float(np.mean(hits)), float(np.mean(chance))
    # permutation null: shuffle rememberer labels within batch
    print(f"[{label}] identification: n={len(hits)} top-1 {acc:.3f} vs chance {ch:.3f}; MRR {np.mean(mrr):.3f}")
    return dict(n=len(hits), top1=acc, chance=ch, mrr=float(np.mean(mrr)))


def icc_oneway(vals, groups):
    """ICC(1) from one-way ANOVA."""
    g = defaultdict(list)
    for v, k in zip(vals, groups):
        if not np.isnan(v):
            g[k].append(v)
    g = {k: v for k, v in g.items() if len(v) >= 2}
    allv = np.concatenate([np.array(v) for v in g.values()])
    k0 = (len(allv) - sum(len(v) ** 2 for v in g.values()) / len(allv)) / (len(g) - 1)
    gm = allv.mean()
    msb = sum(len(v) * (np.mean(v) - gm) ** 2 for v in g.values()) / (len(g) - 1)
    msw = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in g.values()) / (len(allv) - len(g))
    return float((msb - msw) / (msb + (k0 - 1) * msw)), len(g)


def main():
    rows = load()
    print("memories", len(rows), "rememberers", len({r["p"] for r in rows}))
    out = {}
    for field in ("mem", "why"):
        E = embed([r[field] or "(blank)" for r in rows])
        S = E @ E.T
        out[field] = dict(decomp=decompose(rows, S, field), ident=identify(rows, S, field))
        np.save(os.path.join(OUT, f"emb_{field}.npy"), E)
    print("\nICC(1) of memory ratings: grouped by rememberer vs by partner")
    out["icc"] = {}
    for k, name in (("timing", "timing (0=beginning,100=end; NOT vividness)"), ("a", "arousal"), ("d", "dominance"), ("pl", "pleasure")):
        v = [r[k] for r in rows]
        ip, gp = icc_oneway(v, [r["p"] for r in rows])
        it, gt = icc_oneway(v, [r["t"] for r in rows])
        out["icc"][name] = dict(rememberer=ip, partner=it)
        print(f"  {name:10s} rememberer {ip:+.3f} (k={gp})   partner {it:+.3f} (k={gt})")
    json.dump(out, open(os.path.join(OUT, "style_ceiling.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
