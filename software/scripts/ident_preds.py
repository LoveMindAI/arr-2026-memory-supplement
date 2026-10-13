#!/usr/bin/env python3
"""D) Whose memory is this, for PREDICTED memories. For each memory i of person p in batch b: score every person q in the batch
(with >=1 other memory) by the mean cosine similarity between text i and q's other texts, excluding any text about the same partner as i
(and i itself). Hit if p ranks first. Same procedure on the real memories (human reference) and on each model x condition
(5 models; none, history, card_narr). Text = moment + reason; embeddings all-mpnet-base-v2 (as style_ceiling.py). analysis/ident_preds.json."""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
M5 = ["deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"]
rows = {r["perspectiveID"]: r for r in P.rows()}
texts = {("human", "human"): {k: (r["conv_SP_Mem_descr"].strip() + " " + r["conv_SP_Mem_reason"].strip()) for k, r in rows.items()}}
for run in ["main_text_cheap", "cards_narr_cheap"]:
    for l in open(os.path.join(ROOT, "runs", run, "preds.jsonl")):
        j = json.loads(l); m = j["model"].split("/")[1]
        if j.get("ok") and m in M5 and j["info"] in ("none", "history", "card_narr") and j["modality"] == "text":
            texts.setdefault((m, j["info"]), {})[j["perspectiveID"]] = str(j["pred"].get("moment", "")) + " " + str(j["pred"].get("reason", ""))
from sentence_transformers import SentenceTransformer
os.environ.setdefault("HF_HUB_OFFLINE", "1")
st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")


def ident(tx):
    keys = [k for k in tx if k in rows]
    E = st.encode([tx[k] for k in keys], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    idx = {k: i for i, k in enumerate(keys)}
    by_b = collections.defaultdict(list)
    for k in keys:
        by_b[rows[k]["batchID"]].append(k)
    hits = []; chance = []
    for k in keys:
        r = rows[k]; me, partner = r["selfPID"], r["otherPID"]
        cand = collections.defaultdict(list)
        for k2 in by_b[r["batchID"]]:
            if k2 == k: continue
            r2 = rows[k2]
            if r2["otherPID"] == partner or r2["selfPID"] == partner and r2["otherPID"] == me:   # drop texts about the same call/partner
                continue
            cand[r2["selfPID"]].append(idx[k2])
        cand = {q: v for q, v in cand.items() if v}
        if me not in cand or len(cand) < 2:
            continue
        sc = {q: float((E[v] @ E[idx[k]]).mean()) for q, v in cand.items()}
        hits.append(max(sc, key=sc.get) == me); chance.append(1 / len(cand))
    return dict(n=len(hits), accuracy=float(np.mean(hits)), chance=float(np.mean(chance)))


res = {}
for key, tx in texts.items():
    res["|".join(key)] = ident(tx)
    print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["|".join(key)].items()}, flush=True)
summ = {info: float(np.mean([res[f"{m}|{info}"]["accuracy"] for m in M5 if f"{m}|{info}" in res])) for info in ("none", "history", "card_narr")}
res["mean_over_models"] = summ; print("mean over models", summ)
json.dump(res, open(os.path.join(ROOT, "analysis", "ident_preds.json"), "w"), indent=1)
