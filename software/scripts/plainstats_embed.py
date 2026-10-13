#!/usr/bin/env python3
"""Plain-statistics check, step 2: embed the MOMENT text of every human report and every prediction in texts.json.
Embedders (A1, Oct 11): qwen/qwen3-embedding-8b (with an instruction, applied to every text: symmetric comparison), google/gemini-embedding-2
(both via OpenRouter), Cohere embed-v5.0-pro (input_type clustering), and the local all-mpnet-base-v2 the paper already uses.
Usage: plainstats_embed.py qwen|gemini|cohere|mpnet -> analysis/plainstats/emb_<name>.npz (keys H:<pid>, P<k>), cost in emb_<name>.log"""
import json, os, sys, time, requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = os.path.join(ROOT, "analysis", "plainstats")
T = json.load(open(os.path.join(D, "texts.json"))); name = sys.argv[1]
keys = [f"H:{p}" for p in T["human"]] + [f"P{k}" for k in range(len(T["preds"]))]
texts = [h["moment"] for h in T["human"].values()] + [p["moment"] for p in T["preds"]]
texts = [t if t.strip() else "(empty)" for t in texts]
INSTR = "Instruct: Identify what this memory of a conversation is about: its topic and the event it describes\nQuery: "
UA = {"User-Agent": "curl/8.4.0"}
def openrouter(model, batch, pre=""):
    for a in range(6):
        try:
            r = requests.post("https://openrouter.ai/api/v1/embeddings", headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], **UA},
                              json={"model": model, "input": [pre + t for t in batch]}, timeout=120)
            j = r.json(); d = sorted(j["data"], key=lambda x: x["index"])
            assert len(d) == len(batch)
            return [x["embedding"] for x in d], j.get("usage", {})
        except Exception as e:
            err = str(e)[:200]; time.sleep(2 * (a + 1))
    raise RuntimeError(err)
def cohere(batch):
    for a in range(6):
        try:
            r = requests.post("https://api.cohere.com/v2/embed", headers={"Authorization": "Bearer " + os.environ["COHERE_API_KEY"], **UA},
                              json={"model": "embed-v5.0-pro", "texts": batch, "input_type": "clustering", "embedding_types": ["float"]}, timeout=120)
            j = r.json(); e = j["embeddings"]["float"]; assert len(e) == len(batch)
            return e, j.get("meta", {}).get("billed_units", {})
        except Exception as ex:
            err = str(ex)[:200] + " " + str(r.text)[:200] if 'r' in dir() else str(ex); time.sleep(3 * (a + 1))
    raise RuntimeError(err)
GTASK = "task: identify what a memory of a conversation is about, its topic and the event it describes | query: "
def gdirect(batch):
    k = open(os.path.expanduser("~/.config/LAB/credentials/gemini-direct.key")).read().strip()
    for a in range(8):
        try:
            r = requests.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents", headers={"x-goog-api-key": k, **UA},
                              json={"requests": [{"model": "models/gemini-embedding-2", "content": {"parts": [{"text": GTASK + t}]}} for t in batch]}, timeout=120)
            j = r.json(); e = [x["values"] for x in j["embeddings"]]; assert len(e) == len(batch)
            return e, {"texts": len(batch)}
        except Exception as ex:
            err = str(ex)[:150] + " " + r.text[:300]; time.sleep(4 * (a + 1))
    raise RuntimeError(err)
if name == "mpnet":
    from sentence_transformers import SentenceTransformer
    E = SentenceTransformer("all-mpnet-base-v2").encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False); usage = {}
else:
    B = 96 if name == "cohere" else 100
    batches = [texts[i:i + B] for i in range(0, len(texts), B)]
    f = {"qwen": lambda b: openrouter("qwen/qwen3-embedding-8b", b, INSTR), "gemini": lambda b: openrouter("google/gemini-embedding-2", b),
         "cohere": cohere, "gemini_instr": gdirect}[name]
    with ThreadPoolExecutor(8) as ex: res = list(ex.map(f, batches))
    E = np.array([v for r in res for v in r[0]], np.float32); usage = [r[1] for r in res]
assert len(E) == len(keys)
np.savez_compressed(os.path.join(D, f"emb_{name}.npz"), keys=np.array(keys), E=np.asarray(E, np.float16))
tot = {}
for u in (usage if isinstance(usage, list) else []):
    for k, v in (u or {}).items():
        if isinstance(v, (int, float)): tot[k] = tot.get(k, 0) + v
open(os.path.join(D, f"emb_{name}.log"), "w").write(json.dumps(dict(n=len(keys), dim=int(np.asarray(E).shape[1]), usage=tot)) + "\n")
print(name, len(keys), np.asarray(E).shape, tot)
