"""Frontier-writer comparison cards (A1: lcw150 vs a frontier model on 50 people; Opus said Oct 7, Sonnet earlier -> both, ~$3).
Same no-Altruism card prompt; kit card sampling (T .7, top-p .95, max 4096)."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import orc
OUT = os.path.join(HERE, "out_frontier"); os.makedirs(OUT, exist_ok=True)
P = {json.loads(l)["pid"]: json.loads(l)["messages"] for l in open(os.path.join(HERE, "inputs", "card_prompts.jsonl"))}
pids = json.load(open(os.path.join(HERE, "inputs", "frontier50.json")))
def job(a):
    model, pid = a; p = os.path.join(OUT, f"{model.split('/')[1]}__{pid}.json")
    if os.path.exists(p): return 0.0
    r = orc.call(model, P[pid], os.path.join(OUT, "calls.jsonl"), tag=f"{model}|{pid}", max_tokens=4096, temperature=0.7)
    if r.get("ok") and r.get("text"):
        json.dump(dict(pid=pid, writer=model, text=r["text"], finish=r.get("finish")), open(p, "w"))
    return orc.cost(r)
jobs = [(m, p) for m in ("anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5") for p in pids]
tot = sum(ThreadPoolExecutor(12).map(job, jobs)); print("cost $", round(tot, 3), "files", len(os.listdir(OUT)) - 1)
