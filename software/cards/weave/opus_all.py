#!/usr/bin/env python3
"""Bake-off winner (Opus 5.5; Oct 7, carte blanche): woven cards (weave v3) for all 289 and fact narratives ("who not how" v1) for all 292.
Same prompts as the 31B writers; T .7. Appends writer="opus" rows to weave_out/weave_all.jsonl and factnarr.jsonl (idempotent by pid)."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts")); import orc
IN = os.path.join(HERE, "weave_in"); OUT = os.path.join(HERE, "run_weave1", "weave_out")
bullets = lambda xs: "\n".join(f"- {x}" for x in xs)
JOBS = [("weave_all.jsonl", "weave_all_jobs.jsonl", "WEAVE_PROMPT_v3.md", lambda T, j: T.format(traits=j["traits"], card=j["card"], facts=bullets(j["facts"]))),
        ("factnarr.jsonl", "factnarr_jobs.jsonl", "FACT_NARRATIVE_PROMPT_v1.md", lambda T, j: T.format(facts=bullets(j["facts"])))]
for out, jobs, tpl, fill in JOBS:
    T = open(os.path.join(IN, tpl)).read(); J = [json.loads(l) for l in open(os.path.join(IN, jobs))]
    done = {d["pid"] for d in map(json.loads, open(os.path.join(OUT, out))) if d["writer"] == "opus" and d.get("finish") == "stop"}
    todo = [j for j in J if j["pid"] not in done]
    def one(j):
        r = orc.call("anthropic/claude-opus-5.5", [{"role": "user", "content": fill(T, j)}], os.path.join(OUT, "opus_all_calls.jsonl"), tag=out + j["pid"],
                     max_tokens=6000, temperature=0.7)
        return dict(pid=j["pid"], writer="opus", text=r.get("text") or "", finish=r.get("finish")), orc.cost(r)
    res = list(ThreadPoolExecutor(12).map(one, todo))
    with open(os.path.join(OUT, out), "a") as f:
        for d, _ in res:
            if d["text"]:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(out, "todo", len(todo), "ok", sum(1 for d, _ in res if d["finish"] == "stop"), "cost $", round(sum(c for _, c in res), 2), flush=True)
