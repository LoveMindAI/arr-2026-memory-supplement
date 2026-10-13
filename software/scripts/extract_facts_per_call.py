#!/usr/bin/env python3
"""Per-CALL personal-fact extraction, so person-specific cards can exclude the call being predicted (leakage fix).
Prompts are A2's VERBATIM (ClaudeFirstImpressions/First Impressions/A2 Code/PARSEL_first_impressions-main/llm_facts_from_transcripts.py),
same model (google/gemini-3-flash-preview). Only change: one dyad transcript per request instead of a whole batch; transcripts are our v2
(speaker-labelled by PID). Output: analysis/facts_per_call/<dyadID>.json = {PID: [facts]}."""
import ast, json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "analysis", "facts_per_call"); os.makedirs(OUT, exist_ok=True)
SRC = os.path.expanduser("<LOCAL_FILE> Impressions/A2 Code/PARSEL_first_impressions-main/llm_facts_from_transcripts.py")
MODEL = "google/gemini-3-flash-preview"


def A2_prompts():
    tree = ast.parse(open(SRC).read()); d = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE"):
            d[n.targets[0].id] = n.value.value
    return d["SYSTEM_PROMPT"], d["USER_PROMPT_TEMPLATE"]


SYS, USER = A2_prompts()


def job(dyad):
    p = os.path.join(OUT, dyad + ".json")
    if os.path.exists(p):
        return 0.0, True
    j = json.load(open(os.path.join(P.TX, dyad + ".json")))
    tx = "\n".join(f"[{int(t['start']//60)}:{int(t['start']%60):02d}] {t['spk']}: {t['text']}" for t in j["turns"])
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": USER.format(transcripts=f"=== {dyad} ===\n{tx}")}]
    r = orc.call(MODEL, msgs, os.path.join(OUT, "calls.jsonl"), tag=dyad, max_tokens=4000, temperature=0, response_format={"type": "json_object"})
    t = r.get("text") or ""
    try:
        d = json.loads(t[t.find("{"): t.rfind("}") + 1]); assert isinstance(d, dict)
    except Exception:
        return orc.cost(r), False
    json.dump(dict(dyadID=dyad, model=MODEL, facts=d), open(p, "w"))
    return orc.cost(r), True


def main():
    dyads = sorted({r["dyadID"] for r in P.rows()} - {"PID_n+PID_n"})
    tot = 0.0; bad = 0
    with ThreadPoolExecutor(12) as ex:
        for c, ok in ex.map(job, dyads):
            tot += c; bad += not ok
    print("dyads", len(dyads), "failed", bad, "cost $", round(tot, 3))


if __name__ == "__main__":
    main()
