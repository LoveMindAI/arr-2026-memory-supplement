#!/usr/bin/env python3
"""Weave bake-off (A1, Oct 7), local side.
  bake.py jobs   -> pick 20 people (spread over fact counts) and write weave_in/weave_jobs.jsonl {pid, traits, card, facts}
  bake.py opus   -> Opus 5.5 weaves the same 20 with the same prompt (T .7) -> run_weave1/weave_out/opus.jsonl
  bake.py eval   -> for every woven card (lcw150, gemma4_31b_base, opus) and the original narrative card:
       change: share of sentences not in the narrative card; spread: share of new sentences by fifth of the card;
       facts: share of the person's facts present (checker: GPT-6-Luna, one call per card);
       fidelity: a fresh Claude Sonnet 5.5 reads the card with the kit reader prompt (no context) -> profile CCC vs true 24 facets.
     writes analysis/weave_bakeoff.json and a side-by-side file for three people (SIDE_BY_SIDE.md)."""
import difflib, json, os, re, sys, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import orc
sys.path.insert(0, "<LOCAL>/Documents/Codex/LAB/research/personality_models_kit"); import prompts as KIT
sys.path.insert(0, os.path.join(ROOT, "cards")); import score_fidelity as SF
IN = os.path.join(HERE, "weave_in"); OUT = os.path.join(HERE, "run_weave1", "weave_out"); os.makedirs(OUT, exist_ok=True)
TPL = open(os.path.join(IN, "WEAVE_PROMPT_v3.md")).read()
sent = lambda t: [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def jobs():
    facts = json.load(open(os.path.join(ROOT, "facts", "person_facts.json")))
    narr = {json.loads(l)["pid"]: json.loads(l)["text"] for l in open(os.path.join(ROOT, "cards", "run_cards1", "cards_out", "cards_lcw150.jsonl"))}
    traits = {}
    for l in open(os.path.join(ROOT, "cards", "inputs", "person_jobs.jsonl")):
        j = json.loads(l); traits.setdefault(j["pid"], j["traits"])
    pids = sorted([p for p in facts if p in narr and p in traits and len(facts[p]) >= 3], key=lambda p: len(facts[p]))
    pick = [pids[int(i)] for i in np.linspace(0, len(pids) - 1, 20)]          # spread from fewest to most facts
    with open(os.path.join(IN, "weave_jobs.jsonl"), "w") as f:
        for p in pick:
            f.write(json.dumps(dict(pid=p, traits=traits[p], card=narr[p], facts=facts[p]), ensure_ascii=False) + "\n")
    print("20 people; facts per person:", [len(facts[p]) for p in pick])


def opus():
    J = [json.loads(l) for l in open(os.path.join(IN, "weave_jobs.jsonl"))]
    def one(j):
        r = orc.call("anthropic/claude-opus-5.5", [{"role": "user", "content": TPL.format(traits=j["traits"], card=j["card"], facts="\n".join(f"- {x}" for x in j["facts"]))}],
                     os.path.join(OUT, "opus_calls.jsonl"), tag=j["pid"], max_tokens=6000, temperature=0.7)
        return dict(pid=j["pid"], writer="opus", text=r.get("text") or "", finish=r.get("finish")), orc.cost(r)
    res = list(ThreadPoolExecutor(10).map(one, J))
    with open(os.path.join(OUT, "opus.jsonl"), "w") as f:
        for d, _ in res:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("opus cost $", round(sum(c for _, c in res), 3))


CHECK = """Here is a portrait of a person, and a list of facts about them. For each fact, say whether the portrait conveys it (1) or not (0). A fact counts as conveyed if a reader of the portrait would learn it, even in different words.

PORTRAIT:
{card}

FACTS:
{facts}

Return JSON only: {{"present": [1, 0, ...]}} with one number per fact, in order."""


def evaluate():
    J = {json.loads(l)["pid"]: json.loads(l) for l in open(os.path.join(IN, "weave_jobs.jsonl"))}
    cards = [dict(pid=p, writer="narrative_original", text=j["card"]) for p, j in J.items()]
    for fn in ("weave_all.jsonl", "opus.jsonl"):
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            cards += [d for d in map(json.loads, open(p)) if d["pid"] in J]
    def one(c):
        j = J[c["pid"]]; a, b = sent(j["card"]), sent(c["text"])
        sm = difflib.SequenceMatcher(None, a, b); same = set()
        for blk in sm.get_matching_blocks():
            same.update(range(blk.b, blk.b + blk.size))
        new = [k for k in range(len(b)) if k not in same]
        fifths = np.histogram([k / max(1, len(b) - 1) for k in new], bins=5, range=(0, 1))[0]
        r = orc.call("openai/gpt-6-luna", [{"role": "user", "content": CHECK.format(card=c["text"], facts="\n".join(f"{i+1}. {x}" for i, x in enumerate(j["facts"])))}],
                     os.path.join(OUT, "check_calls.jsonl"), tag=c["pid"] + c["writer"], max_tokens=3000, temperature=0, response_format={"type": "json_object"},
                     reasoning={"effort": "low"})
        try:
            pres = json.loads((r.get("text") or "{}"))["present"]; fact_share = float(np.mean(pres)) if len(pres) == len(j["facts"]) else None
        except Exception:
            fact_share = None
        msgs, rf = KIT.reader_messages("hexaco", c["text"])
        rr = orc.call("anthropic/claude-sonnet-5.5", msgs, os.path.join(OUT, "read_calls.jsonl"), tag=c["pid"] + c["writer"], max_tokens=2000, temperature=0, response_format=rf)
        try:
            t = rr.get("text") or ""; fac = json.loads(t[t.find("{"): t.rfind("}") + 1])["facets"]
            v = np.array([fac[i] for i in SF.IDS]); tv = np.array([SF.TRUE[c["pid"]][SF.MAP[i]] for i in SF.IDS]); fid = float(SF.ccc(v, tv))
        except Exception:
            fid = None
        return dict(pid=c["pid"], writer=c["writer"], words=len(c["text"].split()), change=len(new) / max(1, len(b)),
                    fifths=(fifths / max(1, fifths.sum())).tolist(), facts_present=fact_share, fidelity_ccc=fid, finish=c.get("finish"))
    rows = list(ThreadPoolExecutor(16).map(one, cards))
    summ = {}
    for w in sorted({r["writer"] for r in rows}):
        R = [r for r in rows if r["writer"] == w]
        g = lambda k: [r[k] for r in R if r[k] is not None]
        summ[w] = dict(n=len(R), words=float(np.median(g("words"))), change=float(np.median(g("change"))),
                       middle_three_fifths=float(np.mean([sum(r["fifths"][1:4]) for r in R])), facts_present=float(np.mean(g("facts_present"))),
                       fidelity_ccc=float(np.mean(g("fidelity_ccc"))))
        print(w, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ[w].items()})
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(ROOT, "analysis", "weave_bakeoff.json"), "w"), indent=1)
    by = {(r["pid"], r["writer"]): c["text"] for r, c in zip(rows, cards)}
    pick = random.Random(3).sample(sorted(J), 3)
    with open(os.path.join(HERE, "SIDE_BY_SIDE.md"), "w") as f:
        for p in pick:
            f.write(f"# {p} ({len(J[p]['facts'])} facts)\n\n## Facts\n" + "\n".join(f"- {x}" for x in J[p]["facts"]) + "\n\n")
            for w in ("narrative_original", "lcw150", "gemma4_31b_base", "opus"):
                if (p, w) in by:
                    f.write(f"## {w}\n\n{by[(p, w)]}\n\n")


if __name__ == "__main__":
    {"jobs": jobs, "opus": opus, "eval": evaluate}[sys.argv[1]]()
