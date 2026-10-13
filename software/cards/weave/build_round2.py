#!/usr/bin/env python3
"""Inputs for the one-trip GPU job (A1 gave carte blanche, Oct 7): for ALL people, with both 31B writers (lcw150, plain Gemma 4 31B):
woven cards (weave v3 over existing narrative cards), fact narratives ("who not how" prompt v1), and lcw150 narrative cards from the
models' impressions (mean of 4 readers; same card template as self-report, so only the source of the numbers differs).
Also writes thin impression cards locally (same format as the self-report thin cards)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); CARDS = os.path.join(HERE, ".."); ROOT = os.path.join(CARDS, "..")
sys.path.insert(0, CARDS); import build_card_inputs as B, score_fidelity as SF
facts = json.load(open(os.path.join(ROOT, "facts", "person_facts.json")))
narr = {json.loads(l)["pid"]: json.loads(l)["text"] for l in open(os.path.join(CARDS, "run_cards1", "cards_out", "cards_lcw150.jsonl"))}
traits = {}
for l in open(os.path.join(CARDS, "inputs", "person_jobs.jsonl")):
    j = json.loads(l); traits.setdefault(j["pid"], j["traits"])
IN = os.path.join(HERE, "weave_in")
with open(os.path.join(IN, "weave_all_jobs.jsonl"), "w") as f:
    for p in sorted(narr):
        if p in facts and p in traits and facts[p]:
            f.write(json.dumps(dict(pid=p, traits=traits[p], card=narr[p], facts=facts[p]), ensure_ascii=False) + "\n")
with open(os.path.join(IN, "factnarr_jobs.jsonl"), "w") as f:
    for p in sorted(facts):
        if facts[p]:
            f.write(json.dumps(dict(pid=p, facts=facts[p]), ensure_ascii=False) + "\n")
open(os.path.join(IN, "FACT_NARRATIVE_PROMPT_v1.md"), "w").write(open(os.path.join(HERE, "FACT_NARRATIVE_PROMPT_v1.md")).read())
imp = json.load(open(os.path.join(CARDS, "inputs", "impression_facets.json")))
os.makedirs(os.path.join(CARDS, "inputs", "thin_cards_impression"), exist_ok=True)
n = 0
with open(os.path.join(IN, "impression_card_prompts.jsonl"), "w") as f:
    for p, fac in sorted(imp.items()):
        fs = {SF.MAP[i]: fac[i] for i in SF.IDS}          # 24 facets in kit names; Altruism dropped (not in the variant template)
        f.write(json.dumps(dict(pid=p, messages=B.card_messages(fs))) + "\n")
        open(os.path.join(CARDS, "inputs", "thin_cards_impression", p + ".txt"), "w").write(B.thin_card(fs)); n += 1
for fn in ("weave_all_jobs.jsonl", "factnarr_jobs.jsonl", "impression_card_prompts.jsonl"):
    print(fn, sum(1 for _ in open(os.path.join(IN, fn))))
