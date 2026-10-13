#!/usr/bin/env python3
"""Blind, style-stripping coding of each PARSEL memory report (exploratory; scheme is agent1's draft for A1 to revise).
The coder sees ONLY the memory text and the stated reason. No IDs, no partner, no ratings.
Output per memory: neutral paraphrase in a fixed template + categorical codes. Used to test whether rememberer-specific
similarity survives once writing style is removed."""
import csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc

BASE = os.path.expanduser("<LOCAL_FILE> Story/Conv Data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "coded")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "calls.jsonl")
MODEL = os.environ.get("CODER", "google/gemini-3.8-flash")

PROMPT = """A participant in a research study had a 3-minute video call with a stranger. Afterwards they were asked to describe the moment of the conversation they remembered most, and why. Here is what they wrote.

MEMORY: {mem}
REASON: {why}

Code this report. Ignore spelling, grammar and writing style entirely; code only what is remembered.

Return JSON only:
{{
 "neutral": "one sentence in this exact form: 'A moment when <who> <did/said what>, remembered because <reason>.' Use 'the rememberer', 'the partner' or 'both' for people. Plain words, no quotes from the original, at most 30 words.",
 "focus": "partner | self | both | task_or_study | setting_or_technical | none",
 "kind": "personal_fact (a fact about someone's life: job, home, family, study) | commonality (discovering something shared) | humor | feeling (an emotion the rememberer felt, e.g. nerves, comfort) | impression (appearance, voice, accent, manner, first-impression check) | task_or_study (the study, Prolific, the game, performance) | conversation_process (how the talk went: choosing a topic, ice-breaking, awkwardness, flow) | disagreement_or_negative | other",
 "valence": "positive | neutral | negative",
 "specificity": "specific_event (one identifiable moment) | topic (a subject discussed) | general (the whole conversation or the person overall)",
 "reason_type": "self_relevance (connects to the rememberer's own life/interests) | novelty (surprising, unexpected) | social_connection (bonding, feeling understood) | emotion (it made them feel something) | importance (it was the main point) | other | none_given"
}}"""


def rows():
    return list(csv.DictReader(open(BASE + "/cleaned_curated_conv_data_combined_ffinal.csv", encoding="utf-8-sig")))


def job(item):
    i, r = item
    path = os.path.join(OUT, f"{i:04d}.json")
    if os.path.exists(path):
        return json.load(open(path))
    msgs = [{"role": "user", "content": PROMPT.format(mem=r["conv_SP_Mem_descr"].strip(), why=r["conv_SP_Mem_reason"].strip())}]
    rec = orc.call(MODEL, msgs, LOG, tag=str(i), max_tokens=2000, temperature=0, response_format={"type": "json_object"},
                   reasoning={"effort": "low"})
    t = rec.get("text") or ""
    try:
        code = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception:
        code = None
    out = dict(i=i, ok=bool(code), code=code, cost=orc.cost(rec), model=MODEL)
    if code:
        json.dump(out, open(path, "w"))
    return out


def main():
    R = rows()
    items = list(enumerate(R))
    if len(sys.argv) > 1:
        items = items[: int(sys.argv[1])]
    tot, bad = 0.0, 0
    with ThreadPoolExecutor(16) as ex:
        for o in ex.map(job, items):
            tot += o.get("cost", 0); bad += (not o["ok"])
    print("done", len(items), "failed", bad, "cost $", round(tot, 4))


if __name__ == "__main__":
    main()
