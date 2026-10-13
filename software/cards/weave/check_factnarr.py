#!/usr/bin/env python3
"""Who-not-how check (A1's rule: no behavioral claims that can't be read straight off a fact). GPT-6-Luna reads each Opus fact narrative
with its facts and lists sentences that state personality, feelings, motives or how the person treats others beyond what a fact states.
Writes run_weave1/factnarr_check.jsonl and prints the share of narratives with any such sentence."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts")); import orc
OUT = os.path.join(HERE, "run_weave1")
PROMPT = """Below is a short portrait of a person, and the facts it was written from. The portrait was meant to say WHO the person is (their circumstances) and never HOW they are (personality, temperament, feelings, motives, or how they treat other people), except where a fact itself states it.

FACTS:
{facts}

PORTRAIT:
{text}

List every sentence (or clause) of the portrait that describes personality, temperament, feelings, motives or how the person treats others in a way that NO fact states. Do not list sentences that only restate a fact. Return JSON only: {{"inferences": ["...", "..."]}} (an empty list if there are none)."""
facts = {json.loads(l)["pid"]: json.loads(l)["facts"] for l in open(os.path.join(HERE, "weave_in", "factnarr_jobs.jsonl"))}
N = [d for d in map(json.loads, open(os.path.join(OUT, "weave_out", "factnarr.jsonl"))) if d["writer"] == "opus"]


def one(d):
    r = orc.call("openai/gpt-6-luna", [{"role": "user", "content": PROMPT.format(facts="\n".join(f"- {x}" for x in facts[d["pid"]]), text=d["text"])}],
                 os.path.join(OUT, "factnarr_check_calls.jsonl"), tag=d["pid"], max_tokens=3000, temperature=0, response_format={"type": "json_object"},
                 reasoning={"effort": "low"})
    t = r.get("text") or ""
    try:
        inf = json.loads(t[t.find("{"): t.rfind("}") + 1])["inferences"]
    except Exception:
        inf = None
    return dict(pid=d["pid"], words=len(d["text"].split()), inferences=inf), orc.cost(r)


res = list(ThreadPoolExecutor(24).map(one, N))
with open(os.path.join(OUT, "factnarr_check.jsonl"), "w") as f:
    for d, _ in res:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
ok = [d for d, _ in res if d["inferences"] is not None]
print("checked", len(ok), "of", len(res), "| with any inference", sum(bool(d["inferences"]) for d in ok), "| total flagged sentences",
      sum(len(d["inferences"]) for d in ok), "| cost $", round(sum(c for _, c in res), 3))
for d in [d for d in ok if d["inferences"]][:8]:
    print(" ", d["pid"], "|", d["inferences"][0][:140])
