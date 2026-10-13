#!/usr/bin/env python3
"""API readers for card fidelity (kit's reader prompt unchanged: personality_models_kit reader_messages('hexaco', text), strict schema).
Recipe follows reader_night frontier_read.py: DeepSeek with reasoning disabled; Luna with reasoning effort none. Greedy (T 0).
Usage: api_read.py <luna|deepseek> <source> ...   source = out_frontier | cards_out/cards_lcw150.jsonl | cards_out/cards_person.jsonl
Writes reads_api/<reader>/<card_id>.json {card_id, pid, writer, valid, facets}. Idempotent."""
import json, os, sys, glob
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "<LOCAL>/Documents/Codex/LAB/research/personality_models_kit"); import prompts as KIT
sys.path.insert(0, os.path.join(HERE, "..", "scripts")); import orc
MODELS = {"luna": "openai/gpt-6-luna", "deepseek": "deepseek/deepseek-v4.1-flash"}
IDS = json.load(open("<LOCAL>/Documents/Codex/LAB/research/personality_models_kit/templates/reader_hexaco.json"))["ids"]


def cards(src):
    p = os.path.join(HERE, src)
    if os.path.isdir(p):
        for f in sorted(glob.glob(os.path.join(p, "*__PID_*.json"))):
            d = json.load(open(f)); w = d["writer"].split("/")[1]
            if d.get("text"):
                yield dict(card_id=f"{w}__{d['pid']}", pid=d["pid"], writer=w, text=d["text"])
    else:
        for l in open(p):
            d = json.loads(l)
            if d.get("text"):
                w = d.get("writer", "lcw150") + ("_person" if "perspectiveID" in d else "")
                cid = f"{w}__{d['perspectiveID']}" if "perspectiveID" in d else f"{w}__{d['pid']}"
                yield dict(card_id=cid, pid=d["pid"], writer=w, text=d["text"])


def one(tag, c, out):
    fp = os.path.join(out, c["card_id"] + ".json")
    if os.path.exists(fp):
        return 0.0
    msgs, rf = KIT.reader_messages("hexaco", c["text"])
    r = orc.call(MODELS[tag], msgs, os.path.join(out, "calls.jsonl"), tag=c["card_id"], max_tokens=2000, temperature=0, response_format=rf,
                 reasoning={"enabled": False} if tag == "deepseek" else {"effort": "none"})
    t = r.get("text") or ""; fac = None; valid = False
    try:
        fac = json.loads(t[t.find("{"): t.rfind("}") + 1]).get("facets", {})
        valid = set(fac) == set(IDS) and all(type(v) in (int, float) and 1 <= v <= 5 for v in fac.values())
    except Exception:
        pass
    if valid:
        json.dump(dict(card_id=c["card_id"], pid=c["pid"], writer=c["writer"], reader=MODELS[tag], valid=True, facets=fac), open(fp, "w"))
    return orc.cost(r)


if __name__ == "__main__":
    tag = sys.argv[1]; out = os.path.join(HERE, "reads_api", tag); os.makedirs(out, exist_ok=True)
    cs = [c for s in sys.argv[2:] for c in cards(s)]
    tot = sum(ThreadPoolExecutor(32).map(lambda c: one(tag, c, out), cs))
    ok = sum(1 for c in cs if os.path.exists(os.path.join(out, c["card_id"] + ".json")))
    print(tag, "cards", len(cs), "valid", ok, "cost $", round(tot, 3))
