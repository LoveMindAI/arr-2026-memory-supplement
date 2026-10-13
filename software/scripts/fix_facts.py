#!/usr/bin/env python3
"""Second, narrow pass over the fact codes (Oct 7 spot-check found: missed private names; dangling redactions; the participant's own
spouse/partner coded as the conversation partner). Candidates: kept facts with a capitalized word after the first word or with
name/named/called/nickname, plus every fact dropped as 'partner'. One fact per call (GPT-6-Luna). Mechanical check afterwards:
a redaction that leaves an empty fact or a dangling 'named'/'called'/'name is' is dropped as name_only.
Writes facts/coded_v2/fix/<key>.json and facts/FIX_LOG.tsv; assemble_facts.py reads the fixes."""
import json, os, re, sys, glob
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "facts", "coded_v2"); FX = os.path.join(D, "fix"); os.makedirs(FX, exist_ok=True)
PROMPT = """A participant in a study had a short video call with a stranger (their "conversation partner"). This fact about the participant was written down from that call:

FACT: {fact}

1. about_conversation_partner: 1 only if the fact is mainly about the stranger they were talking to in this call. A spouse, boyfriend, girlfriend, romantic partner, family member, friend or colleague of the participant is NOT the conversation partner (answer 0 for those).
2. no_names: the fact with every name of a private individual or pet removed (the participant, family, friends, partners, colleagues, pets: first names, surnames, nicknames), changing as little as possible and keeping it a grammatical sentence ("You have a dog named Marmalade who barks" -> "You have a dog who barks"; "Your brother is named Barnaby and your cousin is named Ptolemy" -> "You have a brother and a cousin"). Names of public figures, places, companies and brands stay. If there is no such name, repeat the fact exactly. If nothing is left once the name is removed, return "".

Return JSON only: {{"about_conversation_partner": 0, "no_names": "..."}}"""
CAP = re.compile(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+"); NAMEW = re.compile(r"\b(name|named|names|called|nickname)\b", re.I)
DANGLE = re.compile(r"\b(named|called|name is|names are|nicknamed)\s*[.,]?\s*$", re.I)


def key(x):
    import hashlib
    return hashlib.md5(f"{x['pid']}|{x['dyad']}|{x['fact']}".encode()).hexdigest()[:16]


def job(x):
    p = os.path.join(FX, key(x) + ".json")
    if os.path.exists(p):
        return 0.0
    r = orc.call("openai/gpt-6-luna", [{"role": "user", "content": PROMPT.format(fact=x["fact"])}], os.path.join(FX, "calls.jsonl"), tag=key(x),
                 max_tokens=1500, temperature=0, response_format={"type": "json_object"}, reasoning={"effort": "low"})
    t = r.get("text") or ""
    try:
        j = json.loads(t[t.find("{"): t.rfind("}") + 1]); json.dump(j, open(p, "w"))
    except Exception:
        pass
    return orc.cost(r)


def main():
    items = [json.load(open(f)) for f in glob.glob(os.path.join(D, "items", "*.json"))]
    cand = [x for x in items if (x.get("drop") == "keep" and (CAP.search(x["fact"]) or NAMEW.search(x["fact"]))) or x.get("drop") == "partner"]
    print("candidates", len(cand), "cost $", round(sum(ThreadPoolExecutor(64).map(job, cand)), 3))
    log = []
    for x in cand:
        p = os.path.join(FX, key(x) + ".json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p)); nn = (j.get("no_names") or "").strip()
        new_drop = "partner" if int(j.get("about_conversation_partner", 0)) else "keep"
        if new_drop == "keep" and (not nn or DANGLE.search(nn)):
            new_drop = "name_only"
        x2 = dict(x, drop=new_drop, fact_no_names=nn if nn else x["fact"], has_name=int(nn != x["fact"]), fixed=True)
        json.dump(x2, open(os.path.join(D, "items", key(x) + ".json"), "w"))
        if new_drop != x.get("drop") or x2["fact_no_names"] != x.get("fact_no_names"):
            log.append((x["pid"], x.get("drop"), new_drop, x["fact"], x2["fact_no_names"]))
    with open(os.path.join(ROOT, "facts", "FIX_LOG.tsv"), "w") as f:
        f.write("pid\tdrop_before\tdrop_after\tfact\tfact_no_names\n")
        for l in log:
            f.write("\t".join(map(str, l)) + "\n")
    print("changed", len(log))


if __name__ == "__main__":
    main()
