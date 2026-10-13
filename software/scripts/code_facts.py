#!/usr/bin/env python3
"""Fact codebook pass (A1, Oct 7). One fact per request (coder can't learn a person). Coder GPT-6-Luna; CODER env for a second coder.
Output facts/coded[_<coder>]/<pid>.json : list of {dyad, fact, kind, drop, trait_adjacent, sensitive, has_name, fact_no_names}."""
import json, os, sys, glob, hashlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CODER = os.environ.get("CODER", "openai/gpt-6-luna")
OUT = os.path.join(ROOT, "facts", "coded_v2" + ("" if "luna" in CODER else "_" + CODER.split("/")[1])); os.makedirs(OUT, exist_ok=True)
PROMPT = """A participant in an online study had short video calls with strangers. From one call, an extractor wrote down facts the participant shared about themselves, in the second person. Code this one fact.

FACT: {fact}

1. kind (pick one):
   home_place (where they live or come from, moving), work_money (job, employer, income, side income including paid online studies), education, family_relationships, pets, interests_activities (hobbies, sport, media, travel), health_body, identity (nationality, age, language, religion, name), plans_daily_life (today, this weekend, routines), opinions_beliefs, study_participation_life (their history of taking part in studies, as part of their life), study_procedure (anything about THIS study session: the task, their score or strategy, the money game, technical problems, the camera or screen, how the session is going), about_partner (mainly about the other person, not the participant), other (fits none of these).
2. drop: "procedure" if the fact is about this study session's task, strategy, technology or setup; "session_feeling" if it is about the participant's feelings or state during this study session; "partner" if it is mainly about the other person; "name_only" if the fact only states the participant's or a private individual's name (keep facts about how someone feels about their name); otherwise "keep".
3. trait_adjacent: 1 if the fact describes the participant's own feelings, temperament, habits or tendencies (e.g. "you get nervous meeting new people", "you are a morning person", "you have social anxiety"), else 0.
4. sensitive: 1 if it concerns health, mental health, religion, sexuality, ethnicity, politics or finances in a sensitive way, else 0.
5. has_name: 1 if the fact contains the personal name of a private individual (the participant, their family, friends, partner, colleagues or pets: first name, last name or nickname), else 0. Names of public figures (authors, athletes, celebrities, politicians), places, companies and brands do not count.
6. fact_no_names: the fact with every private individual's name removed and nothing else changed (e.g. "You have a daughter named Marisol who studies psychology" -> "You have a daughter who studies psychology"). If it has no personal name, repeat the fact exactly.

Return JSON only: {{"kind": "...", "drop": "...", "trait_adjacent": 0, "sensitive": 0, "has_name": 0, "fact_no_names": "..."}}"""


def items():
    for f in sorted(glob.glob(os.path.join(ROOT, "analysis", "facts_per_call", "*.json"))):
        d = json.load(open(f))
        for pid, fs in (d.get("facts") or {}).items():
            for x in fs:
                yield pid, d["dyadID"], x


def job(it):
    pid, dyad, fact = it
    key = hashlib.md5(f"{pid}|{dyad}|{fact}".encode()).hexdigest()[:16]
    p = os.path.join(OUT, "items", key + ".json")
    if os.path.exists(p):
        return 0.0
    r = orc.call(CODER, [{"role": "user", "content": PROMPT.format(fact=fact)}], os.path.join(OUT, "calls.jsonl"), tag=key,
                 max_tokens=int(os.environ.get("MAXTOK", 2000)), temperature=0, response_format={"type": "json_object"},
                 reasoning={"effort": "low"} if "luna" in CODER else None)
    t = r.get("text") or ""
    try:
        j = json.loads(t[t.find("{"): t.rfind("}") + 1]); j.update(pid=pid, dyad=dyad, fact=fact)
        json.dump(j, open(p, "w"))
    except Exception:
        pass
    return orc.cost(r)


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "items"), exist_ok=True)
    its = list(items())
    if os.environ.get("SAMPLE"):
        import random; random.Random(7).shuffle(its); its = its[:int(os.environ["SAMPLE"])]
    print("facts", len(its), "cost $", round(sum(ThreadPoolExecutor(48).map(job, its)), 3))
