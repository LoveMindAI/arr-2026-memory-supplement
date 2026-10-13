#!/usr/bin/env python3
"""Disclosure depth in the conversation itself (A1, Oct 10: "how many deep revelations"; "I don't think we necessarily coded disclosure
rates ... from conversational analysis"). The Oct 7 analysis (disclosure_analysis.py) counted FACTS per person per call; this codes DEPTH,
with the same Barak & Gluck-Ofri categories the codebook's personal_kinds question uses (information / thoughts / feelings, at a
surface level 2 and a revealing level 3), but over the whole call and for each speaker.
Transcript: Scribe v2 (the more faithful record; transcripts_v3_scribe). Speakers shown as A and B (A = first to speak).
Coders: Claude Haiku 5.5 and GPT-6 Luna (the paper's two coders), temperature 0, JSON.
Usage: disclosure_depth.py [--limit N] -> analysis/disclosure_depth/<dyad>__<coder>.json"""
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TX = os.path.join(ROOT, "transcripts_v3_scribe"); OUT = os.path.join(ROOT, "analysis", "disclosure_depth"); os.makedirs(OUT, exist_ok=True)
CODERS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
CATS = {
    "INFO_GENERAL": "Surface facts about themselves (age, job, family, where they live, interests, hobbies). Level 2, information.",
    "INFO_REVEALING": "Revealing personal information: their personal traits, a personal experience, or problems in their own or their family's behaviour. Level 3, information.",
    "THOUGHTS_EVENTS_PLANS": "Their thoughts about past events or future plans. Level 2, thoughts.",
    "THOUGHTS_INTIMATE": "Their thoughts about their own character, looks or health, or private wishes. Level 3, thoughts.",
    "FEELINGS_MILD": "Surface feelings: mild feelings such as confusion or inconvenience; ordinary worries or frustrations. Level 2, feelings.",
    "FEELINGS_DEEP": "Deep feelings, such as humiliation, anxiety, depression, fears or pain. Level 3, feelings.",
}
PROMPT = """Below is the transcript of a short video call (about three minutes) between two strangers, A and B, who met in an online research study in the UK to get to know each other. It was transcribed automatically.

{tx}

Find every self-disclosure in the call: a moment where a speaker tells the other something about THEMSELVES. Questions, remarks about the other person, and talk about the weather or the study's mechanics are not disclosures, unless the speaker is revealing something about themselves in them (for example "I'm so nervous about these calls" is a disclosure).

Give each disclosure one category:
{cats}

Count each distinct piece of self-disclosure once, even if it is repeated or spread over several turns. Quote the speaker's own words briefly (at most 12 words) so the disclosure can be found.

Return JSON only:
{{"disclosures": [{{"speaker": "A" or "B", "category": "<one code above>", "quote": "..."}}, ...]}}"""

def transcript(dyad):
    if dyad.startswith("C") and dyad[1:].isdigit():          # Oct 10 night: twin calls (twin/v2/calls_tidy), A/B already labelled
        cj = json.load(open(os.path.join(ROOT, "twin", "v2", "calls_tidy", dyad + ".json")))
        return cj["text"], {"A": cj["A"], "B": cj["B"]}
    j = json.load(open(os.path.join(TX, dyad + ".json")))
    order = []
    for t in j["turns"]:
        if t["spk"] not in order: order.append(t["spk"])
    lab = {s: "AB"[i] for i, s in enumerate(order[:2])}
    lines = [f"{lab.get(t['spk'], '?')}: {t['text']}" for t in j["turns"] if not t.get("bc")]
    return "\n".join(lines), {v: k for k, v in lab.items()}

def one(job):
    dyad, coder = job
    p = os.path.join(OUT, f"{dyad}__{coder.split('/')[1]}.json")
    if os.path.exists(p): return 0.0
    tx, who = transcript(dyad)
    msg = PROMPT.format(tx=tx, cats="\n".join(f"- {k}: {v}" for k, v in CATS.items()))
    r = orc.call(coder, [{"role": "user", "content": msg}], os.path.join(OUT, "calls.jsonl"), tag=f"{dyad}|{coder}", max_tokens=8000,
                 temperature=0, response_format={"type": "json_object"})
    try:
        d = json.loads(r.get("text") or ""); ds = [x for x in d.get("disclosures", []) if x.get("speaker") in ("A", "B") and x.get("category") in CATS]
        ok = True
    except Exception:
        ds, ok = [], False
    for x in ds: x["pid"] = who.get(x["speaker"])
    json.dump(dict(dyad=dyad, coder=coder, ok=ok and r.get("ok", False), speakers=who, disclosures=ds), open(p, "w"), indent=1)
    return orc.cost(r)

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--limit", type=int); a.add_argument("--workers", type=int, default=16); a.add_argument("--coders"); a = a.parse_args()
    coders = a.coders.split(",") if a.coders else CODERS   # Oct 10 (A1): third perspective, google/gemini-3-flash-preview
    dyads = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(TX, "*.json")))
    if a.limit: dyads = dyads[:a.limit]
    cost = sum(ThreadPoolExecutor(a.workers).map(one, [(d, c) for d in dyads for c in coders]))
    files = glob.glob(os.path.join(OUT, "*__*.json"))
    print("files", len(files), "ok", sum(json.load(open(f))["ok"] for f in files), "this run $", round(cost, 3))
