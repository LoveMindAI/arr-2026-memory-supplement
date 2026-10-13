#!/usr/bin/env python3
"""Disclosure depth, scheme v2 (A1, Oct 10: "tighten the scheme and recode"). Written from A1's 435 labels and his notes:
two kinds instead of three ("information vs thoughts/feelings seems more natural"), study/Prolific talk excluded unless it ties to
something bigger, and a written test for DEEP. All examples in the prompt are INVENTED (none from PARSEL), so the same text can go to
Prolific coders later. Calibration caveat: the rule was written from one coder's labels (A1's), so agreement with A1 on the packet is
in-sample; the real test is a second human (and the crowd).
Coders: Claude Haiku 5.5, GPT-6 Luna, Gemini 3 Flash; Scribe transcripts. -> analysis/disclosure_depth_v2/<dyad>__<coder>.json"""
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc
from disclosure_depth import transcript, TX
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "analysis", "disclosure_depth_v2" + ("_twin" if os.environ.get("TWIN") else "")); os.makedirs(OUT, exist_ok=True)
CODERS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3-flash-preview"]
SCHEME = """WHAT COUNTS AS A DISCLOSURE
A moment where a speaker tells the other person something about THEMSELVES and their life.
Not a disclosure:
- pleasantries ("I'm good, thanks", "not too bad");
- talk about the study itself, Prolific, the tasks, their scores, the camera or connection, the weather;
- questions, and remarks about the other person.
Exception: study talk counts if it shows something about their life beyond the study (for example "I do these surveys on my lunch break at the warehouse", "it's been my main income since I got laid off").

KIND
- FACT: a fact about their life (where they live, their job, their family, their history, what they do).
- INNER: their own thoughts, feelings, wishes, opinions, or how they see themselves.

DEPTH
- SURFACE: the ordinary facts and likes you would share with any stranger: where they live, their job or studies, hobbies, a place or food they love, a passing reaction. A bare fact stays SURFACE even when the topic is big ("I moved here from Lisbon last year", "I've got two kids").
- DEEP: it lets the other person see beneath the surface of the speaker's life. Said lightly or as a joke still counts. Deep includes:
  - a vulnerability or struggle: health, disability, money pressure, a habit or vice they admit to, loneliness, not fitting in ("I'm hooked on energy drinks, I know I shouldn't be", "I'm on long-term sick at the moment");
  - close relationships and family history: a partner, children, wanting or losing them, people they've drifted from ("we tried for years before our daughter came along", "I haven't spoken to my brother in ages");
  - roots and life direction, where the meaning or cost shows: leaving home, not belonging, not knowing where life is going, a big decision still open ("I packed up and left everything", "I've not been back home in twenty years", "honestly I don't know what I'm doing after this");
  - how they see themselves or what they deeply value: a self-description or conviction that says who they are ("I'll chat to anyone, me", "I feel like a bit of a fish out of water here", "I'll never work for a boss again").

Count each distinct disclosure once, even if it is repeated or spread over several turns."""
PROMPT = """Below is the transcript of a short video call (about three minutes) between two strangers, A and B, who met in an online research study in the UK to get to know each other. It was transcribed automatically.

{tx}

Code every self-disclosure in the call, using this scheme:

{scheme}

Quote the speaker's own words briefly (at most 12 words) so each disclosure can be found.

Return JSON only:
{{"disclosures": [{{"speaker": "A" or "B", "kind": "FACT" or "INNER", "depth": "SURFACE" or "DEEP", "quote": "..."}}, ...]}}"""

def one(job):
    dyad, coder = job
    p = os.path.join(OUT, f"{dyad}__{coder.split('/')[1]}.json")
    if os.path.exists(p): return 0.0
    tx, who = transcript(dyad)
    r = orc.call(coder, [{"role": "user", "content": PROMPT.format(tx=tx, scheme=SCHEME)}], os.path.join(OUT, "calls.jsonl"), tag=f"{dyad}|{coder}",
                 max_tokens=8000, temperature=0, response_format={"type": "json_object"})
    try:
        d = json.loads(r.get("text") or "")
        ds = [x for x in d.get("disclosures", []) if x.get("speaker") in ("A", "B") and x.get("kind") in ("FACT", "INNER") and x.get("depth") in ("SURFACE", "DEEP")]
        ok = True
    except Exception:
        ds, ok = [], False
    for x in ds: x["pid"] = who.get(x["speaker"])
    json.dump(dict(dyad=dyad, coder=coder, scheme="v2", ok=ok and r.get("ok", False), speakers=who, disclosures=ds), open(p, "w"), indent=1)
    return orc.cost(r)

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--limit", type=int); a.add_argument("--dyads"); a.add_argument("--workers", type=int, default=24); a = a.parse_args()
    dyads = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(TX, "*.json")))
    if os.environ.get("TWIN"):   # code the twin's tidied calls instead (outputs go to analysis/disclosure_depth_v2_twin)
        dyads = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(ROOT, "twin", "v2", "calls_tidy", "C*.json")) if json.load(open(f)).get("ok"))
    if a.dyads: dyads = [d for d in dyads if d in set(json.load(open(a.dyads)))]
    if a.limit: dyads = dyads[:a.limit]
    cost = sum(ThreadPoolExecutor(a.workers).map(one, [(d, c) for d in dyads for c in CODERS]))
    files = glob.glob(os.path.join(OUT, "*__*.json"))
    print("files", len(files), "ok", sum(json.load(open(f))["ok"] for f in files), "this run $", round(cost, 3))
