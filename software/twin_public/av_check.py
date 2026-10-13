#!/usr/bin/env python3
"""Checks for the audio/video-only subset (A1, Oct 11). For every planned call with an event:
 (a) keyword scan: content words of the event description (minus stop words) that appear in the transcript, and
     a cheap judge (Gemini 3 Flash preview) asked whether the transcript mentions or hints at the event;
 (b) the paper's text-only locator (scripts/locate_moments.py PROMPT, v1; Claude Sonnet 5.5, temperature 0, low reasoning) on the noticer's memory.
     Stand-in transcripts have no timestamps, so turns are numbered without times.
-> av_events.json (every event), av_check.json, and a printed summary. Leaks are flagged for rewrite or exclusion from the subset."""
import glob, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "scripts")); import orc
LOC = open(os.path.join(HERE, "..", "scripts", "locate_moments.py")).read(); PROMPT = re.search(r'PROMPT = """(.*?)"""', LOC, re.S).group(1)   # v1 verbatim
JUDGE = "google/gemini-3-flash-preview"; LOCATOR = "anthropic/claude-sonnet-5.5"
STOP = set("the a an and or but to of in on at for with is was it its their they them his her he she you your this that from as by are be into over off out up down one".split())
os.makedirs(os.path.join(HERE, "av"), exist_ok=True); LOG = os.path.join(HERE, "av", "calls.jsonl")
def jp(t):
    t = t or ""; s, e = t.find("{"), t.rfind("}")
    try: return json.loads(t[s:e + 1])
    except Exception: return None
def check(f):
    m = json.load(open(f))
    if not m.get("requires_av") or not m.get("ok"): return None
    cj = json.load(open(os.path.join(HERE, "calls", m["call"] + ".json"))); ev = m["av_event"]; lines = cj["text"].splitlines()
    names = {n.lower() for n in cj["names"].values()}
    words = {w for w in re.findall(r"[a-z]+", str(ev.get("what", "")).lower()) if len(w) > 3 and w not in STOP and w not in names}
    hits = sorted(w for w in words if re.search(r"\b" + w + r"\b", cj["text"].lower()))
    j = orc.call(JUDGE, [{"role": "user", "content": f"Here is a call transcript.\n\n{cj['text']}\n\nDuring the call, this happened on video and was supposed to stay out of the words: {ev.get('what')}\n"
                          'Does the transcript mention, react to, or hint at it in any way (including stage directions)? Return JSON only: {"reveals": true or false, "evidence": "quote or empty"}'}],
                 LOG, tag="judge|" + m["call"], max_tokens=2000, temperature=0, response_format={"type": "json_object"})
    jd = jp(j.get("text")) or {}
    s = m["seat"]; tx = "\n".join(f"[{i}] " + ("You: " if l.startswith(s + ":") else "Partner: ") + l[3:] for i, l in enumerate(lines))
    a = m["answers"]
    r = orc.call(LOCATOR, [{"role": "user", "content": PROMPT.format(tx=tx, mem=str(a["moment"]).strip(), why=str(a["reason"]).strip())}], LOG, tag="locate|" + m["call"],
                 max_tokens=3000, temperature=0, response_format={"type": "json_object"}, reasoning={"effort": "low"})
    lo = jp(r.get("text")) or {}
    return dict(call=m["call"], noticer=m["person"], event=ev, memory=a, keyword_hits=hits, judge=jd, locator=dict(kind=lo.get("kind"), turns=lo.get("turns"), confidence=lo.get("confidence"), note=lo.get("note")),
                leak=bool(jd.get("reveals")), placed=(lo.get("kind") == "in_conversation"), cost=orc.cost(j) + orc.cost(r))
if __name__ == "__main__":
    ev = {}
    for f in sorted(glob.glob(os.path.join(HERE, "calls", "C*.json"))):
        c = json.load(open(f))
        if c.get("av_event"): ev[c["id"]] = c["av_event"]
    json.dump(ev, open(os.path.join(HERE, "av_events.json"), "w"), indent=1)
    res = [x for x in ThreadPoolExecutor(12).map(check, sorted(glob.glob(os.path.join(HERE, "memories", "C*.json")))) if x]
    json.dump(res, open(os.path.join(HERE, "av_check.json"), "w"), indent=1)
    n = len(res); kinds = {k: sum(x["locator"]["kind"] == k for x in res) for k in ("not_in_transcript", "general", "in_conversation")}
    print(f"{len(ev)} events; {n} AV memories; judge says transcript reveals: {sum(x['leak'] for x in res)}; keyword hits in {sum(bool(x['keyword_hits']) for x in res)}; locator kinds {kinds}; cost ${sum(x['cost'] for x in res):.2f}")
