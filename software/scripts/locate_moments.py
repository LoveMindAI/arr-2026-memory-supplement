#!/usr/bin/env python3
"""Locate each remembered moment in the transcript (rememberer's seat, numbered turns). Locator model is NOT in the test set.
Output analysis/located/<perspectiveID>.json: {turns: [indices], kind: in_conversation|general|not_in_transcript, confidence}.
Validation built in: the located turns' time position vs the person's own timing slider (0=beginning, 100=end)."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "analysis", os.environ.get("LOCATED_DIR", "located")); os.makedirs(OUT, exist_ok=True)   # Oct 11: second/third locators and prompt v2 write elsewhere
LOG = os.path.join(OUT, "calls.jsonl")
MODEL = os.environ.get("LOCATOR", "anthropic/claude-sonnet-5.5")

PROMPT = """Below is the transcript of a 3-minute video call between two strangers, numbered by turn. "You" is the participant whose memory we are locating; "Partner" is the other person. The transcript was made automatically and may contain errors.

{tx}

Right after the call, the participant wrote this about the moment they remembered most clearly:
MEMORY: {mem}
WHY: {why}

Which turns does this memory refer to? Pick the smallest set of consecutive turns that contains the remembered moment. If the memory is about the conversation as a whole or a general impression, say "general". If it refers to something that is not in the words of the transcript (for example someone's appearance, a noise, a technical problem, or something that happened before the call), say "not_in_transcript" and give the nearest turns if any.

Return JSON only: {{"kind": "in_conversation|general|not_in_transcript", "turns": [first, ..., last], "confidence": "high|medium|low", "note": "one short sentence"}}"""


PROMPT_V2 = """Below is the transcript of a 3-minute video call between two strangers, numbered by turn. "You" is the participant whose memory we are locating; "Partner" is the other person. The transcript was made automatically and may contain errors.

{tx}

Right after the call, the participant wrote this about the moment they remembered most clearly:
MEMORY: {mem}
WHY: {why}

Which turns does this memory refer to?
- If it refers to one moment, give the smallest set of consecutive turns that contains it.
- If it names several separate things said at different points (for example a job, then pets, then where someone is from), give the turns for each of them; they need not be consecutive.
- If it is a summary of the whole conversation or a general impression of the person (for example "we had a lot in common", "a nice chat", "she seemed friendly"), say "general", even if some turns fit loosely.
- If it is about something that is not in the words: how someone looked or sounded (accent, voice, appearance, smoking, background), laughter, a pause or silence, nerves, talking over each other, a technical problem, or something before the call, say "not_in_transcript". Do not substitute the nearest turns.
Use "low" confidence when the words only loosely fit.

Return JSON only: {{"kind": "in_conversation|general|not_in_transcript", "turns": [indices], "confidence": "high|medium|low", "note": "one short sentence"}}"""
if os.environ.get("LOC_PROMPT") == "v2": PROMPT = PROMPT_V2


def numbered(dyad, self_pid):
    j = json.load(open(os.path.join(os.environ.get("LOC_TX") or P.TX, dyad + ".json")))   # Oct 11 (A1): LOC_TX=transcripts_v3_scribe for the vocal-cue re-check
    lines = []
    for i, t in enumerate(j["turns"]):
        who = "You" if t["spk"] == self_pid else "Partner"
        lines.append(f"[{i}] ({int(t['start'] // 60)}:{int(t['start'] % 60):02d}) {who}: {t['text']}")
    return "\n".join(lines), j["turns"]


def job(r):
    path = os.path.join(OUT, r["perspectiveID"] + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    tx, turns = numbered(r["dyadID"], r["selfPID"])
    msg = [{"role": "user", "content": PROMPT.format(tx=tx, mem=r["conv_SP_Mem_descr"].strip(), why=r["conv_SP_Mem_reason"].strip())}]
    c = orc.call(MODEL, msg, LOG, tag=r["perspectiveID"], max_tokens=3000, temperature=0, response_format={"type": "json_object"},
                 reasoning={"effort": "low"})
    t = c.get("text") or ""
    try:
        j = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception:
        j = None
    if not j:
        return dict(perspectiveID=r["perspectiveID"], ok=False, cost=orc.cost(c))
    idx = [i for i in (j.get("turns") or []) if isinstance(i, int) and 0 <= i < len(turns)]
    dur = max(t_["end"] for t_ in turns) if turns else 180
    pos = (100 * ((turns[idx[0]]["start"] + turns[idx[-1]]["end"]) / 2) / dur) if idx else None
    out = dict(perspectiveID=r["perspectiveID"], ok=True, kind=j.get("kind"), turns=idx, confidence=j.get("confidence"), note=j.get("note"),
               located_pos=pos, human_timing=float(r["conv_SP_Mem"]) if r["conv_SP_Mem"].strip() else None, n_turns=len(turns),
               model=MODEL, cost=orc.cost(c))
    json.dump(out, open(path, "w"))
    return out


def main():
    rows = [r for r in P.rows() if r["dyadID"] != "PID_n+PID_n"]
    if len(sys.argv) > 1:
        rows = rows[: int(sys.argv[1])]
    if os.environ.get("PIDS_FILE"):   # Oct 11: re-check a chosen subset (e.g. on Scribe transcripts)
        keep = set(json.load(open(os.environ["PIDS_FILE"]))); rows = [r for r in rows if r["perspectiveID"] in keep]
    tot = 0.0; bad = 0
    with ThreadPoolExecutor(12) as ex:
        for o in ex.map(job, rows):
            tot += o.get("cost") or 0; bad += not o.get("ok")
    print("done", len(rows), "failed", bad, "cost $", round(tot, 3))


if __name__ == "__main__":
    main()
