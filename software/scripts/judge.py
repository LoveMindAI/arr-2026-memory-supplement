#!/usr/bin/env python3
"""Blind judge: does a predicted memory match the person's real memory? (DRAFT for A1's review)
The judge sees two reports labelled A and B in random order. It never learns which is human, which model, or which condition.
Scores (Feb 23 continuity + additions): topic_match, event_match, similarity (0-1), same_focus, same_reason.
Also scores the partner's own memory of the same call against the rememberer's (human reference) and a random same-batch
memory (floor), so model scores sit on a human scale.
Usage: judge.py --run NAME [--judges anthropic/claude-sonnet-5.5 openai/gpt-6-luna] [--refs]
"""
import argparse, hashlib, json, os, random, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JUDGE = """Two people were each asked to describe the moment they remember most clearly from a short conversation, and why they remember it. Compare the two reports.

REPORT A
Moment: {am}
Why: {ar}

REPORT B
Moment: {bm}
Why: {br}

Judge only what the reports are about, not how well they are written.
- topic_match: 1 if they are about the same topic or subject of conversation, else 0.
- event_match: 1 if they describe the same specific moment or exchange, else 0.
- similarity: from 0.0 (nothing in common) to 1.0 (the same memory), how close the two remembered moments are.
- same_focus: 1 if both centre on the same person (the other person / the rememberer / both / the task or setting), else 0.
- same_reason: 1 if the stated reasons are of the same kind (e.g. both about feeling connected, both about something surprising, both about personal relevance), else 0.

Return JSON only: {{"topic_match": 0, "event_match": 0, "similarity": 0.0, "same_focus": 0, "same_reason": 0}}"""


def jcall(judge, a, b, log, tag):
    rnd = random.Random(int(hashlib.md5(tag.encode()).hexdigest()[:8], 16))
    flip = rnd.random() < 0.5
    x, y = (b, a) if flip else (a, b)
    msg = [{"role": "user", "content": JUDGE.format(am=x[0], ar=x[1], bm=y[0], br=y[1])}]
    r = orc.call(judge, msg, log, tag=tag, max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", 4000)), temperature=0, response_format={"type": "json_object"},
                 reasoning={"effort": "low"})
    t = r.get("text") or ""
    try:
        j = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception:
        j = None
    return j, orc.cost(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judges", nargs="+", default=["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"])  # A1, Oct 7: Luna + DeepSeek; GLM stands in for DeepSeek on DeepSeek's own predictions
    ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args()
    rd = os.path.join(ROOT, "runs", a.run)
    rows = {r["perspectiveID"]: r for r in P.rows()}
    preds = [json.loads(l) for l in open(os.path.join(rd, "preds.jsonl"))]
    preds = [p for p in preds if p.get("ok")]
    items = []
    for p in preds:
        h = rows[p["perspectiveID"]]
        items.append(dict(kind="model", id=f"{p['perspectiveID']}|{p['model']}|{p['modality']}|{p['info']}|{p['frame']}|{p['rep']}",
                          human=(h["conv_SP_Mem_descr"].strip(), h["conv_SP_Mem_reason"].strip()),
                          other=(str(p["pred"].get("moment", "")), str(p["pred"].get("reason", "")))))
    # human references for every perspective that appears
    rng = random.Random(3)
    for pid in sorted({p["perspectiveID"] for p in preds}):
        h = rows[pid]
        partner = rows.get(f"{h['otherPID']}+{h['selfPID']}")
        if partner:
            items.append(dict(kind="ref_partner", id=f"{pid}|REF_partner", human=(h["conv_SP_Mem_descr"].strip(), h["conv_SP_Mem_reason"].strip()),
                              other=(partner["conv_SP_Mem_descr"].strip(), partner["conv_SP_Mem_reason"].strip())))
        pool = [r for r in rows.values() if r["batchID"] == h["batchID"] and r["selfPID"] not in (h["selfPID"], h["otherPID"])
                and r["otherPID"] not in (h["selfPID"], h["otherPID"])]
        if pool:
            o = rng.choice(pool)
            items.append(dict(kind="ref_random", id=f"{pid}|REF_random", human=(h["conv_SP_Mem_descr"].strip(), h["conv_SP_Mem_reason"].strip()),
                              other=(o["conv_SP_Mem_descr"].strip(), o["conv_SP_Mem_reason"].strip())))
    out = os.path.join(rd, "judged.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(x["id"], x["judge"]) for x in map(json.loads, open(out)) if x["scores"]}  # empty-score lines are retried; readers keep the last non-empty line
    jobs = [(it, j) for it in items for j in a.judges if (it["id"], j) not in done]
    print("judgements to make", len(jobs))
    log = os.path.join(rd, "judge_calls.jsonl")
    tot = 0.0
    def actual(it, j):
        # never let a model judge its own predictions: DeepSeek's slot is filled by GLM-5.3-FlashX for DeepSeek predictions
        if j.startswith("deepseek/") and it["kind"] == "model" and it["id"].split("|")[1].startswith("deepseek/"):
            return "z-ai/glm-5.3-flashx"
        return j
    with ThreadPoolExecutor(a.conc) as ex:
        for (it, j), (res, c) in zip(jobs, ex.map(lambda x: jcall(actual(x[0], x[1]), x[0]["human"], x[0]["other"], log, x[0]["id"] + "|" + x[1]), jobs)):
            tot += c
            with open(out, "a") as f:
                f.write(json.dumps(dict(id=it["id"], kind=it["kind"], judge=j, judge_model=actual(it, j), scores=res)) + "\n")
    print("judge cost $", round(tot, 4))


if __name__ == "__main__":
    main()
