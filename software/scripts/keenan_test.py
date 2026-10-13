#!/usr/bin/env python3
"""Keenan, MacWhinney & Mayhew (1977) test on PARSEL: is the remembered stretch of a call richer in 'interactional content'
than a same-length stretch from elsewhere in the same call? Coder (GPT-6-Luna) sees one stretch at a time, blind to which is which.
Codes: humor, teasing, compliment, self_disclosure, disagreement, surprise (all 0/1).
Paired comparison per memory (McNemar-style counts + exact sign test)."""
import json, os, random, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CODER = os.environ.get("CODER", "openai/gpt-6-luna")
OUT = os.path.join(ROOT, "analysis", "keenan" + ("" if CODER == "openai/gpt-6-luna" else "_" + CODER.split("/")[1])); os.makedirs(OUT, exist_ok=True)
PROMPT = """Below is a short stretch of a video call between two strangers (speakers A and B), transcribed automatically.

{seg}

Code what happens in this stretch. Answer 1 if it clearly occurs, else 0.
- humor: someone jokes or says something meant to be funny
- teasing: playful ribbing or banter aimed at the other person
- compliment: someone praises or flatters the other
- self_disclosure: someone shares something personal about their own life, feelings or circumstances
- disagreement: the speakers disagree or one corrects the other
- surprise: someone expresses surprise or something unexpected comes up
Return JSON only: {{"humor":0,"teasing":0,"compliment":0,"self_disclosure":0,"disagreement":0,"surprise":0}}"""


def seg_text(turns, a, b, self_pid):
    lab = lambda t: "A" if t["spk"] == self_pid else "B"
    return "\n".join(f"{lab(t)}: {t['text']}" for t in turns[a:b + 1])


def job(item):
    key, seg = item
    p = os.path.join(OUT, key + ".json")
    if os.path.exists(p):
        return 0.0
    r = orc.call(CODER, [{"role": "user", "content": PROMPT.format(seg=seg)}], os.path.join(OUT, "calls.jsonl"), tag=key,
                 max_tokens=int(os.environ.get("MAXTOK", 1500)), temperature=0, response_format={"type": "json_object"}, reasoning={"effort": "low"} if "luna" in CODER else None)
    t = r.get("text") or ""
    try:
        json.dump(json.loads(t[t.find("{"): t.rfind("}") + 1]), open(p, "w"))
    except Exception:
        pass
    return orc.cost(r)


def main():
    rows = {r["perspectiveID"]: r for r in P.rows()}
    rng = random.Random(1977); items = []
    import glob
    for f in glob.glob(os.path.join(ROOT, "analysis", "located", "PID_*.json")):
        o = json.load(open(f))
        if o.get("kind") != "in_conversation" or not o.get("turns"):
            continue
        r = rows[o["perspectiveID"]]
        turns = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]
        a, b = min(o["turns"]), max(o["turns"]); L = b - a + 1
        cand = [s for s in range(0, len(turns) - L + 1) if s + L - 1 < a or s > b]
        if not cand:
            continue
        s = rng.choice(cand)
        items.append((o["perspectiveID"] + "__mem", seg_text(turns, a, b, r["selfPID"])))
        items.append((o["perspectiveID"] + "__ctl", seg_text(turns, s, s + L - 1, r["selfPID"])))
    rng.shuffle(items)
    if os.environ.get('LIMIT'):
        lim=int(os.environ['LIMIT']); ks=sorted({k.rsplit('__',1)[0] for k,_ in items})[:lim]; items=[x for x in items if x[0].rsplit('__',1)[0] in ks]
    tot = sum(ThreadPoolExecutor(32).map(job, items)); print("coded", len(items), "cost $", round(tot, 3))
    # paired analysis
    from math import comb
    codes = ["humor", "teasing", "compliment", "self_disclosure", "disagreement", "surprise"]
    res = {}
    keys = {k.rsplit("__", 1)[0] for k, _ in items}
    for c in codes:
        b10 = b01 = n = 0; m = ctl = 0
        for k in keys:
            pm, pc = os.path.join(OUT, k + "__mem.json"), os.path.join(OUT, k + "__ctl.json")
            if not (os.path.exists(pm) and os.path.exists(pc)):
                continue
            x, y = int(json.load(open(pm)).get(c, 0)), int(json.load(open(pc)).get(c, 0)); n += 1; m += x; ctl += y
            b10 += x and not y; b01 += y and not x
        k_ = b10 + b01; p = sum(comb(k_, i) for i in range(0, min(b10, b01) + 1)) * 2 / 2 ** k_ if k_ else 1.0
        if not n: continue
        res[c] = dict(n=n, remembered_rate=m / n, control_rate=ctl / n, discordant_mem=b10, discordant_ctl=b01, p_sign=min(1.0, p))
        print(f"{c:16s} remembered {m/n:.3f}  control {ctl/n:.3f}  discordant {b10}:{b01}  p={min(1.0,p):.2g}")
    json.dump(res, open(OUT + "_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
