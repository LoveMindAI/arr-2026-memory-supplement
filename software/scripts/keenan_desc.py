#!/usr/bin/env python3
"""Companion to keenan_test.py: code each MEMORY REPORT (description + reason) for the same six categories, so the stretch comparison
can be repeated on memories whose report does not mention the category (the locator matched stretches to reports, so a report about a
compliment guarantees a compliment in its stretch). Coder GPT-6-Luna. Then re-run the paired test within report-negative memories."""
import json, os, sys, glob
from concurrent.futures import ThreadPoolExecutor
from math import comb
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "analysis", "keenan_desc"); os.makedirs(OUT, exist_ok=True)
CODES = ["humor", "teasing", "compliment", "self_disclosure", "disagreement", "surprise"]
PROMPT = """After a short video call with a stranger, a participant described the moment they remembered most clearly and why.

Moment: {m}
Why: {w}

Does the REPORTED MOMENT involve each of the following? Answer 1 if the report says or clearly implies it, else 0.
- humor: someone joked or said something meant to be funny
- teasing: playful ribbing or banter aimed at the other person
- compliment: someone praised or flattered the other
- self_disclosure: someone shared something personal about their own life, feelings or circumstances
- disagreement: the speakers disagreed or one corrected the other
- surprise: someone expressed surprise or something unexpected came up
Return JSON only: {{"humor":0,"teasing":0,"compliment":0,"self_disclosure":0,"disagreement":0,"surprise":0}}"""


def job(r):
    p = os.path.join(OUT, r["perspectiveID"] + ".json")
    if os.path.exists(p):
        return 0.0
    c = orc.call("openai/gpt-6-luna", [{"role": "user", "content": PROMPT.format(m=r["conv_SP_Mem_descr"].strip(), w=r["conv_SP_Mem_reason"].strip())}],
                 os.path.join(OUT, "calls.jsonl"), tag=r["perspectiveID"], max_tokens=1500, temperature=0, response_format={"type": "json_object"}, reasoning={"effort": "low"})
    t = c.get("text") or ""
    try:
        json.dump(json.loads(t[t.find("{"): t.rfind("}") + 1]), open(p, "w"))
    except Exception:
        pass
    return orc.cost(c)


def main():
    rows = [r for r in P.rows() if os.path.exists(os.path.join(ROOT, "analysis", "keenan", r["perspectiveID"] + "__mem.json"))]
    print("cost $", round(sum(ThreadPoolExecutor(32).map(job, rows)), 3))
    res = {}
    for coder, d in (("luna", "keenan"), ("deepseek", "keenan_deepseek-v4.1-flash")):
        for c in CODES:
            b10 = b01 = n = m = ctl = mentioned = 0
            for r in rows:
                k = r["perspectiveID"]; fd = os.path.join(OUT, k + ".json")
                pm, pc = os.path.join(ROOT, "analysis", d, k + "__mem.json"), os.path.join(ROOT, "analysis", d, k + "__ctl.json")
                if not (os.path.exists(fd) and os.path.exists(pm) and os.path.exists(pc)):
                    continue
                if int(json.load(open(fd)).get(c, 0)):
                    mentioned += 1; continue
                x, y = int(json.load(open(pm)).get(c, 0)), int(json.load(open(pc)).get(c, 0)); n += 1; m += x; ctl += y
                b10 += x and not y; b01 += y and not x
            k_ = b10 + b01; p = min(1.0, sum(comb(k_, i) for i in range(min(b10, b01) + 1)) * 2 / 2 ** k_) if k_ else 1.0
            res[f"{coder}|{c}"] = dict(n_report_negative=n, n_report_mentions=mentioned, remembered_rate=m / n, control_rate=ctl / n, discordant_mem=b10, discordant_ctl=b01, p_sign=p)
            print(f"{coder:8s} {c:16s} report-negative n={n:4d} (mentions {mentioned:3d})  remembered {m/n:.3f}  control {ctl/n:.3f}  {b10}:{b01}  p={p:.2g}")
    json.dump(res, open(os.path.join(ROOT, "analysis", "keenan_report_negative.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
