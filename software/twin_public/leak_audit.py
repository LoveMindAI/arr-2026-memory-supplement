#!/usr/bin/env python3
"""Leak audit of the public stand-in against PARSEL (an audit, so it reads PARSEL; nothing it reads reaches the stand-in).
1. Distinctive 10-word runs (in at most two real calls) from transcripts_v2 and transcripts_v3_scribe, in any stand-in call, memory or card.
2. Any real memory report or reason of 40+ characters appearing verbatim (normalized) in any stand-in text (PROLIFIC_OVERNIGHT/finalize.py rule).
3. Capitalized words in stand-in text that occur in at most two real transcripts (rare there): listed for a human read; common UK first names
   and towns are expected and fine, anything distinctive is a finding.
-> LEAK_AUDIT.md, leak_audit.json"""
import collections, glob, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import prompts as P
def grams(t, n=10):
    out = set()
    for line in t.splitlines():
        w = re.findall(r"[a-z']+", re.sub(r"^[^:]{1,30}:", "", line).lower()); out |= {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    return out
real = []
for d in ("transcripts_v2", "transcripts_v3_scribe"):
    for f in glob.glob(os.path.join(ROOT, d, "*.json")):
        j = json.load(open(f)); real.append("\n".join(t.get("text", "") for t in j.get("turns", [])))
df = collections.Counter(g for t in real for g in grams(t)); RARE = {g for g, c in df.items() if c <= 2}
cap = collections.Counter(w for t in real for w in set(re.findall(r"\b[A-Z][a-z]{2,}\b", t)))
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
reports = {norm(str(r[k])) for r in P.rows() for k in ("conv_SP_Mem_descr", "conv_SP_Mem_reason")}; reports = {x for x in reports if len(x) >= 40}
texts = {}
for f in glob.glob(os.path.join(HERE, "calls", "C*.json")): texts["call:" + os.path.basename(f)[:-5]] = json.load(open(f))["text"]
for f in glob.glob(os.path.join(HERE, "memories", "C*.json")):
    j = json.load(open(f)); a = j.get("answers") or {}; texts["mem:" + os.path.basename(f)[:-5]] = f"{a.get('moment', '')}\n{a.get('reason', '')}"
for f in glob.glob(os.path.join(HERE, "cards", "P*.json")): texts["card:" + os.path.basename(f)[:-5]] = json.dumps(json.load(open(f)).get("card"), ensure_ascii=False)
runs = {k: sorted(grams(t) & RARE) for k, t in texts.items()}; runs = {k: v for k, v in runs.items() if v}
rep = {k: [x for x in reports if x in norm(t)] for k, t in texts.items()}; rep = {k: v for k, v in rep.items() if v}
rare_caps = collections.Counter()
for t in texts.values():
    for w in set(re.findall(r"\b[A-Z][a-z]{2,}\b", t)):
        if 0 < cap[w] <= 2: rare_caps[w] += 1
res = dict(n_texts=len(texts), runs=runs, reports=rep, rare_capitalized=dict(rare_caps.most_common()))
json.dump(res, open(os.path.join(HERE, "leak_audit.json"), "w"), indent=1)
with open(os.path.join(HERE, "LEAK_AUDIT.md"), "w") as f:
    f.write(f"# Leak audit (leak_audit.py)\n\nTexts checked: {len(texts)} (calls, memories, persona cards).\n\n"
            f"1. Distinctive 10-word runs from real transcripts: **{sum(map(len, runs.values()))}** in {len(runs)} texts.\n"
            f"2. Real reports of 40+ characters: **{sum(map(len, rep.values()))}**.\n"
            f"3. Capitalized words that are rare in the real transcripts (at most two calls) and appear in the stand-in: {len(rare_caps)} distinct; listed in leak_audit.json for a human read.\n")
print(json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in res.items()}))
