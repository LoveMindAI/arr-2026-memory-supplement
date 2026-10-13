#!/usr/bin/env python3
"""Code the stand-in memories with codebook 3.5.1 exactly as the paper's coders ran (scripts/code_v35.py: same rendered prompt, settings and
parser), with two coders only and no escalation tiers (A1, Oct 11): Claude Haiku 5.5 and GPT-6 Luna. -> coding/<key>__<coder>.json"""
import glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import code_v35 as CV   # the paper's coding job (prompt rendered from codebook/spec_v351_FROZEN.py)
CODERS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
OUT = os.path.join(HERE, "coding"); os.makedirs(OUT, exist_ok=True)
jobs = []
for f in sorted(glob.glob(os.path.join(HERE, "memories", "C*.json"))):
    j = json.load(open(f))
    if j.get("ok"):
        for c in CODERS: jobs.append((OUT, f"{j['call']}_{j['person']}", str(j["answers"]["moment"]).strip(), str(j["answers"]["reason"]).strip(), c))
only = sys.argv[1:] and set(sys.argv[1].split(","))
if only: jobs = [x for x in jobs if x[1].split("_")[0] in only]
cost = sum(ThreadPoolExecutor(int(os.environ.get("WORKERS", 24))).map(CV.job, jobs))
print(len(jobs), "coding jobs; this run $", round(cost, 2), "; codebook sha", CV.CODEBOOK_SHA)
