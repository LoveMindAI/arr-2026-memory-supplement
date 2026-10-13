#!/usr/bin/env python3
"""Codebook v3.5 (FROZEN, A1 Oct 8) model coding. The prompt is rendered from codebook/spec_v35_FROZEN.py by codebook_render.py, so models
see the same words as the human coding page. Reasoning ON (A1 agreed Oct 8). Coders: Claude Haiku 5.5, GPT-6-Luna, DeepSeek v4.1 Flash;
escalation tiers as in v2/v3 (A1, Oct 8). Usage: code_v35.py --set pilot40 [--adjudicate]  -> analysis/code_v35/<set>/"""
import argparse, hashlib, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P, codebook_render as CR
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_frozen = open(os.path.join(ROOT, "codebook", "spec_v351_FROZEN.py")).read(); _live = open(os.path.join(ROOT, "codebook", "spec_v351.py")).read()
assert _frozen == _live, "spec_v351.py differs from the frozen copy"   # v3.5.1 (A1, Oct 8); renderer imports spec_v351
INSTR = CR.prompt()
CODEBOOK_SHA = hashlib.sha256(INSTR.encode()).hexdigest()[:12]
CODERS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"]
TIER2 = ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5"]   # A1, Oct 8: strong disagreements go to larger models (call rarely)
TIER3 = ["anthropic/claude-opus-5.5"]                             # ... and "if something is truly bonkers, pass it to opus 5.5"
CORE_S = ["new_info", "extent", "moment_text", "bigger_meaning"]
CORE_M = ["topic", "remembered", "feelings", "personal_kinds", "matters_for", "reason"]
TEMPLATE = INSTR + "\n\n---\nREPORT TO CODE\nMOMENT: {mem}\nREASON: {why}\n"


def parse(t):
    s, e = (t or "").find("{"), (t or "").rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def job(a):
    out, key, mem, why, coder = a
    p = os.path.join(out, f"{key}__{coder.split('/')[1]}.json")
    if os.path.exists(p) and json.load(open(p)).get("ok"):
        return 0.0                                            # failed parses are retried
    msg = [{"role": "user", "content": TEMPLATE.replace("{mem}", mem).replace("{why}", why)}]
    kw = dict(max_tokens=int(os.environ.get("CODE_MAX_TOKENS", 12000)), temperature=0, response_format={"type": "json_object"},
              reasoning=({"effort": "none"} if "gpt" in coder else {"enabled": False}) if os.environ.get("REASONING", "on") == "off" else None)   # ON for v3.5 (A1, Oct 8)
    r = orc.call(coder, msg, os.path.join(out, "calls.jsonl"), tag=f"{key}|{coder}", **kw)
    d = parse(r.get("text"))
    json.dump(dict(key=key, coder=coder, codebook_sha=CODEBOOK_SHA, codes=d, ok=d is not None), open(p, "w"))
    return orc.cost(r)


def items(name):
    rows = [r for r in P.rows() if r["dyadID"] != "PID_n+PID_n"]
    if name == "pilot40":
        rng = random.Random(40); by = {}
        for r in rows:
            by.setdefault(r["selfPID"], []).append(r)
        people = sorted(by); rng.shuffle(people)
        pick = [rng.choice(by[p]) for p in people[:40]]          # one report per person, 40 people
        return [(r["perspectiveID"], r["conv_SP_Mem_descr"].strip(), r["conv_SP_Mem_reason"].strip()) for r in pick]
    if name == "pilot40b":                                       # pilot 2: 40 DIFFERENT people (not in pilot40)
        rng = random.Random(40); by = {}
        for r in rows:
            by.setdefault(r["selfPID"], []).append(r)
        people = sorted(by); rng.shuffle(people)
        rng2 = random.Random(41); pick = [rng2.choice(by[p]) for p in people[40:80]]
        return [(r["perspectiveID"], r["conv_SP_Mem_descr"].strip(), r["conv_SP_Mem_reason"].strip()) for r in pick]
    if name.startswith("twin"):                             # Oct 10 night: the internal twin's confederate reports (twin/v2/memories; v2b = firmer spec test)
        import glob as _g
        out = []
        for f in sorted(_g.glob(os.path.join(ROOT, "twin", "v2", "memories" if name == "twin" else "memories_" + name.split("_", 1)[1], "C*.json"))):
            j = json.load(open(f))
            if j.get("ok") and j["answers"].get("moment"): out.append((f"{j['call']}_{j['twin']}", str(j["answers"]["moment"]).strip(), str(j["answers"].get("reason", "")).strip()))
        return out
    if name == "human":
        return [(r["perspectiveID"], r["conv_SP_Mem_descr"].strip(), r["conv_SP_Mem_reason"].strip()) for r in rows]
    M5 = {"deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"}   # the 5 models with controls
    run, info = name.split(":")                                  # model predictions from a run, one condition
    out = []
    for l in open(os.path.join(ROOT, "runs", run, "preds.jsonl")):
        j = json.loads(l)
        if j.get("ok") and j["info"] == info and (j["model"].split("/")[1] in M5 or (run == "human_predict" and j["model"].startswith("human/"))) and j.get("modality", "text") == "text":   # people predicting (Oct 12)
            out.append((f"{j['perspectiveID']}|{j['model'].split('/')[1]}|{info}", str(j["pred"].get("moment", "")), str(j["pred"].get("reason", ""))))
    return out


def nset(v):
    if isinstance(v, dict): v = list(v.get("about") or []) + list(v.get("details") or [])   # the ranked reason: all choices
    if isinstance(v, str): v = [v]
    return frozenset(str(x).strip().lower() for x in (v or []))


def splits(codes):
    """Core dimensions with a clearly different reading: single-label, no value shared by 2+ coders when 3 coded (no majority
    among 5 when 5 coded = no value held by 3+); multi-label, mean pairwise Jaccard below .34."""
    import itertools, collections
    n = len(codes); need = 2 if n <= 4 else 3; out = []
    for d in CORE_S:
        vals = [str(c.get(d)).upper() for c in codes]
        if max(collections.Counter(vals).values()) < need: out.append(d)
    for d in CORE_M:
        sets = [nset(c.get(d)) for c in codes]
        jac = [len(a & b) / len(a | b) if a | b else 1 for a, b in itertools.combinations(sets, 2)]
        if sum(jac) / len(jac) < 0.34: out.append(d)
    return out


def consensus(codes_by_coder, tier3):
    """Per dimension: majority across all coders; ties broken by the tier-3 (Opus) code when present, else 'TIE'.
    Multi-label: each label kept if a majority of coders chose it."""
    import collections
    keys = set(k for c in codes_by_coder.values() for k in c)
    out = {}
    cs = list(codes_by_coder.values()); n = len(cs)
    for k in keys:
        v = [c.get(k) for c in cs]
        if any(isinstance(x, dict) for x in v):                    # nested reason / appraisal: keep each coder's object for the record
            out[k] = v; continue
        if any(isinstance(x, list) for x in v):
            cnt = collections.Counter(l for x in v for l in nset(x)); out[k] = sorted(l for l, m in cnt.items() if m > n / 2)
        else:
            cnt = collections.Counter(str(x).upper() for x in v if x is not None); top = cnt.most_common()
            if not top: out[k] = None
            elif len(top) == 1 or top[0][1] > top[1][1]: out[k] = top[0][0]
            else: out[k] = str(tier3.get(k)).upper() if tier3 else "TIE"
    return out


def adjudicate(out, its, conc):
    """Tier 2 on items with >= 3 split core dimensions among the base coders; tier 3 (Opus) on items still >= 3 split among 5.
    Writes FINAL.jsonl (consensus codes + provenance). Agreement statistics are computed on the base coders only."""
    import glob, collections
    C = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(out, "*__*.json")):
        j = json.load(open(f))
        if j.get("ok"): C[j["key"]][j["coder"]] = j["codes"]
    item = {k: (m, w) for k, m, w in its}
    t2 = [k for k in C if len([c for c in C[k] if c in CODERS]) >= 2 and len(splits([C[k][c] for c in C[k] if c in CODERS])) >= 3]
    print("tier 2 items", len(t2), "of", len(C), flush=True)
    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(job, [(out, k, item[k][0], item[k][1], c) for k in t2 for c in TIER2]))
    for f in glob.glob(os.path.join(out, "*__*.json")):
        j = json.load(open(f))
        if j.get("ok"): C[j["key"]][j["coder"]] = j["codes"]
    t3 = [k for k in t2 if len(splits([C[k][c] for c in C[k] if c in CODERS + TIER2])) >= 3]
    print("tier 3 items", len(t3), flush=True)
    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(job, [(out, k, item[k][0], item[k][1], c) for k in t3 for c in TIER3]))
    for f in glob.glob(os.path.join(out, "*__*.json")):
        j = json.load(open(f))
        if j.get("ok"): C[j["key"]][j["coder"]] = j["codes"]
    with open(os.path.join(out, "FINAL.jsonl"), "w") as fh:
        for k in sorted(C):
            panel = {c: v for c, v in C[k].items() if c in CODERS + TIER2}
            tier3 = next((v for c, v in C[k].items() if c in TIER3), None)
            fh.write(json.dumps(dict(key=k, tier=3 if tier3 else (2 if k in t2 else 1), coders=sorted(C[k]), codebook_sha=CODEBOOK_SHA,
                                     codes=consensus(panel, tier3))) + "\n")
    json.dump(dict(items=len(C), tier2=len(t2), tier3=len(t3), codebook_sha=CODEBOOK_SHA), open(os.path.join(out, "ADJUDICATION.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--conc", type=int, default=24)
    ap.add_argument("--adjudicate", action="store_true"); ap.add_argument("--coders", default="", help="comma list of short names to restrict base coders (A1, Oct 8: predictions get Haiku + Luna only)")
    a = ap.parse_args()
    if a.coders:
        want = a.coders.split(","); CODERS[:] = [c for c in CODERS if c.split("/")[1] in want]; assert CODERS, "no coders matched"
    out = os.path.join(ROOT, "analysis", "code_v351" + ("_reasoning_off" if os.environ.get("REASONING") == "off" else ""), a.set.replace(":", "__")); os.makedirs(out, exist_ok=True)
    its = items(a.set); rnd = random.Random(7); rnd.shuffle(its)      # random order; a person's reports dispersed
    jobs = []
    for key, mem, why in its:
        for c in CODERS:
            if "|" in key and c.split("/")[1].split("-")[0] in key.split("|")[1]:
                continue                                           # never code your own family's predictions
            jobs.append((out, key, mem, why, c))
    print("codebook", CODEBOOK_SHA, "items", len(its), "jobs", len(jobs), flush=True)
    with ThreadPoolExecutor(a.conc) as ex:
        cost = sum(ex.map(job, jobs))
    print("cost $", round(cost, 3))
    if a.adjudicate:
        adjudicate(out, its, a.conc)
