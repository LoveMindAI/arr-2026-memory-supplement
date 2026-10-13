#!/usr/bin/env python3
"""Impression cards (A1, Oct 7): DeepSeek v4.1 Flash and GLM-5.3 FlashX read all of a person's conversation transcripts (their seat:
'You' / 'Partner') and estimate the HEXACO facets, blind to the scores. Kit reader prompt (reader_hexaco) unchanged except the opening of
the source text (approved wording in FOR_A1_FACTS_AND_PROMPTS.md §6). Kit reader settings: greedy, reasoning off, strict schema.
Writes analysis/impressions/<reader>/<pid>.json and, with 'score', analysis/impression_accuracy.json + cards/inputs/impression_facets.json (mean of readers)."""
import json, os, sys, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P
import importlib.util as _u
_sp = _u.spec_from_file_location("kit_prompts", "<LOCAL>/Documents/Codex/LAB/research/personality_models_kit/prompts.py")
KIT = _u.module_from_spec(_sp); _sp.loader.exec_module(KIT)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
READERS = {"deepseek": "deepseek/deepseek-v4.1-flash", "glm": "z-ai/glm-5.3-flashx", "luna": "openai/gpt-6-luna", "gemini": "google/gemini-3-flash-preview"}  # Luna + Gemini added (A1, Oct 7)
IDS = json.load(open("<LOCAL>/Documents/Codex/LAB/research/personality_models_kit/templates/reader_hexaco.json"))["ids"]
OPEN = ('SOURCE TEXT: transcripts of {n} short video calls in which the person labelled "You" talked with a different stranger each time. '
        'Judge only the person labelled "You"; the other speakers are different people.\n\n')


def people():
    calls = collections.defaultdict(list)
    for r in P.rows():
        if r["dyadID"] == "PID_n+PID_n":
            continue
        calls[r["selfPID"]].append(r)
    return {p: sorted(v, key=lambda r: int(r["roundIdx"])) for p, v in calls.items()}


def messages(rows):
    tx = "\n\n".join(f"Call {i+1}:\n" + P.transcript(r["dyadID"], r["selfPID"]) for i, r in enumerate(rows))
    msgs, rf = KIT.reader_messages("hexaco", tx)
    u = msgs[1]["content"]; assert u.startswith("SOURCE TEXT:\n"), u[:40]
    msgs[1]["content"] = OPEN.format(n=len(rows)) + u[len("SOURCE TEXT:\n"):]
    return msgs, rf


def read(tag, pid, rows):
    d = os.path.join(ROOT, "analysis", "impressions", tag); os.makedirs(d, exist_ok=True); p = os.path.join(d, pid + ".json")
    if os.path.exists(p):
        return 0.0
    msgs, rf = messages(rows)
    # GLM-5.3 FlashX refuses reasoning off ("Reasoning is mandatory for this endpoint"), so it runs at low effort with more room (logged deviation).
    glm = tag == "glm"
    r = orc.call(READERS[tag], msgs, os.path.join(d, "calls.jsonl"), tag=pid, max_tokens=12000 if glm else 2000, temperature=0, response_format=rf,
                 reasoning={"effort": "low"} if glm else ({"effort": "none"} if tag == "luna" else {"enabled": False}))
    t = r.get("text") or ""
    try:
        fac = json.loads(t[t.find("{"): t.rfind("}") + 1])["facets"]
        if set(fac) == set(IDS) and all(isinstance(v, (int, float)) and 1 <= v <= 5 for v in fac.values()):
            json.dump(dict(pid=pid, reader=READERS[tag], n_calls=len(rows), facets=fac), open(p, "w"))
    except Exception:
        pass
    return orc.cost(r)


def score():
    sys.path.insert(0, os.path.join(ROOT, "cards")); import score_fidelity as SF
    E = {}
    for tag in READERS:
        d = os.path.join(ROOT, "analysis", "impressions", tag)
        if not os.path.isdir(d): continue
        E[tag] = {f[:-5]: json.load(open(os.path.join(d, f)))["facets"] for f in os.listdir(d) if f.startswith("PID_")}
    tags = [t for t in READERS if E.get(t)]
    common = sorted(set.intersection(*[set(E[t]) for t in tags]))
    mean = {p: {i: sum(E[t][p][i] for t in tags) / len(tags) for i in IDS} for p in common}
    json.dump(mean, open(os.path.join(ROOT, "cards", "inputs", "impression_facets.json"), "w"), indent=1)
    res = {}
    for name, src in [(t, E[t]) for t in tags] + [("mean", mean)]:
        ps = [p for p in src if p in SF.TRUE]
        Pm = np.array([[src[p][i] for i in SF.IDS] for p in ps]); T = np.array([[SF.TRUE[p][SF.MAP[i]] for i in SF.IDS] for p in ps])
        fr = [float(np.corrcoef(Pm[:, j], T[:, j])[0, 1]) for j in range(24)]
        dom = lambda M: np.stack([M[:, [j for j, i in enumerate(SF.IDS) if i[0] == d]].mean(1) for d in "HEXACO"], 1)
        dr = {d: float(np.corrcoef(dom(Pm)[:, k], dom(T)[:, k])[0, 1]) for k, d in enumerate("HEXACO")}
        res[name] = dict(n=len(ps), facet_r_mean=float(np.mean(fr)), facet_r=dict(zip(SF.IDS, fr)), domain_r=dr, pooled_ccc=float(SF.ccc(Pm.ravel(), T.ravel())))
        print(name, "n", len(ps), "mean facet r", round(np.mean(fr), 3), "domain r", {k: round(v, 2) for k, v in dr.items()})
    json.dump(res, open(os.path.join(ROOT, "analysis", "impression_accuracy.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["score"]:
        score(); sys.exit()
    ppl = people()
    if os.environ.get("LIMIT"):
        ppl = dict(list(ppl.items())[:int(os.environ["LIMIT"])])
    for tag in READERS:
        print(tag, "people", len(ppl), "cost $", round(sum(ThreadPoolExecutor(24).map(lambda kv: read(tag, *kv), ppl.items())), 3))
