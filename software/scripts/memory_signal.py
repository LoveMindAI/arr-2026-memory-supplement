#!/usr/bin/env python3
"""Which turns get remembered? Diachek & Brown-Schmidt (2024)-style test on PARSEL.
Unit = turn (v2 transcript turns) within each perspective whose memory was located in the conversation.
Outcome = turn is inside the located remembered span. Predictors:
  own (rememberer's own turn), backchanneled (other person backchannels right after it), filler (um/uh/er/hmm... in the turn;
  from v3 verbatim words when available, else v2), question ('?'), log length, relative position + position^2.
Logit with cluster-robust SEs by perspective. Exploratory; locator = Claude Sonnet 5.5 (validated vs timing slider, rho .65)."""
import glob, json, math, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V3 = os.path.join(ROOT, "transcripts_v3_verbatim")
FILL = re.compile(r"^(um+|uh+|erm+|er+|hmm+|mm+|ah+)$", re.I)
BC = re.compile(r"^(yeah|yes|yep|ok|okay|right|mhm|mm-?hmm|uh-?huh|oh|wow|sure|cool|nice|really|lovely|ah|aw|ha)([ ,.!?]+(yeah|yes|right|okay|ok|wow|really|nice|lovely))*[.!?,]*$", re.I)


def fillers_in(words, spk, a, b):
    return sum(1 for w in words if w["spk"] == spk and a - 0.05 <= w["start"] < b + 0.05 and FILL.match(re.sub(r"[^\w-]", "", w["word"]) or "x"))


def main(source="auto"):
    rows = {r["perspectiveID"]: r for r in P.rows()}
    X, y, groups = [], [], []
    used_v3 = 0
    for f in glob.glob(os.path.join(ROOT, "analysis", "located", "PID_*.json")):
        o = json.load(open(f))
        if o.get("kind") != "in_conversation" or not o.get("turns"):
            continue
        r = rows[o["perspectiveID"]]
        j = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))
        turns = j["turns"]; dur = max(t["end"] for t in turns)
        words = j["words"]
        v3p = os.path.join(V3, r["dyadID"] + ".json")
        if source != "v2" and os.path.exists(v3p):
            words = json.load(open(v3p))["words"]; used_v3 += 1
        elif source == "v3":
            continue
        span = set(range(min(o["turns"]), max(o["turns"]) + 1))
        for i, t in enumerate(turns):
            is_bc = t.get("bc") or bool(BC.match(t["text"].strip()))
            if is_bc:
                continue  # backchannel turns themselves are not candidate "moments"
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            backchanneled = bool(nxt and nxt["spk"] != t["spk"] and (nxt.get("bc") or BC.match(nxt["text"].strip())))
            n = max(1, len(t["text"].split()))
            pos = t["start"] / dur
            X.append([1.0, float(t["spk"] == r["selfPID"]), float(backchanneled), float(fillers_in(words, t["spk"], t["start"], t["end"]) > 0),
                      float("?" in t["text"]), math.log(n), pos, pos * pos])
            y.append(float(i in span)); groups.append(o["perspectiveID"])
    X, y = np.array(X), np.array(y)
    import statsmodels.api as sm
    names = ["const", "own_turn", "backchanneled", "filler", "question", "log_words", "rel_pos", "rel_pos^2"]
    m = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": np.unique(groups, return_inverse=True)[1]})
    print(f"source={source} perspectives={len(set(groups))} (v3 words used for {used_v3}) turns={len(y)} remembered-turn rate={y.mean():.3f}")
    print(f"base rates: own {X[:,1].mean():.2f}  backchanneled {X[:,2].mean():.2f}  filler {X[:,3].mean():.2f}  question {X[:,4].mean():.2f}")
    for k, nm in enumerate(names):
        if k == 0:
            continue
        b, se = m.params[k], m.bse[k]
        print(f"  {nm:14s} OR={math.exp(b):5.2f}  [{math.exp(b-1.96*se):.2f}, {math.exp(b+1.96*se):.2f}]  p={m.pvalues[k]:.3g}")
    out = os.path.join(ROOT, "analysis", f"memory_signal_{source}.json")
    json.dump(dict(source=source, n_turns=int(len(y)), n_persp=len(set(groups)), params=dict(zip(names, map(float, m.params))),
                   se=dict(zip(names, map(float, m.bse))), p=dict(zip(names, map(float, m.pvalues)))), open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
