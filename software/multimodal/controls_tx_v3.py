#!/usr/bin/env python3
"""The paper's controls analysis (scripts/controls_analysis.py, run unchanged otherwise) on the Scribe-transcript reruns:
none/history from runs/tx_v3_cheap, history_prior / history_wrong / card_thin_wrong / card_narr_wrong from runs/tx_v3_r3c, same five
models, three judges. Cards with the RIGHT person were not rerun (A1: no card reruns), so the two right-vs-wrong CARD comparisons and
right-card coupling are left out here rather than mixing transcripts; everything else is computed with both arms on Scribe.
Second part: the transcriber effect inside each control condition (Scribe minus Whisper, paired per model x memory, mean of three
judges, bootstrap over people). -> analysis/controls_tx_v3.json"""
import collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = os.path.join(HERE, "..", "scripts"); ROOT = os.path.join(HERE, "..")
src = open(os.path.join(SCRIPTS, "controls_analysis.py")).read()
reps = [('RUNS = ["main_text_cheap", "cards_text_cheap", "cards_narr_cheap"] + [f"r3c_{t}" for t in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")]',
         'RUNS = ["tx_v3_cheap", "tx_v3_r3c"]'),
        ('    for info in ("card_thin", "card_narr", "card_thin_wrong", "card_narr_wrong"):\n        rung', '    for info in ("card_thin_wrong", "card_narr_wrong"):\n        rung'),
        ('"controls.json"', '"controls_tx_v3.json"'),
        ('if __name__ == "__main__":\n    main()', '')]
for a, b in reps:
    assert src.count(a) == 1, a[:60]; src = src.replace(a, b)
os.chdir(SCRIPTS); sys.path.insert(0, SCRIPTS)
g = {"__file__": os.path.join(SCRIPTS, "controls_analysis.py"), "__name__": "controls_tx_v3"}
exec(compile(src, "controls_tx_v3", "exec"), g)
g["main"]()
FM = g["FM"]; M5 = g["M5"]; MET = g["MET"]
res = json.load(open(os.path.join(ROOT, "analysis", "controls_tx_v3.json")))
W, S3 = {}, {}
for r in [f"r3c_{t}" for t in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")] + ["main_text_cheap"]:
    s, _ = FM.load(r); W.update(s)
for r in ["tx_v3_r3c", "tx_v3_cheap"]:
    s, _ = FM.load(r); S3.update(s)
rng = np.random.default_rng(0); res["transcriber_delta"] = {}
for info in ("none", "history", "history_prior", "history_wrong", "card_thin_wrong", "card_narr_wrong"):
    res["transcriber_delta"][info] = {}
    for m in MET:
        d, who, per = [], [], collections.defaultdict(list)
        for k, v3 in S3.items():
            if k[2] != info or k[1] != "text" or k[0] not in M5: continue
            v2 = W.get(k)
            if not v2: continue
            x, y = FM.avg2(v2, m), FM.avg2(v3, m)
            if x is None or y is None: continue
            d.append(y - x); who.append(k[3])
        if len(d) >= 20:
            lo, hi = FM.cboot(np.array(d), who, rng)
            res["transcriber_delta"][info][m] = dict(mean=round(float(np.mean(d)), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n=len(d))
json.dump(res, open(os.path.join(ROOT, "analysis", "controls_tx_v3.json"), "w"), indent=1)
print("== transcriber effect inside each condition (Scribe minus Whisper, mean of 3 judges)")
for info, D in res["transcriber_delta"].items():
    print(f"  {info:16s} " + "  ".join(f"{m[:10]} {v['mean']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] n={v['n']}" for m, v in D.items()))
