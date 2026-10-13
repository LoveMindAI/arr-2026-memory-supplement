#!/usr/bin/env python3
"""Session-level sensitivity (A3's note 6, Oct 13): the main judged contrasts with 95% intervals from resampling whole SESSIONS
(PARSEL groups, batchID) instead of people, so partners and shared comparison reports stay together. Same paired differences,
same three judges, same counting as the paper (models whose interval is above / below zero). Both resamplings reported side by side.
-> analysis/session_bootstrap.json"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figs_models as FM, controls_analysis as CA, prompts as P
ROOT = FM.ROOT
ROWS = {r["perspectiveID"]: r for r in P.rows()}
PERSON_BOOT = FM.cboot
def session_boot(d, pids, rng, B=2000):
    g = collections.defaultdict(list)
    for i, p in enumerate(pids): g[ROWS[p]["batchID"]].append(i)
    G = list(g.values()); d = np.asarray(d, float)
    gs = np.array([d[ix].sum() for ix in G]); gn = np.array([len(ix) for ix in G])
    bs = [gs[k].sum() / gn[k].sum() for k in (rng.integers(0, len(G), len(G)) for _ in range(B))]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
MET = ["same_focus", "same_reason", "similarity", "event_match"]
S = {}
for r in ["main_text_cheap", "main_text_pro", "cards_text_cheap", "cards_narr_cheap", "r2_weave_factnarr"] + [f"r3c_{t}" for t in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")]:
    s, _ = FM.load(r); S.update(s)
ALL = sorted({k[0] for k in S if k[1] == "text" and k[2] == "history"})
C = [("history vs none (all models)", "none", "history", ALL, None),
     ("own vs stranger history", "history_wrong", "history", CA.M5, None),
     ("stranger history vs none", "none", "history_wrong", CA.M5, None),
     ("stranger written profile vs none", "none", "card_narr_wrong", CA.M5, None),
     ("own vs stranger written profile", "card_narr_wrong", "card_narr", CA.M5, None),
     ("woven profile vs none", "none", "card_weave", CA.M5, None)]
out = {}
for name, a, b, models, keep in C:
    out[name] = {}
    for m in MET:
        row = {}
        for lab, fn in (("person", PERSON_BOOT), ("session", session_boot)):
            FM.cboot = fn; rng = np.random.default_rng(0)
            cs = [CA.paired_sub(S, mdl, a, b, m, rng, keep) for mdl in models]; cs = [c for c in cs if c]
            row[lab] = dict(models=len(cs), up=sum(c["lo"] > 0 for c in cs), down=sum(c["hi"] < 0 for c in cs),
                            mean=round(float(np.mean([c["mean"] for c in cs])), 3) if cs else None,
                            mean_width=round(float(np.mean([c["hi"] - c["lo"] for c in cs])), 3) if cs else None)
        out[name][m] = row
        print(f"{name:34s} {m:12s} person up/down {row['person']['up']}/{row['person']['down']} of {row['person']['models']} (w {row['person']['mean_width']})"
              f" | session {row['session']['up']}/{row['session']['down']} (w {row['session']['mean_width']})", flush=True)
FM.cboot = PERSON_BOOT
json.dump(out, open(os.path.join(ROOT, "analysis", "session_bootstrap.json"), "w"), indent=1)
