#!/usr/bin/env python3
"""Summarize a judged run: means per (model, modality, info, judge) and paired contrasts (info B minus info A on the same memories),
bootstrap 95% CIs over memories. Usage: summarize_run.py RUN [--a none --b history]"""
import argparse, collections, json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
METRICS = ["topic_match", "event_match", "similarity", "same_focus", "same_reason"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--a", default="none"); ap.add_argument("--b", default="history")
    a = ap.parse_args()
    J = [json.loads(l) for l in open(os.path.join(ROOT, "runs", a.run, "judged.jsonl"))]
    means = collections.defaultdict(list); S = collections.defaultdict(dict)
    for j in J:
        if not j["scores"]:
            continue
        jd = j["judge"].split("/")[1]
        if j["kind"] != "model":
            means[(j["kind"], "", "", jd)].append(j["scores"]); continue
        pid, model, mod, info, frame, rep = j["id"].split("|")
        means[(model.split("/")[1], mod, info, jd)].append(j["scores"])
        S[(model.split("/")[1], mod, jd, pid)][info] = j["scores"]
    print(f"{'model':28s} {'mod':6s} {'info':8s} {'judge':18s}  n    " + "  ".join(m[:6] for m in METRICS))
    for k in sorted(means):
        v = means[k]
        print(f"{k[0]:28s} {k[1]:6s} {k[2]:8s} {k[3]:18s} {len(v):4d}  " + "  ".join(f"{np.mean([s[m] for s in v]):.3f}" for m in METRICS))
    print(f"\nPaired {a.b} − {a.a} (bootstrap 95%)")
    rng = np.random.default_rng(0); out = {}
    for (model, mod, jd) in sorted({(k[0], k[1], k[2]) for k in S}):
        keys = [k for k in S if k[:3] == (model, mod, jd) and a.a in S[k] and a.b in S[k]]
        if len(keys) < 20:
            continue
        cells = []
        for m in ["similarity", "same_focus", "same_reason", "event_match"]:
            d = np.array([S[k][a.b][m] - S[k][a.a][m] for k in keys])
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5]); out[f"{model}|{mod}|{jd}|{m}"] = dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=len(d))
            cells.append(f"{m[:6]} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{model:28s} {mod:6s} {jd:18s} n={len(keys):4d}  " + "  ".join(cells))
    json.dump(out, open(os.path.join(ROOT, "runs", a.run, f"PAIRED_{a.b}_minus_{a.a}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
