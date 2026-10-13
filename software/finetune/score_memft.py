#!/usr/bin/env python3
"""Score memft1 eval outputs: parse JSON answers, write runs/memft1_eval/preds.jsonl, run the blind judges (same judge.py),
then report per tag (base / epoch1 / epoch2 / epoch4) on val and test, plus timing error and parse rate.
Epoch selection on VAL by Sonnet similarity; the headline is TEST for base vs selected epoch."""
import glob, json, os, subprocess, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prompts as P
RUN = "memft1_eval"; RD = os.path.join(ROOT, "runs", RUN); os.makedirs(RD, exist_ok=True)


def parse(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def main():
    rows = {r["perspectiveID"]: r for r in P.rows()}
    split = {}
    out = open(os.path.join(RD, "preds.jsonl"), "w")
    stats = defaultdict(lambda: [0, 0])
    for f in glob.glob(os.path.join(HERE, "run_memft1", "eval_out", "*.jsonl")):
        for l in open(f):
            x = json.loads(l); p = parse(x["text"]); split[x["perspectiveID"]] = x["split"]
            ok = bool(p and p.get("moment")); stats[(x["tag"], x["split"])][0] += 1; stats[(x["tag"], x["split"])][1] += ok
            out.write(json.dumps(dict(perspectiveID=x["perspectiveID"], selfPID=rows[x["perspectiveID"]]["selfPID"], model=f"gemma4-31b:{x['tag']}",
                                      modality="text", info="none", frame="seat", rep=0, ok=ok, pred=p, split=x["split"])) + "\n")
    out.close()
    for k, v in sorted(stats.items()):
        print("parse", k, f"{v[1]}/{v[0]}")
    if "--nojudge" not in sys.argv:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "judge.py"), "--run", RUN], check=True)
    J = [json.loads(l) for l in open(os.path.join(RD, "judged.jsonl"))]
    agg = defaultdict(list)
    for j in J:
        if not j["scores"]:
            continue
        pid = j["id"].split("|")[0]
        tag = j["id"].split("|")[1] if j["kind"] == "model" else j["kind"]
        agg[(tag, split.get(pid, "?"), j["judge"].split("/")[1])].append(j["scores"])
    preds = [json.loads(l) for l in open(os.path.join(RD, "preds.jsonl"))]
    terr = defaultdict(list)
    for p in preds:
        if p["ok"]:
            try:
                terr[(p["model"], p["split"])].append(abs(float(p["pred"].get("timing")) - float(rows[p["perspectiveID"]]["conv_SP_Mem"])))
            except Exception:
                pass
    print(f"\n{'tag':22s} {'split':5s} {'judge':18s}  n   topic event  sim  focus reason | timing |err|")
    for k in sorted(agg):
        v = agg[k]; te = terr.get((k[0], k[1]))
        print(f"{k[0]:22s} {k[1]:5s} {k[2]:18s} {len(v):3d}  " + "  ".join(f"{np.mean([s[m] for s in v]):.2f}" for m in
              ["topic_match", "event_match", "similarity", "same_focus", "same_reason"]) + (f" | {np.median(te):.0f}" if te else ""))


if __name__ == "__main__":
    main()
