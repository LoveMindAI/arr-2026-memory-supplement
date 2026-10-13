#!/usr/bin/env python3
"""Load the coders' 12-call prediction files (PREDICT_<NAME>_v1.json from the predict pages) into a run folder that the
model pipelines read (judge.py --run human_predict; code_v35.py --set human_predict:none).

Usage: load_human_predictions.py PREDICT_A1_v1.json [PREDICT_A4_v1.json ...]
Each file is {"file": "PREDICT_A1_v1", "saved_at": ..., "answers": {"P07": {"moment": ..., "reason": ...}, ...}}.
Call ids map to reports through analysis/human_predict_key.json (P01-P06 shared by every coder).
Writes runs/human_predict/preds.jsonl in the model format, with model = "human/<NAME>", modality text, info none, frame seat.
Re-running replaces that coder's rows and keeps everyone else's. Unanswered calls are reported and skipped."""
import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KEY = json.load(open(os.path.join(ROOT, "analysis", "human_predict_key.json")))
OUT = os.path.join(ROOT, "runs", "human_predict"); os.makedirs(OUT, exist_ok=True)
PRED = os.path.join(OUT, "preds.jsonl")


def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    old = [json.loads(l) for l in open(PRED)] if os.path.exists(PRED) else []
    for f in sys.argv[1:]:
        j = json.load(open(os.path.expanduser(f)))
        name = j.get("file", os.path.basename(f)).split("_")[1].upper()
        old = [r for r in old if r["model"] != f"human/{name}"]
        n, skipped = 0, []
        for pid, a in sorted(j.get("answers", {}).items()):
            k = KEY.get(pid)
            m, r = (a.get("moment") or "").strip(), (a.get("reason") or "").strip()
            if not k or not m or not r: skipped.append(pid); continue
            key = k["perspectiveID"]
            old.append({"perspectiveID": key, "selfPID": key.split("+")[0], "dyadID": key, "model": f"human/{name}",
                        "modality": "text", "info": "none", "frame": "seat", "rep": 0, "ok": True, "call": pid,
                        "shared": bool(k.get("shared")), "pred": {"moment": m, "reason": r}, "saved_at": j.get("saved_at")})
            n += 1
        print(f"{name}: {n} predictions loaded" + (f"; unanswered or unknown: {skipped}" if skipped else ""))
    with open(PRED, "w") as fh:
        for r in old: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(old)} rows in {os.path.relpath(PRED, ROOT)}")


if __name__ == "__main__":
    main()
