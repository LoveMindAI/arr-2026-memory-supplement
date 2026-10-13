#!/usr/bin/env python3
"""Load human coders' GUI saves (codebook v3.5.x) into per-item files the agreement scripts read.

Usage: load_human_coders.py OUTDIR SAVE.json [SAVE.json ...]
  e.g. load_human_coders.py analysis/code_v351/humans <LOCAL_FILE> A4_first50_v3.5.json

Each save has {"coder": "A1", "codebook": "3.5.1", "mode": "code", "codes": {"B001": {...}, ...}}.
B-ids map to report keys (PID_x+PID_y) through analysis/A3_item_key.json (the same key for every coder's page).
Writes OUTDIR/<key>__human-<INI>.json in the model-file format ({"key","coder","codes","ok"}), and copies the
three AI coders' files for the same keys from analysis/code_v351/human/, so one folder holds every coder.
Prints a completeness report per coder (questions left blank, unknown codes). Never deletes anything.
"""
import json, os, shutil, sys, collections, importlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "codebook"))
SPEC = importlib.import_module(os.environ.get("SPEC", "spec_v351_FROZEN")).SPEC
Q = [q for p in SPEC["parts"] if p.get("models") is not False for q in p["questions"]]
QIDS = [q["id"] for q in Q]
KEY = json.load(open(os.environ.get("KEY_FILE") or os.path.join(ROOT, "analysis", "A3_item_key.json")))   # env KEY_FILE: e.g. twin_public/a1_page_key.json
MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(ROOT, "analysis", "code_v351", "human")   # env MODEL_DIR: e.g. twin_public/coding
MODELS = ["claude-haiku-5.5", "gpt-6-luna", "deepseek-v4.1-flash"]


def main():
    if len(sys.argv) < 3: sys.exit(__doc__)
    out = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(ROOT, sys.argv[1]); os.makedirs(out, exist_ok=True)
    manifest = {}
    for f in sys.argv[2:]:
        j = json.load(open(os.path.expanduser(f)))
        ini = str(j.get("coder", "")).strip().upper() or os.path.basename(f).split("_")[0][:2].upper()
        if j.get("mode") not in (None, "code"): print(f"SKIP {f}: mode={j.get('mode')} (prediction files use a different loader)"); continue
        if not str(j.get("codebook", "")).startswith("3.5"): print(f"WARNING {f}: codebook {j.get('codebook')} (expected 3.5.x)")
        blanks, n = collections.Counter(), 0
        for bid, codes in j["codes"].items():
            if bid not in KEY: print(f"  {ini}: unknown item id {bid}"); continue
            key = KEY[bid]; n += 1
            for q in QIDS:
                v = codes.get(q)
                if v in (None, "", [], {}) or (q == "reason" and not (v or {}).get("about")): blanks[q] += 1
            keep = {k: v for k, v in codes.items() if k in QIDS or k.endswith("_other")}
            json.dump({"key": key, "coder": f"human-{ini}", "codes": keep, "ok": True, "item": bid,
                       "source": os.path.basename(f), "saved_at": j.get("saved_at")},
                      open(os.path.join(out, f"{key}__human-{ini}.json"), "w"), indent=1)
            for m in MODELS:
                src = os.path.join(MODEL_DIR, f"{key}__{m}.json")
                if os.path.exists(src): shutil.copy(src, os.path.join(out, f"{key}__{m}.json"))
        manifest.setdefault(ini, []).append({"file": os.path.abspath(os.path.expanduser(f)), "items": n, "saved_at": j.get("saved_at"),
                                             "codebook": j.get("codebook"), "blank_answers": dict(blanks)})
        print(f"{ini}: {n} items from {os.path.basename(f)} (saved {j.get('saved_at')}); blank answers: {dict(blanks) or 'none'}")
    old = json.load(open(os.path.join(out, "MANIFEST.json"))) if os.path.exists(os.path.join(out, "MANIFEST.json")) else {}
    old.update(manifest); json.dump(old, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
