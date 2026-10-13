#!/usr/bin/env python3
"""Run memory predictions. Resumable: one JSONL line per (perspective, model, modality, info, frame, rep).
Usage:
  run_predict.py --run NAME --pids P1 P2 ... --models m1 m2 --modalities text audio --infos none card --frame seat [--reps 1]
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEXT_ONLY = {"deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3-flashx", "openai/gpt-5.6-sol", "moonshotai/kimi-k2-0905", "xiaomi/mimo-v2.5-pro"}
AUDIO_OK = {"xiaomi/mimo-v2.6-pro-ultraspeed", "xiaomi/mimo-v2.6-flash", "qwen/qwen3.8-omni-flash", "google/gemini-3.8-flash",
            "google/gemini-3.1-pro-preview", "thinkingmachines/inkling", "thinkingmachines/inkling-small", "openai/gpt-audio",
            "mistralai/voxtral-small-24b-2507", "xiaomi/mimo-v2.5"}
VIDEO_OK = {"xiaomi/mimo-v2.6-pro-ultraspeed", "xiaomi/mimo-v2.6-flash", "qwen/qwen3.8-omni-flash", "google/gemini-3.8-flash",
            "google/gemini-3.1-pro-preview", "xiaomi/mimo-v2.5"}
NO_TEMP = {"openai/gpt-5.6-sol"}


def supported(model, modality):
    if model == "openai/gpt-audio" and "audio" not in modality:
        return False  # requires audio in the input
    if "audio" in modality and model not in AUDIO_OK:
        return False
    if "video" in modality and model not in VIDEO_OK:
        return False
    return True


def parse(t):
    if not t:
        return None
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def key(r):
    return f"{r['perspectiveID']}|{r['model']}|{r['modality']}|{r['info']}|{r['frame']}|{r['rep']}"


def job(spec, run_dir, temperature):
    row, model, modality, info, frame, rep = spec
    rec = dict(perspectiveID=row["perspectiveID"], selfPID=row["selfPID"], dyadID=row["dyadID"], model=model, modality=modality,
               info=info, frame=frame, rep=rep)
    try:
        msgs = P.build(row, modality, info, frame, model)
    except Exception as e:
        rec.update(ok=False, error=f"build: {e}")
        return rec
    r = orc.call(model, msgs, os.path.join(run_dir, "calls.jsonl"), tag=key(rec), max_tokens=16000,
                 temperature=None if model in NO_TEMP else temperature)
    p = parse(r.get("text"))
    rec.update(ok=bool(p and p.get("moment")), pred=p, raw=(r.get("text") or "")[:4000] if not p else None,
               error=None if r.get("ok") else json.dumps(r.get("error"))[:500], cost=orc.cost(r), usage=r.get("usage"),
               finish=r.get("finish"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pids", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--modalities", nargs="+", default=["text"])
    ap.add_argument("--infos", nargs="+", default=["none"])
    ap.add_argument("--frame", default="seat")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--budget", type=float, default=5.0, help="stop launching new calls after this many dollars")
    a = ap.parse_args()
    run_dir = os.path.join(ROOT, "runs", a.run)
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, "preds.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            if r.get("ok"):
                done.add(key(r))
    rows = {r["perspectiveID"]: r for r in P.rows()}
    specs = []
    for pid in a.pids:
        for m in a.models:
            for mod in a.modalities:
                if not supported(m, mod):
                    continue
                for info in a.infos:
                    for rep in range(a.reps):
                        s = (rows[pid], m, mod, info, a.frame, rep)
                        if f"{pid}|{m}|{mod}|{info}|{a.frame}|{rep}" not in done:
                            specs.append(s)
    print("calls to make", len(specs), flush=True)
    spent = 0.0
    with ThreadPoolExecutor(a.conc) as ex:
        futs = []
        for s in specs:
            futs.append(ex.submit(job, s, run_dir, a.temperature))
        for f in futs:
            r = f.result()
            spent += r.get("cost") or 0
            with open(out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(("OK " if r["ok"] else "ERR"), r["model"], r["modality"], r["info"], r["perspectiveID"], f"${r.get('cost') or 0:.4f}",
                  (r.get("error") or "")[:160], flush=True)
            if spent > a.budget:
                print("BUDGET REACHED; cancelling remaining", flush=True)
                for g in futs:
                    g.cancel()
                break
    print("spent $", round(spent, 4))


if __name__ == "__main__":
    main()
