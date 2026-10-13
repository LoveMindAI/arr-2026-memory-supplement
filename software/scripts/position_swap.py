#!/usr/bin/env python3
"""Position-swap experiment (A1, Oct 9; prompts approved with his change: a literal faithfulness check by Haiku + Luna, Sol if they disagree).
Bank: memories the person placed mid-call (locator 30-70% of the call, high/medium confidence, own slider 30-70) where >=3 of the 5 main
models put their predicted moment in the last fifth. Versions: SWAP (the person's remembered stretch moved to just before any closing,
later material moved earlier; editor Sonnet 5.5) and CONTROL (same editor, same edit allowance, order unchanged). Edits must pass the code's
word-for-word checks AND the faithfulness check. Then: the 5 main models predict (no history, prompt otherwise identical), the same locator
places their moments in the edited transcript, and the same blind judges score them against the person's memory.
Stages: --stage edit | predict | locate. Judging: judge.py --run position_swap. Cap $30 overall. -> runs/position_swap/, analysis/position_swap/"""
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
import orc, prompts as P, locate_moments as LM
ROOT = ".."; OUT = os.path.join(ROOT, "analysis", "position_swap"); os.makedirs(OUT, exist_ok=True)
RUN = os.path.join(ROOT, "runs", "position_swap"); os.makedirs(RUN, exist_ok=True)
EDITOR, CHECKERS, ARBITER = "anthropic/claude-sonnet-5.5", ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"], "openai/gpt-6.1-sol"
M5 = ["deepseek/deepseek-v4.1-flash", "google/gemini-3.8-flash", "google/gemma-4-31b-it", "qwen/qwen3.8-omni-flash", "xiaomi/mimo-v2.6-flash"]
CAP = 30.0; spent = [0.0]; lock = threading.Lock(); RETRY = False
rows = {r["perspectiveID"]: r for r in P.rows()}
def cost(c):
    with lock: spent[0] += orc.cost(c)
def pj(t):
    t = t or ""; a, b = (t.find("["), t.rfind("]")) if t.strip().startswith("[") or ('"from"' in t and t.find("[") < t.find("{")) else (t.find("{"), t.rfind("}"))
    try: return json.loads(t[a:b + 1])
    except Exception: return None
def loc(p):
    if not os.path.exists(p): return None
    L = json.load(open(p))
    if L.get("kind") != "in_conversation" or not L.get("turns"): return None
    t = L["turns"]; t = json.loads(t) if isinstance(t, str) else t
    return L, t
def bank():
    out = []
    for pid in sorted(rows):
        x = loc(os.path.join(ROOT, "analysis", "located", pid + ".json"))
        if not x: continue
        L, t = x; n = int(L["n_turns"]); pos = ((t[0] + t[-1]) / 2) / max(1, n - 1)
        try: sl = float(L.get("human_timing"))
        except Exception: continue
        if not (0.3 <= pos <= 0.7 and 30 <= sl <= 70 and L.get("confidence") in ("high", "medium")): continue
        late = 0
        for m in M5:
            y = loc(os.path.join(ROOT, "analysis", "located_preds", m.split("/")[1], pid + ".json"))
            if y and ((y[1][0] + y[1][-1]) / 2) / max(1, int(y[0]["n_turns"]) - 1) >= 0.8: late += 1
        if late >= 3: out.append(dict(pid=pid, i=min(t), j=max(t), late=late))
    return out
def turns_of(pid):
    r = rows[pid]; return json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"], r["selfPID"]
def numbered(T, me, mark=None):
    return "\n".join(f"[{k}] {'You' if t['spk'] == me else 'Partner'}: {t['text']}" for k, t in enumerate(T))
SWAP = """Below is the transcript of a 3-minute video call between two strangers, as a numbered list of turns. Turns {i} to {j} are marked. Rewrite the conversation so that the marked turns happen near the end of the call, just before any closing remarks or goodbyes, and the turns that originally came after them happen earlier instead. The result should read like a real conversation that simply happened in this order.

Rules:
- Keep every marked turn word for word, with the same speaker.
- Keep all other turns as close to word for word as you can. You may add at most three short bridging phrases (up to 8 words each, such as "Oh, going back to what you said...") where the new order needs them.
- Don't add any new information, opinions, events or feelings, and don't remove anything except at most two filler turns ("yeah", "mm") where needed for flow.
- If the call ends with goodbyes or closing remarks, they stay at the very end.

Return JSON only: a list of turns in the new order, each {{"from": <original turn number, or null for a bridge you added>, "speaker": "You" or "Partner", "text": "..."}}.

TRANSCRIPT:
{tx}"""
CTRL = """Below is the transcript of a 3-minute video call between two strangers, as a numbered list of turns. Lightly smooth this conversation without changing its order. You may add at most three short bridging phrases (up to 8 words each) at natural points, and drop at most two filler turns. Keep everything else word for word, and don't add any new information, opinions, events or feelings.

Return JSON only: a list of turns in order, each {{"from": <original turn number, or null for a bridge you added>, "speaker": "You" or "Partner", "text": "..."}}.

TRANSCRIPT:
{tx}"""
CHECK = """You will see an original transcript, an edited version, and the instructions the editor was given. Check whether the edit followed the instructions exactly:
1. {q1}
2. {q2}
3. Is everything else word for word, apart from at most three short bridges (8 words or fewer) and at most two dropped filler turns?
4. Was any new information, opinion, event or feeling added?

Return JSON only: {{"faithful": true or false, "problems": "..." or ""}}.

INSTRUCTIONS GIVEN TO THE EDITOR:
{instr}

ORIGINAL:
{orig}

EDITED:
{edit}"""
def code_check(T, me, E, version, i=None, j=None):
    probs = []
    if not isinstance(E, list) or not E: return ["not a list"]
    srcs = [e.get("from") for e in E]
    bridges = [e for e in E if e.get("from") is None]
    if len(bridges) > 3: probs.append("more than 3 bridges")
    if any(len(str(e.get("text", "")).split()) > 8 for e in bridges): probs.append("bridge over 8 words")
    used = [s for s in srcs if isinstance(s, int)]
    if len(used) != len(set(used)): probs.append("a turn repeated")
    dropped = [k for k in range(len(T)) if k not in used]
    if len(dropped) > 2: probs.append(f"{len(dropped)} turns dropped")
    if any(len(T[k]["text"].split()) > 3 for k in dropped): probs.append("dropped a non-filler turn")
    if version == "swap" and any(k in dropped for k in range(i, j + 1)): probs.append("dropped a marked turn")
    changed = 0
    for e in E:
        s = e.get("from")
        if isinstance(s, int) and 0 <= s < len(T):
            same = str(e.get("text", "")).strip() == T[s]["text"].strip()
            spk_ok = (e.get("speaker") == "You") == (T[s]["spk"] == me)
            if version == "swap" and i <= s <= j and not same: probs.append(f"marked turn {s} not word for word")
            if not spk_ok: probs.append(f"speaker changed on {s}")
            changed += (not same)
    if changed > 3: probs.append(f"{changed} turns reworded")
    if version == "control" and used != sorted(used): probs.append("order changed")
    if version == "swap":
        newpos = [k for k, e in enumerate(E) if isinstance(e.get("from"), int) and i <= e["from"] <= j]
        if not newpos or (sum(newpos) / len(newpos)) / max(1, len(E) - 1) < 0.65: probs.append("marked turns not moved near the end")
    return probs
def build_turns(T, me, E):
    """New transcript in the original JSON shape, with each turn's original duration, laid out in the new order."""
    out, t0 = [], T[0]["start"] if T else 0.0
    for e in E:
        s = e.get("from")
        if isinstance(s, int) and 0 <= s < len(T):
            d = max(0.5, T[s]["end"] - T[s]["start"]); spk = T[s]["spk"]
        else:
            d = max(1.0, len(str(e.get("text", "")).split()) / 2.5); spk = me if e.get("speaker") == "You" else next((t["spk"] for t in T if t["spk"] != me), "partner")
        out.append(dict(spk=spk, start=round(t0, 2), end=round(t0 + d, 2), text=str(e.get("text", "")), src=s)); t0 += d
    return out
def edit_one(b):
    pid = b["pid"]; T, me = turns_of(pid); tx = numbered(T, me); res = {}
    for version, prompt in (("swap", SWAP.format(i=b["i"], j=b["j"], tx=tx)), ("control", CTRL.format(tx=tx))):
        p = os.path.join(OUT, f"{pid}__{version}.json"); prev = None
        if os.path.exists(p):
            prev = json.load(open(p))
            if prev["faithful"] or prev.get("retried") or not RETRY: res[version] = prev; continue
        if spent[0] > CAP: return
        msgs = [{"role": "user", "content": prompt}]
        if prev:   # one retry, told exactly what the checks found (Oct 9); first attempt kept in the record
            fb = "; ".join(prev["code_problems"] + [str(c.get("problems", "")) for c in prev["checks"].values() if c and not c.get("faithful")])
            msgs = [{"role": "user", "content": prompt + "\n\nA previous attempt at this edit had these problems: " + fb + "\nPlease redo the edit, following the rules exactly."}]
        c = orc.call(EDITOR, msgs, os.path.join(OUT, "edit_calls.jsonl"), tag=f"{pid}|{version}|{'retry' if prev else 'first'}", max_tokens=16000, temperature=0); cost(c)
        E = pj(c.get("text")); E = E.get("turns") if isinstance(E, dict) else E
        probs = code_check(T, me, E, version, b["i"], b["j"])
        rec = dict(pid=pid, version=version, i=b["i"], j=b["j"], code_problems=probs, edit=E, checks={})
        if not probs:
            q1 = f"Are the marked turns ({b['i']} to {b['j']}) present word for word, with the same speakers?" if version == "swap" else "Does the order of the turns stay the same?"
            q2 = "Do the marked turns now come near the end, before any closing remarks, with the later material moved earlier?" if version == "swap" else "Were only small bridging phrases added, if any?"
            instr = prompt.split("TRANSCRIPT:")[0].strip()
            et = "\n".join(f"{'You' if e.get('speaker') == 'You' else 'Partner'}: {e.get('text', '')}" for e in E)
            msg = [{"role": "user", "content": CHECK.format(q1=q1, q2=q2, instr=instr, orig=tx, edit=et)}]
            for m in CHECKERS:
                cc = orc.call(m, msg, os.path.join(OUT, "check_calls.jsonl"), tag=f"{pid}|{version}|{m}", max_tokens=8000, temperature=0, response_format={"type": "json_object"}); cost(cc)
                rec["checks"][m] = pj(cc.get("text")) or {}
            v = [bool(rec["checks"][m].get("faithful")) for m in CHECKERS]
            if v[0] != v[1]:
                cc = orc.call(ARBITER, msg, os.path.join(OUT, "check_calls.jsonl"), tag=f"{pid}|{version}|sol", max_tokens=8000, temperature=0, response_format={"type": "json_object"}); cost(cc)
                rec["checks"][ARBITER] = pj(cc.get("text")) or {}; rec["faithful"] = bool(rec["checks"][ARBITER].get("faithful"))
            else: rec["faithful"] = v[0]
        else: rec["faithful"] = False
        if rec["faithful"]: rec["turns"] = build_turns(T, me, E)
        if prev: rec["retried"] = True; rec["first_attempt"] = {k: prev.get(k) for k in ("code_problems", "checks", "edit")}
        json.dump(rec, open(p, "w")); res[version] = rec
def custom_messages(pid, new_turns):
    r = rows[pid]; msgs = P.build(r, "text", "none"); orig = P.transcript(r["dyadID"], r["selfPID"])
    new = "\n".join(f"[{int(t['start'] // 60)}:{int(t['start'] % 60):02d}] {'You' if t['spk'] == r['selfPID'] else 'Partner'}: {t['text']}" for t in new_turns)
    for part in msgs[-1]["content"]:
        if part["type"] == "text" and orig in part["text"]: part["text"] = part["text"].replace(orig, new); return msgs
    raise RuntimeError("transcript not found in prompt")
def predict_one(args):
    pid, version, model = args; key = f"{pid}|{model}|text|{version}|seat|0"
    with lock:
        if key in DONE or spent[0] > CAP: return
    rec = json.load(open(os.path.join(OUT, f"{pid}__{version}.json"))); msgs = custom_messages(pid, rec["turns"])
    c = orc.call(model, msgs, os.path.join(RUN, "calls.jsonl"), tag=key, max_tokens=16000, temperature=0.7); cost(c)
    d = pj(c.get("text")); ok = isinstance(d, dict) and bool(d.get("moment"))
    with lock:
        open(os.path.join(RUN, "preds.jsonl"), "a").write(json.dumps(dict(perspectiveID=pid, selfPID=rows[pid]["selfPID"], dyadID=rows[pid]["dyadID"], model=model, modality="text", info=version, frame="seat", rep=0, ok=ok, pred=d if ok else None)) + "\n")
def locate_one(p):
    pid, model, version = p["perspectiveID"], p["model"], p["info"]; path = os.path.join(OUT, "located", f"{pid}__{model.split('/')[1]}__{version}.json")
    if os.path.exists(path) or spent[0] > CAP: return
    rec = json.load(open(os.path.join(OUT, f"{pid}__{version}.json"))); r = rows[pid]
    tx = "\n".join(f"[{k}] ({int(t['start'] // 60)}:{int(t['start'] % 60):02d}) {'You' if t['spk'] == r['selfPID'] else 'Partner'}: {t['text']}" for k, t in enumerate(rec["turns"]))
    c = orc.call(LM.MODEL, [{"role": "user", "content": LM.PROMPT.format(tx=tx, mem=str(p["pred"].get("moment", "")), why=str(p["pred"].get("reason", "")))}],
                 os.path.join(OUT, "locate_calls.jsonl"), tag=f"{pid}|{model}|{version}", max_tokens=3000, temperature=0, response_format={"type": "json_object"}, reasoning={"effort": "low"}); cost(c)
    o = pj(c.get("text")) or {}
    idx = [i for i in o.get("turns", []) if isinstance(i, int) and 0 <= i < len(rec["turns"])]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(dict(pid=pid, model=model, version=version, kind=o.get("kind"), turns=idx, n_turns=len(rec["turns"]),
                   moved=[k for k, t in enumerate(rec["turns"]) if isinstance(t.get("src"), int) and rec["i"] <= t["src"] <= rec["j"]]), open(path, "w"))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--retry", action="store_true"); a = ap.parse_args(); RETRY = a.retry
    B = bank(); B = B[:a.limit] if a.limit else B
    if a.stage == "edit":
        with ThreadPoolExecutor(8) as ex: list(ex.map(edit_one, B))
        R = [json.load(open(os.path.join(OUT, f"{b['pid']}__{v}.json"))) for b in B for v in ("swap", "control") if os.path.exists(os.path.join(OUT, f"{b['pid']}__{v}.json"))]
        for v in ("swap", "control"):
            rv = [r for r in R if r["version"] == v]
            print(v, "edited", len(rv), "passed code checks", sum(not r["code_problems"] for r in rv), "faithful", sum(r["faithful"] for r in rv))
        print("bank", len(B), "spent $", round(spent[0], 2))
    if a.stage == "predict":
        DONE = {f"{p['perspectiveID']}|{p['model']}|text|{p['info']}|seat|0" for p in map(json.loads, open(os.path.join(RUN, "preds.jsonl")))} if os.path.exists(os.path.join(RUN, "preds.jsonl")) else set()
        ok = [b["pid"] for b in B if all(os.path.exists(os.path.join(OUT, f"{b['pid']}__{v}.json")) and json.load(open(os.path.join(OUT, f"{b['pid']}__{v}.json")))["faithful"] for v in ("swap", "control"))]
        jobs = [(pid, v, m) for pid in ok for v in ("swap", "control") for m in M5]
        with ThreadPoolExecutor(24) as ex: list(ex.map(predict_one, jobs))
        print("calls usable", len(ok), "predictions", len(jobs), "spent $", round(spent[0], 2))
    if a.stage == "locate":
        P_ = [p for p in map(json.loads, open(os.path.join(RUN, "preds.jsonl"))) if p["ok"]]
        with ThreadPoolExecutor(16) as ex: list(ex.map(locate_one, P_))
        print("located", len(P_), "spent $", round(spent[0], 2))
