#!/usr/bin/env python3
# Ties in typical() follow the neutral rule in tie_rule.py (order-free).
"""Item 1 (A1, Oct 9): one ladder of predictors on the codebook scale, per question, coder-matched where the predictor is a text.
Rungs: chance (random batch-mate's memory, another call) | most common human answer (all other people) | the person's own EARLIER reports
(their typical answer: mode / majority labels / median) | partner's memory of the same call | models: no history, history, earlier-calls history,
norms note, norms + history (the last two exist for the held-out half only) | people predicting from the transcript (added when files arrive).
Common set: reports with at least one earlier report by the same person. Two panels: all such reports, and the held-out half.
Scores as correspondence_v351.score; means with person-clustered 95% intervals. Two numbers per rung: raw closeness, and EXCESS =
closeness to this person minus mean closeness to other people's memories from the same batch (other calls). A constant guess (e.g. the most
common answer) has excess 0 by construction, so excess measures person-level prediction, raw closeness mixes it with base rates.
-> analysis/ladder_v351.json"""
import collections, json, os
import numpy as np
src = open("correspondence_v351.py").read().split("res = {")[0]; exec(src)
_tp = os.path.join(ROOT, "analysis", "norms_test_pids.json")   # held-out half for the norms note; absent outside PARSEL (e.g. the stand-in)
test = set(json.load(open(_tp))) if os.path.exists(_tp) else set()
import tie_rule
def typical(vals, k): return tie_rule.typical(vals, k, ORD)   # neutral tie rule (Oct 12)
score = tie_rule.wrap_score(score)
SETS = {"model: no history": ["main_text_cheap__none"], "model: history": ["main_text_cheap__history"],
        "model: earlier-calls history": [f"r3c_{m}__history_prior" for m in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")],
        "model: norms note": ["norms_note_v2__norms"], "model: norms + history": ["norms_history_v2__norms_history"]}
out = {}
for coder in CODERS:
    H = load("human", coder); out[coder] = {"all": {}, "held_out_half": {}}
    by_person = collections.defaultdict(list); by_batch = collections.defaultdict(list)
    for pid in H: by_person[rows[pid]["selfPID"]].append(pid); by_batch[rows[pid]["batchID"]].append(pid)
    earlier = {pid: [o for o in by_person[rows[pid]["selfPID"]] if int(rows[o]["roundIdx"]) < int(rows[pid]["roundIdx"])] for pid in H}
    common = [pid for pid in H if earlier[pid]]
    preds = {}
    for name, sets in SETS.items():
        d = collections.defaultdict(list)
        for sd in sets:
            for key, v in load(sd, coder).items(): d[key.split("|")[0]].append(v)
        preds[name] = d
    for panel, pids in (("all", common), ("held_out_half", [p for p in common if p in test])):
        R = out[coder][panel]
        for k in KEYS:
            rung = collections.defaultdict(lambda: ([], [])); ex = collections.defaultdict(lambda: ([], []))
            pop = {}
            def add(name, pred, person, pid, h, exclude=()):
                s = score(k, h, pred)
                if s is None: return
                rung[name][0].append(s); rung[name][1].append(person)
                oth = [o for o in by_batch[rows[pid]["batchID"]] if rows[o]["dyadID"] != rows[pid]["dyadID"] and o not in exclude and rows[o]["selfPID"] != rows[pid]["selfPID"]]  # Oct 9: exclude own other memories (A1 OK)
                c = [score(k, H[o][k], pred) for o in oth]; c = [x for x in c if x is not None]
                if c: ex[name][0].append(s - float(np.mean(c))); ex[name][1].append(person)
            for pid in pids:
                h = H[pid][k]
                if h is None: continue
                person = pid.split("+")[0]
                others = [o for o in by_batch[rows[pid]["batchID"]] if rows[o]["dyadID"] != rows[pid]["dyadID"]]
                c = [score(k, h, H[o][k]) for o in others]; c = [x for x in c if x is not None]
                if c: rung["chance"][0].append(float(np.mean(c))); rung["chance"][1].append(person)
                if k not in pop: pop[k] = typical([H[o][k] for o in H if rows[o]["selfPID"] != rows[pid]["selfPID"]], k)
                add("most common human answer", pop[k], person, pid, h)
                add("person's own earlier reports", typical([H[o][k] for o in earlier[pid]], k), person, pid, h)
                r = rows[pid]; pp = f"{r['otherPID']}+{r['selfPID']}"
                if pp in H: add("partner's memory", H[pp][k], person, pid, h, exclude=(pp,))
                for name in SETS:
                    for p in preds[name].get(pid, []): add(name, p[k], person, pid, h)
            R[k] = {name: dict(raw=cboot(v, w), excess=cboot(*ex[name]) if len(ex[name][0]) > 20 else None) for name, (v, w) in rung.items() if len(v) > 20}
json.dump(out, open(os.path.join(ROOT, "analysis", "ladder_v351" + ("_comply" if os.environ.get("COMPLY_MAX3") == "1" else "") + ".json"), "w"), indent=1)
for coder in CODERS:
    for panel in ("all",):
        print("==", coder, panel)
        for k, d in out[coder][panel].items():
            print(f"  {k:24s} " + " | ".join(f"{n.replace('model: ','M:')[:16]} {v['excess'][0]:+.3f}" for n, v in d.items() if v.get('excess') and n != 'chance'))
