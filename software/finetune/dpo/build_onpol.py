#!/usr/bin/env python3
"""On-policy memory DPO (A1, Oct 8: "worth running it overnight"). Aim (b) of the paper: can a model learn to remember like one person.
Rejected side = the UNTUNED model's own guess for that memory (on-policy), instead of another human's answer.
  gen   : write ../data_v4_onpol/gen_train_s{0..3}.jsonl (the 845 history-condition training prompts, 4 seeds each) + copy val/test eval files
  pairs : read the pod's samples (run_onpol/eval_out/gen_s*.jsonl), judge each against the person's real memory with the paper's two judges
          (GPT-6-Luna + DeepSeek v4.1 Flash, judge.py prompt), keep as REJECTED only samples BOTH judges call a different moment
          (event_match 0) -- a correct guess must never be pushed away. Up to 2 rejects per memory.
          Writes pairs_onpol_train.jsonl (on-policy only) and pairs_mixed_train.jsonl (1 on-policy + 1 original human pair per memory).
Chosen = the person's real answer (same text as the first DPO runs). Rejected = the guess re-serialized in the chosen answer's exact JSON format. All pairs keep the NLL anchor (chosen is off-policy)."""
import json, os, random, shutil, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import judge as J
V3 = os.path.join(HERE, "..", "data_v3_dpo"); V4 = os.path.join(HERE, "..", "data_v4_onpol"); os.makedirs(V4, exist_ok=True)
RUN = os.path.join(HERE, "run_onpol"); JUDGES = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"]


def train_rows():
    rows = {}
    for l in open(os.path.join(V3, "pairs_history_train.jsonl")):
        r = json.loads(l); rows.setdefault(r["perspectiveID"], dict(messages=r["messages"], chosen=r["chosen"], orig=[]))["orig"].append(r)
    return rows


def gen():
    R = train_rows()
    for k in range(4):
        with open(os.path.join(V4, f"gen_train_s{k}.jsonl"), "w") as f:
            for pid, r in sorted(R.items()):
                f.write(json.dumps(dict(messages=r["messages"], perspectiveID=f"{pid}#s{k}", split="train", person_key=pid)) + "\n")
    for s in ("val", "test"):
        shutil.copy(os.path.join(V3, f"eval_history_{s}.jsonl"), V4)
    print("prompts", len(R), "x 4 seeds")


def parse(t):
    s, e = (t or "").find("{"), (t or "").rfind("}")
    try:
        d = json.loads(t[s:e + 1]); return d if str(d.get("moment", "")).strip() else None
    except Exception:
        return None


def same_format(d, chosen):
    """The model's guess in exactly the human answer's JSON format (same keys, order and json.dumps style), so the only
    difference between chosen and rejected is content, not code fences or key order (a free DPO shortcut otherwise)."""
    keys = list(json.loads(chosen).keys())
    if not all(k in d for k in keys):
        return None
    return json.dumps({k: d[k] for k in keys})


def pairs():
    R = train_rows(); S = []
    for k in range(4):
        for l in open(os.path.join(RUN, "eval_out", f"gen_s{k}.jsonl")):
            x = json.loads(l); pid = x["perspectiveID"].split("#")[0]; d = parse(x["text"])
            t = same_format(d, R[pid]["chosen"]) if d and x.get("finish") == "stop" else None
            if t:
                S.append(dict(pid=pid, k=k, text=t, d=d))
    print("parsed samples", len(S), "of", 4 * len(R))
    out = os.path.join(RUN, "onpol_judged.jsonl"); done = {}
    if os.path.exists(out):
        for l in open(out):
            z = json.loads(l)
            if z["scores"]: done[(z["pid"], z["k"], z["judge"])] = z["scores"]
    jobs = [(s, j) for s in S for j in JUDGES if (s["pid"], s["k"], j) not in done]
    print("judgements", len(jobs))
    def one(x):
        s, j = x; h = json.loads(R[s["pid"]]["chosen"])
        res, c = J.jcall(j, (h["moment"], h["reason"]), (str(s["d"]["moment"]), str(s["d"].get("reason", ""))), os.path.join(RUN, "onpol_judge_calls.jsonl"), f"{s['pid']}#s{s['k']}|{j}")
        return s, j, res
    with ThreadPoolExecutor(48) as ex, open(out, "a") as f:
        for s, j, res in ex.map(one, jobs):
            f.write(json.dumps(dict(pid=s["pid"], k=s["k"], judge=j, scores=res)) + "\n"); f.flush()
            if res: done[(s["pid"], s["k"], j)] = res
    rng = random.Random(8); P_on, P_mix = [], []; stats = dict(memories=len(R), with_reject=0, all_right=0, rejects=0)
    by = {}
    for s in S:
        by.setdefault(s["pid"], []).append(s)
    for pid, r in sorted(R.items()):
        wrong = [s for s in sorted(by.get(pid, []), key=lambda s: s["k"])
                 if all((pid, s["k"], j) in done and done[(pid, s["k"], j)].get("event_match") == 0 for j in JUDGES)]
        if not wrong:
            stats["all_right" if by.get(pid) else "no_samples"] = stats.get("all_right" if by.get(pid) else "no_samples", 0) + 1
            continue
        stats["with_reject"] += 1
        for s in wrong[:2]:
            P_on.append(dict(messages=r["messages"], chosen=r["chosen"], rejected=s["text"], perspectiveID=pid, arm=f"onpol_s{s['k']}",
                             person_key=f"{pid}|onpol_s{s['k']}", nll=True)); stats["rejects"] += 1
        s = wrong[0]
        P_mix.append(dict(messages=r["messages"], chosen=r["chosen"], rejected=s["text"], perspectiveID=pid, arm=f"onpol_s{s['k']}",
                          person_key=f"{pid}|onpol_s{s['k']}", nll=True))
        P_mix.append(rng.choice(r["orig"]))
    for name, P in (("pairs_onpol_train.jsonl", P_on), ("pairs_mixed_train.jsonl", P_mix)):
        rng.shuffle(P)
        assert len({p["person_key"] for p in P}) == len(P), "person_key must be unique per pair"
        with open(os.path.join(V4, name), "w") as f:
            for p in P: f.write(json.dumps(p) + "\n")
    stats.update(onpol_pairs=len(P_on), mixed_pairs=len(P_mix)); json.dump(stats, open(os.path.join(RUN, "PAIRS_STATS.json"), "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    {"gen": gen, "pairs": pairs}[sys.argv[1]]()
