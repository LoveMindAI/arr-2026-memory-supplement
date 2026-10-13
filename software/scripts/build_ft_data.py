#!/usr/bin/env python3
"""Build SFT data for the memory fine-tune (text, info=none, seat frame) in the reader-lane trainer format.
Rows: {"messages": [...], "completion": "<human JSON answer>", "person_key": PID, "perspectiveID": ..., "split": ...}
Split by BATCH (PARSEL batches are closed round-robins), so no person or partner crosses splits.
Seeded: test ~20% of batches, val ~8%, train the rest. Excludes the near-empty call PID_n+PID_n.
The completion reproduces the person's actual survey answers in the schema the prompt asks for."""
import json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finetune", "data_v1")
os.makedirs(OUT, exist_ok=True)
EXCLUDE_DYADS = {"PID_n+PID_n"}


def flat(msgs):
    out = []
    for m in msgs:
        c = m["content"]
        if not isinstance(c, str):
            assert all(p["type"] == "text" for p in c), "text-only fine-tune"
            c = "\n\n".join(p["text"] for p in c)
        out.append({"role": m["role"], "content": c})
    return out


def num(x, nd=None):
    try:
        v = float(x)
    except Exception:
        return None
    return round(v, nd) if nd is not None else int(round(v))


def completion(r):
    d = {"moment": r["conv_SP_Mem_descr"].strip(), "reason": r["conv_SP_Mem_reason"].strip(), "timing": num(r["conv_SP_Mem"]),
         "getting_along": num(r["conv_TP_Mem_sc2"]), "minimize_conflict": num(r["conv_TP_Mem_sc3"]), "unit": num(r["conv_TP_Mem_ent1"]),
         "pleasure": num(r["conv_SP_Mem_p"], 2), "arousal": num(r["conv_SP_Mem_a"], 2), "dominance": num(r["conv_SP_Mem_d"], 2)}
    return json.dumps(d, ensure_ascii=False)


def main(seed=2026):
    rows = [r for r in P.rows() if r["dyadID"] not in EXCLUDE_DYADS]
    batches = sorted({r["batchID"] for r in rows})
    rng = random.Random(seed); rng.shuffle(batches)
    n = len(batches); nt = round(n * 0.20); nv = round(n * 0.08)
    split = {b: ("test" if i < nt else "val" if i < nt + nv else "train") for i, b in enumerate(batches)}
    out = defaultdict(list)
    for r in rows:
        msgs = flat(P.build(r, "text", "none", "seat"))
        out[split[r["batchID"]]].append(dict(messages=msgs, completion=completion(r), person_key=r["selfPID"],
                                             perspectiveID=r["perspectiveID"], batchID=r["batchID"], split=split[r["batchID"]]))
    for k, v in out.items():
        with open(os.path.join(OUT, f"memory_none_{k}.jsonl"), "w") as f:
            for x in v:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    meta = {k: dict(rows=len(v), people=len({x["person_key"] for x in v}), batches=sorted({x["batchID"] for x in v})) for k, v in out.items()}
    # leakage proof: no person in two splits, no partner in two splits
    people = defaultdict(set)
    for k, v in out.items():
        for x in v:
            people[x["person_key"]].add(k)
    meta["people_in_multiple_splits"] = [p for p, s in people.items() if len(s) > 1]
    meta["seed"] = seed
    json.dump(meta, open(os.path.join(OUT, "SPLIT.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "batches"}) for k, v in meta.items()}, indent=1))


if __name__ == "__main__":
    main()
