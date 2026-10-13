#!/usr/bin/env python3
"""DPO pairs for history-conditioned memory prediction (FSPO-style: the person's own other memories are in context).
Split = the same batch split as SFT v1 (finetune/data_v1/SPLIT.json). Prompt = text + info=history, seat frame (flattened).
chosen  = the person's actual answer (same JSON as SFT targets).
rejected (arm A, "wrong person")       = the PARTNER's answer about the same call (same conversation, human-written, same length class).
rejected (arm B, "wrong conversation") = the SAME person's answer about a different call (right person, wrong call).
Writes data_v2_dpo/{pairs_A,pairs_B}_train.jsonl and eval prompts (history) for val/test, plus length stats.
Use with reader-lane dpo_worker.py --len-norm (per-token scoring) so length cannot buy margin."""
import json, os, random, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import prompts as P
from build_ft_data import flat, completion
OUT = os.path.join(HERE, "data_v2_dpo"); os.makedirs(OUT, exist_ok=True)


def main():
    split_meta = json.load(open(os.path.join(HERE, "data_v1", "SPLIT.json")))
    bsplit = {b: k for k in ("train", "val", "test") for b in split_meta[k]["batches"]}
    rows = [r for r in P.rows() if r["dyadID"] != "PID_n+PID_n"]
    by_pid = {r["perspectiveID"]: r for r in rows}
    by_person = defaultdict(list)
    for r in rows:
        by_person[r["selfPID"]].append(r)
    rng = random.Random(2026)
    A, B, ev = [], [], defaultdict(list)
    lens = defaultdict(list)
    for r in rows:
        sp = bsplit[r["batchID"]]
        msgs = flat(P.build(r, "text", "history", "seat"))
        if sp != "train":
            ev[sp].append(dict(messages=msgs, perspectiveID=r["perspectiveID"], split=sp, person_key=r["selfPID"]))
            continue
        ch = completion(r)
        partner = by_pid.get(f"{r['otherPID']}+{r['selfPID']}")
        if partner:
            rj = completion(partner); A.append(dict(messages=msgs, chosen=ch, rejected=rj, perspectiveID=r["perspectiveID"], arm="A_partner"))
            lens["A"].append(len(rj) / max(1, len(ch)))
        others = [x for x in by_person[r["selfPID"]] if x["perspectiveID"] != r["perspectiveID"]]
        if others:
            o = rng.choice(others); rj = completion(o)
            B.append(dict(messages=msgs, chosen=ch, rejected=rj, perspectiveID=r["perspectiveID"], arm="B_other_call"))
            lens["B"].append(len(rj) / max(1, len(ch)))
    for name, data in (("pairs_A_train.jsonl", A), ("pairs_B_train.jsonl", B)):
        with open(os.path.join(OUT, name), "w") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    for sp, data in ev.items():
        with open(os.path.join(OUT, f"eval_history_{sp}.jsonl"), "w") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import numpy as np
    stats = {k: dict(n=len(v), median_len_ratio=float(np.median(v)), iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))]) for k, v in lens.items()}
    stats.update({f"eval_{k}": len(v) for k, v in ev.items()})
    json.dump(stats, open(os.path.join(OUT, "STATS.json"), "w"), indent=1); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
