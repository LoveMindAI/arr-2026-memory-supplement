#!/usr/bin/env python3
"""Score the memory DPO runs (plan: DPO_PLAN_v2.md). Reads run_dpo1/eval_out/{base_history,base_card,dpo_*}.jsonl.
1) parse each answer (JSON) -> runs/dpo1_eval/preds.jsonl (model = gemma4-31b:<tag>, info = history | card_narr); report parse rate and
   moment length per tag (the collapse check);
2) judge with judge.py (GPT-6-Luna + DeepSeek v4.1 Flash; GLM is not involved: no DeepSeek predictions here);
3) paired DPO - base on the same held-out memories (test, and val+test), mean of the three judges, bootstrap 95% CI, per metric;
4) card runs: trait-to-behavior coupling (CV R^2 of the model's ratings from the card's traits), base_card vs dpo_card_*, on val+test.
Writes analysis/dpo_results.json."""
import json, os, subprocess, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", ".."); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import figs_models as FM, trait_steering as T, prompts as P
EV = os.path.join(HERE, "run_dpo1", "eval_out"); RD = os.path.join(ROOT, "runs", "dpo1_eval"); os.makedirs(RD, exist_ok=True)
TAGS = {"base_history": "history", "base_card": "card_narr", "dpo_history_lr2e-05": "history", "dpo_history_lr5e-05": "history",
        "dpo_card_lr2e-05": "card_narr", "dpo_card_lr5e-05": "card_narr"}


def parse(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def main():
    rows = {r["perspectiveID"]: r for r in P.rows()}; out = []; stats = {}
    for tag, info in TAGS.items():
        p = os.path.join(EV, tag + ".jsonl")
        if not os.path.exists(p):
            print("missing", tag); continue
        L = [json.loads(l) for l in open(p)]; ok = 0; words = []
        for x in L:
            d = parse(x["text"]); r = rows[x["perspectiveID"]]
            if d and str(d.get("moment", "")).strip():
                ok += 1; words.append(len(str(d["moment"]).split()))
            out.append(dict(perspectiveID=x["perspectiveID"], selfPID=r["selfPID"], dyadID=r["dyadID"], model=f"gemma4-31b/{tag}", modality="text",
                            info=info, frame="seat", rep=0, ok=bool(d and str(d.get("moment", "")).strip()), pred=d or {}, split=x.get("split"),
                            finish=x.get("finish")))
        stats[tag] = dict(n=len(L), parse_rate=ok / max(1, len(L)), moment_words_median=float(np.median(words)) if words else None,
                          finish_length=sum(x.get("finish") == "length" for x in L))
        print(tag, stats[tag])
    with open(os.path.join(RD, "preds.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    subprocess.run(["python3", "judge.py", "--run", "dpo1_eval", "--conc", "64"], cwd=os.path.join(ROOT, "scripts"),
                   env=dict(os.environ, JUDGE_MAX_TOKENS="12000"), check=False)
    S, R = FM.load("dpo1_eval")
    split = {o["perspectiveID"]: o["split"] for o in out}
    rng = np.random.default_rng(0); res = {"stats": stats, "paired": {}, "levels": {}, "coupling": {}}
    for tag, info in TAGS.items():
        if tag.startswith("base"): continue
        base = "base_history" if info == "history" else "base_card"
        for sp in ("test", "all"):
            for m in ("similarity", "same_focus", "same_reason", "event_match"):
                d = []; who = []
                for pid in split:
                    if sp == "test" and split[pid] != "test": continue
                    x, y = S.get((base, "text", info, pid)), S.get((tag, "text", info, pid))
                    if x and y:
                        u, v = FM.avg2(x, m), FM.avg2(y, m)
                        if u is not None and v is not None: d.append(v - u); who.append(pid)
                d = np.array(d)
                if len(d) >= 20:
                    lo_, hi_ = FM.cboot(d, who, rng)   # Oct 8: bootstrap over people
                    bs = [lo_, hi_]
                    res["paired"][f"{tag}|{sp}|{m}"] = dict(n=len(d), mean=float(d.mean()), lo=bs[0], hi=bs[1])
    for tag, info in TAGS.items():
        for m in ("similarity", "same_focus", "same_reason", "event_match"):
            v = [FM.avg2(S[k], m) for k in S if k[0] == tag]; v = [x for x in v if x is not None]
            res["levels"][f"{tag}|{m}"] = float(np.mean(v)) if v else None
    for tag in ("base_card", "dpo_card_lr2e-05", "dpo_card_lr5e-05"):
        F = {o["perspectiveID"]: T.feats(o["pred"].get("moment", ""), o["pred"]) for o in out if o["model"].endswith(tag) and o["ok"]}
        t = T.r2_table(F); res["coupling"][tag] = {o: t[o]["cv_r2"] for o in ("pleasure", "arousal", "dominance") if o in t}
    json.dump(res, open(os.path.join(ROOT, "analysis", "dpo_results.json"), "w"), indent=1)
    for k, v in res["paired"].items():
        print(f"{k:45s} {v['mean']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] n={v['n']}")
    print("coupling", res["coupling"])


if __name__ == "__main__":
    main()
