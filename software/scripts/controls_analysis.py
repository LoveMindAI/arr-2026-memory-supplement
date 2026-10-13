#!/usr/bin/env python3
"""Oct 8 controls (A2/agent2 review; A1: "Provide the wrong history, provide the wrong profile ... trivially easy"):
5 round-2 models x 1,147 memories x {history_prior, history_wrong, card_thin_wrong, card_narr_wrong} (runs r3c_*), against the
existing right-person conditions on the same models (none, history, card_thin, card_narr).
(1) right vs wrong person: history - history_wrong, card_thin - card_thin_wrong, card_narr - card_narr_wrong (paired, mean of 2 judges).
(2) wrong person vs nothing: does someone else's material help? history_wrong - none, card_*_wrong - none.
(3) prospective history: history_prior - none, and history (all other calls) - history_prior, on targets with >= 1 earlier call.
(4) coupling on wrong cards: CV R^2 of the model's ratings from the DONOR's traits (on the card) vs from the target person's own traits.
All intervals: bootstrap over people (FM.cboot). Per model, then mean over models with up/down counts. Writes analysis/controls.json."""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figs_models as FM, trait_steering as T, prompts as P
ROOT = FM.ROOT
M5 = ["deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"]
RUNS = ["main_text_cheap", "cards_text_cheap", "cards_narr_cheap"] + [f"r3c_{t}" for t in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")]
MET = ["same_focus", "similarity", "same_reason", "event_match"]
OUTS = ["pleasure", "arousal", "dominance", "getting_along", "unit"]


def paired_sub(S, mdl, a, b, m, rng, keep=None):
    d, who = [], []
    for k in list(S):
        if k[0] != mdl or k[1] != "text" or k[2] != a:
            continue
        pid = k[3]
        if keep is not None and pid not in keep:
            continue
        x, y = S.get((mdl, "text", a, pid)), S.get((mdl, "text", b, pid))
        if x and y:
            u, v = FM.avg2(x, m), FM.avg2(y, m)
            if u is not None and v is not None:
                d.append(v - u); who.append(pid)
    if len(d) < 20:
        return None
    lo, hi = FM.cboot(np.array(d), who, rng)
    return dict(mean=float(np.mean(d)), lo=lo, hi=hi, n=len(d))


def pool(cs):
    cs = [c for c in cs if c]
    return dict(models=len(cs), mean=float(np.mean([c["mean"] for c in cs])) if cs else None,
                up=sum(c["lo"] > 0 for c in cs), down=sum(c["hi"] < 0 for c in cs))


def main():
    S = {}
    for r in RUNS:
        s, _ = FM.load(r); S.update(s)
    rng = np.random.default_rng(0)
    rows = {r["perspectiveID"]: r for r in P.rows()}
    prior = {pid for pid, r in rows.items() if P.history(r, prior_only=True)}
    res = {"n_with_prior": len(prior), "comparisons": {}, "per_model": {}, "coupling_wrong": {}}
    # (b - a) comparisons
    comps = [("right_vs_wrong_history", "history_wrong", "history", None), ("right_vs_wrong_scores", "card_thin_wrong", "card_thin", None),
             ("right_vs_wrong_narr", "card_narr_wrong", "card_narr", None),
             ("wrong_history_vs_none", "none", "history_wrong", None), ("wrong_scores_vs_none", "none", "card_thin_wrong", None),
             ("wrong_narr_vs_none", "none", "card_narr_wrong", None),
             ("prior_history_vs_none", "none", "history_prior", prior), ("all_vs_prior_history", "history_prior", "history", prior),
             ("all_history_vs_none_same_targets", "none", "history", prior)]
    for name, a, b, keep in comps:
        res["per_model"][name] = {}
        res["comparisons"][name] = {}
        for m in MET:
            cs = {mdl: paired_sub(S, mdl, a, b, m, rng, keep) for mdl in M5}
            res["per_model"][name][m] = cs
            res["comparisons"][name][m] = pool(cs.values())
    # coupling on wrong cards: donor traits (on the card) vs own traits
    donor_of = {}
    for rung in ("thin", "narr"):
        for pid, r in rows.items():
            try:
                donor_of[(rung, pid)] = P.donor(r, rung=rung)[0]["selfPID"]
            except Exception:
                pass
    preds = collections.defaultdict(dict)
    for r in RUNS:
        p = os.path.join(ROOT, "runs", r, "preds.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            j = json.loads(l); m = j["model"].split("/")[1]
            if j.get("ok") and m in M5 and j["info"] in ("card_thin", "card_narr", "card_thin_wrong", "card_narr_wrong"):
                preds[(m, j["info"])][j["perspectiveID"]] = T.feats(j["pred"].get("moment", ""), j["pred"])
    for info in ("card_thin", "card_narr", "card_thin_wrong", "card_narr_wrong"):
        rung = info.split("_")[1]
        out = {"card_traits": {o: [] for o in OUTS}, "own_traits": {o: [] for o in OUTS}}
        for mdl in M5:
            D = preds.get((mdl, info), {})
            if info.endswith("_wrong"):   # key by the donor so r2_table looks up the traits that were ON the card
                Dc = {f"{donor_of[(rung, k)]}+{k}": v for k, v in D.items() if (rung, k) in donor_of}
            else:
                Dc = D
            for kind, data in (("card_traits", Dc), ("own_traits", D)):
                t = T.r2_table(data)
                for o in OUTS:
                    if o in t:
                        out[kind][o].append(t[o]["cv_r2"])
        res["coupling_wrong"][info] = {k: {o: (float(np.median(v)) if v else None) for o, v in d.items()} for k, d in out.items()}
    json.dump(res, open(os.path.join(ROOT, "analysis", "controls.json"), "w"), indent=1)
    print("targets with >=1 earlier call:", len(prior))
    for name in res["comparisons"]:
        print(f"{name:34s} " + "  ".join(f"{m[:10]} {c['mean']:+.3f} {c['up']}u/{c['down']}d" for m, c in res["comparisons"][name].items() if c["mean"] is not None))
    for info, d in res["coupling_wrong"].items():
        print(info, "card traits:", {o: (round(v, 2) if v is not None else None) for o, v in d["card_traits"].items()},
              "| own traits:", {o: (round(v, 2) if v is not None else None) for o, v in d["own_traits"].items()})


if __name__ == "__main__":
    main()
