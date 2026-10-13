#!/usr/bin/env python3
"""Round-2 conditions (Oct 7; 5 models x 1,147 memories, each condition alone):
card_facts (bare facts), card_impthin (impression scores), card_weave (woven card), card_factnarr (who-not-how), card_impnarr (impression card).
Compared with the round-1 conditions on the SAME 5 models: none, card_thin, card_narr.
(1) Fidelity lens: paired change vs no information (mean of three judges), mean over models and up/down counts.
(2) A1's lens: trait-to-behavior coupling, CV R^2 of the model's ratings from the traits ON THE CARD (self-report traits for self-report
    cards and facts/weave/factnarr; impression traits for impression cards). Writes analysis/round2.json."""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figs_models as FM, trait_steering as T
ROOT = FM.ROOT
M5 = ["deepseek-v4.1-flash", "gemini-3.8-flash", "gemma-4-31b-it", "qwen3.8-omni-flash", "mimo-v2.6-flash"]
RUNS = ["main_text_cheap", "cards_text_cheap", "cards_narr_cheap", "r2_facts_impthin", "r2_weave_factnarr_impnarr", "r2_weave_factnarr", "r2_history_a", "r2_history_b"]
CONDS = [("card_thin", "Self-report scores", "self"), ("card_narr", "Narrative card", "self"), ("card_impthin", "Impression scores", "imp"),
         ("card_impnarr", "Impression narrative card", "imp"), ("card_facts", "Facts (list)", "self"), ("card_factnarr", "Facts (narrative, who not how)", "self"),
         ("card_weave", "Narrative card with facts woven in", "self")]
MET = ["same_focus", "similarity", "same_reason", "event_match"]
OUTS = ["pleasure", "arousal", "dominance", "getting_along", "unit"]


def main():
    S = {}
    for r in RUNS:
        s, _ = FM.load(r); S.update(s)
    rng = np.random.default_rng(0); res = {"fidelity": {}, "coupling": {}}
    imp = json.load(open(os.path.join(ROOT, "cards", "inputs", "impression_facets.json")))
    sys.path.insert(0, os.path.join(ROOT, "cards")); import score_fidelity as SF
    TR_IMP = {p: np.array([np.mean([v[i] for i in SF.IDS if i[0] == d]) for d in "HEXACO"]) for p, v in imp.items()}
    preds = collections.defaultdict(dict)
    for r in RUNS:
        p = os.path.join(ROOT, "runs", r, "preds.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            j = json.loads(l); m = j["model"].split("/")[1]
            if j.get("ok") and m in M5:
                preds[(m, j["info"])][j["perspectiveID"]] = T.feats(j["pred"].get("moment", ""), j["pred"])
    print(f"{'condition':40s} " + " ".join(f"{m:>22s}" for m in MET[:3]) + "   coupling R2 (dom, pleasure, arousal)")
    for info, name, src in CONDS:
        if not any(k[0] in M5 and k[2] == info for k in S):
            continue
        fid = {}
        for m in MET:
            cs = [c for c in (FM.paired(S, mdl, "text", "none", info, m, rng) for mdl in M5) if c]
            fid[m] = dict(models=len(cs), mean=float(np.mean([c["mean"] for c in cs])) if cs else None,
                          up=sum(c["lo"] > 0 for c in cs), down=sum(c["hi"] < 0 for c in cs))
        res["fidelity"][info] = fid
        backup = T.TR; T.TR = TR_IMP if src == "imp" else backup
        cp = {o: [] for o in OUTS}
        for mdl in M5:
            t = T.r2_table(preds.get((mdl, info), {}))
            for o in OUTS:
                if o in t:
                    cp[o].append(t[o]["cv_r2"])
        T.TR = backup
        res["coupling"][info] = {o: float(np.median(v)) if v else None for o, v in cp.items()}
        f = lambda m: f"{fid[m]['mean']:+.3f} {fid[m]['up']}up/{fid[m]['down']}dn" if fid[m]["mean"] is not None else "--"
        c = res["coupling"][info]; g = lambda o: f"{c[o]:.2f}" if c[o] is not None else "--"
        print(f"{name:40s} " + " ".join(f"{f(m):>22s}" for m in MET[:3]) + f"   {g('dominance')} {g('pleasure')} {g('arousal')}")
    json.dump(res, open(os.path.join(ROOT, "analysis", "round2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
