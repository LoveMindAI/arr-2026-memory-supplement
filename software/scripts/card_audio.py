#!/usr/bin/env python3
"""Personality x audio (A1's queue, Oct 7). Six omni models, 300-memory sample, narrative card.
(1) Does the card cost as much when the model also hears the call? paired card-vs-none within each modality (text; text+audio).
(2) Does hearing the call change how strongly the card steers? coupling CV R^2 (traits on the card -> the model's ratings) for card text vs card text+audio.
Writes analysis/card_audio.json."""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figs_models as FM, trait_steering as T
ROOT = FM.ROOT
S = {}
for r in ["main_text_cheap", "main_text_pro", "main_media_cheap", "main_media_pro", "cards_narr_cheap", "cards_narr_pro", "card_x_audio"]:
    s, _ = FM.load(r); S.update(s)
pro = set(json.load(open(os.path.join(ROOT, "runs", "MAIN_SETS.json")))["pro300"])
OMNI = ["mimo-v2.5", "mimo-v2.6-flash", "qwen3.8-omni-flash", "gemini-3.8-flash", "gemini-3.1-pro-preview", "mimo-v2.6-pro-ultraspeed"]
rng = np.random.default_rng(0); res = {"cost": {}, "coupling": {}}


def paired(model, a, b, m):  # a, b = (modality, info)
    d = []; who = []
    for pid in pro:
        x, y = S.get((model, a[0], a[1], pid)), S.get((model, b[0], b[1], pid))
        if x and y:
            u, v = FM.avg2(x, m), FM.avg2(y, m)
            if u is not None and v is not None:
                d.append(v - u); who.append(pid)
    d = np.array(d)
    if len(d) < 20:
        return None
    lo, hi = FM.cboot(d, who, rng)   # Oct 8: bootstrap over people
    return dict(mean=float(d.mean()), lo=lo, hi=hi, n=len(d))


for mod in ("text", "text_audio"):
    for m in ("same_focus", "similarity"):
        cs = [c for c in (paired(x, (mod, "none"), (mod, "card_narr"), m) for x in OMNI) if c]
        res["cost"][f"{mod}|{m}"] = dict(models=len(cs), mean=float(np.mean([c["mean"] for c in cs])), up=sum(c["lo"] > 0 for c in cs), down=sum(c["hi"] < 0 for c in cs), n=[c["n"] for c in cs])
        print(f"card vs none, {mod:10s} {m:11s}: mean {res['cost'][f'{mod}|{m}']['mean']:+.3f}  up {res['cost'][f'{mod}|{m}']['up']} down {res['cost'][f'{mod}|{m}']['down']}  models {len(cs)}")
P = collections.defaultdict(dict)
for r in ["cards_narr_cheap", "cards_narr_pro", "card_x_audio"]:
    for l in open(os.path.join(ROOT, "runs", r, "preds.jsonl")):
        j = json.loads(l); mdl = j["model"].split("/")[1]
        if j.get("ok") and j["info"] == "card_narr" and mdl in OMNI and j["perspectiveID"] in pro:
            P[(mdl, j["modality"])][j["perspectiveID"]] = T.feats(j["pred"].get("moment", ""), j["pred"])
for mod in ("text", "text_audio"):
    v = collections.defaultdict(list)
    for mdl in OMNI:
        t = T.r2_table(P.get((mdl, mod), {}))
        for o in ("dominance", "pleasure", "arousal"):
            if o in t:
                v[o].append(t[o]["cv_r2"])
    res["coupling"][mod] = {o: float(np.median(x)) for o, x in v.items()}
    print("coupling", mod, {o: round(x, 3) for o, x in res["coupling"][mod].items()}, "models", len(v["dominance"]))
json.dump(res, open(os.path.join(ROOT, "analysis", "card_audio.json"), "w"), indent=1)
