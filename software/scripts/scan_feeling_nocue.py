#!/usr/bin/env python3
"""A1 (Oct 9): find memories about feelings or the vibe that have no clear cue in the transcript, as candidates for what only audio/video
would catch, and ask whether hearing/seeing the call helps models more on those. Flags (both coders must agree on the topic):
FEELING_VIBE = Haiku and Luna both tag topic 'feelings or mood' or 'how the partner came across'; NO_CUE = the locator found no specific exchange
(kind general / not_in_transcript, or low confidence). Media gain = media minus text, same model, same person info (none), mean of three judges,
on the 300-memory media sample; bootstrap over people. -> analysis/feeling_nocue_scan.json"""
import collections, glob, json, os
import numpy as np
import figs_models as FM
ROOT = ".."; up = lambda x: str(x).upper()
def codes(c, pid):
    p = os.path.join(ROOT, "analysis", "code_v351", "human", f"{pid}__{c}.json")
    j = json.load(open(p)) if os.path.exists(p) else {}
    return j.get("codes") or {}
flags = {}
for f in glob.glob(os.path.join(ROOT, "analysis", "located", "*.json")):
    L = json.load(open(f)); pid = L["perspectiveID"]
    tv = [set(up(x) for x in ((codes(c, pid).get("topic") or []) if isinstance(codes(c, pid).get("topic"), list) else [codes(c, pid).get("topic")])) for c in ("claude-haiku-5.5", "gpt-6-luna")]
    fv = all(t & {"FEELINGS_MOOD", "PARTNER_IMPRESSION"} for t in tv)
    nc = L.get("kind") in ("general", "not_in_transcript") or L.get("confidence") == "low"
    flags[pid] = dict(feeling_vibe=fv, no_cue=nc)
cnt = collections.Counter((v["feeling_vibe"], v["no_cue"]) for v in flags.values())
print("all memories (feeling_vibe, no_cue):", dict(cnt))
S = {}
for r in ["main_text_cheap", "main_text_pro", "main_media_cheap", "main_media_pro"]:
    s, _ = FM.load(r); S.update(s)
pro = set(json.load(open(os.path.join(ROOT, "runs", "MAIN_SETS.json")))["pro300"]) | set(json.load(open(os.path.join(ROOT, "analysis", "feeling_nocue_pids.json")))["new_for_media"])   # + the 59 run Oct 9
models = sorted({k[0] for k in S if k[1] != "text"}); rng = np.random.default_rng(0)
grp = lambda pid: "feeling_vibe_no_cue" if flags.get(pid, {}).get("feeling_vibe") and flags[pid]["no_cue"] else ("feeling_vibe_cued" if flags.get(pid, {}).get("feeling_vibe") else "other")
res = {"counts_all": {f"{k[0]}|{k[1]}": v for k, v in cnt.items()}, "counts_media_sample": dict(collections.Counter(grp(p) for p in pro)), "gain": {}}
for mod in ("audio", "text_audio", "text_audio_video"):
    for m in ("similarity", "same_focus", "event_match"):
        d = collections.defaultdict(list); who = collections.defaultdict(list)
        for mdl in models:
            for pid in pro:
                x, y = S.get((mdl, "text", "none", pid)), S.get((mdl, mod, "none", pid))
                if x and y:
                    u, v = FM.avg2(x, m), FM.avg2(y, m)
                    if u is not None and v is not None: g = grp(pid); d[g].append(v - u); who[g].append(pid)
        for g in d:
            lo, hi = FM.cboot(np.array(d[g]), who[g], rng)
            res["gain"][f"{mod}|{m}|{g}"] = dict(mean=round(float(np.mean(d[g])), 3), lo=round(lo, 3), hi=round(hi, 3), n=len(d[g]), people=len(set(who[g])))
json.dump(res, open(os.path.join(ROOT, "analysis", "feeling_nocue_scan.json"), "w"), indent=1)
print("pro300 groups:", res["counts_media_sample"])
for k, v in res["gain"].items(): print(f"{k:50s} {v['mean']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] n={v['n']} people={v['people']}")
