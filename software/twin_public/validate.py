#!/usr/bin/env python3
"""Validation of the public stand-in against the ONLY targets it was built to (A1, Oct 11): memory lengths, call shape, and the two
memory types' shares and defining profiles. Nothing else is compared, because nothing else was tuned. Reads stand-in files and
targets_from_parsel.json only. -> VALIDATION.md, validation.json"""
import collections, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); T = json.load(open(os.path.join(HERE, "targets_from_parsel.json")))
PEOPLE = json.load(open(os.path.join(HERE, "people.json")))
q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)} if len(a) else {}
mem = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "memories", "C*.json")))]; mem = [m for m in mem if m.get("ok")]; mem_av = [m for m in mem if m.get("requires_av")]   # the audio/video-only subset is reported, not counted in the type profiles
calls = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "calls", "C*.json")))]; calls = [c for c in calls if c.get("ok")]
wm = [len(str(m["answers"]["moment"]).split()) for m in mem]; wr = [len(str(m["answers"]["reason"]).split()) for m in mem]
def split(key):
    by = collections.defaultdict(list)
    for m in mem: by[m["person"]].append(np.log1p(len(str(m["answers"][key]).split())))
    v = [x for x in by.values() if len(x) >= 2]
    return dict(between_person_sd=float(np.std([np.mean(x) for x in v])) if v else None, within_person_sd=float(np.sqrt(np.mean([np.var(x) for x in v]))) if v else None)
res = dict(n_memories=len(mem), n_calls=len(calls),
           moment_words=dict(standin=q(wm), target=T["memory_moment_words"]["raw_quantiles"], standin_split=split("moment"),
                             target_split={k: T["memory_moment_words"][k] for k in ("between_person_sd", "within_person_sd")}),
           reason_words=dict(standin=q(wr), target=T["memory_reason_words"]["raw_quantiles"], standin_split=split("reason"),
                             target_split={k: T["memory_reason_words"][k] for k in ("between_person_sd", "within_person_sd")}),
           turns_per_call=dict(standin=q([c["turns"] for c in calls]), target=T["call_shape"]["turns_per_call"]),
           words_per_call=dict(standin=q([c["words"] for c in calls]), target=T["call_shape"]["words_per_call"]))
# types: share of people per type, and each type's coded profile (mean of the two coders) against the target profile
codes = collections.defaultdict(dict)
for f in glob.glob(os.path.join(HERE, "coding", "C*__*.json")):
    j = json.load(open(f))
    if j.get("ok") and j.get("codes"): codes[j["key"]][j["coder"]] = j["codes"]
def feats(c):
    ext = c.get("extent"); rem = {str(x).lower() for x in (c.get("remembered") or [])}; ni = c.get("new_info")
    return {"scope:one_moment": float(ext == "ONE_MOMENT"), "scope:stretch": float(ext == "STRETCH"), "what:did_together": float("did_together" in rem),
            "what:partner_said_or_did": float("partner_said_or_did" in rem), "new_fact": float(ni == "YES")}
prof = {}
for t in ("shared_stretch", "what_they_were_told"):
    rows = [np.mean([list(feats(c).values()) for c in codes[f"{m['call']}_{m['person']}"].values()], axis=0) for m in mem
            if m["memory_type"] == t and not m.get("requires_av") and m.get("spec") and codes.get(f"{m['call']}_{m['person']}")]
    names = list(feats({}).keys())
    prof[t] = dict(n_memories_coded=len(rows), standin={n: float(v) for n, v in zip(names, np.mean(rows, axis=0))} if rows else {},
                   target=T["memory_types"][t]["profile"], share_people=float(np.mean([p["memory_type"] == t for p in PEOPLE.values()])),
                   target_share=T["memory_types"][t]["share"])
res["memory_types"] = prof
json.dump(res, open(os.path.join(HERE, "validation.json"), "w"), indent=1)
fmt = lambda d: ", ".join(f"{k} {v:.0f}" for k, v in d.items())
with open(os.path.join(HERE, "VALIDATION.md"), "w") as f:
    f.write(f"# Validation of the stand-in (validate.py)\n\nCompared ONLY on what it was built to match: lengths, call shape and the two memory types. "
            f"{len(mem)} memories, {len(calls)} calls.\n\n## Lengths (word quantiles)\n")
    for k in ("moment_words", "reason_words", "turns_per_call", "words_per_call"):
        f.write(f"- **{k}**: stand-in {fmt(res[k]['standin'])}; target {fmt({x: v for x, v in res[k]['target'].items() if x in res[k]['standin']})}\n")
    for k in ("moment_words", "reason_words"):
        s, t = res[k]["standin_split"], res[k]["target_split"]
        if s["between_person_sd"] is not None: f.write(f"- {k}, log-length SD between people {s['between_person_sd']:.2f} (target {t['between_person_sd']:.2f}), within {s['within_person_sd']:.2f} (target {t['within_person_sd']:.2f})\n")
    f.write("\n## Memory types (share of people; coded profile of each type's memories, mean of Haiku and Luna)\n")
    for t, v in prof.items():
        f.write(f"- **{t}**: {v['share_people']:.2f} of people (target {v['target_share']:.2f}); {v['n_memories_coded']} coded memories\n")
        for n, tv in v["target"].items(): f.write(f"  - {n}: {v['standin'].get(n, float('nan')):.2f} (target {tv:.2f})\n")
if os.path.exists(os.path.join(HERE, "av_summary.json")):
    a = json.load(open(os.path.join(HERE, "av_summary.json"))); k = a["locator_on_kept"]
    open(os.path.join(HERE, "VALIDATION.md"), "a").write(
        f"\n## Audio/video-only subset\nOne call per session, chosen at random by design (seed in av_plan.py), carries one non-verbal event on one person's side, written into a hidden track "
        f"(av_events.json) and kept out of the transcript; the other person's memory is about it (requires_av: true). Of {a['planned']} planned, {a['dropped_for_leak']} were dropped from the subset "
        f"because a judge (Gemini 3 Flash preview) found the transcript hinting at the event; {a['kept']} remain. On those, the paper's text-only locator (prompt v1, Claude Sonnet 5.5) said "
        f"not_in_transcript for {k['not_in_transcript']}, general for {k['general']}, and placed {k['in_conversation']} in the conversation; the placed ones are mostly a face or voice changing "
        f"during a spoken line, where the words give the timing but not what was noticed (av_check.json has each locator note). A keyword scan of the event description against the transcript "
        f"flagged common words in nearly every call and was not informative.\n")
if os.path.exists(os.path.join(HERE, "CHECKPOINT.md")): open(os.path.join(HERE, "VALIDATION.md"), "a").write("\n" + open(os.path.join(HERE, "CHECKPOINT.md")).read())
print(json.dumps({k: res[k] for k in ("n_memories", "n_calls")}))
