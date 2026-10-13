#!/usr/bin/env python3
"""Generate coding pages from codebook/spec_v31.py (same words as the model prompt).
  calibrate: 4 invented examples with agent1's answers (reveal after coding)  -> human_coding/CALIBRATION_v35.html
  code:      real reports + transcript part (A1: overlap 50; A3: 100) -> human_coding/A1_CODER_v35.html, A3_CODER_v35.html"""
import json, os, sys
import prompts as P
ROOT = ".."; sys.path.insert(0, os.path.join(ROOT, "codebook")); from spec_v35 import SPEC
TPL = open("coder_gui_v35.html").read()
def write(path, **d):
    d["spec"] = SPEC
    open(path, "w").write(TPL.replace("/*DATA*/null", json.dumps(d, ensure_ascii=False)))
A = lambda v, why="": dict(v=v, why=why)
EX = [dict(id="E1", moment="She told me she has a pet snake called Doris.", reason="I didn't expect that at all!"),
      dict(id="E2", moment="We both laughed about how awkward the start was.", reason="It broke the ice and made me relax."),
      dict(id="E3", moment="When I mentioned I'm a nurse, he said his mum had been in hospital for months last year and how much the nurses helped him get through it.",
           reason="It made me feel proud of what I do, and it was moving that he'd share something that personal with a stranger."),
      dict(id="E4", moment="We talked about the study and how long the waiting room took.", reason="It was the main thing we talked about.")]
ANS = {
 "E1": dict(topic=A(["FAMILY_RELATIONSHIPS_PETS"]), remembered=A(["PARTNER_SAID_OR_DID", "PARTNER_FACT_OR_QUALITY"], "she said it, and it's a fact about her"),
   new_info=A("YES"), extent=A("ONE_MOMENT"), details=A(2, "a pet snake; called Doris ('she told me' no longer counts)"), moment_text=A("WHAT_HAPPENED"),
   shared_by=A("PARTNER"), personal_kinds=A(["INFO_GENERAL"], "a pet: Barak's general information"), feelings=A(["SURPRISE"]), intensity=A(4, "'at all!'"),
   appraisal=A(dict(predict=1, pleasant="CANT_TELL", unpleasant=1, own_behavior=1, other_behavior=4)),
   reason=A(dict(about=["PARTNER"], details=["UNEXPECTED"]), "A2's unexpected_info"), matters_for=A(["NONE"], "it says why it stood out, not what it's for"),
   bigger_meaning=A("NO"), ),
 "E2": dict(topic=A(["THE_STUDY_OR_CALL", "FEELINGS_MOOD"]), remembered=A(["DID_TOGETHER", "SITUATION"], "they laughed together, about the situation"),
   new_info=A("NOT_ABOUT_INFORMATION"), extent=A("ONE_MOMENT"), details=A(2, "both laughed; the awkward start"), moment_text=A("WHAT_HAPPENED", "'awkward' is a judgement, so Both is defensible"), shared_by=A("BOTH"),
   personal_kinds=A(["FEELINGS_MILD"], "the start felt awkward: a mild, ordinary feeling"), feelings=A(["JOY", "RELIEF"]), intensity=A(3),
   appraisal=A(dict(predict=3, pleasant=4, unpleasant=2, own_behavior=3, other_behavior=3)),
   reason=A(dict(about=["US", "ME"], details=["IN_COMMON", "MY_FEELINGS"]), "a shared feeling about the start; it relaxed me"),
   matters_for=A(["CONNECTION"]), bigger_meaning=A("NO"), ),
 "E3": dict(topic=A(["WORK_EDUCATION_SERVICE", "HEALTH_WELLBEING", "FAMILY_RELATIONSHIPS_PETS"]), remembered=A(["PARTNER_SAID_OR_DID", "REPORTER_SAID_OR_DID"]), new_info=A("YES"), extent=A("ONE_MOMENT"), details=A(3, "a nurse; mum in hospital; for months; last year; the nurses helped"),
   moment_text=A("WHAT_HAPPENED"), shared_by=A("BOTH"), personal_kinds=A(["INFO_GENERAL", "INFO_REVEALING", "THOUGHTS_EVENTS_PLANS"], "'I'm a nurse'; his mum's months in hospital; how the nurses helped him get through it"), feelings=A(["PRIDE", "TRUST"], "'moving that he'd share' read as trust"), intensity=A(4),
   appraisal=A(dict(predict=2, pleasant=4, unpleasant=2, own_behavior=3, other_behavior=4)),
   reason=A(dict(about=["ME", "PARTNER"], details=["MY_FEELINGS", "MY_LIFE", "PARTNER_LIFE", "EMOTION"])), matters_for=A(["WHO_I_AM", "CONNECTION"]),
   bigger_meaning=A("YES", "ties the moment to who she is"), ),
 "E4": dict(topic=A(["THE_STUDY_OR_CALL"]), remembered=A(["SITUATION"]), new_info=A("NO"), extent=A("STRETCH"),
   details=A(1, "the topic alone is 0; 'how long the waiting room took' = 1"), moment_text=A("WHAT_HAPPENED"),
   shared_by=A("NOBODY"), personal_kinds=A(["NOTHING"]), feelings=A(["NONE"]), intensity=A(1, "no feeling shown, so not at all"),
   appraisal=A(dict(predict="CANT_TELL", pleasant="CANT_TELL", unpleasant="CANT_TELL", own_behavior="CANT_TELL", other_behavior="CANT_TELL")),
   reason=A(dict(about=["CALL"], details=["WHOLE_CALL"]), "with 'the whole conversation' now an option, 'the main thing we talked about' fits it; in v3.2 I'd said no reason"), matters_for=A(["NONE"]), bigger_meaning=A("NO"), )}
# leak check: no wording a coder sees may echo a calibration example
import re
STOP = set("that with from they them about their when what this have didn't would just made make feel told also there been where were your which will".split())
words = lambda t: {w for w in re.findall(r"[a-z']+", t.lower()) if len(w) >= 4 and w not in STOP}
seen = " ".join(SPEC["intro"] + [json.dumps(q) for p in SPEC["parts"] for q in p["questions"]]).lower()
seenw = words(seen)
for e in EX:
    hits = sorted(words(e["moment"] + " " + e["reason"]) & seenw)
    print("LEAK?", e["id"], hits)
qids = {q["id"] for p in SPEC["parts"] if p["id"] != "C" for q in p["questions"] if q["type"] != "text"}
assert all(set(ANS[e]) == qids for e in ANS), [qids - set(ANS[e]) for e in ANS]
write(os.path.join(ROOT, "human_coding", "CALIBRATION_v35.html"), mode="calibrate", key="calibration_v35", title="Calibration v3.5", coder="", items=EX, answers=ANS)
key = json.load(open(os.path.join(ROOT, "analysis", "A3_item_key.json"))); rows = {r["perspectiveID"]: r for r in P.rows()}
def real(ids):
    out = []
    for iid in ids:
        r = rows[key[iid]]; loc = json.load(open(os.path.join(ROOT, "analysis", "located", key[iid] + ".json")))
        j = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))
        tx = [dict(i=k, t=f"{int(t['start'] // 60)}:{int(t['start'] % 60):02d}", who="You" if t["spk"] == r["selfPID"] else "Partner", text=t["text"]) for k, t in enumerate(j["turns"])]
        out.append(dict(id=iid, moment=r["conv_SP_Mem_descr"].strip(), reason=r["conv_SP_Mem_reason"].strip(), tx=tx, loc=sorted(loc.get("turns") or []), lockind=loc.get("kind"), locnote=loc.get("note")))
    return out
ids = sorted(key)
write(os.path.join(ROOT, "human_coding", "A1_CODER_v35.html"), mode="code", key="A1_v35_overlap50", title="A1", coder="A1", items=real(ids[:50]))
write(os.path.join(ROOT, "human_coding", "A3_CODER_v35.html"), mode="code", key="A3_v35_100", title="A3", coder="A3", items=real(ids))
print("written")
