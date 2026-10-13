#!/usr/bin/env python3
"""A1 (Oct 11): 'for anything that looks like it may be for hidden vocal cues ... when there's disagreement, run it on the annotated Scribe
transcripts'. Disagreement among the three locators on a person's memory = different kinds, or all three place it with no turn in common.
Vocal-looking = a locator called it not_in_transcript, or a note or the memory mentions laughter, voice, tone, accent, a pause or silence,
nerves, talking over, awkwardness. -> analysis/scribe_recheck_pids.json"""
import glob, json, os, re
A = "../analysis"
def load(d): return {json.load(open(f))["perspectiveID"]: json.load(open(f)) for f in glob.glob(os.path.join(A, d, "PID_*.json"))}
D = {l: load(d) for l, d in (("sonnet", "located"), ("gemini", "located_g3f"), ("luna", "located_luna"))}
def rng_(j):
    t = j.get("turns") or []
    return set(range(min(t), max(t) + 1)) if (j.get("kind") == "in_conversation" and t) else None
import sys; sys.path.insert(0, "."); import prompts as P
R = {r["perspectiveID"]: r for r in P.rows()}
VOC = re.compile(r"laugh|giggl|chuckl|voice|tone|accent|pause|silen|quiet|nervous|interrupt|talk(ed|ing)? over|same time|awkward|sound|spoke|soft|loud|smil", re.I)
dis, voc = [], []
for k in set.intersection(*(set(v) for v in D.values())):
    js = [D[l][k] for l in D]; kinds = {j.get("kind") for j in js}; rs = [rng_(j) for j in js]
    disagree = len(kinds) > 1 or (all(rs) and not set.intersection(*rs))
    if not disagree: continue
    dis.append(k)
    text = " ".join([R[k]["conv_SP_Mem_descr"], R[k]["conv_SP_Mem_reason"]] + [str(j.get("note") or "") for j in js])
    if "not_in_transcript" in kinds or VOC.search(text): voc.append(k)
json.dump(sorted(voc), open(os.path.join(A, "scribe_recheck_pids.json"), "w")); json.dump(sorted(dis), open(os.path.join(A, "locator_disagree_pids.json"), "w"))
print("disagreements", len(dis), "vocal-looking", len(voc))
