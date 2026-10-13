#!/usr/bin/env python3
"""A1's locator check on MODEL guesses (Oct 11: "run it and build the packet"). His earlier 150 checks were all placements of people's
memories; this packet checks the locator on what the five models wrote (no history and history conditions), the placements the new
position/span analysis uses. 150 items: 75 per condition; within each, half high and half medium confidence (plus every low or
not-in-transcript placement there is, which are rare). Same page, same four answers as LOCATOR_A1_2. Real PARSEL transcripts: A1 only.
-> human_coding/LOCATOR_MODELS_A1.html, key analysis/locator_models_packet_key.json"""
import glob, json, os, random, re
import prompts as P
ROOT = ".."; rng = random.Random(1511)
R = {r["perspectiveID"]: r for r in P.rows()}
PRED = {}
for l in open(os.path.join(ROOT, "runs", "main_text_cheap", "preds.jsonl")):
    j = json.loads(l)
    if j.get("ok") and j["info"] in ("none", "history") and j.get("modality", "text") == "text":
        PRED[(j["info"], j["model"].split("/")[1], j["perspectiveID"])] = j["pred"]
seen = set()
for f in ("locator_a1_packet_key.json", "locator_a1_packet2_key.json"):
    d = json.load(open(os.path.join(ROOT, "analysis", f))); seen |= {v["pid"] for v in (d.values() if isinstance(d, dict) else d)}
pick = []
for cond, sub in (("none", "located_preds"), ("history", "located_preds_history")):
    L = []
    for f in glob.glob(os.path.join(ROOT, "analysis", sub, "*", "PID_*.json")):
        j = json.load(open(f)); j["cond"] = cond
        if j["perspectiveID"] in R and (cond, j["model"], j["perspectiveID"]) in PRED and j["perspectiveID"] not in seen: L.append(j)
    rare = [j for j in L if j.get("kind") != "in_conversation" or j.get("confidence") == "low"]
    hi = [j for j in L if j not in rare and j.get("confidence") == "high"]; me = [j for j in L if j not in rare and j.get("confidence") == "medium"]
    rng.shuffle(hi); rng.shuffle(me); rng.shuffle(rare)
    pool = rare[:10] + [x for pair in zip(hi, me) for x in pair]   # rare first, then high and medium alternating
    used, got = set(), 0                                            # one item per call per condition
    for j in pool:
        if got == 75: break
        dy = R[j["perspectiveID"]]["dyadID"]
        if dy in used: continue
        used.add(dy); pick.append(j); got += 1
rng.shuffle(pick); items, key = [], {}
for n, j in enumerate(pick[:150], 1):
    r = R[j["perspectiveID"]]; tx = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]
    p = PRED[(j["cond"], j["model"], j["perspectiveID"])]; iid = f"G{n:03d}"
    hl = j["turns"] if j.get("kind") == "in_conversation" else []
    items.append(dict(id=iid, mem=str(p.get("moment", "")).strip(), why=str(p.get("reason", "")).strip(),
                      turns=[["You" if t["spk"] == r["selfPID"] else "Partner", t["text"]] for t in tx], hl=hl))
    key[iid] = dict(pid=j["perspectiveID"], model=j["model"], cond=j["cond"], kind=j.get("kind"), confidence=j.get("confidence"), turns=hl, note=j.get("note"))
page = open(os.path.join(ROOT, "human_coding", "LOCATOR_A1_2.html")).read()
i = page.find("const ITEMS=") + len("const ITEMS="); _, end = json.JSONDecoder().raw_decode(page[i:])
page = page[:i] + json.dumps(items) + page[i + end:]
page = page.replace('KEY="locator_a1_v2"', 'KEY="locator_models_A1"').replace("LOCATOR_A1_2_answers.json", "LOCATOR_MODELS_A1_answers.json")
page = page.replace("Left: what the participant wrote right after the call.", "Left: an AI model's guess at what the participant remembered (not their real answer).")
page = page.replace("Answer for the highlighted lines", "The question is only whether the guess is in the highlighted lines, not whether it's a good guess. Answer for the highlighted lines")
open(os.path.join(ROOT, "human_coding", "LOCATOR_MODELS_A1.html"), "w").write(page)
json.dump(key, open(os.path.join(ROOT, "analysis", "locator_models_packet_key.json"), "w"), indent=1)
import collections; print(len(items), collections.Counter((v["cond"], v["confidence"], v["kind"]) for v in key.values()))
