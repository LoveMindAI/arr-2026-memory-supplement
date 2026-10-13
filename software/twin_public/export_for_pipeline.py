#!/usr/bin/env python3
"""Write the public stand-in in the layout the paper's pipeline reads, so the method runs end to end without PARSEL:
  pipeline_data/conv.csv               one row per memory report (perspectiveID, selfPID, otherPID, dyadID, batchID, roundIdx,
                                       conv_SP_Mem_descr, conv_SP_Mem_reason), the columns scripts/prompts.py uses
  pipeline_data/transcripts/<dyad>.json  speaker-attributed turns ({"spk", "start", "end", "text"})
The stand-in's calls carry no timing, so turn times are ESTIMATED from word counts (2.5 words per second, 0.4 s gaps); they only
set the [m:ss] stamps in the prompt. Then: MEMORY_DATA=<this folder>/pipeline_data python3 scripts/run_predict.py ...
Usage: python3 export_for_pipeline.py [--sessions S01,S02] [--out DIR] [--src DIR]"""
import argparse, csv, glob, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pipeline_data")
ap = argparse.ArgumentParser(); ap.add_argument("--sessions", default=""); ap.add_argument("--out", default=OUT); ap.add_argument("--src", default=HERE, help="folder holding calls/, sessions/, memories/ (the packet ships them in data/twin_public)"); a = ap.parse_args(); OUT = a.out; HERE = a.src
want = set(a.sessions.split(",")) if a.sessions else None
os.makedirs(os.path.join(OUT, "transcripts"), exist_ok=True)
have = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(HERE, "sessions", "S*.json"))}
if not have: raise SystemExit(f"no sessions/S*.json under {HERE}; pass --src pointing at data/twin_public")
if want and not want <= have: raise SystemExit(f"unknown session(s): {sorted(want - have)}; available S01..{max(have)}")
calls = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "calls", "C*.json"))}
rows, nt = [], 0
for sf in sorted(glob.glob(os.path.join(HERE, "sessions", "S*.json"))):
    sid = os.path.basename(sf)[:-5]
    if want and sid not in want: continue
    S = json.load(open(sf))
    for cid, text in S["calls"].items():
        c = calls[cid]; A, B = c["A"], c["B"]; name2pid = {c["names"]["A"]: A, c["names"]["B"]: B}
        dyad = f"{A}+{B}"; t0, turns = 0.0, []
        for line in text.split("\n"):
            m = re.match(r"^([^:]{1,40}):\s*(.+)$", line.strip())
            if not m or m.group(1) not in name2pid: continue
            dur = max(1.0, len(m.group(2).split()) / 2.5)
            turns.append(dict(spk=name2pid[m.group(1)], start=round(t0, 2), end=round(t0 + dur, 2), text=m.group(2)))
            t0 += dur + 0.4
        json.dump(dict(dyadID=dyad, source="public stand-in (synthetic); times estimated", turns=turns),
                  open(os.path.join(OUT, "transcripts", dyad + ".json"), "w"), indent=1); nt += 1
        for seat, me, other in (("A", A, B), ("B", B, A)):
            mf = os.path.join(HERE, "memories", f"{cid}_{me}.json")
            if not os.path.exists(mf): continue
            mj = json.load(open(mf))
            if not mj.get("ok"): continue
            rows.append(dict(perspectiveID=f"{me}+{other}", selfPID=me, otherPID=other, dyadID=dyad, batchID=sid, roundIdx=c["round"],
                             conv_SP_Mem_descr=mj["answers"]["moment"], conv_SP_Mem_reason=mj["answers"].get("reason", "")))
if not rows: raise SystemExit("no memory reports in the selected sessions; nothing written")
with open(os.path.join(OUT, "conv.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} memory reports, {nt} transcripts -> {os.path.relpath(OUT)}")
