#!/usr/bin/env python3
"""Audio/video-only subset (A1, Oct 11): one call per session, chosen at random by design (not tuned to anything), carries one non-verbal
event that a camera or microphone would catch and the transcript never shows. The noticer (random side) remembers it; the event happens on
the other person's side. 55 calls = 4.7% of memories. -> av_plan.json {call: {noticer, actor}}"""
import json, os, random
HERE = os.path.dirname(os.path.abspath(__file__)); S = json.load(open(os.path.join(HERE, "sessions.json"))); rng = random.Random(5511)
plan = {}
for sid in sorted(S["sessions"]):
    c = rng.choice([c for c in S["calls"] if c["session"] == sid]); n = rng.choice(["A", "B"])
    plan[c["id"]] = dict(session=sid, noticer=c[n], noticer_seat=n, actor=c["B" if n == "A" else "A"])
json.dump(plan, open(os.path.join(HERE, "av_plan.json"), "w"), indent=1); print(len(plan), "audio/video-only calls")
