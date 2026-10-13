#!/usr/bin/env python3
"""Transcripts v3 (A1, Oct 10): the same turn structure as transcripts_v2 (scripts/transcribe_v2.to_turns, same per-speaker
speech regions from the same source tracks, same remote-copy shift), but built from ElevenLabs Scribe v2 words, which keep
fillers and mark non-speech events ([laughs], [chuckles], ...) with timestamps. Events stay inline as tokens, so a listener's
laugh during the partner's turn becomes a backchannel listed after that turn, exactly like "yeah".
Output: transcripts_v3_scribe/<dyad>.json and .txt. Dyads with a missing Scribe track are skipped and listed."""
import glob, json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from transcribe_v2 import to_turns
OUT = os.path.join(ROOT, "transcripts_v3_scribe"); os.makedirs(OUT, exist_ok=True)
A = os.path.join(ROOT, "multimodal", "asr_v3")
skipped, n = [], 0
for f in sorted(glob.glob(os.path.join(ROOT, "transcripts_v2", "*.json"))):
    j = json.load(open(f)); d = j["dyadID"]; words, ok = [], True
    for pid, m in j["meta"].items():
        p = os.path.join(A, f"{d}__{pid}__scribe-v2.json")
        sc = json.load(open(p)) if os.path.exists(p) else {}
        if not sc.get("words"):
            ok = False; break
        shift = -0.12 if m.get("kind") == "remote_copy" else 0.0
        for w in sc["words"]:
            if w.get("type") in ("word", "audio_event") and w.get("text", "").strip():
                words.append(dict(spk=pid, word=w["text"].strip(), start=max(0, w["start"] + shift), end=max(0, w["end"] + shift),
                                  event=w.get("type") == "audio_event"))
    if not ok:
        skipped.append(d); continue
    words.sort(key=lambda w: (w["start"], w["spk"]))
    regions = {pid: m.get("speech_regions") for pid, m in j["meta"].items()}
    turns = to_turns(words, regions)
    meta = {pid: dict(m, n_words=sum(1 for w in words if w["spk"] == pid and not w["event"]),
                      n_events=sum(1 for w in words if w["spk"] == pid and w["event"])) for pid, m in j["meta"].items()}
    json.dump(dict(dyadID=d, model="elevenlabs/scribe_v2 (verbatim, audio events tagged)", speakers=j["speakers"], meta=meta,
                   turns=turns, words=words), open(os.path.join(OUT, d + ".json"), "w"))
    with open(os.path.join(OUT, d + ".txt"), "w") as fh:
        for t in turns:
            fh.write(f"[{int(t['start'] // 60)}:{int(t['start'] % 60):02d}] {t['spk']}{' (bc)' if t.get('bc') else ''}: {t['text']}\n")
    n += 1
print("built", n, "skipped", len(skipped), skipped[:10])
