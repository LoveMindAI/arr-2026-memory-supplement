#!/usr/bin/env python3
"""Transcripts v2: word-level Whisper (Groq whisper-large-v3) on each speaker's clean own-mic track, merged into turns.

Per dyad: transcribe self_local of each person (partner muted at source). Missing self_local -> that person's voice as
recorded on the partner's machine (self_remote), shifted back by the measured latency. Both clocks agree to ~0.1-0.2 s
(align_probe.py), so words are merged on their own timestamps.
Outputs transcripts_v2/<dyadID>.json (words per speaker + turns + provenance) and .txt (turns with [mm:ss]).
Rendering for prompts relabels speakers as You / Partner per rememberer (see prompts.py), never PIDs.
"""
import csv, glob, io, json, os, re, subprocess, sys, time, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.expanduser("<LOCAL_FILE> Story")
CONV = BASE + "/Conv Data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", os.environ.get("TX_OUT", "transcripts_v2"))
ASR_PROMPT = os.environ.get("ASR_PROMPT", "")
os.makedirs(OUT, exist_ok=True)
KEY = os.environ.get("GROQ_API_KEY") or re.search(r"gsk_[A-Za-z0-9]+", open(BASE + "/TRANSCRIPTION_PIPELINE_DESIGN.md").read()).group(0)
MODEL = "whisper-large-v3"
GAP = 1.2
FORCE = set(os.environ.get("FORCE", "").split(","))


def mp3_bytes(path):
    return subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", "-f", "mp3", "-"],
                          capture_output=True, check=True).stdout


def groq(audio, name):
    b = uuid.uuid4().hex
    parts = []
    def field(k, v):
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    field("model", MODEL); field("response_format", "verbose_json"); field("language", "en"); field("temperature", "0")
    field("timestamp_granularities[]", "word"); field("timestamp_granularities[]", "segment")
    if ASR_PROMPT:
        field("prompt", ASR_PROMPT)
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}.mp3\"\r\nContent-Type: audio/mpeg\r\n\r\n".encode() + audio + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    body = b"".join(parts)
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://api.groq.com/openai/v1/audio/transcriptions", data=body,
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": f"multipart/form-data; boundary={b}",
                                                  "User-Agent": "LAB-memory/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code == 429:
                time.sleep(10 * (attempt + 1)); continue
            raise RuntimeError(f"groq {e.code} {msg}")
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("groq retries exhausted")


def speech_regions(path, min_gap=0.35, min_len=0.12):
    """Energy VAD on a clean single-speaker track: frames (10 ms) above max(noise_floor+12 dB, peak-35 dB),
    regions merged across gaps < min_gap. Returns [[start, end], ...] in seconds."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(x) // 160
    e = 20 * np.log10(np.sqrt((x[: n * 160].reshape(n, 160) ** 2).mean(1) + 1e-10))
    thr = max(np.percentile(e, 10) + 12, np.percentile(e, 99) - 35)
    on = np.convolve(e > thr, np.ones(5) / 5, mode="same") > 0.4
    regs, st = [], None
    for i, v in enumerate(on):
        if v and st is None:
            st = i
        if not v and st is not None:
            regs.append([st / 100, i / 100]); st = None
    if st is not None:
        regs.append([st / 100, n / 100])
    merged = []
    for r in regs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_len]


def find(stem, vids):
    return vids.get(stem)


PHRASE_GAP = 0.6   # pause that ends a phrase within one speaker
BACKCHANNEL = 2    # phrases of <= this many words don't take the floor ("yeah", "oh wow")


def to_turns(words, regions=None):
    """1) Per speaker, keep Whisper's word ORDER. Cut phrases where the speaker's own clean mic track shows a pause
       (energy-VAD speech regions; Whisper's word end-times stretch over silences so they can't find pauses).
       Without regions, fall back to start-gap > PHRASE_GAP.
    2) Walk phrases in start-time order. A speaker holds the floor until the OTHER speaker produces a substantive phrase
       (> BACKCHANNEL words); backchannels during someone's turn are listed right after that turn, marked bc=True."""
    import bisect
    phrases = []
    for spk in sorted({w["spk"] for w in words}):
        regs = (regions or {}).get(spk) or []
        starts = [r[0] for r in regs]
        def region_of(t):
            if not regs:
                return None
            k = bisect.bisect_right(starts, t + 0.15) - 1
            return max(k, 0)
        cur, cur_reg, prev_start = None, None, None
        for w in [x for x in words if x["spk"] == spk]:
            rg = region_of(w["start"])
            same = (rg == cur_reg) if regs else (prev_start is not None and w["start"] - prev_start <= PHRASE_GAP + 0.4)
            if cur and same:
                cur["text"] += " " + w["word"].strip(); cur["end"] = max(cur["end"], w["end"]); cur["n"] += 1
            else:
                if cur:
                    phrases.append(cur)
                cur = dict(spk=spk, start=w["start"], end=w["end"], text=w["word"].strip(), n=1)
            cur_reg, prev_start = rg, w["start"]
        if cur:
            phrases.append(cur)
    phrases.sort(key=lambda p: (p["start"], p["spk"]))
    turns, main, bcs = [], None, []
    for p in phrases:
        if main and p["spk"] == main["spk"]:
            main["text"] += " " + p["text"]; main["end"] = max(main["end"], p["end"]); continue
        if main and p["n"] <= BACKCHANNEL:
            bcs.append(dict(spk=p["spk"], start=p["start"], end=p["end"], text=p["text"], bc=True)); continue
        if main:
            turns.append(main); turns.extend(bcs); bcs = []
        main = dict(spk=p["spk"], start=p["start"], end=p["end"], text=p["text"], bc=False)
    if main:
        turns.append(main); turns.extend(bcs)
    return turns


def do_dyad(row, vids, force=False):
    d = row["dyadID"]
    jp = os.path.join(OUT, d + ".json")
    if os.path.exists(jp) and not force:
        return d, "skip"
    a, b = row["selfPID"], row["otherPID"]
    src = {}
    # person a: own mic = self_local; fallback = self_remote (a's voice on b's machine)
    src[a] = (find(row["conv_rec_self_local"], vids), "self_local") if find(row["conv_rec_self_local"], vids) else (find(row["conv_rec_self_remote"], vids), "remote_copy")
    src[b] = (find(row["conv_rec_other_local"], vids), "self_local") if find(row["conv_rec_other_local"], vids) else (find(row["conv_rec_other_remote"], vids), "remote_copy")
    words, meta = [], {}
    for pid, (path, kind) in src.items():
        if not path:
            meta[pid] = dict(source=None); continue
        j = groq(mp3_bytes(path), pid)
        shift = -0.12 if kind == "remote_copy" else 0.0  # remote copies arrive ~0.1-0.2 s late
        ws = [dict(spk=pid, word=w["word"], start=max(0, w["start"] + shift), end=max(0, w["end"] + shift)) for w in j.get("words", [])]
        words += ws
        regs = [[a + shift, b + shift] for a, b in speech_regions(path)]
        meta[pid] = dict(source=os.path.basename(path), kind=kind, n_words=len(ws), duration=j.get("duration"), speech_regions=regs,
                         segments=[dict(start=s["start"], end=s["end"], text=s["text"], no_speech_prob=s.get("no_speech_prob"),
                                        avg_logprob=s.get("avg_logprob")) for s in j.get("segments", [])])
    turns = to_turns(words, {p: m.get("speech_regions") for p, m in meta.items()})
    json.dump(dict(dyadID=d, model=MODEL, asr_prompt=ASR_PROMPT, speakers=[a, b], meta=meta, turns=turns, words=words), open(jp, "w"))
    with open(os.path.join(OUT, d + ".txt"), "w") as f:
        for t in turns:
            f.write(f"[{int(t['start']//60):02d}:{int(t['start']%60):02d}] {t['spk']}: {t['text']}\n")
    return d, f"ok {len(turns)} turns " + " ".join(f"{p}:{m.get('kind')}" for p, m in meta.items())


def main():
    vids = {os.path.basename(p)[:-4]: p for p in glob.glob(CONV + "/Conversations/*/*.mp4")}
    rows = {r["dyadID"]: r for r in csv.DictReader(open(CONV + "/LIST_unique_dyads_and_clean_videos_Manual.tsv"), delimiter="\t")}
    mem_dyads = {r["dyadID"] for r in csv.DictReader(open(CONV + "/cleaned_curated_conv_data_combined_ffinal.csv", encoding="utf-8-sig"))}
    todo = [rows[d] for d in sorted(mem_dyads) if d in rows]
    if len(sys.argv) > 1:
        todo = [rows[d] for d in sys.argv[1:]]
    print("dyads", len(todo), "missing from TSV", len(mem_dyads - set(rows)))
    with ThreadPoolExecutor(int(os.environ.get("CONC", "6"))) as ex:
        for d, st in ex.map(lambda r: _safe(r, vids), todo):
            print(d, st, flush=True)


def _safe(r, vids):
    try:
        return do_dyad(r, vids, force=r["dyadID"] in FORCE)
    except Exception as e:
        return r["dyadID"], f"ERROR {e}"


if __name__ == "__main__":
    main()
