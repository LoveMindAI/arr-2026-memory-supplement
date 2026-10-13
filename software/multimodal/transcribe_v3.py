#!/usr/bin/env python3
"""Second and third ASR instruments (A1, Oct 10): ElevenLabs Scribe v2 (tags non-speech events such as [laughs]) and
Microsoft MAI-Transcribe-2, both via OpenRouter /audio/transcriptions, on EXACTLY the same per-speaker source tracks that
transcribe_v2.py (Whisper large-v3) used: transcripts_v2/<dyad>.json meta[pid].source (own mic, or remote copy where missing).
Output: multimodal/asr_v3/<dyad>__<pid>__<model>.json (raw response). Resumable; failed calls are retried on rerun."""
import base64, glob, json, os, subprocess, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONV = os.path.expanduser("<LOCAL_FILE> Story/Conv Data/Conversations")
OUT = os.path.join(ROOT, "multimodal", "asr_v3"); os.makedirs(OUT, exist_ok=True)
KEY = os.environ["OPENROUTER_API_KEY"]
MODELS = ["elevenlabs/scribe-v2", "microsoft/mai-transcribe-2"]   # Scribe: direct ElevenLabs API (word + audio-event timestamps); MAI: OpenRouter
EL_KEY = open(os.path.expanduser("~/.config/LAB/elevenlabs_key")).read().strip()
vids = {os.path.basename(p): p for p in glob.glob(CONV + "/*/*.mp4")}

def mp3_bytes(path):   # exactly the encoding transcribe_v2.py sent to Whisper (mono, 16 kHz, 48 kbps mp3)
    return subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", "-f", "mp3", "-"],
                          capture_output=True, check=True).stdout

def wav_b64(path):
    return base64.b64encode(mp3_bytes(path)).decode()

def wav_bytes(path):
    return subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-"], capture_output=True, check=True).stdout

def scribe_direct(path):
    import uuid
    b = uuid.uuid4().hex; parts = []
    for k, v in [("model_id", "scribe_v2"), ("tag_audio_events", "true"), ("timestamps_granularity", "word"), ("diarize", "false"), ("language_code", "en")]:
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\nContent-Type: audio/mpeg\r\n\r\n".encode() + mp3_bytes(path) + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request("https://api.elevenlabs.io/v1/speech-to-text", data=b"".join(parts),
                                 headers={"xi-api-key": EL_KEY, "Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())

def job(a):
    dyad, pid, src, model = a
    p = os.path.join(OUT, f"{dyad}__{pid}__{model.split('/')[1]}.json")
    if os.path.exists(p):
        old = json.load(open(p))
        if old.get("text") is not None and old.get("_enc") == "mp3_16k_48kbps" and old.get("_model") == model:
            return 0.0
    if model.startswith("elevenlabs/"):      # NOT `"scribe" in model`: "mai-transcribe-2" contains "scribe" (bug caught Oct 10)
        for att in range(4):
            try:
                r = scribe_direct(vids[src]); r["_src"] = src; r["_via"] = "elevenlabs-direct"; r["_enc"] = "mp3_16k_48kbps"; r["_model"] = model; json.dump(r, open(p, "w")); return 0.0
            except Exception as e:
                err = str(e)[:300]; time.sleep(5 * (att + 1))
        json.dump({"text": None, "error": err, "_src": src}, open(p, "w")); return 0.0
    body = {"model": model, "input_audio": {"data": wav_b64(vids[src]), "format": "mp3"}, "response_format": "verbose_json", "language": "en"}
    for att in range(4):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/audio/transcriptions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            r["_src"] = src; r["_enc"] = "mp3_16k_48kbps"; r["_model"] = model; r["_via"] = "openrouter"; json.dump(r, open(p, "w")); return float((r.get("usage") or {}).get("cost") or 0)
        except Exception as e:
            err = str(e)[:300]; time.sleep(3 * (att + 1))
    json.dump({"text": None, "error": err, "_src": src}, open(p, "w")); return 0.0

jobs = []
for f in sorted(glob.glob(os.path.join(ROOT, "transcripts_v2", "*.json"))):
    j = json.load(open(f))
    for pid, m in j["meta"].items():
        if m.get("source") in vids:
            for mo in MODELS: jobs.append((j["dyadID"], pid, m["source"], mo))
        else:
            print("MISSING SOURCE", j["dyadID"], pid, m.get("source"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else len(jobs)
jobs = jobs[:n]
if os.environ.get('REVERSE'): jobs = jobs[::-1]
if os.environ.get('ONLY'): jobs = [x for x in jobs if x[3].startswith(os.environ['ONLY'])]
print("jobs", len(jobs), flush=True)
t = time.time(); done = 0; cost = 0.0
with ThreadPoolExecutor(int(os.environ.get("CONC", 24))) as ex:
    for c in ex.map(job, jobs):
        cost += c; done += 1
        if done % 100 == 0: print(done, "done", round(time.time() - t), "s, $", round(cost, 3), flush=True)
bad = [os.path.basename(p) for p in glob.glob(OUT + "/*.json") if json.load(open(p)).get("text") is None]
print("FINISHED", done, "jobs", round(time.time() - t), "s, cost $", round(cost, 3), "| failed", len(bad), flush=True)
