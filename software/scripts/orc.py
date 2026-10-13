#!/usr/bin/env python3
"""Minimal OpenRouter client for the memory project: text / audio / video parts, cost logging, retries.
Every call is appended to a JSONL log with request summary (no base64 payloads), reply, usage and cost."""
import base64, json, os, time, urllib.request, urllib.error

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY", "")


WAV_ONLY = ("thinkingmachines/",)


def as_wav(path):
    """16 kHz mono PCM WAV next to the source (cached)."""
    import subprocess
    w = os.path.splitext(path)[0] + ".wav"
    if not os.path.exists(w):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", w], check=True)
    return w


def audio_part(path, model=None):
    if model and model.startswith(WAV_ONLY):
        path = as_wav(path)
    fmt = os.path.splitext(path)[1][1:].lower()
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(open(path, "rb").read()).decode(), "format": fmt}}


def video_part(path):
    b = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b}"}}


def text_part(t):
    return {"type": "text", "text": t}


def _summ(messages):
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "text": c})
        else:
            out.append({"role": m["role"], "parts": [p["text"] if p["type"] == "text" else f"<{p['type']}>" for p in c]})
    return out


def call(model, messages, log_path, tag="", max_tokens=16000, temperature=None, reasoning=None, provider=None,
         response_format=None, retries=3, timeout=600):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True}}
    if temperature is not None:
        body["temperature"] = temperature
    if reasoning is not None:
        body["reasoning"] = reasoning
    if provider:
        body["provider"] = provider
    if response_format:
        body["response_format"] = response_format
    data = json.dumps(body).encode()
    rec = {"t": time.time(), "tag": tag, "model": model, "req": _summ(messages), "params": {k: v for k, v in body.items() if k not in ("messages",)}}
    err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                                                   "X-Title": "LAB memory project"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.loads(r.read())
            if "choices" not in j:
                err = j.get("error", j)
                time.sleep(3 * (attempt + 1))
                continue
            ch = j["choices"][0]
            rec.update(ok=True, text=ch["message"].get("content"), reasoning_text=ch["message"].get("reasoning"),
                       finish=ch.get("finish_reason"), usage=j.get("usage"), provider_used=j.get("provider"), id=j.get("id"))
            break
        except urllib.error.HTTPError as e:
            err = {"http": e.code, "body": e.read().decode()[:2000]}
            if e.code in (400, 401, 402, 403, 404):
                break
            time.sleep(3 * (attempt + 1))
        except Exception as e:  # network
            err = {"exc": repr(e)}
            time.sleep(3 * (attempt + 1))
    else:
        pass
    if not rec.get("ok"):
        rec.update(ok=False, error=err)
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def cost(rec):
    u = rec.get("usage") or {}
    return float(u.get("cost") or 0)
