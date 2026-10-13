#!/usr/bin/env python3
"""Build perspective-specific media for PARSEL memory rows.

For each memory row (rememberer S, partner O) we rebuild what S experienced:
  S's own mic   = conv_rec_self_local   (partner muted at source)
  O as S heard  = conv_rec_other_remote (recorded on S's machine, same clock)
Fallbacks when one is missing use the other machine's copy, shifted by an offset measured
from a same-voice pair (local vs remote of the same speaker, r ~ .97-.99 in probes).

Outputs in media/<perspectiveID>/:
  self.mp3, partner.mp3          single-voice tracks (16 kHz mono)
  mix.mp3                        both voices, mono (what S heard incl. own voice)
  stereo.mp3                     L = S (rememberer), R = partner
  video.mp4                      side by side: LEFT = partner (as S saw them), RIGHT = S; mixed mono audio; 640x240 @10fps
  meta.json                      sources, offsets, durations, method
Usage: build_media.py --pids PID_n+PID_n ... | --all [--no-video] [--jobs 6]
"""
import argparse, csv, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

BASE = os.path.expanduser("<LOCAL_FILE> Story/Conv Data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")
SR = 16000
HOP = 160


def mp4(stem, vids):
    return vids.get(stem)


def pcm(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def env(x):
    n = len(x) // HOP
    return 20 * np.log10(np.sqrt((x[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-10))


def lag_seconds(a_path, b_path, maxlag=1500):
    """Lag (s) of b relative to a for the SAME voice; positive = b later. Returns (lag, r)."""
    a, b = env(pcm(a_path)), env(pcm(b_path))
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    best = (-9.0, 0)
    for lag in range(-maxlag, maxlag + 1):
        x, y = (a[: len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: len(b) + lag])
        m = min(len(x), len(y))
        if m < 1000:
            continue
        c = float(np.mean(x[:m] * y[:m]))
        if c > best[0]:
            best = (c, lag)
    return best[1] * HOP / SR, best[0]


def plan(row, vids):
    """Choose sources. Returns dict with self_src, partner_src, partner_shift_s (delay applied to partner), method."""
    sl, orr = mp4(row["conv_rec_self_local"], vids), mp4(row["conv_rec_other_remote"], vids)
    ol, sr = mp4(row["conv_rec_other_local"], vids), mp4(row["conv_rec_self_remote"], vids)
    if sl and orr:
        return dict(self_src=sl, partner_src=orr, self_video=sl, partner_video=orr, shift=0.0, method="A_own_machine")
    if sl and ol:
        # other_local is on O's clock. Offset O->S: S's voice appears on O's machine (self_remote) `lag` later than self_local.
        if sr:
            lag, r = lag_seconds(sl, sr)
            return dict(self_src=sl, partner_src=ol, self_video=sl, partner_video=ol, shift=-lag + 0.1, method=f"B_cross_machine_lag{lag:+.2f}_r{r:.2f}")
        return dict(self_src=sl, partner_src=ol, self_video=sl, partner_video=ol, shift=0.1, method="B_cross_machine_unaligned")
    if sr and orr:
        return dict(self_src=sr, partner_src=orr, self_video=sr, partner_video=orr, shift=0.0, method="C_remote_pair_unaligned")
    if sr and ol:
        return dict(self_src=sr, partner_src=ol, self_video=sr, partner_video=ol, shift=0.0, method="D_O_machine")
    return None


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def build(row, vids, video=True, force=False):
    pid = row["perspectiveID"]
    d = os.path.join(OUT, pid)
    if os.path.exists(os.path.join(d, "meta.json")) and not force and (not video or os.path.exists(os.path.join(d, "video_silent.mp4"))):
        return pid, "skip"
    p = plan(row, vids)
    if not p:
        return pid, "no_sources"
    os.makedirs(d, exist_ok=True)
    sh = p["shift"]
    # partner delay filter (adelay for positive shift, atrim for negative)
    pf = f"adelay={int(sh*1000)}" if sh > 0 else (f"atrim=start={-sh},asetpts=PTS-STARTPTS" if sh < 0 else "anull")
    a16 = ["-ar", str(SR), "-ac", "1", "-b:a", "48k"]
    run(["ffmpeg", "-v", "error", "-y", "-i", p["self_src"], "-vn", *a16, os.path.join(d, "self.mp3")])
    run(["ffmpeg", "-v", "error", "-y", "-i", p["partner_src"], "-vn", "-af", pf, *a16, os.path.join(d, "partner.mp3")])
    fc = (f"[0:a]aresample={SR},pan=mono|c0=c0[s];[1:a]aresample={SR},pan=mono|c0=c0,{pf}[o];"
          f"[s][o]amix=inputs=2:duration=shortest:normalize=0[m]")
    run(["ffmpeg", "-v", "error", "-y", "-i", p["self_src"], "-i", p["partner_src"], "-filter_complex", fc, "-map", "[m]",
         *a16, os.path.join(d, "mix.mp3")])
    fs = (f"[0:a]aresample={SR},pan=mono|c0=c0[s];[1:a]aresample={SR},pan=mono|c0=c0,{pf}[o];"
          f"[s][o]amerge=inputs=2[m]")
    run(["ffmpeg", "-v", "error", "-y", "-i", p["self_src"], "-i", p["partner_src"], "-filter_complex", fs, "-map", "[m]",
         "-ar", str(SR), "-ac", "2", "-b:a", "64k", "-shortest", os.path.join(d, "stereo.mp3")])
    if video:
        vpf = f"tpad=start_duration={sh}" if sh > 0 else (f"trim=start={-sh},setpts=PTS-STARTPTS" if sh < 0 else "null")
        fv = (f"[1:v]{vpf},scale=320:240,setsar=1[pv];[0:v]scale=320:240,setsar=1[sv];[pv][sv]hstack=inputs=2,fps=10[v]")
        run(["ffmpeg", "-v", "error", "-y", "-i", p["self_video"], "-i", p["partner_video"], "-i", os.path.join(d, "mix.mp3"),
             "-filter_complex", fv, "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
             "-c:a", "aac", "-b:a", "48k", "-shortest", "-movflags", "+faststart", os.path.join(d, "video.mp4")])
        run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(d, "video.mp4"), "-an", "-c:v", "copy", os.path.join(d, "video_silent.mp4")])
    def dur(f):
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f], capture_output=True, text=True).stdout
        return float(o.strip() or 0)
    meta = dict(perspectiveID=pid, selfPID=row["selfPID"], otherPID=row["otherPID"], dyadID=row["dyadID"], method=p["method"],
                partner_shift_s=sh, sources={k: os.path.basename(v) for k, v in p.items() if k.endswith(("src", "video"))},
                durations={f: dur(os.path.join(d, f)) for f in os.listdir(d) if f.endswith((".mp3", ".mp4"))},
                sizes={f: os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith((".mp3", ".mp4"))},
                layout="video.mp4: LEFT=partner as rememberer saw them, RIGHT=rememberer; stereo.mp3: L=rememberer, R=partner")
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    return pid, p["method"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    vids = {os.path.basename(p)[:-4]: p for p in glob.glob(BASE + "/Conversations/*/*.mp4")}
    rows = list(csv.DictReader(open(BASE + "/cleaned_curated_conv_data_combined_ffinal.csv", encoding="utf-8-sig")))
    if not a.all:
        rows = [r for r in rows if r["perspectiveID"] in set(a.pids or [])]
    with ThreadPoolExecutor(a.jobs) as ex:
        for pid, st in ex.map(lambda r: build(r, vids, not a.no_video, a.force), rows):
            print(pid, st, flush=True)


if __name__ == "__main__":
    main()
