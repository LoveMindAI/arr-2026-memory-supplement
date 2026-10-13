#!/usr/bin/env python3
"""Probe PARSEL recording alignment for a few dyads.

Per dyad row of LIST_unique_dyads_and_clean_videos_Manual.tsv we have four recordings:
  self_local   = S's own camera/mic (partner muted, per Feb 22 notes)
  other_remote = O's feed as received/recorded on S's machine
  other_local  = O's own camera/mic
  self_remote  = S's feed as received/recorded on O's machine
Questions:
  1. Does self_local contain only S? (envelope corr with other_local should be low at every lag)
  2. Is other_remote O's voice, and what's its lag vs other_local? (should correlate strongly)
  3. Is other_remote time-aligned with self_local on S's machine? (lag of other_remote vs other_local tells us)
Usage: python3 align_probe.py PID_n+PID_n [...]
"""
import csv, subprocess, sys, os
import numpy as np

BASE = os.path.expanduser("<LOCAL_FILE> Story/Conv Data")
SR = 16000
HOP = 160  # 10 ms frames


def find_mp4(stem):
    if not stem:
        return None
    batch = stem.split("-Batch_")[1].split("-")[0] if "-Batch_" in stem else None
    p = os.path.join(BASE, "Conversations", f"Batch_{batch}", stem + ".mp4")
    return p if os.path.exists(p) else None


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def env(x):
    n = len(x) // HOP
    e = np.sqrt((x[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-10)
    return 20 * np.log10(e)


def xcorr(a, b, maxlag=1500):
    """corr of z-scored envelopes; positive lag means b is delayed relative to a."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best = (-9, 0)
    out = {}
    for lag in range(-maxlag, maxlag + 1, 2):
        if lag >= 0:
            x, y = a[: len(a) - lag] if lag else a, b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        m = min(len(x), len(y))
        if m < 1000:
            continue
        c = float(np.mean(x[:m] * y[:m]))
        out[lag] = c
        if c > best[0]:
            best = (c, lag)
    return best, out.get(0, float("nan"))


def main(dyads):
    rows = {r["dyadID"]: r for r in csv.DictReader(open(os.path.join(BASE, "LIST_unique_dyads_and_clean_videos_Manual.tsv")), delimiter="\t")}
    for d in dyads:
        r = rows[d]
        P = {k: find_mp4(r[k]) for k in ["conv_rec_self_local", "conv_rec_other_remote", "conv_rec_other_local", "conv_rec_self_remote"]}
        print(f"\n== {d}  perspective {r['perspectiveID']}  (self={r['selfPID']})")
        for k, v in P.items():
            print(f"   {k:22s} {'OK' if v else 'missing'}  {os.path.basename(v) if v else r[k]}")
        E = {}
        for k, v in P.items():
            if v:
                x = load(v)
                E[k] = env(x)
                print(f"   {k:22s} dur {len(x)/SR:6.1f}s  active-frames {np.mean(E[k] > -40):.2f}")
        def rep(a, b, label):
            if a in E and b in E:
                (c, lag), c0 = xcorr(E[a], E[b])
                print(f"   {label:46s} best r={c:+.2f} at lag {lag*HOP/SR:+.2f}s   r@0={c0:+.2f}")
        rep("conv_rec_self_local", "conv_rec_other_local", "self_local vs other_local (diff speakers)")
        rep("conv_rec_other_local", "conv_rec_other_remote", "other_local vs other_remote (same voice O)")
        rep("conv_rec_self_local", "conv_rec_self_remote", "self_local vs self_remote (same voice S)")
        rep("conv_rec_self_local", "conv_rec_other_remote", "self_local vs other_remote (S's machine)")


if __name__ == "__main__":
    main(sys.argv[1:])
