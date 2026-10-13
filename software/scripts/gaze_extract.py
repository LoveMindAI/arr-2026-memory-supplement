#!/usr/bin/env python3
"""Per-frame face, head-pose and eye-direction signals from the PARSEL call videos, for a gaze-aversion test
(does the rememberer look away from the screen more during the stretch they later report?).

Input: media/<perspectiveID>/video_silent.mp4, 640x240 at 10 fps; LEFT half = partner (as the rememberer saw them),
RIGHT half = rememberer (media/*/meta.json "layout"). Each half is upscaled 2x before landmarking.
Model: MediaPipe FaceLandmarker (VIDEO mode) with blendshapes and the facial transformation matrix.
Per sampled frame and per person: face found (0/1), head yaw and pitch in degrees (from the transformation matrix),
and the eye-look blendshapes (in/out/up/down, both eyes). Nothing visual is saved, only these numbers.
Output: analysis/gaze/<perspectiveID>.npz (arrays t, and for side in {self, partner}: found, yaw, pitch, look_* ...).

Usage: gaze_extract.py [--n 20] [--fps 5] [--all] [--workers 6]
"""
import argparse, glob, json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL = os.environ.get("FACE_MODEL", "<LOCAL_TMP>")
OUT = os.path.join(ROOT, "analysis", "gaze"); os.makedirs(OUT, exist_ok=True)
LOOK = ["eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight", "eyeBlinkLeft", "eyeBlinkRight"]


def euler(M):
    R = np.array(M)[:3, :3]
    yaw = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch


def run(pid, fps):
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    out_f = os.path.join(OUT, f"{pid}.npz")
    if os.path.exists(out_f): return pid, "cached"
    vid = os.path.join(ROOT, "media", pid, "video_silent.mp4")
    if not os.path.exists(vid): return pid, "no video"
    mk = lambda: vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    lm = {"partner": mk(), "self": mk()}
    cap = cv2.VideoCapture(vid); src_fps = cap.get(cv2.CAP_PROP_FPS) or 10; step = max(1, round(src_fps / fps))
    rec = {s: {k: [] for k in ["found", "yaw", "pitch"] + LOOK} for s in lm}; ts = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            t_ms = int(i * 1000 / src_fps); ts.append(t_ms / 1000)
            h, w = fr.shape[:2]
            for side, half in (("partner", fr[:, : w // 2]), ("self", fr[:, w // 2:])):
                img = cv2.cvtColor(cv2.resize(half, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
                r = lm[side].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)), t_ms)
                d = rec[side]
                if r.face_landmarks:
                    y, p = euler(r.facial_transformation_matrixes[0]); bs = {b.category_name: b.score for b in r.face_blendshapes[0]}
                    d["found"].append(1); d["yaw"].append(y); d["pitch"].append(p)
                    for k in LOOK: d[k].append(bs.get(k, np.nan))
                else:
                    d["found"].append(0); d["yaw"].append(np.nan); d["pitch"].append(np.nan)
                    for k in LOOK: d[k].append(np.nan)
        i += 1
    cap.release()
    arrs = {"t": np.array(ts)}
    for s, d in rec.items():
        for k, v in d.items(): arrs[f"{s}__{k}"] = np.array(v, dtype=float)
    np.savez_compressed(out_f, **arrs)
    return pid, f"{len(ts)} frames; face found self {np.mean(rec['self']['found']):.2f}, partner {np.mean(rec['partner']['found']):.2f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    pids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "media", "PID_*")))
    if not a.all: random.Random(20261009).shuffle(pids); pids = pids[: a.n]
    with ProcessPoolExecutor(a.workers) as ex:
        for pid, msg in ex.map(run, pids, [a.fps] * len(pids)): print(pid, msg, flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
