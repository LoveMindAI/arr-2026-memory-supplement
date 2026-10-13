#!/usr/bin/env python3
"""Merge the rented-GPU emotion outputs (multimodal/voice_remote_out/) into analysis/voice_v2/dyad/. A dyad already computed on
the Mac keeps the Mac file; for those, CUDA vs MPS agreement is recorded (same code path, device check). Writes
multimodal/VOICE_DEVICE_PARITY.json."""
import glob, json, os, shutil
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "multimodal", "voice_remote_out"); DST = os.path.join(ROOT, "analysis", "voice_v2", "dyad")
par = {"adv": [], "e2v": []}; copied = {"adv": 0, "e2v": 0}
for f in sorted(glob.glob(os.path.join(SRC, "*.npz"))):
    b = os.path.basename(f); kind = "adv" if b.endswith("__adv.npz") else "e2v"; dst = os.path.join(DST, b)
    if os.path.exists(dst):
        R, L = np.load(f)["Y"], np.load(dst)["Y"]; m = np.isfinite(R).all(1) & np.isfinite(L).all(1)
        if m.sum(): par[kind].append(float(np.abs(R[m] - L[m]).max()))
    else:
        shutil.copyfile(f, dst); copied[kind] += 1
out = {"copied": copied, "parity_n_dyads": {k: len(v) for k, v in par.items()},
       "parity_max_abs_diff": {k: (round(max(v), 5) if v else None) for k, v in par.items()},
       "parity_median_max_abs_diff": {k: (round(float(np.median(v)), 5) if v else None) for k, v in par.items()}}
json.dump(out, open(os.path.join(ROOT, "multimodal", "VOICE_DEVICE_PARITY.json"), "w"), indent=1); print(out)
