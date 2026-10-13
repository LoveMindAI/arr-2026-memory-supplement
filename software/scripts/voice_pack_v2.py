#!/usr/bin/env python3
"""Pack the exact turn windows the emotion models read (voice_extract_v2.seg on each speaker's own-mic track) as lossless
16 kHz FLAC per dyad + offsets, for a rented GPU if the Mac is too contended. Only dyads still missing __adv or __e2v.
-> multimodal/voice_pack/<dyad>.flac and <dyad>.json (list of [start_sample, n_samples] per turn, in turn order)."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_extract_v2 as VX
OUT = os.path.join(VX.ROOT, "multimodal", "voice_pack"); os.makedirs(OUT, exist_ok=True)


def pack(d):
    import soundfile as sf
    if os.path.exists(os.path.join(OUT, d + ".json")): return d, "cached"
    j = json.load(open(os.path.join(VX.TXD, d + ".json"))); T = VX.dyad_tracks(j, VX.vids())
    parts, off, pos = [], [], 0
    for k, t, a, b in VX.windows(j["turns"]):
        s = VX.seg(T[t["spk"]], a, b); parts.append(s); off.append([pos, len(s)]); pos += len(s)
    x = np.concatenate(parts) if parts else np.zeros(1, np.float32)
    sf.write(os.path.join(OUT, d + ".flac"), np.clip(x, -1, 1), VX.SR, subtype="PCM_16")
    json.dump({"dyad": d, "offsets": off}, open(os.path.join(OUT, d + ".json"), "w"))
    return d, f"{len(off)} turns {pos / VX.SR:.0f}s"


if __name__ == "__main__":
    import prompts as P
    rows = {r["perspectiveID"]: r for r in P.rows()}
    dyads = sorted({rows[p]["dyadID"] for p in VX.located()})
    need = [d for d in dyads if not (os.path.exists(os.path.join(VX.DY, f"{d}__adv.npz")) and os.path.exists(os.path.join(VX.DY, f"{d}__e2v.npz")))]
    print("dyads needing emotion", len(need), flush=True)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for d, m in ex.map(pack, need): print(d, m, flush=True)
    print("DONE pack")
