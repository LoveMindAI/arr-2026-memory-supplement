#!/usr/bin/env python3
"""Voice v2 (A1, Oct 10: "the voice stuff done right from the right source"). Per transcript turn, every speaker is measured
on THEIR OWN microphone track: the exact source transcribe_v2.py used for that speaker's words (transcripts_v2/<dyad>.json
meta[PID].source; kind self_local = own machine, or remote_copy = their voice as recorded on the partner's machine, shifted
-0.12 s as in transcribe_v2.py; those tracks are flagged). Old voice_extract.py measured the partner on the RECEIVED stream.

Stage cpu (ProcessPool): openSMILE eGeMAPSv02 functionals (same 10 as voice_extract.py) + Praat via parselmouth
  (F0 mean and SD in semitones re 27.5 Hz, intensity mean and SD in dB) + the listener's own-track level during the turn.
Stage emo (one process, MPS): audEERING wav2vec2-large-robust-12-ft-emotion-msp-dim (arousal, dominance, valence; turns
  longer than 10 s are split into <=10 s windows and averaged by duration) and emotion2vec_plus_large (9-class scores, FunASR).
Raw per-dyad arrays -> analysis/voice_v2/dyad/; assemble -> analysis/voice_v2/<perspectiveID>.npz (speaker features centered on
each speaker's own mean in the call; listener level centered over the call), same turn order as transcripts_v2 turns.
Usage: voice_extract_v2.py --stage cpu|adv|e2v|assemble [--n 10] [--all] [--workers 12]"""
import argparse, glob, json, os, random, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONV = os.path.expanduser("<LOCAL_FILE> Story/Conv Data")
TXD = os.path.join(ROOT, "transcripts_v2")
OUT = os.path.join(ROOT, "analysis", "voice_v2"); DY = os.path.join(OUT, "dyad"); os.makedirs(DY, exist_ok=True)
SR = 16000
KEEP = ["F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm", "loudness_sma3_amean", "loudness_sma3_stddevNorm",
        "VoicedSegmentsPerSec", "MeanUnvoicedSegmentLength", "jitterLocal_sma3nz_amean", "shimmerLocaldB_sma3nz_amean",
        "HNRdBACF_sma3nz_amean", "slopeV0-500_sma3nz_amean"]
CPU_NAMES = ["pitch", "pitch_var", "loud", "loud_var", "voiced_rate", "pause_len", "jitter", "shimmer", "hnr", "slope",
             "praat_f0_st", "praat_f0_sd_st", "praat_int_db", "praat_int_sd_db", "listener_loud"]
EMO_A = ["arousal", "dominance", "valence"]
E2V = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]


def vids():
    return {os.path.basename(p)[:-4]: p for p in glob.glob(CONV + "/Conversations/*/*.mp4")}


def load_track(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def dyad_tracks(j, V):
    """{PID: (signal, shift_s, kind, source)} from the transcript's own provenance."""
    out = {}
    for pid, m in j["meta"].items():
        src = m.get("source")
        if not src:
            out[pid] = (None, 0.0, None, None); continue
        path = V.get(src[:-4] if src.endswith(".mp4") else src)
        shift = -0.12 if m.get("kind") == "remote_copy" else 0.0
        out[pid] = (load_track(path) if path else None, shift, m.get("kind"), src)
    return out


def windows(turns):
    for k, t in enumerate(turns):
        a = t["start"]; b = turns[k + 1]["start"] if k + 1 < len(turns) else a + 5; b = max(b, a + 0.5)
        yield k, t, a, b


def seg(tr, a, b):
    x, shift = tr[0], tr[1]
    if x is None:
        return np.zeros(0, np.float32)
    i, j = int(max(0, a - shift) * SR), int(max(0, b - shift) * SR)          # transcript time t = audio time + shift
    return x[i:j]


def praat_feats(s):
    import parselmouth
    snd = parselmouth.Sound(s.astype(np.float64), sampling_frequency=SR)
    f0 = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600).selected_array["frequency"]
    f0 = f0[f0 > 0]
    st = 12 * np.log2(f0 / 27.5) if len(f0) else np.array([])
    it = snd.to_intensity(minimum_pitch=75).values.flatten()
    it = it[np.isfinite(it)]
    return [float(st.mean()) if len(st) >= 3 else np.nan, float(st.std()) if len(st) >= 3 else np.nan,
            float(it.mean()) if len(it) else np.nan, float(it.std()) if len(it) else np.nan]


def run_cpu(dyad):
    import opensmile
    of = os.path.join(DY, f"{dyad}__cpu.npz")
    if os.path.exists(of): return dyad, "cached"
    t0 = time.time(); j = json.load(open(os.path.join(TXD, dyad + ".json"))); turns = j["turns"]
    T = dyad_tracks(j, vids())
    sm = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    X, spk = [], []
    for k, t, a, b in windows(turns):
        s = seg(T[t["spk"]], a, b); row = [np.nan] * (len(KEEP) + 4)
        if len(s) > 1600:
            try:
                f = sm.process_signal(s, SR); row[:len(KEEP)] = [float(f[c].iloc[0]) for c in KEEP]
            except Exception:
                pass
            try:
                row[len(KEEP):] = praat_feats(s)
            except Exception:
                pass
        lis = [p for p in T if p != t["spk"]]
        ls = seg(T[lis[0]], a, b) if lis else np.zeros(0)
        row.append(float(20 * np.log10(np.sqrt(np.mean(ls ** 2)) + 1e-6)) if len(ls) else np.nan)
        X.append(row); spk.append(t["spk"])
    kinds = {p: T[p][2] for p in T}; srcs = {p: T[p][3] for p in T}
    np.savez_compressed(of, X=np.array(X, float), spk=np.array(spk), kinds=json.dumps(kinds), sources=json.dumps(srcs))
    return dyad, f"{len(turns)} turns {time.time() - t0:.1f}s"


class _Emo:
    def __init__(self, which):
        import torch, torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class RegressionHead(nn.Module):                     # verbatim structure from the model card
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size); self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features, **kwargs):
                x = self.dropout(features); x = torch.tanh(self.dense(x)); x = self.dropout(x); return self.out_proj(x)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config); self.config = config
                self.wav2vec2 = Wav2Vec2Model(config); self.classifier = RegressionHead(config); self.init_weights()

            def forward(self, input_values):
                h = self.wav2vec2(input_values)[0]; h = torch.mean(h, dim=1); return h, self.classifier(h)

        name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.torch = torch; self.e2v = None
        if which == "e2v":
            from funasr import AutoModel
            self.e2v = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True, device="cpu", disable_pbar=True, log_level="ERROR")
            return
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = Wav2Vec2Processor.from_pretrained(name)
        self.model = EmotionModel.from_pretrained(name).to(self.dev).eval()

    def adv(self, s):
        out, w = [], []
        for i in range(0, len(s), 10 * SR):
            c = s[i:i + 10 * SR]
            if len(c) < SR and i > 0: continue
            y = self.proc(c, sampling_rate=SR)["input_values"][0].reshape(1, -1)
            with self.torch.no_grad():
                out.append(self.model(self.torch.from_numpy(y).to(self.dev))[1].cpu().numpy()[0]); w.append(len(c))
        return np.average(np.array(out), axis=0, weights=w) if out else np.full(3, np.nan)

    def adv_batch(self, segs, max_samples=SR * 120):
        """Batched version of adv(): every turn split into <=10 s windows (windows < 1 s dropped unless the first), each window
        normalized on its own by the processor (as in adv), windows sorted by length, padded, attention-masked; hidden states
        mean-pooled over valid frames only, then the model's own regression head. Duration-weighted mean per turn."""
        torch = self.torch
        items = []
        for ti, s in enumerate(segs):
            if s is None or len(s) <= 1600: continue
            for i in range(0, len(s), 10 * SR):
                c = s[i:i + 10 * SR]
                if len(c) < SR and i > 0: continue
                items.append((ti, self.proc(c, sampling_rate=SR)["input_values"][0].astype(np.float32)))
        out = {}
        items.sort(key=lambda x: len(x[1]))
        k = 0
        while k < len(items):
            L = len(items[k][1]); n = max(1, min(len(items) - k, max_samples // max(L, 1)))
            batch = items[k:k + n]; L = max(len(x[1]) for x in batch)
            X = np.zeros((len(batch), L), np.float32); M = np.zeros((len(batch), L), np.int64)
            for b, (_, x) in enumerate(batch): X[b, :len(x)] = x; M[b, :len(x)] = 1
            with torch.no_grad():
                xt, mt = torch.from_numpy(X).to(self.dev), torch.from_numpy(M).to(self.dev)
                h = self.model.wav2vec2(xt, attention_mask=mt)[0]
                fm = self.model.wav2vec2._get_feature_vector_attention_mask(h.shape[1], mt).unsqueeze(-1).to(h.dtype)
                pooled = (h * fm).sum(1) / fm.sum(1).clamp(min=1)
                y = self.model.classifier(pooled).cpu().numpy()
            for b, (ti, x) in enumerate(batch): out.setdefault(ti, []).append((y[b], len(x)))
            k += n
        res = np.full((len(segs), 3), np.nan)
        for ti, L in out.items():
            res[ti] = np.average(np.array([a for a, _ in L]), axis=0, weights=[w for _, w in L])
        return res

    def cat(self, s):
        if self.e2v is None: return np.full(len(E2V), np.nan)
        r = self.e2v.generate(s, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        lab = ["unknown" if l == "<unk>" else l.split("/")[-1].lower() for l in r["labels"]]
        sc = dict(zip(lab, r["scores"]))
        return np.array([sc.get(k, np.nan) for k in E2V], float)


def run_emo(dyads, which):
    """which = 'adv' (audEERING on MPS) or 'e2v' (emotion2vec+ on CPU); separate processes so GPU and CPU work concurrently."""
    E = _Emo(which); V = vids()
    for d in dyads:
        of = os.path.join(DY, f"{d}__{which}.npz")
        if os.path.exists(of): print(d, "cached", flush=True); continue
        t0 = time.time(); j = json.load(open(os.path.join(TXD, d + ".json"))); T = dyad_tracks(j, V)
        segs = [seg(T[t["spk"]], a, b) for k, t, a, b in windows(j["turns"])]
        if which == "adv":
            Y = E.adv_batch(segs)
        else:
            Y = np.array([E.cat(s) if len(s) > 1600 else np.full(len(E2V), np.nan) for s in segs], float)
        np.savez_compressed(of, Y=Y)
        print(d, f"{len(j['turns'])} turns {time.time() - t0:.1f}s", flush=True)


def located():
    P = []
    for f in sorted(glob.glob(os.path.join(ROOT, "analysis", "located", "PID_*.json"))):
        L = json.load(open(f))
        if L.get("kind") == "in_conversation": P.append(L["perspectiveID"])
    return P


def assemble():
    import prompts as P
    rows = {r["perspectiveID"]: r for r in P.rows()}; n = 0
    for pid in located():
        r = rows[pid]; d = r["dyadID"]
        fc, fa, fb = (os.path.join(DY, f"{d}__{x}.npz") for x in ("cpu", "adv", "e2v"))
        if not os.path.exists(fc): continue
        C = np.load(fc); spk = C["spk"]; nt = len(spk)                # emotion columns are NaN until their stage has run
        YA = np.load(fa)["Y"] if os.path.exists(fa) else np.full((nt, 3), np.nan)
        YB = np.load(fb)["Y"] if os.path.exists(fb) else np.full((nt, len(E2V)), np.nan)
        X = np.concatenate([C["X"], YA, YB], 1)
        who = np.where(spk == r["selfPID"], "self", "partner")
        nsp = X.shape[1] - 1 - 3 - len(E2V)                       # speaker features before listener_loud
        spk_cols = list(range(nsp)) + list(range(nsp + 1, X.shape[1]))
        for s in ("self", "partner"):
            m = who == s
            if m.any(): X[np.ix_(m, spk_cols)] = X[np.ix_(m, spk_cols)] - np.nanmean(X[np.ix_(m, spk_cols)], 0)
        X[:, nsp] = X[:, nsp] - np.nanmean(X[:, nsp])
        kinds = json.loads(str(C["kinds"]))
        np.savez_compressed(os.path.join(OUT, f"{pid}.npz"), X=X, who=who, names=np.array(CPU_NAMES + EMO_A + ["e2v_" + e for e in E2V]),
                            self_kind=kinds.get(r["selfPID"]), partner_kind=kinds.get(r["otherPID"]))
        n += 1
    print("assembled", n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    import prompts as P
    rows = {r["perspectiveID"]: r for r in P.rows()}
    dyads = sorted({rows[p]["dyadID"] for p in located()})
    if not a.all: random.Random(20261010).shuffle(dyads); dyads = sorted(dyads[: a.n])
    t0 = time.time()
    if a.stage == "cpu":
        with ProcessPoolExecutor(a.workers) as ex:
            for d, msg in ex.map(run_cpu, dyads): print(d, msg, flush=True)
    elif a.stage in ("adv", "e2v"):
        import torch; torch.set_num_threads(a.threads)
        run_emo(dyads[a.shard::a.nshards], a.stage)
    elif a.stage == "assemble":
        assemble()
    print("DONE", a.stage, len(dyads), "dyads", f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
