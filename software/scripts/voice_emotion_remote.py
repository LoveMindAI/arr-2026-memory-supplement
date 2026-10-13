#!/usr/bin/env python3
"""Rented-GPU runner for the two speech-emotion models, reading voice_pack_v2.py output (lossless FLAC of the exact turn
windows). Same computation as voice_extract_v2.py: audEERING arousal/dominance/valence with <=10 s windows, per-window
processor normalization, attention-masked mean pooling, duration-weighted mean per turn (adv_batch); emotion2vec_plus_large
utterance scores per turn. Turns of <= 1,600 samples get NaN, as locally. Writes <out>/<dyad>__adv.npz and __e2v.npz (key Y).
Usage: python3 voice_emotion_remote.py PACK_DIR OUT_DIR"""
import glob, json, os, sys, time
import numpy as np, soundfile as sf, torch, torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
SR = 16000
E2V = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
REV = "6eba34a2485ea31cb03600241787c3a5edab8626"


class RegressionHead(nn.Module):
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


def main(pack, out):
    os.makedirs(out, exist_ok=True); dev = "cuda"
    proc = Wav2Vec2Processor.from_pretrained(NAME, revision=REV)
    model = EmotionModel.from_pretrained(NAME, revision=REV).to(dev).eval()
    from funasr import AutoModel
    e2v = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True, device=dev, disable_pbar=True, log_level="ERROR")
    t00 = time.time(); n = 0
    while True:                                   # the pack may still be arriving: only read complete files, loop until done
        files = sorted(glob.glob(os.path.join(pack, "*.json")))
        todo = []
        for jf in files:
            d = os.path.basename(jf)[:-5]
            if os.path.exists(os.path.join(out, f"{d}__adv.npz")) and os.path.exists(os.path.join(out, f"{d}__e2v.npz")): continue
            try:
                offs = json.load(open(jf))["offsets"]; need = sum(L for _, L in offs)
                if sf.info(jf[:-5] + ".flac").frames < need: continue
            except Exception:
                continue
            todo.append(jf)
        if not todo:
            if os.path.exists(os.path.join(os.path.dirname(pack.rstrip("/")), "UPLOAD_DONE")) and len(files) >= 557: break
            time.sleep(10); continue
        for jf in todo:
            n += 1; process(jf, out, proc, model, e2v, dev, n, len(files), t00)
    print("DONE remote")


def process(jf, out, proc, model, e2v, dev, n, nfiles, t00):
    if True:
        d = os.path.basename(jf)[:-5]
        fa, fb = os.path.join(out, f"{d}__adv.npz"), os.path.join(out, f"{d}__e2v.npz")
        x, _ = sf.read(jf[:-5] + ".flac", dtype="float32"); offs = json.load(open(jf))["offsets"]
        segs = [x[a:a + L] for a, L in offs]
        # audEERING, batched + masked (identical to voice_extract_v2._Emo.adv_batch)
        items = []
        for ti, s in enumerate(segs):
            if len(s) <= 1600: continue
            for i in range(0, len(s), 10 * SR):
                c = s[i:i + 10 * SR]
                if len(c) < SR and i > 0: continue
                items.append((ti, proc(c, sampling_rate=SR)["input_values"][0].astype(np.float32)))
        items.sort(key=lambda z: len(z[1])); res = {}; k = 0
        while k < len(items):
            L = len(items[k][1]); m = max(1, min(len(items) - k, (SR * 240) // max(L, 1)))
            batch = items[k:k + m]; L = max(len(z[1]) for z in batch)
            X = np.zeros((len(batch), L), np.float32); M = np.zeros((len(batch), L), np.int64)
            for b, (_, z) in enumerate(batch): X[b, :len(z)] = z; M[b, :len(z)] = 1
            with torch.no_grad():
                xt, mt = torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev)
                h = model.wav2vec2(xt, attention_mask=mt)[0]
                fm = model.wav2vec2._get_feature_vector_attention_mask(h.shape[1], mt).unsqueeze(-1).to(h.dtype)
                y = model.classifier((h * fm).sum(1) / fm.sum(1).clamp(min=1)).cpu().numpy()
            for b, (ti, z) in enumerate(batch): res.setdefault(ti, []).append((y[b], len(z)))
            k += m
        A = np.full((len(segs), 3), np.nan)
        for ti, Lw in res.items(): A[ti] = np.average(np.array([a for a, _ in Lw]), axis=0, weights=[w for _, w in Lw])
        np.savez_compressed(fa, Y=A)
        B = np.full((len(segs), len(E2V)), np.nan)
        for ti, s in enumerate(segs):
            if len(s) <= 1600: continue
            r = e2v.generate(s, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
            lab = ["unknown" if l == "<unk>" else l.split("/")[-1].lower() for l in r["labels"]]
            sc = dict(zip(lab, r["scores"])); B[ti] = [sc.get(q, np.nan) for q in E2V]
        np.savez_compressed(fb, Y=B)
        print(f"{n} done (of {nfiles} seen) {d} {len(segs)} turns | {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
