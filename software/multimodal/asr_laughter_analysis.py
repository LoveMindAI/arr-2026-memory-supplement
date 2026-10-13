#!/usr/bin/env python3
"""Transcript instruments and laughter (A1, Oct 10). Reads multimodal/asr_v3/ (Scribe v2 direct with word + audio-event
timestamps; MAI-Transcribe-2 text) and transcripts_v2 (Whisper large-v3 words). Writes analysis/asr_laughter.json.
1. Agreement: word error rate between each pair of ASR instruments, per speaker track (same source audio).
2. Fillers: um/uh/er/erm/hmm counts per instrument.
3. Laughter: Scribe audio events (laughs, chuckles, giggles ...) placed in the v2 turn structure by time; laughter by the turn's
   speaker vs by the listener.
4. Remembered turns: does laughter in a turn's window mark the stretch the person later reported? Same unit and controls as
   scripts/memory_signal.py (located in-conversation memories; non-backchannel turns; own turn, log words, position, position^2),
   logit with SEs clustered by perspective.
5. Reports about laughter or humor: does the amount of laughter in a call predict a report that mentions laughing or a joke?"""
import glob, json, math, os, re, sys, collections
import numpy as np
from rapidfuzz.distance import Levenshtein
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import prompts as P
A = os.path.join(ROOT, "multimodal", "asr_v3")
FILL = re.compile(r"^(um+|uh+|erm+|er+|hmm+|mm+|ah+)$")
LAUGHY = re.compile(r"laugh|chuckl|giggl|snicker|cackl|snort", re.I)
BC = re.compile(r"^(yeah|yes|yep|ok|okay|right|mhm|mm-?hmm|uh-?huh|oh|wow|sure|cool|nice|really|lovely|ah|aw|ha)([ ,.!?]+(yeah|yes|right|okay|ok|wow|really|nice|lovely))*[.!?,]*$", re.I)
HUMOR = re.compile(r"\b(laugh\w*|joke\w*|joking|funny|humou?r\w*|giggl\w*|chuckl\w*|banter|hilarious|amus\w*|lol|haha\w*)\b", re.I)


def toks(s):
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s or "").lower().replace("’", "'")
    return [t for t in re.sub(r"[^a-z0-9' ]+", " ", s).split() if t]


def wer(ref, hyp):
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def load(dyad, pid, model):
    p = os.path.join(A, f"{dyad}__{pid}__{model}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    want = {"scribe-v2": "elevenlabs/scribe-v2", "mai-transcribe-2": "microsoft/mai-transcribe-2"}[model]
    assert j.get("_model") == want and j.get("_enc") == "mp3_16k_48kbps", (p, j.get("_model"), j.get("_via"))   # provenance check (Oct 10 routing bug)
    return j if j.get("text") is not None else None


def main():
    out = {"agreement": {}, "fillers": {}, "laughter": {}, "remembered_turns": {}, "humor_reports": {}}
    pairs = collections.defaultdict(list); fills = collections.defaultdict(list); pooled = collections.Counter()
    laughs = {}                                       # dyad -> list of (spk, start, end, label)
    ev_labels = collections.Counter(); n_tracks = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "transcripts_v2", "*.json"))):
        j = json.load(open(f)); d = j["dyadID"]; laughs[d] = []
        for pid in j["meta"]:
            sc, mai = load(d, pid, "scribe-v2"), load(d, pid, "mai-transcribe-2")
            if not sc or not mai or not sc.get("words"):
                continue
            n_tracks += 1
            wh = toks(" ".join(w["word"] for w in j["words"] if w["spk"] == pid))
            scw = toks(" ".join(w["text"] for w in sc["words"] if w.get("type") == "word"))
            mw = toks(mai["text"])
            pairs["whisper_vs_scribe"].append(wer(scw, wh)); pairs["whisper_vs_mai"].append(wer(mw, wh)); pairs["scribe_vs_mai"].append(wer(scw, mw))
            nf = lambda t: [x for x in t if not FILL.match(x)]
            whn, scn, mwn = nf(wh), nf(scw), nf(mw)
            pairs["nofill_whisper_vs_scribe"].append(wer(scn, whn)); pairs["nofill_whisper_vs_mai"].append(wer(mwn, whn)); pairs["nofill_scribe_vs_mai"].append(wer(scn, mwn))
            ops = Levenshtein.editops(scn, whn)
            pooled["ref_words"] += len(scn)
            for o in ops: pooled[o.tag] += 1
            for tag in ("delete", "replace", "insert"):
                pairs[f"nofill_whisper_{tag}s_per_word"].append(sum(o.tag == tag for o in ops) / max(1, len(scn)))
            for name, t in (("whisper", wh), ("scribe", scw), ("mai", mw)):
                fills[name].append(sum(bool(FILL.match(x)) for x in t) / max(1, len(t)) * 100)
            shift = -0.12 if j["meta"][pid].get("kind") == "remote_copy" else 0.0     # same shift transcribe_v2 applied
            for w in sc["words"]:
                if w.get("type") == "audio_event":
                    ev_labels[w["text"].lower()] += 1
                    if LAUGHY.search(w["text"]):
                        laughs[d].append((pid, w["start"] + shift, w["end"] + shift, w["text"]))
    q = lambda v: dict(median=round(float(np.median(v)), 3), mean=round(float(np.mean(v)), 3), p25=round(float(np.percentile(v, 25)), 3),
                       p75=round(float(np.percentile(v, 75)), 3), n=len(v))
    out["agreement"] = {k: q(v) for k, v in pairs.items()}
    out["agreement"]["pooled_nofill_whisper_vs_scribe"] = {k: (round(pooled[k] / pooled["ref_words"], 4) if k != "ref_words" else pooled[k])
                                                          for k in ("ref_words", "delete", "replace", "insert")}
    out["agreement"]["note_nofill"] = "nofill_*: same, after dropping um/uh/er/erm/hmm/mm/ah from all three; whisper deletes/replaces/inserts are against Scribe as reference"
    out["agreement"]["note"] = "word error rate of the second instrument against the first (reference) on the same speaker track; tags removed"
    out["fillers"] = {k: q(v) for k, v in fills.items()}; out["fillers"]["note"] = "fillers per 100 words per track"
    out["laughter"]["event_labels_top"] = ev_labels.most_common(25); out["laughter"]["n_tracks"] = n_tracks
    nl = [len(v) for v in laughs.values() if v is not None]
    out["laughter"]["per_call"] = q(nl); out["laughter"]["calls_with_any"] = round(float(np.mean([x > 0 for x in nl])), 3)

    rows = {r["perspectiveID"]: r for r in P.rows()}
    X, y, groups = [], [], []; own_l, lis_l = 0, 0
    for fo in glob.glob(os.path.join(ROOT, "analysis", "located", "PID_*.json")):
        o = json.load(open(fo))
        if o.get("kind") != "in_conversation" or not o.get("turns"):
            continue
        r = rows.get(o["perspectiveID"])
        if not r or r["dyadID"] not in laughs:
            continue
        j = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json"))); turns = j["turns"]; dur = max(t["end"] for t in turns)
        L = laughs[r["dyadID"]]
        span = set(range(min(o["turns"]), max(o["turns"]) + 1))
        for i, t in enumerate(turns):
            if t.get("bc") or BC.match(t["text"].strip()):
                continue
            nxt_start = next((u["start"] for u in turns[i + 1:] if not u.get("bc")), t["end"])
            a, b = t["start"], max(t["end"], nxt_start)
            mid = lambda e: (e[1] + e[2]) / 2
            by_spk = sum(1 for e in L if e[0] == t["spk"] and a <= mid(e) < b)
            by_lis = sum(1 for e in L if e[0] != t["spk"] and a <= mid(e) < b)
            own_l += by_spk; lis_l += by_lis
            n = max(1, len(t["text"].split())); pos = t["start"] / dur
            X.append([1.0, float(t["spk"] == r["selfPID"]), float(by_spk > 0), float(by_lis > 0), math.log(n), pos, pos * pos])
            y.append(float(i in span)); groups.append(o["perspectiveID"])
    X, y = np.array(X), np.array(y)
    import statsmodels.api as sm
    names = ["const", "own_turn", "speaker_laughs", "listener_laughs", "log_words", "rel_pos", "rel_pos^2"]
    m = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": np.unique(groups, return_inverse=True)[1]})
    ci = m.conf_int()
    rt = out["remembered_turns"]
    rt.update(perspectives=len(set(groups)), turns=int(len(y)), remembered_rate=round(float(y.mean()), 3),
              share_turns_speaker_laughs=round(float(X[:, 2].mean()), 3), share_turns_listener_laughs=round(float(X[:, 3].mean()), 3),
              laugh_events_in_turn_windows=dict(by_speaker=own_l, by_listener=lis_l))
    for k, nm in enumerate(names[1:4], 1):
        rt[nm] = dict(odds_ratio=round(float(math.exp(m.params[k])), 3), ci=[round(float(math.exp(ci[k][0])), 3), round(float(math.exp(ci[k][1])), 3)], p=float(m.pvalues[k]))
    for k, nm in ((2, "speaker_laughs"), (3, "listener_laughs")):
        rt[nm]["remembered_rate_with"] = round(float(y[X[:, k] == 1].mean()), 3) if (X[:, k] == 1).any() else None
        rt[nm]["remembered_rate_without"] = round(float(y[X[:, k] == 0].mean()), 3)

    H, Z, G = [], [], []
    for pid_, r in rows.items():
        if r["dyadID"] not in laughs or r["dyadID"] == "PID_n+PID_n":
            continue
        H.append(float(bool(HUMOR.search(r["conv_SP_Mem_descr"] + " " + r["conv_SP_Mem_reason"]))))
        Z.append([1.0, math.log1p(len(laughs[r["dyadID"]]))]); G.append(r["selfPID"])
    H, Z = np.array(H), np.array(Z)
    m2 = sm.Logit(H, Z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": np.unique(G, return_inverse=True)[1]})
    c2 = m2.conf_int()
    out["humor_reports"] = dict(reports=int(len(H)), rate=round(float(H.mean()), 3),
                                log1p_laughs_odds_ratio=round(float(math.exp(m2.params[1])), 3),
                                ci=[round(float(math.exp(c2[1][0])), 3), round(float(math.exp(c2[1][1])), 3)], p=float(m2.pvalues[1]),
                                note="report (moment or reason) mentions laughing/joke/funny etc.; predictor log(1 + Scribe laugh events in the call); clustered by person")
    json.dump(out, open(os.path.join(ROOT, "analysis", "asr_laughter.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
