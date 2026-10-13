#!/usr/bin/env python3
"""The gate for the transcript ablation (TRANSCRIPT_ABLATION_PLAN.md §0.3): audio models LISTEN to short clips and judge which
transcript is more faithful, blind to source. Unit: one speaker's turn (v2 turn, 3-25 s) from that speaker's own track.
Candidates: Whisper (the v2 turn text) vs Scribe (Scribe words in the same window). Strata, 12 turns each, chosen at random
(seed 20261010) among turns of that kind, at most one per call:
  dropped   Scribe has >= 3 more non-filler words than Whisper (Whisper deletions dominate)
  swapped   >= 2 word substitutions, word counts within 2
  fillers   identical once fillers are removed; differ only in um/uh etc.
  laughter  Scribe marks a laugh in the turn (tags shown to the listener)
  extra     Whisper has >= 3 more non-filler words than Scribe (the opposite direction; includes background speech cases)
Judges: Gemini 3.1 Pro (Google) and Qwen3.8 Omni Flash (Alibaba), plus MiMo v2.6 Pro (Xiaomi) as a third family.
Each gets the clip and A/B in random order (per item, seeded), returns which is more faithful (A, B or equal) and what each misses.
Two-instrument rule: the difference is real if the judges, separately, prefer Scribe well above chance (binomial vs .5 over
decisive answers) and agree with each other. -> multimodal/gate/items.jsonl, judgments.jsonl, analysis/transcript_gate.json.
Clips stay local in multimodal/gate/clips (private; never released)."""
import glob, json, os, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz.distance import Levenshtein
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import orc
CONV = os.path.expanduser("<LOCAL_FILE> Story/Conv Data/Conversations")
G = os.path.join(ROOT, "multimodal", "gate"); os.makedirs(os.path.join(G, "clips"), exist_ok=True)
FILL = re.compile(r"^(um+|uh+|erm+|er+|hmm+|mm+|ah+)$")
LAUGH = re.compile(r"laugh|chuckl|giggl", re.I)
JUDGES = ["google/gemini-3.1-pro-preview", "qwen/qwen3.8-omni-flash", "xiaomi/mimo-v2.6-pro-ultraspeed"]
vids = {os.path.basename(p): p for p in glob.glob(CONV + "/*/*.mp4")}
toks = lambda s: [t for t in re.sub(r"[^a-z0-9' ]+", " ", re.sub(r"\[[^\]]*\]", " ", (s or "").lower().replace("’", "'"))).split() if t]
nf = lambda t: [x for x in t if not FILL.match(x)]


def candidates():
    C = {k: [] for k in ("dropped", "swapped", "fillers", "laughter", "extra")}
    for f in sorted(glob.glob(os.path.join(ROOT, "transcripts_v2", "*.json"))):
        j = json.load(open(f)); d = j["dyadID"]
        for pid, m in j["meta"].items():
            sc = json.load(open(os.path.join(ROOT, "multimodal", "asr_v3", f"{d}__{pid}__scribe-v2.json")))
            shift = -0.12 if m.get("kind") == "remote_copy" else 0.0
            SW = [dict(w, start=w["start"] + shift, end=w["end"] + shift) for w in sc["words"] if w.get("type") in ("word", "audio_event")]
            for t in j["turns"]:
                if t["spk"] != pid or t.get("bc") or not (3 <= t["end"] - t["start"] <= 25):
                    continue
                a, b = t["start"] - 0.3, t["end"] + 0.3
                inw = [w for w in SW if a <= (w["start"] + w["end"]) / 2 < b]
                s_txt = " ".join(w["text"] for w in inw); w_txt = t["text"]
                ws, ss = toks(w_txt), toks(s_txt)
                if not ss or not ws:
                    continue
                wn, sn = nf(ws), nf(ss)
                ops = Levenshtein.editops(sn, wn)
                dele = sum(o.tag == "delete" for o in ops); rep = sum(o.tag == "replace" for o in ops); ins = sum(o.tag == "insert" for o in ops)
                it = dict(dyad=d, pid=pid, src=m["source"], shift=shift, start=t["start"], end=t["end"], whisper=w_txt, scribe=s_txt)
                if any(w.get("type") == "audio_event" and LAUGH.search(w["text"]) for w in inw):
                    C["laughter"].append(it)
                elif dele >= 3 and dele > rep + ins:
                    C["dropped"].append(it)
                elif ins >= 3 and ins > rep + dele:
                    C["extra"].append(it)
                elif rep >= 2 and abs(len(wn) - len(sn)) <= 2:
                    C["swapped"].append(it)
                elif sn == wn and ws != ss:
                    C["fillers"].append(it)
    return C


def build():
    C = candidates(); rng = random.Random(20261010); items = []
    for k, L in C.items():
        rng.shuffle(L); seen = set()
        for it in L:
            if it["dyad"] in seen: continue
            seen.add(it["dyad"]); items.append(dict(it, stratum=k, id=f"{k}_{len(seen):02d}"))
            if len(seen) == 12: break
    for it in items:
        it["scribe_first"] = rng.random() < 0.5
        clip = os.path.join(G, "clips", it["id"] + ".mp3")
        a = max(0, it["start"] - 0.5 - it["shift"]); dur = it["end"] - it["start"] + 1.0
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-t", f"{dur:.2f}", "-i", vids[it["src"]], "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", clip], check=True)
        it["clip"] = clip
    with open(os.path.join(G, "items.jsonl"), "w") as fh:
        for it in items: fh.write(json.dumps(it) + "\n")
    print({k: len(v) for k, v in C.items()}, "->", len(items), "items")
    return items


PROMPT = """You will hear a short audio clip of ONE person speaking on a video call, recorded from their own microphone (the other person is mostly silent on this track). Below are two automatic transcripts of this clip, A and B. Listen carefully to the whole clip.

Transcript A:
{A}

Transcript B:
{B}

Which transcript more faithfully records what THIS speaker actually said in the clip: every word they said, including fillers like "um" or "uh", false starts, and (if a transcript marks them) non-speech sounds such as laughter? A transcript should not get credit for words this speaker did not say (for example background speech, a TV, or words cut from outside the clip's edges).

Answer in JSON only: {{"more_faithful": "A" or "B" or "equal", "A_misses_or_errs": "short description", "B_misses_or_errs": "short description", "confidence": "high" or "medium" or "low"}}"""


def judge(args):
    it, jm = args
    out = os.path.join(G, "judgments", f"{it['id']}__{jm.split('/')[1]}.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and json.load(open(out)).get("parsed"):
        return
    A, B = (it["scribe"], it["whisper"]) if it["scribe_first"] else (it["whisper"], it["scribe"])
    msgs = [{"role": "user", "content": [orc.audio_part(it["clip"], jm), orc.text_part(PROMPT.format(A=A, B=B))]}]
    r = orc.call(jm, msgs, os.path.join(G, "calls.jsonl"), tag=f"{it['id']}|{jm}", max_tokens=4000, temperature=0)
    t = r.get("text") or ""; s, e = t.find("{"), t.rfind("}")
    try: p = json.loads(t[s:e + 1])
    except Exception: p = None
    pick = None
    if p and p.get("more_faithful") in ("A", "B", "equal"):
        mf = p["more_faithful"]; pick = "equal" if mf == "equal" else ("scribe" if (mf == "A") == it["scribe_first"] else "whisper")
    json.dump(dict(id=it["id"], judge=jm, stratum=it["stratum"], parsed=p, pick=pick, cost=orc.cost(r)), open(out, "w"))


def analyze(items):
    from scipy.stats import binomtest
    J = [json.load(open(p)) for p in glob.glob(os.path.join(G, "judgments", "*.json"))]
    res = {"by_judge": {}, "by_judge_stratum": {}, "agreement": {}}
    for jm in JUDGES:
        js = [x for x in J if x["judge"] == jm and x["pick"]]
        def summ(xs):
            s = sum(x["pick"] == "scribe" for x in xs); w = sum(x["pick"] == "whisper" for x in xs); e = sum(x["pick"] == "equal" for x in xs)
            p = binomtest(s, s + w, 0.5).pvalue if s + w else None
            return dict(scribe=s, whisper=w, equal=e, scribe_share_decisive=round(s / (s + w), 3) if s + w else None, p=p)
        res["by_judge"][jm] = summ(js)
        for k in ("dropped", "swapped", "fillers", "laughter", "extra"):
            res["by_judge_stratum"][f"{jm}|{k}"] = summ([x for x in js if x["stratum"] == k])
    by = {}
    for x in J:
        if x["pick"]: by.setdefault(x["id"], {})[x["judge"]] = x["pick"]
    for i, a in enumerate(JUDGES):
        for b in JUDGES[i + 1:]:
            both = [v for v in by.values() if a in v and b in v]
            res["agreement"][f"{a.split('/')[1]} vs {b.split('/')[1]}"] = dict(n=len(both), same=round(sum(v[a] == v[b] for v in both) / max(1, len(both)), 3))
    res["n_items"] = len(items); res["cost"] = round(sum(x.get("cost") or 0 for x in J), 3)
    json.dump(res, open(os.path.join(ROOT, "analysis", "transcript_gate.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:3500])


if __name__ == "__main__":
    items = build() if not os.path.exists(os.path.join(G, "items.jsonl")) else [json.loads(l) for l in open(os.path.join(G, "items.jsonl"))]
    with ThreadPoolExecutor(24) as ex:
        list(ex.map(judge, [(it, jm) for it in items for jm in JUDGES]))
    analyze(items)
