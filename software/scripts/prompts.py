#!/usr/bin/env python3
"""Prompt builder for the memory-prediction task. DRAFT v0.1 for A1's review (agent1, Oct 6 2026).

Dimensions:
  modality : text | audio | text_audio | text_audio_video
  info     : none | card | history | card_history      (what the model knows about the rememberer)
  frame    : seat (model IS the rememberer, 'you')  | observer (model predicts for 'the participant')
The question block reproduces the PARSEL survey items verbatim (surveyPARSEL_Qualtrics.docx, block "Conversation-task"),
except the character-limit note and the AffectButton, which is replaced by three numbers on its own -1..1 axes.
"""
import csv, json, os
import orc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MEDIA = os.path.join(ROOT, "media")
TX = os.path.join(ROOT, "transcripts_v2")
# MEMORY_DATA (Oct 13): read the memory reports and transcripts from another folder in the same layout, e.g. the public stand-in
# exported by twin_public/export_for_pipeline.py (conv.csv + transcripts/<dyadID>.json). Unset = the PARSEL data as used in the paper.
_DATA = os.environ.get("MEMORY_DATA")
if _DATA:
    TX = os.path.join(_DATA, "transcripts")
CONV = os.path.expanduser("<LOCAL_FILE> Story/Conv Data")
CARDS = {
    "card_2A": os.path.expanduser("<LOCAL_FILE>"),
    "card_psych_narrative": os.path.expanduser("<LOCAL_FILE>"
                                               "Canonical_PsychometricOnly/Canonical_Opus_Profiles_PsychometricOnly/v7_v4_ccv0_production_opus46_290pid/profiles"),
}
DEFAULT_CARD = "card_2A"

# ---------------------------------------------------------------- verbatim survey items (PARSEL)
Q_DESCR = ("Take a moment to reflect on your experiences during the interaction with the other person.\n"
           "Please recall and describe the moment you remember most clearly from the interaction. Describe this moment in as much "
           "detail as you can remember.\n"
           "The description could be about the content you've discussed, the other participants, your feelings, your words, others' "
           "words, timing, anything that happened throughout the interaction. (You can use up to a maximum of 4 sentences.)")
Q_REASON = "Why do you think you remembered this specific moment? (You can use up to a maximum of 4 sentences.)"
Q_TIMING = "When during your conversation did this moment occur? (Answer on a scale from 0 = Beginning, 50 = Middle, to 100 = End.)"
Q_SC2 = "This moment resulted in us getting along ... with each other. (1 = much worse, 2 = a little worse, 3 = neither worse nor better, 4 = a little better, 5 = much better)"
Q_SC3 = "This moment resulted in us trying ... to minimize interpersonal conflict. (1 = much less, 2 = a little less, 3 = neither less nor more, 4 = a little more, 5 = much more)"
Q_ENT1 = "This moment resulted in me feeling ... that we were a unit. (1 = much less, 2 = a little less, 3 = neither more nor less, 4 = a little more, 5 = much more)"
Q_FEEL = ("How did experiencing this moment make you feel? Give three numbers from -1 to 1: pleasure (very unpleasant -1 to very "
          "pleasant 1), arousal (very calm -1 to very activated 1), dominance (very submissive -1 to very dominant 1).")

SCHEMA = ('Reply with JSON only, no other text:\n{"moment": "...", "reason": "...", "timing": 0-100, "getting_along": 1-5, '
          '"minimize_conflict": 1-5, "unit": 1-5, "pleasure": -1..1, "arousal": -1..1, "dominance": -1..1}')

STUDY = ("You took part in a research study run online in the UK. You met other participants one at a time in short video calls "
         "of up to 3 minutes, to get to know each other. You have just finished one of these calls.")
STUDY_OBS = ("Two participants in a research study run online in the UK met in a short video call of up to 3 minutes, to get to know "
             "each other. We call them the participant and the partner. Right after the call, the participant answered the questions "
             "below about it.")


def _rows():
    if _DATA:
        return list(csv.DictReader(open(os.path.join(_DATA, "conv.csv"), encoding="utf-8")))
    return list(csv.DictReader(open(CONV + "/cleaned_curated_conv_data_combined_ffinal.csv", encoding="utf-8-sig")))


ROWS = None


def rows():
    global ROWS
    if ROWS is None:
        ROWS = _rows()
    return ROWS


def transcript(dyad, self_pid, observer=False):
    j = json.load(open(os.path.join(TX, dyad + ".json")))
    you, other = ("Participant", "Partner") if observer else ("You", "Partner")
    out = []
    for t in j["turns"]:
        who = you if t["spk"] == self_pid else other
        out.append(f"[{int(t['start'] // 60)}:{int(t['start'] % 60):02d}] {who}: {t['text']}")
    return "\n".join(out)


def card(pid, which=DEFAULT_CARD):
    p = os.path.join(CARDS[which], pid + ".txt")
    if not os.path.exists(p):
        # canonical narratives use other filename patterns
        c = [f for f in os.listdir(CARDS[which]) if f.startswith(pid + "_") or f.startswith(pid + ".")]
        if not c:
            return None
        p = os.path.join(CARDS[which], c[0])
    return open(p).read().strip()


def history(row, prior_only=False, max_items=None):
    """The rememberer's memories from their OTHER calls in this study (never this call). prior_only -> earlier rounds only."""
    me, this_round = row["selfPID"], int(row["roundIdx"])
    items = [r for r in rows() if r["selfPID"] == me and r["perspectiveID"] != row["perspectiveID"]]
    if prior_only:
        items = [r for r in items if int(r["roundIdx"]) < this_round]
    items.sort(key=lambda r: int(r["roundIdx"]))
    if max_items:
        items = items[-max_items:]
    return [dict(moment=r["conv_SP_Mem_descr"].strip(), reason=r["conv_SP_Mem_reason"].strip(), round=int(r["roundIdx"])) for r in items]


CARDS_OUT = os.path.join(ROOT, "cards", "run_cards1", "cards_out")
_RUNG = {}


W2 = os.path.join(ROOT, "cards", "weave", "run_weave1", "weave_out")
_R2 = {}


def round2_card(row, rung):
    """Oct 7 conditions (A1): impthin/impnarr = cards from 4 models' impressions; facts = bare fact list; weave = woven card; factnarr = who-not-how.
    Writer for weave/factnarr chosen by the bake-off: env WEAVE_WRITER (default lcw150)."""
    pid = row["selfPID"]
    if rung == "impthin":
        p = os.path.join(ROOT, "cards", "inputs", "thin_cards_impression", pid + ".txt")
        return open(p).read().strip() if os.path.exists(p) else None
    if rung == "facts":
        if "facts" not in _R2:
            _R2["facts"] = json.load(open(os.path.join(ROOT, "facts", "person_facts.json")))
        xs = _R2["facts"].get(pid)
        return "\n".join(f"- {x}" for x in xs) if xs else None
    fn, wr = {"weave": ("weave_all.jsonl", os.environ.get("WEAVE_WRITER", "lcw150")), "factnarr": ("factnarr.jsonl", os.environ.get("WEAVE_WRITER", "lcw150")),
              "impnarr": ("cards_impression.jsonl", "lcw150")}[rung]
    if rung not in _R2:
        _R2[rung] = {}
        for l in open(os.path.join(W2, fn)):
            d = json.loads(l)
            if d["writer"] == wr and d.get("finish") == "stop":
                _R2[rung][d["pid"]] = d["text"].strip()
    return _R2[rung].get(pid)


def rung_card(row, rung):
    if rung in ("impthin", "facts", "weave", "factnarr", "impnarr"):
        return round2_card(row, rung)
    if rung == "thin":
        p = os.path.join(ROOT, "cards", "inputs", "thin_cards", row["selfPID"] + ".txt")
        return open(p).read().strip() if os.path.exists(p) else None
    if not _RUNG:
        for l in open(os.path.join(CARDS_OUT, "cards_lcw150.jsonl")):
            d = json.loads(l); _RUNG[("narr", d["pid"])] = d["text"].strip()
        pp = os.path.join(CARDS_OUT, "cards_person.jsonl")
        for l in (open(pp) if os.path.exists(pp) else []):
            d = json.loads(l); _RUNG[("person", d["perspectiveID"])] = d["text"].strip()
    return _RUNG.get(("narr", row["selfPID"])) if rung == "narr" else _RUNG.get(("person", row["perspectiveID"]))


def donor(row, need_hist=0, rung=None):
    """Oct 8 controls (A1: "Provide the wrong history, provide the wrong profile"): a different participant from a DIFFERENT batch
    (so they never met this partner), seeded by the target perspective. need_hist: donor must have at least that many memories;
    rung: donor must have that card."""
    import hashlib, random as _r
    me = row["selfPID"]
    by = {}
    for r in rows():
        by.setdefault(r["selfPID"], []).append(r)
    cands = sorted(q for q, rs in by.items() if q != me and q != row.get("otherPID") and all(x["batchID"] != row["batchID"] for x in rs)
                   and len(rs) >= need_hist)
    rng = _r.Random(int(hashlib.sha256((row["perspectiveID"] + "|" + str(rung)).encode()).hexdigest()[:8], 16)); rng.shuffle(cands)
    for q in cands:
        if rung is None or rung_card(by[q][0], rung) is not None:
            return by[q]
    raise KeyError("no donor")


def identity_block(row, info, observer, card_which=DEFAULT_CARD, prior_only=False):
    pid = row["selfPID"]
    parts = []
    if info == "norms_history":          # Oct 9 (A1: "do it all"): the approved norms note, then the person's history block, unchanged
        return identity_block(row, "norms", observer) + "\n\n---\n\n" + identity_block(row, "history", observer)
    if info == "norms":                  # Oct 9 (A1 approved text): population-norm note from the TRAIN half (analysis/norms_split.json); test half only
        return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NORMS_NOTE_v2_APPROVED.txt")).read().strip()
    if info == "history_prior":          # Oct 8: prospective history, only calls BEFORE this one (the prompt's "Earlier" is then literally true)
        info, prior_only = "history", True
    if info.endswith("_wrong"):          # Oct 8 controls: same prompt text, another participant's material
        base = info[:-len("_wrong")]
        if base == "history":
            n = len(history(row))
            d = sorted(donor(row, need_hist=n), key=lambda r: int(r["roundIdx"]))[:n]
            h = [dict(moment=r["conv_SP_Mem_descr"].strip(), reason=r["conv_SP_Mem_reason"].strip(), round=int(r["roundIdx"])) for r in d]
            lead = ("Earlier in the same study you also had calls with other people. After each one you were asked what moment you "
                    "remembered most clearly, and why. Here is what you wrote:")
            assert not observer
            return lead + "\n\n" + ("\n\n".join(f"Another call:\nMoment: {x['moment']}\nWhy: {x['reason']}" for x in h) or "(no other calls)")
        rung = base.split("_")[1]
        drow = donor(row, rung=rung)[0]
        return identity_block(drow, base, observer, card_which, prior_only)
    if info.startswith("card") and info.split("_")[1] in ("thin", "narr", "person", "impthin", "impnarr", "facts", "weave", "factnarr") if "_" in info else False:
        rung = info.split("_")[1]
        c = rung_card(row, rung)
        if c is None:
            raise KeyError(f"no {rung} card for {row['perspectiveID']}")
        if rung in ("thin", "impthin"):
            parts.append("Here are your scores on a personality questionnaire you completed:\n\n" + c)
        elif rung == "facts":
            parts.append("Here are some facts about you:\n\n" + c)
        else:
            parts.append("Here is a description of you:\n\n" + c + "\n\nThat is who you are.")
        if info.endswith("_history"):
            info = "history"  # fall through to the history block below
    elif info in ("card", "card_history"):
        c = card(pid, card_which)
        if c is None:
            raise KeyError(f"no card for {pid}")
        if observer:
            parts.append("Here is a description of the participant, written from personality questionnaires they completed:\n\n" + c)
        else:
            parts.append("Here is a description of you, written from personality questionnaires you completed:\n\n" + c +
                         "\n\nThat is who you are.")
    if info in ("history", "card_history") or info.startswith("history_k"):
        h = history(row, prior_only)
        if info.startswith("history_k"):   # nested random subset of size k (seeded by perspective), for the history gradient
            import random as _r
            k = int(info.split("history_k")[1])
            order = list(range(len(h))); _r.Random(row["perspectiveID"]).shuffle(order)
            keep = sorted(order[:k]); h = [h[i] for i in keep]
        lead = ("In the same study the participant also had calls with other people. After each one they were asked what moment "
                "they remembered most clearly, and why. Here is what they wrote:" if observer else
                "Earlier in the same study you also had calls with other people. After each one you were asked what moment you "
                "remembered most clearly, and why. Here is what you wrote:")
        body = "\n\n".join(f"Another call:\nMoment: {x['moment']}\nWhy: {x['reason']}" for x in h) or "(no other calls)"
        parts.append(lead + "\n\n" + body)
    return "\n\n---\n\n".join(parts) if parts else None


def media_parts(row, modality, observer, model=None):
    d = os.path.join(MEDIA, row["perspectiveID"])
    you = "the participant" if observer else "you"
    yours = "the participant's" if observer else "your own"
    P = []
    if "text" in modality:
        P.append(orc.text_part(
            "Here is a transcript of the call, transcribed automatically (expect some errors). "
            + ("Lines marked Participant are what the participant said; lines marked Partner are what the other person said. "
               if observer else "Lines marked You are what you said; lines marked Partner are what the other person said. ")
            + "Times are minutes:seconds from the start.\n\n" + transcript(row["dyadID"], row["selfPID"], observer)))
    if "audio" in modality:
        P.append(orc.text_part(f"Recording 1 is {yours} microphone (only {'their' if observer else 'your'} voice):"))
        P.append(orc.audio_part(os.path.join(d, "self.mp3"), model))
        P.append(orc.text_part(f"Recording 2 is the other person as {you} heard them (only their voice). Both recordings start at "
                               "the same moment:"))
        P.append(orc.audio_part(os.path.join(d, "partner.mp3"), model))
    if "video" in modality:
        P.append(orc.text_part(f"The video shows the call side by side: the left half is the other person as {you} saw them, the "
                               f"right half is {you}. The video has no sound; the sound is in the recordings above."))
        P.append(orc.video_part(os.path.join(d, "video_silent.mp4")))
    return P


def questions(observer):
    qs = [Q_DESCR, Q_REASON, Q_TIMING, Q_SC2, Q_SC3, Q_ENT1, Q_FEEL]
    head = ("Answer these questions exactly as the participant did, the way you expect they answered them (in their own voice, "
            "as 'I'):" if observer else "Now please answer the questions you were given right after the call:")
    return head + "\n\n" + "\n\n".join(f"{i+1}. {q}" for i, q in enumerate(qs)) + "\n\n" + SCHEMA


def build(row, modality="text", info="none", frame="seat", model=None, card_which=DEFAULT_CARD, prior_only=False):
    observer = frame == "observer"
    msgs = []
    ib = identity_block(row, info, observer, card_which, prior_only)
    if ib:
        msgs.append({"role": "system", "content": ib})
    content = [orc.text_part(STUDY_OBS if observer else STUDY)] + media_parts(row, modality, observer, model) + [orc.text_part(questions(observer))]
    msgs.append({"role": "user", "content": content})
    return msgs


def render_text(msgs):
    """Human-readable dump (media shown as placeholders) for review."""
    out = []
    for m in msgs:
        out.append(f"===== {m['role'].upper()} =====")
        if isinstance(m["content"], str):
            out.append(m["content"])
        else:
            for p in m["content"]:
                out.append(p["text"] if p["type"] == "text" else f"<<{p['type']}>>")
    return "\n\n".join(out)
