#!/usr/bin/env python3
"""Public stand-in generator (A1, Oct 11). Reads ONLY public inputs and the numbers in targets_from_parsel.json:
  people.json / sessions.json (build_people.py: ONS 2021 teaching sample, OGL), hexaco_norms.json (published norms; SOURCES.md),
  style_guide.md (public descriptions of British conversational English + the call-shape numbers). No PARSEL code or data is imported.
Stages (each resumable, one file per item; every API call logged with its cost in <stage>/calls.jsonl):
  traits  local: HEXACO-60 domain and facet scores from published norms by age and sex
  card    Claude Opus 5.5: a Nemotron-style persona card, English voice, typing habits, webcam-photo description
  rate    Gemini 3 Flash (separate, cheaper): each person rates the others' photos on first-impression items
  convo   Claude Opus 5.5: one thread per session, the calls written round by round in order
  memory  Claude Opus 5.5: each person's post-call answers (moment, reason, timing), in persona, with the type notes and length targets
Usage: gen.py STAGE --sessions S01,S02 [--workers N]   (--sessions all for every session)"""
import argparse, collections, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts")); import orc   # API plumbing only
OPUS, RATER = "anthropic/claude-opus-5.5", "google/gemini-3-flash-preview"
PEOPLE = json.load(open(os.path.join(HERE, "people.json"))); SS = json.load(open(os.path.join(HERE, "sessions.json")))
SESS, CALLS = SS["sessions"], {c["id"]: c for c in SS["calls"]}
T = json.load(open(os.path.join(HERE, "targets_from_parsel.json")))
AV = json.load(open(os.path.join(HERE, "av_plan.json"))) if os.path.exists(os.path.join(HERE, "av_plan.json")) else {}
for d in ("traits", "cards", "ratings", "sessions", "calls", "memories"): os.makedirs(os.path.join(HERE, d), exist_ok=True)
log = lambda stage: os.path.join(HERE, stage, "calls.jsonl")
SFX = os.environ.get("TAGSFX", "")   # set to regenerate a session whose cached answers must not be reused
def jparse(t):
    t = (t or "").strip(); s, e = t.find("{"), t.rfind("}")
    try: return json.loads(t[s:e + 1])
    except Exception: return None
def from_log(stage, tag):
    p = log(stage)
    if os.path.exists(p):
        for l in open(p):
            j = json.loads(l)
            if j.get("tag") == tag and j.get("ok") and j.get("text"): return dict(j, usage={"cost": 0})
    return None

# ---------- 0. traits (local)
DOM = {"Honesty-Humility": ["Sincerity", "Fairness", "Greed Avoidance", "Modesty"], "Emotionality": ["Fearfulness", "Anxiety", "Dependence", "Sentimentality"],
       "Extraversion": ["Social Self-Esteem", "Social Boldness", "Sociability", "Liveliness"], "Agreeableness": ["Forgivingness", "Gentleness", "Flexibility", "Patience"],
       "Conscientiousness": ["Organization", "Diligence", "Perfectionism", "Prudence"], "Openness to Experience": ["Aesthetic Appreciation", "Inquisitiveness", "Creativity", "Unconventionality"]}
def traits():
    """Domains ~ multivariate normal (Garcia SI Table A10 intercorrelations; SD = within-sex s) around the Western-8 adult centre split by the
    UK sex gap, clipped to 1-5; Honesty-Humility age term (hexaco_norms.json "recipe"). No facets (cards use domain levels only)."""
    N = json.load(open(os.path.join(HERE, "hexaco_norms.json"))); rng = np.random.default_rng(1011); h = N["hh_age"]; doms = list(DOM)
    R = np.array(N["domain_correlations"]); out = {}
    for pid, p in sorted(PEOPLE.items()):
        g = N["by_sex"][p["sex"]]; sd = np.array([g[d]["sd"] for d in doms])
        mu = np.array([g[d]["mean"] + (h["r"] / h["age_sd"] * g[d]["sd"] * (p["age"] - h["centre_age"]) if d == "Honesty-Humility" else 0.0) for d in doms])
        x = np.clip(mu + rng.multivariate_normal(np.zeros(6), R) * sd, 1, 5)
        out[pid] = dict(domains={d: float(x[i]) for i, d in enumerate(doms)})
    json.dump(out, open(os.path.join(HERE, "traits", "traits.json"), "w"), indent=1); print("traits for", len(out))
def TR(pid): return json.load(open(os.path.join(HERE, "traits", "traits.json")))[pid]
LVL = ["very low", "low", "average", "high", "very high"]
def lvl(x, d):
    N = json.load(open(os.path.join(HERE, "hexaco_norms.json"))); all_ = [g[d]["mean"] for g in N["by_sex"].values()]; m = np.mean(all_)
    sd = np.mean([g[d]["sd"] for g in N["by_sex"].values()]); z = (x - m) / sd
    return LVL[int(np.searchsorted([-1.28, -0.52, 0.52, 1.28], z))]

# ---------- 1. cards
CARD = """Create a persona card for an invented participant in a UK online research study (recruited through an online participant panel). The card is an honest attempt to instantiate a real-seeming, data-backed person in the style of census-grounded persona datasets: ordinary, specific, consistent. It is invented: do not base it on anyone real or famous.

The data behind this person (from census and personality-norm statistics):
- {age}-year-old {sex}, living in {region}; {born}; ethnic group: {ethnic}; religion: {religion}.
- Partnership: {partnership}; household: {household}.
- Work: {work}; approximate social grade: {grade}; health: {health}.
- Personality (HEXACO, relative to the general population): {traits}.

Write the life around these facts (job, town, household, hobbies) so that it fits them. Their English should fit where they live, their age and their background, as described in this guide to British conversational English:
{voice_guide}

Return JSON only, with these fields:
{{"first_name": "...", "age": {age}, "lives_in": "town or city, region", "occupation": "...", "education_detail": "...", "household": "...",
 "cultural_background": "...", "professional_persona": "...", "sports_persona": "...", "arts_persona": "...", "travel_persona": "...", "culinary_persona": "...",
 "hobbies_and_interests": ["..."], "skills": ["..."], "career_goals": "...",
 "voice": "how they actually talk on a call: register, pace, regional features if any (light, never caricature), fillers, how they handle silence",
 "typing": "how they type a quick answer into a web form: capitals, punctuation, apostrophes, abbreviations",
 "with_strangers": "how they come across in a first conversation with a stranger",
 "small_things": ["ordinary things they might mention"], "deeper_things": ["more personal things they would only mention if comfortable"],
 "webcam_photo": "a specific description of a webcam snapshot of them: framing, angle and lighting, their face and expression, hair, clothing, glasses or not, and exactly what is visible in the room behind them, as a plain factual description of the image"}}"""
def voice_guide(): return open(os.path.join(HERE, "style_guide.md")).read()
def card(pid):
    f = os.path.join(HERE, "cards", pid + ".json")
    if os.path.exists(f) and json.load(open(f)).get("ok"): return 0.0
    p = PEOPLE[pid]; t = TR(pid)
    work = ("full-time student" if p["full_time_student"] else "") + ("; " if p["full_time_student"] and p["occupation"] != "no current occupation" else "") + \
           (f"{p['occupation']} ({p['hours']})" if p["occupation"] != "no current occupation" else ("" if p["full_time_student"] else "not currently in paid work"))
    msg = CARD.format(age=p["age"], sex={"Female": "woman", "Male": "man"}[p["sex"]], region=p["region"], born=p["born"].lower(), ethnic=p["ethnic_group"],
                      religion=p["religion"], partnership=p["partnership"].lower(), household=p["household"].lower(), work=work, grade=p["social_grade"],
                      health=p["health"].lower(), traits="; ".join(f"{d} {lvl(v, d)}" for d, v in t["domains"].items()), voice_guide=voice_guide())
    r = from_log("cards", pid); c = jparse((r or {}).get("text"))
    if not (c and c.get("webcam_photo") and c.get("first_name")):   # no usable cached answer: call fresh (pilot 2: one card truncated at 4,000 tokens)
        r = orc.call(OPUS, [{"role": "user", "content": msg}], log("cards"), tag=pid, max_tokens=10000, temperature=1.0, response_format={"type": "json_object"}); c = jparse(r.get("text"))
    ok = bool(c and c.get("webcam_photo") and c.get("first_name"))
    json.dump(dict(id=pid, ok=ok, card=c), open(f, "w"), indent=1); return orc.cost(r)
def C(pid): return json.load(open(os.path.join(HERE, "cards", pid + ".json")))["card"]
def label(pid):   # transcript label: first name, made unique within the session (two Gareths in S10 broke C0085)
    sid = next(s for s, m in SESS.items() if pid in m); same = [m for m in SESS[sid] if C(m)["first_name"] == C(pid)["first_name"]]
    return C(pid)["first_name"] + (f" {'ABCDEF'[same.index(pid)]}" if len(same) > 1 else "")
def card_text(pid, photo=True):
    c = C(pid); keys = [k for k in c if (k != "webcam_photo" or photo)]
    return "\n".join(f"- {k.replace('_', ' ')}: {', '.join(map(str, c[k])) if isinstance(c[k], list) else c[k]}" for k in keys)

# ---------- 2. first impressions of the photos (separate model)
ITEMS = [("friendly", "How friendly would you say this person is?"), ("like", "How likable would you say this person is?"),
         ("trust", "How trustworthy would you say this person is?"), ("comp", "How competent would you say this person is?"),
         ("sim", "How similar to you would you say this person is?")]
RATE = """Stay in character as this person, who is taking part in an online study in the UK:
{me}

Before a round of short video calls, the study shows you webcam photos of the other people in your group and asks for first impressions. Their photos (described):
{photos}

For EACH person, answer from the photo alone, 1 (not at all) to 7 (extremely):
{items}
Then: would you like to be paired with them for a later task (0 = no, 1 = yes), and your gut impression in a few words.

Return JSON only: {{"ratings": [{{"photo": <number>, {keys}, "pair": 0 or 1, "gut": "..."}}, ...]}}"""
def rate(job):
    sid, pid = job; f = os.path.join(HERE, "ratings", f"{sid}_{pid}.json")
    if os.path.exists(f) and json.load(open(f)).get("ok"): return 0.0
    others = [o for o in SESS[sid] if o != pid]
    msg = RATE.format(me=card_text(pid, photo=False), photos="\n".join(f"Photo {i + 1}: {C(o)['webcam_photo']}" for i, o in enumerate(others)),
                      items="\n".join(f"{i + 1}. {q}" for i, (_, q) in enumerate(ITEMS)), keys=", ".join(f'"{k}": 1-7' for k, _ in ITEMS))
    r = from_log("ratings", f"{sid}|{pid}") or orc.call(RATER, [{"role": "user", "content": msg}], log("ratings"), tag=f"{sid}|{pid}", max_tokens=4000, temperature=1.0, response_format={"type": "json_object"})
    d = jparse(r.get("text")); out = {}
    for x in (d or {}).get("ratings", []):
        try: out[others[int(x["photo"]) - 1]] = x
        except Exception: pass
    json.dump(dict(session=sid, rater=pid, ok=len(out) == len(others), ratings=out), open(f, "w"), indent=1); return orc.cost(r)
def R(sid, pid):
    f = os.path.join(HERE, "ratings", f"{sid}_{pid}.json"); return json.load(open(f))["ratings"] if os.path.exists(f) else {}

# ---------- 3. calls, a whole session in order
cs = T["call_shape"]
SHAPE = (f"Call shape (from the original study's numbers): a call is cut off by a three-minute timer. A typical call has about {int(cs['turns_per_call']['p50'])} turns "
         f"(most between {int(cs['turns_per_call']['p10'])} and {int(cs['turns_per_call']['p90'])}) and about {int(cs['words_per_call']['p50'])} words in total "
         f"(most between {int(cs['words_per_call']['p10'])} and {int(cs['words_per_call']['p90'])}); half of all turns are {int(cs['words_per_turn']['p50'])} words or fewer, "
         f"and a quarter are {int(cs['words_per_turn']['p75'])} or more. Short acknowledgements (\"yeah\", \"mm\") count as turns too. "
         f"So aim for about {int(cs['words_per_call']['p50'])} words in about {int(cs['turns_per_call']['p50'])} turns: alongside the quick back-and-forth, people often talk "
         f"in longer stretches (explaining their job, telling a story), and those long turns carry much of the call.")
FLOW = ("The study, in order (from its published design): a short puzzle task done alone; first impressions of the others' webcam photos; then a series of "
        "three-minute, one-on-one video calls in rounds, each person meeting the others in their group; after each call a short survey.")
def session_thread(sid):
    f = os.path.join(HERE, "sessions", sid + ".json")
    done = json.load(open(f)) if os.path.exists(f) else {"turns": [], "calls": {}}
    members = SESS[sid]; cards = "\n\n".join(f"{m} ({C(m)['first_name']}; transcript label: {label(m)}):\n{card_text(m)}" for m in members)
    imp = [f"{C(m)['first_name']} on {C(o)['first_name']}'s photo: likable {x.get('like')}/7, friendly {x.get('friendly')}/7, would pair: {'yes' if x.get('pair') else 'no'}; gut: {x.get('gut')}"
           for m in members for o, x in R(sid, m).items()]
    sysmsg = ("You write transcripts of realistic video calls between strangers for a fully synthetic research dataset set in the UK. " + FLOW + "\n\n" + SHAPE +
              "\n\nGuide to British conversational English (use it for how people talk; it is a description of speech, not a script):\n" + voice_guide())
    intro = (f"The group (session {sid}). Persona cards, including the webcam photo each person was rated on:\n\n{cards}\n\nTheir first impressions of each other's photos (private):\n" +
             "\n".join(imp) + "\n\nI will give you the calls one round at a time, in the order they happened. Write each call as a transcript, one turn per line starting with the speaker's transcript label (their first name, plus a letter only where two people share a name) and a colon. "
             "Start each call with a line '=== CALL <id> ===' and nothing else. People keep to their cards, and can refer to things from earlier in the study.")
    msgs = [{"role": "system", "content": sysmsg}]; rounds = collections.defaultdict(list)
    for c in CALLS.values():
        if c["session"] == sid: rounds[c["round"]].append(c)
    cost = 0.0
    for k, rd in enumerate(sorted(rounds)):
        ask = (intro + "\n\n" if k == 0 else "") + f"Round {rd + 1}. These calls happen at the same time:\n" + "\n".join(
            f"- CALL {c['id']}: {label(c['A'])} ({c['A']}) and {label(c['B'])} ({c['B']})" for c in rounds[rd])
        for c in rounds[rd]:
            if c["id"] in AV:
                v = AV[c["id"]]; an, nn = C(v["actor"])["first_name"], C(v["noticer"])["first_name"]
                ask += (f"\n\nPRIVATE, for CALL {c['id']} only: during this call one non-verbal thing happens on {an}'s side that {nn} sees or hears on the video, "
                        f"for example something visible in {an}'s room (their webcam description helps), a change in {an}'s face, a long silence, a silent laugh, a sigh, "
                        f"their voice catching, someone appearing behind them, or them glancing away repeatedly. It must NOT appear in the transcript in any way: no stage "
                        f"directions or brackets, nobody mentions it, reacts to it in words, or hints at it, and the talk carries on as if it hadn't happened. After that "
                        f"call's transcript, add a block '=== EVENT {c['id']} ===' with JSON only: {{\"turn\": <number of the transcript line it happens at, counting "
                        f"that call's lines from 1>, \"line_text\": \"<that line's text>\", \"who\": \"{an}\", \"what\": \"<what a camera or microphone would show>\"}}.")
        msgs.append({"role": "user", "content": ask})
        if len(done["turns"]) > k: msgs.append({"role": "assistant", "content": done["turns"][k]}); continue
        r = from_log("sessions", f"{sid}|r{rd}{SFX}") or orc.call(OPUS, msgs, log("sessions"), tag=f"{sid}|r{rd}{SFX}", max_tokens=14000, temperature=1.0)
        cost += orc.cost(r); text = r.get("text") or ""; msgs.append({"role": "assistant", "content": text}); done["turns"].append(text)
        for m in re.finditer(r"=== CALL (C\d+) ===\s*\n(.*?)(?=\n=== |\Z)", text, re.S): done["calls"][m.group(1)] = m.group(2).strip()
        for m in re.finditer(r"=== EVENT (C\d+) ===\s*\n(.*?)(?=\n=== |\Z)", text, re.S): done.setdefault("events", {})[m.group(1)] = jparse(m.group(2))
        json.dump(done, open(f, "w"), indent=1)
    for c in (x for x in CALLS.values() if x["session"] == sid):
        na, nb = label(c["A"]), label(c["B"]); lines = []
        for l in done["calls"].get(c["id"], "").splitlines():
            m = re.match(r"^\s*\**([^:*]{1,30})\**:\s*(.*)$", l)
            if m and m.group(1).strip() in (na, nb) and na != nb: lines.append(("A" if m.group(1).strip() == na else "B") + ": " + m.group(2).strip())
        ev = (done.get("events") or {}).get(c["id"]) if c["id"] in AV else None
        json.dump(dict(**{k: c[k] for k in ("id", "session", "round", "A", "B")}, names={"A": na, "B": nb}, turns=len(lines), av_event=ev,
                       words=sum(len(l[3:].split()) for l in lines), ok=len(lines) >= 8, text="\n".join(lines)), open(os.path.join(HERE, "calls", c["id"] + ".json"), "w"), indent=1)
    return cost

# ---------- 4. memories
Q1 = ("Please recall and describe the moment you remember most clearly from the interaction. Describe this moment in as much detail as you can remember.")
Q2 = "Why do you think you remembered this specific moment?"
Q3 = "When during your conversation did this moment occur? (0 = beginning, 100 = end)"
def notes(spec, L):
    """Every slot gives an explicit instruction (pilot 1: unguided slots defaulted to one exchange and the partner's words)."""
    n = []
    n.append({"one_moment": "The moment is one particular exchange.",
              "stretch": "It can be a stretch of the call that you talked about for a while, rather than one particular line.",
              "free": "It is about the call as a whole or a general impression of it, not one particular moment."}[spec["scope"]])
    n.append({"did_together": "Describe it in 'we' terms, as something you both talked about or did together; do not quote or paraphrase anything the other person said.",
              "partner_said_or_did": "It is something the other person said or did.",
              "free": "Do not report what the other person said or did; keep it to the situation, your own thoughts, or what they were like."}[spec["what"]])
    n.append("Include one specific thing you learned about them that you hadn't known." if spec["new_fact"]
             else "Do not mention any specific new fact about them (no details of their job, family, places, names or numbers).")
    n.append(f"Your answer about the moment is about {L['moment']} words long (no more than {int(L['moment'] * 1.25) + 2}), and your reason about {L['reason']} words (no more than {int(L['reason'] * 1.25) + 2}).")
    return " ".join(n)
MEM_SYS = ("You play one participant in a study and fill in their post-call survey exactly as that person would, typing into a web form a minute after the call. "
           "You will be told privately how this person's answer comes out. Follow those notes in the person's own voice and typing habits, choosing a real moment from the call that fits. Never mention the notes.")
def seat(cid, pid):
    cj = json.load(open(os.path.join(HERE, "calls", cid + ".json"))); s = "A" if cj["A"] == pid else "B"
    return cj, s, "\n".join(("You: " if l.startswith(s + ":") else "Partner: ") + l[3:] for l in cj["text"].splitlines())
def memory(job):
    cid, pid = job; f = os.path.join(HERE, "memories", f"{cid}_{pid}.json")
    if os.path.exists(f) and json.load(open(f)).get("ok"): return 0.0
    cj, s, tx = seat(cid, pid)
    if not cj["ok"]: return 0.0
    c = CALLS[cid]; partner = c["B"] if c["A"] == pid else c["A"]
    mine = sorted((x for x in CALLS.values() if x["session"] == c["session"] and pid in (x["A"], x["B"]) and x["round"] < c["round"]), key=lambda x: x["round"])
    earlier = "\n\n".join(f"Earlier call with {C(x['B'] if x['A'] == pid else x['A'])['first_name']}:\n" + seat(x["id"], pid)[2] for x in mine) or "(this was your first call)"
    pi = R(c["session"], pid).get(partner, {}); spec, L = c[f"spec_{s}"], c[f"len_{s}"]
    av = AV.get(cid, {}).get("noticer") == pid and cj.get("av_event")
    note = (f"The moment you remember is something you saw or heard on the video that nobody said anything about: {cj['av_event'].get('what')} "
            f"Describe it as you noticed it. Your answer about the moment is about {L['moment']} words long (no more than {int(L['moment'] * 1.25) + 2}), and your reason "
            f"about {L['reason']} words (no more than {int(L['reason'] * 1.25) + 2}).") if av else notes(spec, L)
    user = (f"You are this person:\n{card_text(pid, photo=False)}\n\n{FLOW}\n\nBefore the calls you saw {C(partner)['first_name']}'s photo ({C(partner)['webcam_photo']}); your gut impression was: {pi.get('gut', 'no strong impression')}.\n\n"
            f"Your earlier calls today:\n{earlier}\n\nThe call you have just finished (You = you, Partner = {C(partner)['first_name']}):\n{tx}\n\n"
            f"PRIVATE NOTES on how your answer comes out: {note}\n\nThe survey asks:\n1. {Q1}\n2. {Q2}\n3. {Q3}\n\n"
            'Reply with JSON only: {"moment": "...", "reason": "...", "timing": 0-100}')
    r = from_log("memories", f"{cid}|{pid}{SFX}") or orc.call(OPUS, [{"role": "system", "content": MEM_SYS}, {"role": "user", "content": user}], log("memories"), tag=f"{cid}|{pid}{SFX}",
                                                         max_tokens=3000, temperature=1.0, response_format={"type": "json_object"})
    a = jparse(r.get("text")); ok = bool(a and a.get("moment") and a.get("reason"))
    json.dump(dict(call=cid, person=pid, seat=s, memory_type=PEOPLE[pid]["memory_type"], spec=(None if av else spec), length_target=L, ok=ok, answers=a,
                   requires_av=bool(av), av_event=({"call": cid, **cj["av_event"]} if av else None)), open(f, "w"), indent=1)
    return orc.cost(r)

def spent():   # every API call this build has made, pilots included
    import glob as _g
    return sum(float((json.loads(l).get("usage") or {}).get("cost") or 0) for p in _g.glob(os.path.join(HERE, "**", "calls.jsonl"), recursive=True) for l in open(p))

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("stage"); a.add_argument("--sessions", default="all"); a.add_argument("--workers", type=int, default=12); a.add_argument("--cap", type=float, default=200.0); a = a.parse_args()
    if a.stage == "traits": traits(); sys.exit()
    if a.stage == "spent": print(round(spent(), 2)); sys.exit()
    assert spent() < a.cap, f"budget cap reached: ${spent():.2f}"
    ss = sorted(SESS) if a.sessions == "all" else sorted(a.sessions.split(","))
    if a.stage in ("rate", "convo", "memory"):
        bad = [p for s in ss for p in SESS[s] if not (os.path.exists(os.path.join(HERE, "cards", p + ".json")) and json.load(open(os.path.join(HERE, "cards", p + ".json")))["ok"])]
        assert not bad, f"cards missing or failed: {bad[:10]} (rerun gen.py card)"
    pids = [p for s in ss for p in SESS[s]]; cids = [c for c, x in CALLS.items() if x["session"] in ss]
    ex = ThreadPoolExecutor(a.workers)
    if a.stage == "card": cost = sum(ex.map(card, pids))
    elif a.stage == "rate": cost = sum(ex.map(rate, [(s, p) for s in ss for p in SESS[s]]))
    elif a.stage == "convo": cost = sum(ex.map(session_thread, ss))
    elif a.stage == "memory": cost = sum(ex.map(memory, [(c, p) for c in cids for p in (CALLS[c]["A"], CALLS[c]["B"])]))
    print(a.stage, len(ss), "sessions; this run $", round(cost, 2), "; total spent $", round(spent(), 2))
