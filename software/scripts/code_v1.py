#!/usr/bin/env python3
"""Codebook v1 coding of memory reports (human now; model predictions later with the same prompt).
Reasons: Tsfasman et al. 2022/2026 MeMo scheme, labels + descriptions VERBATIM (lit/MEMO_REASONS_CODEBOOK_Tsfasman2022.md), multi-label
  (her scheme is multi-layer), adapted only in wording of the frame ("annotator" = the rememberer; "speaker" = whoever was talking).
Structure: Singer & Blagov two-level specificity, adapted to 3-min calls (their speech rule: quoting/paraphrasing a particular
  instance of speaking = a specific event). Meaning: Singer & Blagov integrative vs non-integrative, applied to the stated reason.
One memory per request (coders can't learn a person's style across items). Two coders not in the test set."""
import csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc, prompts as P

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "analysis", "code_v1"); os.makedirs(OUT, exist_ok=True)
CODERS = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]

PROMPT = """You are coding one memory report from a research study. Two strangers had a 3-minute video call. Afterwards, one of them (the "rememberer") described the moment they remembered most clearly and why. Code ONLY what the report says; ignore spelling and writing style.

MEMORY: {mem}
WHY THEY REMEMBERED IT: {why}

A) REASON LABELS (from the MeMo scheme of Tsfasman et al.). Choose every sub-label that applies to the stated reason (and the memory where the reason is implicit). "Annotator" means the rememberer; "speaker" means whoever was talking in the remembered moment.
- fact_about_others.view: agreement or disagreement with the view of another participant
- fact_about_others.social_facts: speaker describes something about his/her activities, family or friends
- fact_about_others.unexpected_info: speaker describes something which is unexpected from the annotator's perspective
- fact_about_world.entities: view about entities such as lockdown, vaccination, etc.
- fact_about_world.people: view about people
- self_perception.annotator_feelings: feelings of the annotator such as happy, sad, embarrassed, etc.
- self_perception.annotator_stories: stories or views of the annotator
- shared_experience.shared_story: annotator has a similar experience or feeling with the speaker
- meta_behaviour_of_other.emotional_moment: emotional moments of other participants
- meta_behaviour_of_other.behaviour: behaviour of the participants during the session such as laughter, anger, etc.
- cognitive.cognitive_empathy: annotator sees the situation from the perspective of the speaker in a logical way
- time_label.first: annotator remembers the first moments of the session
- time_label.last: annotator remembers the last moments of the session
- none: no label fits

B) STRUCTURE (Singer & Blagov, adapted): "specific" if the memory points to a particular moment in the call — a single exchange, remark, event or reaction (quoting or paraphrasing a particular instance of speaking counts as specific); "general" if it summarises a topic discussed, the conversation as a whole, or a general impression of the person.

C) MEANING (Singer & Blagov): "integrative" if the reason states what the moment means beyond itself — a lesson, insight or connection to the rememberer's own life or self-understanding; otherwise "non_integrative" (it just narrates, or labels it by importance, emotion or attribute).

Return JSON only: {{"reasons": ["label.sublabel", ...], "structure": "specific|general", "meaning": "integrative|non_integrative"}}"""


def job(args):
    key, mem, why, coder = args
    p = os.path.join(OUT, f"{key}__{coder.split('/')[1]}.json")
    if os.path.exists(p):
        return 0.0, True
    r = orc.call(coder, [{"role": "user", "content": PROMPT.format(mem=mem, why=why)}], os.path.join(OUT, "calls.jsonl"), tag=f"{key}|{coder}",
                 max_tokens=2000, temperature=0, response_format={"type": "json_object"}, reasoning={"effort": "low"})
    t = r.get("text") or ""
    try:
        d = json.loads(t[t.find("{"): t.rfind("}") + 1]); assert "reasons" in d
    except Exception:
        return orc.cost(r), False
    json.dump(dict(key=key, coder=coder, code=d), open(p, "w"))
    return orc.cost(r), True


def main():
    rows = P.rows()
    jobs = [(r["perspectiveID"], r["conv_SP_Mem_descr"].strip(), r["conv_SP_Mem_reason"].strip(), c) for r in rows for c in CODERS]
    if len(sys.argv) > 1:
        jobs = jobs[: int(sys.argv[1]) * 2]
    tot = 0.0; bad = 0
    with ThreadPoolExecutor(16) as ex:
        for c, ok in ex.map(job, jobs):
            tot += c; bad += not ok
    print("codings", len(jobs), "failed", bad, "cost $", round(tot, 3))


if __name__ == "__main__":
    main()
