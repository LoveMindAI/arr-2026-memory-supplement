#!/usr/bin/env python3
"""THE single source of truth for codebook v3.5.1 (Oct 8 2026; v3.5 frozen copy kept as spec_v35_FROZEN.py).
Changes v3.5 -> v3.5.1 (A1 approved after the v3.5 model pilot, analysis/code_v35/pilot40): models chose "Can't tell" on most feeling and
appraisal items (intensity 67%, 'could have predicted' 93%). (1) 'could have predicted' cut (A1: "just so weird"; surprise is covered by the
Surprise feeling and the reason's 'Something unexpected about them'); (2) intensity: "If the text shows no feeling, choose Not at all."
(crowd-enVENT's 1 = 'not at all'); (3) "Can't tell" clarified as "only if the text gives you nothing to go on; a sensible reading is fine".
storage_version stays "3.5" so coders' in-progress autosaves carry over. Everything people and models see is generated from SPEC:
  codebook/CODEBOOK_v3.1.md (readable), the model prompt (scripts/code_v31.py), and the coding GUI (human_coding/*.html, Prolific).
Grounding and adaptations: see CODEBOOK_v3.md header. Changes v3.1 -> v3.2 (A1 + A3 calibration, Oct 8):
 - reporter/partner defined up front; topic, about and 'what is remembered' take up to 3 answers.
 - details: 'who said it' no longer counts (our adaptation: Levine's internal details include the people involved, as summarised by
   van Genugten & Schacter 2024, but in a two-person call the people are always the same two, so naming them carries no information); worked examples no longer reuse any calibration item (checked by leak_check in make_gui_v32.py).
 - 'funny' became 'funny or unusual'. Grounding per lit room (paper_notes/CODEBOOK_LIT_C_HUMOUR_DISTINCTIVENESS_20261008.md): humour = Schmidt & Williams 2001 (READ;
   effect is within-list, gist not detail); distinctiveness = Hunt 1995 (READ) + Hunt 2013 (abstract): context-relative, so the code is 'unusual as described',
   NOT distinctiveness. Schmidt 1994 and Hunt 2006 are abstract-only. (Question removed in v3.4.)
 - disclosure ladder: level 2 is now 'surface feelings, likes or opinions', next to level 1 'surface facts'.
 - valence dropped (crowd-enVENT's own pleasant/unpleasant statements carry it); intensity follows crowd-enVENT's validator question 'How intense do you think the emotion was?' (1-5; paper text: 'not at all' to 'extremely').
 - appraisal: crowd-enVENT's VALIDATOR (third-person) statements, 'experiencer' -> 'the reporter', agreement scale as in crowd-enVENT
   (1 "I don't agree at all", 5 "I completely agree"; middle labels ours). The 'put yourself in the shoes of the writer' instruction is dropped:
   coders judge what the text says, never imagine another person (A1, Oct 8). 'sudden or abrupt' swapped for crowd-enVENT's predictability item.
Changes v3.2 -> v3.3 (A1's v3.2 calibration, Oct 8):
 - 'how personal' (one level) became 'what kinds of personal things were shared' (all that fit); depth = the highest kind chosen.
 - 'clashed with standards and ideals' dropped ("too open ended and weird").
 - topic 'Work or study' -> 'Work, education or service'; 'The study...' -> 'This research study...' (we call the call 'the study').
 - reason 'The call itself' gained MIDDLE, WHOLE_CALL and MEDIUM. MeMo has only first/last (time_label), so these three have no MeMo label.
 - 'none'-type answers are exclusive (exclusive=True): choosing one clears the rest.
Changes v3.3 -> v3.4 (principled cuts, A1 Oct 8: "the less complicated we can make this, the better"; evidence in codebook/CUTS_v34.md):
 - removed: funny/unusual (A1); 'who drove the moment' (our invention, no source; agency now carried by crowd-enVENT's own/other-behaviour items and
   source by 'what is remembered', as Brown-Schmidt et al. 2024 code source); 'two or more moments' (a procedural rule, kept in the intro);
   'who or what is the moment about' (overlaps topic + what is remembered); 'feeling toward the partner' (no source).
 - appraisal: removed 'familiar' and 'important consequences' (lowest model agreement in both pilots; near-constant in a first call with a stranger;
   'consequences' overlaps 'matters for... the future', Bluck's directive function).
 - intensity: crowd-enVENT's 1-5 ('not at all' to 'extremely') with "Can't tell" FIRST so it isn't read as an end of the scale; 'No feeling shown' removed
   (it duplicated 'Not at all').
Changes v3.4 -> v3.5 (lit room D, paper_notes/CODEBOOK_LIT_D_*):
 - topic re-grounded on Kellermann's conversation MOP (Kellermann et al. 1989; Kellermann 1991; Kellermann & Palomares 2004, topics coded from recall):
   added 'plans, goals or recent events' (MOP goals/intentions, 25 of 43 dyads; recent life events); people both know -> family/relationships;
   introductions, names, swapping contacts -> the call itself (MOP present situation, reason for presence, rituals). Ours, not the MOP's:
   health as a substantive topic (MOP 'health' is the how-are-you ritual) and partner impressions not said aloud.
 - personal disclosure re-grounded on Barak & Gluck-Ofri 2007 (READ; Table 1 p. 410): each option is one of their cells
   (information / thoughts / feelings x level 2 'little' or 3 'high'); their level 1 = not chosen. Their overall score = sum of the
   three dimension levels (p. 409), derivable from the ticks. Adaptation: their unit is the discloser's own message; ours is the
   remembered moment, for either person. Altman & Taylor no longer needed as grounding.
 - details: a second logged departure from Levine (via van Genugten & Schacter 2024, p. 2244): thoughts and feelings count as internal
   details there; we don't count them because Part B codes them separately.
 - 'Something the partner said' -> 'said or did' (PARTNER_SAID_OR_DID), matching the reporter option; found while writing E6.
 - write-in 'other' (text=True; answer stored as <qid>_other): feelings 'Another feeling' (A1: admiration fit none of crowd-enVENT's 12;
   crowd-enVENT has no write-in, so this is ours, and the 12-label analysis treats OTHER as its own label), topic 'Something else',
   reason 'Something else' (no MeMo label), and Part C interaction/acts 'Other'."""
import json, os

AGREE5 = [dict(code=1, label="Don't agree at all"), dict(code=2, label="Slightly agree"), dict(code=3, label="Somewhat agree"),
          dict(code=4, label="Mostly agree"), dict(code=5, label="Completely agree"), dict(code="CANT_TELL", label="Can't tell")]

INTRO = [
 "Two strangers had a 3-minute video call. Right afterwards, one of them wrote down the moment they remembered most clearly (MOMENT) and why they think they remembered it (REASON).",
 "\"The reporter\" is the person reporting their memory of the conversation. \"The partner\" is the person they had the conversation with.",
 "Some reports were written by people and some by computer models imitating people; code them all the same way.",
 "Code only what the text says. Don't guess what probably happened, and don't imagine how you would have felt.",
 "Each question says which text to read: MOMENT, REASON, or both.",
 "Choose \"Can't tell\" only if the text gives you nothing to go on. A sensible reading of the text is fine. Where a question says 'up to 3', choose every answer that fits.",
 "If the MOMENT describes two or more separate moments, answer about the first one."]

Q = lambda **k: k
O = lambda code, label, help=None, **k: dict(code=code, label=label, **({"help": help} if help else {}), **k)

PARTS = [
 dict(id="A", title="The moment", questions=[
  Q(id="topic", read="MOMENT", type="multi", max=3, q="What was the moment about?", options=[
    O("WORK_EDUCATION_SERVICE", "Work, education or service"), O("PLACE_TRAVEL", "Where someone lives or comes from, places, travel"),
    O("FAMILY_RELATIONSHIPS_PETS", "Family, relationships or pets", "including people they both know"), O("HOBBIES_INTERESTS_MEDIA", "Hobbies, interests, TV, music, sport"),
    O("PLANS_EVENTS", "Plans, goals or recent events in someone's life"), O("FOOD", "Food"), O("HEALTH_WELLBEING", "Health or wellbeing"), O("FEELINGS_MOOD", "Feelings or the mood of the call"),
    O("THE_STUDY_OR_CALL", "This research study, the call itself or the technology", "including introductions, names, swapping contacts"), O("PARTNER_IMPRESSION", "How the partner looked, sounded or came across"),
    O("OTHER", "Something else", text=True), O("CANT_TELL", "Can't tell", exclusive=True)]),
  Q(id="remembered", read="MOMENT", type="multi", max=3, q="What is remembered?", options=[
    O("PARTNER_SAID_OR_DID", "Something the partner said or did"), O("PARTNER_FACT_OR_QUALITY", "A fact or quality of the partner", "not one particular remark"),
    O("REPORTER_SAID_OR_DID", "Something the reporter said or did"), O("DID_TOGETHER", "Something they did together", "e.g. discovered something in common, made plans"),
    O("REPORTER_THOUGHT", "A thought, realisation or feeling the reporter had"), O("SITUATION", "The situation", "the call, the study, the technology"),
    O("CANT_TELL", "Can't tell", exclusive=True)]),
  Q(id="new_info", read="MOMENT", type="single", q="Is a specific new piece of information remembered?", options=[
    O("YES", "Yes: a specific fact learned in the call", "e.g. where they grew up; a train strike I hadn't heard of"),
    O("NO", "No: it is about information, but nothing specific and new"), O("NOT_ABOUT_INFORMATION", "It isn't about information", "a feeling, a laugh, the mood")]),
  Q(id="extent", read="MOMENT", type="single", q="How much of the call does the moment cover?", options=[
    O("ONE_MOMENT", "One moment", "a single remark, exchange or reaction"), O("STRETCH", "A stretch of the call", "a longer part: 'the conversation about travel'"),
    O("WHOLE_CALL_OR_PATTERN", "The whole call, or a pattern across it", "'we got on well', 'he was really friendly'"),
    O("NOT_AN_EVENT", "Not an event", "a fact or opinion about someone, with nothing happening")]),
  Q(id="details", read="MOMENT", type="single", q="How many details?",
    help="Count the separate, concrete pieces of information about what happened. Each counts once: a separate fact (a place, a name, a number, an object); a reaction ('she gasped' = 1); a quote, or something seen or heard ('her accent' = 1). Don't count who said it ('he told me'), a general topic on its own ('we talked about work' = 0), feelings or opinions, or framing like 'during the call'. Examples: 'He told me he grew up on a farm in Wales' = 2 (a farm; Wales); 'She went quiet when I asked about her brother' = 2; 'We talked about films' = 0.",
    options=[O(0, "0"), O(1, "1"), O(2, "2"), O(3, "3 or more")]),
  Q(id="moment_text", read="MOMENT", type="single", q="Is the MOMENT text a report of what happened, or of the reporter's thoughts about it?", help="Read the MOMENT text only.", options=[
    O("WHAT_HAPPENED", "What happened"), O("THOUGHTS", "The reporter's thoughts or feelings about it"), O("BOTH", "Both")]),
  Q(id="shared_by", read="MOMENT", type="single", q="Did someone share something personal in the moment?", options=[
    O("REPORTER", "The reporter"), O("PARTNER", "The partner"), O("BOTH", "Both"), O("NOBODY", "Nobody")]),
  Q(id="personal_kinds", read="MOMENT", type="multi", q="What did someone share about themselves?", help="Anyone in the moment: the reporter or the partner.", options=[
    O("INFO_GENERAL", "Surface facts about themselves", "age, job, family, interests, hobbies", barak="information", level=2),
    O("INFO_REVEALING", "Revealing personal information", "their personal traits, a personal experience, or problems in their own or their family's behaviour", barak="information", level=3),
    O("THOUGHTS_EVENTS_PLANS", "Their thoughts about past events or future plans", barak="thoughts", level=2),
    O("THOUGHTS_INTIMATE", "Their thoughts about their own character, looks or health, or private wishes", barak="thoughts", level=3),
    O("FEELINGS_MILD", "Surface feelings", "mild feelings such as confusion or inconvenience; ordinary worries or frustrations", barak="feelings", level=2),
    O("FEELINGS_DEEP", "Deep feelings", "such as humiliation, anxiety, depression, fears or pain", barak="feelings", level=3),
    O("NOTHING", "Nothing personal", exclusive=True, level=1)])]),
 dict(id="B", title="Feelings and the reason", questions=[
  Q(id="feelings", read="BOTH", type="multi", max=3, q="Which feelings does the text show the reporter had about this moment?",
    help="Only feelings the text names or clearly shows. These are the emotions of the crowd-enVENT study (Troiano et al. 2023).", options=[
    O("JOY", "Joy"), O("PRIDE", "Pride"), O("RELIEF", "Relief"), O("TRUST", "Trust"), O("SURPRISE", "Surprise"), O("BOREDOM", "Boredom"),
    O("ANGER", "Anger"), O("DISGUST", "Disgust"), O("FEAR", "Fear"), O("GUILT", "Guilt"), O("SADNESS", "Sadness"), O("SHAME", "Shame"),
    O("OTHER", "Another feeling", "write it in", text=True), O("NONE", "No feeling shown", exclusive=True)]),
  Q(id="intensity", read="BOTH", type="single", q="How intense was that feeling, as the text shows it?", help="If the text shows no feeling, choose Not at all.", options=[
    O("CANT_TELL", "Can't tell"), O(1, "Not at all"), O(2, "A little"), O(3, "Moderately"), O(4, "Very"), O(5, "Extremely")]),
  Q(id="appraisal", read="BOTH", type="statements", q="Based on the text, how much do you agree with each statement?",
    help="'The event' is the remembered moment. Judge from what the text says; a sensible reading is fine, and Can't tell is only for when the text gives you nothing to go on. (Statements from the crowd-enVENT study; 'the experiencer' is called 'the reporter' here.)",
    scale=AGREE5, statements=[
     dict(code="pleasant", text="The event was pleasant for the reporter."), dict(code="unpleasant", text="The event was unpleasant for the reporter."),
     dict(code="own_behavior", text="The event was caused by the reporter’s own behavior."),
     dict(code="other_behavior", text="The event was caused by somebody else’s behavior.")]),
  Q(id="reason", read="REASON", type="nested_ranked", q="What is the reason mainly about?",
    help="Pick what the reason is about, in order of importance (first pick = most important). Each pick opens its options: choose all that apply.",
    options=[
     O("PARTNER", "The partner", "what they said, did or are like", sub=[
       O("VIEW", "I agreed or disagreed with their view", memo="fact_about_others.view"),
       O("PARTNER_LIFE", "Something about their life", "job, family, friends, activities", memo="fact_about_others.social_facts"),
       O("UNEXPECTED", "Something unexpected about them", memo="fact_about_others.unexpected_info"),
       O("BEHAVIOUR", "How they behaved", "a reaction, how they acted", memo="meta_behaviour_of_other.behaviour"),
       O("EMOTION", "An emotion they showed", memo="meta_behaviour_of_other.emotional_moment"),
       O("UNDERSTOOD", "I understood their point of view", memo="cognitive.cognitive_empathy")]),
     O("ME", "Me", "my feelings, my life or my views", sub=[
       O("MY_FEELINGS", "My feelings", "e.g. happy, embarrassed, nostalgic", memo="self_perception.annotator_feelings"),
       O("MY_LIFE", "My own life, stories or views", memo="self_perception.annotator_stories")]),
     O("US", "Us", "something between us", sub=[
       O("IN_COMMON", "We had something in common", "a similar experience or feeling", memo="shared_experience.shared_story")]),
     O("CALL", "The call itself", "when it happened, or the call as a whole", sub=[
       O("START", "It was at the very start", memo="time_label.first"), O("MIDDLE", "It was in the middle", memo=None),
       O("END", "It was at the very end", memo="time_label.last"), O("WHOLE_CALL", "The whole conversation", memo=None),
       O("MEDIUM", "The video call itself", "the format, the technology, being on camera", memo=None)]),
     O("WORLD", "The wider world", "an issue, an event, people in general", sub=[
       O("ISSUE", "An issue or event", "e.g. a strike, a lockdown", memo="fact_about_world.entities"),
       O("PEOPLE", "People in general", memo="fact_about_world.people")]),
     O("OTHER", "Something else", "write it in", memo=None, text=True), O("NO_REASON", "No real reason given", memo="none", exclusive=True)]),
  Q(id="matters_for", read="REASON", type="multi", q="Does the REASON say this moment matters for...", options=[
    O("FUTURE", "What they'll think or do in future", "they learned something useful, or will act differently"),
    O("WHO_I_AM", "Who they are, or how they see themselves"),
    O("CONNECTION", "Their connection with the partner", "getting to know them, feeling closer"),
    O("NONE", "None of these", "it only says why the moment stood out", exclusive=True)]),
  Q(id="bigger_meaning", read="REASON", type="single", q="Does the REASON connect the moment to something bigger than the moment itself?",
    help="A lesson, an insight about themselves, or about who the partner is.", options=[O("YES", "Yes"), O("NO", "No")])]),
 dict(id="C", title="With the transcript", models=False, questions=[
  Q(id="traceable", read="TRANSCRIPT", type="single", q="Can the moment be found in the transcript?", options=[
    O("SPECIFIC_EVENT", "One specific exchange"), O("GENERAL_TOPIC", "A general topic or stretch, not one exchange"),
    O("VISUAL_NONVERBAL", "Something seen or felt, not in the words"), O("TOO_VAGUE", "Too vague to tell"),
    O("CONTRADICTED", "The transcript says something different"), O("ASR_FAILURE", "The transcript is garbled where it would be")]),
  Q(id="locator", read="TRANSCRIPT", type="single", q="Did the locator (highlighted turns) pick the right turns?", options=[
    O("RIGHT", "Right"), O("PARTLY_RIGHT", "Partly right"), O("WRONG", "Wrong"), O("NOT_FINDABLE", "Nothing in the transcript matches")]),
  Q(id="locator_turns", read="TRANSCRIPT", type="text", q="If not right: the turns you would pick", help="e.g. 12-15"),
  Q(id="interaction", read="TRANSCRIPT", type="multi", max=2, q="What kind of interaction was it?", options=[
    O("QA", "Asking and answering questions"), O("TELLING", "One person telling about themselves"), O("FACT", "Sharing a fact or piece of information"),
    O("COMMON", "Discovering something in common"), O("JOKING", "Joking around"), O("DISAGREEING", "Disagreeing or debating"),
    O("META", "Talking about the study, the call or the technology"), O("OTHER", "Other", text=True)]),
  Q(id="acts", read="TRANSCRIPT", type="multi", q="What was said in the exchange?", options=[
    O("QUESTION", "A question"), O("ANSWER", "An answer"), O("STORY", "A personal story or statement"), O("OPINION", "An opinion"),
    O("AGREE", "Agreement"), O("DISAGREE", "Disagreement"), O("COMPLIMENT", "A compliment or thanks"), O("JOKE", "A joke"),
    O("RITUAL", "A greeting, introduction or goodbye"), O("OTHER", "Other", text=True)]),
  Q(id="notes", read="TRANSCRIPT", type="text", q="Notes (optional)")])]

SPEC = dict(version="3.5.1", storage_version="3.5", intro=INTRO, parts=PARTS)

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(SPEC, open(os.path.join(here, "SPEC_v3.5.1.json"), "w"), indent=1, ensure_ascii=False)
    print("questions", sum(len(p["questions"]) for p in PARTS))
