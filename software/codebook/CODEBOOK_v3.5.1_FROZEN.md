# Codebook v3.5.1 (generated from codebook/spec_v351.py; do not edit by hand)

Grounding, adaptations and procedure: see the header of codebook/CODEBOOK_v3.md.
Model coders see the text below with the codes in square brackets, Parts A and B only, plus the output format. People see the same words in the coding page.

```
HOW THIS WORKS
- Two strangers had a 3-minute video call. Right afterwards, one of them wrote down the moment they remembered most clearly (MOMENT) and why they think they remembered it (REASON).
- "The reporter" is the person reporting their memory of the conversation. "The partner" is the person they had the conversation with.
- Some reports were written by people and some by computer models imitating people; code them all the same way.
- Code only what the text says. Don't guess what probably happened, and don't imagine how you would have felt.
- Each question says which text to read: MOMENT, REASON, or both.
- Choose "Can't tell" only if the text gives you nothing to go on. A sensible reading of the text is fine. Where a question says 'up to 3', choose every answer that fits.
- If the MOMENT describes two or more separate moments, answer about the first one.

PART A: THE MOMENT

A1. What was the moment about? (Read: MOMENT. Choose up to 3.)
- Work, education or service [WORK_EDUCATION_SERVICE]
- Where someone lives or comes from, places, travel [PLACE_TRAVEL]
- Family, relationships or pets (including people they both know) [FAMILY_RELATIONSHIPS_PETS]
- Hobbies, interests, TV, music, sport [HOBBIES_INTERESTS_MEDIA]
- Plans, goals or recent events in someone's life [PLANS_EVENTS]
- Food [FOOD]
- Health or wellbeing [HEALTH_WELLBEING]
- Feelings or the mood of the call [FEELINGS_MOOD]
- This research study, the call itself or the technology (including introductions, names, swapping contacts) [THE_STUDY_OR_CALL]
- How the partner looked, sounded or came across [PARTNER_IMPRESSION]
- Something else (write what it is in the matching _other field) [OTHER]
- Can't tell (only on its own) [CANT_TELL]

A2. What is remembered? (Read: MOMENT. Choose up to 3.)
- Something the partner said or did [PARTNER_SAID_OR_DID]
- A fact or quality of the partner (not one particular remark) [PARTNER_FACT_OR_QUALITY]
- Something the reporter said or did [REPORTER_SAID_OR_DID]
- Something they did together (e.g. discovered something in common, made plans) [DID_TOGETHER]
- A thought, realisation or feeling the reporter had [REPORTER_THOUGHT]
- The situation (the call, the study, the technology) [SITUATION]
- Can't tell (only on its own) [CANT_TELL]

A3. Is a specific new piece of information remembered? (Read: MOMENT. Choose one.)
- Yes: a specific fact learned in the call (e.g. where they grew up; a train strike I hadn't heard of) [YES]
- No: it is about information, but nothing specific and new [NO]
- It isn't about information (a feeling, a laugh, the mood) [NOT_ABOUT_INFORMATION]

A4. How much of the call does the moment cover? (Read: MOMENT. Choose one.)
- One moment (a single remark, exchange or reaction) [ONE_MOMENT]
- A stretch of the call (a longer part: 'the conversation about travel') [STRETCH]
- The whole call, or a pattern across it ('we got on well', 'he was really friendly') [WHOLE_CALL_OR_PATTERN]
- Not an event (a fact or opinion about someone, with nothing happening) [NOT_AN_EVENT]

A5. How many details? (Read: MOMENT. Choose one.)
Count the separate, concrete pieces of information about what happened. Each counts once: a separate fact (a place, a name, a number, an object); a reaction ('she gasped' = 1); a quote, or something seen or heard ('her accent' = 1). Don't count who said it ('he told me'), a general topic on its own ('we talked about work' = 0), feelings or opinions, or framing like 'during the call'. Examples: 'He told me he grew up on a farm in Wales' = 2 (a farm; Wales); 'She went quiet when I asked about her brother' = 2; 'We talked about films' = 0.
- 0 [0]
- 1 [1]
- 2 [2]
- 3 or more [3]

A6. Is the MOMENT text a report of what happened, or of the reporter's thoughts about it? (Read: MOMENT. Choose one.)
Read the MOMENT text only.
- What happened [WHAT_HAPPENED]
- The reporter's thoughts or feelings about it [THOUGHTS]
- Both [BOTH]

A7. Did someone share something personal in the moment? (Read: MOMENT. Choose one.)
- The reporter [REPORTER]
- The partner [PARTNER]
- Both [BOTH]
- Nobody [NOBODY]

A8. What did someone share about themselves? (Read: MOMENT. Choose all that apply.)
Anyone in the moment: the reporter or the partner.
- Surface facts about themselves (age, job, family, interests, hobbies) [INFO_GENERAL]
- Revealing personal information (their personal traits, a personal experience, or problems in their own or their family's behaviour) [INFO_REVEALING]
- Their thoughts about past events or future plans [THOUGHTS_EVENTS_PLANS]
- Their thoughts about their own character, looks or health, or private wishes [THOUGHTS_INTIMATE]
- Surface feelings (mild feelings such as confusion or inconvenience; ordinary worries or frustrations) [FEELINGS_MILD]
- Deep feelings (such as humiliation, anxiety, depression, fears or pain) [FEELINGS_DEEP]
- Nothing personal (only on its own) [NOTHING]

PART B: FEELINGS AND THE REASON

B1. Which feelings does the text show the reporter had about this moment? (Read: MOMENT and REASON. Choose up to 3.)
Only feelings the text names or clearly shows. These are the emotions of the crowd-enVENT study (Troiano et al. 2023).
- Joy [JOY]
- Pride [PRIDE]
- Relief [RELIEF]
- Trust [TRUST]
- Surprise [SURPRISE]
- Boredom [BOREDOM]
- Anger [ANGER]
- Disgust [DISGUST]
- Fear [FEAR]
- Guilt [GUILT]
- Sadness [SADNESS]
- Shame [SHAME]
- Another feeling (write it in) (write what it is in the matching _other field) [OTHER]
- No feeling shown (only on its own) [NONE]

B2. How intense was that feeling, as the text shows it? (Read: MOMENT and REASON. Choose one.)
If the text shows no feeling, choose Not at all.
- Can't tell [CANT_TELL]
- Not at all [1]
- A little [2]
- Moderately [3]
- Very [4]
- Extremely [5]

B3. Based on the text, how much do you agree with each statement? (Read: MOMENT and REASON. Rate every statement.)
'The event' is the remembered moment. Judge from what the text says; a sensible reading is fine, and Can't tell is only for when the text gives you nothing to go on. (Statements from the crowd-enVENT study; 'the experiencer' is called 'the reporter' here.)
Answer scale: Don't agree at all [1] · Slightly agree [2] · Somewhat agree [3] · Mostly agree [4] · Completely agree [5] · Can't tell [CANT_TELL]
- The event was pleasant for the reporter. [pleasant]
- The event was unpleasant for the reporter. [unpleasant]
- The event was caused by the reporter’s own behavior. [own_behavior]
- The event was caused by somebody else’s behavior. [other_behavior]

B4. What is the reason mainly about? (Read: REASON. Rank the main targets; then choose all options that apply under each.)
Pick what the reason is about, in order of importance (first pick = most important). Each pick opens its options: choose all that apply.
- The partner (what they said, did or are like) [PARTNER]
    - I agreed or disagreed with their view [VIEW]
    - Something about their life (job, family, friends, activities) [PARTNER_LIFE]
    - Something unexpected about them [UNEXPECTED]
    - How they behaved (a reaction, how they acted) [BEHAVIOUR]
    - An emotion they showed [EMOTION]
    - I understood their point of view [UNDERSTOOD]
- Me (my feelings, my life or my views) [ME]
    - My feelings (e.g. happy, embarrassed, nostalgic) [MY_FEELINGS]
    - My own life, stories or views [MY_LIFE]
- Us (something between us) [US]
    - We had something in common (a similar experience or feeling) [IN_COMMON]
- The call itself (when it happened, or the call as a whole) [CALL]
    - It was at the very start [START]
    - It was in the middle [MIDDLE]
    - It was at the very end [END]
    - The whole conversation [WHOLE_CALL]
    - The video call itself (the format, the technology, being on camera) [MEDIUM]
- The wider world (an issue, an event, people in general) [WORLD]
    - An issue or event (e.g. a strike, a lockdown) [ISSUE]
    - People in general [PEOPLE]
- Something else (write it in) (write what it is in the matching _other field) [OTHER]
- No real reason given (only on its own) [NO_REASON]

B5. Does the REASON say this moment matters for... (Read: REASON. Choose all that apply.)
- What they'll think or do in future (they learned something useful, or will act differently) [FUTURE]
- Who they are, or how they see themselves [WHO_I_AM]
- Their connection with the partner (getting to know them, feeling closer) [CONNECTION]
- None of these (it only says why the moment stood out) (only on its own) [NONE]

B6. Does the REASON connect the moment to something bigger than the moment itself? (Read: REASON. Choose one.)
A lesson, an insight about themselves, or about who the partner is.
- Yes [YES]
- No [NO]

PART C: WITH THE TRANSCRIPT

C1. Can the moment be found in the transcript? (Read: the transcript. Choose one.)
- One specific exchange [SPECIFIC_EVENT]
- A general topic or stretch, not one exchange [GENERAL_TOPIC]
- Something seen or felt, not in the words [VISUAL_NONVERBAL]
- Too vague to tell [TOO_VAGUE]
- The transcript says something different [CONTRADICTED]
- The transcript is garbled where it would be [ASR_FAILURE]

C2. Did the locator (highlighted turns) pick the right turns? (Read: the transcript. Choose one.)
- Right [RIGHT]
- Partly right [PARTLY_RIGHT]
- Wrong [WRONG]
- Nothing in the transcript matches [NOT_FINDABLE]

C3. If not right: the turns you would pick (Read: the transcript. Write a short answer.)
e.g. 12-15

C4. What kind of interaction was it? (Read: the transcript. Choose up to 2.)
- Asking and answering questions [QA]
- One person telling about themselves [TELLING]
- Sharing a fact or piece of information [FACT]
- Discovering something in common [COMMON]
- Joking around [JOKING]
- Disagreeing or debating [DISAGREEING]
- Talking about the study, the call or the technology [META]
- Other (write what it is in the matching _other field) [OTHER]

C5. What was said in the exchange? (Read: the transcript. Choose all that apply.)
- A question [QUESTION]
- An answer [ANSWER]
- A personal story or statement [STORY]
- An opinion [OPINION]
- Agreement [AGREE]
- Disagreement [DISAGREE]
- A compliment or thanks [COMPLIMENT]
- A joke [JOKE]
- A greeting, introduction or goodbye [RITUAL]
- Other (write what it is in the matching _other field) [OTHER]

C6. Notes (optional) (Read: the transcript. Write a short answer.)
```
