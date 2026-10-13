# Validation of the stand-in (validate.py)

Compared ONLY on what it was built to match: lengths, call shape and the two memory types. 1166 memories, 583 calls.

## Lengths (word quantiles)
- **moment_words**: stand-in p10 8, p25 12, p50 18, p75 27, p90 38; target p10 6, p25 10, p50 16, p75 24, p90 36
- **reason_words**: stand-in p10 6, p25 8, p50 13, p75 20, p90 29; target p10 5, p25 8, p50 12, p75 18, p90 26
- **turns_per_call**: stand-in p10 27, p25 29, p50 31, p75 34, p90 38; target p10 20, p25 24, p50 30, p75 35, p90 39
- **words_per_call**: stand-in p10 416, p25 454, p50 500, p75 551, p90 603; target p10 440, p25 500, p50 570, p75 624, p90 683
- moment_words, log-length SD between people 0.51 (target 0.52), within 0.27 (target 0.33)
- reason_words, log-length SD between people 0.50 (target 0.46), within 0.29 (target 0.35)

## Memory types (share of people; coded profile of each type's memories, mean of Haiku and Luna)
- **shared_stretch**: 0.51 of people (target 0.51); 565 coded memories
  - scope:one_moment: 0.42 (target 0.33)
  - scope:stretch: 0.44 (target 0.48)
  - what:did_together: 0.62 (target 0.61)
  - what:partner_said_or_did: 0.40 (target 0.22)
  - new_fact: 0.36 (target 0.25)
- **what_they_were_told**: 0.49 of people (target 0.49); 546 coded memories
  - scope:one_moment: 0.62 (target 0.65)
  - scope:stretch: 0.18 (target 0.16)
  - what:did_together: 0.23 (target 0.20)
  - what:partner_said_or_did: 0.71 (target 0.62)
  - new_fact: 0.70 (target 0.72)

## Audio/video-only subset
One call per session, chosen at random by design (seed in av_plan.py), carries one non-verbal event on one person's side, written into a hidden track (av_events.json) and kept out of the transcript; the other person's memory is about it (requires_av: true). Of 55 planned, 2 were dropped from the subset because a judge (Gemini 3 Flash preview) found the transcript hinting at the event; 53 remain. On those, the paper's text-only locator (prompt v1, Claude Sonnet 5.5) said not_in_transcript for 42, general for 0, and placed 11 in the conversation; the placed ones are mostly a face or voice changing during a spoken line, where the words give the timing but not what was noticed (av_check.json has each locator note). A keyword scan of the event description against the transcript flagged common words in nearly every call and was not informative.

## Checkpoint (after S01–S12; Oct 11 2026, 19:54 EDT)
At 12 sessions (246 memories, 234 outside the audio/video subset) both memory types matched their targets closely on scope and on what is remembered. The stretch type was a little high on the partner's words (.38 vs .22) and new facts (.37 vs .25); the told type a little high on stretch (.26 vs .16). Lengths: moment median 17 (16), reason 13 (12), between-person log SD .53 (.52); calls 516 words over 31 turns (570 over 30). **Nothing was clearly off, so nothing was adjusted** (the coordinator's rule: one checkpoint, adjust once only if clearly off).

Changes made before the checkpoint, during pilots:
- (pilot 1) explicit notes for every memory slot;
- (pilot 2) a call-length instruction, a no-quoting rule for "together", and length ceilings;
- (Oct 11) the stretch note softened one notch.

Execution fix at the checkpoint: session S10 had two people named Gareth, and one call couldn't be parsed. Transcript labels are now unique within a session, and S10 was regenerated.
