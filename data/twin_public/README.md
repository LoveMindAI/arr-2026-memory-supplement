# A public, fully synthetic stand-in for the memory study

The stand-in is a fully synthetic dataset for running the paper's pipeline (prompts, coding and scoring scripts) without access to the original study. It is built from public sources. Its numbers are not the paper's numbers. The paper's results reproduce from the aggregate result files.

## What comes from where
- **Demographics.** The Office for National Statistics Census 2021 public microdata teaching sample, England and Wales (`public_data/ons2021_teaching_sample.csv`). Category labels come from the ONS API (`public_data/ons2021_labels.json`).
  - ONS: Adapted from data from the Office for National Statistics licensed under the Open Government Licence v.3.0 (https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
  - No census record becomes a person. Each person is assembled from four census draws (with replacement within a matched cell) (see `build_people.py`).
- **Personality.** Personality is sampled from published HEXACO-60 adult norms. These are not UK norms.
  - **Base:** the unweighted average of the eight Western European and US adult samples in García et al. 2022 (Supporting Information, Table A3; mean age about 42).
  - **Sex differences** are set to those reported for UK online respondents (Lee & Ashton 2020, Table 2, p. 1082), split around each domain's centre (women at centre + d·s/2, men at centre − d·s/2). Those d values are age-corrected and come from the 100-item form; on the 60-item form they would be about 4% smaller, which we ignore.
  - **Correlation:** the six domains are drawn together, using García's 18-country average intercorrelations (Supporting Information, Table A10) and the within-sex SDs, then clipped to 1–5.
  - **Age:** Honesty-Humility rises with age, by about .008 SD per year around age 41.6 (García et al. 2022, Table 4, p. 267; age SD, p. 259). No facets are sampled.
  - The parameters are in `hexaco_norms.json`, and the sources and pages are in `SOURCES.md` §1C and §1F.
- **How people talk.** A style guide (`style_guide.md`) written from public descriptions of British conversational English (`SOURCES.md`).
- **From the original study: numbers only** (`targets_from_parsel.json`, written by `extract_targets.py`, the only script that reads it):
  - memory word lengths;
  - call shape (turns, words, duration, group sizes, number of calls);
  - the shares and defining profiles of the two memory types.

  No text, record, score or code from the original study reaches any generator. The two survey questions use the wording the paper quotes.

## Pipeline (rerun in this order)
1. `python3 build_people.py` builds people, sessions, the call schedule, memory types and length targets. It's local and free.
2. `python3 gen.py traits` samples personality scores from the norms. Local.
3. `python3 gen.py card` writes persona cards with Claude Opus 5.5.
4. `python3 gen.py rate` has Gemini 3 Flash preview record first impressions of the webcam photos.
5. `python3 gen.py convo` writes whole sessions, round by round, with Claude Opus 5.5.
6. `python3 gen.py memory` writes the post-call answers in persona with Claude Opus 5.5.
7. `python3 code_standin.py` codes the memories with codebook 3.5.1 (Claude Haiku 5.5 and GPT-6 Luna, the paper's coding prompt).
8. `python3 leak_audit.py` and `python3 validate.py` produce `LEAK_AUDIT.md` and `VALIDATION.md`.

Every stage is resumable, and every API call is logged with its cost in `<stage>/calls.jsonl`. `python3 gen.py spent` prints the total spent so far. All stages accept `--sessions S01,S02` to run a pilot.

## The audio/video-only subset
One call per session (55), chosen at random by design (`av_plan.py`), carries one non-verbal event on one person's side. The event is something a camera or microphone would catch, written into a hidden track (`av_events.json`) and kept out of the transcript. The other person's memory is about it, and those memories are marked `requires_av: true`, with an `av_event` pointer.

`av_check.py` checks the subset in three ways: a keyword scan, a judge asking whether the transcript hints at the event, and the paper's text-only locator. Two memories were dropped from the subset because their transcripts hinted at the event (marked `av_excluded`). The results are in `VALIDATION.md`.

## Ages and sex
An author, Oct 11: these follow the published PARSEL summary (Hrkalovic et al. 2025, p. 3491: 297 participants, 166 women, mean age 37.39, SD 11.42).
- Ages are drawn from a normal distribution with that mean and SD, and any draw below 18 is redrawn. Truncating at 18 lifts the realized mean to about 39.
- Each person's census donors match their census age band and sex. Region and everything else follow the census within that band.

## Status (complete)
293 people, 55 sessions, 583 calls, 1,166 memories, every memory coded by both coders. See `VALIDATION.md` and `LEAK_AUDIT.md`. An author's codes for 40 of these memories are in `author_codes/A1_publicstandin_v3.5.json`, with the item map in `author_codes/items.json`.

Not for the packet: the pilot folders (`pilot1/`, `pilot2/`, `pilot2_aborted/`, `redo_S10/`), and the source texts (moved to `../lit/twin_public_sources/`).
