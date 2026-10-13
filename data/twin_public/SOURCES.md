# Public sources for the stand-in dataset (lit room, Oct 11 2026, 19:05 EDT)
*For twin_public/PLAN.md. Every source below was read by the lit room: the PDF was opened and pages were pinned. Bib records come from Crossref (`twin_public/sources.bib`), with three year quirks noted at the end. Nothing restricted was downloaded. Paywalled papers are listed for A1 to fetch and have not been read.*

## The short version
1. **No readable source gives HEXACO-60 means and SDs by sex and age band together.** What we have is:
   - HEXACO-60 domain norms by sex, from the scale authors (hexaco.org, Canadian students);
   - HEXACO-100 facet norms by sex, from the same page;
   - an open 18-country HEXACO-60 study that gives age-band design and age correlations, and says age effects are small except for Honesty-Humility.
   
   The age-by-sex means are in two places we can't read yet: that study's supplement (Wiley blocked the download) and a paywalled 2016 paper. A1 could fetch both (list below).
2. **Usable now:** a HEXACO-60 base from §1A, with UK sex differences from Lee & Ashton 2020 (§1C, read Oct 11) swapped in for the students' gap, and an age trend for Honesty-Humility from §1B. The recipe, numbers and traps are under "What the builder could do". Combining them is A1's call.
3. **Spoken British English:** eight open sources, seven of them peer-reviewed, cover fillers, hedges, tag questions, politeness words and named regional differences in words and grammar. All are freely readable, and paraphrasing them into a style guide is fine under their licences (CC BY for six; CC BY-NC-ND for one; one conference slide deck with no licence).

## 1. HEXACO-60 norms

### 1A. Domain means and SDs by sex (scale authors; hexaco.org only)
Source: Ashton & Lee, "Descriptive Statistics and Internal Consistency Reliabilities of the HEXACO-60 Scales in a College Student Sample", hexaco.org/downloads/descriptives_60.pdf. It's a one-page table linked from hexaco.org/hexaco-inventory as "Descriptive Statistics (English version; Canadian undergraduate students)". **Only on hexaco.org; not peer-reviewed.**

Self-report form, 1–5 scale. N = 1,126 in total, 691 women and 429 men (sex missing for 6):

| Domain | α | Total M (SD) | Women M (SD) | Men M (SD) |
|---|---|---|---|---|
| Honesty-Humility | .76 | 3.23 (.66) | 3.31 (.64) | 3.09 (.69) |
| Emotionality | .80 | 3.36 (.70) | 3.66 (.55) | 2.87 (.64) |
| Extraversion | .80 | 3.51 (.62) | 3.48 (.64) | 3.54 (.60) |
| Agreeableness | .77 | 3.10 (.63) | 3.06 (.59) | 3.16 (.68) |
| Conscientiousness | .76 | 3.47 (.61) | 3.54 (.58) | 3.34 (.63) |
| Openness | .78 | 3.49 (.67) | 3.52 (.64) | 3.44 (.71) |

Caveats:
- These are students, so the sample is young.
- The peer-reviewed HEXACO-60 paper (Ashton & Lee 2009, *JPA* 91(4):340–345) reports "samples of college students and community adults" (abstract, read via OpenAlex). It's paywalled, so I can't confirm that this table matches it, or give the community sample's numbers.

**Facets:** the same page offers HEXACO-100 facet means and SDs by sex for the same students (hexaco.org/downloads/descriptives_100.pdf, read). Note, though, that HEXACO-60 facets have only 2–3 items each and "are very short ... and thus rather unreliable" (García et al. 2022, p. 260, citing Ashton & Lee 2009). HEXACO-100 facet SDs (4 items each) will therefore understate the spread of HEXACO-60 facet scores. hexaco.org also says "The items of the 60-item version are a subset of the items of the 100-item version" (hexaco-inventory page).

**hexaco.org terms (read):** the forms are free "only for the purpose of non-profit academic research", and data collection through public online surveys is not allowed. We use only published summary numbers and never administer items, so this doesn't touch the stand-in. Still, don't put HEXACO item text in any persona or prompt.

### 1B. Age, from an open peer-reviewed HEXACO-60 study
García et al. 2022, *Journal of Personality* 90(2):256–276, doi:10.1111/jopy.12664. **Open access, CC BY-NC-ND** (the numbers are facts we can use; the licence restricts redistributing the article itself). Read in full.
- **Design** (p. 260): community adults in 17 of 18 countries, collected by snowball sampling, each student recruiting "four men and four women" across four age bands: 18–30, 31–45, 46–60 and over 60. The USA sample came from MTurk with the same quotas. **There is no UK sample**; the 18 countries are listed in Table 4.
- **Sample** (p. 259): 10,298 people. Mean age 40.31 (SD 17.32); 36.5% aged 18–30, 23.1% 31–45, 23.4% 46–60, 17% over 60.
- **Age** (p. 264):
  - "only Honesty-Humility scores increase from 18 to 60 years in both genders, being higher in women. In the other five traits, the largest difference is lower than 4 points in the same scale, suggesting a negligible effect" (that's in their summed-score metric).
  - Average partial η² for age is below .03 for H-H and .01–.02 elsewhere.
  - "women and men tend to show the same pattern with age".
- **Per-country partial correlations with age** (Table 4, p. 267): e.g. Honesty-Humility r = .37 in the USA and .23 in Germany; 14 of 18 countries are positive and significant for H-H (p. 264).
- **Sex** (p. 264, Table 4, p. 267): Emotionality is the only large sex difference across countries; "Women score higher than men in all significant differences."
- **Domain intercorrelations:** averaged across countries in Supporting Information Table A10 (p. 264). **Blocked**; A1 could fetch it.
- **Means by age band × sex:** Supporting Information Figure A2 (p. 264). **Blocked.**

### 1C. UK sex differences: Lee & Ashton 2020 (read in full; A1's copy)
Lee & Ashton 2020, "Sex differences in HEXACO personality characteristics across countries and ethnicities", *J. Personality* 88(6):1075–1090, doi:10.1111/jopy.12551.
- **Form and scale** (p. 1080, §2.2.1): "The English-language version of the HEXACO-100 (Lee & Ashton, 2018)", taken online. "Responses were made on a 5-point scale (1 = strongly disagree, 5 = strongly agree)." **So it's the 100-item form, not the 60.**
- **Sample** (p. 1078):
  - hexaco.org, Oct 2014 – Oct 2018; people took it "primarily for self-exploration purposes";
  - kept: those who completed all items, passed three attention checks and two quality checks;
  - 370,857 in total, mean age 30.2 (SD 12.7).
- **UK** (Table 1, p. 1079): 30,597 men and 23,840 women; mean age 31.47 (SD 13.69).
- **How d was computed** (footnote 4, p. 1089):
  - "Scores on each personality variable were regressed on age, and the resulting residualized scores were used"; so the d values are **age-corrected**;
  - "d = (Mean women − Mean men)/[(SD women + SD men)/2]", so positive means women higher, and **the denominator is the average of the two sexes' SDs** (not a pooled SD).
- **UK row, Table 2 (p. 1082), d (women − men):**

| Domains | H .44 | E .97 | X −.07 | A −.04 | C .06 | O −.15 |
|---|---|---|---|---|---|---|

| Facets | | | | |
|---|---|---|---|---|
| H: Sincerity .10, Fairness .38, Greed Avoidance .29, Modesty .54 | E: Fearfulness .77, Anxiety .58, Dependence .60, Sentimentality .80 | X: Social Self-Esteem −.22, Social Boldness −.15, Sociability .15, Liveliness −.01 | A: Forgiveness −.04, Gentleness .11, Flexibility .05, Patience −.20 | C: Organization .17, Diligence .03, Perfectionism .07, Prudence −.12 |
| O: Aesthetic Appreciation .26, Inquisitiveness −.49, Creativity −.02, Unconventionality −.24 | Altruism .55 (HEXACO-100 only; **the HEXACO-60 has no Altruism**) | Mahalanobis D: 1.05 (6 domains), 1.32 (25 facets) | | |

- **No raw means or SDs anywhere in the article.**
  - Table S4 (supplement, not fetched) gives means and SDs of the *age-corrected* scores per sex × country (footnote 4, p. 1089). Those scores are residuals ("a mean of zero"), so S4 can't give a raw base, though its SDs could give the UK spread by sex.
  - Table 3 (p. 1085) gives age-corrected Emotionality by ethnic group, which we shouldn't use.
- **Weird-check, and it matters:** every page of this PDF (and the .txt beside it) carries a Wiley download stamp with A1's full name and the date. It must **not** go into the ARR packet: that would break anonymity and Wiley's terms. See "Packet hygiene" below.

### 1D. Domain intercorrelations (bonus), as far as readable
Lee & Ashton (HEXACO-100; *Assessment* 25(5):543–556) say in their abstract (read via OpenAlex): "the HEXACO-100 factor scales showed fairly low intercorrelations, with only one pair of scales (Honesty-Humility and Agreeableness) having an absolute correlation above .20 in self-report data." The numbers are in the paywalled paper.

Sampling the six domains independently would therefore be a reasonable first approximation. With the paper (or García's Table A10), we could add the H–A correlation.

### 1E. Prolific's UK participant pool, for reweighting demographics (read Oct 11 2026, 19:18 EDT)
**Bottom line: no published, citable statistics on the age or sex make-up of Prolific's UK pool turned up.** The only current numbers live in Prolific's audience checker, which needs a person with a browser. One closer option exists, but it's A1's call (last bullet).

- **Prolific's own help pages (read):**
  - "Who are the participants on Prolific?" (researcher-help.prolific.com, article 445224, dated June 18, 2026) lists the supported countries and says "All participants on Prolific are at least 18 years old". It points to the audience checker and gives no age, sex or student breakdown.
  - "What are representative samples on Prolific" (article 445161, dated July 23, 2026) gives no pool statistics either. It says their UK representative samples use "Combined 2021/2022 data for England & Wales, Northern Ireland and Scotland" by age × sex × ethnicity. That's census data, not the pool.
- **Prolific audience checker** (app.prolific.com/audience-checker; public, no login):
  - It shows "Participants active in the past 90 days" for any set of filters. Small counts are obfuscated ("Fewer than ...", read from the app's own text).
  - It's a live JavaScript tool, so I didn't automate it; that would mean calling an undocumented API.
  - **A 5-minute manual read gets the numbers:** set Country of residence = United Kingdom, add each age band × sex in turn, then student status, and write down each count with the date. Keep a screenshot.
  - Cite as "Prolific (2026), audience checker, accessed <date>". It's as current as it gets (the past 90 days).
  - It measures who is *active*, not who ends up completing a given study. Fast, frequent participants fill studies first.
- **Papers checked (both read):**
  - Douglas et al. 2023 (*PLOS ONE* 18(3):e0279720, CC BY): US only ("adult participants (N = 2729) from the United States", p. 5).
  - Peer et al. 2022 (*Behavior Research Methods* 54:1643–1662): US only; both studies recruited participants "residing in the United States" (Europe PMC full text).
  - So neither gives UK figures.
- **Not read:** Palan & Schitter 2018, "Prolific.ac—A subject pool for online experiments" (*J. Behavioral and Experimental Finance* 17:22–27, CC BY). Elsevier blocked the download. It describes the platform around 2017, so it's too old for today's pool anyway.
- **A1-GATE option:** PARSEL's own published summary is a Prolific UK sample: "297 participants (166 females; Mage = 37.39, SDage = 11.42)", mostly of UK nationality (Hrkalovic et al. 2025, p. 3491).
  - It's the closest match by far, and it's a published aggregate, not a row.
  - But PLAN.md's line says nothing from PARSEL beyond memory, turn and call lengths and the memory-type shares. Using its age and sex would cross that line, so it's A1's call.

### 1F. Adult community base: García et al. Supporting Information, Tables A1–A3 and A10 (A1's copy; read Oct 11 2026, 19:26 EDT)
The supplement belongs to `garcia2022hexaco60`. It's a Word file with no page numbers, so cite it by table.

**What's there:**
- **Table A1:** sample N, sex split and age per country.
- **Table A2:** HEXACO-60 domain M, SD and α per country (sexes combined).
- **Table A3:** domain M and SD **per sex** per country.
- **Tables A4–A9:** facet scores per sex. These are item *sums* of 2–3 items, not means.
- **Table A10:** domain intercorrelations averaged over the 18 countries.

**Which sample is closest to the UK?** There's no UK sample.
- **USA** is the only English-language one (N = 438: 222 women, 216 men; mean age 44.65, SD 15.08; Table A1). It was recruited on MTurk (article, p. 260), and among the Western samples it sits at the edge: lowest Extraversion (3.19), highest Conscientiousness (3.79), and the largest SDs (.62–.80; Table A3).
- The **seven Western European community samples** (Belgium, Catalonia, Germany, Italy, Spain, Switzerland F and G) are close to one another. They used paper and pencil, were sex-balanced by design, and had mean ages of 36–45.
- The article reports that scalar invariance fails across countries (p. 264), so country mean differences are partly measurement artefacts. That argues against betting on any single country.

**Lit room's recommendation (A1's call):** use the **unweighted average of the eight Western samples**: the seven above plus the USA. N = 4,370; the samples' mean ages average 41.6 (Table A1). Germany alone (N = 700, mean age 45.22) is the best single-country fallback; it sits close to that average except for slightly higher Extraversion.

**Calculation (lit room, from Table A3):** for each sample, centre = (M women + M men)/2 and s = (SD women + SD men)/2. Then average the centres and the s values over the eight samples. Finally split by the UK d from §1C: women = centre + d·s/2, men = centre − d·s/2.

| Domain | Western-8 centre | s | UK d (§1C) | **Women M** | **Men M** | for comparison: USA centre / Germany centre |
|---|---|---|---|---|---|---|
| Honesty-Humility | 3.554 | .661 | .44 | **3.70** | **3.41** | 3.470 / 3.560 |
| Emotionality | 3.167 | .568 | .97 | **3.44** | **2.89** | 3.055 / 3.155 |
| Extraversion | 3.432 | .583 | −.07 | **3.41** | **3.45** | 3.185 / 3.550 |
| Agreeableness | 3.210 | .574 | −.04 | **3.20** | **3.22** | 3.325 / 3.265 |
| Conscientiousness | 3.648 | .574 | .06 | **3.67** | **3.63** | 3.785 / 3.610 |
| Openness | 3.398 | .656 | −.15 | **3.35** | **3.45** | 3.575 / 3.395 |

Use SD = s within each sex, and clip to 1–5.

**For comparison:**
- Each sample's own sex gap (d) differs from the UK one. For Emotionality, the USA's is +.70 and Germany's +1.20, against the UK's .97. That's why we split around the centre rather than keep a country's own gap.

**Age:**
- The centre represents adults of about 42, not students. That's an improvement on the hexaco.org student base in §1A, which this replaces.
- Honesty-Humility (H-H) is the only domain that changes much with age. A person of age a gets **centre + (.145 / 17.32) · s · (a − 41.6)** added to their H-H. Here .145 is the median of García's per-country partial r for H-H with age (lit room's calculation from Table 4, p. 267), and 17.32 is the SD of age (p. 259). That's roughly +.06 on the 1–5 scale per decade.
- The other domains get no age term (p. 264).
- The UK d values are already age-corrected, so nothing is counted twice.

**Domain correlations (bonus, Table A10, averaged over 18 countries):**

|  | E | X | A | C | O |
|---|---|---|---|---|---|
| H | .02 | .05 | .29 | .25 | .08 |
| E |  | −.18 | −.08 | −.02 | −.06 |
| X |  |  | .13 | .19 | .16 |
| A |  |  |  | .12 | .09 |
| C |  |  |  |  | .13 |

On the HEXACO-60, two pairs exceed .20: H–A .29 and H–C .25. (The HEXACO-100 abstract in §1D says only H–A does.) A sampler can use this matrix with the per-sex SDs. The lit room checked that it's positive definite (smallest eigenvalue .635).

**What changes for the builder:** the base table to use is this one, not the §1A student table. Everything else in "What the builder could do" stands.

### Paywalled; A1 to fetch if he wants age-by-sex numbers (none read unless marked)
- Ashton & Lee 2016, "Age trends in HEXACO-PI-R self-reports", *J. Research in Personality* 64:102–111, doi:10.1016/j.jrp.2016.08.008. **The best source for age-by-sex trends at domain and facet level.** Note it uses the longer HEXACO-PI-R, not the 60.
- ~~Lee & Ashton 2020~~: now read (§1C). It has UK sex differences but no raw means. Its supplement Table S4 (age-corrected SDs by sex × country) is still unread.
- Ashton & Lee 2009 (above): the HEXACO-60's own paper, with the community adult sample.
- Lee & Ashton (*Assessment* 25(5):543–556), doi:10.1177/1073191116659134: the domain and facet intercorrelations.
- ~~García et al. 2022 Supporting Information~~: now read (§1F), from A1's copy. Tables A1–A3 and A10 are used; Figure A2 is an image and wasn't read.
- **Weird-check:** a copy of the *Assessment* paper is hosted on a University of Chicago site (found in a web search). It's a third-party copy, so don't use or cite it.

### What the builder could do now (a design choice for A1, not a decision)
**Recipe (lit room's judgment: defensible if the README says what it is):**
1. **Base:** for each domain, take the centre and the within-sex SD from the hexaco.org HEXACO-60 table (§1A):
   - centre c = (M women + M men)/2;
   - s = (SD women + SD men)/2, which matches Lee & Ashton's denominator.
2. **Replace the student sex gap with the UK gap.** Don't add the UK gap on top of the student one:
   - women's mean = c + d·s/2, men's mean = c − d·s/2, with d from Table 2 (§1C);
   - SD within each sex: s, or the per-sex SDs from §1A.
3. **Age, Honesty-Humility only** (García p. 264):
   - add a slope per year in SD units of about r / SD(age);
   - García's per-country partial r for H-H with age has a median of .145 across the 18 countries (lit room's calculation from Table 4, p. 267), and SD(age) = 17.32 (p. 259). That's about .008 SD per year, so roughly a third of an SD from age 20 to 60;
   - this is an illustration; how big to make it is A1's call.
   
   This doesn't double count, because the UK d values are already age-corrected.
4. **Domains independent** (§1D), clipped to 1–5.

**Resulting domain means by sex** (lit room's calculation from §1A and §1C; HEXACO-60 metric):

| Domain | student d (W−M), from §1A | UK d | women M | men M | s |
|---|---|---|---|---|---|
| H | .33 | .44 | 3.35 | 3.05 | .665 |
| E | 1.33 | .97 | 3.55 | 2.98 | .595 |
| X | −.10 | −.07 | 3.49 | 3.53 | .62 |
| A | −.16 | −.04 | 3.10 | 3.12 | .635 |
| C | .33 | .06 | 3.46 | 3.42 | .605 |
| O | .12 | −.15 | 3.43 | 3.53 | .675 |

**Traps:**
- **Double counting:** the §1A table already has a sex gap, and for E it's much bigger than the UK one (1.33 vs .97). Shifting the per-sex means by the UK d would stack two gaps. Use the centre, as in step 2.
- **100 vs 60:** the UK d values come from the HEXACO-100. A shorter scale is a little less reliable, so its true d is a little smaller. With the hexaco.org αs (H .76 on the 60 vs .83 on the 100), the shrink factor is about √(.76/.83) ≈ .96. That's small, and fine to ignore if the README says so.
- **Different denominator:** none. Lee & Ashton divide by the average of the two SDs, and step 1 uses that same average.
- **Facets:**
  - The facet d values are from 4-item HEXACO-100 facets. HEXACO-60 facets have 2–3 items (García p. 260), so facet gaps shrink more on the 60.
  - Drop Altruism.
  - There's no HEXACO-60 facet base on hexaco.org; only the HEXACO-100 one in descriptives_100.pdf.
  - Simplest: sample domains, then generate each facet as domain + noise, nudged by its facet d. Or skip facets if the persona cards don't need them.
- **Populations:**
  - The base is Canadian undergraduates, whose age isn't given on the page; treating them as about 20 is an assumption.
  - The UK d values come from self-selected hexaco.org visitors (mean age 31.5).
  - README wording: "*Personality is sampled from published HEXACO-60 student norms, with sex differences set to those reported for UK online respondents (Lee & Ashton 2020) and an age trend for Honesty-Humility from an 18-country community study (García et al. 2022); not UK norms.*"

**Packet hygiene:**
- Don't ship `lit/twin_public_sources/`:
  - The Lee & Ashton PDF and .txt carry A1's name.
  - The hexaco.org PDFs and the article texts are other people's copyrighted work (most of them CC BY, but there's no reason to ship them).
- The packet should cite, not include. The lit room suggests moving `../lit/twin_public_sources/` out of `twin_public/` (e.g. to `lit/twin_public_sources/`) before any packet build. That's the writer room's call.

## 2. Spoken British English, for a writer's style guide (all read)

| Source | Licence | What it gives the style guide (page) |
|---|---|---|
| Love, Dembry, Hardie, Brezina & McEnery 2017, "The Spoken BNC2014", *Int. J. Corpus Linguistics* 22(3):319–344, doi:10.1075/ijcl.22.3.02lov | CC BY | The anchor: 11.5M words of informal conversation among L1 British English speakers, 2012–2016 (p. 319). Dialect categories by region: North East, Yorkshire & Humberside, North West, Merseyside, East and West Midlands, Eastern, South West, South East, London, Scotland, Wales, Northern Ireland (Table 2, p. 331). The tag question *isn't it* is transcribed as "isn't it, ain't it, innit, etc." (p. 337). Fillers were transcribed by sound form only (pp. 334–335). |
| Wieling, Grieve, Bouma, Fruehwald, Coleman & Liberman 2016, "Variation and change in the use of hesitation markers in Germanic languages", *Language Dynamics and Change* 6(2):199–234, doi:10.1163/22105832-00602001 | author version (Aston repository); no licence stated | British spelling of fillers: "erm and er in British English" (§1). In the spoken BNC, women and younger speakers favour *erm* over *er* (§5, Table 1); the UM form is increasing across Germanic languages. Page numbers are from the author version, so cite by section. |
| Reichelt 2021, "Recent developments of the pragmatic markers *kind of* and *sort of* in spoken British English", *English Language and Linguistics* 25(3):563–580, doi:10.1017/S1360674321000253 | CC BY | Hedges: *kind of* is spreading into contexts that belong to *sort of*, with younger speakers leading; *sort of* is stable (abstract, p. 563). |
| Islentyeva, Pesendorfer & Tolochin 2023, "'Can I have a cup of tea please?' Politeness markers in the Spoken BNC2014", *J. Politeness Research* 19(2):297–322, doi:10.1515/pr-2022-0010 | CC BY | *please*, *thank you*, *thanks* and *sorry* are used mainly in their core senses; other senses include "self-correcting and being confrontational or ironic" (abstract, p. 297). In short, *sorry* works as a repair word too. |
| Torgersen & Gabrielatos 2009, "A corpus-based study of invariant tags in London English", Corpus Linguistics 2009 (Liverpool), Lancaster eprint 26837 | **slides plus abstract, not a peer-reviewed paper; no licence stated** | London tags: *innit*, *yeah*, *right*, *okay*, *you know*, *you get me*, *do you know what I mean* (abstract, p. 1). *innit* and *you get me* are features of young people's speech and are rising; *right* and *okay* are falling (p. 1). **Caution:** they link the innovative tags to young, male, non-Anglo inner-London speakers. The writer should never use ethnicity as a dialect switch; age and region only. |
| MacKenzie, Bailey & Turton 2022, "Towards an updated dialect atlas of British English", *J. Linguistic Geography* 10:46–66, doi:10.1017/jlg.2022.2 | CC BY | **Regional differences we can state in words** (14,438 respondents; see the regional table below). |
| Grieve, Montgomery, Nini, Murakami & Guo 2019, "Mapping lexical dialect variation in British English using Twitter", *Frontiers in AI* 2:11, doi:10.3389/frai.2019.00011 | CC BY | A ready list of everyday British word choices from the BBC Voices survey (Table 1, p. 4), e.g. *knackered/shattered*, *chuffed/made up*, *skint/brassic*, *mardy/stroppy*, *bairn/wean*, *mum/mam*, *mate/pal*, *settee/sofa/couch*, *ginnel*, *beck/burn*. The table doesn't say which region uses which word (that's in their maps), so use it as a pool of British words, not as regional rules. |
| Tantucci & Wang 2024, "British conversation is changing: resonance and engagement in the BNC1994 and the BNC2014", *Applied Linguistics* 46(6):821–840, doi:10.1093/applin/amae040 | CC BY (accepted version read: abstract and intro) | Speakers in engaged conversation re-use parts of each other's words ("resonance"). That's a good general instruction for the writer. Their class finding (more resonance among upper-class speakers in 2014) shouldn't become a persona rule. |

### Regional differences in words and grammar (MacKenzie, Bailey & Turton 2022)

| Feature | Where (and how much) | Page |
|---|---|---|
| *barm* (bread roll) | "confined to the North West", from Manchester west to Liverpool and north into western Lancashire | p. 55 |
| *cob* | "largely concentrated in the Midlands around Nottinghamshire" | p. 56 |
| *bap* | most concentrated in Staffordshire, the West Midlands and North Wales | p. 56 |
| *batch* | Liverpool and Coventry | p. 56 |
| *lolly ice* (for *ice lolly*) | the Liverpool area | p. 57 |
| *tea* vs *dinner* (evening meal) | The North West, North East and Yorkshire are "tea strongholds" (67–69%), and the South East says *dinner* (84%; London 95%). Some people say both, depending on the meal. | p. 58 |
| *yous(e)* (you, plural) | North East: Newcastle 51%, Teesside 44%. Liverpool 34%, Manchester 25%. Scotland 33%. Low in York, Leeds and Sheffield. | p. 58 |
| *give it me* (dative) | A North West and West Midlands form, accepted in Sheffield 87%, York 56% and Newcastle 25%. The boundary lies just south of Birmingham, Coventry and Leicester. | p. 59 |
| *you was* (was-levelling) | accepted by 52% of 11,846 responses; historically Northern and spreading south as far as London | p. 60 |

## Gaps
- **Backchannels** (*yeah*, *mm*, *right*): no open, peer-reviewed British source found in this pass. Torgersen & Gabrielatos cover *yeah*, *right* and *okay* as tags in London only.
- **Scotland, Wales and Northern Ireland:** beyond the MacKenzie survey points above, I found nothing open in this pass.
- **Not fetched:** the BBC Voices maps (in Grieve et al.'s supplement), and the English Dialects App paper (Leemann, Kolly & Britain, *Ampersand* 2018, CC BY-NC-ND), whose download was blocked.

## Bib (`twin_public/sources.bib`, from Crossref; three year quirks to fix by hand)
- Love et al.: Crossref gives 2022 and no volume. The article itself says "International Journal of Corpus Linguistics 22:3 (2017), 319–344" (PDF p. 1). **Use 2017, 22(3).**
- Lee & Ashton (HEXACO-100): Crossref says 2016 (online first); the volume is 25(5). Check the print year before citing.
- García et al. and Tantucci & Wang: Crossref gives the online years (2021, 2024); the print volumes are 90(2) and 46(6).
