# Field count: the rule, written before any paper was collected
*Lit room, Oct 9 2026, 18:37 EDT (stamp). Nothing below may change after collection starts without a dated note saying what changed and why.*

## The question
How does recent NLP work evaluate an AI system's memory of conversations with people? Specifically, how many papers score that memory against **what a particular person themselves reports remembering** from the conversation, as opposed to recall of stated facts, QA over past sessions, judged response quality, or retrieval of annotator-chosen items? Our point would be that almost none use the first-party measure. If the count isn't clean, we say so and don't use it.

## Population
1. **Source:** the ACL Anthology's full BibTeX export with abstracts (`anthology+abstracts.bib.gz`, downloaded once; its download date is recorded in FIELD_COUNT.md). That's every venue the Anthology hosts: ACL, EMNLP, NAACL, EACL, AACL, COLING, LREC(-COLING), TACL, CL, Findings, SIGDIAL and workshops.
2. **Years:** 2024, 2025, 2026 (the `year` field).
3. **Keyword screen** (case-insensitive, over title + abstract). A paper passes if it matches either:
   - a *memory phrase*: `long-term memory`, `long term memory`, `conversational memory`, `dialogue memory`, `dialog memory`, `episodic memory`, `personal memory`, `user memory`, `memory-augmented`, `memory augmented`, `memory bank`, `memory module`, `memory mechanism`, `memory management`, `memory system`; **and** a *conversation word*: `dialog`, `dialogue`, `conversation`, `chat`, `chatbot`, `assistant`, `companion`, `session`;
   - or a *long-conversation phrase* on its own: `long-term conversation`, `long-term dialogue`, `long-term dialog`, `multi-session`, `multi session`, `long-horizon conversation`, `lifelong dialogue`.
4. **Inclusion (judged from the abstract, then the paper if the abstract is unclear):** the paper proposes, trains or evaluates **an AI system's memory of earlier conversation with a user**, across sessions or within a long conversation (a memory method, a memory-equipped agent, or a benchmark/evaluation of such memory). Personalization from dialogue history counts if the system must use what the user said earlier.
5. **Exclusion:**
   - parametric memorization or training-data leakage;
   - KV-cache, context compression or efficiency work not evaluated on remembering conversation content;
   - agent memory for tasks, tools or the web, not conversation with a person;
   - surveys and position papers (counted separately and listed, not coded);
   - studies of human memory with no AI system (these go to Related Work).

## Coding: how is memory scored? (multi-label; every included paper gets at least one)
- **A. Fact QA / recall**: questions about facts stated in earlier sessions (e.g., single-hop, multi-hop, temporal QA), scored against gold answers.
- **B. Response quality**: responses judged (by people or an LLM) for consistency, engagingness, personalization or use of memory.
- **C. Retrieval / memory-content match**: retrieved or stored memory items compared with gold items or summaries chosen by annotators, an LLM or the dataset creators.
- **D. Downstream task or user outcome**: task success, user satisfaction or ratings of the system, with no check of what was remembered.
- **E. First-party memory**: the system's memory (or its prediction of what a person will remember) compared against **what a human participant in the conversation themselves reported remembering, finding memorable, or wanting kept**.
- **F. Other**: described in a note.

Gold produced by crowd workers or an LLM, by reading a transcript, is **not** E. E requires the conversation's own participant as the source.

## What gets reported
Every included paper with its venue, year, codes, and the page or section where the evaluation is described. Plus the count of E, and the counts of A–D for context. Plus a list of borderline decisions with reasons. Then one candidate opening sentence. If E > 0, every E paper gets named and credited in Related Work.

## Honesty limits stated in advance
- The keyword screen will miss papers that never use these phrases. Arguing that the screen is complete is not the claim. The claim is a count within a stated, reproducible population.
- arXiv-only papers are out of the main count (no reproducible population). A separate, smaller arXiv keyword search may be listed as a sanity check, labeled as such.
- Coding is done by me, from abstracts and evaluation sections. A random 20% of included papers gets a second check against the full evaluation section before the count is used.
