---
name: "ai-exam-builder"
description: Generate MCQ practice questions for a 90-question AI assessment, either as a single topic set or as a full 90-question mock exam built topic by topic. Use this whenever the user asks for exam questions by topic number ("Topic 3, 30 questions", "give me topic 6"), asks to build or continue a combined mock exam, asks to assemble finished topic sets into a paper, or asks for practice questions on the assessment's domains (statistics, prompts and agents, RAI governance, PyTorch/TensorFlow, ML fundamentals, deep learning, RAI technical, NLP, computer vision, LLMs, RAG). Also use it when the user asks to audit or check an existing question set for gameable options, length tells, or reading load. Always use this skill for any request touching this exam, even when phrased casually or without the word "topic".
---

# Exam Builder

Builds MCQ practice material for the AI assessment: 90 questions, 150 minutes, 11 domains, roughly 40% code-based (about 35 of 90 questions, none in Topics 1, 3 and 7).

The whole point of this skill is that **the questions must not be answerable by reading strategy alone**. A previous paper was scored 80/90 by a candidate who simply picked the longest option — that heuristic worked on 90% of questions. Everything below exists to prevent that recurring.

## Two modes

Decide which mode applies before doing anything else.

**Topic mode** — the user wants questions on one domain, for practice. Triggered by "Topic 3", "Topic 3, 30 questions", "give me deep learning questions". Produces the requested count. No reading-load quota, no whole-paper audit, no state.

**Combined mode** — the user is building a full 90-question mock, one topic per turn. Triggered by "make me a combined exam, start with topic 1", "continue with topic 4", "assemble the mock". Produces the block size from the domain table (8 or 9), enforces a share of the reading-load quota, tracks which blocks are done, and assembles plus audits at the end.

**If the request does not clearly indicate which, ask before doing anything else.** Do not guess. Ask only when genuinely ambiguous — if the user says "Topic 7" and a combined build is already underway in this conversation, that is a continuation, not an ambiguity.

## What differs between modes

| | Topic mode | Combined mode |
|---|---|---|
| Question count | as requested | 8, or 9 for Statistics and Deep Learning |
| Reading-load quota | not enforced | per-block share enforced |
| Block files on disk | not needed | required, one per block |
| Cross-block duplicate check | not needed | required |
| Whole-paper audit | not run | run at assembly |

Everything else is identical. Both modes obey the format contract, the authoring rules and the option rules in full.

## Procedure — topic mode

1. Read `references/subtopics.md` for the requested topic's coverage surface.
2. Read `references/confirmed-topics.md` and pull the entries belonging to that topic. These are question types confirmed to appear in the real exam; include a substantial share.
3. Read `references/authoring.md` and apply every rule while writing. For Topic 2, also read `references/prompt-ranking.md`.
4. Write the set to a file using the format in `references/format.md`.
5. Compute every numerical answer in code and confirm no second option is also correct.
6. Run `scripts/validate_set.py <file>` and **paste its raw output into the reply**.
7. Fix anything it flags, re-run, then deliver.

## Procedure — combined mode

Same steps, plus:

- Before writing, list existing `block*.md` files in the working directory and read their questions, so the new block repeats nothing.
- Write to `block<NN>_<domain>.md` so progress survives in files rather than in conversation memory.
- Carry the block's share of reading load: the code-question count for the topic comes from the code-expectation table in `references/subtopics.md`, plus about 2 questions containing a data table per block.
- When the user asks to assemble, read `references/assembly.md` and follow it.

## Non-negotiables

**Show the validator output.** If the script output is not in the reply, the set has not been delivered. This is the only hard check in the system; everything else is a judgement call that can quietly slip.

**Never state the defect in the stem.** Do not write "`model.eval()` is not called" or "the scaler was fitted before the split". Embed the target line inside a longer realistic code block and let the candidate find it. This single rule is what turns an easy question into a properly difficult one, and, where code is involved, it is also what produces the code-heavy stems the paper needs. In topics with no code expectation, apply it with a scenario or table instead of forcing code.

**Strip the correct answer to its claim; put the reasoning in the distractors.** The question asks *which*, not *why*. A correct option that explains itself is how the length tell appears in the first place.

**Verify numericals in code before writing the key.** Three wrong keys shipped in one early paper because answers were written from intent rather than arithmetic. Also confirm no second option is correct — a kernel-size question once admitted two valid answers.

## Reference files

Read these as needed rather than all at once.

- `references/format.md` — output format. Parser-critical; read before writing any set.
- `references/subtopics.md` — the 11 domains, their subtopics, and the topic-number to block mapping.
- `references/authoring.md` — the authoring and option-construction rules. Read before writing any set.
- `references/prompt-ranking.md` — rules for the prompt-ranking questions. Topic 2 only.
- `references/confirmed-topics.md` — question types confirmed seen in the real exam, organized by topic.
- `references/assembly.md` — ordering, scatter, audit thresholds, letter rebalancing. Combined mode, at assembly only.

## Scripts

Run with `python scripts/<name>.py <file>`. Each reads the answer key out of the file itself.

- `validate_set.py` — format, option parity, hedge and absolute distribution, key and explanation agreement. Run on every set.
- `audit.py` — full adversarial thresholds. Assembly only.
- `load_check.py` — reading-load quotas. Assembly only, or on a block to check its share.
- `rebalance.py` — seeded letter rebalancing. Assembly only; rewrites the file in place.

## Difficulty

Easy to Medium. Difficulty comes from option design, reading load and time pressure — not from reaching for obscure topics. A well-prepared candidate should need 130–140 minutes for a full paper. If it can be finished in under 120, the reading load is too light.
