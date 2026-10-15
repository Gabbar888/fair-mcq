# Assembling blocks into a mock

Read this only when the user asks to assemble a combined exam. It assumes eleven block files exist.

## Steps

1. **Collect** the eleven `block*.md` files in block order (Topic 1 first, Topic 11 last).
2. **Renumber** questions 1–90 continuously.
3. **Apply the scatter rule** — displace about 10 questions from their home block into an adjacent one, so the block boundary is not exactly where the count predicts. Do not scatter so heavily that the paper reads as interleaved; the blocked trend must stay obvious.
4. **Merge the answer keys** into one three-column table, with the domain tag per question.
5. **Merge the explanations** in question order, renumbered to match.
6. **Run `scripts/load_check.py`** on the assembled file. Fix shortfalls by lengthening stems or adding code, not by adding questions.
7. **Run `scripts/audit.py`**. Fix what fails. Re-run. Repeat until all thresholds pass — budget three passes.
8. **Run `scripts/rebalance.py`** to distribute correct-answer letters. It rewrites the file in place and prints the new key.
9. **Run `scripts/validate_set.py`** one final time on the assembled paper to confirm keys and explanations still agree after rebalancing.
10. **Paste the output of steps 6–9 into the reply.**

## Reading-load quotas

| Metric | Minimum |
|---|---|
| Questions containing a code block | 35 |
| Total lines of code in the paper | 165 |
| Questions containing a data table | 17 |
| Mean stem length (words) | 47 |
| Median stem length (words) | 38 |
| Stems over 80 words | 17 |
| Stems over 120 words | 7 |
| Total paper word count | 11,000 |

A paper meeting the option rules but failing these is not acceptable. This is the quota most often missed — it was missed once already, and the resulting paper was finished in 100 of 150 minutes.

Code is not expected in every block: Topics 1, 3 and 7 carry none, and Topics 2 and 11 about two each (see `subtopics.md`). Those blocks carry their reading load through longer scenario stems and data tables instead. Spread the heavy questions across all eleven blocks rather than concentrating them. Because domains are blocked, concentrating load in one block makes that domain a time sink and breaks the pacing checkpoints.

## Audit thresholds

| Check | Threshold |
|---|---|
| Correct answer uniquely longest | 20–30% (chance is 25%) |
| Correct answer uniquely shortest | 20–30% (chance is 25%) |
| Mean word gap, correct vs distractors | within ±1.0 |
| Mean within-question spread | ≤ 3 words |
| Questions with spread > 5 words | ≤ 3 |
| Correct is the only hedged option | 0 |
| Eliminating options with absolutes isolates the correct answer | 0 |
| Answer-letter distribution | 22–23 each, no run of 4 identical |
| Questions that hinge on niche library behaviour (manual screen: could someone who knows the concept but not this library quirk still answer it?) | ≤ 5; replace any extras with a reasoning-based question on the same subtopic, without repeating other questions, then re-audit |

Both directions matter. Uniquely-shortest at 4% is as exploitable as uniquely-longest at 90% — it licenses "never pick the shortest". Drive both toward chance, not toward zero.

## Letter rebalancing

Never assign letters by a formula. A formulaic assignment once produced a repeating A-B-C cycle with no D at all, which is worse than the problem it was fixing. `rebalance.py` uses a seeded shuffle of a balanced multiset, rejects any draw with four identical letters in a row, and rotates each question's options so the correct answer lands on its target letter. Code-block options are rotated intact.

## Final checklist

- [ ] 90 questions, 360 options, 90 explanations, key has 90 entries
- [ ] Every explanation letter matches the key
- [ ] Every numerical computed in code and matched to the key
- [ ] Answer uniqueness confirmed — no second correct option
- [ ] Reading-load quotas met
- [ ] Audit thresholds pass after three passes
- [ ] Letter distribution 22–23 each, no run of four
- [ ] Domains blocked in syllabus order, ~10 questions scattered
- [ ] Heavy-load questions spread across all eleven blocks
- [ ] 6–7 prompt-ranking sets with varied winning labels and varied principles
- [ ] At least one multi-bug question with a real bug absent from the options
- [ ] No question recycles an earlier paper's question with changed values
- [ ] Niche-library-behaviour questions within the limit; extras replaced
