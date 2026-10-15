# General principles for writing MCQ mock exams

Follow every rule below when writing multiple-choice questions with four options (A–D) and exactly one correct answer. The goal: no question can be answered by reading strategy alone, only by knowing the subject.

---

## 1. Questions

- Write each set in two passes. Pass 1: write questions on the topic. Pass 2: rewrite every question to defeat reading strategy and raise difficulty.
- Never state the key fact in the stem. Embed it in a longer, realistic scenario, data table or code block, and let the candidate find it.
- In debugging questions, say only that something failed. Ask what went wrong and where. Never name the error.
- Combine two related subtopics in one question where natural.
- No question may be answerable by general reasoning or careful reading without subject knowledge.
- No question may depend on niche trivia, such as one library's default setting or an obscure quirk. Test standard, well-known behaviour.
- Use claim-style phrasing: give a descriptive claim the candidate must unpack instead of asking for a formula or definition.

## 2. Options

- Keep all four options within ±4 words of each other.
- Across a set, keep the average length of correct options equal to that of wrong options, within ±1 word.
- Strip the correct answer to its claim. It states *which*, not *why*.
- Give each wrong option a correct-sounding, relevant mechanism that leads to a false conclusion. It must be believable to someone who does not know the concept.
- Use statements that are true in general but do not answer the question as distractors.
- When options are explanation-style, make at least three of the four equally detailed.
- Use hedging words (*typically, roughly, often, may*) in wrong options as often as in correct ones.
- Do not cluster absolute words (*always, never, only, cannot, every*) in wrong options. Use them in correct answers where they are true.
- For numerical answers, include two options with the same values swapped, so only computing decides.

### Example

Question:

> A team trains four models on the same tabular dataset, where `income` ranges from 20,000 to 2,000,000 and `age` from 18 to 90: k-nearest neighbours, ridge regression, a random forest and gradient-boosted trees. Before training, they consider standardizing every feature to zero mean and unit variance.
>
> For which models will standardization change the predictions?

Not acceptable (correct answer explains itself and is longest):

> D. k-nearest neighbours and ridge regression require it — one because it computes distances, the other because the penalty acts on coefficient magnitudes — while the two tree ensembles are invariant to monotone rescaling of individual features.

Acceptable (correct answer is C: shortest, no justification; every wrong option carries a plausible mechanism):

> A. All four, since each minimizes a loss whose curvature depends on the relative scale of its inputs.
>
> B. k-NN and both tree ensembles, since all three compare values across columns when choosing split points.
>
> C. k-NN and ridge regression only.
>
> D. Ridge regression only, since the L2 penalty is the sole component that reads coefficient magnitude.

## 3. Correctness

- Compute every numerical answer in code before writing the answer key.
- Check every question for a second correct option. Rewrite until exactly one is correct.
- In each explanation, state why the correct option is right and why each tempting wrong option is wrong.

## 4. Whole paper

- Balance answer letters: about N/4 per letter, no run of four identical letters. Assign them with a seeded random shuffle, never a pattern or formula.
- Keep concepts easy to medium. Create difficulty through reading load, option design and time pressure, not obscure topics.
- Size the reading load so a well-prepared candidate needs about 87–93% of the time limit.
- Carry reading load with long stems, data tables and code blocks. Where code is unnatural for the subject, use longer scenarios, tables and output snippets instead.
- Spread long and heavy questions across all topics. Do not concentrate them in one topic.
- Do not repeat questions, including earlier questions reused with changed numbers.
- Cover each topic's full width. For sets over ten questions, write in batches of ten that each span the whole topic; do not order easy-first then hard.
- Do not label topics inside the questions.
