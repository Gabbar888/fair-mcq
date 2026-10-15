# Authoring and option construction

## Why these rules exist

An earlier paper was scored 80/90 by a candidate who, for the questions he was unsure of, simply picked the longest option. That heuristic worked on **90%** of the paper. The correct answers averaged 27% more words than the distractors, because each one was written as a complete thought — verdict, then mechanism, then qualification — while the distractors were bare assertions. Two smaller tells compounded it: calibrated hedging appeared only in correct answers, and absolute words clustered in distractors.

None of that is about the concepts. It is entirely about how the options were written. Hence the rules below.

---

## Authoring

**Two-pass authoring.** Pass 1: write questions aligned to the topic. Pass 2: rewrite each one specifically to make it airtight against reading strategy, and to raise difficulty. The second pass is not optional — most of the work happens there.

**Never state the defect in the stem.** Do not write "`model.eval()` is not called" or "the scaler was fitted before the split". Embed the target line inside a longer realistic code block — setup, configuration, training, then the serving path — and ask the question against that fuller context. This rule does double duty: it makes easy questions hard, and it produces the long code-bearing stems the paper needs. It applies wherever a question involves code. In topics with no code expectation (see `subtopics.md`), do the equivalent without code: embed the key fact inside a longer realistic scenario or data table rather than forcing a code block.

**Debugging questions state only that a runtime error occurred.** Then ask what the error is and which specific line is responsible. Never name the error in the stem.

**Multi-concept stems.** Real exam questions frequently test more than one concept from more than one subtopic. Combine related subtopics rather than keeping every question single-concept.

**No overly direct questions.** Nothing answerable with little domain knowledge by reading and general reasoning alone.

**No questions built on niche library behaviour.** Don't write a question whose answer depends on a specific library quirk the candidate either knows or doesn't, such as how a scikit-learn `RandomForestClassifier` combines trees (it averages probabilities rather than majority voting) or how `TimeSeriesSplit(gap=5)` shrinks the training folds. Standard behaviour the topic is known for is fine, and code-heavy stems are unaffected.

**Claim-style phrasing for statistics and evaluation metrics.** Embed the concept in a tricky descriptive claim the candidate must unpack — "the analyst queue is dominated by noise despite excellent specificity" — rather than asking for a formula. Use this regularly for specificity/sensitivity, precision/recall and similar.

---

## Option construction

**Length parity.** All four options within ±4 words. Mean length of correct options equals mean length of distractors within ±1 word across the set. An option noticeably longer than the others gets picked as the safe answer on that cue alone.

**Strip the correct answer to its claim.** Delete the mechanism clause. The question asks *which*, not *why*. The why goes in the explanation.

**Put the reasoning in the distractors.** Wrong options should carry a correct-sounding, relevant mechanism that leads somewhere false. A distractor that explains itself is far harder to dismiss than a bare assertion. Distractors must be believable to someone who does *not* know the concept.

**Align options with what the question asks**, not with what is true in general. A statement can be perfectly true and still not answer the question — that makes an excellent distractor.

**Explanation-style options all comparable in depth.** Never rely on one clearly longer, correct-sounding option. Make at least three of the four similarly detailed.

**Distribute hedges.** *Typically, roughly, often, may* must appear in distractors at a similar rate to correct answers. Calibrated language in the correct answer alone is a free signal.

**Distribute absolutes.** *Always, never, only, cannot, every* must not cluster in distractors. Include them in correct answers where genuinely true — "gradients are *exactly* zero".

**Paired numeric options.** Where possible give two options with the same values transposed — precision 0.60/recall 0.75 against precision 0.75/recall 0.60 — so the question is decidable only by computing.

---

## Worked example

The concept is trivial: which models need feature standardization. The first version is gameable, the second is not.

**Before** — correct answer 36 words against a 20-word distractor mean:

> D. *k-nearest neighbours and ridge regression require it — one because it computes distances, the other because the penalty acts on coefficient magnitudes — while the two tree ensembles are invariant to monotone rescaling of individual features.*

**After:**

> A. All four, since each minimizes a loss whose curvature depends on the relative scale of its inputs.
>
> B. k-NN and both tree ensembles, since all three compare values across columns when choosing split points.
>
> C. k-NN and ridge regression only.
>
> D. Ridge regression only, since the L2 penalty is the sole component that reads coefficient magnitude.

The correct answer is now **C — the shortest option, with no justification at all.** Every wrong option carries a mechanism that sounds like it was written by someone who knew something. To answer it you have to actually know that trees are invariant to monotone rescaling and that distance metrics and coefficient penalties are not.

---

## Numerical verification

Compute every numerical answer in code before writing the key, not after. Writing keys from intent rather than arithmetic once shipped three wrong answers in a single paper.

Also confirm **no second option is correct**. Two real failures: a kernel-size question where two values both produced the stated flatten length, and a token-budget question where two configurations both exceeded the limit.
