# Prompt-ranking questions

Topic 2 carries these. The real exam had 6–7 of them, so a full paper should contain that many; a Topic 2 set should contain a proportionate share.

## Shape

Five candidate prompts labelled P1–P5, and a stated objective. The four options are **complete orderings of all five prompts**, most to least effective against that objective.

## Rules

**All five prompts lean the same direction.** None is obviously bad. The quality gap must be subtle, sometimes barely articulable — that is the point of the format.

**Vary which label wins.** Not always P2. Across six sets the winners should land on different labels.

**Vary the governing principle.** This is the rule most easily broken. If "the most detailed prompt wins" holds in every set, the candidate stops reading the prompts and answers from the pattern — which happened, and was reported as the sets feeling uniform. Rotate among:

- explicit rule beats illustrative example — for semantics and edge cases
- example beats every verbal description — for exact surface format, where punctuation and spacing cannot be described completely in prose
- persona wins outright — when the objective *is* register or voice
- the most precisely specified prompt ranks **last** — precision aimed at the wrong target, e.g. a hard word cap that forces dropping the content the objective requires
- the detailed prompt carries a subtly harmful instruction — e.g. a confidence filter that suppresses exactly the recall being optimized
- negative constraints backfire; positive reframing generalizes better
- chain-of-thought wins and terse-output ranks last — multi-step arithmetic

**Vary the objective** across sets: security (SQL injection risk), correctness, efficiency, clarity, format adherence, faithfulness, recall, latency and token cost, consistency on ambiguous input.

**Place the discrimination at positions 3–4** of the ranking, and let two options share their first two entries. The answer must not be readable off the top of the list.

## Worked examples of the less obvious principles

**Precision aimed at the wrong target.** Objective: preserve every condition, exception and deadline when summarizing a contractual obligation. A prompt saying "summarize in exactly 10 words" is the most precisely specified of the five and ranks **last**, because the cap guarantees the required content gets dropped.

**Persona wins.** Objective: consistency with a character's voice in period drama dialogue. A prompt specifying "Lady Ashworth, 62, aristocrat, 1924 Yorkshire, speaks formally, never uses contractions" beats every rule-based alternative, because the objective is register and that is what persona specification is for.

**The example wins.** Objective: reproduce a house citation format exactly. Whether there is a space before §, whether the page range uses an en-dash or hyphen — these cannot be stated completely in prose but are trivial to show. The worked example beats the field-order description.

**Harmful detail.** Objective: maximize recall of security issues, with human triage downstream. A prompt adding "report only issues you are confident are exploitable in production" is the most specific and ranks **last**, because the confidence filter suppresses the recall the task optimizes for.

**Negative constraint backfires.** Objective: avoid naming any competitor. "Do not mention Zenith, Orbital or Kestrel" covers only three, leaves every other competitor unaddressed, and writes those names into context where they can be echoed. "Answer using only information about our own products; if asked to compare, say you can only speak to our own range" generalizes and supplies a substitute behaviour.
