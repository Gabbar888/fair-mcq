# Output format

Machine-parsed. Do not alter the markers.

## Header

```
# <Exam title>
## <N> questions | <M> minutes | MCQ, single correct

<optional instructions paragraph>

---
```

## Questions

```
**Q1.** <question text: Markdown allowed, including tables and code blocks>

A. <option text>

B. <option text>

C. <option text>

D. <option text>

---
```

Rules:
- Each question starts with `**Q<number>.**` at the very start of a line.
- Exactly 4 options, each starting at the start of a line with `A. `, `B. `, `C. `, `D. `. Blank line between options.
- Options come after all question text. Nothing follows D except `---`.
- If an option is a code block, put the letter alone on its line (`A.`) with the code block on the following lines.
- Outside code blocks, no line of question text may start with `A.`, `B.`, `C.` or `D.`
- Do not label domains inside the questions.

## Answer key

```
# ANSWER KEY

| Q | Ans | Domain | Q | Ans | Domain | Q | Ans | Domain |
|---|-----|--------|---|-----|--------|---|-----|--------|
| 1 | A | Stats | 31 | B | ML | 61 | C | NLP |
```

For a single topic set the two-column form is fine:

```
# ANSWER KEY

| Q | Ans | Domain |
|---|-----|--------|
| 1 | A | Deep Learning |
```

## Explanations

```
# EXPLANATIONS

**Q1 — A.** <explanation>

**Q2 — C.** <explanation>
```

- Entry format is `**Q<number> — <letter>.**` with an em dash and spaces either side. The letter must match the key.
- One entry per question, in order.
- A line containing only `---` after the last explanation.

Explanations say why the correct option is right AND why each tempting distractor is wrong. This is where the reasoning stripped out of the options goes — it belongs here, not in the option text.

## Errata

Only if a key entry turns out to be wrong:

```
# ERRATA

- **Q69 → D** (not A). <one-line reason>
```

Format is `**Q<number> → <letter>**`. It overrides the answer key.

Anything after that (for example `# SCORING AND DIAGNOSIS`) is ignored by the parser. Section headings use a single `# `.
