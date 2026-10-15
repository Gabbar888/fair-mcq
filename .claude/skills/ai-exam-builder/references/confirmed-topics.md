# Question types confirmed seen in the real exam

Organized by topic number so a request for one topic only needs its own section. Not every item must appear in a given set, but pull a substantial share of whatever belongs to the requested topic.

Within a topic, draw on important concepts whether or not they are confirmed-seen. Coverage is not limited to this list.

---

## Topic 1 — Statistics

- **Two-way ANOVA: F-statistic with interaction effect.** Both forms appeared — a pre-filled SS/df/MS/F table, and a raw experimental data table (crops and fertilizers, yield per configuration) to compute from scratch. Generate both.
- **VIF + multicollinearity.** Confirmed; generate the two together as paired concepts, never separately.
- **PCA.**
- **R² and variance.**
- **Hypothesis testing with t-tests.**
- **Sensitivity and specificity in claim-style phrasing.** The candidate reported being unable to answer these, and specifically struggles with the claim framing rather than the formulas. Use it often.
- **Odds ratio combined with Bayes' theorem** in a single problem.
- **Sequential Bayesian updating** — compute a posterior, then use that posterior as the prior for a second step.
- **Two drugs, two groups each.** Accuracy or impact per group; which drug had the greater overall effect. Requires correct aggregation across grouped data rather than reading a single figure (Simpson's paradox).
- **Logistic regression.**

## Topic 2 — Prompts, Agents, RAG

- **Prompt ranking sets.** 6–7 in the real paper. See `prompt-ranking.md`.
- **Scenario → choose the appropriate ML algorithm** also appeared; it can sit here or in Topic 5.

## Topic 3 — RAI Governance

- **Governance / feedback loop for deployed local models.** A company deploying local models broadly across its organization asks how to collect feedback and manage them post-deployment. The correct answer emphasizes a continuous, structured feedback collection and response process gathered through the model's users, as opposed to one-off or purely technical solutions.

## Topic 4 — PyTorch / TensorFlow

- **PyTorch code debugging.**
- **Debugging — multiple bugs, identify the blocking bug.** Code with several bugs; which single one prevents it running at all, as opposed to bugs that are real but non-blocking. Some actual bugs appear among the options and **other real bugs are deliberately absent from the option list**, so the candidate must find all bugs and reason about severity rather than pattern-match. Reported as one of the harder types encountered. Include at least one per paper.

## Topic 5 — ML Fundamentals

- **Random forest** — numerical or calculation-style.
- **Churn.** Used repetitively across the real exam. The candidate reported not understanding what the churn questions were even asking, so give enough setup that the task is unambiguous — what churn means in context, what is being predicted, over what window. Generate at multiple difficulty levels.
- **Cross-validation, broadly** — stratified k-fold, leave-one-out, train/val/test splitting pitfalls, data leakage in CV.
- **Scenario → choose the appropriate ML algorithm.** A business or data situation described; select the fitting model (k-NN, k-means and similar).

## Topic 6 — Deep Learning

- **LSTM/GRU internal gate order — fill-in-the-blank pseudocode.** The sequence of gate computations given in code or pseudocode with one line missing; identify the missing line. Tests the correct order and form of each gate's computation, distinct from a find-the-bug question. **At least 4–5 questions in this style appeared** — this is a high-frequency format, not a one-off.
- **LSTM / GRU / RNN / batch normalization / dropout — depth.** Self-identified weak area; generate detailed practice beyond the gate-order format.
- **CNN output dimension and flattening.** Full forward-dimension tracing through conv and pool layers into the flatten step, and the resulting input size to a fully connected layer. Appeared multiple times.
- **Weight initialization** — zero-centered, fan-in, fan-out, geometric and average strategies, conceptual and numerical.
- **Spatial dimension calculation.**

## Topic 7 — RAI Technical

No specific items recalled. Build from the subtopic list: reliability, robustness, explainability, understanding model behaviour.

## Topic 8 — NLP

- **NER.**
- **TF-IDF.**
- **Naive Bayes.**
- **Trigram document similarity** across three documents A, B and C.
- **Similarity** measures generally.

## Topic 9 — Computer Vision

- **OpenCV in genuine code form** — not definitional questions about enumerators.
- **Canny edge detection, conceptual.** How changing a parameter or threshold affects which edges appear, fade or persist. Options along the lines of new edges becoming visible, old edges fading, weaker edges remaining.
- **Vision Transformer calculations** — patch count, sequence length and similar numericals.
- **AlexNet** — architecture specifics, historical significance, key design choices.
- **Computer vision broadly.** Many CV questions appeared beyond those specifically recalled, so cover the subtopic list widely.

## Topic 10 — LLMs

- **Multi-head attention → multi-query attention** — what happens if you change MHA to MQA, effects and trade-offs.

## Topic 11 — RAG

- **Transformer + RAG token budgeting.** Total context budget → subtract reserved and cache tokens → sequentially add chunk token sizes in the given order → how many whole chunks fit before exceeding the remaining budget. Chunks are supplied in order and may not be skipped.
- **Reciprocal Rank Fusion (RRF)** in reranking.
- **Precision@k and recall@k.**
- **Document ranking numericals.**
- **Context-window management.**

---

## Not seen — deprioritize

- GPU idle-time diagnosis
- BatchNorm in transfer learning
