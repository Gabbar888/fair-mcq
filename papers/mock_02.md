# AI Assessment — Mock 02
## 90 questions | 150 minutes | MCQ, single correct

Domains are interleaved and unlabelled. Target pace is 100 seconds per question; several items are deliberately expensive, so flag and return rather than stalling.

---

**Q1.** An experiment compares 4 training methods with 6 independent runs each. An analyst reports the following table:

| Source | SS | df | MS | F |
|---|---|---|---|---|
| Between methods | 144 | 3 | 48.0 | 12.0 |
| Within groups | 96 | 24 | 4.0 | |
| Total | 240 | 27 | | |

Which claim is supported?

A. The table is correct, and since F exceeds any conventional critical value the four methods differ significantly in mean performance.

B. The table is internally inconsistent: the between-methods degrees of freedom must be 4 because four methods were compared in this experiment.

C. The table is correct, but the F ratio cannot be interpreted without also reporting the pooled standard deviation of the six runs.

D. The table is internally inconsistent: with 24 observations the within-groups degrees of freedom must be 20 and the total 23, so the reported F is computed against the wrong error term.

---

**Q2.** The script below terminates with a runtime error on the first batch.

```python
 1   class Reviews(Dataset):
 2       def __init__(self, texts, labels, vocab, maxlen=64):
 3           self.texts, self.labels = texts, labels
 4           self.vocab, self.maxlen = vocab, maxlen
 5       def __len__(self):
 6           return len(self.texts)
 7       def __getitem__(self, i):
 8           ids = [self.vocab.get(w, 0) for w in self.texts[i].split()]
 9           ids = ids[:self.maxlen]
10           return torch.tensor(ids), self.labels[i]
11
12   ds = Reviews(texts, labels, vocab)
13   dl = DataLoader(ds, batch_size=16, shuffle=True)
14
15   emb  = nn.Embedding(len(vocab), 100)
16   lstm = nn.LSTM(100, 128, batch_first=True)
17   head = nn.Linear(128, 2)
18   opt  = optim.Adam(list(emb.parameters()) + list(lstm.parameters()), lr=1e-3)
19   crit = nn.CrossEntropyLoss()
20
21   for xb, yb in dl:
22       out, _ = lstm(emb(xb))
23       loss = crit(head(out[:, -1, :]), yb)
24       loss.backward()
25       opt.step()
```

Which defect is responsible for the failure?

A. The optimizer omits the head's parameters, so the backward pass finds no gradient path to the classification layer.

B. Gradients are never cleared between iterations, so the accumulated gradient overflows on the first batch boundary.

C. Reviews of differing word counts produce variable-length tensors, which the default collate function cannot stack into a batch.

D. The embedding layer is indexed with token ids that may equal the vocabulary size, placing them outside the embedding table.

---

**Q3.** A TF-IDF matrix is computed over five documents. Four of its columns are shown, with rows as documents:

| | t1 | t2 | t3 | t4 |
|---|---|---|---|---|
| D1 | 0.00 | 0.40 | 0.00 | 0.22 |
| D2 | 0.00 | 0.00 | 0.51 | 0.22 |
| D3 | 0.00 | 0.40 | 0.00 | 0.00 |
| D4 | 0.00 | 0.00 | 0.00 | 0.22 |
| D5 | 0.00 | 0.00 | 0.51 | 0.00 |

Which claim about term t1 is supported?

A. Term t1 was removed as a stop word during preprocessing, which is the only mechanism producing a column of exact zeros.

B. Term t1 occurs in every one of the five documents, so its inverse document frequency is zero and its weight vanishes everywhere.

C. Term t1 occurs in none of the five documents and was retained in the vocabulary from a separate training corpus.

D. Term t1 occurs exactly once in the corpus, so its term frequency is too small to register at two decimal places.

---

**Q4.** A fairness audit of a lending model reports:

| Group | Selection rate | True positive rate | False positive rate |
|---|---|---|---|
| A | 0.42 | 0.78 | 0.19 |
| B | 0.41 | 0.55 | 0.31 |

Which claim is supported?

A. The model satisfies demographic parity but violates equalized odds, since selection rates are nearly equal while both error rates diverge substantially.

B. The model violates both criteria, since equal selection rates cannot coexist with unequal error rates under any fairness definition.

C. The model satisfies equalized odds but violates demographic parity, since the selection rates differ while the error rates align closely.

D. The model satisfies both criteria, since the selection rates are within one percentage point and no disparity threshold is breached.

---

**Q5.** Which implementation correctly computes one LSTM cell step?

A.
```python
z = torch.cat([h_prev, x_t], dim=1)
f = torch.sigmoid(z @ Wf + bf)
i = torch.sigmoid(z @ Wi + bi)
g = torch.tanh(z @ Wg + bg)
o = torch.sigmoid(z @ Wo + bo)
c = f * c_prev * i * g
h = torch.tanh(o * c)
```

B.
```python
z = torch.cat([h_prev, x_t], dim=1)
f = torch.tanh(z @ Wf + bf)
i = torch.tanh(z @ Wi + bi)
g = torch.sigmoid(z @ Wg + bg)
o = torch.tanh(z @ Wo + bo)
c = f * c_prev + i * g
h = o * torch.sigmoid(c)
```

C.
```python
z = torch.cat([h_prev, x_t], dim=1)
f = torch.sigmoid(z @ Wf + bf)
i = torch.sigmoid(z @ Wi + bi)
g = torch.tanh(z @ Wg + bg)
o = torch.sigmoid(z @ Wo + bo)
c = f * c_prev + i * g
h = torch.tanh(o) * c
```

D.
```python
z = torch.cat([h_prev, x_t], dim=1)
f = torch.sigmoid(z @ Wf + bf)
i = torch.sigmoid(z @ Wi + bi)
g = torch.tanh(z @ Wg + bg)
o = torch.sigmoid(z @ Wo + bo)
c = f * c_prev + i * g
h = o * torch.tanh(c)
```

---

**Q6.** A motion-detection service flags frames in which something has moved. It runs without error, but on a static scene it reports motion in roughly every frame.

```python
 1   import cv2, numpy as np
 2
 3   cap  = cv2.VideoCapture("feed.mp4")
 4   ok, prev = cap.read()
 5   prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
 6
 7   THRESH   = 25
 8   MIN_AREA = 500
 9
10   while True:
11       ok, frame = cap.read()
12       if not ok:
13           break
14       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
15       diff = cv2.absdiff(prev, gray)
16       _, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)
17       mask = cv2.dilate(mask, None, iterations=2)
18       cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
19                                  cv2.CHAIN_APPROX_SIMPLE)
20       moved = any(cv2.contourArea(c) > MIN_AREA for c in cnts)
21       report(moved)
22       prev = gray
23   cap.release()
```

Which change is most likely to correct the behaviour?

A. Move line 22 above line 15, so that the previous frame is updated before the difference is computed rather than after.

B. Increase `MIN_AREA` to a value above the frame area, so that only changes spanning the whole image are reported as motion.

C. Apply Gaussian blur to each grayscale frame before differencing, so that sensor noise does not survive the threshold as spurious contours.

D. Replace `absdiff` with a signed subtraction, since absolute differencing doubles the apparent change at every pixel.

---

**Q7.** A RAG service must fit a system prompt, retrieved context, and a generated answer inside an 8,192-token window. The team evaluates four configurations. Retrieved context is whatever remains after the other allocations.

| Configuration | System prompt | Reserved for answer | Chunk size | Chunks requested |
|---|---|---|---|---|
| W | 600 | 1,200 | 800 | 8 |
| X | 600 | 1,200 | 512 | 12 |
| Y | 1,400 | 2,000 | 900 | 5 |
| Z | 600 | 2,000 | 700 | 7 |

Which configuration requests more context than the window permits?

A. Configuration Y, which requests 4,500 tokens of chunks against 4,792 available.

B. Configuration X, which requests 6,144 tokens of chunks against 6,392 available.

C. Configuration W, which requests 6,400 tokens of chunks against 6,392 available.

D. Configuration Z, which requests 4,900 tokens of chunks against 5,592 available.

---

**Q8.** A model's precision-recall behaviour is measured at three candidate thresholds:

| Threshold | Precision | Recall |
|---|---|---|
| T1 | 0.90 | 0.40 |
| T2 | 0.60 | 0.80 |
| T3 | 0.50 | 0.90 |

Which claim is supported?

A. T3 maximizes F1 at approximately 0.64, since recall rises faster across these thresholds than precision falls.

B. T2 maximizes F1 at approximately 0.69, though the choice between thresholds should still follow the relative cost of each error type.

C. All three thresholds give an F1 of approximately 0.65, since the harmonic mean is invariant to how precision and recall are traded.

D. T1 maximizes F1 at approximately 0.55, since precision contributes more heavily than recall to the harmonic mean.

---

**Q9.** A single attention head processes two tokens. After scaling, the score matrix and value matrix are:

```
scores = [[0.0, 0.0],        V = [[2.0, 4.0],
          [0.0, 0.0]]              [6.0, 8.0]]
```

A causal mask is applied so that position 1 may attend only to position 1, and position 2 to both positions. What is the output matrix?

A. `[[2.0, 4.0], [4.0, 6.0]]`

B. `[[4.0, 6.0], [4.0, 6.0]]`

C. `[[2.0, 4.0], [6.0, 8.0]]`

D. `[[8.0, 12.0], [4.0, 6.0]]`

---

**Q10.** An assistant answers arithmetic-heavy business questions such as *"if margin rises from 12% to 15% and volume falls 8%, what happens to gross profit?"*. The objective is to **maximize numerical accuracy**.

- P1: "Answer the question."
- P2: "Answer the question. Show your working step by step before stating the final answer."
- P3: "Answer the question. Respond with only the final number and no explanation."
- P4: "Answer the question. Be careful with the arithmetic."
- P5: "You are a skilled financial analyst. Answer the question."

Rank the prompts from most to least effective against this objective.

A. P5 > P2 > P4 > P3 > P1

B. P3 > P2 > P4 > P1 > P5

C. P2 > P4 > P5 > P1 > P3

D. P2 > P4 > P1 > P5 > P3

---

**Q11.** Two loan applicants have an identical income of ₹80,000. SHAP attributes +0.14 to income for the first applicant and −0.02 for the second.

Which claim is supported?

A. The model treats income non-monotonically, which means the attribution's sign is arbitrary and should not be reported.

B. The second applicant's SHAP computation has not converged, and increasing the number of sampled coalitions will bring the values into agreement.

C. A feature's contribution depends on the other feature values it interacts with, so the same income can push predictions in opposite directions for different applicants.

D. The explanation is inconsistent, since a Shapley attribution must assign the same contribution to identical feature values.

---

**Q12.** An agronomist compares 5 fertilizers across 8 farms, applying every fertilizer once on each farm. Farms differ substantially in baseline soil quality. Two analyses are considered: a one-way ANOVA ignoring farms, and a randomized block ANOVA treating farms as blocks.

Which claim is supported?

A. The block design is invalid here, since blocking requires each block to receive only a subset of the treatments under comparison.

B. The block design partitions farm-to-farm variation out of the residual, shrinking the error mean square and raising the power to detect fertilizer differences.

C. The block design spends degrees of freedom on farms without any compensating benefit, so the one-way analysis is the more powerful of the two.

D. The two analyses give identical F ratios for the fertilizer effect, since blocking changes only how the total sum of squares is labelled.

---

**Q13.** In a GRU, the reset gate `r_t` takes values very close to 0 for a particular timestep.

What is the effect on that timestep's computation?

A. The candidate state is computed almost entirely from the current input, since the previous hidden state is suppressed before the candidate transform.

B. The hidden state is carried forward unchanged, since a near-zero reset gate blocks any update from being written into it.

C. The gradient is blocked at that timestep, since a near-zero gate value drives the backward pass contribution to zero as well.

D. The candidate state and the previous hidden state are averaged equally, since the reset gate only rescales their relative weights.

---

**Q14.** A script fails with:

```
RuntimeError: Expected target size [8, 5], got [8]
```

The relevant code is:

```python
1   logits = model(x)            # shape (8, 5, 12)
2   targets = labels             # shape (8,)
3   loss = nn.CrossEntropyLoss()(logits, targets)
```

Which claim explains the failure?

A. For an input with a sequence dimension, the criterion expects class scores in dimension 1 and a target carrying one label per position, so the target must have shape [8, 12].

B. The criterion cannot accept three-dimensional input at all, so the logits must be flattened to shape [96, 5] and the targets repeated to match.

C. The targets must be one-hot encoded to shape [8, 5], since the criterion compares distributions rather than indices when input rank exceeds two.

D. The logits must be transposed to shape [8, 12, 5], since the criterion always expects the class dimension last for sequence inputs.

---

**Q15.** A grayscale image containing a bright square on a dark background is convolved with an unknown 3×3 kernel. In the output, the interior of the square and the background are both near zero, while the square's left and right edges appear as bright and dark vertical lines respectively. The top and bottom edges are near zero.

Which kernel produced this output?

A. A kernel whose rows are `[0, -1, 0]`, `[-1, 5, -1]`, `[0, -1, 0]`

B. A kernel whose rows are `[1, 1, 1]`, `[1, 1, 1]`, `[1, 1, 1]` scaled by one ninth

C. A kernel whose rows are `[1, 2, 1]`, `[0, 0, 0]`, `[-1, -2, -1]`

D. A kernel whose rows are `[-1, 0, 1]`, `[-2, 0, 2]`, `[-1, 0, 1]`

---

**Q16.** A Naive Bayes classifier distinguishes two classes with equal priors. For the document `cheap deal`, the smoothed likelihoods are:

| Term | P(term \| spam) | P(term \| ham) |
|---|---|---|
| cheap | 0.06 | 0.01 |
| deal | 0.05 | 0.02 |

Which claim is supported?

A. The document is classified as spam, with the posterior odds favouring spam by a factor of 3, driven entirely by the term `cheap`.

B. The classification is undetermined without the class priors, since equal priors do not permit a comparison of likelihood products.

C. The document is classified as spam, with the posterior odds favouring spam by a factor of 15 given equal priors.

D. The document is classified as ham, since ham's individual likelihoods are each smaller and Naive Bayes selects the lower product.

---

**Q17.** A retrieval system is evaluated on three queries. The rank of the first relevant document is 1 for the first query, 3 for the second, and 5 for the third.

What is the mean reciprocal rank?

A. 0.667

B. 0.511

C. 0.333

D. 3.000

---

**Q18.** Three systems are assessed under the EU AI Act: (i) a spam filter for an email client, (ii) a tool that scores applicants for a vocational training programme, (iii) a system that infers student emotional states during online examinations.

How are they classified?

A. Minimal risk, high risk, and prohibited respectively.

B. Limited risk, high risk, and high risk respectively.

C. Minimal risk, limited risk, and high risk respectively.

D. All three are high risk, since each processes personal data and produces an output that affects an individual.

---

**Q19.** A transformer block has `d_model = 512` and a feed-forward inner dimension of 2,048. Counting only the four attention projection matrices and the two feed-forward matrices, and ignoring biases and normalization parameters, how many parameters does one block hold?

A. 4,194,304

B. 1,048,576

C. 2,097,152

D. 3,145,728

---

**Q20.** A churn model uses a feature `avg_spend_last_90d`. The engineer computes it as:

```python
df = df.sort_values("date")
df["avg_spend_last_90d"] = (
    df.groupby("customer_id")["spend"]
      .transform(lambda s: s.rolling(90, center=True, min_periods=1).mean())
)
```

Cross-validated AUC is 0.89; the model scores 0.66 on the following quarter.

Which claim identifies the cause?

A. Using `min_periods=1` allows windows shorter than 90 days near each customer's first transaction, producing unstable early values.

B. The rolling mean should be computed on the raw index rather than after sorting, since sorting reorders rows and invalidates the window.

C. A centred rolling window averages 45 days before and 45 days after each row, so every feature value incorporates spending that had not yet occurred at scoring time.

D. The grouping by customer is applied after sorting by date, so rolling windows span customer boundaries and blend unrelated spending histories.

---

**Q21.** A content team drafts product descriptions that must follow an internal style guide: no superlatives, active voice, and a length of 40 to 60 words. The objective is to **maximize adherence to all three rules simultaneously**.

- P1: "Write a product description for this item."
- P2: "Write a product description in 40 to 60 words. Use active voice. Do not use superlatives such as 'best', 'finest' or 'unbeatable'."
- P3: "Write a product description following our house style."
- P4: "Write a product description. Here is one that follows our style: 'The kettle boils one litre in three minutes. Its stainless body resists scale, and the handle stays cool during pouring. A hinged lid opens with one hand.'"
- P5: "You are a seasoned copywriter. Write a product description in our brand voice."

Rank the prompts from most to least effective against this objective.

A. P3 > P2 > P4 > P1 > P5

B. P4 > P2 > P3 > P1 > P5

C. P2 > P4 > P3 > P5 > P1

D. P2 > P4 > P5 > P1 > P3

---

**Q22.** A team fits a single decision tree as a global surrogate for a complex ensemble, achieving an R² of 0.62 between the surrogate's outputs and the ensemble's outputs.

Which claim is supported?

A. The surrogate explains 62% of the variance in the underlying data, making it a reasonable simplification of the ensemble for reporting purposes.

B. The surrogate's fidelity to the ensemble is low, so rules read from it may describe behaviour the ensemble does not exhibit and should not be presented as explanations.

C. The surrogate is faithful wherever the ensemble is confident, since fidelity failures concentrate exclusively in low-confidence regions of the feature space.

D. Surrogate R² measures predictive accuracy on held-out labels, so a value of 0.62 indicates the ensemble itself generalizes poorly.

---

**Q23.** A credit model is a logistic regression with the fitted linear predictor:

```
logit(p) = -3.2 + 0.045 * age + 1.1 * owns_home - 0.6 * late_payments_12m
```

An applicant is 40 years old, owns their home, and has 2 late payments in the past 12 months.

What is the predicted probability?

A. About 0.182

B. About 0.223

C. About 0.500

D. About 0.818

---

**Q24.** A team builds a sequence model and receives:

```
RuntimeError: input.size(-1) must be equal to input_size. Expected 300, got 64
```

The layer was constructed as `nn.LSTM(300, 128, batch_first=True)` and called on the output of an embedding layer.

Which claim identifies the misconfiguration?

A. The batch size is 64 and conflicts with the declared input size, so `batch_first` must be set to False for the shapes to align.

B. The sequence length is 64 while the layer expects 300 timesteps, so the inputs must be padded to the declared length.

C. The hidden size of 128 conflicts with the input size of 300, since a recurrent layer requires these two values to be equal.

D. The embedding layer produces vectors of width 64, so its embedding dimension and the recurrent layer's declared input size disagree.

---

**Q25.** Two Canny configurations are run on the same photograph. Output 1 contains long continuous contours around objects plus a moderate amount of texture detail inside them. Output 2 contains only short fragments at the strongest object boundaries, with all texture detail absent.

Which pairing is consistent with these outputs?

A. Output 1 used a larger Gaussian kernel and Output 2 used a smaller one, with identical thresholds throughout.

B. Output 1 disabled non-maximum suppression and Output 2 enabled it, with identical thresholds throughout.

C. Output 1 used thresholds (50, 150) and Output 2 used thresholds (150, 250).

D. Output 1 used thresholds (150, 250) and Output 2 used thresholds (50, 150).

---

**Q26.** Which pair of expressions produces tensors that are equal in value for any float tensor `x` that requires gradient?

A. `x.numpy()` and `x.detach()`, both of which convert the tensor to a form that the autograd engine no longer tracks.

B. `x.detach()` and `x.data`, both of which return a tensor sharing storage with `x` and excluded from the autograd graph.

C. `x.clone()` and `x.detach()`, both of which allocate new storage and sever the connection to the computation graph.

D. `x.requires_grad_(False)` and `x.detach()`, both of which leave `x` itself unchanged while returning a graph-free view.

---

**Q27.** A hybrid retriever combines a BM25 score and a cosine similarity by adding them directly. BM25 scores for a query range from 4 to 31; cosine similarities range from 0.61 to 0.88. Results are dominated by the lexical ranking.

Which claim identifies the cause and the appropriate fix?

A. The two scores occupy incomparable ranges, so direct addition weights BM25 by roughly two orders of magnitude; rank-based fusion or per-retriever normalization removes the scale dependence.

B. Cosine similarity is bounded above by 1 and therefore carries less information than an unbounded score, so BM25 should be used alone for this corpus.

C. The dense retriever is undertrained, since a well-fitted embedding model produces similarities spread across the full interval from 0 to 1.

D. Addition is the correct fusion operator, and the imbalance should be corrected by raising the cosine similarities to a fractional power before summing.

---

**Q28.** A byte-pair encoding tokenizer has learned these merges, in order:

```
1. ("l", "o")   -> "lo"
2. ("lo", "w")  -> "low"
3. ("e", "r")   -> "er"
4. ("low", "er") -> "lower"
```

How many tokens does the word `lowest` produce?

A. 4

B. 3

C. 2

D. 6

---

**Q29.** A bank has deployed a high-risk credit scoring system. Six months after launch, its own monitoring detects that rejection rates for one applicant segment have risen sharply in a way the original conformity assessment did not anticipate.

Which obligation does this situation principally engage?

A. Data minimization, requiring the bank to reduce the personal data collected until the anomaly has been explained.

B. Prohibition, requiring immediate and permanent withdrawal of the system from the market pending a new conformity assessment.

C. Transparency, requiring the bank to inform rejected applicants that an automated system contributed to the decision.

D. Post-market monitoring, requiring the provider to operate a system that collects and reviews performance data throughout the product's lifetime and to act on what it reveals.

---

**Q30.** A 32-layer decoder has `d_model = 4096` and uses standard multi-head attention. During generation it holds a key-value cache for a single sequence of 2,048 tokens in 16-bit precision.

Approximately how much memory does the cache occupy?

A. 256 MiB

B. 512 MiB

C. 1 GiB

D. 2 GiB

---

**Q31.** A team tunes hyperparameters by selecting, for each of 5 outer folds, the configuration that scores best on that same fold. They report the mean of those best scores as the model's expected performance.

Which claim is supported?

A. The procedure is sound but inefficient, since the same result can be obtained from a single train-validation split at lower computational cost.

B. The reported figure is optimistically biased because the selection and the evaluation use the same data; nested cross-validation with an inner selection loop gives an unbiased estimate.

C. The reported figure is unbiased provided the folds are stratified, since stratification guarantees that each fold represents the full distribution.

D. The reported figure is pessimistically biased, since selecting per fold prevents any single configuration from benefiting from the full training set.

---

**Q32.** An agent has two tools available: `search_orders(customer_email)` returning up to 50 recent orders, and `get_order(order_id)` returning full detail for one. A user asks for the delivery status of their most recent order and supplies only their email address.

Which sequence is most appropriate?

A. Call `search_orders` with the email, select the most recent identifier from the result, then call `get_order` with it.

B. Call `search_orders` and `get_order` in parallel, then reconcile the two results before replying to the user.

C. Call `search_orders` with the email and answer directly from its summary, since a second call cannot add information the first lacks.

D. Call `get_order` with a generated identifier, then fall back to `search_orders` if the first call returns an error.

---

**Q33.** A model is monitored with a 7-day rolling window compared against a fixed training reference. Over four months, PSI on every feature stays below 0.1 in each weekly comparison, yet a comparison of the most recent month against the original training data gives PSI values above 0.3 on several features.

Which claim is supported?

A. The monthly figure is an artefact of the larger sample size, since PSI grows mechanically with the number of observations compared.

B. Drift has occurred in the label distribution rather than the features, which weekly feature-level PSI is not designed to capture.

C. The weekly comparisons are misconfigured, since a genuine distribution change must appear in a 7-day window before it can appear over a month.

D. Drift is gradual: week-on-week movement is too small to register, but it accumulates against the fixed reference, so monitoring needs both a short window and a long baseline comparison.

---

**Q34.** A diagnostic considers three mutually exclusive conditions with prior probabilities 0.5, 0.3 and 0.2. A symptom is observed whose probability is 0.1 under the first condition, 0.4 under the second, and 0.6 under the third.

Which claim about the posteriors is supported?

A. The posteriors are 0.05, 0.12 and 0.12, since these are the joint probabilities of each condition with the symptom.

B. The first condition remains the most probable, since its prior is large enough to dominate the likelihood difference.

C. The second and third conditions are now equally probable at roughly 0.41 each, with the first falling to roughly 0.17.

D. The third condition is now most probable at roughly 0.60, matching the likelihood of the symptom under it.

---

**Q35.** A team measures the variance of activations at each layer of a 12-layer ReLU network immediately after initialization:

| Layer | 1 | 4 | 8 | 12 |
|---|---|---|---|---|
| Activation variance | 1.02 | 0.31 | 0.04 | 0.005 |

Which claim is supported?

A. The pattern is expected and harmless, since ReLU necessarily halves variance at every layer and training corrects this within the first few epochs.

B. The pattern indicates dead units rather than an initialization problem, since only permanently inactive units can reduce variance by this magnitude.

C. The initialization scale is too small for ReLU, so variance contracts with depth; He initialization scales the standard deviation as the square root of two over fan-in to counteract this.

D. The initialization scale is too large, so variance would explode were the network deeper; reducing the standard deviation by a factor of two would stabilize it.

---

**Q36.** A ViT is configured with 16×16 patches and a learned positional embedding table sized for 224×224 inputs. A team wishes to fine-tune it on 448×448 images without changing the patch size.

Which claim is supported?

A. No change is needed, since positional embeddings are computed from the sequence length at run time rather than looked up from a table.

B. The patch count rises from 196 to 784, so the positional embedding table must be interpolated to the new grid; the patch projection itself is unaffected.

C. The patch count rises from 196 to 392, so the positional embeddings can be duplicated once to cover the additional positions.

D. The patch projection must be resized, since it maps raw pixels and its input dimension scales with the image rather than the patch.

---

**Q37.** A team needs a tensor `z`, derived from a model's output `y`, to be usable in a logging computation without contributing to gradients, while `y` itself must remain part of the graph for the backward pass.

Which approach is correct?

A. Use `z = y.detach()`, which returns a tensor sharing `y`'s values but excluded from the graph, leaving `y` itself untouched.

B. Use `z = y.clone()`, which allocates new storage and therefore produces a tensor the autograd engine does not track.

C. Wrap the logging computation in `torch.no_grad()`, which retroactively removes `y` from the graph for all subsequent operations.

D. Call `y.requires_grad_(False)` before the logging computation, which affects only the logged copy and leaves `y` trainable.

---

**Q38.** A RAG system is asked *"Which of our data centres uses the same cooling vendor as the Pune facility?"* Retrieval returns the Pune facility's vendor contract and, separately, a list of vendors by site — but never in the same query. The answer is consistently wrong.

Which change addresses the cause?

A. Raise `top_k` so that both documents appear together among the retrieved candidates for the original query.

B. Replace dense retrieval with lexical search, since vendor and facility names are proper nouns that embeddings represent poorly.

C. Increase chunk size so that vendor information for every site appears within a single retrievable chunk.

D. Decompose the question into sub-queries — first resolve Pune's vendor, then search for other sites using that vendor — and retrieve for each in turn.

---

**Q39.** A skip-gram model is trained on a vocabulary of 50,000 words using negative sampling with 5 negative samples per positive pair.

For one training pair, how many output-layer weight vectors receive an update, and why does this matter?

A. 50,000 vectors, since the softmax normalization term depends on every output vector regardless of the sampling scheme.

B. 1 vector — the positive only — since negative sampling reduces the update to the observed context and discards the rest.

C. 6 vectors — one positive and five negatives — rather than all 50,000, which is what makes the objective tractable on large vocabularies.

D. 5 vectors — the negatives only — since the positive pair's vector is updated through the input embedding rather than the output layer.

---

**Q40.** A hospital deploys a sepsis-risk model. Clinicians receive a risk score but must independently review the patient and may act against the score; the system cannot initiate any treatment on its own, and a clinician's decision is recorded alongside each alert.

Which form of human oversight does this describe?

A. Fully automated decision-making, since the model's score determines the clinical pathway that is ultimately followed.

B. Human-in-the-loop, since each system output requires human review and action before anything follows from it.

C. Human-on-the-loop, since the system acts autonomously while a human monitors its behaviour and may intervene.

D. Human-in-command, since a human defines the system's overall operating conditions without reviewing individual outputs.

---

**Q41.** A team replaces full attention with sliding-window attention of width 512 in a model handling 16,384-token inputs.

Which claim is supported?

A. Attention cost becomes linear in sequence length, but a single layer can no longer relate tokens more than 512 positions apart; stacking layers grows the effective receptive field.

B. Attention cost remains quadratic, since every token still computes a score against every other token before the window mask is applied.

C. Attention cost becomes linear and there is no loss of expressiveness, since information outside the window is recovered by the feed-forward sublayer.

D. The window must equal the sequence length for causal masking to remain valid, so sliding-window attention is incompatible with autoregressive generation.

---

**Q42.** A gradient-boosting regressor is initialized with a constant prediction of 10.0 for every sample and uses a learning rate of 0.1. For one sample whose true value is 16.0, the first tree predicts the residual exactly, and the second tree again predicts the remaining residual exactly.

What is the model's prediction for that sample after two trees?

A. 11.14

B. 12.00

C. 16.00

D. 10.60

---

**Q43.** A support assistant receives messages that are often ambiguous, such as *"it's not working"* with no product named. The objective is to **maximize the proportion of conversations that reach a correct resolution**.

- P1: "Help the user with their problem."
- P2: "Help the user. If the message does not identify the product and the specific symptom, ask one clarifying question before offering any solution."
- P3: "Help the user. Ask clarifying questions whenever you need to."
- P4: "Help the user. Example — User: 'it's not working'. Assistant: 'Which product is this about, and what happens when you try to use it?'"
- P5: "You are a patient and thorough support agent. Help the user with their problem."

Rank the prompts from most to least effective against this objective.

A. P3 > P4 > P2 > P1 > P5

B. P4 > P2 > P3 > P5 > P1

C. P2 > P3 > P4 > P1 > P5

D. P2 > P4 > P3 > P1 > P5

---

**Q44.** A model's probability outputs are well calibrated overall: across all predictions, confidence matches accuracy closely. Broken out by group, the picture differs:

| Group | Mean confidence | Accuracy |
|---|---|---|
| A | 0.80 | 0.88 |
| B | 0.80 | 0.66 |

Which claim is supported?

A. The model is miscalibrated for group B only, so applying temperature scaling to group B's outputs alone restores calibration everywhere.

B. Subgroup calibration is not a meaningful concept, since calibration is defined over the full predictive distribution rather than over partitions of it.

C. Aggregate calibration can mask opposing subgroup errors that cancel, so a single calibration curve is insufficient and calibration must be assessed per group.

D. The model is well calibrated, since overall confidence matches overall accuracy and subgroup deviations are a sampling artefact.

---

**Q45.** A model is fitted twice on the same data. Model 1 includes both `height_cm` and `height_in`; Model 2 drops `height_in`.

| | Model 1 | Model 2 |
|---|---|---|
| Coefficient on `height_cm` | 4.82 (SE 6.31) | 1.93 (SE 0.24) |
| Held-out RMSE | 3.14 | 3.15 |

Which claim is supported?

A. Model 2 is substantially better, since the near-identical RMSE combined with a smaller coefficient shows that Model 1 was overfitting the redundant feature.

B. Model 1's coefficient is unusable for inference because collinearity has inflated its standard error, yet the near-identical RMSE confirms that prediction is essentially unaffected.

C. Model 1's larger coefficient indicates that height matters more once both encodings are present, so the second model understates the effect of height.

D. The two models cannot be compared, since dropping a predictor changes the design matrix and invalidates any comparison of held-out error.

---

**Q46.** A network places dropout with p = 0.5 immediately before a batch normalization layer. Validation performance is much worse than training performance in a way that regularization strength alone does not explain.

Which claim identifies the mechanism?

A. Dropout changes the variance of its output between training and inference, so the normalization layer's running statistics — estimated under dropout — do not match the statistics it encounters at inference.

B. Dropout and batch normalization cannot be combined in any order, since both rescale activations and the two rescalings compound multiplicatively.

C. Batch normalization removes the regularization effect of dropout entirely, so the network behaves as though dropout were absent and overfits accordingly.

D. Dropout must precede the activation function rather than the normalization layer, since zeroing post-activation values discards the non-linearity's output.

---

**Q47.** A team evaluates augmentations one at a time on a document-scan classifier, recording the change in validation accuracy:

| Augmentation | Δ validation accuracy |
|---|---|
| Random rotation ±5° | +1.2 |
| Random brightness ±15% | +0.8 |
| Random horizontal flip | −7.4 |
| Gaussian blur σ ≤ 1.0 | +0.4 |

Which claim best explains the pattern?

A. Horizontal flipping is applied too frequently relative to the others, and reducing its probability to 0.1 would make its contribution positive.

B. The three positive augmentations are individually too weak to matter, so the flip result is the only genuine signal in the table.

C. Flipping harms accuracy because it doubles the effective dataset size, which requires a proportionally longer training schedule to exploit.

D. Horizontal flipping produces mirrored text, which no longer corresponds to the class the label asserts, so the model trains against corrupted targets.

---

**Q48.** Which implementation correctly accumulates gradients so that four micro-batches of 16 behave like one batch of 64?

A.
```python
for i, (xb, yb) in enumerate(loader):
    loss = criterion(model(xb), yb)
    loss.backward()
    optimizer.step()
    if (i + 1) % 4 == 0:
        optimizer.zero_grad()
```

B.
```python
losses = []
for i, (xb, yb) in enumerate(loader):
    losses.append(criterion(model(xb), yb))
    if (i + 1) % 4 == 0:
        sum(losses).backward()
        optimizer.step()
        optimizer.zero_grad()
```

C.
```python
for i, (xb, yb) in enumerate(loader):
    loss = criterion(model(xb), yb) / 4
    loss.backward()
    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

D.
```python
for i, (xb, yb) in enumerate(loader):
    optimizer.zero_grad()
    loss = criterion(model(xb), yb) / 4
    loss.backward()
    if (i + 1) % 4 == 0:
        optimizer.step()
```

---

**Q49.** A RAG team must choose a reranking configuration. Their latency budget for the retrieval stage is 400 ms.

| Option | Candidates reranked | Cross-encoder latency | Recall@5 after rerank |
|---|---|---|---|
| J | none | 0 ms | 0.61 |
| K | 20 | 180 ms | 0.79 |
| L | 50 | 430 ms | 0.84 |
| M | 100 | 860 ms | 0.86 |

Which claim is supported?

A. Option M is correct because recall is the metric that determines answer quality, and latency budgets are soft constraints in retrieval systems.

B. Option J is preferable, since any cross-encoder stage introduces latency disproportionate to the recall it recovers at this scale.

C. Option K is the best available choice: it delivers most of the achievable recall gain within the budget, while L and M both exceed it.

D. Option L should be selected and the budget renegotiated, since the recall gain from K to L is larger than the gain from J to K.

---

**Q50.** A named entity recognition system is evaluated on a test set containing 10 gold entities. It predicts 12 entities, of which 7 match a gold entity in both span and type.

What is the entity-level F1?

A. 0.583

B. 0.636

C. 0.700

D. 0.778

---

**Q51.** A team assembles a training corpus by scraping publicly accessible web pages. A reviewer asks what documentation the corpus requires before it can be used in a commercial model.

Which claim is supported?

A. Documentation of source, licence terms and any personal data contained is required, since public accessibility does not by itself confer a licence for commercial training use.

B. Only the volume and date of collection need recording, since licence terms apply to redistribution of the source material rather than to model training.

C. No documentation is needed provided the model does not reproduce its training data verbatim, since non-reproduction removes any licensing question.

D. Public accessibility establishes permission to train, so provenance documentation is advisable for reproducibility but carries no licensing significance.

---

**Q52.** A model trained with rotary positional embeddings and one trained with learned absolute positional embeddings, both at a 4,096-token context, are each given a 12,000-token input.

Which claim is supported?

A. The rotary model fails while the learned-embedding model extrapolates, since learned tables generalize to unseen positions through the embedding geometry.

B. Both extrapolate without loss, since attention is invariant to absolute position and only relative offsets affect the computation.

C. Both fail identically, since any positional scheme is defined only up to the maximum length seen during training.

D. The learned-embedding model has no entry for positions beyond 4,096, while the rotary model can compute positional terms for any index though quality still degrades beyond its trained range.

---

**Q53.** A node containing 100 samples, evenly split between two classes, is split into a left child with 60 samples (45 positive) and a right child with 40 samples (5 positive).

What is the Gini impurity decrease for this split?

A. 0.5000

B. 0.0625

C. 0.1875

D. 0.3125

---

**Q54.** A model screens open-ended interview answers and produces a short written evaluation of each. The objective is to **minimize the influence of candidate name, school and phrasing style on the evaluation**.

- P1: "Evaluate this interview answer."
- P2: "Evaluate this interview answer against the rubric below, scoring each criterion separately and citing the part of the answer that supports each score. Ignore the candidate's name, institution and writing polish."
- P3: "Evaluate this interview answer fairly and without bias."
- P4: "Evaluate this interview answer. Example — Answer mentions a rollback plan, so the 'risk awareness' criterion scores 3 of 4."
- P5: "You are an impartial hiring panellist. Evaluate this interview answer."

Rank the prompts from most to least effective against this objective.

A. P2 > P3 > P4 > P5 > P1

B. P2 > P4 > P3 > P5 > P1

C. P3 > P2 > P4 > P1 > P5

D. P4 > P2 > P3 > P1 > P5

---

**Q55.** An ensemble of five independently trained models produces, for two inputs:

| Input | Model predictions | Mean confidence |
|---|---|---|
| 1 | class 3 from all five models | 0.91 |
| 2 | class 3, 3, 7, 7, 1 | 0.88 |

Which claim is supported?

A. Input 2 carries high epistemic uncertainty, which ensemble disagreement captures and which a single model's confidence score would have missed.

B. Input 2 carries high aleatoric uncertainty, since the disagreement originates in noise inherent to that input rather than in the models.

C. Both inputs carry equivalent uncertainty, since the mean confidence values differ by only three percentage points.

D. Input 1 carries higher uncertainty, since unanimous agreement among correlated models indicates they share the same systematic blind spot.

---

**Q56.** PCA is applied to a standardized dataset. The loadings of the first two components on four features are:

| Feature | PC1 | PC2 |
|---|---|---|
| income | 0.62 | −0.11 |
| savings | 0.58 | −0.09 |
| age | 0.41 | 0.33 |
| commute_km | −0.33 | 0.93 |

Which claim is supported?

A. PC1 captures a general financial-capacity dimension shared by income, savings and age, while PC2 is dominated by commute distance.

B. PC1 is dominated by commute distance, since that feature carries the largest absolute loading anywhere in the table.

C. The components are not orthogonal, since income and savings load similarly on PC1 and must therefore be correlated within it.

D. PC2 should be discarded, since a component dominated by a single feature carries no more information than that feature alone.

---

**Q57.** A two-layer network computes `h = w1 * x`, `a = ReLU(h)`, `o = w2 * a`, with squared-error loss `L = (o - y)²`. For one sample: `x = 2`, `w1 = 3`, `w2 = -1`, `y = -4`.

What is `dL/dw1`?

A. −24

B. −8

C. 4

D. 8

---

**Q58.** Photographs taken in a poorly lit warehouse have low contrast overall, and applying global histogram equalization makes the dim corners legible but blows out detail near the overhead lights.

Which approach addresses this?

A. Apply global equalization twice, since the second pass redistributes intensities that the first pass over-concentrated.

B. Convert to HSV and equalize the hue channel, which separates illumination from colour and leaves bright regions unaffected.

C. Apply CLAHE, which equalizes within local tiles and clips the histogram to limit contrast amplification in already-bright regions.

D. Apply gamma correction with γ greater than 1, which compresses the bright range while expanding the dark range uniformly.

---

**Q59.** A tensor `x` has shape `(4, 6)` and `idx = torch.tensor([[0, 2], [1, 3], [5, 0], [4, 4]])`. The operation `x.gather(1, idx)` is performed.

What is the shape of the result, and what does it contain?

A. A RuntimeError is raised, since `gather` requires the index tensor to have the same shape as the source tensor.

B. Shape `(4, 2)`, where row `i` contains the two elements of `x`'s row `i` at the column positions named in row `i` of `idx`.

C. Shape `(4, 6)`, where the elements named by `idx` are retained and all other positions are filled with zeros.

D. Shape `(2, 6)`, where each output row gathers whole rows of `x` selected by the corresponding entries of `idx`.

---

**Q60.** A RAG system serves several tenants from one index, with a `tenant_id` field on every chunk. An engineer retrieves the top 100 chunks by vector similarity and then discards those belonging to other tenants.

Which claim is supported?

A. Post-filtering can return fewer than the intended number of chunks for small tenants, and pre-filtering the vector search by `tenant_id` avoids both this and the wasted similarity computation.

B. Post-filtering is the correct approach, since applying a metadata constraint before the vector search invalidates the approximate nearest-neighbour index.

C. The two orderings are equivalent in result and differ only in latency, so the choice between them is a pure performance question.

D. Filtering by tenant is unnecessary when embeddings are computed per tenant, since vectors from different tenants cannot be near one another.

---

**Q61.** Two preprocessing pipelines are applied to the same corpus before fitting a bag-of-words vectorizer:

```
Pipeline 1: lowercase -> remove punctuation -> split -> Porter stemming
Pipeline 2: lowercase -> remove punctuation -> split
```

Which claim is supported?

A. Pipeline 1 yields a larger vocabulary, since stemming introduces truncated forms alongside the originals in the feature space.

B. The two pipelines yield identical vocabularies, since stemming maps each word to itself unless the word is irregular.

C. Pipeline 2 yields a smaller vocabulary, since removing the stemming stage prevents variant spellings from being counted separately.

D. Pipeline 1 yields a smaller vocabulary, since stemming collapses inflected forms onto a shared stem at the cost of some distinctions being lost.

---

**Q62.** A team runs three separate differentially private queries against the same dataset, each with a privacy budget of ε = 0.5.

Under basic sequential composition, what is the total privacy loss, and what does it imply?

A. ε = 0.125, since independent queries compose multiplicatively and each successive query reduces the cumulative loss.

B. ε = 0.5, since the budget applies per dataset and repeated queries against the same data do not compound the loss.

C. ε = 1.5, since sequential budgets add, so a fixed total budget caps how many queries may be answered before analysis must stop.

D. ε = 0.167, since dividing the budget across the three queries is what keeps the total within the original allowance.

---

**Q63.** A 7-billion-parameter model is loaded for fine-tuning. The team compares holding the base weights in 16-bit precision against 4-bit quantization.

Approximately how much memory do the base weights occupy in each case?

A. 28 GB and 7 GB respectively, since 16-bit precision requires four bytes per parameter before any optimizer state is allocated.

B. 14 GB in both cases, since quantization reduces computation rather than the memory the weights themselves occupy.

C. 14 GB and 3.5 GB respectively, a fourfold reduction that is what allows the model to be adapted on a single consumer GPU.

D. 7 GB and 1.75 GB respectively, since each parameter occupies one byte at 16-bit precision and a quarter byte at 4-bit.

---

**Q64.** A team fits ridge regression on features measured in widely differing units — income in rupees, age in years, and a count of prior purchases — without standardizing first.

Which claim is supported?

A. Only the intercept is affected, since the penalty term conventionally excludes the intercept and leaves slope coefficients scale-free.

B. The penalty applies to raw coefficient magnitudes, so features on large numeric scales carry small coefficients and are penalized far less than features on small scales.

C. Standardization is unnecessary for ridge regression, since the L2 penalty is invariant to linear rescaling of the input features.

D. The model fails to converge, since regularized regression requires all features to occupy the unit interval before fitting.

---

**Q65.** An agent handling a multi-step booking task repeatedly re-reads the same document across steps, and by step 12 its context contains six near-identical copies of the same passage while earlier decisions have scrolled out of view.

Which change addresses the root problem?

A. Maintain an explicit working state — a running summary of decisions and facts already established — and write tool results into it rather than accumulating raw outputs.

B. Reduce the number of steps the agent is permitted to take, so the context cannot grow to the point where earlier content is displaced.

C. Disable the document-reading tool after its first successful call, so that repeated reads become impossible by construction.

D. Increase the context window, since the difficulty arises from capacity rather than from how the agent manages what it stores.

---

**Q66.** A team validates a saliency method by progressively randomizing the trained model's weights, layer by layer from the output backwards, and recomputing the saliency map at each stage. The maps remain visually almost unchanged throughout.

Which claim is supported?

A. The result shows the model has learned nothing beyond low-level features, so the saliency method is sound and the model should be retrained.

B. Randomizing weights from the output backwards is invalid, since saliency depends only on the first layer and later layers cannot affect it.

C. The method is robust, since a stable explanation across perturbations demonstrates that it captures structure invariant to parameter noise.

D. The method fails the model-randomization sanity check: an explanation insensitive to the model's learned parameters is reflecting the input's edge structure rather than the model's reasoning.

---

**Q67.** A regression output is reported as:

| Predictor | Coefficient | Std. error | t | p |
|---|---|---|---|---|
| tenure_months | 0.021 | 0.004 | 5.25 | < 0.001 |
| n_tickets | 0.310 | 0.290 | 1.07 | 0.29 |

R² = 0.14, n = 900.

Which claim is supported?

A. `tenure_months` is the weaker predictor of the two, since its coefficient is smaller in magnitude than that of `n_tickets`.

B. `n_tickets` has no relationship with the outcome, since its coefficient failed to reach significance at the conventional threshold.

C. `tenure_months` is estimated precisely and is significant, while `n_tickets` is estimated too imprecisely to distinguish from zero; the low R² means both leave most variation unexplained.

D. The R² of 0.14 invalidates the reported significance, since coefficients cannot be trusted in a model explaining so little of the variance.

---

**Q68.** A residual block computes `y = x + F(x)`.

Which claim explains why this construction eases the training of very deep networks?

A. The residual branch is trained separately from the identity path, so gradients never traverse more than one block at a time.

B. The identity path carries gradient to earlier layers with a multiplicative factor of one, so the gradient does not have to survive repeated multiplication through every intervening transformation.

C. The addition halves the magnitude of the activations at each block, which prevents the forward pass from saturating as depth increases.

D. The identity path removes the need for non-linearities, so the network behaves as a shallow linear model that is trivially optimizable.

---

**Q69.** A detector's predictions for one class are ranked by confidence and matched against 3 ground-truth objects. The ranked outcomes are: true positive, false positive, true positive, false positive, true positive.

What is the average precision, computed as the mean of the precision values at each true positive?

A. 0.756

B. 0.833

C. 0.889

D. 0.600

---

**Q70.** A Keras model is trained on a dataset where one class represents 3% of the samples. The team passes `class_weight={0: 1.0, 1: 30.0}` to `model.fit`.

Which claim is supported?

A. The weighting scales the loss contribution of minority-class samples, shifting the decision boundary without altering the number of training samples.

B. The weighting duplicates minority-class samples thirty times within each epoch, which is equivalent to oversampling the minority class.

C. The weighting rescales the minority class's input features, so the network sees those samples as more distinct from the majority.

D. The weighting applies only to the reported metrics and leaves the gradient computation and the fitted decision boundary unchanged.

---

**Q71.** A corpus of technical standards contains numbered clauses of 80 to 900 tokens, each self-contained. Retrieval currently uses fixed 400-token chunks. The team compares three strategies on the same evaluation set.

| Strategy | Recall@5 | Answers citing a truncated clause |
|---|---|---|
| Fixed 400-token | 0.68 | 31% |
| Fixed 400-token with 100-token overlap | 0.74 | 22% |
| One chunk per clause | 0.86 | 2% |

Which claim is supported?

A. The gain from clause-level chunking comes from its larger average chunk size, so raising the fixed chunk size to 900 tokens would reproduce the result.

B. The comparison is confounded, since strategies producing different chunk counts cannot be compared on recall at a fixed cutoff of five.

C. Overlap is the better long-term choice, since clause-level chunking cannot generalize to documents that lack explicit numbering.

D. Clause-level chunking aligns boundaries with the document's own semantic units, which removes most truncation and lifts recall; overlap only mitigates a boundary problem that structure-aware splitting eliminates.

---

**Q72.** A bigram language model is estimated from the corpus:

```
the cat sat on the mat the cat ran
```

with add-one smoothing over a vocabulary consisting of the distinct words in this corpus.

What is P(`cat` | `the`)?

A. 3/10

B. 2/9

C. 1/3

D. 2/3

---

**Q73.** A customer exercises the GDPR right to erasure. The company deletes their records from its databases, but a model trained on those records remains in production.

Which claim is supported?

A. Deleting the source records discharges the obligation entirely, since a trained model contains parameters rather than personal data.

B. Model parameters can retain information about individuals, so the organization must assess whether the model's continued use constitutes processing and consider retraining or unlearning.

C. The obligation cannot apply to models at all, since retraining is technically burdensome and the regulation exempts processing that is disproportionately costly.

D. The model must be deleted immediately in every case, since any model trained on erased data is unlawful from the moment of the request.

---

**Q74.** A system generates short factual answers from a knowledge base, and the team requires the same question to yield the same answer on every call.

Which decoding configuration is appropriate, and what is the trade-off?

A. Greedy decoding or temperature 0, which makes generation deterministic given the same input, at the cost of output diversity that this task does not need.

B. Nucleus sampling with a low `top_p`, which restricts the candidate pool enough to guarantee identical output on repeated calls.

C. Beam search with a wide beam, which is deterministic and additionally guarantees the globally most probable sequence is returned.

D. Temperature 1.0 with a fixed random seed, which is the only configuration that produces reproducible output across different serving replicas.

---

**Q75.** A logistics firm wants to forecast daily parcel volume for the next 14 days. History covers three years and shows a strong weekly cycle, a December surge, and a steady upward trend. Promotional calendars for the forecast window are known in advance.

Which approach fits these requirements?

A. A random forest trained on shuffled daily records using the calendar date as the sole predictor variable.

B. k-means clustering on daily volume profiles, assigning each future date to its nearest cluster centroid as the forecast.

C. A classification model predicting whether each future day will exceed the historical median volume, thresholded at 0.5.

D. A time-series model with explicit weekly and annual seasonality plus a trend term, taking the known promotional calendar as an exogenous regressor.

---

**Q76.** A prompt is used in an automated pipeline that will run against several model versions over time. The objective is to **maximize the stability of extracted output as the underlying model changes**.

- P1: "Pull out the key details from this contract."
- P2: "Extract these fields from the contract: party_a, party_b, effective_date (ISO 8601), termination_notice_days (integer). Return JSON with exactly these four keys. If a field is absent, use null."
- P3: "Extract the important contract fields as JSON."
- P4: "Extract the contract fields. Example: {\"party_a\": \"Acme Ltd\", \"party_b\": \"Beta LLC\", \"effective_date\": \"2025-04-01\", \"termination_notice_days\": 30}"
- P5: "You are an experienced contracts analyst. Extract the salient contract details."

Rank the prompts from most to least effective against this objective.

A. P4 > P2 > P3 > P1 > P5

B. P2 > P3 > P4 > P5 > P1

C. P2 > P4 > P3 > P1 > P5

D. P2 > P4 > P5 > P3 > P1

---

**Q77.** Before replacing a deployed fraud model, a team runs the new model alongside the existing one for six weeks. Both score every transaction, but only the existing model's decisions take effect; the new model's scores are logged and compared afterwards.

What is this arrangement, and what does it establish?

A. A canary release, which routes a small fraction of live decisions to the new model so that failures are contained to that fraction.

B. A backtest, which evaluates the new model on historical data and therefore carries the same limitations as any retrospective evaluation.

C. A shadow deployment, which measures the new model's behaviour on live traffic without exposing customers to its decisions, though it cannot measure outcomes that depend on acting differently.

D. An A/B test, which establishes the causal effect of the new model on downstream fraud losses by comparing two randomized treatment arms.

---

**Q78.** A trial records outcomes for two training methods:

| | Passed | Failed |
|---|---|---|
| Method A | 40 | 60 |
| Method B | 60 | 40 |

A chi-squared test of independence is performed. The critical value is 3.84 at α = 0.05 with 1 degree of freedom.

Which claim is supported?

A. The test is inapplicable here, since chi-squared requires expected cell counts below five and these counts are far larger.

B. The statistic is 8.0, so the association between method and outcome is significant at this threshold, though the test does not quantify the size of the effect.

C. The statistic is 2.0, which falls below the critical value, so no association between method and outcome is established.

D. The statistic is 8.0, which establishes that Method B causes a higher pass rate than Method A in this population.

---

**Q79.** A network takes input of shape `(N, 3, 64, 64)` and applies:

```
Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
MaxPool2d(2, stride=2)
Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
MaxPool2d(2, stride=2)
Conv2d(64, 128, kernel_size=k, stride=1, padding=0)
Flatten()
```

The flattened vector has length 25,088. What is `k`?

A. 3

B. 5

C. 7

D. 2

---

**Q80.** A pipeline segments red safety markings from factory photographs:

```python
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
```

Roughly half the markings are detected and the rest are missed, despite appearing identical to the eye.

Which claim explains this?

A. The saturation and value floors are too permissive, so the mask saturates with background pixels and the markings are lost within it.

B. Red has no stable representation in HSV and colour segmentation of red must be performed in the LAB colour space instead.

C. The conversion requires an RGB input, so passing a BGR array from `imread` shifts every hue by a fixed offset.

D. OpenCV represents hue on a 0 to 179 scale, so red straddles both ends of the range and a single low-end interval captures only part of it.

---

**Q81.** A checkpoint saved from a GPU training run is loaded on a CPU-only machine with `torch.load("ckpt.pt")` and fails.

Which claim is supported?

A. The failure arises from precision rather than device, so the tensors must be cast to 32-bit floating point before the file can be read.

B. The state dict must be re-saved with `_use_new_zipfile_serialization=False`, since the default format embeds CUDA kernels in the file.

C. The saved tensors carry their device placement, so loading requires `map_location="cpu"` to redirect them onto available hardware.

D. Checkpoints are device-specific binary formats and a GPU-saved file cannot be read on a CPU-only machine under any configuration.

---

**Q82.** An assistant drafts replies to customer complaints. The objective is to **minimize the rate at which the assistant commits the company to remedies it is not authorized to offer**, such as refunds or delivery date guarantees.

- P1: "Draft a reply to this complaint."
- P2: "Draft a reply to this complaint. Do not promise refunds, credits, replacements or delivery dates. Where the customer asks for one, say the request has been passed to the team that can authorize it."
- P3: "Draft a reply to this complaint. Be careful not to over-promise."
- P4: "Draft a reply to this complaint. Example — Customer asks for a refund; reply: 'I've escalated your refund request to our billing team, who will contact you within two working days.'"
- P5: "You are a cautious customer service representative. Draft a reply to this complaint."

Rank the prompts from most to least effective against this objective.

A. P2 > P4 > P5 > P3 > P1

B. P2 > P4 > P3 > P1 > P5

C. P4 > P2 > P3 > P1 > P5

D. P2 > P3 > P4 > P5 > P1

---

**Q83.** Two documents are represented as binary term vectors over the vocabulary `[data, model, train, test]`:

- `d1 = [1, 0, 1, 1]`
- `d2 = [1, 1, 0, 1]`

What is the cosine similarity between them?

A. 0.667

B. 0.750

C. 0.816

D. 0.500

---

**Q84.** A research group prepares to release open weights for a capable general-purpose model.

Which pre-release practice most directly addresses the risks specific to an irreversible open-weights release?

A. Structured red-teaming for dangerous capabilities before release, since safety mitigations applied to open weights can be removed by anyone and the release cannot be recalled.

B. Rate-limiting the download endpoint, which constrains the speed at which the weights propagate through the developer ecosystem.

C. Requiring users to accept an acceptable-use licence, which establishes the legal basis for pursuing downstream misuse after it occurs.

D. Publishing a model card describing intended use and known limitations, which informs downstream developers of the boundaries of appropriate application.

---

**Q85.** A team fine-tunes a base language model with supervised instruction data and then applies reinforcement learning from human feedback.

Which claim describes what each stage contributes?

A. Instruction tuning adds new factual knowledge to the model, while RLHF compresses that knowledge into fewer parameters for efficient serving.

B. Instruction tuning and RLHF optimize the same objective, differing only in whether the gradient is computed from labelled examples or from sampled rollouts.

C. RLHF must precede instruction tuning, since preference optimization requires a base model that has not yet been shaped by supervised examples.

D. Instruction tuning teaches the model to follow the form of a request, while RLHF optimizes against a learned preference model to shape which of several valid responses is preferred.

---

**Q86.** A screening model is deployed where a missed positive costs ₹500 and a false alarm costs ₹20. On a population of 10,000 with 200 true positives, three thresholds give:

| Threshold | True positives caught | False positives |
|---|---|---|
| A | 180 | 400 |
| B | 150 | 120 |
| C | 195 | 1,500 |

Which threshold minimizes expected cost?

A. Threshold C, at ₹32,500

B. All three are equivalent at approximately ₹25,000

C. Threshold A, at ₹18,000

D. Threshold B, at ₹27,400

---

**Q87.** A RAG system answers questions about a product's configuration options. Users phrase questions as problems — *"my exports keep timing out"* — while the documentation is written as feature descriptions — *"the batch_timeout parameter controls..."*. Retrieval consistently misses the relevant page.

Which change addresses the cause?

A. Re-embed the corpus with a larger model, since capacity is what determines whether problem phrasings and feature phrasings align.

B. Generate a hypothetical answer to the user's question and embed that for retrieval, so the query vector sits in the same region of the space as the documentation's phrasing.

C. Increase `top_k` so that the relevant page appears somewhere in a longer candidate list even if it ranks poorly.

D. Reduce chunk size so that each configuration parameter occupies its own chunk and becomes individually retrievable.

---

**Q88.** A team distinguishes two threats to a deployed image classifier: an attacker who modifies inputs at inference time to force a misclassification, and an attacker who contributes corrupted examples to the data used for the next retraining cycle.

Which claim correctly names and separates these?

A. The first is an evasion attack and the second a poisoning attack; evasion is addressed by adversarial training and input checks, poisoning by data provenance controls and training-set auditing.

B. The first is a poisoning attack and the second an evasion attack; both are addressed by the same adversarial training procedure applied at different stages.

C. Both are evasion attacks, differing only in whether the perturbation is applied before or after the model's parameters are fixed.

D. The first is a model-inversion attack and the second a membership-inference attack, both of which concern training-data confidentiality.

---

**Q89.** Two group means are reported with 95% confidence intervals: Group A [10.2, 14.8] and Group B [14.0, 18.6]. The intervals overlap.

Which claim is supported?

A. The overlap region [14.0, 14.8] contains the true value of both means with 95% probability.

B. The intervals cannot overlap if the groups differ, so the reported figures must contain a computational error.

C. Overlapping intervals establish that the two group means do not differ significantly at the 5% level.

D. Overlapping individual intervals do not by themselves establish non-significance; the confidence interval for the difference must be computed, and it may exclude zero.

---

**Q90.** A team records training loss across epochs for one run:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Training loss | 2.31 | 1.84 | 1.52 | 1.49 | 1.48 | 1.48 |
| Validation loss | 2.28 | 1.86 | 1.55 | 1.71 | 1.94 | 2.21 |

Which claim is supported and what should be done?

A. The model is underfitting, since the training loss has stopped improving before reaching a low value; increasing capacity would reduce both curves.

B. The validation set is unrepresentative, since a correctly constructed split cannot produce a validation loss that rises while the training loss falls.

C. Training and validation loss diverge after epoch 3, so the model begins fitting noise; early stopping at epoch 3 together with stronger regularization is the appropriate response.

D. The learning rate is too high, since the training loss has plateaued while the validation loss rises; reducing it would allow both to continue falling together.

---
# ANSWER KEY

| Q | Ans | Domain | Q | Ans | Domain | Q | Ans | Domain |
|---|-----|--------|---|-----|--------|---|-----|--------|
| 1 | D | Stats | 31 | B | ML | 61 | D | NLP |
| 2 | C | PyTorch | 32 | A | Prompts | 62 | C | Governance |
| 3 | B | NLP | 33 | D | RAI Tech | 63 | C | LLM |
| 4 | A | Governance | 34 | C | Stats | 64 | B | ML |
| 5 | D | Deep Learning | 35 | C | Deep Learning | 65 | A | Prompts |
| 6 | C | Vision | 36 | B | Vision | 66 | D | RAI Tech |
| 7 | C | RAG | 37 | A | PyTorch | 67 | C | Stats |
| 8 | B | ML | 38 | D | RAG | 68 | B | Deep Learning |
| 9 | A | LLM | 39 | C | NLP | 69 | A | Vision |
| 10 | D | Prompts | 40 | B | Governance | 70 | A | PyTorch |
| 11 | C | RAI Tech | 41 | A | LLM | 71 | D | RAG |
| 12 | B | Stats | 42 | A | ML | 72 | C | NLP |
| 13 | A | Deep Learning | 43 | D | Prompts | 73 | B | Governance |
| 14 | A | PyTorch | 44 | C | RAI Tech | 74 | A | LLM |
| 15 | D | Vision | 45 | B | Stats | 75 | D | ML |
| 16 | C | NLP | 46 | A | Deep Learning | 76 | C | Prompts |
| 17 | B | RAG | 47 | D | Vision | 77 | C | RAI Tech |
| 18 | A | Governance | 48 | C | PyTorch | 78 | B | Stats |
| 19 | D | LLM | 49 | C | RAG | 79 | A | Deep Learning |
| 20 | C | ML | 50 | B | NLP | 80 | D | Vision |
| 21 | C | Prompts | 51 | A | Governance | 81 | C | PyTorch |
| 22 | B | RAI Tech | 52 | D | LLM | 82 | B | Prompts |
| 23 | A | Stats | 53 | C | ML | 83 | A | NLP |
| 24 | D | Deep Learning | 54 | B | Prompts | 84 | A | Governance |
| 25 | C | Vision | 55 | A | RAI Tech | 85 | D | LLM |
| 26 | B | PyTorch | 56 | A | Stats | 86 | C | ML |
| 27 | A | RAG | 57 | D | Deep Learning | 87 | B | RAG |
| 28 | A | NLP | 58 | C | Vision | 88 | A | RAI Tech |
| 29 | D | Governance | 59 | B | PyTorch | 89 | D | Stats |
| 30 | C | LLM | 60 | A | RAG | 90 | C | Deep Learning |

# EXPLANATIONS

**Q1 — D.** With 4 groups of 6 runs, N = 24: df between = 3, df within = 24 − 4 = **20**, df total = **23**. The table reports 24 and 27, which correspond to 28 observations. The within-groups mean square is therefore computed against the wrong divisor and the F ratio is not trustworthy. Checking that df between + df within = df total = N − 1 is a two-second sanity check worth doing on every ANOVA table.

**Q2 — C.** `__getitem__` returns a tensor whose length is the review's word count, capped at 64 but not padded up to it. The default collate function calls `torch.stack`, which requires identical shapes, so the first batch containing two reviews of different lengths fails. The fix is a custom `collate_fn` using `pad_sequence`. The missing head parameters and absent `zero_grad` are genuine defects but neither raises an error.

**Q3 — B.** A TF-IDF column of exact zeros across every document means `idf = log(N/df) = log(1) = 0`, which happens precisely when `df = N` — the term appears in all five documents. A term appearing in *no* document would not be in the vocabulary at all, and stop-word removal would drop the column rather than zero it. This is TF-IDF working as designed: a term present everywhere cannot discriminate between documents.

**Q4 — A.** Selection rates 0.42 and 0.41 are near-identical, so demographic parity holds. But TPR differs by 23 points and FPR by 12, so equalized odds — which requires both to match across groups — fails badly. The two criteria are distinct and can be satisfied independently, which is why "violates both" is wrong. The practical reading: equal selection rates are being achieved by selecting the wrong people in group B.

**Q5 — D.** The correct cell has forget, input and output gates as **sigmoids** (they must lie in [0,1]) and the candidate as **tanh** (it supplies signed content). The cell update is **additive**: `c = f*c_prev + i*g` — that additive path is what preserves gradients across long sequences. The hidden state is `o * tanh(c)`, with the tanh applied to the cell state *inside* the product. The distractors break exactly one of these: a multiplicative cell update, swapped non-linearities, and tanh applied to the gate rather than the cell.

**Q6 — C.** The comparison frame is updated at line 22, *after* the difference is computed — which is correct. The real defect is that `prev` is never blurred while `gray` is not either, so sensor noise at every pixel survives the threshold of 25; after two dilations those specks merge into contours exceeding 500 px. Gaussian blurring both frames before differencing is the standard remedy and is what makes frame differencing usable on real video. Moving line 22 above line 15 would compare a frame against itself and report no motion ever.

**Q7 — C.** Available context per configuration is 8,192 minus the system prompt minus the answer reservation. W: 8,192 − 600 − 1,200 = 6,392 available against 800 × 8 = **6,400 requested — it overflows by 8 tokens**. X requests 6,144 against 6,392. Y requests 4,500 against 4,792. Z requests 4,900 against 5,592. Only W exceeds, and it does so by a margin small enough that it would pass casual inspection.

**Q8 — B.** F1 = 2PR/(P+R). T1: 2(0.9)(0.4)/1.3 = **0.554**. T2: 2(0.6)(0.8)/1.4 = **0.686**. T3: 2(0.5)(0.9)/1.4 = **0.643**. T2 maximizes F1. The caveat matters as much as the arithmetic: F1 weights the two errors equally, which is rarely what a deployment wants — the threshold should follow the relative cost of a false negative against a false positive.

**Q9 — A.** Position 1 attends only to itself, so after masking its row is a distribution placing all weight on position 1 and its output is V's first row, **[2.0, 4.0]**. Position 2 attends to both with equal scores of 0.0, so softmax gives 0.5 and 0.5, and its output is the elementwise mean of V's rows: **[(2+6)/2, (4+8)/2] = [4.0, 6.0]**. Equal scores producing a uniform average is the detail most easily missed.

**Q10 — D.** This is the case where step-by-step reasoning genuinely wins: multi-step arithmetic benefits from the model externalizing intermediate values rather than producing a single token. P2 therefore leads. P4's "be careful" is a weak nudge but on-topic. P1 is neutral. P5's persona adds no computational scaffolding. **P3 ranks last** and is the trap — demanding only the final number actively suppresses the reasoning that produces accuracy. Note this inverts the usual ranking, where terse output constraints help: what helps depends on the objective.

**Q11 — C.** Shapley values attribute a feature's contribution *in the context of the other features present*, so identical values legitimately receive different attributions when they interact differently with the rest of the profile — an income of ₹80,000 may be strong alongside low debt and weak alongside heavy obligations. This is a property of the method, not a convergence failure or an inconsistency.

**Q12 — B.** In a randomized block design each farm receives every fertilizer, so farm-to-farm variation can be estimated and removed from the residual. The error mean square shrinks, the F ratio for fertilizer rises, and power increases — the cost is 7 degrees of freedom spent on blocks, which is repaid when blocks genuinely differ. Blocking does not require partial treatment assignment; that describes incomplete block designs.

**Q13 — A.** The reset gate multiplies the previous hidden state *before* the candidate transform: `n_t = tanh(W [r_t ⊙ h_prev, x_t])`. With `r_t ≈ 0` the previous state is suppressed, so the candidate depends almost entirely on the current input — the mechanism by which a GRU can discard accumulated context at a sequence boundary. What gets *carried forward* is governed by the update gate, not the reset gate.

**Q14 — A.** For input of shape `(N, C, d1)`, `CrossEntropyLoss` expects the class dimension in position 1 and a target of shape `(N, d1)` — one label per position. Here the logits are `(8, 5, 12)`, so C = 5 and the sequence dimension is 12, giving a required target of `[8, 12]`. The error message reports the size it expected, `[8, 5]`, based on a different reading of the axes, which is why the message itself is a poor guide to the fix.

**Q15 — D.** Vertical edges become visible while horizontal edges vanish, so the kernel differences along the **horizontal** axis — left column against right column. That is the Sobel-x kernel with rows `[-1,0,1]`, `[-2,0,2]`, `[-1,0,1]`. Option A is Sobel-y and would show the opposite pattern. The sharpening and box kernels would leave the interior non-zero rather than near zero, which the stem rules out.

**Q16 — C.** With equal priors the comparison reduces to the likelihood products: spam gives 0.06 × 0.05 = 0.0030, ham gives 0.01 × 0.02 = 0.0002. The ratio is **15**, so spam is favoured by a factor of 15 and the document is classified as spam. The per-term ratios are 6 and 2.5; their product is 15, so neither term is solely responsible — which rules out the option attributing it to `cheap` alone.

**Q17 — B.** MRR is the mean of the reciprocal of the first relevant rank: (1/1 + 1/3 + 1/5)/3 = (1 + 0.3333 + 0.2)/3 = 1.5333/3 = **0.511**. MRR captures only where the *first* relevant result lands, which makes it suitable for known-item search and poorly suited to tasks needing several relevant results.

**Q18 — A.** A spam filter is **minimal risk** — the Act's recitals use it as the canonical example. Scoring applicants for vocational training is **high risk**, since access to education and vocational training is a listed category. Inferring emotional states of students during examinations is **prohibited**: emotion recognition in education and the workplace sits in the unacceptable-risk tier. The trap is assuming all three regulated systems fall into one band.

**Q19 — D.** Attention uses four `d_model × d_model` matrices: 4 × 512² = 1,048,576. The feed-forward uses two matrices of 512 × 2,048: 2 × 1,048,576 = 2,097,152. Total = **3,145,728**, which is 12·d² — a useful shortcut, since attention contributes 4d² and a 4× feed-forward contributes 8d².

**Q20 — C.** `center=True` places the window symmetrically, so each row's average spans roughly 45 days before *and* 45 days after that row's date. Every feature value therefore contains future spending that would not exist at scoring time — textbook temporal leakage, and it inflates cross-validated AUC while contributing nothing in production. The fix is a trailing window, `rolling(90).mean()` with `closed="left"`.

**Q21 — C.** P2 states all three rules explicitly, including a concrete list of banned superlatives, so each is independently checkable. P4's example demonstrates the style but never states the word range or the superlative prohibition, so compliance with two of the three rules is left to inference. P3 references a house style the model cannot see, P1 is neutral, and P5's persona plus "brand voice" supplies nothing verifiable.

**Q22 — B.** Surrogate R² measures **fidelity** — how well the surrogate reproduces the *ensemble's outputs*, not the data's labels. At 0.62 the surrogate disagrees with the ensemble across much of the input space, so any rule extracted from it may describe behaviour the ensemble does not have. Presenting such rules as explanations is the failure mode; high fidelity is the precondition for a global surrogate to be usable at all.

**Q23 — A.** logit = −3.2 + 0.045(40) + 1.1(1) − 0.6(2) = −3.2 + 1.8 + 1.1 − 1.2 = **−1.5**. p = 1/(1 + e^1.5) = 1/(1 + 4.4817) = **0.182**. The common slips are dropping a term's sign and forgetting that the linear predictor lives on the log-odds scale, so it must be pushed through the logistic function rather than read as a probability.

**Q24 — D.** `nn.LSTM(300, 128)` declares an input feature width of 300, and the error reports that the last dimension actually arriving is 64 — the embedding dimension. The two must agree. The message names `input.size(-1)`, which is the feature axis regardless of `batch_first`, so neither batch size nor sequence length is implicated; hidden size is independent of input size.

**Q25 — C.** Lower thresholds admit more edge pixels — long contours plus interior texture — while higher thresholds admit only the strongest gradients and leave short fragments at object boundaries. So Output 1 corresponds to (50, 150) and Output 2 to (150, 250). A larger Gaussian kernel would reduce detail in Output 1 rather than increase it, and non-maximum suppression is not an optional stage of Canny.

**Q26 — B.** Both `detach()` and `data` return tensors sharing storage and excluded from autograd, so their **values** are equal — the difference is that `data` bypasses autograd's correctness checks and can silently corrupt gradients, which is why `detach()` is preferred. `clone()` remains attached to the graph; `requires_grad_(False)` mutates the tensor in place rather than returning a view; `numpy()` returns an array, not a tensor, and fails outright on a tensor requiring grad.

**Q27 — A.** BM25 spans a range of roughly 27 units while cosine spans roughly 0.27 — two orders of magnitude apart — so a raw sum is effectively the BM25 ranking with rounding noise. The fix is to remove the scale dependence: rank-based fusion such as RRF, or per-retriever normalization before weighted combination. Applying a fractional power is an arbitrary patch that does not make the scales commensurable.

**Q28 — A.** Characters `l o w e s t`. Merge 1 gives `lo w e s t`; merge 2 gives `low e s t`; merge 3 requires an adjacent `e r` and no `r` is present; merge 4 requires `low er`. Final tokens: `low`, `e`, `s`, `t` = **4**. The word `lower` would collapse to a single token, which is the contrast the question is built on: BPE handles unseen words by decomposing them into learned pieces rather than emitting an unknown token.

**Q29 — D.** The Act requires providers of high-risk systems to operate a post-market monitoring system that collects and analyses performance data across the product's lifetime, with reporting duties for serious incidents. A divergence the conformity assessment did not anticipate is precisely what that obligation exists to surface. Transparency is a separate, lesser duty; nothing here triggers prohibition.

**Q30 — C.** Cache size = 2 (keys and values) × layers × sequence length × d_model × bytes per element = 2 × 32 × 2,048 × 4,096 × 2 = 1,073,741,824 bytes = **1 GiB** — for a single sequence. Multiplying by batch size is what makes KV cache the binding memory constraint in serving, and is the pressure that motivates multi-query and grouped-query attention.

**Q31 — B.** Choosing the best configuration on a fold and then reporting that same fold's score uses one dataset for both selection and evaluation, so the reported figure is optimistically biased — it captures the maximum over configurations, including whatever noise favoured the winner. Nested cross-validation selects within an inner loop and evaluates on an untouched outer fold. Stratification addresses class balance and does nothing about this.

**Q32 — A.** The user supplied only an email, so the identifier must be discovered before the detail call: `search_orders` first, then select the most recent id, then `get_order`. Generating an identifier is the hallucinated-argument failure mode. Parallel calls are impossible here because the second call's argument is the first call's output — a genuine sequential dependency.

**Q33 — D.** Each weekly window is compared against the training reference and moves too little to register, but the drift accumulates: a month's worth of small movements aggregates into PSI above 0.3. Gradual drift is precisely what a short window against a fixed baseline misses, which is why production monitoring runs both a short window for sudden shifts and a long-horizon comparison for creeping ones. PSI does not grow mechanically with sample size.

**Q34 — C.** Joint probabilities: 0.5 × 0.1 = 0.05, 0.3 × 0.4 = 0.12, 0.2 × 0.6 = 0.12, summing to 0.29. Posteriors: 0.05/0.29 = **0.172**, 0.12/0.29 = **0.414**, 0.12/0.29 = **0.414**. The second and third conditions tie despite quite different priors, because the likelihood ratio exactly offsets the prior ratio. Note that the joint probabilities are not posteriors — they must be normalized by their sum.

**Q35 — C.** Variance contracts by roughly a factor of three every four layers, so the initialization is too small for ReLU, which discards about half the variance passing through it. He initialization compensates with a standard deviation of √(2/fan_in). Left uncorrected, activations and then gradients vanish with depth. Dead units would show activations pinned at exactly zero, not a smooth geometric decay.

**Q36 — B.** Patch count scales with area: (224/16)² = 196 rises to (448/16)² = **784**. The patch projection maps a 16×16×3 patch to the embedding dimension and is therefore unaffected by image size. What must change is the learned positional embedding table, which has 196 entries and must be interpolated onto the new 28×28 grid — the standard procedure when fine-tuning a ViT at higher resolution.

**Q37 — A.** `detach()` returns a new tensor that shares `y`'s values but sits outside the autograd graph, leaving `y` itself fully differentiable. `no_grad()` affects only operations executed inside its scope and does not detach `y`. `requires_grad_(False)` mutates `y` in place, breaking the backward pass. `clone()` is a differentiable operation — the clone remains attached to the graph.

**Q38 — D.** The question requires two retrieval hops — resolve Pune's vendor, then find other sites using that vendor — and no single query embedding sits near both documents. Increasing k or chunk size does not create the missing intermediate step. Query decomposition issues one retrieval per hop and feeds the first result into the second, which is the standard remedy for multi-hop failures.

**Q39 — C.** Negative sampling updates the output vector for the observed context word plus k sampled negatives: 1 + 5 = **6** vectors, instead of all 50,000 that a full softmax normalization would touch. This is the approximation that makes the objective tractable at scale, and it converts the task from a 50,000-way classification into six independent binary ones.

**Q40 — B.** Each output is reviewed by a clinician who must act before anything follows, and who may override the score — that is human-in-the-loop. Human-on-the-loop describes a system that acts autonomously under supervision, and human-in-command describes oversight of operating conditions rather than individual outputs. The distinction matters because the EU AI Act's oversight requirements are specified in these terms.

**Q41 — A.** Restricting each token's attention to a fixed window of 512 makes cost linear in sequence length rather than quadratic. The price is that a single layer cannot relate distant tokens directly; information propagates across windows layer by layer, so the effective receptive field grows with depth. Sliding-window attention is entirely compatible with causal masking — the window is simply applied to the already-masked positions.

**Q42 — A.** Boosting adds each tree's prediction scaled by the learning rate. After tree 1: 10.0 + 0.1 × (16.0 − 10.0) = **10.60**. The remaining residual is 16.0 − 10.60 = 5.40, so after tree 2: 10.60 + 0.1 × 5.40 = **11.14**. The slow approach to the target is the point — a small learning rate requires many trees but generalizes better, which is the central trade-off in boosting.

**Q43 — D.** The recall of correct resolutions turns on handling under-specified messages, and P2 states the trigger condition and the required action precisely: ask one clarifying question when product and symptom are both absent. P4's example demonstrates the behaviour on one case without generalizing the rule. P3 permits clarifying questions but supplies no trigger, so behaviour stays inconsistent. P1 and P5 address neither.

**Q44 — C.** Group A is underconfident and group B overconfident; averaged over both, the errors cancel and the aggregate curve looks calibrated. This is why calibration must be assessed within groups, not only overall — the aggregate figure conceals two failures of opposite sign. Applying temperature scaling to one group alone is a partial remedy at best and raises its own fairness questions.

**Q45 — B.** The coefficient moves from 4.82 with a standard error of 6.31 — an estimate indistinguishable from zero — to 1.93 with a standard error of 0.24, while held-out RMSE is unchanged at 3.14 against 3.15. That is the signature of collinearity: coefficient estimates become unstable while predictions remain intact, because the linear combination is well determined even though its decomposition is not. Collinearity is an inference problem, not a prediction problem.

**Q46 — A.** Dropout has different output variance in training and inference. The batch normalization layer downstream estimates its running statistics under training-time dropout, then at inference encounters activations with different variance because dropout is off — so the normalization is applied with mismatched statistics. This is the documented variance-shift problem, and the usual remedies are placing dropout after normalization or using them in separate blocks.

**Q47 — D.** Small rotations, brightness changes and mild blur all preserve the content of a document scan and help modestly. Horizontal flipping produces mirrored text, which is not a valid instance of the class the label asserts — the model is trained on corrupted targets, and the 7.4-point drop follows. Augmentation validity is a property of the data, not of the transform: the same flip is harmless on natural-scene photographs.

**Q48 — C.** Because `.grad` accumulates by default, the correct pattern divides the loss by the accumulation count so the summed gradient equals the mean over the full effective batch, calls `backward()` every micro-batch, and calls `step()` and `zero_grad()` only every fourth. Option B zeroes every iteration, destroying the accumulation. Option D holds four graphs alive and fails on the second backward. Option A is correct in structure but is not the option shown as such here — check the ordering carefully.

**Q49 — C.** The budget is 400 ms. Option L at 430 ms and M at 860 ms both breach it, so the choice is between J and K. K delivers 0.79 against J's 0.61 — most of the total achievable gain, since L and M add only 0.05 and 0.07 beyond it — within 180 ms. Constraint-bounded selection is the skill being tested: the highest-recall option is not available.

**Q50 — B.** Precision = 7/12 = 0.583, recall = 7/10 = 0.700. F1 = 2(0.583)(0.700)/(0.583 + 0.700) = 0.8167/1.2833 = **0.636**. Entity-level scoring requires both the span boundaries and the type to match exactly, which is far stricter than token-level accuracy and is the reason token accuracy overstates NER performance.

**Q51 — A.** Public accessibility is not a licence. Web pages carry copyright by default and may carry terms restricting reuse, and scraped text routinely contains personal data engaging separate obligations. Documenting source, licence terms and personal-data content is what allows a lawful-basis assessment to be made at all — and it is exactly what a datasheet for the dataset is designed to record.

**Q52 — D.** A learned absolute embedding table has one row per position up to its trained maximum, so position 4,097 simply has no entry and the model cannot process the input without interpolation or extension. Rotary embeddings are computed as a function of the index, so they are *defined* at any position — but quality still degrades well beyond the trained range, which is why long-context work still requires explicit scaling techniques.

**Q53 — C.** Parent Gini = 1 − 0.5² − 0.5² = 0.5. Left child (45/60 positive): 1 − 0.75² − 0.25² = 0.375. Right child (5/40 positive): 1 − 0.125² − 0.875² = 0.21875. Weighted child impurity = 0.6(0.375) + 0.4(0.21875) = 0.225 + 0.0875 = 0.3125. Decrease = 0.5 − 0.3125 = **0.1875**. The weighting by child size is the step most often skipped.

**Q54 — B.** P2 replaces holistic judgement with a rubric scored criterion by criterion, requires evidence from the answer for each score, and explicitly excludes the attributes at issue — three mechanisms working together, and the evidence requirement is what makes a biased score visible. P4's example shows the rubric format on one criterion without excluding the confounds. P3 and P5 assert impartiality without any mechanism that produces it.

**Q55 — A.** Ensemble disagreement is a practical proxy for **epistemic** uncertainty: five independently trained models converging on class 3 indicates the input is well covered by the training distribution, while a 2-2-1 split indicates it is not. Crucially the mean confidence is nearly identical for both inputs, so a single model's softmax would have missed this entirely — which is the argument for ensembles in uncertainty estimation.

**Q56 — A.** PC1 loads positively and substantially on income, savings and age, so it represents a shared financial-capacity dimension; commute distance loads only −0.33 there but **0.93** on PC2, which is therefore dominated by it. Reading the largest single loading in the table as belonging to PC1 is the error the layout invites. Components are orthogonal by construction, and a component dominated by one feature still contributes a distinct direction.

**Q57 — D.** Forward: h = 3(2) = 6, a = ReLU(6) = 6, o = −1(6) = −6, L = (−6 − (−4))² = 4. Backward: dL/do = 2(o − y) = 2(−2) = −4; dL/da = dL/do × w2 = −4 × −1 = 4; ReLU passes gradient unchanged since h > 0, so dL/dh = 4; dL/dw1 = dL/dh × x = 4 × 2 = **8**. The sign flips twice, which is where this most often goes wrong.

**Q58 — C.** Global equalization applies one intensity mapping to the whole image, so making the dark corners legible necessarily pushes the already-bright regions into saturation. CLAHE equalizes within local tiles — so each region gets its own mapping — and clips the histogram before redistribution to cap the amplification, which is what prevents blow-out. Equalizing hue would corrupt colour rather than brightness.

**Q59 — B.** `gather(1, idx)` returns a tensor with the **same shape as `idx`**, here `(4, 2)`, where output position `[i][j]` takes `x[i][idx[i][j]]`. So row 0 collects columns 0 and 2 of x's row 0, row 1 collects columns 1 and 3 of row 1, and so on. The index tensor need not match the source's shape — only its rank — which disposes of the error option.

**Q60 — A.** With post-filtering, a tenant holding a small share of the index may have few or none of its chunks among the global top 100, so the filter can return fewer results than requested — and the similarity computation spent on other tenants is wasted. Pre-filtering restricts the search space before ranking. Modern vector stores support filtered ANN search directly, so the claim that pre-filtering invalidates the index is false.

**Q61 — D.** Stemming maps inflected forms — `train`, `training`, `trained` — onto a shared stem, so distinct surface forms collapse into one feature and the vocabulary **shrinks**. The trade-off is lost distinctions, since stemming is aggressive and can conflate words with different meanings. It never adds forms, which is why the larger-vocabulary options are wrong.

**Q62 — C.** Under basic sequential composition, privacy budgets **add**: 3 × 0.5 = **ε = 1.5**. This is why a differentially private system must maintain a budget ledger and refuse queries once the total is exhausted — privacy loss is cumulative over everything ever released about the dataset. Advanced composition gives a tighter bound than simple addition but the loss still grows with query count.

**Q63 — C.** At 16-bit precision each parameter occupies 2 bytes: 7 × 10⁹ × 2 = **14 GB**. At 4-bit, half a byte each: 7 × 10⁹ × 0.5 = **3.5 GB**. The fourfold reduction is what brings a 7B model within reach of a single consumer GPU, and combining it with LoRA adapters — quantized base, small trainable adapters — is the QLoRA recipe.

**Q64 — B.** The L2 penalty sums squared coefficient magnitudes, and a coefficient's magnitude depends on its feature's units: income measured in rupees produces a tiny coefficient that the penalty barely touches, while a feature on a small scale carries a large coefficient and absorbs most of the shrinkage. Regularized regression is therefore not scale-invariant and standardization is required — unlike ordinary least squares, where rescaling merely rescales the coefficient.

**Q65 — A.** The problem is unmanaged context accumulation, not window size — a larger window postpones the failure and multiplies cost without fixing it. The remedy is an explicit working state that the agent updates: decisions made, facts established, and tool results written in once, so the context holds a compact representation rather than an append-only log of raw outputs.

**Q66 — D.** If an explanation is unchanged after the model's learned weights are randomized, it cannot be reflecting anything the model learned — it is responding to the input's structure, typically its edges. This is the model-randomization sanity check, and methods failing it produce plausible-looking maps that carry no information about the model's reasoning. Stability under perturbation is desirable only when the perturbation should not matter.

**Q67 — C.** `tenure_months` has a t of 5.25 and p < 0.001 — estimated precisely and clearly non-zero. `n_tickets` has a coefficient of 0.310 with a standard error of 0.290, so it cannot be distinguished from zero; that is a statement about precision, not proof of no relationship. And R² = 0.14 means both predictors together leave 86% of the variance unexplained. Comparing raw coefficient magnitudes across predictors on different scales is meaningless.

**Q68 — B.** The identity path contributes a gradient term of exactly 1 to the Jacobian, so gradient reaching a deep block also reaches earlier ones without having to survive multiplication through every intervening transformation. That is what makes networks of hundreds of layers trainable. The non-linearities remain within F(x), and gradients do traverse the whole stack — just with a path that does not attenuate.

**Q69 — A.** Precision is computed at each true positive in the ranking: rank 1 → 1/1 = 1.000; rank 3 → 2/3 = 0.667; rank 5 → 3/5 = 0.600. Average = (1.000 + 0.667 + 0.600)/3 = **0.756**. mAP is this quantity averaged across classes. Computing precision only at the final rank, or averaging over all five positions rather than the true positives, are the two usual errors.

**Q70 — A.** `class_weight` multiplies each sample's loss contribution by its class weight, so minority errors carry thirty times the gradient signal — the decision boundary shifts without any change to the dataset. It does not duplicate samples (that is oversampling, which changes epoch length), does not touch features, and does affect training rather than only reporting.

**Q71 — D.** Clauses are self-contained semantic units of varying length, so splitting on clause boundaries means every chunk is a complete statement — truncation falls from 31% to 2% and recall rises to 0.86. Overlap merely reduces the damage from boundaries that should not have been placed there. Raising the fixed size to 900 would not reproduce the result, since boundaries would still land arbitrarily within clauses.

**Q72 — C.** Tokens: `the cat sat on the mat the cat ran`. Count(`the`) = 3; count(`the cat`) = 2. Distinct vocabulary = {the, cat, sat, on, mat, ran} = 6. Add-one smoothing: (2 + 1)/(3 + 6) = 3/9 = **1/3**. The vocabulary size is added to the denominator so the smoothed conditional distribution still sums to one over all possible next words.

**Q73 — B.** A model's parameters can encode information about individuals in the training data — memorization and membership-inference attacks demonstrate this — so deleting source records does not automatically end the processing. The organization must assess whether continued use constitutes processing of that person's data and consider retraining or machine unlearning. Neither blanket exemption nor automatic deletion is the correct answer; it is an assessment obligation.

**Q74 — A.** Greedy decoding, equivalently temperature 0, makes generation deterministic for a given input, which is what reproducibility requires; the cost is output diversity, which this task does not want. Nucleus sampling remains stochastic at any `top_p` above zero. Beam search is deterministic but returns the highest-scoring beam, not a global optimum. A fixed seed does not survive changes in batching or hardware across replicas.

**Q75 — D.** The data has weekly seasonality, annual seasonality, a trend, and a known future driver — a time-series model with explicit seasonal and trend components plus the promotional calendar as an exogenous regressor uses all four. Clustering discards the temporal structure, thresholding at the median throws away the quantity being forecast, and shuffling daily records for a random forest destroys the ordering the task depends on.

**Q76 — C.** Stability across model versions comes from leaving as little to the model's discretion as possible: P2 names every field, fixes the type and format of each, mandates exactly four keys, and specifies the behaviour for missing values. P4's example implies the schema but does not constrain it, so a different model may add or rename fields. P3 names the format only; P1 and P5 leave the field set to interpretation.

**Q77 — C.** Running the candidate model on live traffic while only the incumbent's decisions take effect is a shadow deployment. It measures the new model's scores under real conditions without customer risk — but because its decisions are never enacted, outcomes that depend on acting differently cannot be observed, so it cannot establish causal effect. A canary routes a fraction of real decisions; an A/B test randomizes them.

**Q78 — B.** Row and column totals are all 100 and 200 overall, so every expected count is (100 × 100)/200 = 50. Each cell contributes (|40 − 50|)²/50 = 2.0, and there are four cells, giving χ² = **8.0** > 3.84 — significant at the 5% level. Significance establishes association, not causation, and says nothing about effect size; that would need a measure such as the odds ratio or Cramér's V.

**Q79 — A.** Trace forward: 64 → conv(k=3, p=1) → 64 → pool → 32 → conv(k=3, p=1) → 32 → pool → **16**. Then the flattened length 25,088 divided by 128 channels gives 196 = 14², so the final convolution must map 16 to 14, requiring 16 − k + 1 = 14, hence **k = 3**. Working backwards from the flatten size is the shape this question tests, and it is considerably slower than forward tracing.

**Q80 — D.** OpenCV packs hue into 0–179 to fit a uint8, so the hue circle wraps and red occupies bands at **both** ends — roughly 0–10 and roughly 170–179. A single low-end `inRange` captures only one of them, which is why about half the markings are found. The standard fix builds two masks and combines them with `cv2.bitwise_or`. `COLOR_BGR2HSV` is the correct conversion for an array from `imread`.

**Q81 — C.** Saved tensors record the device they were on, so loading a CUDA checkpoint on a machine without CUDA fails unless the tensors are redirected with `map_location="cpu"` (or `map_location=torch.device("cpu")`). The file format itself is portable — nothing device-specific is embedded beyond that placement metadata, and precision is unrelated.

**Q82 — B.** P2 enumerates the prohibited commitments and supplies the replacement behaviour, so the model has both a rule and something to say instead — the combination is what prevents it from improvising. P4's example demonstrates the replacement for refunds alone and leaves delivery dates and credits unaddressed. P3's "don't over-promise" has no definition of the boundary, and P1 and P5 provide neither rule nor substitute.

**Q83 — A.** Dot product = (1)(1) + (0)(1) + (1)(0) + (1)(1) = 2. Norms: |d1| = √3, |d2| = √3. Cosine = 2/(√3 × √3) = 2/3 = **0.667**. For binary vectors the cosine reduces to shared terms divided by the geometric mean of the document lengths, which is why it normalizes away differences in document size in a way raw overlap counts do not.

**Q84 — A.** Open weights cannot be recalled, and any safety mitigation applied to them can be fine-tuned away by whoever holds the file — so the only intervention that reliably reduces risk is the one performed *before* release: structured evaluation for dangerous capabilities, which informs whether and in what form to release at all. Model cards, rate limits and licences all assume ongoing control the releaser does not retain.

**Q85 — D.** Instruction tuning uses supervised examples to teach the model to respond in the *form* a request calls for. RLHF then optimizes against a reward model learned from human preference comparisons, shaping *which* of several formally valid responses is preferred — helpfulness, tone, refusal behaviour. Neither adds factual knowledge in any substantial way, and the order cannot be reversed, since preference optimization needs a model that already follows instructions.

**Q86 — C.** Expected cost = (missed positives × 500) + (false positives × 20). A: 20 missed × 500 + 400 × 20 = 10,000 + 8,000 = **₹18,000**. B: 50 × 500 + 120 × 20 = 25,000 + 2,400 = ₹27,400. C: 5 × 500 + 1,500 × 20 = 2,500 + 30,000 = ₹32,500. A is cheapest. Neither the highest-recall nor the highest-precision threshold wins — the optimum sits between them, which is why asymmetric costs must be made explicit rather than left to F1.

**Q87 — B.** The query and the documentation occupy different regions of the embedding space because they are written in different registers — symptom language against feature language. HyDE generates a hypothetical answer to the question and embeds *that*, which is phrased like documentation and therefore lands near the right passage. Retrieving more candidates or re-chunking does not change the register mismatch.

**Q88 — A.** Modifying inputs at inference to force a misclassification is an **evasion** attack; contributing corrupted examples to the training set is a **poisoning** attack. They occur at different stages and need different controls: adversarial training and input validation for evasion, data provenance and training-set auditing for poisoning. Model inversion and membership inference are confidentiality attacks, a separate category again.

**Q89 — D.** Overlapping confidence intervals do **not** imply a non-significant difference. The standard error of a difference is smaller than the sum of the individual standard errors, so the interval for the difference can exclude zero even when the individual intervals overlap substantially. The correct procedure is to compute the interval for the difference directly. Overlap does imply non-significance only in the reverse direction and only for non-overlap.

**Q90 — C.** Training loss continues to fall and then flattens while validation loss reaches its minimum at epoch 3 and rises steadily thereafter — the classic overfitting signature, and the divergence point identifies where to stop. Early stopping at epoch 3 plus stronger regularization is the response. An excessive learning rate would show instability in the training loss itself, and underfitting would show both curves high and falling together.

---

# SCORING AND DIAGNOSIS

| Domain | Questions | Max |
|---|---|---|
| Statistics | 1, 12, 23, 34, 45, 56, 67, 78, 89 | 9 |
| Deep Learning | 5, 13, 24, 35, 46, 57, 68, 79, 90 | 9 |
| PyTorch / TF | 2, 14, 26, 37, 48, 59, 70, 81 | 8 |
| Computer Vision | 6, 15, 25, 36, 47, 58, 69, 80 | 8 |
| RAG | 7, 17, 27, 38, 49, 60, 71, 87 | 8 |
| NLP | 3, 16, 28, 39, 50, 61, 72, 83 | 8 |
| Governance | 4, 18, 29, 40, 51, 62, 73, 84 | 8 |
| LLM | 9, 19, 30, 41, 52, 63, 74, 85 | 8 |
| ML Fundamentals | 8, 20, 31, 42, 53, 64, 75, 86 | 8 |
| Prompts / Agents | 10, 21, 32, 43, 54, 65, 76, 82 | 8 |
| RAI Technical | 11, 22, 33, 44, 55, 66, 77, 88 | 8 |

Pacing checkpoints: Q30 by 50 minutes, Q60 by 100 minutes, Q90 by 150.

Deliberately expensive items to flag on sight: Q2 and Q6 (extended code), Q7 (four-way budget arithmetic), Q34 (three-hypothesis posterior), Q57 (backward pass by hand), Q69 (average precision), Q79 (reverse dimension trace), Q86 (three-way expected cost).
