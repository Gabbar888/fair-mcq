# AI Assessment — Mock 03
## 90 questions | 150 minutes | MCQ, single correct

Domains are interleaved and unlabelled. Target pace is 100 seconds per question; several items are deliberately expensive, so flag and return rather than stalling.

---

**Q1.** A one-way ANOVA across 4 groups returns a significant omnibus F. The analyst then runs every pairwise t-test at α = 0.05 to find which groups differ.

Which claim is supported?

A. Six pairwise tests are performed, but because the omnibus F was significant the family-wise error rate is already controlled and each test may use α = 0.05.

B. Twelve pairwise tests are performed, since each ordered pair must be tested separately, and the family-wise error rate rises to roughly 46%.

C. Six pairwise tests are performed, and if no true differences existed the probability of at least one spurious significant result is roughly 26%, so the per-test level must be adjusted.

D. Four pairwise tests are performed, one per group, and since the omnibus test already reached significance no further adjustment to the per-test level is required.

---

**Q2.** The script below terminates with a runtime error partway through the first epoch.

```python
 1   device = "cuda" if torch.cuda.is_available() else "cpu"
 2
 3   class TwoTower(nn.Module):
 4       def __init__(self, n_user, n_item, dim=64):
 5           super().__init__()
 6           self.user = nn.Embedding(n_user, dim)
 7           self.item = nn.Embedding(n_item, dim)
 8           self.bias = torch.zeros(n_item)
 9       def forward(self, u, i):
10           dot = (self.user(u) * self.item(i)).sum(dim=1)
11           return dot + self.bias[i]
12
13   model = TwoTower(10_000, 5_000).to(device)
14   opt   = optim.Adam(model.parameters(), lr=1e-3)
15   crit  = nn.BCEWithLogitsLoss()
16
17   for u, i, y in loader:
18       u, i, y = u.to(device), i.to(device), y.to(device).float()
19       opt.zero_grad()
20       loss = crit(model(u, i), y)
21       loss.backward()
22       opt.step()
```

Which defect is responsible for the failure?

A. `BCEWithLogitsLoss` cannot accept a one-dimensional prediction tensor, so the dot product must be reshaped before the loss is computed.

B. The item bias is a plain tensor rather than a registered parameter or buffer, so `.to(device)` leaves it on the CPU while its index arrives on the GPU.

C. The two embedding tables have different row counts, so multiplying their outputs elementwise produces a shape mismatch at line 10.

D. The optimizer is constructed before the model is moved to the device, so its parameter references point at tensors that no longer exist.

---

**Q3.** TF-IDF weights are computed with `idf(t) = log10(N / df(t))` over 100 documents. In document D, the term `latency` occurs 6 times out of 200 tokens and appears in 10 of the 100 documents. Term frequency is used as the raw count.

What is the TF-IDF weight of `latency` in D?

A. 6.00

B. 0.30

C. 0.06

D. 1.00

---

**Q4.** A lender removes postcode from its model after an audit. Approval rates by ethnic group barely change. Investigation shows the model still uses `distance_to_nearest_branch` and `local_area_default_rate`, both of which are computed from the applicant's address.

Which claim is supported?

A. Removing a feature cannot reduce disparity in any model, since the remaining features are refitted and absorb the removed feature's predictive content exactly.

B. The remaining features act as proxies for the removed attribute, so the disparity persists; proxy auditing must examine features correlated with location rather than only the location field itself.

C. The audit is inconclusive because approval rates are the wrong measure, and only a change in the model's internal coefficients could demonstrate that bias was removed.

D. The persistence proves the original disparity was unrelated to geography, so the postcode removal was unnecessary and should be reversed.

---

**Q5.** A layer is constructed as `nn.LSTM(input_size=100, hidden_size=256, num_layers=1)`.

Counting four gates, each with an input-to-hidden matrix, a hidden-to-hidden matrix and a single bias vector, how many parameters does the layer hold?

A. 262,144

B. 102,400

C. 91,392

D. 365,568

---

**Q6.** A document-scanning service straightens photographed pages before OCR. It runs without error, but output pages are consistently cropped, with the right and bottom edges of the document missing.

```python
 1   import cv2, numpy as np
 2
 3   def deskew(img, corners):
 4       # corners: 4x2 float32, ordered tl, tr, br, bl
 5       (tl, tr, br, bl) = corners
 6       wA = np.linalg.norm(br - bl)
 7       wB = np.linalg.norm(tr - tl)
 8       hA = np.linalg.norm(tr - br)
 9       hB = np.linalg.norm(tl - bl)
10       W  = int(max(wA, wB))
11       H  = int(max(hA, hB))
12       dst = np.array([[0, 0], [W - 1, 0],
13                       [W - 1, H - 1], [0, H - 1]], dtype="float32")
14       M = cv2.getPerspectiveTransform(corners, dst)
15       return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
16
17   page = deskew(photo, detected_corners)
18   text = ocr(page)
```

Which change corrects the behaviour?

A. Use `cv2.warpAffine` instead of `cv2.warpPerspective`, since a four-point correction is an affine rather than a projective transform.

B. Pass `(W, H)` as the output size on line 15 instead of the source image's dimensions, so the warped result matches the rectangle the transform maps onto.

C. Swap `wA` and `hA` on lines 6 and 8, since perspective transforms measure width along the vertical edge pair.

D. Invert the transform with `cv2.invert(M)` before warping, since `getPerspectiveTransform` returns the mapping from destination to source.

---

**Q7.** A RAG service handles 1,000,000 queries per month. Each query sends 6,000 input tokens and generates 500 output tokens. Input tokens are billed at $0.50 per million and output tokens at $1.50 per million.

What is the monthly cost?

A. $3,000

B. $9,750

C. $750

D. $3,750

---

**Q8.** A binary classifier reports an ROC AUC of 0.82.

Which claim states what this value means?

A. The model classifies 82% of test cases correctly when its output is thresholded at 0.5.

B. For a randomly chosen positive and a randomly chosen negative case, the model assigns the positive a higher score 82% of the time.

C. The model's predicted probabilities deviate from observed frequencies by 18% on average across the confidence range.

D. 82% of the cases the model flags as positive are genuinely positive, averaged over all possible thresholds.

---

**Q9.** A single attention head computes scores for one query against three keys. After scaling, the scores are `[2.0, 1.0, 1.0]`, and the corresponding value vectors are `v1 = [1, 0]`, `v2 = [0, 2]`, `v3 = [0, 0]`.

Using `e^2 ≈ 7.389` and `e^1 ≈ 2.718`, what is the attention output, to two decimal places?

A. `[1.00, 2.00]`

B. `[0.58, 0.42]`

C. `[0.33, 0.67]`

D. `[0.58, 0.84]`

---

**Q10.** A model summarizes contractual obligations for a compliance team. The objective is to **preserve every condition, exception and deadline** stated in the source.

- P1: "Summarize the obligation accurately."
- P2: "Summarize the obligation in exactly 10 words."
- P3: "Summarize the obligation, preserving every condition, exception and deadline it states. Do not constrain the length of your answer."
- P4: "You are a contracts lawyer. Summarize the obligation."
- P5: "Summarize the obligation briefly."

Rank the prompts from most to least effective against this objective.

A. P2 > P3 > P1 > P4 > P5

B. P3 > P4 > P1 > P2 > P5

C. P1 > P3 > P5 > P4 > P2

D. P3 > P1 > P4 > P5 > P2

---

**Q11.** A team computes SHAP values for a credit model using a background dataset drawn entirely from approved applicants, then reports the resulting attributions as explanations for declined applicants.

Which claim is supported?

A. Using approved applicants as the background is required, since the baseline must represent the model's positive decision class to make contributions interpretable.

B. The attributions are invalid because SHAP requires a background dataset drawn from the same individuals whose predictions are being explained.

C. SHAP attributions are measured against the background distribution, so explanations for declined applicants are stated relative to an approved-applicant baseline rather than the population as a whole.

D. The background dataset affects only computation time, so its composition has no bearing on the attributions that are ultimately reported.

---

**Q12.** A factorial experiment reports these cell means for a response variable:

| | B = low | B = high |
|---|---|---|
| A = low | 10 | 14 |
| A = high | 18 | 16 |

Which claim is supported?

A. There is an interaction between A and B, since the effect of moving B from low to high is +4 at one level of A and −2 at the other.

B. There is no interaction, since both factors raise the response on average and their main effects are therefore additive.

C. There is no interaction, since an interaction requires the two lines to cross when the cell means are plotted.

D. An interaction cannot be assessed from cell means alone, since it is defined by the residual variance within each cell.

---

**Q13.** A team replaces an LSTM layer with a GRU layer of the same input and hidden size.

Approximately how does the parameter count change, and why?

A. It falls to about three quarters, since a GRU uses three weight groups where an LSTM uses four.

B. It falls to about half, since a GRU has no cell state and therefore needs no hidden-to-hidden matrices.

C. It rises by about a third, since a GRU's candidate computation requires an additional gated projection of the hidden state.

D. It is unchanged, since both cells use the same number of weight matrices and differ only in how they are combined.

---

**Q14.** A script fails with:

```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation
```

The relevant code is:

```python
1   h = self.fc1(x)
2   h = torch.relu(h)
3   h = self.bn(h)
4   h += self.shortcut(x)
5   h = torch.relu_(h)
6   out = self.fc2(h)
```

Which line is responsible, and why?

A. Line 4, because `+=` modifies a tensor that the shortcut branch still needs for its own backward pass.

B. Line 5, because the in-place ReLU overwrites the tensor whose pre-activation values the preceding operation requires to compute its gradient.

C. Line 3, because batch normalization stores running statistics in place and those buffers are reused during the backward pass.

D. Line 2, because the out-of-place ReLU allocates a new tensor and breaks the autograd graph's reference to the linear layer's output.

---

**Q15.** A team implements 2-D filtering with `cv2.filter2D` and compares the result against a reference implementation that performs true mathematical convolution, using the asymmetric kernel with rows `[1, 2, 3]`, `[4, 5, 6]`, `[7, 8, 9]`.

Which claim is supported?

A. `filter2D` performs convolution and the reference implementation performs correlation, so the reference must flip the kernel instead.

B. `filter2D` performs correlation rather than convolution, so the results differ unless the kernel is first flipped in both axes.

C. The results are identical, since correlation and convolution coincide for any kernel whose entries are all positive.

D. The results differ because `filter2D` normalizes the kernel by the sum of its entries while true convolution does not.

---

**Q16.** A Naive Bayes text classifier multiplies several hundred per-term probabilities for each document and begins returning identical scores of 0.0 for every class on long documents.

Which claim identifies the cause and the standard remedy?

A. The vocabulary is too large, so the per-term probabilities become uniformly small; restricting the feature set to the top 1,000 terms resolves it.

B. Long documents violate the independence assumption more severely, so the products collapse; document length normalization is the remedy.

C. The product of many probabilities below one underflows the floating-point representation; summing log probabilities instead preserves the comparison.

D. The smoothing constant is too small, so individual zero probabilities annihilate the product; raising it to 10 restores non-zero scores.

---

**Q17.** A ranked list of three results is graded for relevance on a 0-to-3 scale, giving relevance 3 at rank 1, relevance 1 at rank 2, and relevance 2 at rank 3. Discounted cumulative gain uses `rel / log2(rank + 1)`.

What is nDCG@3, to three decimal places?

A. 0.792

B. 0.973

C. 0.875

D. 1.000

---

**Q18.** A company develops a general-purpose AI model trained with more than 10²⁵ floating-point operations and releases it to downstream developers in the EU.

Which claim is supported?

A. The training compute threshold creates a presumption of systemic risk, bringing additional obligations such as model evaluation, adversarial testing and serious-incident reporting.

B. General-purpose models fall outside the Act entirely, since obligations attach to specific high-risk applications rather than to models themselves.

C. The model is automatically prohibited, since no general-purpose model above the compute threshold may be released within the EU.

D. The only obligation is a transparency notice to downstream developers, since general-purpose models are uniformly classified as limited risk.

---

**Q19.** A dense transformer has 7 billion parameters. A forward pass costs approximately 2 floating-point operations per parameter per token.

Approximately how many floating-point operations does a single forward pass over a 1,000-token prompt require?

A. 1.4 × 10¹³

B. 7 × 10⁹

C. 1.4 × 10¹⁰

D. 7 × 10¹²

---

**Q20.** A team builds a model to predict equipment failure. The dataset was assembled by joining sensor logs to a maintenance table, and the join introduced duplicate rows where a single maintenance event matched several log entries. The rows are then split randomly into training and test sets.

Which claim identifies the resulting problem?

A. Random splitting is always inappropriate for sensor data regardless of duplication, and only a chronological split is ever valid.

B. Duplicated rows land on both sides of the split, so the test set contains records the model has already seen, and the reported score overstates performance on genuinely new equipment.

C. Duplicated rows increase the effective sample size, so the model is better regularized and the reported score understates its true performance.

D. Duplication affects only the class balance, which can be corrected by resampling after the split without changing the split itself.

---

**Q21.** A clinical assistant explains diagnoses to patients. The objective is to **produce an explanation a person with no medical training can act on**.

- P1: "Explain the diagnosis. Example: 'Your thyroid — the gland in your neck that sets how fast your body uses energy — is making less hormone than it should.'"
- P2: "Explain the diagnosis. Use short sentences and avoid any word longer than three syllables."
- P3: "You are a compassionate physician. Explain the diagnosis."
- P4: "Explain the diagnosis as you would to someone with no medical training, defining any clinical term the first time you use it."
- P5: "Explain the diagnosis to the patient."

Rank the prompts from most to least effective against this objective.

A. P2 > P4 > P1 > P3 > P5

B. P1 > P4 > P5 > P2 > P3

C. P4 > P3 > P2 > P1 > P5

D. P4 > P1 > P3 > P5 > P2

---

**Q22.** A team runs LIME on the same instance ten times with different random seeds and finds that the top three features differ between runs.

Which claim is supported?

A. LIME is deterministic given a fixed instance, so differing outputs indicate the implementation has a defect that must be reported.

B. Instability across seeds is the expected and desirable behaviour, since it demonstrates that no single feature dominates the prediction.

C. LIME samples perturbations randomly and fits a local surrogate to them, so its explanations carry sampling variance; stability must be checked and the sample size raised until the ranking settles.

D. The instability proves the model itself is unstable, since a well-fitted model produces the same local behaviour regardless of how it is probed.

---

**Q23.** In a logistic regression, the coefficient on a binary indicator is 0.70.

Which claim is supported?

A. The odds ratio is approximately 2.01, so the indicator roughly doubles the predicted probability at every baseline value uniformly.

B. The coefficient cannot be converted to an odds ratio without knowing the intercept, since odds are defined only relative to the full linear predictor.

C. The odds ratio is approximately 2.01, so the indicator roughly doubles the odds; the change in probability depends on the baseline, moving 0.10 to about 0.18 but 0.50 to about 0.67.

D. The odds ratio is approximately 0.70, so the indicator raises the predicted probability by 70 percentage points regardless of the baseline value.

---

**Q24.** A batch of variable-length sequences is padded to a common length and passed to an `nn.LSTM`. A colleague recommends wrapping the batch with `pack_padded_sequence` first.

Which claim states what this achieves?

A. The padded positions are replaced with learned padding embeddings, which the recurrent layer treats as ordinary input.

B. The batch is reordered so that all sequences share a length, allowing the layer to process them without padding at all.

C. Gradients from padded positions are scaled by zero at the loss, which is equivalent and makes the packing step redundant.

D. The recurrent layer skips the padded timesteps, so the final hidden state reflects each sequence's true last element rather than trailing padding.

---

**Q25.** Photographs from a production line show fine-grained sensor noise, but the team must preserve sharp component boundaries for a downstream edge-detection stage.

Which filter is most appropriate?

A. A bilateral filter, which smooths using both spatial proximity and intensity similarity so that pixels across an edge contribute little to one another.

B. A Gaussian filter with a large kernel, which suppresses noise most effectively and affects edges and flat regions equally.

C. A median filter with a large window, which is the only filter that removes noise while leaving every edge mathematically unchanged.

D. A box filter, which averages uniformly and therefore introduces no directional bias into the subsequent gradient computation.

---

**Q26.** Two scripts are run on the same machine. Which change is sufficient to make repeated runs of a GPU training loop produce identical results?

A. Setting `torch.manual_seed(0)` alone, since the seed governs every source of randomness the framework uses.

B. Seeding Python, NumPy and PyTorch, and additionally setting deterministic algorithm flags and disabling the autotuning benchmark, since some GPU kernels are non-deterministic by default.

C. Setting `shuffle=False` on the `DataLoader` alone, since data ordering is the only source of run-to-run variation on a fixed machine.

D. Running on CPU rather than GPU, since determinism is impossible on GPU hardware under any configuration.

---

**Q27.** A vector index stores 5,000,000 chunks. Each embedding has 1,536 dimensions and is stored as a 32-bit float.

Approximately how much storage do the vectors themselves occupy?

A. 30.7 GB

B. 15.4 GB

C. 61.4 GB

D. 7.7 GB

---

**Q28.** A tokenizer produces on average 1.4 tokens per English word. A model's context window is 4,096 tokens, and a system prompt consumes 300 of them.

Approximately how many words of document text can be supplied?

A. 2,711

B. 5,314

C. 3,796

D. 2,926

---

**Q29.** A provider of a deployed high-risk system discovers that a malfunction contributed to a serious incident affecting several users.

Which obligation does the EU AI Act place on the provider?

A. To notify affected users directly, with no obligation towards any regulatory authority unless those users lodge a complaint.

B. To record the incident in internal logs for review at the next scheduled conformity assessment, with no interim reporting requirement.

C. To report the serious incident to the relevant market surveillance authorities within a defined period and to investigate the malfunction and its causal link.

D. To withdraw the system permanently from the market, since a serious incident reclassifies it into the prohibited tier.

---

**Q30.** A serving team adopts speculative decoding, in which a small draft model proposes several tokens and the large target model verifies them in a single forward pass.

Which claim is supported?

A. Output quality falls to that of the draft model, since the draft model determines which tokens are proposed and therefore which appear.

B. The technique reduces the total number of floating-point operations, since the target model no longer processes every token it emits.

C. Speculative decoding only helps when the draft model is larger than the target model, so that its proposals are rarely rejected.

D. Accepted tokens are exactly those the target model would have produced alone, so throughput rises without altering the output distribution.

---

**Q31.** A team plots training and validation error against training-set size:

| Training samples | 500 | 2,000 | 8,000 | 32,000 |
|---|---|---|---|---|
| Training error | 0.04 | 0.09 | 0.17 | 0.19 |
| Validation error | 0.41 | 0.31 | 0.23 | 0.21 |

Which claim is supported?

A. The curves are converging towards a similar error, so the model is approaching its capacity limit; collecting further data will yield diminishing returns and a higher-capacity model is the next lever.

B. The rising training error shows the model is degrading as data is added, so the additional samples are of lower quality and should be discarded.

C. The gap at 500 samples proves irreducible noise dominates, so no amount of additional data or capacity will reduce validation error further.

D. The model is underfitting throughout, so the correct response is to collect substantially more data while leaving the model capacity unchanged.

---

**Q32.** An agent calls a pricing API that returns HTTP 503 twice in succession. Its trace shows it then reported to the user that the product is unavailable.

Which claim identifies the correct handling?

A. Two consecutive failures establish that the endpoint is permanently unavailable, so reporting unavailability to the user is the correct inference.

B. The agent should fall back to its parametric knowledge of typical prices for the product category, since an approximate answer serves the user better than none.

C. The agent should call the same endpoint repeatedly without delay until it succeeds, since transient errors resolve within milliseconds in practice.

D. A 503 indicates a transient server condition, so the agent should retry with backoff and, if it still fails, report a temporary lookup failure rather than asserting a fact about the product.

---

**Q33.** A team must monitor drift for a model consuming free-text support tickets. Feature-level PSI is not applicable because the input is unstructured.

Which approach is most appropriate?

A. Convert the tickets to a bag-of-words representation and compute PSI on each of the 50,000 vocabulary counts independently.

B. Drift monitoring is not possible for unstructured input, so periodic scheduled retraining is the only available control.

C. Embed incoming tickets with the same encoder used in the pipeline and monitor the distribution of those embeddings against a reference window, using a distance measure over the embedding space.

D. Monitor only the model's output distribution, since any meaningful change in unstructured input must manifest as a change in predictions.

---

**Q34.** A condition affects 1% of a screened population. A test has sensitivity 0.95 and specificity 0.90. A patient tests **negative**.

What is the probability that the patient nevertheless has the condition?

A. About 10.00%

B. About 0.06%

C. About 5.00%

D. About 0.56%

---

**Q35.** A convolutional block contains `nn.BatchNorm2d(64)`.

Which claim correctly describes what this layer holds?

A. 256 learnable parameters, comprising scale, shift, running mean and running variance, all four of which are updated by the optimizer.

B. No learnable parameters, since batch normalization is a fixed statistical transformation applied to each channel independently.

C. 128 learnable parameters — a scale and a shift per channel — plus 128 non-learnable buffers holding the running mean and variance.

D. 64 learnable parameters — a scale per channel — with the shift supplied by the preceding convolution's bias term.

---

**Q36.** A ViT's patch embedding step splits a 224×224×3 image into non-overlapping 16×16 patches, flattens each, and projects it to 768 dimensions.

Which claim is supported?

A. The operation is equivalent to an average pooling layer with a 16×16 window, since flattening and projecting reduces each patch to a summary statistic.

B. The operation is exactly equivalent to `nn.Conv2d(3, 768, kernel_size=16, stride=16)`, which is how it is commonly implemented.

C. The operation cannot be expressed as a convolution, since convolution requires overlapping receptive fields produced by a stride smaller than the kernel.

D. The operation is equivalent to `nn.Conv2d(3, 768, kernel_size=16, stride=1)` followed by discarding all but every sixteenth output position.

---

**Q37.** Which statement correctly describes the effect of calling `model.eval()`?

A. Dropout becomes an identity mapping and batch normalization switches to its stored running statistics, while gradient tracking and the values of all learnable parameters remain unchanged.

B. Gradient computation is disabled throughout the module, and dropout and batch normalization switch to their inference behaviour.

C. All learnable parameters are frozen and batch normalization stops updating its running statistics, while dropout continues to mask activations.

D. Only batch normalization is affected, since dropout is controlled by its own probability argument rather than the module's training flag.

---

**Q38.** A RAG system over an internal wiki returns two retrieved chunks that state different values for the same policy limit — one from a page last edited in 2021 and one from 2024. The generated answer quotes the 2021 figure.

Which change most directly addresses this?

A. Increase the number of retrieved chunks so that additional corroborating sources outweigh the outdated one by majority.

B. Lower the generation temperature, since the choice between two conflicting figures is a sampling artefact rather than a retrieval issue.

C. Surface each chunk's last-modified date in the context and instruct the generator to prefer the most recent source and state the date it relied on.

D. Remove the 2021 page from the corpus, since a corpus containing any outdated document cannot produce reliable answers.

---

**Q39.** Word vectors are given as `king = [0.8, 0.6]`, `man = [0.7, 0.2]`, and `woman = [0.3, 0.5]`.

What is the result of the analogy computation `king − man + woman`?

A. `[0.4, 0.9]`

B. `[1.2, 0.3]`

C. `[0.4, 0.3]`

D. `[1.8, 1.3]`

---

**Q40.** An organization deploying a high-risk AI system assigns oversight to staff who receive no training on the system's limitations, its known failure modes, or how to interpret its confidence output.

Which obligation does this most directly engage?

A. The requirement of data minimization, since untrained overseers will request more data than the oversight task strictly needs.

B. The requirement of technical robustness, since oversight failures are a subcategory of system reliability rather than of organizational practice.

C. No specific obligation, since oversight staffing is an internal management matter outside the scope of the regulation.

D. The requirement that human oversight be effective, which presupposes that the people exercising it have the competence, training and authority to understand and act on the system's output.

---

**Q41.** A mixture-of-experts model has 8 expert feed-forward blocks of 1 billion parameters each and 1 billion parameters in shared components. Each token is routed to the top 2 experts.

What are the total and active parameter counts per token?

A. 9 billion total and 9 billion active.

B. 9 billion total and 3 billion active.

C. 9 billion total and 2 billion active.

D. 8 billion total and 3 billion active.

---

**Q42.** A gradient-boosting model is configured with 2,000 trees and a learning rate of 0.05. Validation error falls until tree 640 and then rises steadily while training error continues to fall.

Which response is appropriate?

A. Increase the number of trees to 5,000, since the validation curve will turn back downwards once the ensemble is large enough.

B. Remove the validation set and select the tree count by training error, since the training curve is the one still improving.

C. Use early stopping to halt at the validation minimum and keep the first 640 trees, since additional trees are fitting residual noise.

D. Increase the learning rate to 0.2, so that the model converges before overfitting can begin to take effect.

---

**Q43.** A production team writes dialogue for a period drama. The objective is to **maximize consistency with the character's established voice and period**.

- P1: "Write the character's next line."
- P2: "You are writing dialogue for Lady Ashworth, a 62-year-old aristocrat in 1924 Yorkshire who speaks formally and never uses contractions. Write her next line."
- P3: "Write the character's next line. Keep it under 25 words."
- P4: "Write the character's next line, avoiding modern slang and anachronistic references."
- P5: "Write the character's next line in period-appropriate language."

Rank the prompts from most to least effective against this objective.

A. P2 > P4 > P5 > P1 > P3

B. P4 > P2 > P5 > P3 > P1

C. P5 > P2 > P4 > P1 > P3

D. P2 > P5 > P3 > P4 > P1

---

**Q44.** A model produces the following probability forecasts and outcomes:

| Forecast | 0.90 | 0.80 | 0.30 | 0.60 |
|---|---|---|---|---|
| Outcome | 1 | 1 | 0 | 0 |

What is the Brier score?

A. 0.250

B. 0.500

C. 0.050

D. 0.125

---

**Q45.** A residual plot from a linear regression shows residuals forming a widening cone: near zero for small fitted values and spreading substantially for large ones.

Which claim is supported?

A. The pattern shows that the coefficient estimates are biased, so ordinary least squares must be replaced by a maximum-likelihood estimator.

B. The residuals exhibit heteroscedasticity, so the coefficient estimates remain unbiased but their standard errors are unreliable; robust standard errors or a variance-stabilizing transform address this.

C. The residuals show non-linearity, so a quadratic term must be added to the model before any inference on the coefficients is valid.

D. The residuals indicate autocorrelation, which arises whenever residual magnitude varies systematically with the fitted value.

---

**Q46.** A classifier is trained with label smoothing of 0.1 rather than hard one-hot targets.

Which claim is supported?

A. The softmax temperature is raised by 10%, which flattens the output distribution at inference without affecting the training objective.

B. The target distribution places 0.9 on the correct class and spreads 0.1 across the others, which discourages the model from driving logit gaps arbitrarily large and typically improves calibration.

C. Ten per cent of the training labels are randomly reassigned each epoch, which acts as a regularizer by injecting label noise.

D. The loss is multiplied by 0.9 for correct predictions and 0.1 for incorrect ones, which rebalances the contribution of easy and hard examples.

---

**Q47.** A team applies test-time augmentation to an image classifier, averaging predictions over the original image and three augmented copies.

Which claim is supported?

A. Accuracy typically improves modestly because averaging reduces variance, at the cost of multiplying inference compute by the number of augmentations.

B. Accuracy improves and inference cost is unchanged, since the augmented copies are processed within the same forward pass as a single batch.

C. Test-time augmentation changes the model's learned parameters, so the model must be re-validated after it is enabled.

D. Test-time augmentation is equivalent to training-time augmentation and therefore makes training-time augmentation unnecessary.

---

**Q48.** A sequence model is trained on padded batches with a padding index of 0. Which implementation correctly excludes padded positions from the loss?

A.
```python
logits = model(x)                       # (B, T, V)
mask = (y != 0)
loss = nn.CrossEntropyLoss()(
    logits.reshape(-1, logits.size(-1)), y.reshape(-1)) * mask.float().mean()
```

B.
```python
logits = model(x)                       # (B, T, V)
loss = nn.CrossEntropyLoss()(
    logits[y != 0], y[y != 0].float())
```

C.
```python
logits = model(x)                       # (B, T, V)
loss = nn.CrossEntropyLoss()(
    logits.reshape(-1, logits.size(-1)), y.reshape(-1))
```

D.
```python
logits = model(x)                       # (B, T, V)
loss = nn.CrossEntropyLoss(ignore_index=0)(
    logits.reshape(-1, logits.size(-1)), y.reshape(-1))
```

---

**Q49.** A RAG pipeline uses 1,200-token chunks. The embedding model has a maximum input length of 512 tokens. Retrieval quality is poor for long chunks, though no error is raised.

Which claim identifies the cause?

A. The maximum input length applies only at training time, so inference is unaffected and the cause must lie in the chunking boundaries.

B. Each chunk is silently truncated at 512 tokens before embedding, so the vector represents only the opening portion while the generator receives the whole chunk.

C. The embedding model pools over the full chunk regardless of its declared maximum, so quality falls because pooling dilutes the signal across too many tokens.

D. Long chunks produce vectors of larger magnitude, which biases cosine similarity towards them and crowds out shorter relevant chunks.

---

**Q50.** A NER annotation scheme must distinguish single-token entities from the first token of a multi-token entity, and must mark where each entity ends.

Which tagging scheme provides this, and what does it cost?

A. BIOES, which adds an E tag for the final token of a multi-token span and an S tag for single-token entities, at the cost of a larger tag set and more sparsely observed classes.

B. BIO, which already distinguishes these cases, since a B tag followed by an O tag implies a single-token entity without further marking.

C. IO tagging, which is sufficient because entity boundaries can always be recovered from consecutive runs of the same entity type.

D. BIOES, which reduces the tag set relative to BIO by merging the B and I categories into a single span marker.

---

**Q51.** A company trains a commercial model on text mined from the web. Several source sites carry machine-readable reservations of rights against text and data mining.

Which claim is supported under the EU framework?

A. Machine-readable reservations bind only non-commercial research, while commercial mining is exempt from them by design.

B. A rights holder's machine-readable opt-out removes the commercial text-and-data-mining exception for that material, so lawful use requires a licence or exclusion of that content.

C. The text-and-data-mining exception applies without qualification to any publicly accessible material, so machine-readable reservations carry no legal effect.

D. Reservations of rights apply only to reproduction of the source text, so training remains permitted provided the model does not output that text verbatim.

---

**Q52.** A pipeline sends documents averaging 3,000 English words to a model with an 8,192-token context. The tokenizer averages 1.35 tokens per word.

Which claim is supported?

A. Token count equals word count for English text, so each document consumes 3,000 tokens and two fit comfortably.

B. Token count cannot be estimated from word count, so the only way to plan context usage is to tokenize every document in advance.

C. Each document consumes roughly 4,050 tokens, so a document plus a 2,000-token instruction block fits, but two documents plus that block would not.

D. Each document consumes roughly 2,222 tokens, so three documents fit alongside a 2,000-token instruction block.

---

**Q53.** A node of 100 samples, evenly split between two classes, is split into a left child of 60 samples containing 48 positives and a right child of 40 samples containing 2 positives.

What is the information gain in bits, using entropy?

A. 0.548

B. 0.721

C. 0.452

D. 0.188

---

**Q54.** An internal tool formats bibliographic references in a house style. The objective is to **reproduce the exact surface format**, including punctuation and spacing.

- P1: "Format this reference in our house style. Example: Chandra, R. (2021) 'Load balancing at the edge', Internal Report TR-88, §4.2, pp. 11–14."
- P2: "Format this reference in our house style."
- P3: "Format this reference: author surname, then initial, then year in parentheses, then title in single quotes, then report number, then section symbol, then page range."
- P4: "You are a reference librarian. Format this reference in our house style."
- P5: "Format this reference carefully and consistently."

Rank the prompts from most to least effective against this objective.

A. P3 > P1 > P2 > P4 > P5

B. P1 > P3 > P4 > P5 > P2

C. P3 > P1 > P5 > P4 > P2

D. P1 > P4 > P3 > P2 > P5

---

**Q55.** A team wraps a classifier with conformal prediction calibrated to 90% coverage, so that the model returns a set of labels rather than a single label.

Which claim is supported?

A. The model's top-1 accuracy rises to 90%, since conformal prediction recalibrates the underlying classifier's scores.

B. Every returned set contains exactly 90% of the available labels, giving a fixed-width guarantee independent of the input.

C. Conformal prediction requires the model to be correctly specified, so the coverage guarantee holds only when the classifier is unbiased.

D. The returned sets contain the true label for approximately 90% of future instances, with set size growing on inputs where the model is less certain.

---

**Q56.** A team applies PCA to reduce 200 features to 20, then splits the reduced data into training and test sets and reports test accuracy.

Which claim is supported?

A. The leakage is negligible because PCA retains only 20 components, which cannot encode enough test information to affect the estimate.

B. The correct fix is to fit PCA separately on the training and test sets, so that each is represented in its own optimal basis.

C. Fitting PCA on the full dataset lets the components absorb variance structure from the test rows, so the reported test accuracy is optimistically biased; PCA must be fitted on the training split alone.

D. PCA is unsupervised and never touches the labels, so fitting it on all rows introduces no leakage and the reported accuracy is valid.

---

**Q57.** A three-class classifier produces logits `[2.0, 1.0, 0.1]` for a sample whose true class is the first. Softmax cross-entropy is used, and `e^2.0 ≈ 7.389`, `e^1.0 ≈ 2.718`, `e^0.1 ≈ 1.105`.

What is the gradient of the loss with respect to the first logit?

A. −0.341

B. 0.659

C. −0.659

D. 0.242

---

**Q58.** A binary image contains 6 white blobs, two of which touch at a single diagonal pixel. A team runs `cv2.connectedComponents` with default settings.

Which claim is supported?

A. The count depends on blob area rather than connectivity, so a minimum-area filter is the only way to control it.

B. With 8-connectivity the two diagonally touching blobs are merged and 5 components are reported; specifying 4-connectivity separates them and reports 6.

C. Connected-component labelling always reports 6, since diagonal adjacency is never treated as a connection under any setting.

D. With 4-connectivity the diagonally touching blobs are merged and 5 components are reported; 8-connectivity separates them into 6.

---

**Q59.** A tensor `scores` of shape `(4, 5)` and a boolean tensor `mask` of the same shape are combined as `torch.where(mask, scores, torch.tensor(float('-inf')))`, and the result is passed to `softmax(dim=1)`.

What does this accomplish?

A. The operation raises an error, since negative infinity cannot participate in the exponential the softmax computes.

B. Positions where the mask is False receive zero probability after the softmax, and the remaining probabilities renormalize to sum to one across each row.

C. Positions where the mask is False are removed from the tensor, so the result has fewer than five columns per row.

D. Positions where the mask is False receive the maximum probability, since negative infinity is the identity element for the softmax.

---

**Q60.** A team builds an evaluation set for a RAG system by asking an LLM to generate questions from randomly sampled chunks, then treating the source chunk as the single correct retrieval target.

Which claim identifies the principal weakness?

A. The principal weakness is sample size, since chunk-level sampling cannot produce enough questions to estimate recall precisely.

B. Questions generated from one chunk tend to be answerable from that chunk alone and to reuse its vocabulary, so the set over-represents single-hop lookups and rewards lexical overlap rather than genuine retrieval quality.

C. LLM-generated questions are unusable for evaluation because a model cannot produce questions it is not also able to answer.

D. The approach is sound only if the generating model differs from the model under evaluation, and is otherwise free of methodological issues.

---

**Q61.** A team compares a stemmer and a lemmatizer on the words `studies`, `better` and `running`.

Which claim is supported?

A. Both return `study`, `good` and `run`, differing only in execution speed rather than in output.

B. A stemmer returns dictionary forms while a lemmatizer applies suffix-stripping rules, which is why stemmer output is always a valid word.

C. Neither transforms `better`, since irregular forms are outside the scope of both stemming and lemmatization.

D. A lemmatizer returns `study`, `good` and `run` by consulting a vocabulary and part-of-speech information, while a rule-based stemmer returns truncations such as `studi` and leaves `better` unchanged.

---

**Q62.** A dataset is transformed by replacing each customer's name with a randomly generated identifier, held in a separate mapping table that the organization retains.

Which claim is supported under the GDPR?

A. The data remains personal data because re-identification is possible via the retained mapping; this is pseudonymization, a security measure, not anonymization.

B. The data is fully anonymized and therefore outside the scope of the regulation, since the identifiers themselves carry no meaning.

C. The data is anonymized provided the mapping table is stored on separate infrastructure, since physical separation breaks the linkage in law.

D. Pseudonymization and anonymization are legally equivalent, so the distinction affects internal practice but not regulatory obligations.

---

**Q63.** A team compares LoRA against prompt tuning for adapting a large model to a downstream task.

Which claim correctly distinguishes them?

A. Both modify the model's weights, differing only in whether the update is constrained to be low rank or is left unconstrained.

B. Prompt tuning requires more trainable parameters than LoRA, since it must learn an embedding for every token position in the context.

C. LoRA injects trainable low-rank matrices into the model's weight layers, while prompt tuning freezes all weights and learns only a set of continuous embeddings prepended to the input.

D. LoRA learns continuous embeddings prepended to the input, while prompt tuning modifies the attention projection matrices directly with low-rank updates.

---

**Q64.** A team must decide which of four models require feature standardization before fitting: k-nearest neighbours, a random forest, ridge regression, and a gradient-boosted tree ensemble.

Which claim is supported?

A. All four require it, since any model fitted by gradient descent is sensitive to the relative scale of its input features.

B. Only k-nearest neighbours requires it, since ridge regression's penalty is applied after the coefficients have been estimated and is therefore scale-free.

C. Only the tree ensembles require it, since axis-aligned splits are chosen by comparing raw feature values across different columns.

D. k-nearest neighbours and ridge regression require it — one because it computes distances, the other because the penalty acts on coefficient magnitudes — while the two tree ensembles are invariant to monotone rescaling of individual features.

---

**Q65.** An agent is given a self-critique step: after drafting an answer it is asked to identify errors in its own draft and revise. On a factual question-answering benchmark this raises accuracy from 71% to 74% while roughly doubling latency and token cost.

Which claim is supported?

A. The improvement proves the model can reliably detect its own errors, so self-critique should be enabled on every task by default.

B. A 3-point gain is within noise for any benchmark, so the result establishes nothing and the technique should be discarded.

C. The latency increase indicates the critique step is misimplemented, since self-critique should add negligible cost when the draft is already in context.

D. Self-critique buys a modest accuracy gain at a substantial cost multiple, so whether to enable it is a task-level judgement — worthwhile where an error is expensive, not where throughput dominates.

---

**Q66.** A team compares two explanation outputs for a declined loan: one reporting that income contributed −0.31 to the score, the other stating that increasing income by ₹15,000 would have reversed the decision.

Which claim is supported?

A. The first is an attribution answering how the model reached this decision; the second is a counterfactual answering what would change it, and only the second gives the applicant a course of action.

B. The two are equivalent, since a feature's attribution magnitude determines directly how much that feature must change to flip the outcome.

C. The first is more useful to the applicant, since a numeric contribution is precise while a counterfactual is a hypothetical without guarantee.

D. The second is invalid because counterfactuals require the model to be monotone in the feature, which cannot be assumed for a general model.

---

**Q67.** Among admitted hospital patients, a study finds a negative association between diabetes and a respiratory condition, although no such association exists in the general population.

Which claim explains this?

A. Admission is a common effect of both conditions, so conditioning on it induces a spurious negative association — a collider or selection bias that adding covariates cannot remove.

B. The finding shows that diabetes is protective against the respiratory condition, and the general-population study lacked the power to detect the effect.

C. The association is confounded by an unmeasured variable, so adjusting for additional patient characteristics would recover the true null relationship.

D. The sample is too small, since hospital populations cannot support inference about associations between two conditions.

---

**Q68.** A transformer is trained with a learning-rate schedule that rises linearly over the first 4,000 steps before decaying.

Which claim explains the warmup phase?

A. Warmup exists to let the data loader fill its prefetch buffers, after which the full learning rate can be applied safely.

B. Warmup prevents the learning rate from exceeding the largest eigenvalue of the loss Hessian, which is only computable after 4,000 steps.

C. Early in training the adaptive optimizer's second-moment estimates are based on few samples and are therefore unreliable, so large steps can destabilize the model before the estimates settle.

D. Warmup compensates for the small batch sizes used at the start of training, which are increased once the schedule reaches its peak.

---

**Q69.** Two bounding boxes are given in `(x1, y1, x2, y2)` format: box P is `(0, 0, 100, 100)` and box Q is `(50, 0, 150, 100)`.

What is their intersection over union?

A. 0.250

B. 0.667

C. 0.333

D. 0.500

---

**Q70.** A `tf.data` pipeline over a file of 1,000,000 records sorted by category is configured as `dataset.shuffle(1000).batch(64)`.

Which claim is supported?

A. The shuffle buffer holds only 1,000 of 1,000,000 records, so each batch is drawn from a narrow window of a sorted file and will be dominated by a single category.

B. The buffer size is irrelevant because `shuffle` performs a full random permutation of the dataset before any element is emitted.

C. The configuration is correct, since shuffling before batching always guarantees that batches are representative of the whole dataset.

D. The buffer should be smaller still, since a large buffer delays the first batch and provides no benefit to randomization quality.

---

**Q71.** A RAG system embeds small 200-token chunks for retrieval precision but finds that the generator often lacks the surrounding context needed to answer correctly.

Which approach resolves this tension?

A. Increase the chunk size to 2,000 tokens so retrieval and generation both operate on the same larger unit.

B. Retrieve the small chunks and instruct the generator to request additional context when it needs it, in a second pass.

C. Embed the large sections and retrieve them directly, since retrieval precision matters less than the generator having full context.

D. Embed and retrieve the small chunks, but pass the larger parent section each retrieved chunk belongs to into the generator's context.

---

**Q72.** A language model assigns probabilities 0.5, 0.25, 0.5 and 0.25 to the four words of a test sentence.

What is the model's perplexity on this sentence?

A. 1.500

B. 4.000

C. 0.354

D. 2.828

---

**Q73.** A retailer wishes to process customer purchase histories to build a recommendation model, relying on legitimate interests rather than consent.

Which claim is supported?

A. Legitimate interests requires a documented balancing test weighing the organization's interest against the data subject's rights and reasonable expectations, and the subject retains a right to object.

B. Legitimate interests is available for any processing that serves a commercial purpose, with no assessment required beyond identifying that purpose.

C. Legitimate interests cannot be used for any profiling activity, so consent is the only available lawful basis for a recommendation model.

D. Legitimate interests removes the obligation to inform data subjects, since the basis does not depend on their awareness of the processing.

---

**Q74.** A system must emit output conforming to a strict JSON schema. The team enables constrained decoding, which masks tokens that would violate the grammar at each step.

Which claim is supported?

A. Constrained decoding modifies the model's weights so that invalid tokens acquire zero probability permanently.

B. Every generated output parses against the schema by construction, though the constraint can degrade content quality where the schema conflicts with the model's preferred phrasing.

C. Constrained decoding raises the probability of valid output but cannot guarantee it, since the model may still emit an invalid token when all candidates are masked.

D. Constrained decoding operates by retrying generation until a valid output appears, so its cost grows with the strictness of the schema.

---

**Q75.** A streaming service must recommend content to users who registered within the past hour and have no viewing history. Rich metadata is available for every title.

Which approach is appropriate?

A. Content-based or metadata-driven recommendation seeded by onboarding preferences, since collaborative filtering has no interaction signal for these users.

B. Matrix factorization over the user-item interaction matrix, since latent factors are inferred for every user including those with no interactions.

C. Item-based collaborative filtering, since item similarities are computed from other users and transfer to new users without history.

D. No recommendation should be served until sufficient history accumulates, since any earlier recommendation is necessarily arbitrary.

---

**Q76.** A code-review assistant flags security issues. The objective is to **maximize the proportion of genuine issues surfaced**, with triage performed by a human afterwards.

- P1: "Review this code for security issues. Report only issues you are confident are exploitable in production."
- P2: "You are a security engineer. Review this code."
- P3: "Review this code for security issues."
- P4: "Review this code for security issues. Be thorough."
- P5: "Review this code for security issues. List every potential issue including ones you are unsure about, marking each as confirmed or possible."

Rank the prompts from most to least effective against this objective.

A. P1 > P5 > P4 > P3 > P2

B. P5 > P1 > P4 > P2 > P3

C. P4 > P5 > P2 > P3 > P1

D. P5 > P4 > P3 > P2 > P1

---

**Q77.** Before each release, a team runs its assistant against a fixed suite of 500 recorded inputs with expected behaviours, and blocks the release if any previously passing case now fails.

What is this, and what is its principal limitation?

A. A red-team exercise, which probes for adversarial failures and therefore covers the space of unanticipated inputs by design.

B. A canary release, which limits exposure by routing a small fraction of live traffic to the new version first.

C. A regression suite, which reliably detects behaviour that has degraded relative to recorded expectations but cannot surface failures on inputs nobody thought to record.

D. An A/B test, which establishes the causal effect of the release on user outcomes by comparing two randomized arms.

---

**Q78.** Support tickets arrive at a help desk at an average rate of 3 per hour, following a Poisson process. Use `e⁻³ ≈ 0.0498`.

What is the probability that exactly 2 tickets arrive in a given hour?

A. 0.149

B. 0.050

C. 0.333

D. 0.224

---

**Q79.** A decoder applies `nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)` to a feature map of spatial size 8×8.

What is the output spatial size?

A. 32

B. 16

C. 14

D. 18

---

**Q80.** A 4,000×3,000 photograph is reduced to 400×300 using `cv2.resize` with `INTER_NEAREST`. The result shows jagged edges and moiré patterns across finely striped regions that were smooth in the original.

Which claim explains this?

A. The artefacts arise from the aspect ratio, which must be preserved exactly to prevent moiré from appearing in striped regions.

B. `INTER_NEAREST` is the correct choice for downscaling, and the artefacts indicate the source image was compressed with a lossy codec.

C. Moiré patterns arise only during upscaling, so the described artefacts must originate in a later stage of the pipeline.

D. Nearest-neighbour sampling discards nine of every ten pixels without averaging, so high-frequency detail aliases into false low-frequency patterns; `INTER_AREA` averages over the source region and avoids this.

---

**Q81.** After a training run finishes, a developer calls `torch.cuda.empty_cache()` and observes that `nvidia-smi` still reports high memory use for the process.

Which claim is supported?

A. `empty_cache` frees all GPU memory, so the reading indicates a driver-level reporting fault rather than genuine allocation.

B. `empty_cache` must be called once per tensor, so a single call releases only the most recently allocated block.

C. `empty_cache` defers its effect until the process exits, so the reading is expected and no memory is ever freed during execution.

D. `empty_cache` releases only cached blocks that no tensor references; memory held by live tensors, and the CUDA context itself, remains allocated.

---

**Q82.** A customer-facing assistant must **avoid naming any competitor**, including ones not known in advance.

- P1: "Answer the customer's question. Do not mention Zenith, Orbital or Kestrel under any circumstances."
- P2: "Answer the customer's question using only information about our own products. If asked to compare against another vendor, say that you can only speak to our own range."
- P3: "Answer the customer's question. Avoid naming competitors."
- P4: "You are a loyal company representative. Answer the customer's question."
- P5: "Answer the customer's question."

Rank the prompts from most to least effective against this objective.

A. P1 > P2 > P3 > P4 > P5

B. P2 > P1 > P3 > P4 > P5

C. P3 > P2 > P5 > P1 > P4

D. P2 > P3 > P1 > P5 > P4

---

**Q83.** What is the Levenshtein edit distance between `kitten` and `sitting`?

A. 1

B. 3

C. 2

D. 4

---

**Q84.** A bank procures a third-party credit-risk model and deploys it under its own brand for lending decisions in the EU.

Which claim is supported?

A. Procurement transfers all obligations to the vendor by contract, so the bank's exposure is limited to whatever the contract specifies.

B. The bank has no obligations until a complaint is raised, at which point the vendor's conformity assessment becomes the operative document.

C. Deploying a high-risk system under its own name makes the bank a provider, so it inherits provider obligations and must obtain from the vendor the documentation needed to discharge them.

D. The bank is a deployer only, so its obligations are limited to operating the system according to the vendor's instructions for use.

---

**Q85.** A team distils a large teacher model into a smaller student by training the student to match the teacher's full output distribution rather than only the argmax label.

Which claim explains why this transfers more than hard labels alone?

A. The student learns the teacher's weights directly through the distribution, so distillation is a form of parameter copying at reduced precision.

B. Soft targets are used only to speed convergence; the final student accuracy is the same as it would be with hard labels given enough epochs.

C. The teacher's probabilities over incorrect classes encode learned similarity structure — which wrong answers are near misses — and that signal is absent from a one-hot label.

D. Matching the full distribution is mathematically identical to training on hard labels, and the benefit comes entirely from the student's smaller capacity.

---

**Q86.** A three-class classifier is evaluated:

| Class | TP | FP | FN |
|---|---|---|---|
| A | 40 | 10 | 5 |
| B | 30 | 5 | 10 |
| C | 5 | 20 | 20 |

Which claim is supported?

A. Neither can be computed from this table, since F1 requires the true negative counts that the table omits.

B. Macro F1 is roughly 0.61 and micro F1 roughly 0.68; the gap arises because macro averaging gives the poorly performing class C the same weight as the others.

C. Macro F1 is roughly 0.68 and micro F1 roughly 0.61; the gap arises because micro averaging weights each class equally regardless of support.

D. Macro and micro F1 coincide at roughly 0.65, since the three classes have similar support in this evaluation.

---

**Q87.** A RAG system over a policy wiki is refreshed nightly by re-embedding pages whose `last_modified` timestamp changed. Users report that answers cite content deleted weeks earlier.

Which claim identifies the cause?

A. The retriever caches its results for several weeks, so deleted content is served from the cache rather than the index.

B. Nightly refresh is too infrequent for a policy wiki, and the fix is to re-embed the entire corpus hourly.

C. Deletions do not change any surviving page's timestamp, so the pipeline never revisits the removed pages and their vectors persist in the index.

D. The embedding model drifts between nightly runs, so vectors for deleted pages gradually migrate towards live queries.

---

**Q88.** An adversary with query access to a deployed model wishes to determine whether a particular individual's record was in its training set.

Which attack is this, and what raises its success rate?

A. A model-inversion attack, whose success rises with the number of output classes the model exposes to the querying party.

B. A poisoning attack, whose success rises when the attacker can contribute records to the next scheduled retraining cycle.

C. An evasion attack, whose success rises when the model's decision boundary lies close to the individual's record in feature space.

D. A membership-inference attack, whose success rises when the model overfits, since memorized training points receive systematically higher confidence than unseen ones.

---

**Q89.** A survey of 1,000 randomly sampled respondents reports a proportion of 0.50. The 95% margin of error uses `1.96 × sqrt(p(1−p)/n)`.

What is the margin of error, and what would halving it require?

A. About ±1.6 percentage points, and halving it would require roughly 2,000 respondents, since the margin falls linearly with sample size.

B. About ±5.0 percentage points, and halving it would require roughly 4,000 respondents, since the margin falls with the square root of sample size.

C. About ±3.1 percentage points, and halving it would require roughly 4,000 respondents, since the margin falls with the square root of sample size.

D. About ±3.1 percentage points, and halving it would require roughly 2,000 respondents, since the margin falls linearly with sample size.

---

**Q90.** A team logs the global gradient norm during training:

| Step | 100 | 200 | 300 | 310 | 320 |
|---|---|---|---|---|---|
| Gradient norm | 1.8 | 2.1 | 1.9 | 340.0 | 1,200.0 |
| Loss | 2.4 | 2.1 | 2.0 | 5.7 | nan |

Which claim is supported?

A. The gradient norms up to step 300 are too small, indicating vanishing gradients, and the spike at 310 is the network finally beginning to learn.

B. The pattern indicates overfitting, since the loss rose after a period of steady decline; early stopping at step 300 is the appropriate response.

C. A gradient spike near step 310 destabilized the update and drove the loss to a non-finite value; clipping the global norm bounds such spikes while leaving ordinary steps unchanged.

D. The loss became non-finite because the learning rate decayed to zero at step 310, halting learning and leaving the loss undefined thereafter.

---
# ANSWER KEY

| Q | Ans | Domain | Q | Ans | Domain | Q | Ans | Domain |
|---|-----|--------|---|-----|--------|---|-----|--------|
| 1 | C | Stats | 31 | A | ML | 61 | D | NLP |
| 2 | B | PyTorch | 32 | D | Prompts | 62 | A | Governance |
| 3 | A | NLP | 33 | C | RAI Tech | 63 | C | LLM |
| 4 | B | Governance | 34 | B | Stats | 64 | D | ML |
| 5 | D | Deep Learning | 35 | C | Deep Learning | 65 | D | Prompts |
| 6 | B | Vision | 36 | B | Vision | 66 | A | RAI Tech |
| 7 | D | RAG | 37 | A | PyTorch | 67 | A | Stats |
| 8 | A | ML | 38 | C | RAG | 68 | C | Deep Learning |
| 9 | B | LLM | 39 | A | NLP | 69 | C | Vision |
| 10 | D | Prompts | 40 | D | Governance | 70 | A | PyTorch |
| 11 | C | RAI Tech | 41 | B | LLM | 71 | D | RAG |
| 12 | A | Stats | 42 | C | ML | 72 | D | NLP |
| 13 | A | Deep Learning | 43 | A | Prompts | 73 | A | Governance |
| 14 | A | PyTorch | 44 | D | RAI Tech | 74 | B | LLM |
| 15 | B | Vision | 45 | B | Stats | 75 | A | ML |
| 16 | C | NLP | 46 | B | Deep Learning | 76 | D | Prompts |
| 17 | B | RAG | 47 | A | Vision | 77 | C | RAI Tech |
| 18 | A | Governance | 48 | D | PyTorch | 78 | D | Stats |
| 19 | A | LLM | 49 | B | RAG | 79 | B | Deep Learning |
| 20 | B | ML | 50 | A | NLP | 80 | D | Vision |
| 21 | D | Prompts | 51 | B | Governance | 81 | D | PyTorch |
| 22 | C | RAI Tech | 52 | C | LLM | 82 | D | Prompts |
| 23 | C | Stats | 53 | C | ML | 83 | B | NLP |
| 24 | D | Deep Learning | 54 | B | Prompts | 84 | C | Governance |
| 25 | A | Vision | 55 | D | RAI Tech | 85 | C | LLM |
| 26 | B | PyTorch | 56 | C | Stats | 86 | B | ML |
| 27 | A | RAG | 57 | A | Deep Learning | 87 | C | RAG |
| 28 | A | NLP | 58 | B | Vision | 88 | D | RAI Tech |
| 29 | C | Governance | 59 | B | PyTorch | 89 | C | Stats |
| 30 | D | LLM | 60 | B | RAG | 90 | C | Deep Learning |

# EXPLANATIONS

**Q1 — C.** Four groups give C(4,2) = **6** pairwise comparisons. If no true differences exist, the probability of at least one false positive is 1 − 0.95⁶ = **0.265**. A significant omnibus F licenses follow-up testing but does not control the family-wise rate, so a correction such as Bonferroni (α = 0.05/6 ≈ 0.0083) or Tukey's HSD is required.

**Q2 — B.** `self.bias` is a plain `torch.zeros` attribute, not an `nn.Parameter` or a registered buffer, so `model.to(device)` never moves it. The indices arrive on the GPU while the bias sits on the CPU, and line 11 fails with a device mismatch. Wrapping it in `nn.Parameter` or calling `register_buffer` fixes both the placement and its absence from `state_dict`.

**Q3 — A.** tf = 6 (raw count), idf = log10(100/10) = log10(10) = **1.0**, so the weight is 6 × 1.0 = **6.00**. The 200-token document length is a distractor — it matters only under length-normalized term frequency, which the question explicitly excludes by specifying raw counts.

**Q4 — B.** Both surviving features are derived from the address, so they carry the geographic signal the postcode removal was meant to eliminate. This is proxy discrimination, and it is why fairness work audits the *correlational neighbourhood* of a protected or proxy attribute rather than deleting a single column. Outcome measurement, not coefficient inspection, is the right instrument.

**Q5 — D.** Each of the four gates holds an input-to-hidden matrix (100 × 256), a hidden-to-hidden matrix (256 × 256) and a bias (256): 25,600 + 65,536 + 256 = 91,392 per gate. Four gates give **365,568**. Option A is the per-gate figure — the trap for stopping one step early. (PyTorch actually stores two bias vectors per gate, giving 366,592; the question fixes a single bias to keep the arithmetic unambiguous.)

**Q6 — B.** The transform maps the four detected corners onto a W×H rectangle, but `warpPerspective` is told to produce an output the size of the *source* image. Where W and H are smaller than the source, content is fine; where the rectangle extends beyond, the right and bottom are clipped. Passing `(W, H)` makes the output canvas match the rectangle the transform actually targets.

**Q7 — D.** Input: 1,000,000 × 6,000 = 6 × 10⁹ tokens = 6,000 million × $0.50 = **$3,000**. Output: 1,000,000 × 500 = 5 × 10⁸ tokens = 500 million × $1.50 = **$750**. Total **$3,750**. Note the asymmetry that drives RAG economics: output is three times the unit price but only 8% of the volume, so retrieved context dominates the bill.

**Q8 — A.** AUC is the probability that a randomly chosen positive receives a higher score than a randomly chosen negative — a ranking property, independent of any threshold and independent of calibration. It says nothing about accuracy at 0.5, and a model can have excellent AUC with badly miscalibrated probabilities.

**Q9 — B.** Softmax over [2.0, 1.0, 1.0]: denominator = 7.389 + 2.718 + 2.718 = 12.825. Weights = [0.576, 0.212, 0.212]. Output = 0.576·[1,0] + 0.212·[0,2] + 0.212·[0,0] = **[0.58, 0.42]**. The common error is forgetting that the second component is 0.212 × 2, not 0.212.

**Q10 — D.** This is the case where the most precisely specified prompt is the **worst**. P3 names the requirement and explicitly removes the length constraint that would force omissions. P1's 'accurately' is vague but aligned. P4's persona primes a relevant register without stating the requirement. P5 pushes towards brevity. **P2 ranks last**: a hard 10-word cap guarantees that conditions, exceptions and deadlines are dropped — the specification is precise and pointed at the wrong target.

**Q11 — C.** SHAP explains a prediction *relative to a background distribution*, and contributions are differences from that baseline. With a background drawn only from approved applicants, an attribution reads as 'relative to a typical approved applicant', which is a different and possibly misleading statement from 'relative to the population'. The background is a modelling choice with interpretive consequences, not a performance setting.

**Q12 — A.** Moving B from low to high changes the response by +4 when A is low (10→14) and by −2 when A is high (18→16). An effect of B that depends on the level of A **is** an interaction. Crossing lines are sufficient but not necessary — non-parallel lines suffice, and here they diverge in sign. Interaction is defined on the cell means, not on within-cell variance.

**Q13 — A.** A GRU has three weight groups (reset gate, update gate, candidate) against an LSTM's four (forget, input, candidate, output), so with identical input and hidden sizes the parameter count is about **three quarters**. GRUs do have hidden-to-hidden matrices; what they lack is the separate cell state, which costs no parameters of its own.

**Q14 — A.** `relu_` writes in place over the tensor produced at line 4. That tensor's values are needed by the backward pass of the operation that produced it, and autograd detects that they have been overwritten. Replacing it with the out-of-place `torch.relu` resolves it. Note that `+=` at line 4 is also in place but happens to be safe here, which is what makes the line attribution non-trivial.

**Q15 — B.** `cv2.filter2D` computes **correlation**: it slides the kernel without flipping it. True convolution flips the kernel through 180° in both axes first. For symmetric kernels the two coincide, which is why the distinction rarely surfaces — but this kernel is asymmetric, so the results differ unless the kernel is flipped before being passed in.

**Q16 — C.** Multiplying several hundred probabilities each well below one underflows double precision to exactly 0.0, at which point all classes tie. The standard remedy is to sum log probabilities instead: the logarithm is monotone, so the argmax is preserved while the magnitudes stay representable. Smoothing addresses a different failure — individual zero probabilities — and does not prevent underflow.

**Q17 — B.** DCG = 3/log2(2) + 1/log2(3) + 2/log2(4) = 3 + 0.631 + 1 = 4.631. The ideal ordering is 3, 2, 1, giving IDCG = 3 + 2/1.585 + 1/2 = 4.762. nDCG@3 = 4.631/4.762 = **0.973**. Unlike MRR, nDCG uses graded relevance and penalizes putting a more relevant document below a less relevant one.

**Q18 — A.** Training compute above 10²⁵ FLOP creates a presumption of systemic risk for a general-purpose model, adding obligations beyond the baseline transparency and copyright-policy duties: model evaluation, adversarial testing, systemic-risk assessment and mitigation, incident reporting and cybersecurity protection. General-purpose models are squarely within scope, which disposes of the option excluding them.

**Q19 — A.** The standard estimate is 2 FLOPs per parameter per token for a forward pass — one multiply and one add per weight. So 2 × 7 × 10⁹ × 1,000 = **1.4 × 10¹³**. (Backward is roughly twice forward again, giving the familiar 6·N·T for a full training step.)

**Q20 — B.** Randomly splitting rows that contain duplicates places copies of the same underlying event on both sides, so the test set is partly memorization rather than generalization and the reported score is inflated. The fix is to deduplicate before splitting, or to split on the entity — equipment unit or maintenance event — rather than on rows.

**Q21 — D.** P4 names the audience and adds an operative rule — define each clinical term on first use — which is what makes the output actionable. P1's worked example demonstrates exactly that register in practice. P3's persona is genuinely relevant here, since compassion and plain speech are part of the target. P5 is neutral. **P2 ranks last**: a syllable ceiling is a brittle proxy that blocks necessary words ('thyroid', 'antibiotic') and produces stilted text while permitting undefined jargon that happens to be short.

**Q22 — C.** LIME perturbs the instance randomly, weights the perturbations by proximity and fits a local surrogate, so its output carries sampling variance and different seeds can reorder the top features. It is not deterministic. The practical response is to measure stability across seeds and increase the number of samples until the ranking settles — instability here reflects the explainer, not the model.

**Q23 — C.** e^0.70 = **2.014**, so the indicator roughly doubles the *odds*. The probability change is not uniform: from a baseline of 0.10 the odds are 0.111, doubling to 0.224, giving p = **0.183**; from 0.50 the odds are 1.0, doubling to 2.014, giving p = **0.668**. So the same coefficient shifts probability by 8 points in one case and 17 in the other — the non-linearity that makes 'per-unit effect on probability' a meaningless phrase in logistic regression.

**Q24 — D.** Packing tells the recurrent layer where each sequence genuinely ends, so it neither computes over padded timesteps nor lets them contaminate the final hidden state. Without it, `h_n` reflects whatever the cell produced after consuming trailing padding — silently wrong for any model that reads the last hidden state. Masking the loss does not fix this, because the corruption is in the recurrence, not the loss.

**Q25 — A.** A bilateral filter weights neighbouring pixels by both spatial distance and intensity difference, so pixels on the far side of an edge contribute almost nothing and the edge survives while flat regions are smoothed. Gaussian and box filters blur edges along with noise. A median filter is excellent for impulse noise but does modify edges, so the claim that it leaves every edge unchanged is too strong.

**Q26 — B.** Determinism on GPU requires more than one seed: seed Python, NumPy and PyTorch (`torch.manual_seed` covers CUDA too), then set `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.benchmark = False`, since cuDNN autotuning selects kernels by timing and several kernels use non-deterministic atomics. Determinism is achievable on GPU, just not by default.

**Q27 — A.** 5,000,000 × 1,536 × 4 bytes = 30,720,000,000 bytes ≈ **30.7 GB** for the raw vectors, before index overhead. Halving to float16 gives about 15.4 GB, and this arithmetic is exactly why quantized indexes and dimensionality reduction matter at scale.

**Q28 — A.** Available tokens = 4,096 − 300 = 3,796. At 1.4 tokens per word, that is 3,796/1.4 ≈ **2,711 words**. Option C is the trap for reporting the token count rather than converting it to words; option B multiplies rather than divides.

**Q29 — C.** The Act requires providers of high-risk systems to report serious incidents to the relevant market surveillance authority within defined deadlines, and to investigate the incident and the causal link to the system. A serious incident triggers reporting and corrective action, not automatic reclassification into the prohibited tier.

**Q30 — D.** The verification step accepts a proposed token only if it matches what the target model's own distribution would have produced, so the output is distributionally identical to running the target model alone — the gain is pure throughput. Total FLOPs actually *rise* slightly, since rejected drafts are wasted work; what falls is wall-clock latency, because several tokens are verified in one pass of a memory-bandwidth-bound model.

**Q31 — A.** Training error rises and validation error falls as data grows, and the two are converging towards roughly 0.20 — the signature of a model at its capacity limit rather than one starved of data. More data will close the remaining 2-point gap slowly; a higher-capacity model is the lever that moves the asymptote. Rising training error with more data is expected, not a data-quality signal.

**Q32 — D.** HTTP 503 is Service Unavailable — a transient server condition. The correct handling is retry with exponential backoff and, on continued failure, an honest report that the lookup failed. Converting a tool failure into a confident factual assertion about the product is the serious error here: the agent stated something it had no evidence for.

**Q33 — C.** For unstructured input the practical approach is to monitor the distribution of embeddings produced by the same encoder the pipeline uses, comparing a recent window against a reference using a distance over the embedding space (MMD, energy distance, or a domain classifier). Per-token PSI over a 50,000-word vocabulary is unusable, and output-only monitoring misses input shifts the model absorbs without changing its predictions.

**Q34 — B.** P(negative ∧ disease) = 0.05 × 0.01 = 0.0005. P(negative ∧ no disease) = 0.90 × 0.99 = 0.891. P(disease | negative) = 0.0005/0.8915 = **0.00056**, about **0.06%**. The negative predictive value is 99.94% — the mirror image of the more familiar positive-predictive-value problem, and the reason a negative result is so informative when prevalence is low.

**Q35 — C.** `BatchNorm2d(64)` learns a per-channel scale γ and shift β: 64 + 64 = **128 learnable parameters**, updated by the optimizer. It additionally maintains `running_mean` and `running_var` as **buffers** — 128 values that appear in `state_dict` and move with `.to(device)` but are updated by the forward pass in training mode rather than by gradients.

**Q36 — B.** Splitting into non-overlapping P×P patches, flattening and projecting is exactly a convolution with kernel size P and stride P: each output position sees one patch and produces one D-dimensional vector. Reference ViT implementations use `nn.Conv2d(3, D, kernel_size=16, stride=16)` for precisely this reason. Stride equal to kernel size is perfectly valid — it simply makes the receptive fields non-overlapping.

**Q37 — A.** `model.eval()` sets the module's training flag, which makes dropout an identity mapping and switches batch normalization to its stored running statistics. It does **not** disable gradient tracking — that is `torch.no_grad()` — and it does not alter any parameter value. Correct validation code therefore uses both, which is why conflating them is such a common source of wrong metrics.

**Q38 — C.** Both chunks were retrieved, so this is a generation-side disambiguation problem: the model had no basis for preferring one figure over the other. Supplying each chunk's last-modified date in the context and instructing the generator to prefer the most recent — and to state which date it used — gives it the missing rule and makes the choice auditable. Deleting history is neither practical nor a general fix.

**Q39 — A.** Componentwise: 0.8 − 0.7 + 0.3 = **0.4** and 0.6 − 0.2 + 0.5 = **0.9**, giving **[0.4, 0.9]**. The result is then compared by cosine similarity against the vocabulary to find the nearest word. The arithmetic is trivial; what it tests is knowing that analogies operate by vector offset in the embedding space.

**Q40 — D.** Human oversight under the Act must be *effective*, which presupposes that the people performing it understand the system's capabilities and limits, can interpret its output, are aware of automation bias, and have the authority to override. Assigning oversight to untrained staff satisfies the form and not the substance — and the Act's AI-literacy provisions reinforce the same point.

**Q41 — B.** Total parameters = 8 experts × 1B + 1B shared = **9 billion**. With top-2 routing, each token activates 2 experts plus the shared components = **3 billion**. That separation is the point of the architecture: memory and storage scale with the total, while per-token compute scales with the active count.

**Q42 — C.** Validation error reaching a minimum at tree 640 and rising thereafter while training error keeps falling is overfitting, and the boosting-specific remedy is early stopping on the validation metric — keep the 640 trees at the minimum and discard the rest. Raising the learning rate makes each tree's contribution larger and typically overfits sooner, not later.

**Q43 — A.** This is the set where the **persona prompt wins outright**, because the objective *is* voice and register. P2 specifies character, age, class, period, locale and a concrete linguistic rule (no contractions) — everything a model needs to produce consistent dialogue. P4's negative constraint prevents the worst anachronisms without supplying a voice. P5 gestures at period language. P1 is neutral. **P3 ranks last**: a word limit is orthogonal to the objective and constrains the one dimension nobody asked about.

**Q44 — D.** Brier score is the mean squared difference between forecast and outcome: [(0.9−1)² + (0.8−1)² + (0.3−0)² + (0.6−0)²]/4 = [0.01 + 0.04 + 0.09 + 0.36]/4 = 0.50/4 = **0.125**. Lower is better, and the score decomposes into calibration and refinement components — it penalizes both miscalibration and indecisiveness.

**Q45 — B.** A cone-shaped residual spread is heteroscedasticity: error variance grows with the fitted value. Ordinary least squares remains **unbiased** under it, but the usual standard errors are wrong, so t-statistics, p-values and confidence intervals are unreliable. Remedies are heteroscedasticity-robust standard errors, weighted least squares, or a variance-stabilizing transform such as a log.

**Q46 — B.** Label smoothing replaces the one-hot target with 0.9 on the true class and 0.1 spread over the rest. Because the target is no longer attainable by driving one logit to infinity, the model stops maximizing the logit gap, which reduces overconfidence and usually improves calibration and robustness at a small cost in top-1 accuracy. No labels are reassigned and no loss weighting occurs.

**Q47 — A.** Test-time augmentation averages predictions over several transformed views, which reduces prediction variance and typically yields a small accuracy gain. The cost is linear in the number of views: four views means four forward passes. Batching them together reduces wall-clock time but not the compute, and the model's parameters are untouched.

**Q48 — D.** `ignore_index=0` tells the criterion to skip target positions equal to the padding index entirely — they contribute neither to the loss nor to its normalization. Option A includes padded positions in both. Option C computes the full loss and then scales it, which changes the magnitude but still lets padded positions shape the gradient. Option D casts integer class indices to float, which the criterion rejects.

**Q49 — B.** The embedding model silently truncates input beyond its maximum length, so a 1,200-token chunk is represented by a vector derived from its first 512 tokens. Content in the remaining two thirds is invisible to retrieval while still being passed to the generator — hence poor retrieval with no error. Chunk size must be chosen against the *embedding model's* limit, not the generator's.

**Q50 — A.** BIOES adds **E** for the last token of a multi-token span and **S** for a single-token entity, making boundaries explicit rather than inferred. The cost is a tag set roughly twice the size of BIO's, with each tag observed less often — which can hurt on small datasets. Under plain BIO a single-token entity is just `B-X`, and its end is inferred from what follows.

**Q51 — B.** The EU text-and-data-mining exception for commercial purposes applies only where rights have not been expressly reserved, and the Directive contemplates machine-readable reservations for content made publicly available online. A valid reservation removes the exception for that material, so lawful use requires a licence or exclusion. Public accessibility is not permission, and the reservation binds commercial mining specifically.

**Q52 — C.** 3,000 words × 1.35 = **4,050 tokens** per document. With an 8,192-token window, one document plus a 2,000-token instruction block uses 6,050 and fits; two documents plus that block would need 10,100 and does not. Token counts are reliably estimable from word counts for English at roughly 1.3 to 1.4 tokens per word.

**Q53 — C.** Parent entropy = 1.0 bit (even split). Left child, 48/60 positive: H(0.8) = 0.722. Right child, 2/40 positive: H(0.05) = 0.286. Weighted = 0.6(0.722) + 0.4(0.286) = 0.433 + 0.115 = 0.548. Gain = 1.0 − 0.548 = **0.452 bits**. Note this differs numerically from the Gini decrease on the same split — the two criteria rank splits similarly but are not the same quantity.

**Q54 — B.** This is the set where **the example wins decisively**. Exact surface format — whether there is a space before §, whether the page range uses an en-dash or a hyphen, whether the year sits in parentheses before or after the title — is very hard to state completely in prose and trivial to show. P1 supplies a fully worked instance. P3's verbal specification names the field order but leaves every punctuation detail underdetermined. P4 and P5 supply neither, and P2 supplies nothing at all.

**Q55 — D.** Conformal prediction converts a point prediction into a set with a distribution-free marginal coverage guarantee: at 90% calibration, roughly 90% of future sets contain the true label. Set size adapts to difficulty — wide on ambiguous inputs, often a singleton on easy ones — which is what makes the output informative. The guarantee requires only exchangeability, not a correctly specified model.

**Q56 — C.** PCA fitted on all rows computes its components from covariance that includes the test rows, so the representation the model is evaluated on has already been shaped by data it is supposed to be held out from. The estimate is optimistically biased. Fit PCA on the training split and apply the fitted transform to the test split — being unsupervised does not exempt a transform from the split.

**Q57 — A.** Softmax: denominator = 7.389 + 2.718 + 1.105 = 11.212, giving p = [0.659, 0.242, 0.099]. For softmax with cross-entropy the gradient with respect to the logits is the elegant `p − y`, so for the true class it is 0.659 − 1 = **−0.341**. The negative sign is what pushes that logit upward under gradient descent.

**Q58 — B.** `cv2.connectedComponents` defaults to 8-connectivity, which treats diagonal neighbours as connected, so the two blobs touching at a corner merge and the count is **5**. Passing `connectivity=4` treats only edge-sharing pixels as connected and separates them, giving 6. The default is the trap, and this is a standard source of off-by-one component counts.

**Q59 — B.** Writing −∞ into masked positions before the softmax makes their exponential zero, so they receive exactly zero probability and the surviving entries renormalize to sum to one across each row. This is the standard masking idiom for attention and for padded sequences. The tensor keeps its shape; nothing is removed, and −∞ is handled correctly by the exponential.

**Q60 — B.** Questions generated from a single chunk are by construction answerable from it and tend to echo its vocabulary, so the evaluation set is biased towards single-hop lookups that reward lexical overlap — exactly what a dense retriever will already do well. It under-represents multi-hop questions, questions with no answer in the corpus, and questions phrased in user register rather than document register.

**Q61 — D.** A lemmatizer consults a lexicon and part-of-speech information, returning valid dictionary forms: `studies`→`study`, `better`→`good`, `running`→`run`. A rule-based stemmer strips suffixes mechanically, producing `studi` and `run` while leaving the irregular `better` untouched. Stemmer output is frequently not a real word, which is the reverse of what the distractor claims.

**Q62 — A.** Replacing names with identifiers while retaining the mapping is **pseudonymization**: re-identification remains possible, so the data is still personal data and remains fully within the regulation's scope. Pseudonymization is recognized as a security and data-protection measure, not an exit from the rules. Genuine anonymization requires that re-identification be reasonably impossible, which a retained key precludes.

**Q63 — C.** LoRA injects trainable low-rank matrices alongside frozen weight matrices, typically in the attention projections, so the adaptation lives inside the model. Prompt tuning freezes every weight and learns only a small set of continuous 'soft prompt' embeddings prepended to the input sequence — usually far fewer trainable parameters than LoRA, though generally less expressive.

**Q64 — D.** k-NN depends on distances, so a feature measured in thousands dominates one measured in single digits; ridge penalizes squared coefficient magnitude, which depends on each feature's units. Both require standardization. Tree ensembles split on thresholds within a single feature at a time and are invariant to any monotone rescaling of that feature, so neither random forests nor gradient-boosted trees need it.

**Q65 — D.** A 3-point accuracy gain for roughly double the latency and cost is a real but bounded improvement, and whether it is worth paying depends on the task: valuable where a wrong answer is expensive to correct, not where throughput or cost dominates. Self-critique helps on some task families and not others, so 'always on' and 'discard it' are both wrong conclusions from this evidence.

**Q66 — A.** An attribution answers 'why this decision' by decomposing the score; a counterfactual answers 'what would change it' by identifying a minimal feasible change. Only the counterfactual supplies recourse. They are not interchangeable: a feature can carry a large attribution while requiring an infeasible change to flip the outcome, and vice versa.

**Q67 — A.** Hospital admission is a common effect of both conditions, so restricting the sample to admitted patients conditions on a collider and induces an association that does not exist in the population — the classic Berkson's paradox. Unlike confounding, this cannot be fixed by adjusting for more covariates; adjusting for a collider is what *created* the problem. The remedy is a sampling frame that does not condition on the collider.

**Q68 — C.** Adaptive optimizers such as Adam estimate gradient second moments from the steps seen so far, and early in training those estimates are based on very few samples and are correspondingly noisy. Large steps taken on unreliable estimates can push the model into a bad region before the statistics settle. Ramping the learning rate over the first few thousand steps avoids that, and the effect is most pronounced in transformers.

**Q69 — C.** Intersection: x from 50 to 100 gives 50, y from 0 to 100 gives 100, so area = 5,000. Each box is 100 × 100 = 10,000. Union = 10,000 + 10,000 − 5,000 = 15,000. IoU = 5,000/15,000 = **0.333**. Subtracting the intersection once from the summed areas is the step most often skipped.

**Q70 — A.** `tf.data.shuffle(buffer_size)` maintains a buffer of that many elements and samples from it — it is not a full permutation. With a buffer of 1,000 over a file of 1,000,000 records sorted by category, the buffer only ever holds records from a narrow contiguous window, so every batch is near-homogeneous in category. The buffer must be comparable to the dataset size, or the file must be pre-shuffled on disk.

**Q71 — D.** This is small-to-big, or parent-document, retrieval: embed and match on small chunks so the similarity signal stays sharp, then expand each hit to the larger section it belongs to before passing context to the generator. It resolves the tension directly, because retrieval precision and generation context no longer have to be served by the same unit of text.

**Q72 — D.** Perplexity is 2 raised to the mean negative log2 probability. The log2 values are −1, −2, −1, −2, summing to −6; the mean is −1.5; perplexity = 2^1.5 = **2.828**. Interpretively, the model is as uncertain as if choosing uniformly among about 2.8 options at each position.

**Q73 — A.** Legitimate interests is a genuine lawful basis but a conditional one: it requires a documented three-part assessment — the interest is legitimate, the processing is necessary for it, and it is not overridden by the data subject's rights and reasonable expectations. Transparency obligations still apply, and the data subject retains a right to object, which is absolute for direct marketing.

**Q74 — B.** Grammar-constrained decoding masks any token that would make the output unparseable, so validity is guaranteed by construction rather than by probability. The trade-off is real: where the schema conflicts with the model's preferred continuation, forcing it onto a permitted token can degrade content quality. It operates at sampling time and does not modify weights or rely on retries.

**Q75 — A.** Users with no interaction history are the cold-start case, and collaborative methods — matrix factorization, item-based CF — have no signal for them, since latent user factors are inferred *from* interactions that do not exist. With rich item metadata available, content-based recommendation seeded by onboarding preferences is the appropriate approach, typically handing over to collaborative signals as history accumulates.

**Q76 — D.** This is the set where the **most specific prompt is the worst**. The objective is recall with human triage downstream, so P5 — which asks for every potential issue, including uncertain ones, tagged by confidence — is exactly right: it preserves recall while giving the triager the information to prioritize. P4's 'be thorough' nudges the same way weakly, P3 is neutral, P2's persona is implicit. **P1 ranks last**: its confidence filter is precisely specified and directly suppresses the recall the task is optimizing for.

**Q77 — C.** A fixed suite of recorded inputs with expected behaviours, run before each release to catch degradation, is a regression suite. It is reliable for what it covers and blind to everything else — it can only detect failures on inputs someone anticipated and recorded. That is why regression testing is paired with red-teaming and production monitoring rather than relied on alone.

**Q78 — D.** Poisson with λ = 3: P(X = 2) = e⁻³ · 3² / 2! = 0.0498 × 9/2 = 0.0498 × 4.5 = **0.224**. Worth remembering that for a Poisson process the mean and variance are both λ, which is the standard check that a count distribution is Poisson rather than over-dispersed.

**Q79 — B.** Transposed convolution output size = (input − 1) × stride − 2 × padding + kernel = (8 − 1) × 2 − 2 + 4 = 14 − 2 + 4 = **16**. This kernel-4, stride-2, padding-1 configuration is the standard exact-doubling block used throughout generator and decoder architectures, chosen because it avoids the checkerboard artefacts that kernel sizes not divisible by the stride produce.

**Q80 — D.** Downscaling by a factor of ten with nearest-neighbour sampling takes one pixel in a hundred and discards the rest, so spatial frequencies above the new Nyquist limit fold back as false low-frequency structure — aliasing, visible as jaggies and moiré on fine stripes. `INTER_AREA` averages over the full source region for each destination pixel, which low-pass filters before sampling and is the correct choice for downscaling.

**Q81 — D.** `torch.cuda.empty_cache()` returns to the driver only those cached blocks that no live tensor references. Memory held by tensors still in scope stays allocated, and the CUDA context itself occupies several hundred megabytes that is never released while the process lives. It is a tool for reducing fragmentation between phases, not a way to free memory that is genuinely in use.

**Q82 — D.** This is the set where a **negative constraint backfires**. P2 reframes positively — answer from our own range, and here is what to say when asked to compare — which generalizes to every competitor including unnamed ones and supplies a substitute behaviour. P3 generalizes but offers no substitute. **P1 is specific and worse**: it covers only three named firms, leaves every other competitor unaddressed, and writes those three names into context where they can be echoed. P5 and P4 supply nothing operative.

**Q83 — B.** `kitten` → `sitten` (substitute k→s) → `sittin` (substitute e→i) → `sitting` (insert g) = **3 operations**. This is the canonical worked example of Levenshtein distance, and the general algorithm is the dynamic program filling an (m+1)×(n+1) table in O(mn) time.

**Q84 — C.** Placing a high-risk system on the market under its own name makes the bank a **provider** under the Act, inheriting provider obligations — risk management, data governance, technical documentation, conformity assessment, post-market monitoring — for a use (creditworthiness assessment) that is itself listed as high risk. Contractual allocation between the parties does not displace regulatory obligations; it determines who bears the cost of meeting them.

**Q85 — C.** The teacher's probabilities over *wrong* classes encode learned similarity structure — that a given image is somewhat like a truck and not at all like a daffodil. That 'dark knowledge' is a far richer training signal than a one-hot label, which asserts only that every wrong class is equally wrong. It is why distillation with soft targets reaches accuracies a student trained on hard labels alone does not.

**Q86 — B.** Per class: A gives P = 0.800, R = 0.889, F1 = 0.842; B gives P = 0.857, R = 0.750, F1 = 0.800; C gives P = 0.200, R = 0.200, F1 = 0.200. Macro F1 = (0.842 + 0.800 + 0.200)/3 = **0.614**. Micro pools the counts: TP = 75, FP = 35, FN = 35, so P = R = F1 = 75/110 = **0.682**. Macro is lower because it weights the failing minority class equally with the two strong ones — which is precisely why it is the metric to report when minority-class performance matters.

**Q87 — C.** The refresh is driven by `last_modified` on surviving pages, and a deletion changes no surviving page's timestamp — so deleted pages are never revisited and their vectors are never removed. Index freshness requires an explicit deletion path: a tombstone stream, a periodic reconciliation of index IDs against the live corpus, or a full rebuild on a slower cadence.

**Q88 — D.** Determining whether a specific record was in the training set is a **membership-inference attack**. Its success rate rises with overfitting, because a memorized training point receives systematically higher confidence and lower loss than an equally typical unseen point — the confidence gap is the signal. Regularization, differential privacy and limiting exposed confidence scores all reduce it.

**Q89 — C.** ME = 1.96 × √(0.25/1000) = 1.96 × 0.01581 = 0.0310, about **±3.1 percentage points**. Because the margin scales with 1/√n, halving it requires quadrupling the sample: roughly **4,000** respondents. That square-root relationship is the reason survey precision gets expensive quickly.

**Q90 — C.** Gradient norms sit near 2 for 300 steps, then jump to 340 and 1,200 while the loss goes to `nan` — a gradient explosion, usually triggered by an outlier batch or a numerically unstable operation. Clipping the global norm (`clip_grad_norm_`) bounds the update magnitude for the rare spike while leaving ordinary steps, which are far below the threshold, completely unaffected. A decayed learning rate would halt progress, not produce `nan`.

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

Deliberately expensive items to flag on sight: Q2 and Q6 (extended code), Q9 (softmax by hand), Q17 (nDCG with an ideal ordering), Q23 (two baselines through the logistic transform), Q53 (entropy gain), Q57 (softmax gradient), Q86 (six F1 computations).
