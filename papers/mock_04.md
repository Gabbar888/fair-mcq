# AI Assessment — Mock 04
## 90 questions | 150 minutes | MCQ, single correct

Domains are interleaved and unlabelled. Target pace is 100 seconds per question. Some stems are long by design; some topics are ones you may half-remember. Flag and return rather than stalling.

---

**Q1.** A team compares mean response latency across four server configurations using a one-way ANOVA. Each configuration was measured 30 times. The residuals are approximately normal, but the sample variances are 4.1, 4.4, 3.9 and 38.2 milliseconds squared.

Which claim is best supported?

A. The variance spread inflates the within-group mean square, which makes the test conservative rather than liberal.

B. Unequal variances violate an assumption of the standard F test, so Welch's ANOVA is safer.

C. ANOVA requires only normality of residuals, so the variance spread is irrelevant to the validity of the F test.

D. The fourth configuration should be excluded as an outlier group, since its variance exceeds the others by nearly tenfold.

---

**Q2.** A team reports that their classifier's loss decreases during training but the model performs no better than chance on held-out data. The training script is below.

```python
 1   model     = Net()
 2   optimizer = optim.SGD(model.parameters(), lr=0.01)
 3   criterion = nn.CrossEntropyLoss()
 4   train_ds  = TensorDataset(X_train, y_train)
 5   val_ds    = TensorDataset(X_val, y_val)
 6   train_dl  = DataLoader(train_ds, batch_size=32, shuffle=True)
 7   val_dl    = DataLoader(val_ds, batch_size=32, shuffle=False)
 8
 9   for epoch in range(30):
10       model.train()
11       for xb, yb in train_dl:
12           logits = model(xb)
13           loss   = criterion(logits, yb)
14           loss.backward()
15           optimizer.step()
16           optimizer.zero_grad()
17
18       model.eval()
19       correct = 0
20       with torch.no_grad():
21           for xb, yb in val_dl:
22               correct += (model(xb).argmax(1) == yb).sum().item()
23       print(epoch, correct / len(val_ds))
```

Which claim about this script is best supported?

A. The validation loader must use `shuffle=True`, since an unshuffled loader biases the accuracy estimate.

B. `model.train()` must be called inside the batch loop, since the mode resets after each optimizer step.

C. The loop is correct as written, so the cause lies outside this script rather than within it.

D. Calling `zero_grad` after `step` rather than before the forward pass leaves stale gradients in place.

---

**Q3.** A search index weights terms by TF-IDF. The word `the` appears in all 10,000 indexed documents.

Which claim is best supported?

A. Its weight is zero in every document, since the inverse document frequency term evaluates to the log of one.

B. Its weight is high in documents where it occurs most often, since term frequency dominates the product.

C. Its weight is undefined, since dividing the corpus size by a document frequency of 10,000 yields unity.

D. Its weight is small but positive, since term frequency still contributes even when document frequency is maximal.

---

**Q4.** A recruitment model is trained on ten years of a company's hiring decisions. Those decisions favoured one demographic group, though no one involved intended that outcome.

Which form of bias does this most directly illustrate?

A. Deployment bias, since the model is being applied to a screening task that differs from the one it learned.

B. Representation bias, since the training sample contains too few members of the disadvantaged group to learn from.

C. Historical bias, since the training labels faithfully record a pattern that already existed in the world.

D. Measurement bias, since the hiring outcome is a flawed proxy for the job performance the model should predict.

---

**Q5.** A team replaces sigmoid activations with ReLU throughout a deep feedforward network.

Which claim is best supported?

A. ReLU's gradient is constant for positive inputs, so it does not shrink the signal passing back through many layers.

B. ReLU outputs lie in a bounded range, which keeps activations stable as they propagate through successive layers.

C. ReLU is differentiable everywhere, which is why gradient descent converges more reliably than it does with sigmoid.

D. ReLU centres its outputs around zero, which is the property that lets gradients survive across many layers.

---

**Q6.** An image classification service loads photographs, prepares them, and runs a model trained on RGB inputs. It produces predictions without error, but accuracy is far below what the same model achieves on the project's own test harness.

```python
 1   import cv2, numpy as np, torch
 2
 3   def load_batch(paths, size=224):
 4       out = []
 5       for p in paths:
 6           img = cv2.imread(p, cv2.IMREAD_COLOR)
 7           img = cv2.resize(img, (size, size))
 8           img = img.astype(np.float32) / 255.0
 9           out.append(img)
10       arr = np.stack(out)                       # (B, 224, 224, 3)
11       t   = torch.from_numpy(arr).permute(0, 3, 1, 2)
12       return t
13
14   batch  = load_batch(image_paths)
15   logits = model(batch)
16   preds  = logits.argmax(1)
```

Which change corrects the behaviour?

A. Replace `np.stack` with `np.concatenate` on line 10, since stacking adds an axis the model does not expect.

B. Convert each image from BGR to RGB, since `imread` reverses the channel order.

C. Call `permute(0, 2, 3, 1)` on line 11, since the channel axis belongs last.

D. Divide by 255.0 after the permute on line 11, since scaling must follow the channel reordering.

---

**Q7.** A team is assembling a retrieval-augmented generation pipeline and must order these stages: embed the query, retrieve nearest chunks, generate the answer, embed and index the documents.

Which ordering is correct?

A. Embed and index the documents, embed the query, retrieve nearest chunks, generate the answer.

B. Embed the query, embed and index the documents, retrieve nearest chunks, generate the answer.

C. Embed and index the documents, retrieve nearest chunks, embed the query, generate the answer.

D. Embed the query, retrieve nearest chunks, embed and index the documents, generate the answer.

---

**Q8.** A team splits its data into training, validation and test sets, tunes hyperparameters against the validation set over roughly 200 configurations, and reports the validation score of the best configuration.

Which claim is best supported?

A. The validation score is an unbiased estimate, since the validation set was never used to fit model parameters.

B. The reported score is optimistic, since selecting the maximum over many configurations fits the validation set.

C. The test set is redundant here, since the validation set already provides a held-out measure of performance.

D. Two hundred configurations is too few to produce selection bias, which requires thousands of repeated trials.

---

**Q9.** A colleague asks what self-attention computes in a transformer layer.

Which description is best supported?

A. It produces for each position a weighted combination of all positions, with weights derived from learned projections.

B. It compares each position only with its immediate neighbours, with the window size fixed by the layer's configuration.

C. It assigns each position a fixed importance score learned during training and reused for every input sequence.

D. It summarizes the whole sequence into a single vector, which every position then reads during the forward pass.

---

**Q10.** A notification system sends alerts by SMS, hard-capped at 160 characters. The objective is to **maximize the proportion of generated alerts that fit without truncation**.

- P1: "Write the alert in no more than 160 characters, including spaces. Count before replying."
- P2: "Write a short alert."
- P3: "Write the alert concisely, keeping it well under a typical message length."
- P4: "You are an experienced operations engineer. Write a brief alert for the on-call team."
- P5: "Write the alert. Example: 'DB-07 replication lag 340s since 14:02. Failover not triggered. Check disk I/O.'"

Rank the prompts from most to least effective against this objective.

A. P3 > P1 > P5 > P2 > P4

B. P1 > P5 > P3 > P2 > P4

C. P5 > P1 > P3 > P4 > P2

D. P1 > P3 > P5 > P4 > P2

---

**Q11.** A team must explain an individual loan decision to the applicant who received it.

Which claim about SHAP and LIME is best supported?

A. Both require the model to be a tree ensemble, since neither can approximate the behaviour of a neural network.

B. LIME attributions sum exactly to the prediction, which is the property that makes them suitable for disclosure.

C. Both explain a single prediction, and both work without access to the model's internals.

D. SHAP explains individual predictions while LIME produces only a ranking of features averaged across the dataset.

---

**Q12.** A team measures the relationship between a service's queue depth and its p99 latency. The relationship is strictly increasing but curves sharply upward, and the data contain several extreme queue-depth readings.

Which measure is most appropriate?

A. Pearson correlation computed after removing the extreme readings, since those points distort any rank-based measure.

B. Pearson correlation, since it is the standard measure and the relationship is strictly increasing throughout.

C. Spearman correlation, since it measures monotone association using ranks rather than raw values.

D. Covariance, since it preserves the original units and therefore conveys the strength of the relationship directly.

---

**Q13.** A colleague asks how a GRU differs from an LSTM.

Which claim is best supported?

A. A GRU replaces the gating mechanism with attention, which is why it handles long dependencies more effectively.

B. A GRU has two gates and no separate cell state, carrying fewer parameters.

C. A GRU has three gates and a separate cell state, so it trains faster.

D. A GRU processes sequences bidirectionally by default, while an LSTM requires an explicit wrapper to do so.

---

**Q14.** A tensor `x` has shape `(32, 3, 8, 8)`. A developer wants a tensor of shape `(32, 192)` to feed a linear layer.

Which claim is best supported?

A. `x.view(32, -1)` and `x.reshape(32, -1)` both fail, since neither can collapse three axes into one.

B. `x.flatten()` produces the required shape, since flatten preserves the leading batch dimension by default.

C. `x.view(-1, 192)` is required, since the batch dimension must be inferred rather than stated explicitly.

D. `x.view(32, -1)` works only if `x` is contiguous, while `x.reshape(32, -1)` succeeds regardless by copying when needed.

---

**Q15.** A 3×3 kernel with every entry equal to one ninth is convolved across a photograph.

Which claim is best supported?

A. Edges are enhanced, since each output pixel is the sum of its neighbourhood and therefore amplifies contrast.

B. The image is unchanged, since the kernel's entries sum to one and the transformation preserves intensity.

C. The image is blurred, since each output pixel becomes the average of its own neighbourhood of nine pixels.

D. The image is sharpened, since uniform weighting removes low-frequency content and leaves high-frequency detail.

---

**Q16.** A team removes stop words before training a document classifier and sees accuracy improve. They then apply the same preprocessing to a sentiment model and accuracy falls.

Which claim is best supported?

A. Sentiment labels are noisier than topic labels, so the same preprocessing yields less stable results on them.

B. Stop-word lists typically contain negation terms, whose removal discards information sentiment models depend on.

C. Sentiment models require larger vocabularies, so any vocabulary reduction degrades them regardless of which words are removed.

D. Stop-word removal should always precede tokenization, and the sentiment pipeline likely applied the two in the wrong order.

---

**Q17.** A retrieval system returns 10 ranked documents for a query. Documents at ranks 1, 2, 4 and 9 are relevant; the corpus contains 12 relevant documents overall.

What are precision@5 and recall@5?

A. Precision 0.40 and recall 0.33

B. Precision 0.60 and recall 0.33

C. Precision 0.40 and recall 0.25

D. Precision 0.60 and recall 0.25

---

**Q18.** A hospital plans to deploy an AI system that triages incoming patients by urgency, directing them to different care pathways.

How does the EU AI Act most likely classify this?

A. Limited risk, since the obligation is to inform patients that an automated system contributed to triage.

B. Minimal risk, since a clinician reviews every pathway assignment before the patient is actually directed.

C. High risk, since healthcare access falls within one of the listed categories.

D. Prohibited, since any system that influences access to medical treatment falls within the unacceptable tier.

---

**Q19.** A team estimates how much text fits in a model's context window and must convert between words and tokens for English prose.

Which claim is best supported?

A. Tokens and words correspond one to one, so a 4,000-word document occupies roughly 4,000 tokens of context.

B. A token is one character, so a 4,000-word document occupies roughly 20,000 tokens of available context.

C. Tokens are subword units, so English runs somewhat above one token per word.

D. A token is one sentence, so length in tokens depends on writing style.

---

**Q20.** A model achieves 0.62 accuracy on its training set and 0.60 on its validation set.

Which claim is best supported?

A. The model is overfitting, since any gap between training and validation accuracy indicates memorization.

B. The model is underfitting, since it performs poorly on data it was directly fitted to in the first place.

C. The model is well regularized, since the small gap between the two scores indicates good generalization.

D. The result is uninterpretable without the test score, since validation data is used for selection rather than evaluation.

---

**Q21.** A question-answering assistant serves an internal knowledge base. The objective is to **minimize confidently stated answers to questions the knowledge base does not cover**.

- P1: "Answer the question using the knowledge base."
- P2: "Answer the question using the knowledge base. Be accurate."
- P3: "You are a meticulous knowledge-base specialist. Answer the question."
- P4: "Answer the question using the knowledge base. Example: 'The retention period is 90 days, per the Data Policy.'"
- P5: "Answer using the knowledge base. If it does not contain the answer, reply 'Not covered' and stop."

Rank the prompts from most to least effective against this objective.

A. P5 > P2 > P4 > P3 > P1

B. P2 > P5 > P4 > P1 > P3

C. P5 > P4 > P2 > P1 > P3

D. P4 > P5 > P2 > P3 > P1

---

**Q22.** A random forest reports that `customer_id` is among its most important features, though the identifier is arbitrary and carries no information.

Which claim is best supported?

A. The identifier must correlate with the target through the order in which records were originally collected.

B. Random forests cannot use identifier columns, so the reported importance indicates a data-type conversion error.

C. The importance is accurate, since a feature the trees split on reduced impurity somewhere.

D. Impurity-based importance favours features with many distinct values, since these offer more candidate split points.

---

**Q23.** A logistic regression predicting subscription cancellation has a coefficient of −0.85 on `months_subscribed`.

Which claim is best supported?

A. Each additional month multiplies the odds of cancellation by about 0.43, lowering the predicted probability.

B. Each additional month lowers the predicted probability of cancellation by 85 percentage points from its baseline.

C. Each additional month multiplies the odds of cancellation by about 2.34, since the coefficient enters as an exponent.

D. The negative sign indicates the feature was not standardized, since logistic coefficients are positive by construction.

---

**Q24.** A dataset holds 50,000 training examples and a `DataLoader` is configured with `batch_size=100`.

How many optimizer steps occur in one epoch, assuming one step per batch?

A. 500

B. 50,000

C. 5,000

D. 100

---

**Q25.** A 4,000×3,000 photograph must be reduced to 400×300 for a thumbnail grid.

Which interpolation method is most appropriate?

A. `INTER_LINEAR`, since bilinear weighting of the four nearest pixels is the standard default for any resize.

B. `INTER_AREA`, since averaging over the source region each output pixel covers avoids aliasing when shrinking.

C. `INTER_NEAREST`, since sampling without averaging preserves the original pixel values exactly as captured.

D. `INTER_CUBIC`, since fitting a cubic surface recovers detail that would otherwise be lost in reduction.

---

**Q26.** A `DataLoader` is created over a dataset of 1,050 samples with `batch_size=100` and `drop_last=True`.

Which claim is best supported?

A. One epoch yields 11 batches, with the final batch padded back up to 100.

B. One epoch yields 10 batches, and the 50 remaining samples are redistributed across the preceding batches.

C. One epoch yields 11 batches, and the final batch contains the 50 remaining samples after the full ones.

D. One epoch yields 10 batches, and the remaining 50 samples are skipped.

---

**Q27.** An embedding model produces vectors that are not unit-normalized. A team must rank documents by similarity to a query.

Which claim is best supported?

A. Dot product and cosine similarity can rank differently, since dot product is also affected by vector magnitude.

B. Euclidean distance must be used, since cosine similarity is defined only for vectors of unit length.

C. Cosine similarity cannot be computed until the vectors are normalized as a separate preprocessing step.

D. Cosine similarity and dot product give identical rankings, since both measure the angle between two vectors.

---

**Q28.** A tokenizer encounters the word `hyperparameterization`, which did not appear in its training corpus.

Which claim is best supported for a subword tokenizer?

A. Tokenization raises an error, requiring the word to be added to the vocabulary before the text is encoded.

B. The word is replaced by a single unknown token, since it is absent from the tokenizer's learned vocabulary.

C. The word is split into several known subword units, so no unknown token is needed.

D. The word is split into individual characters, since subword merges apply only to previously observed words.

---

**Q29.** A company deploys a customer-service chatbot that writes fluently and conversationally about product questions.

Which obligation does this most directly engage under the EU AI Act?

A. Post-market monitoring, requiring the provider to collect performance data throughout the product's lifetime.

B. Data governance, requiring that training data be examined for bias and documented before deployment begins.

C. Transparency, requiring that users be informed they are interacting with an AI system rather than a person.

D. Conformity assessment, requiring independent evaluation before the system may be placed on the market.

---

**Q30.** A generation service raises its sampling temperature from 0.2 to 1.1.

Which claim is best supported?

A. Outputs become more deterministic, since a higher temperature sharpens the distribution towards its mode.

B. Outputs become longer on average, since temperature governs how readily the model emits an end-of-sequence token.

C. Outputs are unaffected in content, since temperature rescales logits uniformly and preserves their ordering.

D. Outputs become more varied, since the distribution over tokens is flattened.

---

**Q31.** A team uses 5-fold cross-validation rather than a single train-validation split on a dataset of 3,000 records.

Which claim is best supported?

A. Every record is validated once and used for training in the other four folds.

B. Every record serves in validation exactly once and in training once, which is what removes the selection bias.

C. The procedure trains a single model five times faster, since each fold uses only a fifth of the data.

D. The five fold scores are independent, so their standard deviation estimates the sampling error directly.

---

**Q32.** An agent is asked to book a meeting room and has tools for listing rooms and for creating a booking. The user says only "book me a room for tomorrow afternoon".

Which behaviour is most appropriate?

A. Create bookings in every available room, then cancel all but the one the user confirms afterwards.

B. Create a booking in the first room the listing returns, then tell the user which room was selected.

C. Ask the user for the duration and attendee count, then list rooms and create the booking accordingly.

D. List the available rooms and take no booking action until the user chooses.

---

**Q33.** A deployed model's inputs shift over several months while the relationship between inputs and the target stays the same.

Which term describes this, and what follows?

A. Covariate shift, where the model may still degrade because it now operates in regions it saw rarely.

B. Label shift, which is detected by comparing the distribution of the target against its training distribution.

C. Data leakage, which arises whenever the deployment distribution differs from the one used during training.

D. Concept drift, which usually requires retraining because the learned mapping no longer holds.

---

**Q34.** A disease affects 2% of a population. A test has sensitivity 0.90 and specificity 0.95. A patient tests positive.

What is the probability the patient has the disease?

A. About 0.02

B. About 0.50

C. About 0.90

D. About 0.27

---

**Q35.** A network uses dropout with p = 0.3 during training.

Which claim is best supported?

A. Thirty per cent of the units are permanently removed, reducing the network's capacity for the whole run.

B. Each unit is independently zeroed with probability 0.3 on every forward pass.

C. Thirty per cent of the training samples are skipped in each epoch.

D. Each unit's output is scaled by 0.3 at inference time, which compensates for the units dropped in training.

---

**Q36.** A colleague proposes replacing a small convolutional network with a fully connected network of similar parameter count for an image task.

Which claim is best supported?

A. The fully connected network will generalize comparably, since both are universal approximators at equal capacity.

B. The convolutional network shares weights across spatial positions, so features transfer anywhere.

C. The fully connected network trains faster, since it avoids the sliding-window computation convolution requires.

D. The two are mathematically equivalent, since a convolution can be written as a sparse fully connected layer.

---

**Q37.** A developer wants a tensor derived from a model's output to be excluded from gradient computation, while leaving the original output differentiable.

Which approach is correct?

A. Wrap the later computation in `torch.no_grad()`, which retroactively detaches `y` from everything before it.

B. Use `y.clone()`, which allocates fresh storage and therefore produces a tensor autograd no longer tracks.

C. Use `y.detach()`, which returns a new tensor sharing the values but excluded from the computation graph.

D. Use `y.requires_grad_(False)`, which marks the derived tensor while leaving the original fully differentiable.

---

**Q38.** A RAG system answers a question incorrectly. Logs show the retrieved chunks did not contain the information needed to answer it, although the corpus does contain it.

Which stage is at fault, and what follows?

A. Chunking, since a chunk containing the answer would always be retrieved if the boundaries were correct.

B. Indexing, since any chunk absent from the retrieved set is necessarily absent from the index as well.

C. Generation, since the model should have recognized the gap and declined rather than producing an answer.

D. Retrieval, since the relevant chunk existed but was not ranked highly enough to reach the generator.

---

**Q39.** A team trains word embeddings on a product-review corpus and inspects the vector for `light`.

Which claim is best supported?

A. The vector is sparse, with one non-zero dimension for each corpus document in which the word occurs.

B. The vector is undefined for ambiguous words, which must be disambiguated before embeddings are trained.

C. The vector blends the word's senses, since static embeddings assign one vector per type.

D. The model stores a separate vector for each sense, selected at lookup time by the surrounding context.

---

**Q40.** A bank deploys a credit model and assigns an analyst to review every borderline decision. The analyst has authority to override the model and is trained on its known failure modes.

Which arrangement does this describe?

A. Human-in-command, since the analyst sets the operating conditions rather than reviewing individual decisions.

B. Automated decision-making, since the model produces the score on which the final outcome ultimately rests.

C. Human-in-the-loop, since a person reviews individual outputs and can change the outcome before it takes effect.

D. Human-on-the-loop, since the system produces decisions autonomously while a person supervises the aggregate.

---

**Q41.** A model has a context window of 8,192 tokens. A user pastes a document of 12,000 tokens.

Which claim is best supported?

A. The excess must be removed before the call, by truncating or retrieving selectively.

B. The model automatically extends its window, since positional encodings generalize to any input length.

C. The model processes the first 8,192 tokens and silently appends the remainder on a second internal pass.

D. The model processes the document at reduced attention resolution, which lowers answer quality proportionally.

---

**Q42.** A team replaces a single decision tree with a random forest of 300 trees on the same data.

Which claim is best supported?

A. Variance falls only if every tree is grown to full depth, since shallow trees are too correlated to average.

B. Variance falls because predictions are averaged across trees trained on different bootstrap samples and feature subsets.

C. Bias falls because each successive tree is fitted to the errors the previous trees in the ensemble made.

D. Both bias and variance fall, since averaging more models always improves every component of the error.

---

**Q43.** A children's educational app explains science concepts. The objective is to **match the register and vocabulary of an eight-year-old reader**.

- P1: "Explain the concept simply."
- P2: "Explain the concept in no more than 60 words."
- P3: "You are writing for an eight-year-old. Use everyday words, short sentences, and a familiar comparison."
- P4: "Explain the concept, avoiding technical jargon wherever you reasonably can."
- P5: "Explain the concept. Example: 'Sound is a wiggle in the air, like the wobble on a drum skin.'"

Rank the prompts from most to least effective against this objective.

A. P3 > P4 > P5 > P2 > P1

B. P4 > P3 > P5 > P2 > P1

C. P5 > P3 > P4 > P1 > P2

D. P3 > P5 > P4 > P1 > P2

---

**Q44.** A weather model forecasts rain with probability 0.70 on 100 occasions. Rain occurs on 52 of them.

Which claim is best supported?

A. The model is accurate, since rain occurred on more than half the occasions it was forecast as likely.

B. The model is well calibrated, since a 0.70 forecast is a statement about confidence rather than frequency.

C. The comparison is invalid, since calibration requires the same weather event to recur under identical conditions.

D. The model is poorly calibrated in this range, since forecast probability and observed frequency diverge substantially.

---

**Q45.** A team reports a 95% confidence interval for a mean and wants it narrower.

Which change achieves this?

A. Raising the confidence level to 99%, which tightens the interval by demanding greater certainty.

B. Increasing the population variance, which spreads the sampling distribution and sharpens the estimate.

C. Reducing the sample size, which concentrates the estimate on fewer and therefore more consistent observations.

D. Increasing the sample size, which reduces the standard error and therefore narrows the interval.

---

**Q46.** A team must choose a loss function for a model predicting house prices in rupees.

Which choice is most appropriate, and why?

A. Mean squared error, since the target is continuous and squared deviation penalizes larger errors more heavily.

B. Cross-entropy loss, since it handles continuous targets by treating each price bracket as a separate class.

C. Hinge loss, since it penalizes predictions on the wrong side of a margin and is robust to price outliers.

D. Binary cross-entropy, since prices can be normalized to the unit interval before the loss is computed.

---

**Q47.** A team trains a model to classify photographs of street signs and adds random horizontal flipping to the augmentation pipeline.

Which claim is best supported?

A. Flipping is safe for all image classification tasks, since a mirrored object remains the same object.

B. Flipping is harmful here, since mirrored text and directional arrows change meaning.

C. Flipping is harmful because it doubles the dataset size, which requires a proportionally longer training schedule.

D. Flipping is safe here, since the network learns orientation-invariant features from the convolution operation.

---

**Q48.** Which implementation correctly performs one standard training iteration?

A.
```python
optimizer.step()
loss = criterion(model(xb), yb)
loss.backward()
optimizer.zero_grad()
```

B.
```python
optimizer.zero_grad()
loss = criterion(model(xb), yb)
loss.backward()
optimizer.step()
```

C.
```python
loss = criterion(model(xb), yb)
optimizer.step()
loss.backward()
optimizer.zero_grad()
```

D.
```python
loss = criterion(model(xb), yb)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

**Q49.** A RAG team considers reducing chunk size from 800 tokens to 200.

Which claim is best supported?

A. Neither is affected, since chunk size influences only the storage footprint of the resulting vector index.

B. Retrieval becomes more precise while each chunk carries less surrounding context for the generator to use.

C. Retrieval becomes less precise, since shorter chunks produce embeddings that are noisier and less distinctive.

D. Both retrieval precision and generation quality improve, since smaller units are easier to match and to read.

---

**Q50.** A named entity recognition model tags the sentence *"Priya joined Infosys in Pune last year"*.

Which output is correctly formed under the BIO scheme?

A. `I-PER O I-ORG O I-LOC O O`

B. `B-PER O O B-ORG B-LOC O O`

C. `PER O ORG O LOC O O`

D. `B-PER O B-ORG O B-LOC O O`

---

**Q51.** A regulator asks a company to document a deployed model's intended uses, its performance across demographic subgroups, and its known limitations.

Which artifact is designed for this purpose?

A. A model card, which records intended use, out-of-scope use, disaggregated evaluation and known limitations.

B. A privacy notice, which informs data subjects how their personal data is processed by the deployed system.

C. A datasheet, which records the provenance, collection method and composition of the training dataset.

D. An audit log, which records the inputs and outputs of the system so decisions can be reconstructed later.

---

**Q52.** A colleague asks why transformers need positional encoding.

Which explanation is best supported?

A. Self-attention would otherwise attend only to nearby tokens, and position information widens that effective window.

B. Self-attention treats its input as an unordered set, so order carries no signal.

C. Positional encoding compresses long sequences, which is what allows attention to scale to large context windows.

D. Positional encoding replaces the recurrence of an RNN, which is the only way gradients reach earlier positions.

---

**Q53.** A binary classifier produces 30 true positives, 20 false positives, 10 false negatives and 940 true negatives.

What are its precision and recall?

A. Precision 0.60 and recall 0.60

B. Precision 0.75 and recall 0.75

C. Precision 0.75 and recall 0.60

D. Precision 0.60 and recall 0.75

---

**Q54.** An internal tool converts meeting notes into a fixed ticket format. The objective is to **match the required output format exactly**, including field order and separators.

- P1: "Convert these notes into a ticket in our standard format."
- P2: "Convert these notes into a ticket. Format: `[PRIORITY] OWNER :: summary -- due YYYY-MM-DD`. Example: `[P2] rahul :: restart ingest worker -- due 2026-03-04`."
- P3: "Convert these notes into a ticket with priority, owner, summary and due date included."
- P4: "You are an experienced project coordinator. Convert these notes into a well-formed ticket."
- P5: "Convert these notes into a ticket, taking care to follow our formatting conventions closely."

Rank the prompts from most to least effective against this objective.

A. P3 > P2 > P4 > P1 > P5

B. P3 > P2 > P5 > P4 > P1

C. P2 > P3 > P5 > P1 > P4

D. P2 > P5 > P3 > P4 > P1

---

**Q55.** A model returns high predictive uncertainty on inputs drawn from a region of the feature space absent from its training data.

Which type of uncertainty is this, and what reduces it?

A. Aleatoric uncertainty, which arises from limited model capacity and falls when a larger architecture is used.

B. Epistemic uncertainty, which reflects the model's ignorance and falls as relevant training data is added.

C. Aleatoric uncertainty, which reflects noise inherent to the measurement process and cannot be reduced by data.

D. Epistemic uncertainty, which reflects irreducible randomness in the outcome and is unaffected by additional data.

---

**Q56.** A team must prepare features measured in rupees, years and counts before fitting a k-nearest-neighbours classifier.

Which claim is best supported?

A. Standardization is needed, since the distance metric is dominated by whichever feature has the largest numeric range.

B. Standardization is harmful, since rescaling distorts the geometry that nearest-neighbour search depends upon.

C. Standardization matters only for the target variable, since predictor scale affects distance computations negligibly.

D. Standardization is unnecessary, since k-NN compares feature values within each column rather than across columns.

---

**Q57.** A colleague asks what backpropagation computes.

Which description is best supported?

A. The second derivative of the loss, which determines the curvature used to choose an appropriate step size.

B. The loss value for each training example, which is then averaged across the batch before the update.

C. The gradient of the loss with respect to every parameter, obtained by applying the chain rule backwards through the graph.

D. The updated parameter values for the next step, which the optimizer then applies directly to the model.

---

**Q58.** A team converts colour photographs to grayscale before edge detection.

Which claim is best supported?

A. Grayscale conversion weights the channels unequally, reflecting the eye's differing sensitivity to each colour.

B. Grayscale conversion discards intensity information, which is why edge detection is applied to colour images instead.

C. Grayscale conversion is required by edge detectors, which cannot operate on arrays of more than two dimensions.

D. Grayscale conversion averages the three channels equally, since each contributes identically to perceived brightness.

---

**Q59.** A tensor `a` has shape `(8, 1, 6)` and a tensor `b` has shape `(5, 6)`. The expression `a + b` is evaluated.

What is the resulting shape?

A. A RuntimeError from incompatible shapes

B. `(5, 8, 6)`

C. `(8, 5, 6)`

D. `(8, 1, 6)`

---

**Q60.** A team stores 2,000,000 document embeddings and must retrieve the nearest neighbours of a query vector in a few milliseconds.

Which claim is best supported?

A. An approximate index trades a small amount of recall for a large reduction in search latency at this scale.

B. Latency depends only on embedding dimension, so reducing dimensionality is the sole available lever here.

C. Exact search is already fast at this scale, since comparing two million vectors is a trivial computation.

D. Exact nearest-neighbour search is required, since approximate methods cannot guarantee any useful recall level.

---

**Q61.** A team compares stemming and lemmatization on the word `was`.

Which claim is best supported?

A. A lemmatizer returns `be`, since it maps inflected forms to their dictionary entry using lexical knowledge.

B. A stemmer returns `be`, since suffix-stripping rules handle irregular verbs through an exception table.

C. Both return `wa`, since the two techniques differ only in the aggressiveness of the suffix rules applied.

D. Neither alters the word, since three-letter tokens fall below the minimum length both techniques require.

---

**Q62.** A dataset has customer names replaced by random codes, with the mapping held in a separate table the company retains.

Which claim is best supported?

A. The dataset is anonymized and falls outside the regulation, since the codes themselves reveal no identity.

B. The dataset is pseudonymized and remains personal data, since the retained mapping permits re-identification.

C. The dataset is anonymized provided the mapping is held by a different team within the same organization.

D. The distinction has no practical consequence, since both treatments satisfy the same regulatory obligations.

---

**Q63.** A team must adapt a large language model to a specialized domain and is weighing fine-tuning against prompting with examples.

Which claim is best supported?

A. Fine-tuning and prompting are equivalent in effect, differing only in whether the cost is paid once or per call.

B. Fine-tuning requires no labelled examples, since the model infers the task from the domain corpus alone.

C. Fine-tuning changes the model's weights, while prompting changes only what the model is shown at inference.

D. Prompting changes the model's weights gradually, which is why its effects persist across separate sessions.

---

**Q64.** A team evaluating a rare-disease screening model reports 99.1% accuracy. The disease affects 0.9% of those screened.

Which claim is best supported?

A. The accuracy is misleading only if the test set was resampled, since a natural class balance keeps it valid.

B. Accuracy is the correct headline metric, provided precision and recall are reported alongside it for context.

C. The accuracy is strong evidence of a working model, since it substantially exceeds random guessing at two classes.

D. A model predicting "no disease" for everyone would score 99.1%, so the figure carries no evidence by itself.

---

**Q65.** An agent is given a calculator tool and asked to compute a compound figure across several steps. Its trace shows it performed all arithmetic internally and never invoked the tool.

Which claim is best supported?

A. The system prompt should direct the agent to use the calculator for all arithmetic rather than leaving it optional.

B. The calculator schema must be malformed, since an available tool is always invoked when its description matches.

C. Arithmetic errors are impossible at this scale, so tool use would add cost without improving the result.

D. The behaviour is correct, since internal arithmetic avoids the latency a tool call would otherwise introduce.

---

**Q66.** A team must choose between a logistic regression and a gradient-boosted ensemble for a regulated lending decision.

Which claim is best supported?

A. Logistic regression is interpretable by construction, while the ensemble requires post-hoc methods to approximate its behaviour.

B. The ensemble is preferable regardless, since regulators accept any model whose predictions are accurate enough.

C. Logistic regression cannot be used here, since regulated decisions require models that capture feature interactions.

D. The ensemble is equally interpretable, since SHAP attributions reconstruct its decision rule for a given input.

---

**Q67.** A trial fails to reject the null hypothesis at α = 0.05.

Which claim is best supported?

A. The alternative hypothesis is false, since a test at the conventional level would have detected a real effect.

B. The evidence was insufficient to reject at this level, which is not the same as evidence that no effect exists.

C. The test should be repeated at α = 0.10, since the conventional level is too strict for exploratory work.

D. The null hypothesis has been shown to be true, since the data provided no evidence against it at this level.

---

**Q68.** A model trains stably at a learning rate of 0.001. The team raises it to 0.5.

Which outcome is most likely?

A. Training is unaffected, since adaptive optimizers rescale the learning rate internally to a stable value.

B. Training converges faster to the same minimum, since larger steps traverse the loss surface more efficiently.

C. Training becomes unstable, with the loss oscillating or diverging as updates overshoot the descent direction.

D. Training slows, since large steps require more iterations before the parameters settle near a minimum.

---

**Q69.** Two bounding boxes are given as `(x1, y1, x2, y2)`: box A is `(0, 0, 20, 20)` and box B is `(10, 10, 30, 30)`.

What is their intersection over union?

A. 0.143

B. 0.500

C. 0.100

D. 0.250

---

**Q70.** A Keras model is compiled with `metrics=['accuracy']` alongside a categorical cross-entropy loss.

Which claim is best supported?

A. The metric is minimized alongside the loss, since every quantity passed to `compile` enters the objective.

B. The metric is reported during training but does not contribute to the gradients the optimizer uses.

C. The metric replaces the loss once training accuracy exceeds the loss value on a given epoch.

D. The metric must be differentiable, since Keras computes gradients for every quantity it tracks.

---

**Q71.** A RAG team adds a 100-token overlap between consecutive 600-token chunks.

Which claim is best supported?

A. Overlap guarantees that any chunk retrieved will contain the complete section it was drawn from.

B. Overlap reduces index size, since shared regions are stored once and referenced by both neighbouring chunks.

C. Overlap raises the chance that a statement spanning a boundary appears intact within at least one chunk.

D. Overlap improves embedding quality, since repeated text gives the encoder more context to work with.

---

**Q72.** A bag-of-words representation is built for the sentences *"the model beat the baseline"* and *"the baseline beat the model"*.

Which claim is best supported?

A. The two differ in one dimension, since relative word position enters the count vector.

B. The two differ substantially, since each word's weight depends on where it occurs.

C. The representation is undefined, since both sentences contain the word `the` more than once.

D. The two sentences receive identical representations, since bag-of-words records counts and discards word order.

---

**Q73.** A company wishes to use customer purchase data to train a recommendation model. The data was collected for order fulfilment.

Which claim is best supported under the GDPR?

A. The new use must be assessed for compatibility with the original purpose, or a fresh lawful basis obtained.

B. No assessment is needed, since the company already holds the data lawfully for its original stated purpose.

C. Training is exempt from purpose limitation, since a model does not store the underlying records themselves.

D. Consent is the only available basis for any secondary use, regardless of how compatible it may be.

---

**Q74.** A team compares top-k and nucleus (top-p) sampling for text generation.

Which claim is best supported?

A. The two are identical when k equals p, since both truncate the distribution at the same cumulative point.

B. Top-k is deterministic while top-p introduces randomness, which is the main consideration in choosing between them.

C. Top-k keeps a fixed number of candidates while top-p keeps the smallest set whose probabilities reach a threshold.

D. Top-p keeps a fixed number of candidates while top-k adapts the count to the shape of the distribution.

---

**Q75.** A logistics firm has six months of delivery records with no outcome labels and wants to discover recurring patterns in route behaviour.

Which approach fits this situation?

A. Supervised regression, predicting delivery duration so that the residuals reveal the underlying patterns.

B. Reinforcement learning, since the absence of labels means the model must learn from a reward signal instead.

C. Supervised classification, using the route identifier as a label so each pattern becomes its own class.

D. Unsupervised clustering, which groups records by similarity without requiring any outcome labels at all.

---

**Q76.** An assistant answers quantitative business questions involving several arithmetic steps. The objective is to **maximize the proportion of answers that are numerically correct**.

- P1: "Answer the question."
- P2: "Answer the question. Respond with only the final figure and no explanation."
- P3: "You are a seasoned financial analyst. Answer the question."
- P4: "Answer the question. Work through the calculation step by step before giving the final figure."
- P5: "Answer the question. Check your arithmetic carefully before replying."

Rank the prompts from most to least effective against this objective.

A. P4 > P5 > P3 > P1 > P2

B. P4 > P5 > P1 > P3 > P2

C. P5 > P4 > P1 > P2 > P3

D. P2 > P4 > P5 > P1 > P3

---

**Q77.** A team deploys a model whose ground-truth labels arrive six months after each prediction.

Which monitoring approach is most appropriate in the interim?

A. Monitor nothing until labels arrive, since any earlier measurement cannot establish whether accuracy has fallen.

B. Monitor the training loss, since it continues to reflect the model's behaviour after deployment has begun.

C. Monitor only the volume of predictions served, since throughput is the one signal available before labels mature.

D. Monitor the input feature distributions and the distribution of predicted scores against a reference window.

---

**Q78.** A survey must estimate satisfaction across a population containing a small but important group of enterprise customers alongside many individual users.

Which sampling method is most appropriate?

A. Convenience sampling, since the enterprise customers are easier to reach and respond at higher rates.

B. Cluster sampling, since drawing whole customer segments reduces cost while preserving overall representativeness.

C. Stratified sampling, since dividing by customer type ensures the small group is adequately represented.

D. Simple random sampling, since every member has an equal chance and the estimate is therefore unbiased.

---

**Q79.** An input of shape `(N, 3, 32, 32)` passes through `Conv2d(3, 16, kernel_size=3, padding=1)` followed by `MaxPool2d(2)`, then `Conv2d(16, 32, kernel_size=3, padding=1)` followed by `MaxPool2d(2)`, then a flatten.

What is the flattened length per sample?

A. 1,024

B. 8,192

C. 4,096

D. 2,048

---

**Q80.** A photograph is loaded and its shape is reported as `(1080, 1920, 3)`.

Which claim is best supported?

A. The image is 1,080 pixels wide and 1,920 tall, with three channels, since NumPy reports width before height.

B. The image is 1,920 pixels wide and 1,080 tall, with three channels, since NumPy reports rows before columns.

C. The image has 1,080 channels, since the leading axis of a NumPy array always holds the channel dimension.

D. The image is 1,080 by 1,920 in an unspecified order, since the array shape cannot distinguish the two axes.

---

**Q81.** A model is saved with `torch.save(model.state_dict(), "m.pt")` and must be loaded in a different script.

Which claim is best supported?

A. The optimizer state is included by default, so training can resume from the file without further saving.

B. The file is device-specific, so a model saved on GPU cannot be loaded on a CPU-only machine at all.

C. The architecture must be instantiated first, after which `load_state_dict` populates it with the saved tensors.

D. The file contains the full model, so `torch.load` alone reconstructs it without the class definition present.

---

**Q82.** A security assistant answers developer questions. It currently refuses many benign requests, such as explaining how SQL injection works so a developer can defend against it. The objective is to **reduce unnecessary refusals while keeping genuinely harmful requests refused**.

- P1: "Answer security questions, including how attacks work, so developers can defend their systems. Decline only requests for working exploit code against a named live target."
- P2: "Answer security questions carefully."
- P3: "You are a responsible security engineer. Answer the developer's question."
- P4: "Answer security questions. Do not help with anything that could possibly be misused."
- P5: "Answer security questions. Example: explaining parameterized queries is fine; writing a working exploit is not."

Rank the prompts from most to least effective against this objective.

A. P1 > P5 > P2 > P3 > P4

B. P1 > P5 > P3 > P4 > P2

C. P4 > P1 > P5 > P2 > P3

D. P5 > P1 > P2 > P3 > P4

---

**Q83.** How many bigrams does the sentence *"the quick brown fox jumps"* contain?

A. 5

B. 3

C. 6

D. 4

---

**Q84.** A company buys a third-party model, fine-tunes it on its own data, and sells the result under its own brand.

Which claim is best supported?

A. Selling under its own name makes the company a provider, which carries obligations the original vendor's work does not discharge.

B. The company remains a deployer, since the underlying capabilities originate in a model it did not itself train.

C. Obligations transfer entirely to the purchaser of the final product, who controls how the system is used.

D. Fine-tuning removes any obligation, since the resulting weights differ from those the vendor originally supplied.

---

**Q85.** A team must choose an architecture for classifying support tickets into one of twelve categories.

Which claim is best supported?

A. A decoder-only model is required, since classification is a generation task producing the label as output.

B. An encoder-decoder model is required, since the input and output are different lengths in this task.

C. The architecture is irrelevant, since classification accuracy depends only on the size of the labelled set.

D. An encoder-only model suits this, since bidirectional context is available and no text needs to be generated.

---

**Q86.** A fraud dataset contains 998 legitimate transactions for every 2 fraudulent ones.

Which claim is best supported?

A. The minority class should be discarded, since two examples per thousand cannot support reliable learning.

B. Class weights or resampling should be applied during training, with evaluation using precision and recall.

C. The imbalance is mild enough to ignore, since any ratio above one in a thousand is considered balanced.

D. Accuracy remains the best metric, provided the test set preserves the original ratio of the two classes.

---

**Q87.** A RAG system retrieves 20 chunks and concatenates them into a prompt that exceeds the model's context window.

Which claim is best supported?

A. Context windows apply only to generated output, so the input length imposes no constraint on the call.

B. The model processes all 20 chunks at reduced resolution, which degrades the answer in proportion to the excess.

C. The excess is silently dropped or the call fails, so the pipeline must bound the retrieved context before sending.

D. The model prioritizes the most relevant chunks automatically, discarding the remainder before it generates.

---

**Q88.** An attacker adds small perturbations to an input image that are imperceptible to a human but cause a classifier to change its prediction.

What is this called, and what mitigates it?

A. A membership-inference attack, mitigated by regularizing the model so training points are not memorized.

B. An evasion attack, mitigated by adversarial training though usually at some cost to clean-data accuracy.

C. A poisoning attack, mitigated by auditing the training data for examples the attacker may have contributed.

D. A model-inversion attack, mitigated by limiting the confidence scores the model returns to external callers.

---

**Q89.** A dataset of household incomes has a long right tail, with a small number of very high values.

Which claim is best supported?

A. The mean is undefined for skewed distributions, so the median is the only summary that may be reported.

B. The mean exceeds the median, since extreme high values pull the mean towards the tail of the distribution.

C. The median exceeds the mean, since the median is computed from the upper half of the ordered values.

D. The two coincide, since both are measures of central tendency computed over the same set of observations.

---

**Q90.** A team monitors validation loss during training and stops when it has not improved for 10 consecutive epochs, keeping the weights from the best epoch.

Which claim is best supported?

A. This is learning-rate scheduling, which reduces the step size once progress on the validation set stalls.

B. This is gradient clipping, which bounds the update magnitude once the validation loss ceases to improve.

C. This is cross-validation, which selects the epoch that generalizes best across the available data folds.

D. This is early stopping, which limits overfitting by halting before the model continues fitting training noise.

---
# ANSWER KEY

| Q | Ans | Domain | Q | Ans | Domain | Q | Ans | Domain |
|---|-----|--------|---|-----|--------|---|-----|--------|
| 1 | B | Stats | 31 | A | ML | 61 | A | NLP |
| 2 | D | PyTorch | 32 | D | Prompts | 62 | B | Governance |
| 3 | A | NLP | 33 | A | RAI Tech | 63 | C | LLM |
| 4 | C | Governance | 34 | D | Stats | 64 | D | ML |
| 5 | A | Deep Learning | 35 | B | Deep Learning | 65 | A | Prompts |
| 6 | B | Vision | 36 | B | Vision | 66 | A | RAI Tech |
| 7 | A | RAG | 37 | C | PyTorch | 67 | B | Stats |
| 8 | B | ML | 38 | D | RAG | 68 | C | Deep Learning |
| 9 | A | LLM | 39 | C | NLP | 69 | A | Vision |
| 10 | B | Prompts | 40 | C | Governance | 70 | B | PyTorch |
| 11 | C | RAI Tech | 41 | A | LLM | 71 | C | RAG |
| 12 | C | Stats | 42 | B | ML | 72 | D | NLP |
| 13 | B | Deep Learning | 43 | D | Prompts | 73 | A | Governance |
| 14 | D | PyTorch | 44 | D | RAI Tech | 74 | C | LLM |
| 15 | C | Vision | 45 | D | Stats | 75 | D | ML |
| 16 | B | NLP | 46 | A | Deep Learning | 76 | A | Prompts |
| 17 | D | RAG | 47 | B | Vision | 77 | D | RAI Tech |
| 18 | C | Governance | 48 | B | PyTorch | 78 | C | Stats |
| 19 | C | LLM | 49 | B | RAG | 79 | D | Deep Learning |
| 20 | B | ML | 50 | D | NLP | 80 | B | Vision |
| 21 | C | Prompts | 51 | A | Governance | 81 | C | PyTorch |
| 22 | D | RAI Tech | 52 | B | LLM | 82 | A | Prompts |
| 23 | A | Stats | 53 | D | ML | 83 | D | NLP |
| 24 | A | Deep Learning | 54 | C | Prompts | 84 | A | Governance |
| 25 | B | Vision | 55 | B | RAI Tech | 85 | D | LLM |
| 26 | D | PyTorch | 56 | A | Stats | 86 | B | ML |
| 27 | A | RAG | 57 | C | Deep Learning | 87 | C | RAG |
| 28 | C | NLP | 58 | A | Vision | 88 | B | RAI Tech |
| 29 | C | Governance | 59 | C | PyTorch | 89 | B | Stats |
| 30 | D | LLM | 60 | A | RAG | 90 | D | Deep Learning |

# EXPLANATIONS

**Q1 — B.** Homogeneity of variance is an assumption of the standard F test, and a variance roughly nine times the others violates it badly enough to distort the error term. Welch's ANOVA does not pool variances and is the standard remedy. Normality of residuals is a separate assumption, and dropping a group because its variance is large discards the data rather than analysing it.

**Q2 — D.** The loop is correct. `zero_grad` after `step` is perfectly valid — gradients are cleared before the next forward pass either way, so nothing stale accumulates. The validation loader should not shuffle, and `model.train()` belongs outside the batch loop where it already sits. The distractors each name a real API whose behaviour is misdescribed, so the question is decidable only if you know the actual contract.

**Q3 — A.** With df equal to N, idf = log(N/N) = log(1) = 0, so the product is zero in every document regardless of how often the term occurs. This is TF-IDF doing exactly what it is designed to do: a term present everywhere cannot discriminate. Option A is the intuition that term frequency must count for something, which is the misconception under test.

**Q4 — C.** The labels record real past decisions, so the pattern the model learns was already present in the world — that is historical bias, and it arises even when every individual decision-maker acted in good faith. Measurement bias would require the label to be a poor proxy for the construct; representation bias a skewed sample; deployment bias a mismatch between the training task and the use.

**Q5 — A.** ReLU's derivative is exactly 1 for positive inputs, so repeated multiplication through many layers does not attenuate the gradient — which is the specific problem sigmoid causes, since its derivative peaks at 0.25. ReLU is unbounded above, is not differentiable at zero, and is not zero-centred, so the other three options each attribute a property to ReLU that it does not have.

**Q6 — B.** `cv2.imread` returns channels in BGR order while the model was trained on RGB, so the red and blue channels are transposed. Nothing errors because the shapes are identical either way, and the only symptom is degraded accuracy. The scaling and the permute in the script are both already correct, which is why the other options propose fixing things that are not broken.

**Q7 — A.** Indexing is an offline step that must complete before any query can be served: documents are embedded and stored, then at query time the query is embedded, nearest chunks are retrieved, and the answer is generated. The distractors each place query-time work before index-time work, which is impossible since there would be nothing to search.

**Q8 — B.** Selecting the maximum over 200 configurations fits the validation set, so the winning score captures both genuine quality and whatever noise favoured that configuration. This is why a separate test set exists. The validation set was not used to fit parameters, which is true and is exactly why option A sounds right — but selection is a form of fitting too.

**Q9 — A.** Self-attention computes, for each position, a weighted sum over all positions, where the weights come from the dot products of learned query and key projections. The weights are input-dependent rather than fixed, the attention span is global rather than local, and no single summary vector is produced. Each distractor describes a real mechanism from some other architecture.

**Q10 — B.** P1 states the constraint in the units that actually bind — characters including spaces — and asks for a count. P5's example demonstrates the target length concretely, which is the next best signal. P3's 'well under a typical message length' is vague about what typical means. P2 is bare. P4 supplies a persona and no constraint, which is useless against a hard character cap.

**Q11 — C.** Both SHAP and LIME produce local explanations for individual predictions, and both are model-agnostic. LIME is local only and does not aggregate to a global ranking; SHAP is the one with exact additivity; and neither is restricted to tree ensembles, though TreeSHAP is a fast exact path for them. Each distractor swaps one property between the two methods.

**Q12 — C.** Spearman operates on ranks, so it captures any monotone relationship regardless of curvature and is far less sensitive to extreme values than Pearson. Pearson measures linear association specifically, so a sharply curving relationship understates its strength. Covariance is unbounded and scale-dependent, and rank-based measures are not distorted by outliers, which makes option D's reasoning backwards.

**Q13 — B.** A GRU has a reset gate and an update gate — two gates — and merges the cell and hidden state into one, so at equal hidden size it holds roughly three quarters of an LSTM's parameters. It is not bidirectional by default and contains no attention mechanism. The three-gates-plus-cell-state description in one distractor is simply the LSTM's structure relabelled.

**Q14 — D.** `view` requires the tensor to be contiguous in memory and fails otherwise; `reshape` returns a view when it can and silently copies when it cannot, so it succeeds in both cases. `flatten()` with no arguments collapses every axis including the batch, giving 6,144 elements rather than the required shape. Both `view` and `reshape` can collapse three axes into one.

**Q15 — C.** A 3×3 kernel of uniform weight 1/9 replaces each pixel with the mean of its neighbourhood — a box blur. The entries summing to one means overall brightness is preserved, which is why the image is not darkened, but it does not mean the image is unchanged. Sharpening requires a negative surround with a positive centre, which this kernel does not have.

**Q16 — B.** Standard stop-word lists include `not`, `no` and `never`. Removing them turns 'not good' into 'good' and inverts exactly the signal a sentiment model depends on, while topic classification is unaffected because topic is carried by content words. The selective nature of the damage — one task improves, the other degrades — is what identifies the mechanism.

**Q17 — D.** At k = 5 the relevant documents are those at ranks 1, 2 and 4, so precision@5 = 3/5 = 0.60. Recall@5 divides by the total relevant in the corpus: 3/12 = 0.25. The two distractors that use 0.33 divide by the number retrieved rather than the number relevant, which is the standard confusion between the two denominators.

**Q18 — C.** Triage determining which care pathway a patient enters affects access to an essential service, which is a listed high-risk category. Clinician review is a safeguard that high-risk systems are required to have, not a reason to reclassify downward. Nothing here falls in the prohibited tier, which covers social scoring, workplace emotion inference and similar practices.

**Q19 — C.** Modern tokenizers use subword units, so common words are a single token while rarer words split into several — English averages somewhat above one token per word, typically around 1.3. A token is neither a character nor a sentence. The one-to-one assumption in option A is the most common planning error and produces context budgets that are roughly 30% too optimistic.

**Q20 — B.** Both scores are low and close together, which is underfitting: the model cannot even fit the data it was trained on. Overfitting requires high training accuracy with a large gap. The small gap does indicate the model is not memorizing, which is what makes 'well regularized' superficially attractive — but regularization is not a virtue when the model has learned too little.

**Q21 — C.** P5 supplies an explicit abstention path, which is the single largest lever against confident answers to uncovered questions — the model is given something permitted to say. P4's example demonstrates grounded answering but never licenses a refusal. P2's 'be accurate' is a weak nudge, P1 is bare, and P3's persona supplies no mechanism at all.

**Q22 — D.** Impurity-based importance is computed on training data and favours high-cardinality features, which offer many candidate split points and therefore many chances to reduce impurity — including by fitting noise. An arbitrary identifier is the extreme case. Permutation importance computed on held-out data is the standard corrective, and the near-zero effect of removing the feature would confirm the diagnosis.

**Q23 — A.** A negative coefficient lowers the log-odds, and e^(−0.85) ≈ 0.43, so each additional month multiplies the odds of cancellation by about 0.43. Coefficients are additive on the log-odds scale and multiplicative on the odds scale, never additive on the probability scale — which is why the percentage-point reading is wrong. 2.34 is e^(+0.85), the sign ignored.

**Q24 — A.** 50,000 samples divided by a batch size of 100 gives 500 batches, and one optimizer step per batch gives 500 steps. The distractors correspond to confusing the batch size with the step count, the sample count with the step count, and dividing by the wrong factor — each a recognizable slip rather than an arbitrary number.

**Q25 — B.** `INTER_AREA` averages over the full source region each destination pixel covers, which low-pass filters the image before sampling and prevents aliasing — the correct choice for downscaling. `INTER_NEAREST` discards most pixels without averaging and produces moiré; cubic and linear interpolation are designed for upscaling and do not suppress the high frequencies that cause the artefacts.

**Q26 — D.** 1,050 divided by 100 gives 10 full batches with 50 samples remaining. `drop_last=True` discards that remainder, so the epoch yields 10 batches and those 50 samples are not seen. With `drop_last=False` it would be 11 batches, the last holding 50. No padding or redistribution occurs under either setting.

**Q27 — A.** Dot product grows with vector magnitude while cosine similarity divides it out, so on unnormalized vectors the two can produce different rankings — a long, loosely related vector can outscore a short, well-aligned one on dot product. Cosine is perfectly computable on unnormalized vectors; the normalization is part of the formula, not a prerequisite.

**Q28 — C.** Subword tokenizers build a vocabulary of fragments, so an unseen word decomposes into known pieces — `hyper`, `parameter`, `ization` or similar — and no unknown token is required. This is the central advantage of subword tokenization over word-level, which would emit an unknown token. Falling back to individual characters happens only when no learned merge applies.

**Q29 — C.** A conversational chatbot sits in the limited-risk tier, where the principal obligation is transparency: users must be told they are dealing with an AI system. Conformity assessment, post-market monitoring and data governance are obligations attaching to high-risk systems, so each distractor names a real duty and applies it at the wrong tier.

**Q30 — D.** Temperature divides the logits before the softmax, so raising it flattens the distribution and makes lower-probability tokens more likely to be selected — output becomes more varied and less predictable. Lowering temperature sharpens towards the mode. It does not directly govern sequence length, and it does change which tokens are sampled even though the ordering of probabilities is preserved.

**Q31 — A.** In k-fold cross-validation each record appears in the validation fold exactly once and in the training set for the other four folds, so every record contributes to both and the estimate averages over five splits rather than depending on one. The folds overlap heavily in training data, so the fold scores are not independent and their standard deviation understates the true sampling error.

**Q32 — D.** The request is under-specified — no duration, no room preference — and booking is a side-effecting action that is awkward to undo. Listing the options and waiting is the appropriate behaviour. Creating a booking on a guess commits the user to something they did not ask for, and creating several bookings to cancel later multiplies the side effects rather than avoiding them.

**Q33 — A.** Inputs shifting while the input-output relationship holds is covariate shift. The model is not necessarily wrong, but it is now operating in regions of the feature space it saw rarely during training, so accuracy can still fall. Concept drift would mean the mapping itself changed; label shift concerns the target distribution; leakage is a training-time methodological fault, not a deployment phenomenon.

**Q34 — D.** P(positive and disease) = 0.90 × 0.02 = 0.018. P(positive and no disease) = 0.05 × 0.98 = 0.049. The posterior is 0.018/0.067 ≈ 0.27. Even with 95% specificity, the large healthy population generates more false positives than the small diseased population generates true ones — which is why 0.90, the sensitivity, is the intuitive but wrong answer.

**Q35 — B.** Dropout zeroes each unit independently with the stated probability, resampling a fresh mask on every forward pass — that randomness is the regularizer. Units are not permanently removed, training samples are not skipped, and under inverted dropout the scaling is applied during training rather than at inference, which is why no adjustment is needed when serving.

**Q36 — B.** A convolution applies the same kernel at every spatial position, so a feature learned in one part of the image is detected everywhere — weight sharing plus locality gives the translation prior that makes CNNs sample-efficient on images. A fully connected layer must learn each feature separately at each location. The equivalence claim is true in a narrow technical sense but misses that the sparsity and sharing are the whole point.

**Q37 — C.** `detach()` returns a new tensor sharing the original's values but excluded from the graph, leaving the original fully differentiable — exactly what is required. `requires_grad_(False)` mutates in place and would break the backward pass; `no_grad()` affects only operations inside its scope and detaches nothing retroactively; `clone()` is itself a differentiable operation, so the clone remains attached.

**Q38 — D.** The information exists in the corpus but was not in the retrieved set, so the failure occurred at retrieval — the chunk was not ranked highly enough. Remedies are hybrid search, reranking, or query reformulation. Generation cannot be blamed for not using evidence it never received, and a chunk absent from the retrieved set is not thereby absent from the index.

**Q39 — C.** Word2Vec-style embeddings are static: one vector per word type, so every sense of an ambiguous word is averaged into a single representation. That limitation is precisely what contextual models later removed by producing a different vector per occurrence. The vectors are dense and low-dimensional, not sparse document indicators, and nothing is undefined for ambiguous words.

**Q40 — C.** A person reviews individual outputs and can change the outcome before it takes effect — human-in-the-loop. Human-on-the-loop describes supervision of a system acting autonomously; human-in-command describes setting operating conditions without reviewing each case. The analyst's training and override authority are what make the oversight effective rather than nominal.

**Q41 — A.** A context window is a hard limit: input exceeding it cannot be processed, so the caller must truncate, summarize, or retrieve only the relevant passages before making the call. There is no graceful degradation to lower resolution, no automatic extension, and no second internal pass that stitches the remainder back in.

**Q42 — B.** A random forest averages many decorrelated trees, each fitted to a different bootstrap sample and considering a random feature subset at each split, which reduces variance. Bias is roughly unchanged — that is boosting's mechanism, not bagging's. Averaging cannot reduce every error component at once, and shallow trees are not required for the variance reduction to work.

**Q43 — D.** P3 names the reader's age and gives three operative instructions — everyday words, short sentences, a familiar comparison — which is what actually controls register. P5's example demonstrates the target voice concretely. P4's jargon avoidance is partial and negatively framed. P1 is vague about what 'simply' means. P2's word cap constrains length, which is orthogonal to register.

**Q44 — D.** A forecast of 0.70 issued 100 times should be followed by rain on roughly 70 occasions; 52 is a substantial shortfall, so the model is overconfident in this range. Calibration is precisely a claim about long-run frequency, which is what makes the 'confidence not frequency' distractor wrong. Repeated forecasts at the same stated probability are exactly how calibration is assessed.

**Q45 — D.** The interval width scales with the standard error, which is the standard deviation divided by the square root of n, so a larger sample narrows it. Raising the confidence level widens the interval rather than narrowing it, greater population variance widens it, and a smaller sample widens it — each distractor inverts one of the two levers.

**Q46 — A.** House price is a continuous target, so a regression loss is required, and mean squared error penalizes larger deviations quadratically. Cross-entropy and binary cross-entropy are classification losses, and discretizing a continuous target into brackets throws away ordering information. Hinge loss is for margin-based classification, not regression.

**Q47 — B.** Street signs frequently contain text and directional arrows, both of which change meaning under mirroring, so a flipped image no longer matches its label and the model trains against corrupted targets. Horizontal flipping is safe for many natural-image tasks but validity is a property of the dataset, not the transform — and convolution provides translation equivariance, not reflection invariance.

**Q48 — B.** The required order is clear gradients, forward pass, compute loss, backward pass, update. Zeroing after the backward pass erases the gradients just computed; stepping before the backward pass applies an update using the previous iteration's gradients; and stepping before the loss is even computed does the same. Only one ordering leaves a correct gradient in place at the moment `step` is called.

**Q49 — B.** Smaller chunks concentrate on fewer ideas, so the embedding is more specific and retrieval precision improves — but each retrieved chunk carries less surrounding context, so the generator may lack what it needs to compose a complete answer. This trade-off is why parent-document retrieval exists: match on small chunks, pass the larger section to the generator.

**Q50 — D.** Each entity here is a single token, so each opens with a `B-` tag of the appropriate type and everything else is `O`. An `I-` tag may not open a span, so the all-`I` sequence is malformed. One distractor misplaces the organization and location tags relative to the words, and another drops the B/I prefixes entirely, which is IO tagging rather than BIO.

**Q51 — A.** A model card documents the model: intended and out-of-scope uses, evaluation disaggregated across subgroups, and known limitations — exactly the three items requested. A datasheet documents the dataset, an audit log records runtime behaviour, and a privacy notice addresses data subjects. Four artifacts with four distinct scopes.

**Q52 — B.** Attention computes a weighted sum over positions with no inherent notion of order, so permuting the input permutes the output identically — position information must be injected explicitly. Positional encoding does not compress sequences, does not substitute for recurrence in gradient flow, and attention is global rather than local before any position information is added.

**Q53 — D.** Precision = TP/(TP+FP) = 30/50 = 0.60. Recall = TP/(TP+FN) = 30/40 = 0.75. The distractor pairing 0.75 with 0.60 is the same two numbers with the denominators exchanged, so it cannot be eliminated by inspection — only by computing. The large true-negative count is irrelevant to both metrics.

**Q54 — C.** P2 supplies the literal format string and a worked example, so the field order, separators and date format are all pinned down. P3 names the four fields but leaves every separator and ordering decision open. P5 and P1 refer to conventions the model cannot see, and P4's persona supplies no format information. For exact surface format, showing beats describing.

**Q55 — B.** Uncertainty arising because the input lies outside the region the model learned from is epistemic — it reflects the model's ignorance and shrinks as relevant data is added. Aleatoric uncertainty is irreducible noise in the data-generating process. Two distractors attach the right definition to the wrong name, which is the discrimination being tested.

**Q56 — A.** k-NN ranks neighbours by distance summed across features, so a feature measured in rupees contributes thousands of units while one measured in years contributes tens — the large-range feature dominates and the others are effectively ignored. Standardization puts them on comparable footing. Scaling the target is irrelevant to a distance computation over predictors.

**Q57 — C.** Backpropagation computes the gradient of the loss with respect to every parameter by applying the chain rule backwards through the computation graph. It does not perform the update — that is the optimizer's job — and it computes first derivatives rather than second. The per-example loss is computed in the forward pass, before backpropagation begins.

**Q58 — A.** Standard grayscale conversion weights the channels unequally — roughly 0.299 red, 0.587 green, 0.114 blue — reflecting the eye's much greater sensitivity to green. Conversion retains intensity and discards colour, which is why edge detection works on it, and edge detectors can operate on colour images; grayscale is a convention of convenience, not a requirement.

**Q59 — C.** Broadcasting aligns shapes from the right and left-pads with ones, so `(5, 6)` becomes `(1, 5, 6)`. Comparing against `(8, 1, 6)`: 8 against 1 gives 8, 1 against 5 gives 5, 6 against 6 gives 6, producing `(8, 5, 6)`. Differing rank is not an error, and the leading dimensions are compatible precisely because one of them is 1.

**Q60 — A.** At two million vectors, exact search means comparing the query against every stored vector, which is too slow for a few-millisecond budget. Approximate indexes such as HNSW or IVF give up a small amount of recall for orders-of-magnitude lower latency, which is the standard trade in production retrieval. Recall is tunable rather than absent, and dimensionality is only one of several levers.

**Q61 — A.** A lemmatizer uses a lexicon and part-of-speech information, so it maps the irregular form `was` to its dictionary entry `be`. A rule-based stemmer strips suffixes mechanically and has no mechanism for irregular verbs, so it leaves `was` essentially untouched. Stemmer output is frequently not a valid word, which is the usual practical difference between the two.

**Q62 — B.** Replacing identifiers while retaining the mapping is pseudonymization: re-identification remains possible, so the data is still personal data and the regulation still applies in full. Pseudonymization is explicitly recognized as a security measure rather than an exemption. Who holds the mapping within the organization does not change the analysis.

**Q63 — C.** Fine-tuning updates the model's weights through gradient descent on examples; prompting leaves the weights untouched and changes only the context supplied at inference. Prompting therefore has no effect that persists across calls, the two are not equivalent in what they can achieve, and fine-tuning does require labelled or at least curated examples for a supervised objective.

**Q64 — D.** With 0.9% prevalence, a model that predicts the majority class for every case scores 99.1% — exactly the reported figure — so accuracy alone provides no evidence that any disease signal was learned. Precision, recall and the confusion matrix are required. Preserving the natural class balance is correct practice for evaluation but does not rescue accuracy as a metric.

**Q65 — A.** The agent had a calculator available and chose not to use it, which is a prompting problem: tool use should be directed rather than left to the model's discretion for tasks where it is known to be unreliable. Language models make arithmetic slips on multi-step computations, and an available tool is not automatically invoked just because its description matches.

**Q66 — A.** Logistic regression is interpretable by construction — the coefficients are the model — while a boosted ensemble requires post-hoc methods such as SHAP that approximate rather than reproduce its behaviour. That approximation is the relevant distinction for a regulated decision. Interactions can be added to a linear model explicitly, and accuracy alone does not satisfy a regulator.

**Q67 — B.** Failing to reject means the evidence was insufficient at the chosen level, which is not the same as evidence that the effect is absent — particularly when power is low. Absence of evidence is not evidence of absence. Rejecting the null never proves it true, and moving the threshold after seeing the result is exactly the practice significance testing exists to prevent.

**Q68 — C.** A learning rate five hundred times larger overshoots the descent direction, so the loss oscillates or diverges rather than settling. Larger steps do not traverse the surface more efficiently past a certain point, do not slow convergence by requiring more iterations, and adaptive optimizers rescale per-parameter but do not neutralize a globally excessive base rate.

**Q69 — A.** The boxes overlap on x from 10 to 20 and y from 10 to 20, giving an intersection of 10 × 10 = 100. Each box is 20 × 20 = 400, so the union is 400 + 400 − 100 = 700. IoU = 100/700 ≈ 0.143. Forgetting to subtract the intersection once from the summed areas gives 0.125, and dividing by one box's area gives 0.25.

**Q70 — B.** Metrics are computed and reported for monitoring but take no part in the gradient computation — only the loss does. This is why accuracy, which is not differentiable, can be used as a metric but never as a loss. The metric does not replace the loss under any condition, and Keras does not require metrics to be differentiable.

**Q71 — C.** Overlap means consecutive chunks share text, so a sentence or clause that would otherwise be severed at a boundary appears whole within at least one chunk. It increases index size rather than reducing it, provides no guarantee about complete sections, and does not improve embedding quality — the duplicated text is the cost paid for boundary safety.

**Q72 — D.** Bag-of-words records only which words occur and how often, discarding order entirely, so two sentences built from the same multiset of words produce identical vectors. That is the representation's defining limitation and the reason n-grams and sequence models exist. Position is not encoded in any dimension, and repeated words are simply counted.

**Q73 — A.** Purpose limitation requires that data collected for one purpose not be further processed in a way incompatible with it, so a compatibility assessment or a fresh lawful basis is needed before the secondary use. Holding the data lawfully for the original purpose does not license a new one, training is not exempt, and consent is one available basis among several rather than the only one.

**Q74 — C.** Top-k always keeps exactly k candidates regardless of how the probability mass is distributed; top-p keeps the smallest set whose cumulative probability reaches the threshold, so the candidate count adapts to the distribution's shape — narrow when the model is confident, wide when it is not. Both are stochastic, and the parameters are not interchangeable.

**Q75 — D.** There are no outcome labels and the goal is to discover structure, which is unsupervised clustering. Manufacturing a label from the route identifier would just recover the identifier. Regression requires a target to predict, and reinforcement learning requires a reward signal and an action loop, neither of which a static set of delivery records provides.

**Q76 — A.** Multi-step arithmetic benefits from the model externalizing intermediate values, so P4's explicit step-by-step instruction leads. P5's 'check your arithmetic' nudges weakly in the same direction. P1 is neutral and P3's persona provides no computational scaffolding. P2 ranks last and is the trap: demanding only the final figure suppresses precisely the reasoning that produces the correct one.

**Q77 — D.** With labels six months away, the available signals are input feature distributions and the distribution of predicted scores, compared against a reference window — both detect change immediately and act as leading indicators. Training loss is a property of the fitted model and does not update after deployment, and prediction volume says nothing about quality.

**Q78 — C.** Stratified sampling divides the population into groups and samples within each, guaranteeing the small enterprise segment is represented in proportion to its importance rather than its size. Simple random sampling may return very few enterprise customers by chance, cluster sampling draws whole groups and can miss the segment entirely, and convenience sampling introduces response bias.

**Q79 — D.** 32 with padding 1 and a 3×3 kernel stays 32, pooling halves it to 16; the second convolution preserves 16 and pooling halves it to 8. With 32 output channels the flattened length is 32 × 8 × 8 = 2,048. The distractors correspond to losing one pooling stage, using the wrong channel count, and halving once too often.

**Q80 — B.** NumPy reports array shape as rows, columns, channels — so `(1080, 1920, 3)` means 1,080 rows of height and 1,920 columns of width, with three colour channels. The row-before-column convention is the source of the most common indexing errors in image code, particularly since `cv2.resize` takes its target size as width then height.

**Q81 — C.** A `state_dict` is a dictionary of tensors with no reference to the class, so the architecture must be instantiated first and then populated with `load_state_dict`. Optimizer state is a separate object and is not included unless saved explicitly. The file is portable across devices provided `map_location` is used when the saving and loading devices differ.

**Q82 — A.** P1 draws the boundary precisely: explain attacks for defensive purposes, and decline only working exploit code against a named live target — a rule that permits the benign cases and still refuses the harmful ones. P5's example illustrates the same boundary on two cases without stating it generally. P2 and P3 give no boundary at all, and P4's 'anything that could possibly be misused' makes over-refusal worse, which is the failure being fixed.

**Q83 — D.** A bigram is a pair of adjacent words, so a sentence of n words yields n − 1 bigrams. Five words give four: the-quick, quick-brown, brown-fox, fox-jumps. Counting five is the off-by-one that comes from counting words rather than adjacent pairs; the formula generalizes to n − k + 1 for k-grams.

**Q84 — A.** Placing a system on the market under its own name makes the company a provider, inheriting obligations that the original vendor's work does not discharge — the vendor's documentation is an input to due diligence, not a transfer of responsibility. Substantial modification plus rebranding is the specific trigger, and obligations do not pass to the purchaser.

**Q85 — D.** Classification needs a representation of the whole input and produces a label rather than text, which is what an encoder-only model is built for: bidirectional context pooled into a classification head. A decoder can classify but discards bidirectional context through causal masking, and an encoder-decoder adds a generation stage the task does not need.

**Q86 — B.** At 2 positives per 1,000 the training signal is overwhelmed by the majority class, so class weights or resampling are needed during training, and evaluation must use precision and recall rather than accuracy. Discarding the minority class discards the thing being predicted, and a 1:499 ratio is severe imbalance by any standard.

**Q87 — C.** Exceeding the context window either causes the call to fail or results in content being silently dropped, depending on the client — so the pipeline must bound retrieved context before sending, typically by reranking and truncating. The model does not degrade gracefully, does not select the most relevant chunks for you, and the window covers input as well as output.

**Q88 — B.** Perturbing inputs at inference to force a misclassification is an evasion attack. Adversarial training — including adversarial examples in the training set — is the standard mitigation, and it typically costs some accuracy on clean data. Poisoning targets the training set, model inversion reconstructs training inputs, and membership inference tests whether a record was used in training.

**Q89 — B.** The mean is sensitive to extreme values while the median is not, so a long right tail pulls the mean above the median — the standard signature of right skew. The median is the middle value of the ordered data, not a summary of the upper half, and the mean is perfectly well defined for skewed distributions even when it is a poor summary.

**Q90 — D.** Monitoring a validation metric, halting after a fixed number of epochs without improvement, and restoring the best weights is early stopping — a regularization technique that prevents the model from continuing to fit training noise. Learning-rate scheduling reduces the step size rather than halting, gradient clipping bounds update magnitude, and cross-validation partitions data rather than selecting an epoch.

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
