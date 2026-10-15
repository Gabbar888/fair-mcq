# AI Assessment — Mock 08
## 90 questions | 150 minutes | MCQ, single correct

Each question has exactly one correct answer. There is no negative marking. The paper is long by design; budget about 100 seconds per question and move on when stuck.

---

**Q1.** A battery manufacturer runs the same automated X-ray inspection model at two plants. The model has a sensitivity of 0.90 and a specificity of 0.95 at both sites, confirmed on labelled audit samples. Each plant produces 10,000 cells a week. Because the Pune line is still running an older electrode supplier, about 20% of its cells are defective; at the mature Chennai line, about 1% are. Flagged cells go to a manual teardown; passed cells ship.

At the quarterly review, the quality head says: "Since sensitivity and specificity are identical at both plants, a flagged cell is equally likely to be truly defective at either plant, and a passed cell is equally safe to ship."

Which statement correctly evaluates this claim?

A. Both halves hold, since predictive values depend only on sensitivity and specificity here.

B. A flag at Chennai is defective about 82% of the time, as specificity is unchanged.

C. A flag means defective about 82% of the time at Pune, but only 15% at Chennai.

D. Passed Pune cells are safer than Chennai's, since Pune's flags are far more accurate.

---

**Q2.** A drone maker sources its altitude sensors from three vendors. Purchasing records and vendor-level field data for the last year are below.

| Vendor | Share of sensors fitted | Field failure rate |
|---|---|---|
| V1 | 50% | 1% |
| V2 | 30% | 2% |
| V3 | 20% | 5% |

A drone crashes and the investigation confirms the altitude sensor failed, but its serial label was destroyed. The operations lead wants to open a supplier audit with the vendor most likely to have supplied the failed sensor, and argues for V1 "because half of everything we fit comes from them."

Which vendor is most likely, and with what probability?

A. V3, with a posterior probability of roughly 0.48.

B. V3, with a posterior of 0.20, matching its share of fitted sensors.

C. V3, with probability 0.05, which is its own field failure rate.

D. V2, with roughly 0.29, as it sits between the two extremes.

---

**Q3.** A shopping app's UX team runs a balanced experiment on checkout time (seconds) with two product-list layouts (Grid, List) across three device types, two sessions per cell.

| Layout | Phone | Tablet | Desktop |
|---|---|---|---|
| Grid | 30, 34 | 26, 28 | 20, 22 |
| List | 36, 38 | 30, 34 | 25, 27 |

A two-way ANOVA with interaction is fitted. Critical values at α = 0.05 are F(1, 6) = 5.99 and F(2, 6) = 5.14. The product manager wants to know whether layout matters and whether that answer depends on the device.

Which statement is supported?

A. The interaction is significant, so layout's effect must be read device by device.

B. Layout F is about 2.4, below its critical value, so layout does not matter.

C. Layout F is 18.75 on 2 degrees of freedom, as there are three devices.

D. Interaction F is zero; layout F is about 18.8, which is significant.

---

**Q4.** A property portal predicts listing prices with a linear regression. Two of its predictors, built-up area and carpet area (both in square feet), have a correlation of 0.97 across listings. When the model is refitted each month, the coefficients on these two features swing widely and sometimes flip sign, although hold-out prediction error barely changes. A junior analyst writes in the review: "The model is broken; the predictions cannot be trusted until we collect far more listings."

Which assessment is correct?

A. VIF is about 1.06 each, so collinearity is negligible and flips are noise.

B. VIF is about 17; coefficients are unstable but predictions are largely unaffected.

C. VIF is about 17, which makes predictions unreliable until many more rows are collected.

D. VIF is about 33, as it is one over one minus the correlation between them.

---

**Q5.** A telecom analytics team runs PCA on two customer features before clustering: monthly spend (in hundreds of rupees, variance 400) and support visits per year (variance 4). The correlation between them is 0.5. The team lead notices that PC1 is almost exactly the spend axis and asks whether the result reflects real structure.

What share of variance does PC1 explain on the raw covariance matrix, and what happens if both features are standardized first?

A. About 99%, and standardizing keeps it near 99%, as the correlation is unchanged.

B. About 75% raw as well, because PCA effectively operates on correlations anyway.

C. About 99% raw, falling to 75% once both features are standardized.

D. About 99% raw, rising further when standardized, as unit noise is removed.

---

**Q6.** A logistics company models delivery time from route features. On 40 historical routes, Model A uses 3 predictors and reaches R² = 0.62. Model B adds 5 more predictors (8 in total) such as driver shirt colour and day-of-month parity, and reaches R² = 0.65. The analyst recommends Model B because "it explains more of the variance."

Using adjusted R² = 1 − (1 − R²)(n − 1)/(n − p − 1), which statement is correct?

A. Adjusted R² falls from about 0.59 to 0.56, so the extra predictors do not pay.

B. Adjusted R² rises from 0.62 to 0.65, since more predictors explain more variance.

C. Adjusted R² rises from about 0.59 to 0.62, so Model B is justified on its merits.

D. Plain R² cannot fall when predictors are added, so Model B is necessarily better.

---

**Q7.** A warehouse trials a new handheld scanner with six pickers. Each picker's rate (picks per hour) is measured for a week on the old scanner and then a week on the new one.

| Picker | Old scanner | New scanner |
|---|---|---|
| P1 | 52 | 55 |
| P2 | 61 | 64 |
| P3 | 48 | 50 |
| P4 | 70 | 74 |
| P5 | 55 | 57 |
| P6 | 64 | 68 |

The analyst runs Welch's two-sample t-test on the two columns, gets p ≈ 0.56, and reports "no evidence the new scanner helps."

Which statement is correct?

A. Welch's test is the right choice here, as the two columns have unequal variances.

B. A paired test gives t ≈ 0.61, agreeing that there is no detectable effect.

C. The paired test is significant, but only because six pickers is too few to trust.

D. A paired t-test gives t ≈ 8.2, p < 0.001, with each picker as their own control.

---

**Q8.** A streaming service's churn model is a logistic regression. The intercept is set so that a customer with no support tickets in the last quarter has a 10% probability of churning, and the coefficient on support-ticket count is 0.4. The customer-success lead reads the model as: "Three tickets roughly triples a customer's churn risk, from 10% to 30%."

For an otherwise identical customer with three tickets, which statement is correct?

A. The odds and the probability both scale by 3.32, giving about 0.33.

B. The odds scale by about 3.32, and the churn probability is about 0.27.

C. The probability triples to 0.30, as the model's effect is linear per ticket.

D. The odds scale by about 4.47, as e^0.4 is multiplied by three; about 0.33.

---

**Q9.** An industrial-equipment company has given 3,000 field technicians a laptop app that runs a small language model entirely offline. The model reads fault codes and service history and suggests repair steps. Technicians often work in plants with no connectivity for days at a time. After two months, regional managers hear anecdotally that suggestions are sometimes wrong for older machine models, but the central AI team has no record of which suggestions failed, on which machines, or whether technicians simply ignored them. Two technicians report that they stopped using the app after it twice recommended a part that does not exist on their plant's machines, and a third says he has "been keeping a list in a notebook" that nobody has asked for. The model is updated centrally twice a year and pushed to laptops when they next connect. Leadership asks how post-deployment feedback should be handled from now on.

Which approach should the company adopt?

A. Add device telemetry on latency, crashes and memory, since user reports from the field are too inconsistent to act on.

B. Hold a technician survey each quarter, and retrain the model on the aggregated survey responses afterwards.

C. Let each regional manager collect complaints in their own way, since field conditions differ widely between regions.

D. Build in-app flagging that queues offline and syncs later, with triage, named owners and responses sent back to technicians.

---

**Q10.** An SRE team monitors a payments API. In normal operation, 5xx errors arrive as a Poisson process averaging 2 per hour. The on-call engineer is paged whenever an hour contains at least a threshold number of errors. The team wants the lowest threshold that keeps the expected number of false pages (pages during normal operation) below one per week (168 hours), so genuine incidents are caught as early as possible.

Which threshold should they choose?

A. At least 5 per hour, as P(X ≥ 5) of about 5% matches the usual significance level.

B. At least 6 per hour, at about 0.017 per hour and so under one page a week.

C. At least 8 per hour, as alert thresholds should sit four standard deviations above the mean.

D. At least 7 per hour, at about 0.0045 per hour, or only about 0.76 false pages a week.

---

**Q11.** A consultancy discovers that staff have been pasting client contracts and financial statements into a free public chatbot to summarize them. The chatbot's terms allow the provider to use submitted content to improve its models. Partners are split: some want a blanket ban on all generative AI, others say "everyone does it; there is no real problem."

Which response best reflects sound AI governance?

A. A blanket ban on generative AI tools, since any use creates unacceptable confidentiality exposure.

B. Provide an approved enterprise tool with data protections, plus a clear usage policy and training.

C. Allow continued use but require staff to delete the chat history after each session.

D. Take no action, as the provider's terms make confidentiality the provider's responsibility.

---

**Q12.** A start-up plans to train a mental-health support chatbot on 400,000 posts scraped from a public peer-support forum. The posts are publicly visible without login, many describe self-harm and medication in detail, and usernames often resemble real names. The forum's terms prohibit automated collection. The CTO argues that "public data is fair game," and the team plans to release the fine-tuned model to a partner clinic network next quarter. No one at the start-up has contacted the forum's moderators, and the dataset has already been copied to two contractors' machines for labelling.

What is the most important governance step before training proceeds?

A. Review consent, licensing and sensitivity of the data before any use, rather than assuming public means permissible.

B. Remove usernames, which fully resolves the privacy concerns raised by the sensitive post content.

C. Proceed, since posts visible without a login carry no expectation of privacy for their authors.

D. Proceed, but add a disclaimer telling chatbot users it was trained on public forum data.

---

**Q13.** A lender's data-science team presents two candidate approval thresholds for a new credit model to the product owner. Group B is a historically under-served applicant group.

| Threshold | Approval rate A | Approval rate B | True positive rate A | True positive rate B | Portfolio default rate |
|---|---|---|---|---|---|
| T1 | 52% | 50% | 0.78 | 0.61 | 3.1% |
| T2 | 48% | 55% | 0.74 | 0.73 | 3.4% |

The product owner asks the data scientists to "just pick the fair one" by the end of the day.

Which course of action is most appropriate?

A. Let the data scientists choose T1, since it has the lowest default rate of the two options.

B. Adopt T2 automatically, since equal true positive rates are the one correct definition of fairness.

C. Escalate to an accountable forum to choose the fairness criterion and document the trade-off.

D. Adopt T1, since equal approval rates are a legal requirement in every jurisdiction where it operates.

---

**Q14.** A telecom's customer-facing chatbot told several users on social media how to bypass their phone's carrier lock, which voids the device warranty and breaches the contract. Screenshots are spreading. The timeline so far:

| Time | Event |
|---|---|
| 09:10 | First screenshot posted publicly |
| 09:40 | Social team alerts the AI product owner |
| 09:55 | Engineers reproduce the behaviour with three prompts |
| 10:05 | Leadership asks what to do first |

Which sequence of actions is most appropriate?

A. Find the root cause first, then retrain the model, then redeploy, then notify affected parties.

B. Contain it (block the topic or disable the bot), assess impact, notify, find root cause, document.

C. Issue a public statement first, then investigate whether the bot is still producing the advice.

D. Delete the public screenshots, then patch the prompt quietly, so the incident does not escalate further.

---

**Q15.** An insurer begins using an LLM to draft replies to customer complaints. Agents review each draft and usually send it unchanged. The marketing team wants to sign every reply with the agent's name only, arguing that "customers prefer a human touch and the agent approved it anyway." Some complaints concern claim denials, and customers often reply asking follow-up questions about their specific case.

Which approach best reflects responsible-AI transparency?

A. Sign replies with the agent's name only, since the agent's approval makes them the true author.

B. Disclose AI involvement only when a customer asks directly whether a person wrote the reply.

C. Disclose AI involvement in every reply, but withhold any option to reach a human reviewer.

D. Disclose in every reply that AI assisted the drafting, with a clear route to a human for follow-ups.

---

**Q16.** A smart-speaker company stores every raw voice recording indefinitely "for future model improvement," including recordings captured after accidental wake-word triggers. Its privacy notice says audio is "used to provide and improve the service." An internal review finds that fewer than 2% of stored recordings have ever been used in training, and that accidental triggers often capture private conversations.

Which change best applies data-governance principles?

A. Set a retention limit, delete accidental-trigger audio, and keep only what has a defined purpose.

B. Encrypt all stored recordings, which on its own resolves the concerns about retaining them.

C. Keep everything, since the privacy notice already covers use for improving the service.

D. Move the recordings to cheaper cold storage, as volume and storage cost are the main issues.

---

**Q17.** A support team is training a text classifier to route customer emails. Each email has already been tokenized into a tensor of word IDs, and emails range from 12 to 400 tokens long. When the script starts, it fails with a runtime error before the first optimizer step.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class EmailDataset(Dataset):
    def __init__(self, token_ids, labels):
        self.token_ids = token_ids          # list of 1-D LongTensors
        self.labels = labels                # list of ints
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.token_ids[i], self.labels[i]

loader = DataLoader(EmailDataset(ids, labels), batch_size=32,
                    shuffle=True)                                  # line 14
model = EmailClassifier(vocab=30000, n_cls=6)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    for x, y in loader:
        probs = model(x).softmax(dim=-1)
        loss = loss_fn(probs, y)
        loss.backward()
        opt.zero_grad()                                            # line 25
        opt.step()
        sched.step()                                               # line 27
    model.eval()                                                   # line 28
    print(epoch, evaluate(model, val_loader))
```

The script contains several bugs. Which one stops it from running?

A. Line 27, as per-batch scheduler steps drive the learning rate to zero and raise an error.

B. Line 28, since a model left in eval mode raises an error on its next backward pass.

C. Line 14, as the default collate cannot stack token tensors of different lengths into one batch.

D. Line 25, since zeroing gradients before the step makes `opt.step()` raise an error.

---

**Q18.** A bank's credit-risk model is built by the retail analytics team. The same team also performs the model's pre-deployment validation and signs it off, and then monitors it in production. An external review lists the roles as follows.

| Activity | Performed by |
|---|---|
| Model development | Retail analytics team |
| Pre-deployment validation | Retail analytics team |
| Production monitoring | Retail analytics team |
| Periodic audit of the model process | None assigned |

The model has performed well so far. Which governance gap is most serious?

A. Monitoring belongs with IT operations, since data scientists lack production expertise.

B. The team needs more data scientists, as one team doing all three activities is overloaded.

C. Validation is not independent of development, and no one audits the model process.

D. There is no gap, since strong performance so far shows the controls are working.

---

**Q19.** A vision team fine-tunes a model that only fits a batch of 16 images in GPU memory. To match a reference run that used a batch of 64, they accumulate gradients over four mini-batches. The accumulated run diverges at the reference learning rate, although it trains stably at a quarter of it.

```python
ACCUM = 4
loss_fn = nn.CrossEntropyLoss()           # reduction="mean"
opt.zero_grad()
for i, (x, y) in enumerate(loader):       # batch_size=16
    x, y = x.cuda(), y.cuda()
    loss = loss_fn(model(x), y)
    loss.backward()
    if (i + 1) % ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()
```

What explains the divergence, and what is the fix?

A. The accumulated gradient is four times the batch-64 gradient; divide each loss by `ACCUM`.

B. Accumulation averages away the gradient signal, so the learning rate should be raised four times.

C. Clipping at 5.0 is applied too late, after four mini-batches have already accumulated.

D. `zero_grad()` runs only every fourth batch, so stale gradients leak into each step.

---

**Q20.** A team training a speech model on rented GPUs saves a checkpoint every epoch, because the cloud provider can preempt their instance at any time. After a preemption at epoch 31, they resume from the latest checkpoint. Training loss, which had been falling smoothly at about 0.42, jumps to 0.95 for the first few hundred steps after resuming and takes two epochs to recover. They use Adam with a cosine learning-rate schedule and warmup.

```python
torch.save({"epoch": epoch, "model": model.state_dict()}, f"ckpt_{epoch}.pt")

ckpt = torch.load("ckpt_31.pt")
model.load_state_dict(ckpt["model"])
start_epoch = ckpt["epoch"] + 1
```

What should the checkpoint also contain to make resumption seamless?

A. The full pickled model object instead of its state dict, so the architecture is restored too.

B. The random seed only, since the jump comes from a different shuffling order of training batches.

C. The training data loader, so that resumed epochs see exactly the same batches as before preemption.

D. The optimizer and scheduler state dicts, so Adam's moment estimates and the learning-rate position resume.

---

**Q21.** An ML engineer notices that a ResNet-50 training job on a single A100 GPU shows only about 30% GPU utilization. Profiling one epoch gives the breakdown below. Images are JPEGs on a local NVMe disk, and the `DataLoader` uses `num_workers=0` with heavy random augmentation.

| Stage per training step | Mean time |
|---|---|
| JPEG decoding and augmentation (main process) | 410 ms |
| Host-to-GPU copy | 25 ms |
| Forward and backward pass on GPU | 140 ms |
| Optimizer step | 8 ms |

Which change is most likely to give the largest speed-up?

A. Enable mixed precision, which roughly halves the forward and backward time on an A100.

B. Use several DataLoader workers with pinned memory, so that image loading overlaps GPU compute.

C. Increase the batch size, so each GPU step processes more images per launch overhead.

D. Switch to a smaller model, since the 140 ms GPU step limits training speed.

---

**Q22.** A defect-inspection model is trained on 224 × 224 RGB images loaded with OpenCV. The pipeline runs without errors, but validation accuracy stays near chance, while the same architecture trained from a torchvision `ImageFolder` reaches 94%.

```python
import cv2
import numpy as np
import torch

def load_batch(paths):
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    arr = np.stack(imgs).astype(np.float32) / 255.0          # line 7
    x = torch.from_numpy(arr)                                 # line 8
    x = x.view(x.size(0), 3, 224, 224)                        # line 9
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std                                   # line 12
```

Which line is responsible?

A. Line 7, as dividing by 255 before normalizing scales the inputs twice over.

B. Line 12, since the ImageNet statistics do not apply to defect-inspection images.

C. Line 8, since `from_numpy` shares memory with the NumPy array and corrupts it.

D. Line 9, which reinterprets the memory layout without reordering any axes.

---

**Q23.** A team adapts a pretrained ResNet-18 to five product categories. They want the new head to train with a learning rate of 1e-3 and the pretrained backbone to train gently at 1e-5, with all parameters updated.

```python
import torch.nn as nn
import torchvision

model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 5)
backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
```

Which optimizer definition achieves this?

A.
```python
opt = torch.optim.AdamW([
    {"params": model.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-3}])
```

B.
```python
opt = torch.optim.AdamW([
    {"params": backbone, "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-3}])
```

C.
```python
for p in backbone:
    p.requires_grad = False
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

D.
```python
opt = torch.optim.AdamW([
    {"params": backbone, "lr": 1e-3},
    {"params": model.fc.parameters(), "lr": 1e-5}])
```

---

**Q24.** A recommendation team writes a residual MLP. Training loss barely moves over ten epochs, and a colleague notices something odd in the parameter count.

```python
import torch
import torch.nn as nn

class ResMLP(nn.Module):
    def __init__(self, dim=128, depth=3, n_out=10):
        super().__init__()
        self.blocks = [nn.Linear(dim, dim) for _ in range(depth)]
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        for block in self.blocks:
            x = x + torch.relu(block(x))
        return self.head(x)

model = ResMLP()
print(sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```

What does the script print, and what is the underlying problem?

A. It prints 1,290; blocks in a plain list are never registered, so they never train.

B. It prints 50,826, with every layer registered; the residual additions are cancelling the gradients.

C. It prints 1,290, because `Linear` layers in a list share one set of weights between them.

D. It prints 49,536; the head is missing, as it is defined after the list of blocks.

---

**Q25.** A music-streaming service predicts churn, defined as a subscriber cancelling within the 30 days after a monthly snapshot date. To get more training rows, the analyst builds one row per subscriber per month for the past 18 months, so each subscriber appears up to 18 times with slowly changing features (listening hours, skips, playlist count, plan). Five-fold cross-validated AUC is 0.91; scored on next month's live subscribers, AUC is 0.74.

```python
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

rows = []
for snap in pd.date_range("2025-01-31", "2026-06-30", freq="ME"):
    feats = build_features(subscribers, events, as_of=snap)      # only data <= snap
    feats["churned"] = cancelled_within(subscribers, snap, days=30)
    feats["snapshot"] = snap
    rows.append(feats)
df = pd.concat(rows)

X = df.drop(columns=["churned", "snapshot", "subscriber_id"])
y = df["churned"]
cv = KFold(n_splits=5, shuffle=True, random_state=1)
print(cross_val_score(GradientBoostingClassifier(), X, y, cv=cv,
                      scoring="roc_auc").mean())
```

What best explains the gap, and what should replace the CV scheme?

A. The features use future data, so `build_features` must exclude events after each snapshot.

B. Gradient boosting overfits this data, so it should be swapped for a regularized logistic model.

C. Shuffled folds leak class balance between folds, so the CV should be stratified on the churn label.

D. The same subscribers appear in train and test folds; split by subscriber or by time.

---

**Q26.** A property-tech firm's random forest for rent estimation has 100 trees. On validation data, each individual tree's prediction has a variance of about 400 (₹ hundreds²), and the average pairwise correlation between trees' predictions is 0.3. The variance of a forest's average prediction is approximately ρσ² + (1 − ρ)σ²/B, where B is the number of trees. The team has budget for one change before the next release.

Which change reduces the forest's prediction variance most, assuming per-tree variance stays roughly the same?

A. Grow 1,000 trees instead of 100, cutting the variance from about 123 to roughly 120.

B. Lower `max_features` so tree correlation falls to 0.15, cutting the variance to about 63.

C. Grow 1,000 trees instead of 100, cutting the variance from about 123 to roughly 12.

D. Use a single fully grown tree, since one deep tree has no correlation term to pay for.

---

**Q27.** A payments company trains a binary fraud classifier with `BCEWithLogitsLoss`. The training set is heavily imbalanced, and the model has learned to predict "legitimate" for almost everything.

| Class | Training examples |
|---|---|
| Legitimate (0) | 49,500 |
| Fraud (1) | 500 |

The team wants each fraud example to contribute to the loss as much in total as the legitimate class does. How should the loss be configured?

A. `BCEWithLogitsLoss(pos_weight=torch.tensor([0.01]))`, matching fraud's 1% share.

B. `CrossEntropyLoss(weight=torch.tensor([99.0, 1.0]))`, weighting each class by the other's count.

C. `BCEWithLogitsLoss(pos_weight=torch.tensor([99.0]))`, the negative-to-positive ratio.

D. `BCEWithLogitsLoss(reduction="sum")`, so rare fraud examples are not averaged away.

---

**Q28.** A bank's fraud team has 5,000 labelled transactions: 4,900 legitimate and 100 fraudulent. The data export is sorted by label, so all 100 fraud rows come last. An analyst evaluates a classifier with the loop below.

```python
import numpy as np
from sklearn.model_selection import KFold

y = np.array([0] * 4900 + [1] * 100)
for i, (tr, te) in enumerate(KFold(n_splits=5).split(y)):
    print(i, y[tr].sum(), y[te].sum())
```

What does the loop reveal, and what is the fix?

A. Four test folds have no fraud and the last training fold has none; use shuffled stratified folds.

B. Each test fold receives 20 fraud cases, so the evaluation is sound as written.

C. Four test folds have no fraud; the fix is to oversample fraud within each fold.

D. Four test folds have no fraud, which is expected and harmless at a 2% fraud rate.

---

**Q29.** A broadband company offers a ₹200 retention voucher to customers its churn model flags. A churner who receives the voucher stays 40% of the time, and a retained customer is worth ₹1,500 in future margin. Vouchers sent to customers who would not have churned are wasted. On a validation sample of ten customers:

```python
import numpy as np

p = np.array([.82, .71, .64, .55, .47, .41, .36, .30, .22, .15])   # predicted churn prob.
churned = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 0])

def net_value(t):
    flag = p >= t
    tp = (flag & (churned == 1)).sum()
    return tp * 0.40 * 1500 - flag.sum() * 200

for t in [0.60, 0.45, 0.35, 0.25]:
    print(t, net_value(t))
```

Which threshold maximizes net value on this sample, and what is that value?

A. 0.60, worth ₹600, since the default 0.5-style cut-off keeps voucher waste lowest.

B. 0.25, worth ₹1,600, since contacting more customers always catches more churners.

C. 0.35, worth ₹1,600, near the break-even probability of 0.33.

D. 0.45, worth ₹800, since the threshold should sit just below 0.5 for cost-sensitive cases.

---

**Q30.** A gym chain is building a decision tree to predict which members will cancel. The analyst checks the root split by hand on a small sample.

```python
import numpy as np

def gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 1 - p**2 - (1 - p)**2

tenure  = np.array([2, 3, 5, 8, 12, 20, 24, 30, 36, 48])     # months
tickets = np.array([5, 4, 0, 3, 1, 0, 2, 2, 0, 0])            # complaints
churned = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0])

best = None
for name, x in [("tenure", tenure), ("tickets", tickets)]:
    for t in np.unique(x)[:-1]:
        left, right = churned[x <= t], churned[x > t]
        g = (len(left) * gini(left) + len(right) * gini(right)) / len(churned)
        if best is None or g < best[2]:
            best = (name, int(t), round(g, 3))
print(best)
```

What does the script print?

A. `('tenure', 24, 0.286)`

B. `('tickets', 0, 0.167)`

C. `('tenure', 12, 0.32)`

D. `('tickets', 2, 0.286)`

---

**Q31.** A retailer clusters 80,000 loyalty members on standardized spending features with k-means, to design targeted campaigns. The marketing team can run between three and six distinct campaigns. Results for different k:

| k | Inertia (within-cluster SS) | Mean silhouette |
|---|---|---|
| 2 | 412,000 | 0.41 |
| 3 | 305,000 | 0.47 |
| 4 | 221,000 | 0.58 |
| 5 | 198,000 | 0.52 |
| 6 | 182,000 | 0.44 |
| 7 | 170,000 | 0.39 |

Which choice is best supported?

A. k = 7, since it has the lowest inertia and therefore the tightest clusters.

B. k = 2, since it has the largest inertia drop when moving to the next value.

C. k = 6, the most campaigns the team can run, as more segments allow finer targeting.

D. k = 4, where silhouette peaks and the inertia curve bends, within the team's range.

---

**Q32.** A food-delivery platform compares two ETA models on ten test orders (minutes). Customers are refunded when an order arrives more than 10 minutes after the promised time, so occasional large misses are far more costly than many small ones.

```python
import numpy as np

actual = np.array([30, 35, 28, 40, 32, 45, 38, 30, 33, 36])
err_A  = np.array([ 2, -2,  3, -3,  2, -2,  3, -3,  2, -2])
err_B  = np.array([ 0,  0,  1,  0, -1,  0,  0,  1,  0, 12])

for name, e in [("A", err_A), ("B", err_B)]:
    print(name, np.abs(e).mean(), round(np.sqrt((e ** 2).mean()), 2))
```

Which statement is correct?

A. Model B, since its MAE of 1.5 beats A's 2.4, and MAE is the most interpretable metric.

B. Model A, as its MAE of 2.4 is lower than B's, which sits at 3.83.

C. Model A: B wins on MAE (1.5 vs 2.4) but loses on RMSE (3.83 vs 2.45).

D. Model B, as the RMSE difference comes from a single order and is therefore noise.

---

**Q33.** A factory monitors vibration sensors on its compressors with an LSTM anomaly detector. To run it on a gateway device without a deep-learning runtime, an engineer re-implements one LSTM step in NumPy. A colleague deleted one line by mistake.

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lstm_step(x, h_prev, c_prev, p):
    i = sigmoid(p.Wi @ x + p.Ui @ h_prev + p.bi)
    f = sigmoid(p.Wf @ x + p.Uf @ h_prev + p.bf)
    g = np.tanh(p.Wg @ x + p.Ug @ h_prev + p.bg)
    o = sigmoid(p.Wo @ x + p.Uo @ h_prev + p.bo)
    c = f * c_prev + i * g
    # ---- missing line ----
    return h, c
```

Which line completes the step?

A.
```python
h = o * c
```

B.
```python
h = o * np.tanh(c)
```

C.
```python
h = np.tanh(o * c)
```

D.
```python
h = f * np.tanh(c)
```

---

**Q34.** An insurer's claim-severity classifier (a shallow logistic model on 12 features) is evaluated on growing amounts of training data. The team plans to spend three months labelling 300,000 more claims to improve it.

| Training examples | Training accuracy | Validation accuracy |
|---|---|---|
| 1,000 | 0.78 | 0.74 |
| 5,000 | 0.77 | 0.75 |
| 20,000 | 0.77 | 0.75 |
| 80,000 | 0.76 | 0.75 |

The target accuracy is 0.85. Which recommendation is best supported?

A. Pause the labelling; add model capacity or features, since both curves have plateaued well short.

B. Proceed with labelling, since validation accuracy rises steadily as more training data is added.

C. Add strong regularization, since the gap between training and validation shows overfitting.

D. Proceed with labelling, as more data always reduces both bias and variance.

---

**Q35.** A hearing-aid company trained a GRU with PyTorch's `nn.GRU` and is porting the trained weights to a microcontroller, where the step below runs in fixed-point C after code generation. For the ported model to reproduce PyTorch's outputs exactly, every line must follow `nn.GRU`'s equations. One line is missing.

```python
import numpy as np

def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def gru_step(x, h_prev, P):
    r = sigmoid(P.W_ir @ x + P.b_ir + P.W_hr @ h_prev + P.b_hr)
    z = sigmoid(P.W_iz @ x + P.b_iz + P.W_hz @ h_prev + P.b_hz)
    n = np.tanh(P.W_in @ x + P.b_in + r * (P.W_hn @ h_prev + P.b_hn))
    # ---- missing line ----
    return h
```

Which line completes the step?

A.
```python
h = z * n + (1 - z) * h_prev
```

B.
```python
h = (1 - r) * n + r * h_prev
```

C.
```python
h = n + z * h_prev
```

D.
```python
h = (1 - z) * n + z * h_prev
```

---

**Q36.** A fintech team's transaction-sequence model uses an LSTM written with concatenated inputs: each gate multiplies a single weight matrix by `[h_prev; x]`. The forget-gate line was corrupted in a refactor and must be rewritten.

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lstm_step(x, h_prev, c_prev, W, b):
    hx = np.concatenate([h_prev, x])
    # ---- missing forget-gate line ----
    i = sigmoid(W["i"] @ hx + b["i"])
    g = np.tanh(W["g"] @ hx + b["g"])
    o = sigmoid(W["o"] @ hx + b["o"])
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h, c
```

Which line correctly restores the forget gate?

A.
```python
f = sigmoid(W["f"] @ hx + b["f"])
```

B.
```python
f = np.tanh(W["f"] @ hx + b["f"])
```

C.
```python
f = sigmoid(W["f"] @ np.concatenate([c_prev, x]) + b["f"])
```

D.
```python
f = 1 - i
```

---

**Q37.** A speech team feeds batches of 16 utterances, each 100 frames of 40 log-mel features, into a stacked bidirectional LSTM. A downstream layer must be sized from the output.

```python
import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=2,
               bidirectional=True, batch_first=True)
x = torch.randn(16, 100, 40)
out, (h_n, c_n) = lstm(x)
print(out.shape, h_n.shape)
```

What does the script print?

A. `out` is (16, 100, 128) and `h_n` is (2, 16, 128)

B. `out` is (16, 100, 256) and `h_n` is (2, 16, 256)

C. `out` is (16, 100, 256) and `h_n` is (4, 16, 128)

D. `out` is (100, 16, 256) and `h_n` is (4, 16, 128)

---

**Q38.** An audio-event classifier processes single-channel 128 × 128 spectrograms. The engineer must set `in_features` for the final linear layer.

```python
import torch.nn as nn

features = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
)
classifier = nn.Linear(IN_FEATURES, 12)
```

What must `IN_FEATURES` be?

A. 6272

B. 8192

C. 25088

D. 28800

---

**Q39.** A hospital trains a 3-D tumour-segmentation network on CT volumes. Memory limits each GPU to 2 volumes per batch, so the team runs 8 GPUs with standard `BatchNorm3d` layers (statistics computed per GPU). Training loss is noisy, and the model scores markedly worse in `eval()` mode than when validated in `train()` mode. Earlier experiments on smaller crops gave:

| Per-GPU batch size | Validation Dice (eval mode) | Validation Dice (train mode) |
|---|---|---|
| 16 (small crops) | 0.84 | 0.83 |
| 8 | 0.81 | 0.82 |
| 2 (full volumes) | 0.62 | 0.79 |

Which change most directly addresses the problem?

A. Raise BatchNorm's momentum, so its running estimates track the latest batches more closely.

B. Remove normalization entirely and compensate with a higher learning rate and more epochs.

C. Put dropout before every BatchNorm layer, so the network relies less on any single statistic.

D. Replace BatchNorm with GroupNorm, or synchronize the BatchNorm statistics across all eight GPUs.

---

**Q40.** A team trains a 20-layer fully connected network with tanh activations, using three initializations. They log the standard deviation of activations at selected layers after initialization, on one batch.

| Init | Layer 1 | Layer 5 | Layer 10 | Layer 20 |
|---|---|---|---|---|
| Normal, std = 0.01 | 0.050 | 2 × 10⁻⁵ | 7 × 10⁻¹² | ≈ 0 |
| Normal, std = 1.0 | 0.97 | 0.98 | 0.98 | 0.98 |
| Xavier (Glorot) | 0.63 | 0.55 | 0.52 | 0.50 |

The std = 1.0 run's loss does not move for the first 2,000 steps. Which diagnosis is correct?

A. std = 0.01 is safest, since small activations keep tanh in its linear region and gradients large.

B. std = 1.0 is ideal, as activations neither shrink nor explode; its slow start is optimizer warm-up.

C. std = 1.0 saturates tanh near ±1, so gradients vanish; Xavier keeps activations in the responsive range.

D. Xavier is best because it sets all weights to the same small value, keeping layers symmetric.

---

**Q41.** A road-maintenance company detects cracks in drone images. Typical cracks span 30–40 pixels, and the team wants each output unit of its feature extractor to see at least one whole crack. The extractor is:

| Order | Layer | Kernel | Stride |
|---|---|---|---|
| 1 | Conv | 3 × 3 | 1 |
| 2 | Conv | 3 × 3 | 1 |
| 3 | Max pool | 2 × 2 | 2 |
| 4 | Conv | 3 × 3 | 1 |
| 5 | Conv | 3 × 3 | 1 |

What is the receptive field of one output unit, and is it sufficient?

A. 10 × 10, from adding each layer's kernel minus one; too small.

B. 14 × 14; too small, so add further downsampling or dilated convolutions.

C. 18 × 18, enough to cover typical cracks once the pooling stride is counted.

D. 9 × 9, as only the four 3 × 3 convolutions contribute to the receptive field.

---

**Q42.** A card issuer's fraud model has been live for nine months. The monitoring dashboard tracks input drift and, with a 45-day lag (the time for chargebacks to confirm fraud), model performance.

| Month | PSI across all input features | Precision at the operating threshold | Recall at the operating threshold |
|---|---|---|---|
| Jan | 0.03 | 0.71 | 0.68 |
| Mar | 0.04 | 0.69 | 0.66 |
| May | 0.03 | 0.58 | 0.49 |
| Jul | 0.05 | 0.52 | 0.41 |

The fraud-ops team reports that criminals have switched to small "test" transactions at merchants the model historically treated as low-risk. The ML lead says: "Inputs haven't drifted, so the model must still be fine — the labels are probably noisy."

Which assessment and response are correct?

A. This is concept drift; retrain on recent labelled data and monitor performance as well as inputs.

B. This is covariate shift; reweight the training data so its inputs match the current input distribution.

C. Label noise is the likely cause, since stable PSI shows that the model's inputs have not changed.

D. This is seasonality, so the team should wait for performance to recover as the year turns.

---

**Q43.** A skin-condition app classifies phone photos into 12 conditions. Users have started uploading photos of pets, food and documents, and the model confidently labels them as conditions. The team measured the maximum softmax probability on two sets:

| Set | Median max-softmax | Share with max-softmax > 0.9 |
|---|---|---|
| In-distribution skin photos (validation) | 0.93 | 61% |
| Out-of-distribution photos (pets, food, documents) | 0.88 | 44% |

The product manager proposes rejecting any image with max-softmax below 0.9. Which design is most appropriate?

A. Use the 0.9 softmax threshold as proposed, since confident predictions are reliable by definition.

B. Retrain with temperature scaling, which teaches the model to reject non-skin photos by itself.

C. Add a dedicated out-of-distribution check before classification, and abstain or ask for a retake.

D. Add a twelfth "other" class trained on one batch of pet photos, which covers all future inputs.

---

**Q44.** A researcher writes a custom loss and checks its gradient against the standard cross-entropy gradient for a single 3-class example whose true class is 0.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
target = torch.tensor([0])

loss = F.cross_entropy(logits.unsqueeze(0), target)
loss.backward()
print(logits.softmax(0).detach().numpy().round(3))
print(logits.grad.numpy().round(3))
```

The first line prints `[0.659 0.242 0.099]`. What does the second line print?

A. `[ 0.659  0.242  0.099]`

B. `[ 0.341 -0.242 -0.099]`

C. `[-0.341  0.242  0.099]`

D. `[-0.659  0.758  0.901]`

---

**Q45.** A lender uses LIME to explain individual credit decisions to its review team. For one rejected applicant, three runs of the explainer on the same model and the same input produce:

| Run | Top three reasons for rejection |
|---|---|
| 1 | Credit utilization, recent enquiries, income |
| 2 | Recent enquiries, employment length, utilization |
| 3 | Income, account age, recent enquiries |

The review team asks whether they can use these explanations in adverse-action letters. What is the most accurate assessment?

A. The model is unstable, since a deterministic model cannot produce differing explanations for one input.

B. LIME's random perturbations make its rankings unstable, so these runs cannot be used as they stand.

C. All three runs are equally valid, so the letter can list the union of every reason shown.

D. The input data must be corrupted, as explanations should depend only on the model weights.

---

**Q46.** A logistics company's invoice-reading model reaches 97% field accuracy on its curated test set, but drivers report frequent failures when photographing invoices in cabs at night. The reliability team breaks down a field sample:

| Capture condition | Share of field photos | Field accuracy |
|---|---|---|
| Flat, well lit, upright | 38% | 96% |
| Rotated 10–30° | 22% | 81% |
| Low light / motion blur | 27% | 63% |
| Crumpled or folded | 13% | 70% |

Which approach should the team take to make the model reliable in the field?

A. Collect more flat, well-lit invoices, since that condition dominates the curated test set.

B. Report the 97% curated accuracy alongside a disclaimer that field results may vary in practice.

C. Raise the confidence threshold until only well-lit images pass, so overall accuracy rises above 96%.

D. Build a stress-test suite per condition, augment for those conditions, and track accuracy per slice over time.

---

**Q47.** A grocery-delivery app wants each ETA shown with a 90% prediction interval. The team applies split conformal prediction: on a held-out calibration set of 19 orders, the absolute errors (minutes) of the point model are:

1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 12, 14, 18, 25

The conformal radius is the ⌈(n + 1)(1 − α)⌉-th smallest absolute error. Which interval should accompany a new prediction of 40 minutes?

A. 40 ± 14.8 minutes, the 90th percentile of the calibration errors.

B. 40 ± 17.6 minutes, using the mean error plus 1.645 standard deviations.

C. 40 ± 18 minutes, the 18th smallest of the 19 calibration errors.

D. 40 ± 25 minutes, the largest calibration error, to guarantee coverage.

---

**Q48.** A regulator requires a bank to tell every rejected credit applicant the specific main reasons for their own rejection. The bank's data-science team proposes attaching the model's global feature-importance chart (income, utilization, account age, enquiries, in that order) to every rejection letter, saying it "explains how the model decides."

Which approach satisfies the requirement?

A. Generate per-applicant attributions and report each rejected applicant's own top negative contributors.

B. Attach the global importance chart, since it reflects the features that drive all of the decisions.

C. Publish the model's full source code and weights, which is the most complete form of transparency.

D. List the four globally important features in alphabetical order to avoid implying any ranking.

---

**Q49.** A ride-hailing company has retrained its surge-pricing demand forecaster. Offline metrics improved by 6%, but the previous version has run reliably for a year, and a pricing error can affect millions of trips within hours. The reliability lead asks for a rollout plan.

Which plan is most appropriate?

A. Replace the old model everywhere at once, since offline metrics already show a clear improvement.

B. Run the old and new models on alternate days for a month, then keep whichever earned more revenue.

C. Deploy to the city with the most trips first, so problems surface as quickly as possible.

D. Run the new model in shadow, then a small canary with predefined rollback triggers, before full rollout.

---

**Q50.** An electronics marketplace wants to merge duplicate listings created by different sellers. As a first-pass similarity score it compares listing titles using character trigrams (spaces replaced by `_`) and cosine similarity on trigram counts.

```python
import math
from collections import Counter

titles = {
    "A": "voltaro nebula s24 ultra 256gb",
    "B": "nebula s24 ultra voltaro 256 gb",
    "C": "voltaro nebula s23 ultra 256gb",
}

def trigrams(s):
    s = s.replace(" ", "_")
    return Counter(s[i:i + 3] for i in range(len(s) - 2))

def cosine(a, b):
    dot = sum(a[k] * b[k] for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)

for p, q in [("A", "B"), ("A", "C"), ("B", "C")]:
    print(p, q, round(cosine(trigrams(titles[p]), trigrams(titles[q])), 3))
```

A and B are the same phone; C is a different model. Which statement about the output, and its implication for the merge pipeline, is correct?

A. A–B scores highest at about 0.81, so trigram cosine alone can safely drive merges.

B. A–B and A–C tie, as both pairs share the same multiset of characters.

C. A–C scores highest at about 0.89; a model-number check must gate any merge.

D. A–C scores highest at about 0.89, which correctly marks them as duplicates.

---

**Q51.** A product-review sentiment model scores 91% accuracy overall. An error analysis on 4,000 validation reviews, sliced by content, shows:

| Slice | Reviews | Accuracy |
|---|---|---|
| All reviews | 4,000 | 91% |
| Mentions brand "Zentra" | 310 | 64% |
| Mentions any other brand | 1,150 | 92% |
| No brand mentioned | 2,540 | 93% |

On inspection, 88% of "Zentra" reviews in the training data were negative, after a product recall, and the model labels nearly every Zentra review negative, even glowing ones. What is the best next step?

A. Remove all reviews mentioning Zentra from the evaluation set, since they are a small slice.

B. Augment with counterfactual examples that vary sentiment for Zentra, then re-test that slice.

C. Accept the result, since the model is right that most Zentra reviews in training were negative.

D. Raise the decision threshold for negative predictions globally until Zentra accuracy reaches 90%.

---

**Q52.** A broadband provider's help-centre search ranks articles by TF-IDF cosine similarity. Customers searching "reset password" complain that a router article appears above the account-password article.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "how to reset your password from the login page",                     # doc 0
    "update your billing address and payment card",                       # doc 1
    "reset the router to factory settings and reset the wifi password",   # doc 2
    "change your account email address",                                  # doc 3
]
vec = TfidfVectorizer()
X = vec.fit_transform(docs)
scores = cosine_similarity(vec.transform(["reset password"]), X)[0]
print(scores.round(3))
```

Which statement is correct?

A. Doc 2 ranks first (≈ 0.51 vs ≈ 0.43), as the repeated "reset" raises its weight.

B. Doc 0 ranks first, since it is the only article that is about account passwords.

C. Doc 2 ranks first, and switching to sublinear term frequency would reverse the ranking.

D. Docs 0 and 2 tie, as each contains both query terms at least once.

---

**Q53.** A bulk-SMS provider uses a multinomial Naive Bayes spam filter with Laplace smoothing (α = 1) over a small vocabulary. Prior P(spam) = 0.3. Spammers have started padding messages with repeated trigger words.

```python
import math

V = 6
spam = dict(winner=30, prize=20, invoice=2, report=1, click=25, family=2)
ham  = dict(winner=4,  prize=1,  invoice=30, report=25, click=5,  family=35)

def log_post(counts, prior, msg):
    total = sum(counts.values())
    return math.log(prior) + sum(
        n * math.log((counts[w] + 1) / (total + V)) for w, n in msg.items())

for msg in [dict(winner=1, invoice=1, report=1, family=1),
            dict(winner=4, invoice=1, report=1, family=1)]:
    s, h = log_post(spam, 0.3, msg), log_post(ham, 0.7, msg)
    print(round(1 / (1 + math.exp(h - s)), 3))
```

What does the script show, and what design change addresses the attack?

A. Both messages score near 0.004, since three ham words outweigh any single spam word.

B. P(spam) rises from about 0.004 to 0.37; raising α to 10 would remove the effect.

C. P(spam) rises from about 0.37 to 0.63, as only the extra copies of "winner" are counted.

D. P(spam) jumps from 0.004 to 0.63; use binary presence features or cap counts.

---

**Q54.** An online pharmacy extracts drug names, doses and frequencies from customer chats. Downstream systems need exact spans: "650" without its unit cannot be used to fill a prescription. On one chat, gold and predicted entities are:

```python
gold = {("Dolo 650", "DRUG"), ("650 mg", "DOSE"), ("Azithral", "DRUG"),
        ("500 mg", "DOSE"), ("twice daily", "FREQ")}
pred = {("Dolo 650", "DRUG"), ("650", "DOSE"), ("Azithral", "DRUG"),
        ("500 mg", "DOSE"), ("daily", "FREQ"), ("Crocin", "DRUG")}

tp = len(gold & pred)
precision, recall = tp / len(pred), tp / len(gold)
f1 = 2 * precision * recall / (precision + recall)
print(round(precision, 2), round(recall, 2), round(f1, 2))
```

What is printed, and is strict span matching the right evaluation for this use?

A. 0.83, 1.0, 0.91; partial overlaps count as matches, which suits a downstream filler.

B. 0.5, 0.6, 0.55; strict matching fits, since partial spans are unusable downstream.

C. 0.6, 0.5, 0.55; strict matching fits, since precision counts gold entities found.

D. 0.5, 0.6, 0.55; token-level scoring would suit better, giving credit for partial dose spans.

---

**Q55.** A logistics company is building an NER model to pull shipment IDs, city names, company names and dates out of operations emails written in mixed English and Hindi. A junior engineer proposes the classic text-classification pipeline: lowercase everything, remove stop words and punctuation, then apply Porter stemming, "to reduce vocabulary size."

Which preprocessing choice is most appropriate for this NER system?

A. Keep case, punctuation and stop words, and never stem, since these cues help mark entities.

B. Apply the full pipeline as proposed, since a smaller vocabulary always improves NER accuracy.

C. Lowercase and stem, but keep stop words, since only stop-word removal can damage entities.

D. Remove punctuation only, as shipment IDs like "SHP-2291/B" are noise to a language model.

---

**Q56.** A laptop maker's review dashboard uses a document-level sentiment classifier (positive, neutral, negative). Product managers say it is useless for prioritizing engineering fixes. A sample of outputs:

| Review | Predicted label |
|---|---|
| "Battery life is superb, but the hinge cracked within a month." | Neutral |
| "Screen is gorgeous; keyboard flexes badly and the fan is loud." | Neutral |
| "Fast, light and the trackpad is the best I've used." | Positive |
| "Speakers are tinny. Everything else is fine." | Neutral |

The managers want to know which components customers praise or criticise, and how often. Which redesign best meets that need?

A. Retrain the classifier with five sentiment levels, so mixed reviews are separated more finely.

B. Drop neutral predictions from the dashboard, as they carry no actionable signal at all.

C. Average the sentiment scores per product, then rank products by their overall mean score.

D. Move to aspect-based sentiment: extract components and assign a polarity to each.

---

**Q57.** A Q&A forum flags duplicate questions using embedding similarity. Its vector index was configured for inner-product search, and the embedding model's outputs are not normalized. Moderators notice that long, generic posts keep appearing as the "most similar" question to short, specific ones. An engineer reproduces the ranking for one query:

```python
import numpy as np

q = np.array([1.0, 1.0, 0.0])
cands = {"Q1": np.array([2.0, 2.0, 0.2]),
         "Q2": np.array([0.9, 1.1, 0.0]),
         "Q3": np.array([4.0, 1.0, 3.0])}

for name, v in cands.items():
    ip = q @ v
    cos = ip / (np.linalg.norm(q) * np.linalg.norm(v))
    print(name, round(ip, 3), round(cos, 3))
```

Which statement is correct?

A. Inner product ranks Q1 first at 4.0; cosine ranks Q3 first, as its norm is largest.

B. Inner product ranks Q3 first at 5.0; cosine ranks Q1 first at ≈ 0.998.

C. Both rank Q3 first, as it has the largest magnitude along the first dimension.

D. Inner product ranks Q3 first; cosine ranks Q2 first, as it is closest in length.

---

**Q58.** A fintech app lets users photograph cheques and flattens each photo into a straight-on 800 × 1100 image before OCR. Corner detection returns the four cheque corners in the order top-left, top-right, bottom-right, bottom-left. Users report that every flattened image comes out as a twisted, "bow-tie" smear instead of a clean rectangle.

```python
import cv2
import numpy as np

img = cv2.imread("cheque.jpg")
corners = detect_corners(img)          # [[112, 80], [930, 140], [880, 1210], [60, 1150]]
src = np.float32(corners)              # TL, TR, BR, BL                     line 6

W, H = 800, 1100
dst = np.float32([[0, 0], [W, 0], [0, H], [W, H]])                        # line 9

M = cv2.getPerspectiveTransform(src, dst)                                 # line 11
flat = cv2.warpPerspective(img, M, (W, H))                                # line 12
```

Which line is responsible, and why?

A. Line 12, since `dsize` must be passed as (height, width), not (width, height).

B. Line 11, since `getPerspectiveTransform` needs at least six point pairs.

C. Line 6, since corner coordinates must be integers for a perspective transform.

D. Line 9, as the destination corners are listed in a different order from the source.

---

**Q59.** A packaging plant measures carton outlines on a conveyor belt with Canny edge detection. The belt has a fine woven texture. Two problems appear in the edge maps: (1) many short speckle edges scattered across the belt, and (2) the carton outline is broken into fragments wherever the carton's colour is close to the belt's.

```python
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blur, 110, 120)          # (low, high) hysteresis thresholds
```

Which change addresses both problems?

A. Raise both thresholds to 200 and 220, so that both the speckle and the gaps disappear together.

B. Use a larger Gaussian kernel, and lower the low threshold while keeping the high threshold.

C. Remove the blur entirely, so that the carton's outline stays sharp and remains connected.

D. Lower the high threshold to 60, so that more outline pixels qualify as strong edges.

---

**Q60.** A review classifier keeps misreading negated phrases. The team switches from unigrams to unigrams plus bigrams, using whitespace tokenization with no stop-word removal, and inspects the vocabulary built from three training sentences:

| # | Sentence |
|---|---|
| 1 | the battery is not good |
| 2 | the battery is good |
| 3 | the screen is not bad |

How many features does the combined unigram and bigram vocabulary contain, and does it capture the negation?

A. 7 features, since bigrams reuse the existing unigram vocabulary entries.

B. 22 features, counting every bigram occurrence across the three sentences.

C. 15 features, including "not good" and "not bad" as distinct bigrams.

D. 15 features, but negation is still lost, because "not" is a stop word.

---

**Q61.** A quality-inspection team is upgrading its Vision Transformer. The current model uses 224 × 224 inputs with 16 × 16 patches; the proposed model uses 448 × 448 inputs with 14 × 14 patches, to resolve finer defects. Both prepend one class token.

| Model | Input | Patch size |
|---|---|---|
| Current | 224 × 224 | 16 × 16 |
| Proposed | 448 × 448 | 14 × 14 |

How many tokens does the proposed model process, and roughly how much larger is each self-attention score matrix than the current model's?

A. 1,025 tokens, with an attention matrix about 27 times larger.

B. 1,024 tokens, about 4 times larger, matching the fourfold rise in pixels.

C. 1,025 tokens, about 5.2 times larger, since attention grows linearly with tokens.

D. 785 tokens, about 16 times larger, from (448/16)² + 1 patches.

---

**Q62.** A pharmaceutical line counts tablets in a backlit tray. Thresholding produces a binary mask with two tablets, but also isolated white specks from dust and small black holes inside the tablets from surface glare. The engineer cleans the mask before counting connected components.

```python
import cv2
import numpy as np

mask = np.zeros((60, 60), np.uint8)
cv2.rectangle(mask, (5, 5), (25, 25), 255, -1)       # tablet 1
cv2.rectangle(mask, (35, 35), (55, 55), 255, -1)     # tablet 2
mask[14:16, 14:16] = 0                               # 2x2 glare hole
mask[44:46, 44:46] = 0                               # 2x2 glare hole
for y, x in [(2, 40), (30, 30), (50, 10), (10, 50)]:
    mask[y, x] = 255                                 # single-pixel dust specks

k = np.ones((3, 3), np.uint8)
clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k)

n_objects = cv2.connectedComponents(clean)[0] - 1    # minus background
n_hole_px = int((clean[5:26, 5:26] == 0).sum() + (clean[35:56, 35:56] == 0).sum())
print(n_objects, n_hole_px)
```

What does the script print?

A. `6 0`

B. `2 8`

C. `2 0`

D. `6 8`

---

**Q63.** A warehouse robot tracks orange totes with a colour threshold. It works near the windows but loses totes in shaded aisles. Sampled tote pixels (BGR) and their HSV conversions:

```python
bright = (0, 128, 255)     # HSV -> (15, 255, 255)
shaded = (0,  64, 128)     # HSV -> (15, 255, 128)

mask = cv2.inRange(frame, (0, 100, 200), (60, 170, 255))   # current BGR range
```

Which change makes detection robust to the lighting difference?

A. Convert to HSV and threshold on a hue band around 15, with a wide value range.

B. Widen the BGR range to (0, 0, 0)–(255, 255, 255), so shaded totes are included.

C. Keep BGR but convert frames to grayscale first, which removes the effect of colour.

D. Threshold on the HSV value channel alone, since brightness separates totes best.

---

**Q64.** A solar-farm operator runs an object detector on drone photos (4000 × 3000) to find cracked panel cells. Images are resized to 640 × 480 before inference. Cracks are typically 15–25 pixels across in the original photos. Validation results:

| Object size (in original image) | Share of labelled defects | Average precision |
|---|---|---|
| Large (> 200 px) — soiled panels | 20% | 0.81 |
| Medium (50–200 px) — broken glass | 35% | 0.58 |
| Small (< 50 px) — cell cracks | 45% | 0.12 |

Which design change most directly targets the weakest category?

A. Train for more epochs with stronger colour augmentation, as small cracks are under-trained.

B. Replace the detector with an image classifier that labels each photo as cracked or not.

C. Lower the confidence threshold, so that faint crack detections are kept in the output.

D. Tile each full-resolution photo into overlapping crops for inference, then merge detections.

---

**Q65.** A city traffic project plans to stream raw (uncompressed) 3840 × 2160 RGB video at 30 frames per second from each roadside camera over a 100 Mbit/s cellular uplink to a central server, which will run vehicle detection.

| Setting | Value |
|---|---|
| Resolution | 3840 × 2160 |
| Channels | 3 (8-bit each) |
| Frame rate | 30 fps |
| Uplink | 100 Mbit/s |

Which assessment of the design is correct?

A. About 93 MB/s of raw video, which a 100 Mbit/s link carries comfortably.

B. About 750 MB/s, which the link carries after standard Wi-Fi compression is applied.

C. About 750 MB/s raw; run detection on the device and send only events or crops.

D. About 25 MB/s, as only one colour channel needs transmitting for detection.

---

**Q66.** A company serves a 40-layer, 13B-parameter decoder on 80 GB GPUs. After model weights and activations, about 50 GiB per GPU remains for the KV cache. The workload is 16 concurrent sequences of 8,192 tokens in fp16, with 40 query heads of dimension 128. The research team has three variants of the model, differing only in the number of key/value heads, each fine-tuned to convergence:

| Variant | K/V heads | KV cache for this workload | Benchmark score |
|---|---|---|---|
| MHA | 40 | 100 GiB | 71.2 |
| GQA-8 | 8 | 20 GiB | 70.9 |
| MQA | 1 | 2.5 GiB | 68.4 |

The product owner wants the best quality that fits the serving budget without reducing concurrency or context length. Which variant should be deployed?

A. MHA, after halving the batch to 8 sequences, since quality should never be traded for memory.

B. GQA-8, which fits in 20 GiB while staying close to MHA's quality.

C. MQA, since its 2.5 GiB cache leaves the most headroom for future growth.

D. MHA, which fits once the cache is stored in fp16 rather than fp32.

---

**Q67.** A self-driving dataset's segmentation masks store class IDs: 0 = background, 1 = pedestrian, 2 = vehicle. A preprocessing script resizes images and masks with the same call. After retraining, the model hallucinates thin "pedestrian" outlines around every vehicle.

```python
import cv2
import numpy as np

mask = np.zeros((4, 4), np.uint8)
mask[:, 2:] = 2                                   # a vehicle on the right

up_lin = cv2.resize(mask, (8, 8), interpolation=cv2.INTER_LINEAR)
up_nn  = cv2.resize(mask, (8, 8), interpolation=cv2.INTER_NEAREST)
print(np.unique(up_lin), np.unique(up_nn))
```

What does the script print, and what explains the hallucinated outlines?

A. `[0 2] [0 2]`; the outlines come from the image resize, not from the mask.

B. `[0 1 2] [0 2]`; linear interpolation creates class 1 at vehicle borders.

C. `[0 1 2] [0 1 2]`; both modes blend classes, so masks should never be resized.

D. `[0 2] [0 1 2]`; nearest-neighbour duplicates pixels and invents the extra class.

---

**Q68.** A support-summary service batches prompts of different lengths for a decoder-only model. Prompts are left-padded with `PAD = 0` so their final tokens align. The same ticket yields a different summary depending on which other tickets share its batch; run alone, it is always correct.

```python
import torch

def attention(q, k, v, ids):                      # q, k, v: (B, T, d); ids: (B, T)
    B, T, d = q.shape
    scores = q @ k.transpose(-2, -1) / d ** 0.5                       # line 4
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)        # line 5
    scores = scores.masked_fill(causal, float("-inf"))                # line 6
    weights = scores.softmax(dim=-1)                                  # line 7
    return weights @ v

batch = left_pad([tok(t) for t in tickets], pad_id=0)                 # line 10
```

What is the cause, and where should it be fixed?

A. Line 4, as the scale must use the unpadded length rather than the padded length `d`.

B. Line 10, since left-padding is invalid for decoder models and must be replaced.

C. Line 6 masks only future positions; pad keys must also be masked, using the `ids == PAD` test.

D. Line 7, since softmax should run over the query axis when sequences are padded.

---

**Q69.** A creative-writing assistant samples the next token with temperature and nucleus (top-p) filtering. For one step, the model's logits over its five candidate tokens are:

```python
import numpy as np

tokens = ["A", "B", "C", "D", "E"]
logits = np.array([3.0, 2.5, 1.0, 0.5, 0.0])

def nucleus(logits, temperature=0.7, top_p=0.9):
    p = np.exp(logits / temperature)
    p /= p.sum()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, top_p) + 1)       # smallest set reaching top_p
    return [tokens[i] for i in order[:k]]

print(nucleus(logits))
```

What does the script print?

A. `['A', 'B']`

B. `['A', 'B', 'C']`

C. `['A']`

D. `['A', 'B', 'C', 'D']`

---

**Q70.** A bank's support assistant is budgeted assuming every customer message costs the same number of tokens, because "messages average 60 words in every language." Measured tokenizer behaviour on real messages:

| Language | Share of traffic | Tokens per 60-word message |
|---|---|---|
| English | 50% | 78 |
| Hindi (Devanagari) | 30% | 240 |
| Tamil | 20% | 330 |

The budget was set from English alone. Which statement is correct?

A. Cost matches the budget, since tokenizers split every language into words at the same rate.

B. The average is roughly 216 tokens, the unweighted mean of the three languages.

C. The average is about 78 tokens, as non-English messages are usually translated into English first.

D. The average is about 177 tokens, roughly 2.3 times the English-based budget.

---

**Q71.** A team fine-tunes a base model into a chat assistant on 40,000 (prompt, response) pairs. Prompts average 600 tokens and responses 80. After training, the assistant often continues past its answer and writes a new "User:" turn, and responses improved less than expected.

```python
def build_example(prompt, response, tok):
    p_ids = tok.encode(f"User: {prompt}\nAssistant:")
    r_ids = tok.encode(f" {response}") + [tok.eos_id]
    input_ids = p_ids + r_ids                                   # line 4
    labels = list(input_ids)                                    # line 5
    return {"input_ids": input_ids, "labels": labels}

# Training uses the standard causal-LM loss, which ignores label value -100.
```

Which line is responsible, and what is the fix?

A. Line 4, as the prompt and response must be placed in separate training examples.

B. Line 5 trains on prompt tokens too; set the prompt positions' labels to -100.

C. Line 4, since the EOS token should end the prompt rather than the response.

D. Line 5, which must shift the labels left by one position before they are returned.

---

**Q72.** A team sizing a custom encoder counts the parameters in one Transformer block: multi-head self-attention with Q, K, V and output projections (all with biases), a two-layer feed-forward network with biases, and two LayerNorms.

```python
d_model, n_heads, d_ff = 1024, 16, 4096

attn = 4 * (d_model * d_model + d_model)
ffn  = d_model * d_ff + d_ff + d_ff * d_model + d_model
norm = 2 * (2 * d_model)
print(attn + ffn + norm)
```

What does the script print?

A. 12,596,224

B. 12,582,912

C. 16,777,216

D. 20,989,952

---

**Q73.** An engineer adds KV caching to a home-grown decoder that uses learned absolute position embeddings. Without the cache, outputs are fluent; with it, text degrades into repetition after the first few generated tokens.

```python
@torch.no_grad()
def generate(model, ids, max_new=50):
    cache = None
    x = ids                                                      # full prompt first
    for _ in range(max_new):
        pos = torch.arange(x.size(1), device=x.device)           # line 6
        logits, cache = model(x, position_ids=pos, past=cache)   # line 7
        next_id = logits[:, -1].argmax(-1, keepdim=True)         # line 8
        ids = torch.cat([ids, next_id], dim=1)                   # line 9
        x = next_id                                              # line 10
    return ids
```

Which line causes the degradation?

A. Line 8, since greedy argmax combined with caching produces repetition by design.

B. Line 10, since only the newest token is fed, so earlier context is lost each step.

C. Line 9, as appending to `ids` while caching duplicates every token in the cache.

D. Line 6 restarts positions at 0 for each new token instead of continuing from the cache length.

---

**Q74.** A marketplace classifies about two million short product reviews a day into positive, negative or mixed. The label is parsed by code and feeds a dashboard; no human reads the model's output. The platform team is billed per input and output token, and the classifier sits on the request path of a seller-facing page, so every extra output token adds visible latency. Accuracy is roughly equal across the prompts below.

**Objective:** the lowest latency and token cost per call while still returning a parseable, correct label.

**P1:** "Classify the review's sentiment. Explain your reasoning step by step, then give the label."

**P2:** "Classify the review's sentiment as positive, negative or mixed."

**P3:** "Classify the review's sentiment. Reply with exactly one word: positive, negative or mixed."

**P4:** "Classify the review's sentiment as positive, negative or mixed, then write one sentence justifying the label."

**P5:** "You are an expert sentiment analyst with deep knowledge of e-commerce marketplaces. Classify the review's sentiment as positive, negative or mixed."

Rank the prompts from most to least effective against the objective.

A. P3 > P2 > P4 > P5 > P1

B. P3 > P2 > P5 > P4 > P1

C. P2 > P3 > P5 > P4 > P1

D. P1 > P3 > P2 > P5 > P4

---

**Q75.** An internal developer assistant generates Python data-access functions for a PostgreSQL reporting database from short natural-language requests. Requests vary widely: exact-match filters, partial-name searches with `LIKE`, filters on a list of IDs, and sorting by a column the end user picks from a dropdown. Every generated function is merged after only a light review, so the security team wants the prompt itself to prevent injection.

**Objective:** no generated function is vulnerable to SQL injection, across all of these request types.

**P1:** "Write the Python function for this request."

**P2:** "Write the Python function for this request. Make sure it is secure."

**P3:** "Write the Python function for this request, following the style of this example: `cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))`"

**P4:** "Write the Python function for this request. Do not concatenate strings into SQL."

**P5:** "Write the Python function for this request. Pass every user-supplied value as a bound parameter, never by string formatting. For identifiers such as a column name in ORDER BY, which cannot be bound, map the user's choice against a fixed allow-list."

Rank the prompts from most to least effective against the objective.

A. P3 > P5 > P4 > P2 > P1

B. P5 > P3 > P2 > P4 > P1

C. P5 > P4 > P3 > P2 > P1

D. P5 > P3 > P4 > P2 > P1

---

**Q76.** A telecom's support desk routes tickets to three queues: Billing, Technical and Account. Many tickets mix categories ("I was charged twice because the app crashed during payment"). The same ticket is sometimes re-triaged after an edit, and supervisors complain that it often lands in a different queue the second time, which resets its service-level clock.

**Objective:** consistency — the same ticket, or two near-identical tickets, should always land in the same queue, including mixed-category tickets.

**P1:** "Assign exactly one queue. If a ticket fits more than one, decide by what the customer needs resolved: money returned or a charge corrected → Billing; something not working → Technical; login, profile or ownership → Account."

**P2:** "Assign the most appropriate queue: Billing, Technical or Account."

**P3:** "You are a senior support lead. Assign the best queue — Billing, Technical or Account — using your judgement on each ticket's particular circumstances."

**P4:** "Assign a queue: Billing, Technical or Account. Examples: 'Card declined at checkout' → Billing. 'App crashes on launch' → Technical."

**P5:** "Assign a queue: Billing, Technical or Account. When a ticket could fit more than one queue, choose Technical."

Rank the prompts from most to least effective against the objective.

A. P1 > P5 > P4 > P2 > P3

B. P1 > P5 > P2 > P4 > P3

C. P1 > P4 > P5 > P3 > P2

D. P3 > P1 > P5 > P4 > P2

---

**Q77.** A retail bank's mobile app was down for 90 minutes because an internal service's TLS certificate expired. The engineering incident report is accurate but dense. It must now be rewritten for 1,400 branch managers, who will read it on their phones between customers. Afterwards, a short quiz checks whether each manager can say what happened, whether it could happen again, and what to tell customers who ask.

**Objective:** managers can correctly answer those three questions after one read.

**P1:** "Explain this incident report."

**P2:** "Rewrite this incident report for bank branch managers with no technical background, reading on a phone between customers. They need to know what happened, whether it could recur, and what to tell customers. Use plain words and short paragraphs."

**P3:** "Summarize this incident report in exactly one sentence."

**P4:** "Rewrite this incident report in simple language. Avoid jargon such as TLS, certificate, handshake and endpoint."

**P5:** "Rewrite this incident report in simple language, keeping all of the technical details so that nothing is lost."

Rank the prompts from most to least effective against the objective.

A. P2 > P4 > P1 > P5 > P3

B. P4 > P2 > P5 > P1 > P3

C. P2 > P4 > P5 > P1 > P3

D. P2 > P3 > P4 > P5 > P1

---

**Q78.** A hospital pilots an LLM that reads discharge notes and lists every medication mentioned, so a pharmacist can reconcile them against the pharmacy record. The pharmacist reviews every list and removes anything irrelevant in seconds. A missed medication, however — say, an anticoagulant mentioned as "held for surgery" — can lead to a dangerous interaction after discharge.

**Objective:** maximize recall of medication mentions in each note.

**P1:** "List the medications mentioned in this note."

**P2:** "List every medication mentioned in this note, including brand and generic names and abbreviations."

**P3:** "Extract the patient's current medications from this note."

**P4:** "List every medication mentioned in this note, including brand and generic names, abbreviations, medications that were stopped or held, and ones mentioned only in passing."

**P5:** "You are a clinical pharmacist. List every medication in this note as a table of drug name, dose, route, frequency and indication. Include only medications whose dose and frequency are explicitly stated, so the list is clinically actionable."

Rank the prompts from most to least effective against the objective.

A. P4 > P2 > P3 > P1 > P5

B. P5 > P4 > P2 > P1 > P3

C. P4 > P5 > P2 > P1 > P3

D. P4 > P2 > P1 > P3 > P5

---

**Q79.** An accounts-payable team extracts invoice details from scanned PDFs, and each invoice must become one line in a file loaded by a 20-year-old importer. The importer rejects any line that deviates from its format: invoice number, then the date as DD/MM/YYYY, then the amount with exactly two decimals and no thousands separator, then the vendor name in double quotes, with any double quote inside the name doubled.

**Objective:** every output line is accepted by the importer.

**P1:** "Output each invoice in a format a spreadsheet can open."

**P2:** "Output each invoice as one CSV line: invoice number, date, amount, vendor."

**P3:** "Output each invoice as one CSV line, exactly like these examples:
`INV-0042,07/03/2026,18450.00,"Shah & Sons"`
`INV-0043,15/03/2026,920.50,"The ""Blue"" Door Ltd"`"

**P4:** "Output each invoice as one CSV line: invoice number, then the date as DD/MM/YYYY, then the amount with two decimals, then the vendor name in quotes."

**P5:** "You are a meticulous data-entry clerk. Output each invoice as a CSV line."

Rank the prompts from most to least effective against the objective.

A. P3 > P4 > P2 > P5 > P1

B. P3 > P4 > P5 > P2 > P1

C. P4 > P3 > P2 > P5 > P1

D. P4 > P3 > P5 > P1 > P2

---

**Q80.** An e-commerce support agent handles about 50,000 customer messages a day. Every message goes through the same planning loop on a large model.

```python
LARGE = "large-model"          # ~2.5 s and ~₹0.40 per call

def handle(message):
    history = [{"role": "system", "content": AGENT_PROMPT},
               {"role": "user", "content": message}]
    for step in range(6):
        reply = llm.chat(model=LARGE, messages=history, tools=TOOLS)
        if not reply.tool_calls:
            return reply.content
        history.append(reply.as_message())
        for call in reply.tool_calls:
            history.append(run_tool(call))
    return ESCALATE_TO_HUMAN
```

Traffic analysis shows the following.

| Message type | Share | Large-model calls today | Example |
|---|---|---|---|
| Simple | 70% | 2 | "Where is order 4471?", "Reset my password" |
| Complex | 30% | 4 | Refund disputes spanning several orders |

A small intent classifier costs about 0.1 s and ₹0.002 per message and identifies simple intents with 98% precision. Order IDs and wording vary from message to message. The team wants to cut median latency and cost without degrading complex cases.

Which change should the team make?

A. Cache each final reply keyed on the exact message text, so repeat questions skip the model.

B. Move every message to a small model, which cuts each step's latency and cost roughly tenfold.

C. Classify intent with the small model, serve simple intents by direct tool call, and escalate the rest.

D. Raise the step limit so the planner finishes complex cases in one pass instead of retrying.

---

**Q81.** A travel company's itinerary agent gathers four independent pieces of data for each request, then makes two LLM calls: one to plan and one to write the itinerary. Profiling one request shows the timings below.

```python
async def build_itinerary(req):
    plan = await llm.plan(req)                         # 1.5 s
    weather = await get_weather(req.city, req.dates)   # 1.2 s
    flights = await get_flights(req.origin, req.city)  # 3.5 s
    hotels  = await get_hotels(req.city, req.dates)    # 2.8 s
    visa    = await get_visa_rules(req.passport)       # 0.9 s
    return await llm.write(plan, weather, flights, hotels, visa)  # 1.5 s
```

Most requests are for unique city, date and passport combinations, and the business wants the largest reduction in end-to-end latency for a typical request.

Which change gives the largest reduction, and to roughly what latency?

A. Cache tool results per city, which brings every request down to about 3 s.

B. Issue the four tool calls concurrently, bringing a request to about 6.5 s.

C. Use a smaller LLM that halves each LLM call, bringing a request to about 9.9 s.

D. Issue the four tool calls concurrently, bringing a request to about 3.5 s.

---

**Q82.** A law firm wants an assistant that drafts clause-level answers in Indian contract law, using the firm's own terminology and house style. Its resources:

| Resource | Size |
|---|---|
| Unlabelled contracts, opinions and memos | ~2 GB of text |
| Curated (question, ideal answer) pairs | 800 |
| Base model | General 7B pretrained decoder |
| Compute | A few days on 8 GPUs |

Which training strategy best fits?

A. Pretrain a new model from scratch on the 2 GB corpus, so it learns only legal language.

B. Fine-tune on the 800 pairs first, then continue pretraining on the 2 GB to add vocabulary.

C. Continue pretraining the base model on the 2 GB corpus, then fine-tune on the 800 pairs.

D. Continue pretraining the base model on the 800 pairs, then fine-tune on the 2 GB corpus.

---

**Q83.** An internal IT agent answers employee questions using retrieved knowledge-base chunks. It runs on a model with a 16,384-token context window. Each request contains the system prompt, tool schemas, conversation history, the reranked chunks and the user question, and a fixed allowance is reserved for the answer. Every chunk is wrapped in a 30-token source header. Evaluation showed answer quality drops sharply when fewer than the top five reranked chunks are included, so the team decides to trim the oldest conversation turns instead.

| Component | Tokens |
|---|---|
| System prompt | 1,200 |
| Tool schemas | 2,300 |
| User question | 200 |
| Reserved for the answer | 1,500 |

| Reranked chunk | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Tokens (without header) | 1,100 | 900 | 1,600 | 700 | 1,000 | 400 |

What is the largest conversation history, in tokens, that still lets the top five chunks fit?

A. 5,884 tokens, after subtracting the fixed components and the five chunks' 5,300 tokens.

B. 7,234 tokens, since the answer reserve is generated later and is not part of the prompt.

C. 5,734 tokens, once the five chunks' 30-token headers are counted as well.

D. 5,304 tokens, after reserving room for all six reranked chunks with their headers.

---

**Q84.** A legal-search team compares two rankers on one query using nDCG@5 with graded relevance (3 = decisive, 2 = relevant, 1 = marginal, 0 = irrelevant) and the standard exponential gain. Six documents in the collection are judged: grades 3, 2, 2, 1, 0, 0.

```python
import numpy as np

def dcg(grades):
    return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(grades))

ideal = [3, 2, 2, 1, 0]
ranker_a = [3, 0, 2, 1, 0]
ranker_b = [2, 2, 3, 0, 1]

for name, r in [("A", ranker_a), ("B", ranker_b)]:
    print(name, round(dcg(r) / dcg(ideal), 3))
```

What does the script print, and which ranker does the team's metric favour?

A. A ≈ 0.825 and B ≈ 0.811, so A is favoured for ranking the decisive document first.

B. A ≈ 0.778 and B ≈ 0.904, so B is favoured for its relevant documents near the top.

C. A ≈ 0.6 and B ≈ 0.8, so B is favoured, as four of its top five are relevant.

D. A and B both score 1.0, since each retrieves every one of the judged-relevant documents.

---

**Q85.** An HR assistant answers multi-hop questions such as "Can an employee on probation carry over leave if they joined mid-quarter?", which need both the probation policy and the leave-carry-over policy. The team evaluates the retriever on 200 such questions, each with exactly two required documents, and passes the top 5 chunks to the generator.

| Metric (top 5) | Value |
|---|---|
| Hit rate (at least one required doc retrieved) | 0.94 |
| Recall@5 (share of required docs retrieved) | 0.61 |
| Questions with both required docs retrieved | 28% |
| Answer accuracy | 31% |

The retrieval lead reports: "Retrieval is fine at 94%; the generator is the bottleneck." Which assessment is correct?

A. Retrieval is fine, as hit rate is the standard metric and 0.94 beats usual targets.

B. The generator is at fault, since answer accuracy is far below retrieval's 94% hit rate.

C. Recall@5 of 0.61 shows that most questions retrieve both of their two required documents.

D. Hit rate hides the gap: both documents arrive only 28% of the time, near the 31% accuracy.

---

**Q86.** A manufacturer's RAG assistant answers questions from equipment manuals that contain many specification tables. The pipeline splits text into fixed 400-token chunks. Users report wrong answers to questions like "What is the maximum torque for model K-220?" On inspection, retrieved chunks often contain table rows such as "K-220 | 45 | 1,800 | IP54" with no column headers, because the header row landed in the previous chunk.

Which change most directly fixes this?

A. Increase the chunk size to 2,000 tokens, so that most tables fit within a single chunk.

B. Chunk by structure: keep each table whole with its header and attach the section title.

C. Switch to a larger embedding model, which can infer the column meaning of each row.

D. Add a 50-token overlap between chunks, so headers sometimes repeat at the chunk starts.

---

**Q87.** A spare-parts portal uses dense retrieval only. Analysis of failed searches by query type:

| Query type | Example | Share of queries | Recall@10 (dense) |
|---|---|---|---|
| Natural-language description | "seal that leaks on cold start" | 55% | 0.86 |
| Exact part number | "XR-4471-B gasket" | 30% | 0.22 |
| Error code | "E-0937 overheat" | 15% | 0.31 |

Which design change most directly addresses the weak query types?

A. Add keyword (BM25) retrieval alongside dense search and fuse the two rankings.

B. Fine-tune the dense model on natural-language queries, raising its recall well above 0.86.

C. Lowercase and stem all part numbers before embedding, so that codes become ordinary words.

D. Increase k from 10 to 100, so that the part-number documents eventually enter the results.

---

**Q88.** A banking FAQ assistant must answer within a 2-second median latency. Most questions are unique (fewer than 3% repeat within a day). Measured median stage latencies:

| Stage | Median latency |
|---|---|
| Query embedding | 40 ms |
| Vector search (top-100) | 30 ms |
| Cross-encoder reranking of all 100 candidates | 900 ms |
| LLM generation | 1,600 ms |
| **Total** | **2,570 ms** |

An offline study showed the correct passage is in the vector search's top 20 for 97% of questions. Which change meets the target while preserving answer quality?

A. Add a semantic cache of previous answers, so that repeated questions skip the whole pipeline.

B. Remove the reranker entirely, bringing the median to about 1,670 ms at no cost in quality.

C. Rerank only the top 20 candidates, bringing the median to about 1,850 ms.

D. Upgrade to a larger LLM, since better answers make up for the longer wait.

---

**Q89.** A RAG evaluation run on 500 insurance questions groups the failures into two clusters, scored with standard RAG metrics:

| Failure cluster | Questions | Context recall | Faithfulness | Answer correctness |
|---|---|---|---|---|
| Cluster X | 120 | 0.32 | 0.91 | 0.29 |
| Cluster Y | 90 | 0.88 | 0.41 | 0.36 |

Context recall measures whether the needed facts were retrieved. Faithfulness measures whether the answer's claims are supported by the retrieved context. Which diagnosis is correct?

A. Both clusters are generator failures, since answer correctness is low in each cluster.

B. Both clusters are retrieval failures, since better context would raise both correctness scores.

C. X is a generator failure and Y a retrieval failure, as faithfulness is high in X.

D. X is a retrieval failure; Y retrieves well, but the generator departs from the context.

---

**Q90.** An HR policy assistant indexes documents nightly. After the leave policy is revised, the assistant sometimes quotes the old carry-over limit and sometimes the new one.

```python
from uuid import uuid4

def sync_document(doc, index):
    chunks = split(doc.text, size=500, overlap=50)
    vectors = embed(chunks)
    ids = [str(uuid4()) for _ in chunks]                            # line 5
    meta = [{"doc_id": doc.id, "version": doc.version}] * len(chunks)
    index.upsert(ids=ids, vectors=vectors, metadata=meta)           # line 7

for doc in changed_documents_since(last_run):
    sync_document(doc, index)
```

What causes the mixed answers, and how should it be fixed?

A. Line 7, since upsert overwrites existing vectors and so leaves too few current chunks.

B. Line 5's random IDs never replace old chunks; delete the doc's old chunks before upserting.

C. The 50-token overlap duplicates text across chunks, so overlap should be set to zero.

D. The nightly schedule is too slow, so re-index after every individual document edit instead.

---

# ANSWER KEY

| Q | Ans | Domain | Q | Ans | Domain | Q | Ans | Domain |
|---|-----|--------|---|-----|--------|---|-----|--------|
| 1 | C | Stats | 31 | D | ML | 61 | A | CV |
| 2 | A | Stats | 32 | C | ML | 62 | C | CV |
| 3 | D | Stats | 33 | B | DL | 63 | A | CV |
| 4 | B | Stats | 34 | A | ML | 64 | D | CV |
| 5 | C | Stats | 35 | D | DL | 65 | C | CV |
| 6 | A | Stats | 36 | A | DL | 66 | B | LLM |
| 7 | D | Stats | 37 | C | DL | 67 | B | CV |
| 8 | B | Stats | 38 | A | DL | 68 | C | LLM |
| 9 | D | RAI Gov | 39 | D | DL | 69 | A | LLM |
| 10 | D | Stats | 40 | C | DL | 70 | D | LLM |
| 11 | B | RAI Gov | 41 | B | DL | 71 | B | LLM |
| 12 | A | RAI Gov | 42 | A | RAI Tech | 72 | A | LLM |
| 13 | C | RAI Gov | 43 | C | RAI Tech | 73 | D | LLM |
| 14 | B | RAI Gov | 44 | C | DL | 74 | B | Prompt ranking |
| 15 | D | RAI Gov | 45 | B | RAI Tech | 75 | D | Prompt ranking |
| 16 | A | RAI Gov | 46 | D | RAI Tech | 76 | A | Prompt ranking |
| 17 | C | PyTorch | 47 | C | RAI Tech | 77 | C | Prompt ranking |
| 18 | C | RAI Gov | 48 | A | RAI Tech | 78 | D | Prompt ranking |
| 19 | A | PyTorch | 49 | D | RAI Tech | 79 | A | Prompt ranking |
| 20 | D | PyTorch | 50 | C | NLP | 80 | C | Agents |
| 21 | B | PyTorch | 51 | B | RAI Tech | 81 | B | Agents |
| 22 | D | PyTorch | 52 | A | NLP | 82 | C | LLM |
| 23 | B | PyTorch | 53 | D | NLP | 83 | C | RAG |
| 24 | A | PyTorch | 54 | B | NLP | 84 | A | RAG |
| 25 | D | ML | 55 | A | NLP | 85 | D | RAG |
| 26 | B | ML | 56 | D | NLP | 86 | B | RAG |
| 27 | C | PyTorch | 57 | B | NLP | 87 | A | RAG |
| 28 | A | ML | 58 | D | CV | 88 | C | RAG |
| 29 | C | ML | 59 | B | CV | 89 | D | RAG |
| 30 | B | ML | 60 | C | NLP | 90 | B | RAG |

# EXPLANATIONS

**Q1 — C.** Pune: 2,000 defective, so TP = 1,800 and FP = 0.05 × 8,000 = 400; PPV = 1,800/2,200 ≈ 0.82. Chennai: 100 defective, so TP = 90 and FP = 0.05 × 9,900 = 495; PPV = 90/585 ≈ 0.15. Predictive values depend on prevalence as well as on sensitivity and specificity, so the first half of the claim fails. NPV is about 97.4% at Pune and 99.9% at Chennai, so passed cells are actually *safer* at Chennai. A ignores prevalence. B applies Pune's PPV to Chennai. D reverses the NPV comparison.

**Q2 — A.** P(failure) = 0.5 × 0.01 + 0.3 × 0.02 + 0.2 × 0.05 = 0.021. Posteriors: V1 = 0.005/0.021 ≈ 0.24, V2 = 0.006/0.021 ≈ 0.29, V3 = 0.010/0.021 ≈ 0.48. V3's much higher failure rate outweighs its smaller share. B reports V3's prior share rather than its posterior. C confuses P(failure | V3) with P(V3 | failure). D reports a correct posterior, but not the largest one.

**Q3 — D.** Cell means: Grid 32, 27, 21; List 37, 32, 26. List is slower by exactly 5 s on every device, so the interaction sum of squares is 0. SS_layout = 75 on 1 df; SS_error = 24 on 6 df, so MS_error = 4 and F_layout = 18.75 > 5.99, which is significant (p ≈ 0.005). The device effect is also significant (F ≈ 30.3). A contradicts the parallel cell means. B divides by the total mean square instead of the error mean square. C gives layout the device factor's degrees of freedom; layout has two levels, so 1 df.

**Q4 — B.** With two predictors, each one's auxiliary R² is r² = 0.9409, so VIF = 1/(1 − 0.9409) ≈ 16.9. High VIF inflates coefficient variance, so individual coefficients swing and flip, but the combined effect of the two is well estimated, and predictions on data like the training data barely change, which matches the stable hold-out error. Remedies are to drop or combine one of them, or to use ridge regression. A computes 1/r². C is wrong because collinearity mainly hurts interpretation, not prediction. D omits the squaring of the correlation.

**Q5 — C.** Covariance = 0.5 × 20 × 2 = 20, so the matrix is [[400, 20], [20, 4]] with eigenvalues ≈ 401.0 and 3.0; PC1 explains ≈ 99.3%, essentially the spend axis, because spend's variance is a hundred times larger. After standardizing, PCA runs on the correlation matrix [[1, 0.5], [0.5, 1]] with eigenvalues 1.5 and 0.5, so PC1 explains 75%. The raw result reflects units, not structure. A is wrong because the change of scale moves the eigenvalues even though the correlation stays the same. B is wrong because standard PCA uses covariance unless the data are standardized. D has the direction backwards.

**Q6 — A.** Model A: 1 − 0.38 × 39/36 ≈ 0.588. Model B: 1 − 0.35 × 39/31 ≈ 0.560. The penalty for five more predictors outweighs a 0.03 gain in R², and predictors like shirt colour are almost certainly noise. B quotes plain R². C computes Model B with p = 3. D states a true fact about plain R², but that is precisely why it cannot be used to compare models with different numbers of predictors.

**Q7 — D.** Differences (new − old) are 3, 3, 2, 4, 2, 4: mean 3.0, standard deviation ≈ 0.894, so t = 3.0/(0.894/√6) ≈ 8.2 on 5 df, p ≈ 0.0004. The unpaired test treats the between-picker spread (48 to 74) as noise and buries a consistent 3-pick gain. A misidentifies the issue, which is pairing, not variance equality. B reports Welch's t as if it were paired. C is wrong because a small sample makes a significant result *harder*, not less trustworthy, when the effect is this consistent.

**Q8 — B.** Baseline odds = 0.1/0.9 ≈ 0.111. Three tickets add 3 × 0.4 = 1.2 to the log-odds, so the odds multiply by e^1.2 ≈ 3.32, giving 0.369 and probability 0.369/1.369 ≈ 0.27. A applies the odds multiplier to the probability. C treats the model as linear in probability. D computes 3e^0.4 instead of e^(3 × 0.4).

**Q9 — D.** Wrong suggestions for older machines are visible only to the technicians who see them fail. Managing a model after deployment needs a continuous, structured channel through its users. Because technicians work offline, flagging must queue locally and sync later, and each report needs triage, an owner and a response back to the reporter so people keep using the channel. A captures operational health but cannot tell whether a repair suggestion was right. B is periodic, loses context, and treats retraining as the only response. C fragments the data, so no one can see patterns across regions.

**Q10 — D.** For a Poisson mean of 2: P(X ≥ 5) ≈ 0.053 (about 8.8 false pages a week), P(X ≥ 6) ≈ 0.0166 (about 2.8 a week), P(X ≥ 7) ≈ 0.0045 (about 0.76 a week). The lowest threshold under one per week is 7. A picks a conventional α without considering 168 hourly checks. B miscounts the weekly rate of the ≥6 threshold. C is stricter than needed (P ≈ 0.0011) and would delay detection of real incidents.

**Q11 — B.** The risk is real (confidential client data shared with a provider that may train on it), but the demand is also real. Governance provides a sanctioned path: an enterprise tool with contractual data protections (no training on inputs, retention controls), backed by a policy on what may be shared and training. A blanket ban tends to push usage underground. C does nothing about what the provider has already received. D misreads the terms: the provider's right to use the content is exactly the problem, and the consultancy remains responsible for its clients' confidentiality.

**Q12 — A.** Publicly visible is not the same as permission to use. The forum's terms forbid scraping, the posts are highly sensitive health content, and authors are plausibly identifiable even without usernames. A data review — legal basis, licensing, consent, sensitivity, de-identification feasibility and whether the use is appropriate at all — must come before training. B is wrong because detailed personal narratives can re-identify people even without usernames. C equates visibility with consent. D discloses a problem rather than resolving it.

**Q13 — C.** T1 equalizes approval rates but leaves qualified B applicants much less likely to be approved (0.61 vs 0.78). T2 equalizes true positive rates at a slightly higher default rate. These fairness criteria conflict, and choosing between them is a value and policy decision with legal, business and community implications. It belongs with an accountable governance forum (risk, compliance, legal, business owner, input from affected groups), with the rationale documented. A optimizes one business metric and ignores fairness. B treats one fairness definition as universally correct. D overstates the law; fairness requirements vary by jurisdiction and context.

**Q14 — B.** Harmful outputs are still being produced, so containment comes first: block the topic with a guardrail or take the bot offline. Then assess scope and impact, notify affected users and stakeholders as required, find the root cause, and document lessons learned. A leaves the harm running during root-cause analysis and retraining. C communicates before stopping the harm. D attempts concealment, which is ethically and often legally unacceptable and tends to escalate incidents.

**Q15 — D.** Customers should know when AI is materially involved in communications that affect them, especially claim denials, and they need a clear way to reach a human who can review their specific case. A misrepresents authorship. Agents who approve most drafts unchanged are not the authors in any meaningful sense. B puts the burden on customers to ask. C discloses AI involvement but removes the human recourse that matters most for contested decisions.

**Q16 — A.** Data minimization and storage limitation: keep data only as long as needed for a specific, legitimate purpose. Recordings from accidental triggers were never intended for the service, and indefinite retention for a speculative future use, with under 2% ever used, is disproportionate. A defined retention period, deletion of accidental captures, and purpose-limited retention (ideally with user controls) address the issue. B reduces breach risk but not the over-collection. C stretches a vague notice beyond what users would reasonably expect. D treats a privacy problem as a cost problem.

**Q17 — C.** The default `collate_fn` calls `torch.stack` on the samples' token tensors, which fails because they have different lengths (RuntimeError: stack expects each tensor to be equal size). That happens when the first batch is drawn, before anything else runs. The fix is a collate function that pads, for example with `pad_sequence`. The other bugs are real but do not stop the script. Stepping a StepLR scheduler per batch makes it decay every 5 batches instead of every 5 epochs, so the learning rate shrinks quickly but never raises an error (A). The model stays in eval mode after the first epoch, which silently disables dropout and changes batch-norm behaviour (B). Calling `zero_grad()` between `backward()` and `step()` wipes the gradients so nothing is learned, but it does not error (D). A further real bug, absent from the options, is that the loss receives softmax probabilities, so `CrossEntropyLoss` normalizes twice and gradients are weakened.

**Q18 — C.** Sound model risk management separates duties: developers build, an independent function validates, and a separate audit function periodically reviews the whole process. When builders validate their own model, errors and optimistic assumptions go unchallenged, and with no audit, nothing checks that the controls work. A and B describe resourcing preferences, not the control weakness. D confuses good outcomes so far with effective controls; independent validation exists precisely to catch problems before they surface in performance.

**Q19 — A.** `CrossEntropyLoss` averages over each 16-image batch. Calling `backward()` four times sums four batch-means, giving 4× the gradient of one batch-64 mean, so every step is effectively four times too large (verified: the accumulated gradient equals 4.0 × the batch-64 gradient). Dividing each loss by `ACCUM` restores equivalence, which is also why the run is stable at a quarter of the learning rate. B has the direction backwards. C is wrong because clipping after accumulation is the correct place for it, and here it only partly masks the oversized gradient. D is wrong because `zero_grad()` after each step is correct for accumulation.

**Q20 — D.** Adam keeps per-parameter running averages of the gradient and squared gradient. When they reset, the first steps after resuming use badly scaled updates. The fresh scheduler also restarts at step 0, re-running warmup and resetting the cosine position, so the learning rate jumps. Both cause the loss spike. A full checkpoint saves `optimizer.state_dict()` and `scheduler.state_dict()` (and often the grad-scaler and RNG states). A is less portable and still omits the optimizer state. B and C would change batch order slightly, but cannot explain a loss jump that takes two epochs to recover.

**Q21 — B.** Each step spends 410 ms decoding and augmenting images in the main process while the GPU waits, against 140 ms of GPU work, so the job is input-bound. Parallel workers (and pinned memory for faster host-to-GPU copies) let loading overlap GPU compute, so the step approaches the time of the slower stage. A and D speed up the 140 ms GPU part, which is not the bottleneck. C makes each step's loading cost grow in proportion, so the GPU still waits.

**Q22 — D.** After line 8, the tensor is (N, 224, 224, 3), channels last. `view` does not move any data; it reinterprets the same memory as (N, 3, 224, 224), so each "channel" is a scrambled mixture of rows and colour values. Because H = W = 224, the shapes match and nothing errors. The fix is `x.permute(0, 3, 1, 2)` (then `.contiguous()` if needed). A is wrong because scaling to [0, 1] before applying ImageNet mean and std is the standard practice. B is wrong because ImageNet statistics are a reasonable default for a pretrained model and would not drop accuracy to chance. C is wrong because sharing memory is harmless here.

**Q23 — B.** Two parameter groups with disjoint parameters, backbone at 1e-5 and head at 1e-3. A puts the head's parameters in both groups, which raises `ValueError: some parameters appear in more than one parameter group`. C freezes the backbone instead of training it gently. D swaps the two learning rates, which would damage the pretrained features.

**Q24 — A.** `nn.Module` registers only submodules assigned as attributes or held in containers like `nn.ModuleList`; a plain Python list is invisible to it. `parameters()` therefore returns only the head: 128 × 10 + 10 = 1,290. The optimizer never updates the three hidden layers, which stay at their random initialization, so the loss barely moves (they also would not move with `.to(device)` or appear in `state_dict()`). The fix is `nn.ModuleList`. B gives the count the model *should* have (3 × 16,512 + 1,290 = 50,826). C misdiagnoses weight sharing. D omits the head rather than the blocks.

**Q25 — D.** Each subscriber contributes up to 18 highly similar rows. Shuffled `KFold` puts some of a subscriber's months in training and others in test, so the model is effectively tested on customers it has already seen. Live scoring has no such overlap. Use `GroupKFold` on `subscriber_id`, or a forward time split (train on earlier snapshots, test on later ones), which also matches how the model is used. A is ruled out because the code builds features only from data up to each snapshot. B guesses at overfitting without addressing the evaluation flaw. C is wrong because stratification fixes class balance across folds, not the repetition of the same subscribers.

**Q26 — B.** At B = 100: 0.3 × 400 + 0.7 × 400/100 = 120 + 2.8 = 122.8. With 1,000 trees: 120 + 0.28 ≈ 120.3, because the correlation term ρσ² = 120 does not shrink as trees are added. With ρ = 0.15: 60 + 0.85 × 4 = 63.4. Decorrelating trees (smaller `max_features`, the core idea of random forests over bagging) attacks the dominant term. C forgets the ρσ² floor. D removes averaging entirely, so the variance is σ² = 400.

**Q27 — C.** `pos_weight` multiplies the loss on positive examples. Setting it to negatives ÷ positives = 49,500/500 = 99 makes the 500 fraud examples contribute as much in total as the 49,500 legitimate ones. A inverts the ratio, which makes fraud matter even less. B changes to a two-logit softmax loss with weights in the wrong order for the class indices (index 0 is legitimate). D sums instead of averaging, which scales every example equally and leaves the imbalance unchanged.

**Q28 — A.** Unshuffled `KFold` takes contiguous blocks of 1,000 rows. Folds 0–3 test only legitimate rows (0 fraud), and fold 4's test set holds all 100 fraud cases while its training set holds none, so that model never sees fraud. Metrics are meaningless or undefined on these folds. `StratifiedKFold(n_splits=5, shuffle=True)` gives each fold about 20 fraud cases. B describes what stratification would produce. C treats this as an imbalance problem, but fold 4's training set has no fraud at all, so there is nothing to oversample. D accepts the empty test folds as harmless, when in fact they make the metrics meaningless and leave fold 4 untrained on fraud.

**Q29 — C.** Net values: 0.60 → 2 × 600 − 3 × 200 = ₹600; 0.45 → 3 × 600 − 5 × 200 = ₹800; 0.35 → 5 × 600 − 7 × 200 = ₹1,600; 0.25 → 5 × 600 − 8 × 200 = ₹1,400. A voucher pays off when p × 0.4 × 1,500 > 200, that is, p > 0.33, so the best threshold sits near that break-even point. The default 0.5 ignores the cost structure. B is wrong: 0.25 adds a voucher for a non-churner and catches no extra churners. A and D fall short of the optimum.

**Q30 — B.** The root Gini is 0.5 (5 of 10 churned). `tickets ≤ 0`: the left node is 4 members with 0 tickets, all retained (Gini 0); the right node is 6 members with 5 churned (Gini ≈ 0.278); weighted impurity = 0.6 × 0.278 ≈ 0.167, the lowest. `tenure ≤ 24` and `tickets ≤ 2` both give ≈ 0.286, and `tenure ≤ 12` gives 0.32. A and D are the runner-up splits, and C is the third best.

**Q31 — D.** Inertia always decreases as k grows, so its minimum (A) tells you nothing. The useful signals are the elbow, where marginal gains flatten (after k = 4, drops fall from 84,000 to 23,000), and the silhouette, which peaks at 0.58 for k = 4. Both point to 4, which also fits the three-to-six campaign range. B misreads the largest drop as the place to stop. C sets the operational maximum as the target, even though the clusters are less well separated (0.44).

**Q32 — C.** Model A: MAE 2.4, RMSE ≈ 2.45. Model B: MAE 1.5, RMSE ≈ 3.83. B is usually closer but has one 12-minute miss, which would trigger a refund. RMSE squares errors and so penalizes large misses heavily, which matches the business cost; model A, which never misses by more than 3 minutes, is the better choice. A optimizes the wrong metric. B misstates the MAE comparison. D dismisses precisely the kind of error that costs money.

**Q33 — B.** The hidden state is the output gate applied to the squashed cell state: hₜ = oₜ ⊙ tanh(cₜ). The tanh bounds h to (−1, 1) while c can grow over time. A omits the squashing, so h is unbounded. C applies tanh after gating, which changes the function. D uses the forget gate, which governs memory retention, not output exposure.

**Q34 — A.** Training and validation accuracy have converged at about 0.75–0.76 and stay flat from 5,000 to 80,000 examples. That is high bias: the model cannot represent the pattern, so more data will not reach 0.85. Increase capacity (non-linear models such as gradient boosting, interactions) or add better features. B misreads a flat curve as rising. C is wrong because the train–validation gap is only 1–4 points, so this is not overfitting. D is false, since more data does not reduce bias.

**Q35 — D.** PyTorch's `nn.GRU` documents h′ = (1 − z) ⊙ n + z ⊙ h. Here z is the fraction of the old state *kept*. A is the mirror convention used in some papers. It is a valid GRU in its own right, but with PyTorch-trained weights it inverts the role of z and does not reproduce PyTorch's outputs. B uses the reset gate, which acts only inside the candidate n. C is not a convex interpolation and lets h grow without bound.

**Q36 — A.** In the concatenated form, every gate applies its own weight matrix to the same [hₜ₋₁; xₜ] vector, with a sigmoid for the gates (i, f, o) and tanh only for the candidate g. B uses tanh, which can make the forget gate negative and flip the memory's sign. C feeds the cell state into the gate, which is a "peephole" variant with a different weight shape that the given W["f"] cannot support. D ties forget to input (a coupled-gate variant) and discards the trained forget weights.

**Q37 — C.** With `batch_first=True`, `out` is (batch, seq, num_directions × hidden) = (16, 100, 256), concatenating the forward and backward states at every step. `h_n` is (num_layers × num_directions, batch, hidden) = (4, 16, 128) and is not affected by `batch_first`. A ignores bidirectionality. B concatenates directions in `h_n`, which PyTorch does not do. D applies the sequence-first layout to `out` despite `batch_first=True`.

**Q38 — A.** 128 → conv k7 s2 p3: 64 → max pool k3 s2 p1: 32 → conv k3 s2 p1: 16 → conv k3 p0: 14 → max pool 2: 7. Flatten: 128 × 7 × 7 = 6,272. B (128 × 8 × 8) gives the last convolution padding 1. C (128 × 14 × 14) forgets the final pool. D (128 × 15 × 15) treats the second convolution as stride 1.

**Q39 — D.** With 2 samples per GPU, per-batch BatchNorm statistics are extremely noisy, and the running averages used in eval mode do not match what the network saw in training. That is why eval-mode Dice collapses (0.62) while train-mode stays near 0.79. GroupNorm normalizes within each sample, independent of batch size; synchronized BatchNorm pools statistics across all GPUs, for an effective batch of 16. A makes the running estimates track noisy batches even more closely. B discards normalization's stabilizing effect. C adds noise without fixing the statistics.

**Q40 — C.** Unit-variance weights on wide layers make pre-activations large, so tanh saturates near ±1 (std ≈ 0.98). Its derivative there is near zero, so gradients vanish even though activations look "healthy". Small init (std 0.01) shrinks the signal geometrically to zero, so both activations and gradients vanish (A is wrong). Xavier scales weight variance by fan-in and fan-out to keep activations in tanh's responsive range. B mistakes saturation for stability. D is wrong because Xavier draws random weights; identical weights would leave units symmetric and unable to differentiate.

**Q41 — B.** Receptive field grows by (k − 1) × the cumulative stride: 1 → 3 → 5 (stride 1); the pool adds 1 → 6, and the cumulative stride becomes 2; each later 3 × 3 convolution adds 2 × 2 = 4 → 10 → 14. 14 px cannot cover a 30–40 px crack, so the team needs more downsampling, dilation or depth. A ignores the stride multiplier. C overcounts. D ignores the pooling layer's own contribution and the stride.

**Q42 — A.** Inputs are stable (PSI ≤ 0.05) while precision and recall fall steadily, and the business reports changed fraud behaviour: the relationship between inputs and the label has changed, which is concept drift. The remedy is retraining on recent labelled data (and possibly new features capturing the tactic), and monitoring performance with delayed labels as well as input drift. B describes covariate shift, which PSI rules out. C blames labels without evidence and ignores the reported change in tactics. D assumes a seasonal cycle that the data does not show.

**Q43 — C.** The softmax distributions overlap heavily: 44% of out-of-distribution photos exceed 0.9, and rejecting below 0.9 would also reject 39% of valid skin photos. Softmax confidence is not a reliable out-of-distribution signal. A dedicated check (a "is this a skin photo" gate, density or energy-based scores, ensembles) placed before classification, with abstention or a retake request, addresses the failure directly. A assumes confidence implies correctness. B is wrong because temperature scaling recalibrates confidence on in-distribution data and does not teach the model what it has never seen. D covers only the outliers it was trained on.

**Q44 — C.** For softmax followed by cross-entropy, ∂L/∂zᵢ = pᵢ − yᵢ: [0.659 − 1, 0.242, 0.099] = [−0.341, 0.242, 0.099]. A forgets to subtract the one-hot target. B flips the sign (the descent direction rather than the gradient). D subtracts the probabilities from 1 element-wise.

**Q45 — B.** LIME fits a local surrogate on randomly sampled perturbations, so different seeds or few samples can yield different top features for the same input, especially when features are correlated. Explanations used in adverse-action letters must be stable and faithful, so these runs cannot be used yet: increase the sample size, fix the seed, check stability, or use a method with consistency guarantees, such as SHAP for tree models. A misattributes the instability to the model. C would produce a misleading, inflated list of reasons. D is wrong because explanations depend on the explainer's procedure as well as the model.

**Q46 — D.** The curated test set over-represents the easy condition. In the field, 62% of photos fall in harder conditions with 63–81% accuracy. Reliability work means making those conditions measurable (a per-slice stress-test suite with realistic rotations, blur, low light and folds), training for them (augmentation and targeted field data), and monitoring per-slice accuracy. A adds more of what already works. B discloses the gap rather than fixing it. C raises reported accuracy by refusing most field photos, which defeats the product's purpose.

**Q47 — C.** n = 19 and α = 0.1, so ⌈20 × 0.9⌉ = 18; the 18th smallest error is 18, giving the interval 40 ± 18 minutes. This finite-sample rank is what gives conformal prediction its coverage guarantee of at least 90%, assuming the calibration and test data are exchangeable. A uses an interpolated percentile (≈ 14.8), which under-covers. B assumes normally distributed errors, which these skewed errors are not. D is needlessly wide.

**Q48 — A.** The requirement is for local explanations: each applicant's own reasons. Per-applicant attributions (SHAP values, or reason codes from a scorecard), with the largest contributors toward rejection translated into plain-language reasons, meet it. B gives global importance, which says what matters on average, not why *this* applicant was rejected; a feature that is globally important may have helped a particular applicant. C is neither required nor understandable to applicants and may expose proprietary details. D still provides no applicant-specific reasons.

**Q49 — D.** For a high-impact model, offline gains must be confirmed under live conditions with controlled exposure. Shadow mode runs the new model on real traffic without acting on its output, to compare predictions and catch failures. Then a small canary with predefined rollback criteria (forecast error, price anomalies, complaint rates) limits the impact of any problem before expanding. A risks a fleet-wide incident. B exposes all traffic to the new model on half the days and confounds the comparison with day-of-week effects. C maximizes the impact of any failure.

**Q50 — C.** The script prints A–B ≈ 0.807, A–C ≈ 0.893 and B–C ≈ 0.702. Reordering words breaks trigrams at the boundaries (and "256gb" vs "256 gb" changes several), while A and C differ by a single character, so character n-gram similarity rates the *different* model as the closest. A dedup pipeline should use similarity to propose candidate pairs but gate merges on structured attributes such as the model number and storage. A misreads the scores. B is wrong because the counts differ. D draws the wrong conclusion from the right number.

**Q51 — B.** The model has learned a spurious shortcut: the brand name stands in for negative sentiment, because training data from the recall period was skewed. Slice analysis exposed it, so the fix should break the correlation. Counterfactual augmentation (positive and neutral Zentra reviews, brand-swapped examples) or rebalancing, followed by re-testing that slice, does that. A hides the failure. C accepts a model that misreads current positive reviews. D shifts every prediction to repair one slice and degrades the rest.

**Q52 — A.** Doc 0 contains each query term once, among 9 words. Doc 2 contains "reset" twice, and "password" once, among 11 words. With TF-IDF and L2 normalization, doc 2 scores ≈ 0.508 against doc 0's ≈ 0.425: the repeated term inflates the router article's match. B ignores the arithmetic. C is wrong: sublinear tf narrows the gap (doc 2 ≈ 0.489) but does not reverse it, so fixing relevance needs more than reweighting (field boosts, phrase matching, or semantic retrieval). D ignores term frequency.

**Q53 — D.** With one "winner", the ham words dominate and P(spam) ≈ 0.004. Multinomial Naive Bayes multiplies in each occurrence as independent evidence, so four copies of "winner" add three more large likelihood ratios (31/86 under spam against 5/106 under ham, each time), and P(spam) jumps to ≈ 0.63. That is the independence assumption being exploited. Binary (Bernoulli-style) presence features, or capping counts, stop repetition from compounding. A ignores repetition. B mismatches the numbers, and raising α only shrinks the ratio slightly. C misstates the first value.

**Q54 — B.** Exact matches (span and type) are "Dolo 650", "Azithral" and "500 mg", so TP = 3. Precision = 3/6 = 0.5 and recall = 3/5 = 0.6, giving F1 ≈ 0.55. Because a dose without its unit or a frequency without "twice" cannot fill a prescription, strict entity-level matching measures exactly what matters. A counts partial overlaps as hits, which overstates usefulness. C swaps precision and recall. D's token-level scoring would reward exactly the partial spans that are unusable downstream.

**Q55 — A.** NER relies on surface cues: capitalization ("Pune", "Bluedart"), punctuation inside IDs ("SHP-2291/B"), function words around entities ("from Pune to Delhi"), and exact token forms. Lowercasing, stop-word and punctuation removal, and stemming are reasonable for some bag-of-words classifiers, but they destroy those signals for sequence labelling. B overgeneralizes vocabulary reduction. C still loses case and word forms. D deletes the characters that make shipment IDs identifiable.

**Q56 — D.** Document-level labels collapse mixed reviews into "neutral" and never say *what* was praised or criticised. Aspect-based sentiment extracts the components (battery, hinge, keyboard, fan, speakers) and assigns a polarity to each, so the dashboard can count complaints per component. A adds granularity to the overall score but still no aspects. B discards most of the actionable reviews. C aggregates further in the wrong direction.

**Q57 — B.** Inner products: Q1 = 4.0, Q2 = 2.0, Q3 = 5.0, so Q3 ranks first because of its large norm. Cosine: Q1 ≈ 0.998, Q2 ≈ 0.995, Q3 ≈ 0.693, so Q1 ranks first. Without normalization, inner product favours long vectors (often long, generic posts), which is the moderators' symptom. Normalize the embeddings or use a cosine index. A reverses which metric rewards magnitude. C is wrong because cosine ignores magnitude. D misreads cosine as length matching.

**Q58 — D.** The source corners are ordered TL, TR, BR, BL, but the destination is TL, TR, BL, BR. The last two points are swapped, so the homography maps the cheque onto a self-intersecting quadrilateral, which is the bow-tie smear (the cheque's centre maps far outside the output). The fix is `dst = [[0, 0], [W, 0], [W, H], [0, H]]`. A is wrong because OpenCV's `dsize` is (width, height), which gives an 1100 × 800 (rows × columns) result, as intended. B is wrong because a perspective transform needs exactly four point pairs. C is wrong because `getPerspectiveTransform` requires float32 points, which line 6 provides.

**Q59 — B.** Speckle comes from high-frequency belt texture, which a larger Gaussian kernel suppresses before gradients are computed. Gaps come from outline pixels whose gradient falls below the low threshold. With hysteresis, pixels between the thresholds are kept when connected to strong edges, so lowering the low threshold (while keeping the high threshold, so that seeds stay selective) reconnects the outline without admitting isolated texture. A removes weak outline pixels too, widening the gaps. C increases texture speckle. D promotes texture gradients to strong seeds, multiplying the speckle.

**Q60 — C.** Unigrams: the, battery, is, not, good, screen, bad = 7. Distinct bigrams: "the battery", "battery is", "is not", "not good", "is good", "the screen", "screen is", "not bad" = 8. Total 15. "not good" and "not bad" become their own features, which is how n-grams capture local negation, at the cost of vocabulary growth. A ignores the bigrams. B counts occurrences rather than distinct types. D contradicts the stem, which says stop words are kept.

**Q61 — A.** Proposed: (448/14)² = 32² = 1,024 patches + 1 class token = 1,025 tokens. Current: (224/16)² + 1 = 197. Each attention score matrix is tokens × tokens, so the ratio is (1,025/197)² ≈ 27. B forgets the class token and treats attention as scaling with pixels. C treats attention as linear in tokens. D uses the old 16-pixel patch size.

**Q62 — C.** Opening (erode then dilate) with a 3 × 3 kernel removes the single-pixel specks while restoring the tablets' shape. Closing (dilate then erode) then fills the 2 × 2 glare holes. Result: 2 objects and 0 hole pixels, the correct count. A would follow from closing alone (the specks remain separate components). B would follow from opening alone (the holes remain). D is the uncleaned mask.

**Q63 — A.** Shading scales all BGR channels down (here G 128 → 64, R 255 → 128), so the shaded pixel falls outside the fixed BGR box. Hue stays at 15 in both conditions, because it encodes colour independently of brightness. Thresholding on a hue band around 15, with reasonable saturation and a wide value range, tracks the totes in both lighting conditions. B accepts every pixel. C discards colour, the very cue being used. D thresholds on brightness, which is exactly what changes between the two conditions.

**Q64 — D.** Downscaling 4000 → 640 shrinks every dimension about 6×, so a 15–25 px crack becomes about 3–4 px, too few pixels for the detector's feature maps. Running inference on overlapping full-resolution tiles (sliced inference), or raising the input resolution, keeps cracks large enough to detect; overlapping detections are merged afterwards. A trains more on information that the resize has already destroyed. B loses localization. C would mainly add false positives, because the cracks are barely represented in the feature maps.

**Q65 — C.** 3840 × 2160 × 3 bytes × 30 fps ≈ 746 MB/s, about 6 Gbit/s, which is around 60 times the 100 Mbit/s uplink. The practical design moves inference to the edge (an on-device accelerator) and transmits only detections, counts or event crops, perhaps with compressed low-resolution video for audit. A confuses bits and bytes and still understates the volume. B invokes compression that the plan explicitly excludes. D is wrong because detection uses all three channels, and even one channel at this rate (≈ 250 MB/s) far exceeds the link.

**Q66 — B.** The KV cache scales with the number of K/V heads: 2 × 40 layers × kv_heads × 128 × 8,192 × 16 × 2 bytes gives 100 GiB (MHA), 20 GiB (GQA-8) and 2.5 GiB (MQA). MHA does not fit in 50 GiB. GQA-8 fits and loses only 0.3 points, while MQA saves memory the workload does not need and costs 2.8 points. Grouped-query attention is the usual middle ground for exactly this trade-off. A violates the concurrency requirement. C optimizes headroom over quality. D is wrong because the 100 GiB is already computed in fp16.

**Q67 — B.** Bilinear interpolation averages neighbouring values, so pixels between background (0) and vehicle (2) become 1, which here is the pedestrian class. The model learns a "pedestrian" rim around vehicles. Label masks must be resized with nearest-neighbour interpolation (as `up_nn` shows, `[0 2]`), while images can use linear or area interpolation. A misattributes the cause. C is wrong because nearest-neighbour never creates new values. D reverses the two behaviours.

**Q68 — C.** With left padding, pad tokens sit before the real tokens. A causal mask hides only future positions, so every real token can attend to the pad keys, and the number of pads depends on the longest ticket in the batch, which is why batch-mates change the output. Mask pad keys as well: combine `causal` with `(ids == PAD)[:, None, :]` before the softmax (making sure no row is masked entirely). A is wrong because the scale uses the head dimension, not the length. B is wrong because left-padding is standard for batched generation when the mask is right. D is wrong because the softmax must normalize over keys.

**Q69 — A.** At temperature 0.7 the probabilities are about [0.629, 0.308, 0.036, 0.018, 0.009]. Cumulative: 0.629, then 0.937 ≥ 0.9, so the nucleus is {A, B}. B is the result without temperature (at T = 1, the cumulative after B is only 0.857, so C is needed). C confuses top-p with greedy decoding. D overshoots the smallest set reaching 0.9.

**Q70 — D.** Weighted tokens per message: 0.5 × 78 + 0.3 × 240 + 0.2 × 330 = 39 + 72 + 66 = 177, about 2.3 times the 78-token English assumption. Subword tokenizers trained mostly on English split Indic scripts into many more tokens per word, so cost, latency and context usage all rise for those users. A states the false premise. B ignores the traffic shares. C assumes a translation step the scenario does not mention.

**Q71 — B.** `labels = input_ids` makes the model learn to predict the prompt tokens too: about 88% of each example (600 of 680 tokens) trains it to write user-style text, which is why it produces new "User:" turns, while the 80 response tokens get proportionally little weight. Setting `labels[:len(p_ids)] = -100` restricts the loss to the response. A and C misplace the problem: concatenation is how the response is conditioned on the prompt, and EOS belongs after the response. D is wrong because standard causal-LM losses shift the labels internally.

**Q72 — A.** Attention: 4 × (1024² + 1024) = 4,198,400. FFN: 1024 × 4096 + 4096 + 4096 × 1024 + 1024 = 8,393,728. LayerNorms: 2 × 2 × 1024 = 4,096. Total: 12,596,224. B is the familiar 12·d² approximation, which omits biases and norms. C (16·d²) and D (FFN counted twice) are plausible miscounts. The number of heads does not change the count, because heads split d_model rather than adding parameters.

**Q73 — D.** On the first step, x is the full prompt, so positions 0…T−1 are correct. Afterwards, x is a single token, and `torch.arange(1)` gives position 0 every time, so each new token is embedded as if it were at the start of the sequence while the cache holds later positions. The model loses track of order and degenerates. The position should be `past_length + arange(x.size(1))`. B is wrong because feeding only the newest token is how caching works: the earlier context lives in the cache. A and C misdiagnose: greedy decoding is deterministic but was fluent without the cache, and appending to `ids` does not touch the cache.

**Q74 — B.** Principle: the most elaborate reasoning prompt ranks last when the objective is latency and token cost. P3 constrains output to one word. P2 is short but leaves the output form open, so the model may write "The sentiment is positive." The decisive comparison is P5 versus P4: P5's persona adds about 15 input tokens per call, while P4 adds a full output sentence, and output tokens are both pricier and generated serially, so they dominate latency. P1's step-by-step reasoning is the most expensive of all and adds nothing when accuracy is already equal. A swaps P4 and P5. C ranks the open-ended P2 above the one-word constraint. D rewards chain-of-thought, which suits multi-step reasoning, not cost-sensitive classification.

**Q75 — D.** Principle: an explicit rule beats an illustrative example when the objective is semantic and has edge cases. P5 states the rule (bind every value) and covers the case parameters cannot handle, identifiers such as an ORDER BY column, with an allow-list. P3's example shows the safe pattern for a simple equality filter, which the model will generalize to many cases, but it says nothing about `LIKE`, ID lists or identifiers. The decisive comparison is P4 versus P2: P4 names the specific dangerous mechanism, while "make sure it is secure" is too vague to steer code generation. P1 gives no guidance. A ranks the example above the rule. B swaps P4 and P2. C ranks the bare prohibition above a concrete safe pattern.

**Q76 — A.** Principle: consistency on ambiguous input comes from an explicit tie-break rule. P1 resolves mixed tickets by a stated, repeatable criterion. P5's crude rule ("choose Technical") is perfectly repeatable, even if it is not always the best queue, and the objective is consistency. The decisive comparison is P4 versus P2: P4's examples anchor clean cases, so at least those are stable, while P2 gives no anchor at all. P3 ranks last because "use your judgement on each ticket's particular circumstances" explicitly invites case-by-case variation. B swaps P4 and P2. C ranks single-category examples above a tie-break rule. D rewards the persona.

**Q77 — C.** Principle: specifying the audience and register wins when the objective is understanding by a particular reader. P2 names the audience, the reading context and the three things they must take away. P4 forces plain language, though its jargon list cannot cover every technical term. The decisive comparison is P5 versus P1: P5 at least asks for simple language, while P1 sets no audience at all, so the model defaults to a technical explanation. P3 ranks last: a single sentence cannot carry what happened, whether it recurs and what to tell customers. A swaps P1 and P5. B puts the jargon list above the audience specification. D rewards brevity, which is precision aimed at the wrong target.

**Q78 — D.** Principle: the most detailed prompt carries a filter that suppresses exactly the recall being optimized. P4 explicitly includes stopped, held and in-passing mentions, which is where the dangerous misses occur. P2 broadens name variants but not status. The decisive comparison is P1 versus P3: "mentioned" covers all mentions, while "the patient's *current* medications" excludes held or stopped drugs, the very anticoagulant case. P5 is the most precise and ranks last, because "only medications whose dose and frequency are explicitly stated" drops every passing or held mention. A swaps P1 and P3. B and C reward P5's detail.

**Q79 — A.** Principle: an example beats any verbal description for an exact surface format. P3's examples show the decimal format, the quoting and the doubled internal quote, which P4's description omits ("in quotes" does not say how to escape a quote inside the name). P2 fixes the field order but not the date or amount format. The decisive comparison is P2 versus P5: P2 fixes the field order, while P5 asks only for "a CSV line" plus a persona. P1 does not even specify CSV. B swaps P2 and P5. C and D put the description above the examples.

**Q80 — C.** Today the average message takes 0.7 × 2 + 0.3 × 4 = 2.6 large-model calls, about 6.5 s and ₹1.04. Routing simple intents through the classifier to a direct tool call cuts those 70% to under half a second, keeps complex cases on the large-model loop (adding only 0.1 s), and brings the average to about 3.3 s and ₹0.48 per message. A cannot work: messages contain varying order IDs and wording, so exact-text cache hits are near zero, and caching final replies risks returning one customer's order status to another. B degrades the complex 30%. D does not address the simple majority at all.

**Q81 — B.** Sequential latency is 1.5 + (1.2 + 3.5 + 2.8 + 0.9) + 1.5 = 11.4 s. The four tool calls are independent, so running them concurrently (for example with `asyncio.gather`) makes that stage take as long as the slowest, 3.5 s, giving 1.5 + 3.5 + 1.5 = 6.5 s. D forgets the two LLM calls. A helps only repeated requests, and most requests are unique. C is a real but smaller gain (11.4 − 1.5 = 9.9 s).

**Q82 — C.** Domain-adaptive continued pretraining on the 2 GB corpus teaches legal vocabulary, phrasing and the firm's style at scale, without labels. Supervised fine-tuning on the 800 curated pairs then teaches the task format, answering questions in the desired way. A is wrong because 2 GB of text and a few days of compute are far too little to pretrain a capable 7B model from scratch. B reverses the order, so continued pretraining afterwards would wash out the instruction-following. D swaps the datasets: 800 pairs are far too few to pretrain on, and unlabelled text cannot teach the question-answer format.

**Q83 — C.** Fixed components: 1,200 + 2,300 + 200 + 1,500 = 5,200. The top five chunks with headers: (1,100 + 900 + 1,600 + 700 + 1,000) + 5 × 30 = 5,450. Maximum history = 16,384 − 5,200 − 5,450 = 5,734 tokens. A forgets the headers (5,884). B drops the answer reserve, which must fit in the same window as the prompt because output tokens share the context (7,234). D budgets for a sixth chunk the team does not need (5,304).

**Q84 — A.** With gain 2^g − 1: DCG(A) = 7/1 + 0 + 3/2 + 1/log₂5 + 0 ≈ 8.93; DCG(B) = 3 + 3/log₂3 + 7/2 + 0 + 1/log₂6 ≈ 8.78; IDCG ≈ 10.82. So nDCG(A) ≈ 0.825 and nDCG(B) ≈ 0.811. Exponential gain rewards putting the decisive (grade 3) document first. B is what linear gain (g instead of 2^g − 1) gives, which favours B. C is precision@5, a different metric. D confuses nDCG with recall.

**Q85 — D.** Hit rate counts a question as a success if *either* required document is retrieved, so for two-document questions it hides missing evidence. Only 28% of questions get both documents, and answer accuracy (31%) tracks that closely: the generator mostly succeeds when it has both, so retrieval is the bottleneck (query decomposition, multi-query retrieval, or larger k with reranking would help). A and B accept the misleading metric. C misreads 0.61: the average share of required documents retrieved is consistent with most questions getting only one of the two.

**Q86 — B.** The failure is structural: rows separated from their headers lose their meaning ("45" of what?). Structure-aware chunking keeps tables intact, repeats or attaches the header, and adds the section and model context, so each chunk is self-describing. A makes chunks bigger but still splits larger tables at arbitrary points, and dilutes retrieval. C cannot recover information that is not in the chunk. D repeats a header only when it happens to fall in the overlap window.

**Q87 — A.** Dense embeddings capture meaning well but handle exact identifiers poorly: a part number like "XR-4471-B" carries no semantics for the model, and near-identical codes embed close together. Lexical (BM25) retrieval matches exact tokens, and fusing the two rankings (for example with Reciprocal Rank Fusion) keeps the dense strengths on natural-language queries while recovering codes. B improves the already-strong query type. C mangles the codes further. D pushes the problem onto the generator, adds cost, and may still miss the codes.

**Q88 — C.** Cross-encoder cost scales with the number of candidates, so reranking 20 instead of 100 cuts that stage from about 900 ms to about 180 ms, giving 40 + 30 + 180 + 1,600 = 1,850 ms. Because the correct passage is in the top 20 for 97% of questions, quality is largely preserved. A barely helps, since fewer than 3% of questions repeat. B meets the target but discards the reranker's precision, so "no cost in quality" is false. D makes latency worse.

**Q89 — D.** Cluster X: low context recall (0.32) means the needed facts were not retrieved, while high faithfulness (0.91) shows the generator stayed true to the poor context it had, so this is a retrieval problem. Cluster Y: high context recall (0.88) means the facts were present, but low faithfulness (0.41) shows the answers made claims the context did not support, so this is a generator problem (prompting, grounding instructions, model choice). A and B ignore the component metrics. C swaps the two diagnoses.

**Q90 — B.** Each sync generates fresh random IDs, so `upsert` inserts new chunks without overwriting the previous version's. Old and new chunks coexist and compete in retrieval, producing mixed answers. Fix: delete existing chunks for the `doc_id` (or the old version) before inserting, or use deterministic IDs such as `f"{doc.id}:{i}"` and remove leftover higher-index chunks; a version filter at query time also helps. A misdescribes upsert, which would overwrite only if the IDs matched. C is wrong because overlap creates near-duplicates within one version, not stale content. D would not remove stale chunks either.

---
