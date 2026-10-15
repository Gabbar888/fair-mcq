# AI Assessment — Mock 05
## 90 questions | 150 minutes | MCQ, single correct

Answer every question; there is no negative marking. Each question has exactly one correct option. Questions move broadly through the eleven syllabus domains in order. Budget about 100 seconds per question and do not stall on any single item.

---

**Q1.**  A retail chain tests two store layouts (A and B) across two trading periods (morning and evening). Each period–layout combination is observed on five randomly chosen days, giving 20 days in a balanced 2 × 2 design. The response is average basket value in dollars. The within-cell mean square error from the full model is 12.0 on 16 degrees of freedom, and the critical value F(1, 16) at the 5% level is 4.49.

| Period | Layout A | Layout B |
|---|---|---|
| Morning | 40 | 46 |
| Evening | 44 | 38 |

A regional manager reads the column means, sees that both layouts average 42, and concludes that layout has no effect on basket value. Which statement is correct?

A. The interaction F is 15.0 and significant, so layout matters, but in opposite directions across periods.

B. The interaction F is 3.75, below 4.49, so the manager's reading stands.

C. The layout main-effect F is 15.0, which contradicts the equal column means of 42.

D. The manager is right, as the layout F is zero and the interaction not significant.

---

**Q2.**  A hospital screens 10,000 adults for a condition with 2% prevalence. Test 1 has sensitivity 0.90 and specificity 0.80. Test 2 has sensitivity 0.85 and specificity 0.95, and the two tests err independently given true status. The protocol is changed from "Test 1 alone" to "flag only if both Test 1 and Test 2 are positive."

A clinical lead says: "The new protocol sharply cuts false alarms, so the review queue becomes mostly genuine cases, but we will now miss more patients who truly have the condition." Which figures support the statement?

A. Combined sensitivity is about 0.99 and specificity 0.99, since both tests must agree.

B. Combined sensitivity falls to about 0.77, specificity rises to 0.99, and PPV rises to 0.61.

C. Combined sensitivity rises to about 0.99 and specificity falls to 0.76, raising predictive value.

D. Combined sensitivity is 0.85 and specificity 0.95, with the stronger test setting both.

---

**Q3.**  An epidemiologist runs a case-control study of a rare liver disease, recruiting 100 cases and 300 disease-free controls and recording whether each person had a particular occupational exposure. Population prevalence of the disease is known from a registry to be 1%.

| | Exposed | Unexposed |
|---|---|---|
| Cases | 60 | 40 |
| Controls | 90 | 210 |

A journalist reads the table and writes that "40% of exposed workers develop the disease." Which pair gives the exposure odds ratio and the probability of disease for an exposed person in the population?

A. Odds ratio 3.5, and probability given exposure 60%, the share of cases exposed.

B. Odds ratio 3.5, and probability of disease given exposure about 2.0%.

C. Odds ratio 3.5, and probability of disease given exposure 40%, read directly off the table.

D. Odds ratio 2.0, the ratio of exposure rates, and probability given exposure about 2.0%.

---

**Q4.**  A churn model predicts 90-day cancellation from four customer features. Before interpreting the coefficients, an analyst computes variance inflation factors by regressing each feature on the other three.

```python
from sklearn.linear_model import LinearRegression

feats = ["tenure_months", "account_age_days", "monthly_fee", "support_calls"]
for f in feats:
    others = [g for g in feats if g != f]
    r2 = LinearRegression().fit(df[others], df[f]).score(df[others], df[f])
    print(f, round(r2, 2), round(1 / (1 - r2), 2))
```

| Feature | Auxiliary R² |
|---|---|
| tenure_months | 0.95 |
| account_age_days | 0.95 |
| monthly_fee | 0.40 |
| support_calls | 0.10 |

Which interpretation of the output is correct?

A. The tenure pair has VIF 20, inflating each coefficient's standard error about 4.5 times.

B. A VIF of 20 means about 20% of tenure variance is shared, so the inflation is mild.

C. Both tenure features must be dropped, since collinearity of this size makes the model's churn predictions unreliable.

D. The VIF of 1.67 for monthly_fee signals severe collinearity, since any VIF above one inflates errors.

---

**Q5.**  A product team compares session length (minutes) between 40 users on a stable app build and 10 users on a beta build. The beta group is small and far more variable.

```python
from scipy import stats

r_pool  = stats.ttest_ind_from_stats(52.0, 4.0, 40, 47.0, 10.0, 10, equal_var=True)
r_welch = stats.ttest_ind_from_stats(52.0, 4.0, 40, 47.0, 10.0, 10, equal_var=False)
print(r_pool)
print(r_welch)
```

The analyst must report whether session length differs at the 5% level. Which statement is correct?

A. Pooled gives t ≈ 1.55 (p ≈ 0.15), while Welch gives t ≈ 2.51 (p ≈ 0.02).

B. Welch gives t ≈ 1.55 with 48 df, the same as the pooled test's df.

C. Both tests reject the null, since the 5-minute gap is large relative to either standard deviation.

D. Pooled gives t ≈ 2.51 (p ≈ 0.02), but Welch gives t ≈ 1.55 (p ≈ 0.15).

---

**Q6.**  A telecom provider tests whether 90-day churn is independent of plan tier. The retention team will fund plan-specific offers only if the association is significant at the 5% level, so the analyst reports the test statistic, its degrees of freedom and one expected count.

```python
import numpy as np
from scipy.stats import chi2_contingency

obs = np.array([[60, 240],     # basic:    churned, stayed
                [30, 270],     # standard: churned, stayed
                [10,  90]])    # premium:  churned, stayed
chi2, p, dof, expected = chi2_contingency(obs)
print(round(chi2, 2), dof, round(expected[0, 0], 2))
```

Which option matches the printed output?

A. 7.0, 2 and 42.86, with the statistic halved by the Yates continuity correction.

B. 14.0, 2 and 60.0, as the first expected count equals its observed value.

C. 14.0, 2 and 42.86, giving a p-value well below the 0.01 level.

D. 14.0, 5 and 42.86, with one degree of freedom per cell after the first.

---

**Q7.**  A data scientist reduces two correlated sensor readings to one component with a hand-rolled PCA.

```python
import numpy as np

X = np.array([[10, 20], [11, 22], [12, 21], [13, 23], [14, 24]], dtype=float)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
ratio = s**2 / (s**2).sum()
print(ratio.round(3), Vt[0].round(2))
```

A reviewer says the first component looks "too good to be true." Which statement is correct?

A. The printed ratio is about 0.95 from uncentred data, and centring raises it.

B. The printed ratio is about 0.999 since X is uncentred; centred PCA gives 0.95.

C. The printed ratio is about 0.95, since the SVD subtracts column means first.

D. The printed ratio is about 0.999, correctly showing nearly all variance on one direction.

---

**Q8.**  A subscription business models monthly customer status as a Markov chain over three states: active, at-risk and churned. Churned is absorbing. This month, 70% of customers are active and 30% are at-risk.

```python
import numpy as np

P = np.array([[0.80, 0.15, 0.05],    # from active
              [0.30, 0.60, 0.10],    # from at-risk
              [0.00, 0.00, 1.00]])   # from churned
v = np.array([0.7, 0.3, 0.0])
v2 = v @ P @ P
print(v2.round(3))
```

Which option gives the share of customers active and churned two months from now?

A. About 0.606 active and 0.269 churned, reading at-risk as churned.

B. About 0.650 active and 0.065 churned, after a single transition.

C. About 0.543 active and 0.000 churned, from right-multiplying.

D. About 0.606 active and 0.126 churned after two transitions.

---

**Q9.**  Objective: consistent date normalization for a UK accounts-payable team. Invoices arrive with dates in mixed forms, and the downstream ledger rejects any row whose date is wrong or ambiguous. Today's batch contains:

```text
14/03/2025    03/04/2025    7/11/24    2025-02-09    31/02/2025    12/12/25
```

Each prompt is sent with the batch appended.

- **P1.** "Convert each date to YYYY-MM-DD. Example: '14/03/2025' → '2025-03-14'."
- **P2.** "Convert each date to YYYY-MM-DD. Examples: '14/03/2025' → '2025-03-14' and '25/12/24' → '2024-12-25'."
- **P3.** "Convert each date to YYYY-MM-DD. Read slash dates as day/month/year even when both numbers are 12 or less, expand two-digit years to 20YY, and output INVALID for impossible dates rather than guessing."
- **P4.** "Convert each date to YYYY-MM-DD, following British date conventions."
- **P5.** "Convert each date to YYYY-MM-DD. Read slash dates as day/month/year. Example: '14/03/2025' → '2025-03-14'."

Which ranking is correct?

A. P3 > P4 > P5 > P1 > P2

B. P3 > P5 > P2 > P4 > P1

C. P5 > P3 > P4 > P2 > P1

D. P3 > P5 > P4 > P2 > P1

---

**Q10.**  A national survey estimates the churn rate of broadband customers. It samples 100 neighbourhoods at random and interviews 20 households in each, for 2,000 respondents in total. Churn decisions cluster within neighbourhoods because of shared local outages and competitor promotions, and the estimated intraclass correlation is 0.05.

Which option gives the design effect and the effective sample size?

A. Design effect 1.95, giving about 1,432 respondents after a square root.

B. Design effect about 1.05, from the correlation alone, and 1,905 respondents.

C. Design effect 1.95, giving about 1,026 effective respondents in the survey overall.

D. Design effect 20, one per household, and 100 effective respondents.

---

**Q11.**  A weather-comparison agent uses the Anthropic Messages API with tool use. For the prompt below, the model answers its first turn by emitting two `tool_use` blocks in a single response, one for each city. The script then raises a runtime error on the second API request.

```python
import anthropic

client = anthropic.Anthropic()
messages = [{"role": "user",
             "content": "Compare today's weather in Pune and Delhi."}]

while True:
    resp = client.messages.create(model=MODEL, max_tokens=1024,      # line 7
                                  tools=TOOLS, messages=messages)
    if resp.stop_reason != "tool_use":
        break
    messages.append({"role": "assistant", "content": resp.content})  # line 11
    call = next(b for b in resp.content if b.type == "tool_use")      # line 12
    out = run_tool(call.name, call.input)
    messages.append({"role": "user", "content": [                     # line 14
        {"type": "tool_result", "tool_use_id": call.id, "content": out}
    ]})

print(resp.content[0].text)
```

Which line is responsible for the error, and why?

A. Line 12, which answers only the first tool call, leaving the second without a result.

B. Line 11, since the SDK content objects must be converted to plain dictionaries first.

C. Line 14, since tool results belong in an assistant turn, not a user turn.

D. Line 7, since 1,024 tokens cannot hold two tool calls plus the final comparison.

---

**Q12.**  Objective: exact surface format. An observability team wants the model to emit log lines that a strict regex-based parser will accept. Spacing, bracket placement, key order, quoting only the `msg` value, and escaping of embedded quotes must all match exactly. A valid line looks like:

```text
[2025-06-01T14:03:22Z] WARN svc=checkout latency_ms=812 msg="upstream said \"retry\""
```

- **P1.** "Write each line as: an ISO-8601 UTC timestamp in square brackets, one space, the level in capitals, one space, then svc, latency_ms and msg as key=value pairs separated by single spaces, with only msg in double quotes and inner quotes backslash-escaped."
- **P2.** "Write each line exactly like this example: `[2025-06-01T14:03:22Z] WARN svc=checkout latency_ms=812 msg="slow upstream"`."
- **P3.** "Write the log lines in a clean, machine-readable form our parser will accept."
- **P4.** "Use this template: `[<timestamp>] <LEVEL> svc=<service> latency_ms=<int> msg="<text>"`."
- **P5.** "Write each line exactly like these examples: `[2025-06-01T14:03:22Z] WARN svc=checkout latency_ms=812 msg="slow upstream"` and `[2025-06-01T14:05:10Z] ERROR svc=payments latency_ms=2045 msg="gateway said \"declined\""`."

Which ranking is correct?

A. P5 > P4 > P2 > P1 > P3

B. P5 > P2 > P4 > P1 > P3

C. P5 > P2 > P1 > P4 > P3

D. P1 > P5 > P2 > P4 > P3

---

**Q13.**  Objective: correct reorder quantities from a warehouse assistant. The policy is: lead-time demand plus safety stock, minus stock on hand and on order, floored at zero, then rounded up to a whole case pack. Most past errors came from skipping the floor or rounding to the nearest case instead of up.

| SKU | Daily demand | Lead time (days) | Safety stock | On hand | On order | Case pack |
|---|---|---|---|---|---|---|
| K-114 | 38 | 6 | 50 | 140 | 60 | 24 |

- **P1.** "Compute the reorder quantity step by step: lead-time demand, add safety stock, subtract on-hand and on-order stock, floor at zero, then round up to the case pack. Check each step against the policy, then give the final quantity on its own line."
- **P2.** "Compute the reorder quantity. Give only the final number of units."
- **P3.** "Compute the reorder quantity from lead-time demand, safety stock, on-hand stock, on-order stock and the case pack. Give the final number."
- **P4.** "Compute the reorder quantity. Example: demand 40/day, lead time 5 days, safety stock 60, on hand 120, on order 50, pack 24 → 40 × 5 + 60 = 260; 260 − 170 = 90; round up to 96. If the net need is negative, order 0. Now do the SKU below."
- **P5.** "Compute the reorder quantity. Show the lead-time demand, the net requirement and the case rounding, then give the final quantity."

Which ranking is correct?

A. P1 > P4 > P5 > P3 > P2

B. P1 > P5 > P4 > P3 > P2

C. P1 > P4 > P3 > P5 > P2

D. P4 > P1 > P5 > P3 > P2

---

**Q14.**  A search team evaluates one query with graded relevance judgments (0 = irrelevant, 3 = perfect). The judged documents for the query are listed below, and the system returns the ranking d4, d1, d7, d2, d9.

| Document | d1 | d2 | d4 | d5 | d7 | d9 |
|---|---|---|---|---|---|---|
| Grade | 3 | 2 | 1 | 3 | 0 | 0 |

```python
import math

def dcg(rels, k):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))

ranked = [1, 3, 0, 2, 0]                     # grades in returned order
ideal  = sorted([3, 2, 1, 3, 0, 0], reverse=True)
print(round(dcg(ranked, 3) / dcg(ideal, 3), 2))
```

Which value is printed?

A. 0.42, normalizing by the ideal ordering of every judged document.

B. 0.49, since the gain uses each grade directly rather than exponentially.

C. 5.42, the discounted cumulative gain of the top three, before normalization.

D. 0.58, normalizing by the best ordering of the documents actually retrieved.

---

**Q15.**  Objective: maximum recall of personal data in application logs before they are shared with a vendor. A privacy engineer reviews every flagged value by hand, so false alarms cost little and misses are serious. A sample of the logs:

```text
2025-05-30 09:12:44 INFO  login ok user=asha.k@example.in ip=203.0.113.24
2025-05-30 09:12:45 DEBUG device=7f3c9e2a-41d1 acct=XXXX-4471 note="call back on 98450 12345"
2025-05-30 09:13:02 INFO  checkout cart=889 test_user=jdoe
```

- **P1.** "Find personal data in these logs."
- **P2.** "Find every value that could identify a person, directly or in combination: names, emails, phone numbers, IP addresses, device IDs, account numbers and free-text notes. Flag uncertain cases too, each with a short reason."
- **P3.** "You are a privacy engineer. Find every value that could identify a person, directly or in combination: names, emails, phone numbers, IP addresses, device IDs, account numbers and free-text notes. To keep review fast, list only values you are certain are real personal data, not test or placeholder values."
- **P4.** "Find personal data such as names, emails and phone numbers in these logs."
- **P5.** "Find every value in these logs that could identify a person, including IP addresses and device IDs."

Which ranking is correct?

A. P3 > P2 > P5 > P4 > P1

B. P2 > P5 > P4 > P1 > P3

C. P2 > P5 > P1 > P4 > P3

D. P2 > P3 > P5 > P4 > P1

---

**Q16.**  Objective: faithfulness in a policy-documents RAG assistant. Answers must contain nothing that the retrieved passages do not support, including on questions the passages cannot answer. The system prompt is inserted into this template:

```python
TEMPLATE = """{system_instruction}

<passages>
{retrieved_chunks}
</passages>

Question: {user_question}"""
```

- **P1.** "Don't make things up."
- **P2.** "Don't use outside knowledge. Don't guess. Don't speculate about anything not in the passages."
- **P3.** "Base your answer on the passages below."
- **P4.** "Answer using only the passages below. If they don't contain the answer, reply 'Not in the provided documents' and say what information is missing."
- **P5.** "Answer using only the passages below."

Which ranking is correct?

A. P4 > P5 > P3 > P2 > P1

B. P4 > P2 > P5 > P3 > P1

C. P4 > P5 > P2 > P3 > P1

D. P5 > P4 > P3 > P2 > P1

---

**Q17.**  A manufacturing group has deployed locally hosted language models across its finance, legal, HR and plant-maintenance teams, about 6,000 users in all. Each team uses a different fine-tuned variant for drafting, summarizing and answering policy questions. Six months in, the central AI office notices that complaints arrive through scattered emails, a few teams have quietly stopped using their assistant, and nobody can say whether answer quality has improved or degraded since launch. Two of the variants have been retrained once on new documents, and the legal team has started double-checking every draft by hand because of a few early errors that were never formally reported or investigated. The office asks how it should manage feedback on the deployed models from here on.

Which approach best fits responsible post-deployment governance?

A. Monitor latency, token usage and error rates on every server, since technical telemetry captures quality problems objectively.

B. Build a continuous, structured channel for user feedback, with triage, named owners and tracked responses feeding every model's updates.

C. Ask each model vendor to re-run its public benchmarks every quarter and publish the scores to all teams.

D. Run a detailed satisfaction survey of all 6,000 users once, then retrain each variant on the aggregated results.

---

**Q18.**  Objective: the right register for an error-explanation feature in a banking app aimed at first-time smartphone users over 65. Usability testers score each explanation on whether it feels warm, unhurried and free of technical vocabulary, as if a trusted helper were speaking. The error is "Session expired. Re-authenticate to continue."

- **P1.** "Explain this error in plain language, without jargon, in short, warm sentences."
- **P2.** "Explain this error for an older adult who is new to smartphones."
- **P3.** "You are Meera, a patient volunteer at a community centre who helps retirees with their first smartphones. You speak warmly and plainly, never use technical words, and gently check that your learner has understood. Explain this error to one of your learners."
- **P4.** "Explain this error for an older adult who is new to smartphones, in plain language, without jargon, in short, warm sentences."
- **P5.** "Explain this error simply."

Which ranking is correct?

A. P3 > P4 > P2 > P1 > P5

B. P3 > P4 > P1 > P2 > P5

C. P4 > P3 > P2 > P1 > P5

D. P3 > P2 > P4 > P5 > P1

---

**Q19.**  A lender audits its approval model across two applicant groups. "Qualified" means the applicant later repaid in full.

```python
import numpy as np

def rates(y_true, y_pred):
    approve = y_pred.mean()
    tpr = y_pred[y_true == 1].mean()
    fpr = y_pred[y_true == 0].mean()
    return round(approve, 2), round(tpr, 2), round(fpr, 3)
```

| Group | Qualified | Unqualified | Approved & qualified | Approved & unqualified |
|---|---|---|---|---|
| A | 400 | 600 | 320 | 180 |
| B | 200 | 800 | 120 | 380 |

The compliance lead writes: "Our approval rates are identical across groups, so the model is fair, yet qualified applicants in group B are approved noticeably less often." Which figures describe the situation?

A. Demographic parity difference is 0, while the equal-opportunity (TPR) difference is 0.20.

B. Demographic parity difference is 0.20, while the TPRs match, satisfying equal opportunity here.

C. The disparate-impact ratio is 0.75, failing the four-fifths rule, while TPRs are equal.

D. The equal-opportunity difference is 0.175, the gap between the two groups' false-positive rates.

---

**Q20.**  A pretrial risk tool outputs scores that are well calibrated within each group: among people scored 0.4, about 40% reoffend, whatever their group. The observed reoffence base rates differ between groups.

| Group | Base rate |
|---|---|
| X | 0.30 |
| Y | 0.60 |

An advocacy group demands that the tool also have equal false-positive and false-negative rates across the groups, without losing calibration. The tool is not a perfect predictor. Which statement is correct?

A. All three criteria can hold together if the decision threshold is tuned separately for each group.

B. With unequal base rates and an imperfect predictor, calibration and equal error rates cannot all hold together.

C. Retraining with the group attribute removed will satisfy all three criteria, since the unfairness comes from that feature.

D. Equal error rates follow automatically from calibration, so the demand is already satisfied by the current tool.

---

**Q21.**  A team ships an internal assistant and adds this logging step to the request handler.

```python
import json, datetime

def log_interaction(user_id, prompt, response):
    record = {
        "ts": datetime.datetime.utcnow().isoformat(),
        "user_id": user_id,
        "prompt": prompt,                 # raw text, may contain customer data
        "response": response,
    }
    with open("/data/llm_logs/all.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

# nightly job: fine-tune the assistant on every logged prompt/response pair
```

Employees often paste customer complaints, contracts and HR cases into the assistant. Users were told that logs are kept "for troubleshooting." Which governance issue is most serious?

A. JSON Lines files are slower to query than a database, which delays the troubleshooting team.

B. Logging the user ID lets administrators see who asked what, which no workplace system should record.

C. UTC timestamps are not timezone-aware, so the logs cannot support a reliable incident timeline.

D. Logged prompts with personal data are kept indefinitely and reused for training beyond the stated purpose.

---

**Q22.**  A recruiting-software vendor plans to release a model that ranks job applicants and automatically rejects the bottom 40% before any recruiter sees them. Risk-based governance frameworks, such as the EU AI Act, treat AI used in employment decisions as high-risk.

Which set of measures best matches what such a framework expects before release?

A. A documented risk assessment, bias testing, technical documentation, logging, and human oversight of rejections.

B. A disclaimer telling applicants that AI is used, since transparency alone discharges the high-risk obligations.

C. An accuracy benchmark above 90% on historical hires, since strong accuracy demonstrates the absence of bias.

D. An internal ethics statement signed by leadership, with no further process, as frameworks focus on intent.

---

**Q23.**  A team publishes this model card for an internal image classifier. Another business unit plans to reuse the model to triage injury photos from factory incident reports.

```yaml
model_name: defect-vision-v3
architecture: ResNet-50, fine-tuned
training_data: 48k product-surface photos, two plants, 2023
metrics:
  accuracy: 0.962
  f1_macro: 0.941
license: internal
contact: vision-team@corp.example
```

From a responsible-use standpoint, which addition matters most before that reuse goes ahead?

A. A model version history, so the other unit can roll back to an earlier checkpoint if needed.

B. The full hyperparameter list and random seeds, so the other unit can reproduce training exactly.

C. A larger headline test set, since 0.962 accuracy on more images would justify the new use.

D. Intended uses, out-of-scope uses and known limitations, which would flag injury triage as unsupported.

---

**Q24.**  A hiring team removes the `gender` column before training a CV-screening model, believing this makes the model gender-blind. An auditor trains a separate probe to predict gender from the remaining features.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = cv_features.drop(columns=["gender"])
probe = LogisticRegression(max_iter=2000)
auc = cross_val_score(probe, X, gender, cv=5, scoring="roc_auc").mean()
print(round(auc, 2))     # prints 0.91
```

| Feature group | Probe importance |
|---|---|
| Extracurricular keywords | 0.34 |
| Career-gap months | 0.29 |
| First-name embedding | 0.22 |
| Other | 0.15 |

Which conclusion is correct?

A. Dropping the three proxy feature groups guarantees that the screening model becomes fair.

B. The model is gender-blind, because the protected column is absent from its training inputs.

C. Gender remains recoverable from proxies, so outcomes still need testing for disparities by gender group.

D. The probe AUC of 0.91 proves that the screening model discriminates against women.

---

**Q25.**  A team trains an image classifier on a workstation with one CUDA GPU. The data pipeline returns CPU tensors, `Net` ends in a plain `nn.Linear` layer, and `evaluate` loops over the validation loader, computing accuracy from `model(xb).argmax(1)`. The script raises a runtime error during the very first training iteration.

```python
import torch, torch.nn as nn

device = "cuda"
model = Net().to(device)                                  # line 4
opt = torch.optim.Adam(model.parameters(), lr=1e-3)       # line 5
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    for xb, yb in train_loader:
        logits = model(xb)                                # line 10
        probs = torch.softmax(logits, dim=1)              # line 11
        loss = loss_fn(probs, yb)                         # line 12
        loss.backward()                                   # line 13
        opt.step()                                        # line 14
    print(epoch, evaluate(model, val_loader))
```

Which line raises the error, and what is the cause?

A. Line 5, since the optimizer must exist before the model moves to the GPU.

B. Line 12, since CrossEntropyLoss raises an error whenever it receives softmax probabilities.

C. Line 10, since the model sits on the GPU while the batch stays on the CPU.

D. Line 14, since stepping without first calling zero_grad raises on the accumulated gradients.

---

**Q26.**  A customer-support chatbot told a user to combine two cleaning products, which is dangerous. The user posted a screenshot publicly. The incident log so far:

| Time | Event |
|---|---|
| 09:05 | Screenshot reported to support team |
| 09:20 | Issue reproduced on current production model |
| 09:25 | Engineering proposes a prompt fix, testing estimated at 2 days |

Which next step best reflects responsible incident handling?

A. Wait for the tested prompt fix, since changing production without testing could cause new harms.

B. Delete the logs of the conversation, since retaining them prolongs the reputational exposure.

C. Issue a public statement that the output was a one-off, since reproduction confirmed it is rare.

D. Contain now by blocking or guarding the topic, notify stakeholders, then fix the root cause and document it.

---

**Q27.**  A developer is preparing features for a downstream classifier from a batch of 32 sequences, each with 4 time steps of 5-dimensional embeddings. A per-position bias of shape (4, 5) is added, the time and feature axes are swapped, and the result is flattened before a projection to 8 outputs. When the script runs, one line raises a runtime error.

```python
import torch

x = torch.randn(32, 4, 5)
y = torch.randn(4, 5)

a = x + y                         # line 6
b = a.transpose(1, 2)             # line 7
c = b.reshape(32, -1)             # line 8
d = b.view(32, -1)                # line 9
e = d @ torch.randn(20, 8)        # line 10
```

Which line raises the error?

A. Line 9, since `view` never copies data and the transposed tensor is non-contiguous in memory.

B. Line 6, since (32, 4, 5) and (4, 5) tensors cannot be broadcast together.

C. Line 8, since reshape cannot flatten a tensor whose dimensions were just transposed.

D. Line 10, since a (32, 20) tensor cannot multiply a (20, 8) weight matrix.

---

**Q28.**  A custom dataset serves product photos for a quality classifier.

```python
from torch.utils.data import Dataset, DataLoader

class PhotoSet(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = self.tf(load_rgb(self.paths[i]))   # tensor of shape (3, 64, 64)
        return img, self.labels[i]               # label is a Python int

loader = DataLoader(PhotoSet(paths, labels, tf), batch_size=32, shuffle=True)
```

| Property | Value |
|---|---|
| Photos | 1,003 |
| batch_size | 32 |
| drop_last | False (default) |

Which option gives the number of batches per epoch, the size of the last batch, and the shape of a full image batch?

A. 31 batches, since the incomplete last batch is dropped, with shape (32, 3, 64, 64).

B. 32 batches, the last padded to 32, with image batches of shape (32, 64, 64, 3).

C. 32 batches, the last holding 11 photos, with image batches of shape (3, 32, 64, 64).

D. 32 batches, the last holding 11 photos, with batches of shape (32, 3, 64, 64).

---

**Q29.**  A small CNN classifies 64 × 64 RGB images of circuit boards into 10 defect types. The developer left the classifier's input width as a placeholder and wants to fill it in so that the first forward pass does not fail with a shape mismatch.

```python
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5),              # no padding
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Linear(IN_FEATURES, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))
```

What must `IN_FEATURES` be?

A. 8,192, since both convolutions preserve the full spatial size before pooling.

B. 6,272, since each convolution trims two pixels from every single side.

C. 7,200, from a 15 × 15 map across 32 channels.

D. 115,200, since flattening keeps the first convolution's 60 × 60 map.

---

**Q30.**  A model is trained on two GPUs and saved, then loaded on a single-device inference server.

```python
import torch, torch.nn as nn

# training machine
model = nn.DataParallel(Net()).cuda()
# ... training loop ...
torch.save(model.state_dict(), "ckpt.pt")

# inference server
net = Net()
net.load_state_dict(torch.load("ckpt.pt", map_location="cpu"))
```

The last line raises a runtime error that lists missing and unexpected keys. Which explanation and fix are correct?

A. DataParallel prefixes every key with `module.`, so save `model.module.state_dict()` or strip the prefix.

B. CUDA tensors cannot load on a CPU machine, so the model must be retrained.

C. `torch.save` stored only optimizer state, so the weights were never written.

D. `map_location` deletes keys tied to GPU devices, so the argument should go.

---

**Q31.**  An engineer reviewing a fraud model's training code finds that the network's final layer already returns raw scores, but a colleague added an explicit sigmoid before the loss. To show the effect, she compares two ways of computing the binary loss for one example whose model output (logit) is 2.0 and whose true label is 1.

```python
import torch, torch.nn as nn

logit = torch.tensor([2.0])
target = torch.tensor([1.0])
crit = nn.BCEWithLogitsLoss()

loss_a = crit(logit, target)
loss_b = crit(torch.sigmoid(logit), target)
print(round(loss_a.item(), 3), round(loss_b.item(), 3))
```

Which option matches the printed values?

A. About 0.347 and 0.127, since the first call applies no sigmoid at all.

B. About 0.127 and 0.000, since a sigmoid output above 0.5 counts as correct.

C. About 0.127 and 0.347, the second call applying the sigmoid twice over.

D. Both are about 0.127, since the loss detects already-squashed inputs.

---

**Q32.**  A single scalar weight is trained with `torch.optim.SGD` and momentum. PyTorch keeps a buffer v, set to the gradient on the first step and updated as v ← μ·v + g afterwards, then applies w ← w − lr·v. The gradient happens to be the same at every step.

| Setting | Value |
|---|---|
| Initial weight | 1.0 |
| Gradient each step | 0.5 |
| Learning rate | 0.1 |
| Momentum μ | 0.9 |

What is the weight after three steps?

A. About 0.72, after three steps of the momentum update as specified.

B. 0.70, since the buffer adds the full previous velocity each step without decay.

C. 0.85, the same as plain SGD, since a constant gradient makes momentum irrelevant.

D. About 0.855, the weight reached once the momentum buffer has built up.

---

**Q33.**  A churn team trains a scikit-learn `RandomForestClassifier` with 200 trees on 64 features, using `max_features="sqrt"`. One feature, `days_since_last_login`, is far more predictive than any other. A colleague worries that the forest will become 200 near-identical trees all splitting on that feature at the root.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                            bootstrap=True, random_state=7)
rf.fit(X_train[features], y_train)          # len(features) == 64
roots = [t.tree_.feature[0] for t in rf.estimators_]
print(sum(features[r] == "days_since_last_login" for r in roots))
```

In expectation, in how many of the 200 trees is `days_since_last_login` among the candidate features at the root split?

A. About 175 trees, since the feature is excluded from seven of every eight candidate draws.

B. About 25 trees, with eight of the 64 features sampled as candidates at each split.

C. All 200 trees, since a forest always evaluates every feature at the root of each tree.

D. About 18.75 trees, since each split samples six candidates, the base-2 logarithm of 64.

---

**Q34.**  A team builds a digit classifier in Keras.

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input((28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```

Which option gives the trainable parameter count and the main problem with this setup?

A. 50,890 parameters, with the Flatten layer adding 784 weights that block gradients.

B. 50,816 parameters, excluding biases, which Dense layers do not train by default.

C. 50,240 parameters, with sparse labels requiring one-hot encoding before training can run.

D. 50,890 parameters, with softmax outputs passed to a loss expecting raw logits instead.

---

**Q35.**  A consultancy is advising two clients. The first is a startup with 2,000 labelled support tickets represented as 30,000-dimensional sparse TF-IDF vectors, which needs a fast, accurate classifier into 12 categories. The second is a bank that must predict next month's credit-card spending, in rupees, from six numeric customer features. A regulator requires that each feature's effect on the prediction be stated as a single signed number, so that a customer can be told, for example, how much each extra year of tenure changes the forecast. Neither client has a budget for heavy infrastructure, and both want something the in-house team can retrain monthly on a single laptop.

Which pair of models fits the two clients, in order?

A. k-means for the tickets, grouping them into 12 clusters, and linear regression for spending.

B. k-NN for the tickets, since sparse TF-IDF distances work well, and logistic regression for spending.

C. A linear SVM for the tickets and logistic regression, giving signed coefficients, for spending.

D. A linear SVM for the tickets and linear regression for the spending amount.

---

**Q36.**  A telecom retention team sends a discount offer to any customer whose predicted 30-day churn probability exceeds a threshold. Sending an offer to a customer who would have stayed costs 10 in wasted discount. Failing to contact a customer who then churns costs 90 in lost margin. The model is well calibrated, and an offer reliably retains a would-be churner.

The analyst sweeps thresholds on a held-out month to confirm the theory.

```python
import numpy as np

C_FP, C_FN = 10, 90
def expected_cost(p, y, t):
    contact = p >= t
    fp = ((contact) & (y == 0)).sum()
    fn = ((~contact) & (y == 1)).sum()
    return C_FP * fp + C_FN * fn

grid = np.arange(0.05, 0.96, 0.01)
best = grid[np.argmin([expected_cost(p_val, y_val, t) for t in grid])]
```

The analyst claims: "The default 0.5 threshold is far too strict for these costs." For a well-calibrated model, which threshold minimizes expected cost in theory?

A. 0.11, the ratio of the false-positive cost to the false-negative cost.

B. 0.90, the share of the total cost carried by a missed churner.

C. 0.50, since a calibrated model needs no adjustment to its default threshold.

D. 0.10, the false-positive cost's share of the combined cost of both errors.

---

**Q37.**  A team predicts 60-day churn for 50,000 customers, using `city` as a feature among others. Many cities have fewer than 20 customers.

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df["city_churn_rate"] = df.groupby("city")["churned"].transform("mean")
X = df[["tenure", "monthly_fee", "support_calls", "city_churn_rate"]]
y = df["churned"]
print(cross_val_score(GradientBoostingClassifier(), X, y, cv=5,
                      scoring="roc_auc").mean())
```

Which statement about the printed AUC is correct?

A. It is pessimistic, since averaging churn by city discards information that one-hot encoding of the city would have kept.

B. It is optimistic, since each customer's own label leaks into the city churn rate used to predict it.

C. It is unbiased, since the encoding uses only the target column and no other features of the test customers.

D. It is optimistic, since gradient boosting overfits whenever any numeric feature takes values between zero and one.

---

**Q38.**  Cost-complexity pruning has produced four candidate subtrees of a churn decision tree. Each is scored as R_α(T) = R(T) + α·|leaves|, where R(T) is the training misclassification rate.

| Subtree | Training error R(T) | Leaves |
|---|---|---|
| T1 | 0.20 | 1 |
| T2 | 0.12 | 4 |
| T3 | 0.08 | 9 |
| T4 | 0.05 | 20 |

Which subtree is selected at α = 0.01?

A. T4, since it has the lowest training error among all four subtrees.

B. T1, since any positive α makes the single-leaf tree optimal in this case.

C. T2, with a penalized score of about 0.16 at this α.

D. T3, with a score of about 0.125, the lowest of the four.

---

**Q39.**  An analyst runs k-means for k = 1 to 6 on standardized customer-behaviour features and records the inertia (within-cluster sum of squares).

| k | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Inertia | 1000 | 520 | 300 | 260 | 235 | 215 |

A colleague argues: "Inertia is still falling at k = 6, so more clusters are clearly better." Which statement is correct?

A. Choose k = 2, since the biggest single drop occurs between one and two.

B. Choose k = 6, since inertia rises beyond it, marking the true cluster count.

C. Inertia falls with every added k, but gains flatten after k = 3, the elbow.

D. The colleague is right, since lower inertia always means a better clustering.

---

**Q40.**  Two fraud models are compared on a stream where 1% of transactions are fraudulent. Model A has ROC-AUC 0.95, and at its chosen threshold TPR 0.80 and FPR 0.05. Model B has ROC-AUC 0.93, and at its threshold TPR 0.70 and FPR 0.01.

An analyst converts each operating point into the alert-queue figures the investigators will actually see.

```python
def queue(tpr, fpr, n=10_000, prevalence=0.01):
    pos = n * prevalence
    neg = n - pos
    tp, fp = tpr * pos, fpr * neg
    return round(tp), round(fp), round(tp / (tp + fp), 2)

print(queue(0.80, 0.05), queue(0.70, 0.01))
```

The risk lead says: "A wins on ROC-AUC, yet most of its alerts would be false, while B's alert queue is far cleaner." Which figures support that statement?

A. Precision is about 0.14 for A and 0.41 for B at these thresholds.

B. Precision is about 0.41 for A and 0.14 for B, lifted by ROC-AUC.

C. Precision is about 0.95 for A and 0.99 for B, the complements of FPR.

D. Precision is about 0.80 for A and 0.70 for B, matching each TPR.

---

**Q41.**  A student implements one time step of a standard LSTM cell from scratch for a sequence model that reads sensor windows. Here σ is the logistic sigmoid, ⊙ is element-wise multiplication, and [h, x] is concatenation. One line has been removed.

```text
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)          # forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)          # input gate
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)       # candidate cell values
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)          # output gate
______________________________________       # missing line
h_t = o_t ⊙ tanh(c_t)                        # new hidden state
```

Which line belongs in the gap?

A. `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, an additive update keeping a gated share of the old state.

B. `c_t = (1 − i_t) ⊙ c_{t-1} + i_t ⊙ g_t`, coupling forgetting to the input gate.

C. `c_t = f_t ⊙ g_t + i_t ⊙ c_{t-1}`, letting the forget gate scale the new candidate values.

D. `c_t = σ(f_t ⊙ c_{t-1} + i_t ⊙ g_t)`, squashing the cell state into the unit interval.

---

**Q42.**  A gradient-boosting churn model is trained with learning rate 0.1, and validation log-loss is recorded every 100 rounds using staged predictions.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

gb = GradientBoostingClassifier(n_estimators=600, learning_rate=0.1,
                                max_depth=3, random_state=0).fit(X_tr, y_tr)
for i, proba in enumerate(gb.staged_predict_proba(X_val), start=1):
    if i % 100 == 0:
        print(i, round(log_loss(y_val, proba), 3))
```

| Rounds | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|
| Validation log-loss | 0.420 | 0.360 | 0.340 | 0.335 | 0.338 | 0.350 |

The team will keep the round count with the lowest validation loss. It also plans a second run at learning rate 0.05. Which option is correct?

A. Keep 300 rounds, where the slope flattens, and the halved rate needs as many.

B. Keep 600 rounds, since boosting always improves with more rounds, needing 300 at 0.05.

C. Keep 400 rounds, and the halved rate likely needs about half the rounds.

D. Keep 400 rounds, and the halved rate will typically need roughly twice as many rounds.

---

**Q43.**  A team re-implements a GRU cell to match PyTorch's `nn.GRU` exactly, so that pretrained weights can be loaded. PyTorch's formulation applies the reset gate to the hidden-state term after its linear map. One line is missing.

```text
r_t = σ(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)            # reset gate
z_t = σ(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)            # update gate
________________________________________________          # missing line
h_t = (1 − z_t) ⊙ n_t + z_t ⊙ h_{t-1}                     # new hidden state
```

Which line belongs in the gap?

A. `n_t = σ(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))`, gating with a sigmoid.

B. `n_t = tanh(r_t ⊙ (W_in x_t + b_in) + W_hn h_{t-1} + b_hn)`, resetting the input term.

C. `n_t = tanh(W_in x_t + b_in + z_t ⊙ (W_hn h_{t-1} + b_hn))`, gating with the update gate.

D. `n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))`, with tanh and the reset gate.

---

**Q44.**  A reviewer checks a colleague's hand-written LSTM step. The gates and the cell update are correct, but the final line has been blanked out.

```text
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
________________________________________     # new hidden state
```

Which line completes the step?

A. `h_t = tanh(o_t ⊙ c_t)`, applying the squashing after the output gating.

B. `h_t = o_t ⊙ tanh(c_{t-1})`, using the previous cell state.

C. `h_t = o_t ⊙ tanh(c_t)`, gating a squashed version of the new cell state.

D. `h_t = o_t ⊙ c_t`, exposing the raw cell state through the output gate.

---

**Q45.**  A speech team builds a single-layer `nn.LSTM(input_size=128, hidden_size=256)` in PyTorch. PyTorch stores, for each of the four gates, an input weight matrix, a hidden weight matrix, and two separate bias vectors (`bias_ih` and `bias_hh`). The team needs the exact count to budget memory for a mobile deployment.

| Quantity | Value |
|---|---|
| Input size | 128 |
| Hidden size | 256 |
| Gates | 4 |
| Bias vectors per gate | 2 |

How many trainable parameters does the layer have?

A. 296,448 parameters, since an LSTM layer has three gates.

B. 98,816 parameters, counting one gate's weights and biases.

C. 394,240 parameters, using a single bias vector for each gate.

D. 395,264 parameters, with both bias vectors in every gate.

---

**Q46.**  A defect detector for a steel-rolling line takes 1 × 100 × 100 grayscale patches cropped from line-scan images and classifies each patch as defective or clean. The first prototype crashed on its first batch with a shape mismatch in the head, because the engineer had guessed the flattened size. The feature extractor and classifier head are:

```python
import torch.nn as nn

features = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3),                       # stride 1, no padding
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 40, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=3, stride=3),
    nn.Flatten(),
)
head = nn.Linear(IN_FEATURES, 2)
```

The engineer traces the spatial size after each layer to set `IN_FEATURES` before the first forward pass. What must it be?

A. 2,560, from an 8 × 8 map across all 40 output channels.

B. 23,040, since average pooling with stride 3 leaves the spatial size unchanged.

C. 2,890, since the stride-2 convolution rounds 98/2 up to 50 before pooling.

D. 3,240, since the padded 5 × 5 convolution shrinks the map.

---

**Q47.**  A team initializes a fully connected layer with 512 inputs and 256 outputs in a deep network that uses ReLU activations throughout. Activations explode or vanish with the framework default: by layer 30, activation variance has drifted by several orders of magnitude, and early layers barely train. They decide to switch to an initialization designed for ReLU, which zeroes roughly half of its inputs, and want the right scale for this layer's weights.

| Scheme | Formula |
|---|---|
| He (Kaiming) normal | std = √(2 / fan_in) |
| Xavier (Glorot) normal | std = √(2 / (fan_in + fan_out)) |
| Xavier (Glorot) uniform | limit = √(6 / (fan_in + fan_out)) |

Which option names the appropriate scheme and its value?

A. He normal, with standard deviation about 0.088, from summed fan-in and fan-out.

B. He normal, with standard deviation 0.0625 from the layer's fan-in of 512.

C. Xavier normal, with standard deviation about 0.051, as averaging fans suits ReLU.

D. Xavier uniform, with limit of about 0.088, since uniform draws avoid dead ReLUs.

---

**Q48.**  A `BatchNorm1d` layer with momentum 0.1 has running mean 2.0 and running variance 4.0 for one feature. A training batch of 8 samples has a mean of 3.0 for that feature and a biased (divide-by-n) variance of 1.0. PyTorch updates running statistics as running = (1 − momentum)·running + momentum·batch, using the unbiased batch variance for the running variance.

```python
import torch, torch.nn as nn

bn = nn.BatchNorm1d(1, momentum=0.1)
bn.running_mean.fill_(2.0); bn.running_var.fill_(4.0)
bn.train(); _ = bn(batch)          # batch: shape (8, 1), as described
print(bn.running_mean.item(), round(bn.running_var.item(), 2))
```

Which option matches the printed values?

A. 2.1 and 3.71, from the momentum update with the unbiased batch variance.

B. 2.9 and 1.43, weighting the batch statistics by 0.9 instead of 0.1.

C. 3.0 and 1.00, since training mode overwrites the running statistics.

D. 2.1 and 3.70, using the biased batch variance in the update.

---

**Q49.**  A regularized classifier contains `nn.Dropout(p=0.2)`. An engineer passes a tensor of ten ones through the layer in each mode.

```python
import torch, torch.nn as nn

drop = nn.Dropout(p=0.2)
x = torch.ones(10)

drop.train(); print(drop(x))
drop.eval();  print(drop(x))
```

Which option describes the two printed tensors?

A. In training mode each element is 0 or 1.0; in evaluation mode every element is 0.8.

B. In training mode each element is 0 or 1.25; in evaluation mode every element is 0.8.

C. In training each element is 0 or 1.25; in evaluation every element is 1.0.

D. In training mode each element is 0 or 0.8; in evaluation mode every element is 1.0.

---

**Q50.**  A dermatology triage model must return a set of candidate diagnoses that contains the true class at least 90% of the time. The team applies split conformal prediction. The nonconformity score is 1 − p̂(true class), computed on 19 held-out calibration images, and q̂ is the ⌈(n + 1)(1 − α)⌉-th smallest calibration score with α = 0.1.

```python
import numpy as np, math

cal = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25,
                0.28, 0.31, 0.35, 0.40, 0.46, 0.52, 0.60, 0.71, 0.85])
k = math.ceil((len(cal) + 1) * 0.9)
qhat = np.sort(cal)[k - 1]

p_test = np.array([0.55, 0.30, 0.10, 0.05])           # four classes
pred_set = [c for c, p in enumerate(p_test) if 1 - p <= qhat]
```

Which option gives q̂ and the prediction set for the test image?

A. q̂ is about 0.62 from the plain 90th percentile, and the set is {0} alone.

B. q̂ is 0.71, and the set is {0, 1}, covering both of the two most probable classes.

C. q̂ is 0.71, and the set is {0}, as only the top class exceeds 0.5.

D. q̂ is 0.85, the largest calibration score, and the set is {0, 1, 2}.

---

**Q51.**  A single sigmoid neuron has weight w = 0.8, bias b = −0.2 and input x = 1.5, with true label y = 1. An engineer compares the gradient ∂L/∂w under two losses: binary cross-entropy, and squared error L = ½(ŷ − y)².

| Quantity | Value |
|---|---|
| x | 1.5 |
| w | 0.8 |
| b | −0.2 |
| y | 1 |

Which option gives the two gradients?

A. About −0.403 under cross-entropy and −0.403 under the squared error.

B. About −0.079 under cross-entropy and −0.403 under the squared error.

C. About −0.403 under cross-entropy and −0.079 under squared error.

D. About −0.269 under both, since each uses the error ŷ − y.

---

**Q52.**  A pricing model's prediction for one customer is explained with exact Shapley values over two features, `tenure` (a) and `plan` (b). The value function v(S) is the model's expected prediction when only the features in S are known.

```python
v = {
    frozenset():           10,
    frozenset({"a"}):      14,
    frozenset({"b"}):      12,
    frozenset({"a", "b"}): 20,
}
```

The two features interact: knowing both adds more than the sum of their solo effects. Which option gives the Shapley values of a and b?

A. 6 and 4, with the interaction shared equally and the total equal to v(a, b) − v(∅).

B. 4 and 2, each feature's gain when it is added first to the empty set.

C. 5 and 5, since an interaction term is always split evenly between the two features.

D. 8 and 6, each feature's gain when it is added last, after the other.

---

**Q53.**  A team evaluates two image classifiers for a roadside-inspection camera on a clean test set and on a corrupted copy containing fog, motion blur and JPEG artefacts at deployment-like severities.

| Model | Clean accuracy | Corrupted accuracy |
|---|---|---|
| A | 0.92 | 0.70 |
| B | 0.95 | 0.68 |

The product lead says: "B looks better on the benchmark, but A degrades more gracefully under the conditions we will actually face." Which figures support that statement?

A. A loses 22 points and keeps about 76% of its accuracy, while B loses 27 points and keeps about 72%.

B. A loses 27 points and keeps about 72% of its accuracy, while B loses 22 points and keeps about 76%.

C. Both lose about the same share, so B's higher clean accuracy makes it the more robust model.

D. A keeps about 92% of its accuracy, since corrupted accuracy is measured relative to clean accuracy.

---

**Q54.**  An auditor runs a model-parameter randomization test on a saliency method for a chest-X-ray classifier. Layers are re-initialized from the top down, and the Spearman rank correlation between the original saliency maps and the maps from the randomized model is recorded.

```python
from scipy.stats import spearmanr

for k in [0, 1, 3, 6, 12]:                   # number of top layers randomized
    m = randomize_top_layers(model, k)
    rho = mean(spearmanr(saliency(model, x).ravel(),
                         saliency(m, x).ravel())[0] for x in audit_set)
    print(k, round(rho, 2))
```

| Layers randomized | 0 | 1 | 3 | 6 | 12 (all) |
|---|---|---|---|---|---|
| Rank correlation | 1.00 | 0.97 | 0.95 | 0.94 | 0.93 |

Which conclusion is correct?

A. The test is inconclusive, since rank correlations above 0.9 are expected for any faithful method.

B. The method is highly robust, since its maps stay stable even when the weights change substantially.

C. The model is overfitting, since its explanations barely change after the layers are re-initialized.

D. The method fails the check, since its maps barely depend on the learned weights.

---

**Q55.**  Two attribution methods are compared on a held-out set with a deletion test. Pixels are removed in order of attributed importance, and the model's accuracy is tracked. A random-order baseline is included.

| Pixels removed | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| E1 order | 0.90 | 0.55 | 0.30 | 0.15 | 0.10 |
| E2 order | 0.90 | 0.80 | 0.65 | 0.45 | 0.30 |
| Random order | 0.90 | 0.85 | 0.78 | 0.70 | 0.62 |

Clinicians in a user study preferred E2's maps, describing them as cleaner and more anatomically plausible. Which conclusion about faithfulness is correct?

A. E2 is more faithful, since a slower decline shows its maps preserve the model's accuracy.

B. E1 is more faithful, since removing its top pixels destroys accuracy fastest.

C. E2 is more faithful, since clinician preference is the accepted ground truth for attribution quality.

D. Neither is faithful, since both curves start at the same accuracy as the random baseline.

---

**Q56.**  A claims-approval model abstains whenever its confidence falls below a threshold, passing those cases to a human. On ten validation cases:

```python
import numpy as np

conf    = np.array([0.99, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80, 0.72, 0.65, 0.55])
correct = np.array([1,    1,    1,    1,    0,    1,    1,    0,    1,    0])

for t in (0.8, 0.9):
    keep = conf >= t
    print(t, keep.mean(), round(1 - correct[keep].mean(), 3))
```

Which option matches the printed coverage and risk at the two thresholds?

A. At 0.8, coverage 0.7 and risk 0.3; at 0.9, coverage 0.5 and risk about 0.143 overall.

B. At 0.8, coverage 0.7 and risk about 0.143; at 0.9, coverage 0.5 and risk 0.2.

C. At 0.8, coverage 0.7 and risk about 0.143; at 0.9, coverage 0.5 and risk falling to 0.0.

D. At 0.8, coverage 0.3 and risk about 0.143; at 0.9, coverage 0.5 and risk 0.2 overall.

---

**Q57.**  A product team wants to show users why a transformer-based sentiment model labelled their review negative. An engineer proposes highlighting the tokens with the largest last-layer attention weights from the classification token, presenting them as the reasons for the decision.

Which assessment is most accurate?

A. Attention weights become faithful explanations once they are averaged across all the heads.

B. Attention weights are faithful explanations by construction, since the model literally uses them.

C. Attention weights should never be shown to users, because they contain no information about the model.

D. Attention weights alone are not a reliable explanation, so faithfulness should be validated first.

---

**Q58.**  An e-commerce catalogue team wants to catch duplicate listings that differ by typos or near-homophones. They compare product titles by the Jaccard similarity of their character trigrams. Titles are lowercased, the space counts as a character, and no padding is added.

```python
def trigrams(s):
    s = s.lower()
    return {s[i:i + 3] for i in range(len(s) - 2)}

def jaccard(a, b):
    A, B = trigrams(a), trigrams(b)
    return len(A & B) / len(A | B)

titles = {"T1": "wireless mouse", "T2": "wirless mouse", "T3": "wireless house"}
```

Which pair is most similar, and what is its score?

A. T1 and T3 at about 0.64, since full-word matches dominate the trigram overlap.

B. T2 and T3 at about 0.35, as each differs from T1 once.

C. T1 and T2 at about 0.64, despite the letter missing from the second title here.

D. T1 and T3 at about 0.60, since only one letter differs between their titles.

---

**Q59.**  A credit-default model was trained in 2024. After a regulation change in March 2025 that altered how lenders report late payments, its monitoring dashboard shows:

| Metric | Jan 2025 | Jun 2025 |
|---|---|---|
| Feature PSI, all inputs (vs training) | 0.03 | 0.04 |
| Predicted default rate | 4.1% | 4.0% |
| Observed default rate | 4.2% | 6.8% |
| AUC on labelled outcomes | 0.86 | 0.71 |

Which diagnosis and response are most appropriate?

A. No drift, since the PSI values for every input stay well below the 0.1 alerting threshold.

B. Covariate shift in the inputs, best handled by reweighting training samples to match the new feature distribution.

C. Concept drift, where P(y | x) has changed, requiring recent labelled data and retraining or recalibration.

D. Label noise from late reporting, so the June outcomes should be discarded until they stabilize.

---

**Q60.**  A logistics company's support-ticket search engine must surface the most relevant past tickets when an agent types a short query, so that agents can reuse earlier resolutions instead of starting from scratch. It scores documents for the query "refund delayed courier" by summing tf × idf over the query terms. Here tf is the raw term count in the document, idf = log₁₀(N / df), and the collection has N = 1,000 tickets.

| Term | Document frequency (df) |
|---|---|
| refund | 50 |
| delayed | 200 |
| courier | 10 |

| Ticket | Term counts |
|---|---|
| D1 | refund × 3, delayed × 1 |
| D2 | refund × 1, courier × 2 |
| D3 | delayed × 4, courier × 1 |

Which option gives the ranking from highest to lowest score?

A. D2, D1, D3, with D2 scoring about 5.30 and D3 scoring the lowest.

B. D3, D1, D2, since D3 contains the most occurrences of query terms in total.

C. D1, D3, D2, since three mentions of "refund" outweigh the rarer courier term.

D. D2, D3, D1, with D2 scoring about 5.30 and D1 scoring the lowest here.

---

**Q61.**  A Naive Bayes spam filter assigns the message "claim prize" these word likelihoods, with independence assumed:

```python
lik_spam = {"claim": 0.05, "prize": 0.02}
lik_ham  = {"claim": 0.01, "prize": 0.01}

def posterior_spam(prior_spam):
    s = prior_spam * lik_spam["claim"] * lik_spam["prize"]
    h = (1 - prior_spam) * lik_ham["claim"] * lik_ham["prize"]
    return s / (s + h)
```

During development the team assumed a 50% spam prior. In production, only 10% of incoming mail is spam. Which option gives the posterior probability of spam under each prior?

A. 0.53 with the equal prior and 0.91 with the production prior of 10%.

B. 0.91 with the equal prior and 0.09 with the production prior, scaled by the prior.

C. 0.91 with the equal prior and about 0.53 with the production prior of 10%.

D. 0.91 under both priors, since a likelihood ratio of 10 dominates any choice of prior.

---

**Q62.**  An NER model is scored on one 20-token sentence with BIO tags. Gold entities are PER at tokens [0, 2), ORG at tokens [5, 8) and LOC at token [15, 16). The model predicts PER [0, 2) correctly and ORG [5, 7), stopping one token early. It tags token 15 as O. Every other token is correctly tagged O.

| Metric | Definition |
|---|---|
| Token accuracy | correctly tagged tokens / all tokens |
| Entity F1 | exact span and type matches, at entity level |

A reviewer says: "Token accuracy looks excellent, yet most of the entities are wrong." Which figures support that statement?

A. Token accuracy 0.90 and entity F1 0.67, with partial ORG counting as a match.

B. Token accuracy 0.90 and entity-level F1 0.40, with only the PER entity matching exactly.

C. Token accuracy 0.40 and entity F1 0.90, as tokens are scored more strictly.

D. Token accuracy 0.85 and entity F1 0.40, counting the O tokens as errors.

---

**Q63.**  A quick sentiment baseline represents each review as the average of its word vectors, then compares it with reference vectors by cosine similarity. Toy two-dimensional vectors are used for illustration.

```python
import numpy as np

E = {"good": np.array([1.0, 2.0]), "not": np.array([-0.2, 0.1]),
     "bad":  np.array([-1.0, -1.8])}
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

review = (E["not"] + E["good"]) / 2          # "not good"
print(round(cos(review, E["good"]), 3), round(cos(review, E["bad"]), 3))
```

Which option matches the output and its implication?

A. About −0.995 and 0.99, since "not" reverses the direction of "good".

B. About 0.10 and 0.99, showing that averaging captures negation through the "not" vector.

C. About 0.50 and −0.50, since averaging two words halves every similarity.

D. About 0.995 and −0.99, so the averaged review is treated as strongly positive.

---

**Q64.**  A multilingual assistant has an 8,192-token context window. Its tokenizer's average fertility (tokens per word), measured on news text, is:

| Language | Tokens per word |
|---|---|
| English | 1.3 |
| Hindi | 2.9 |
| Tamil | 4.1 |

A product manager asks what this means for a 1,000-word customer document in Hindi compared with the same document in English. Which option is correct?

A. Token counts match across languages, since tokenizers normalize every script to the same units first.

B. Hindi needs about 2.2 times as many tokens, and about 6,301 Hindi words fit the window.

C. Hindi needs about 2.2 times as many tokens, and about 2,824 Hindi words fit.

D. Hindi needs about 1.6 more tokens per word, so cost rises by roughly 60% over English.

---

**Q65.**  A sentiment classifier trained on 200,000 smartphone reviews reaches 93% accuracy on held-out phone reviews. Deployed on car-rental reviews, it falls to 71%. Error analysis shows that words such as "unpredictable", "cheap" and "small" carry positive or neutral weight in the training data but signal complaints in car-rental reviews.

Which diagnosis and remedy are most appropriate?

A. Domain shift has changed word polarity, so adapt the model with newly labelled car-rental reviews.

B. The model is underfitting, so more epochs on the smartphone data will restore accuracy.

C. Stop-word removal deleted these words, so restoring them to the vocabulary fixes it.

D. The test set is mislabelled, since a 93% model cannot fall to 71%.

---

**Q66.**  A retail-analytics team fine-tunes a ViT with 224 × 224 RGB inputs, non-overlapping 14 × 14 patches, one class token, and 16 attention heads in every encoder layer. They want to estimate memory for the attention-score matrices, the QKᵀ products before softmax.

Which option gives the sequence length and the number of attention scores computed in one encoder layer, across all heads, for one image?

A. 256 tokens and 1,048,576 scores, as the class token attends to nothing.

B. 257 tokens and 1,056,784 scores across all sixteen heads.

C. 257 tokens and 66,049 scores, since the heads share a single score matrix.

D. 197 tokens and 620,944 scores, using the standard patch size of 16 pixels.

---

**Q67.**  A bigram language model assigns these conditional probabilities to the four predicted tokens of a held-out sentence (the start token is given):

```python
import numpy as np

p = np.array([0.2, 0.5, 0.1, 0.4])            # P(w_i | w_{i-1}) for i = 1..4
ppl = np.exp(-np.mean(np.log(p)))
print(round(ppl, 2))
```

Which value is printed?

A. 0.25, the geometric mean probability assigned to each of the predicted tokens.

B. 3.33, the reciprocal of the mean of the four token probabilities.

C. 3.98, the inverse geometric mean of the four token probabilities.

D. 250.0, the reciprocal of the product of the four token probabilities.

---

**Q68.**  A PCB-inspection line detects copper-trace edges with:

```python
import cv2

gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (5, 5), 1.2)
edges = cv2.Canny(blur, 100, 200)
```

Engineers report that faint sections of long traces show up as broken dashes. One engineer changes the call to `cv2.Canny(blur, 40, 200)`, keeping the upper threshold at 200 and the image unchanged.

Which effect on the edge map is expected?

A. The map is unchanged, since hysteresis depends only on the upper threshold of 200.

B. Weak segments touching strong edges are now kept, closing gaps, while isolated weak segments stay removed.

C. Every pixel with gradient above 40 becomes an edge, so isolated texture specks now appear everywhere.

D. Strong edges above 200 are dropped, since the wider band dilutes the hysteresis seeds.

---

**Q69.**  A colleague claims that AlexNet's 60-million-plus parameters are "mostly in its convolution stack, since that is where the image processing happens." In AlexNet, the final convolution block outputs a 6 × 6 × 256 map, which feeds FC6 (4,096 units), then FC7 (4,096 units), then FC8 (1,000 classes), all with biases.

| Layer | Input | Output |
|---|---|---|
| FC6 | 6 × 6 × 256 = 9,216 | 4,096 |
| FC7 | 4,096 | 4,096 |
| FC8 | 4,096 | 1,000 |

Which statement is correct?

A. FC8 is the largest layer, holding about 4.1 million parameters, since it has 1,000 outputs.

B. FC6 alone holds about 37.8 million parameters, and the three FC layers hold about 96% of the total.

C. FC6 holds about 9.2 million parameters, so the convolutions hold most of the network's weights.

D. The FC layers hold about 37.8 million parameters in total, roughly 60% of the network.

---

**Q70.**  A defect-segmentation pipeline cleans a 12 × 12 binary mask with a morphological opening before counting regions. The mask contains a 4 × 4 blob with a one-pixel spur attached to its edge, a 4 × 5 blob, and two isolated noise pixels.

```python
import cv2, numpy as np

m = np.zeros((12, 12), np.uint8)
m[1:5, 1:5] = 255; m[5, 2] = 255           # blob 1 plus a 1-pixel spur
m[7:11, 6:11] = 255                         # blob 2
m[2, 8] = 255; m[9, 1] = 255                # isolated noise pixels

k = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
n_raw = cv2.connectedComponents(m)[0] - 1
n_open = cv2.connectedComponents(opened)[0] - 1
print(n_raw, n_open, int((opened > 0).sum()))
```

Which option matches the printed output?

A. 5, 2 and 36, since the spur counts as its own component first.

B. 4, 2 and 10, since opening erodes each blob and never restores it.

C. 4, 2 and 36, with the noise pixels and the spur removed.

D. 4, 4 and 39, since opening fills gaps rather than removing structures.

---

**Q71.**  A warehouse detector is evaluated on one image containing 5 ground-truth parcels. Its 7 detections, sorted by confidence, are matched to ground truth at IoU ≥ 0.5 as: TP, TP, FP, TP, FP, FP, TP. Average precision uses the all-point interpolated precision–recall curve.

| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Match | TP | TP | FP | TP | FP | FP | TP |

```python
import numpy as np

tp = np.array([1, 1, 0, 1, 0, 0, 1]); n_gt = 5
prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
rec  = np.cumsum(tp) / n_gt
# all-point interpolation: precision envelope, summed over recall steps
```

Which value is the AP for this image?

A. About 0.57, the precision measured after all seven detections are counted.

B. About 0.80, the final recall reached once all detections are included.

C. About 0.83, the mean precision measured at each of the true positives.

D. About 0.66, the area under the interpolated precision–recall curve.

---

**Q72.**  An ML platform team is standardizing augmentation recipes, and a junior engineer has proposed applying the same aggressive geometric augmentation to every project. Before approving it, the lead must pick plans for three projects: handwritten digit recognition (0–9), chest-X-ray classification where some findings depend on which side of the body they appear, and aerial crop-type classification from nadir satellite tiles.

Which plan preserves labels in all three projects?

A. Digits: small rotations and shifts; X-rays: brightness and contrast, no flips; aerial: flips and 90° rotations.

B. Digits: horizontal flips; X-rays: random crops and flips; aerial: colour inversion only.

C. Digits: no augmentation, since any change breaks the labels; X-rays and aerial tiles: random flips.

D. Digits: random 180° rotations; X-rays: random horizontal flips; aerial: flips and 90° rotations.

---

**Q73.**  A company has two image-classification projects and starts each from an ImageNet-pretrained ResNet-50. The ML lead wants a defensible plan for how much of each network to retrain, given a fixed GPU budget and a strong dislike of overfitting on small datasets. For the radar project, the first convolution is adapted to accept one input channel.

| Project | Labelled images | Similarity to ImageNet |
|---|---|---|
| Retail shelf products | 800 | High (natural photos of objects) |
| Synthetic-aperture radar scenes | 400,000 | Low (single-channel radar returns) |

Which transfer-learning strategy fits each project?

A. Freeze the backbone for both and train only new heads, since ImageNet features transfer universally.

B. Freeze the backbone for retail, and fine-tune all layers for radar.

C. Fine-tune every layer for retail, and freeze the backbone for the larger radar set.

D. Train both from scratch, since pretrained features help only when classes overlap.

---

**Q74.**  An inference team converts a decoder with d_model = 4,096 and 32 heads (d_head = 128) from multi-head attention to multi-query attention. Queries keep 32 heads, but keys and values share a single head, so the K and V projections shrink to 4,096 × 128. Q and output projections are unchanged, and biases are ignored. The model serves long-context chat with many concurrent users on fixed GPU memory, and decoding throughput is limited by how fast cached keys and values can be read from memory at each step. The team briefly uptrains after conversion to recover quality.

```python
d, h, d_head = 4096, 32, 128
mha = 4 * d * d                                   # Wq, Wk, Wv, Wo
mqa = 2 * d * d + 2 * d * d_head                  # Wq, Wo full; Wk, Wv single-head
print(mha, mqa)
```

Which option gives the per-layer attention parameters after the change, and the main reason teams accept it?

A. About 67.1 million, unchanged, since sharing K and V heads alters only the cache.

B. About 34.6 million, a halving whose main benefit is lower prefill computation for long prompts.

C. About 2.1 million, since every projection shrinks to a single head of width 128.

D. About 34.6 million, about half of MHA's 67.1 million, mainly for a far smaller KV cache.

---

**Q75.**  A photo-enhancement script brightens an 8-bit image by 100 levels in two ways and prints one pixel whose original value is 200.

```python
import cv2, numpy as np

img = np.array([[200]], dtype=np.uint8)
delta = np.array([[100]], dtype=np.uint8)

a = img + delta              # NumPy arithmetic
b = cv2.add(img, delta)      # OpenCV arithmetic
print(a[0, 0], b[0, 0])
```

Which option matches the printed values?

A. 255 and 255, since both libraries clip uint8 results to the valid range.

B. 300 and 255, since NumPy promotes the result to a wider integer type.

C. 44 and 255, NumPy wrapping around while OpenCV saturates at 255.

D. 255 and 44, since OpenCV wraps while NumPy saturates at 255.

---

**Q76.**  A team retrains its tokenizer, growing the vocabulary from 32,000 to 128,000 tokens, for a model with d_model = 4,096. The input embedding and output projection are separate (untied) matrices, each of size vocabulary × d_model.

| Setting | Before | After |
|---|---|---|
| Vocabulary | 32,000 | 128,000 |
| d_model | 4,096 | 4,096 |
| Embedding tying | untied | untied |

How many parameters does the change add, and what is a typical upside of the larger vocabulary?

A. About 786 million, with text split into fewer tokens, especially for non-English scripts.

B. About 393 million, with the larger vocabulary shrinking every attention layer.

C. About 786 million, with attention costs unchanged, since tokens get longer but not fewer.

D. About 1.57 billion, counting each matrix twice, with every tokenization stage now lossless.

---

**Q77.**  In a decoder with causal attention, the third token (index 2) of a four-token sequence has raw scaled scores (2.0, 1.0, 0.5, 3.0) against keys 0–3 before masking.

```python
import numpy as np

scores = np.array([2.0, 1.0, 0.5, 3.0])
mask = np.array([0.0, 0.0, 0.0, -np.inf])    # key 3 is in the future
w = np.exp(scores + mask) / np.exp(scores + mask).sum()
print(w.round(3))
```

Which option matches the printed weights?

A. About (0.333, 0.333, 0.333, 0.000), since causal masking spreads weight evenly over past keys.

B. About (0.571, 0.286, 0.143, 0.000), normalizing the raw scores without exponentiating.

C. About (0.629, 0.231, 0.140, 0.000), with the future key excluded.

D. About (0.232, 0.085, 0.052, 0.631), with key 3 receiving most of the attention weight.

---

**Q78.**  A team pretrains a BERT-style encoder with the standard masked-language-model recipe: 15% of tokens are selected for prediction. Of those, 80% are replaced with [MASK], 10% with a random token, and 10% are left unchanged.

For a 512-token sequence, what are the expected numbers of selected tokens and of [MASK] tokens, and why is the 80/10/10 split used?

A. About 76.8 selected and 76.8 [MASK], since the random and unchanged tokens are added after masking.

B. About 76.8 selected and 61.4 [MASK], so that each masked token has fewer clues from context.

C. About 76.8 selected and 61.4 [MASK], since fine-tuning inputs contain no [MASK] tokens.

D. About 61.4 selected and 49.2 [MASK], since the 80% rule applies to the full sequence.

---

**Q79.**  A team implements a prefix-LM attention mask for a sequence of 5 tokens, where the first 2 tokens form a bidirectional prefix and the rest are generated causally. In the mask, 1 means "may attend".

```python
import numpy as np

T, P = 5, 2
causal = np.tril(np.ones((T, T), dtype=int))
prefix = np.zeros((T, T), dtype=int); prefix[:, :P] = 1
mask = np.maximum(causal, prefix)
print(mask[0], mask[3])
```

Which option matches the two printed rows?

A. `[1 1 0 0 0]` and `[1 1 0 0 0]`, since generated tokens may attend only to the prefix.

B. `[1 1 0 0 0]` and `[1 1 1 1 0]`, prefix tokens seeing each other and later tokens causal.

C. `[1 0 0 0 0]` and `[1 1 1 1 0]`, since row 0 stays purely causal under the mask.

D. `[1 1 1 1 1]` and `[1 1 1 1 1]`, since prefix columns open every row fully.

---

**Q80.**  A small research model uses learned absolute position embeddings and was trained with a 2,048-token context. A user runs it on a 3,000-token contract to test long-document summarization, and the script raises a runtime error before any output is produced.

```python
import torch, torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab=50000, d=512, max_len=2048):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)                 # line 6
        self.pos = nn.Embedding(max_len, d)               # line 7
    def forward(self, ids):
        T = ids.size(1)
        pos_ids = torch.arange(T, device=ids.device)      # line 10
        return self.tok(ids) + self.pos(pos_ids)          # line 11

out = TinyLM()(torch.randint(0, 50000, (1, 3000)))
```

Which line raises the error, and why?

A. Line 7, since position tables must be sized to the largest input ever seen.

B. Line 11, since indices 2,048 and above fall outside the learned position table.

C. Line 6, since 50,000-word vocabularies exceed the embedding table's limit.

D. Line 10, since `torch.arange` cannot generate more than 2,048 position values.

---

**Q81.**  A team fine-tunes a decoder on instruction data. Prompt tokens and padding are labelled −100 so that only response tokens contribute to the loss, and the model shifts labels internally for next-token prediction.

```python
batch = [
    {"prompt_len": 37, "response_len": 24},
    {"prompt_len": 52, "response_len": 31},
    {"prompt_len": 18, "response_len": 9},
]
# labels = [-100] * prompt_len + response_ids, then right-padded with -100
```

How many tokens contribute to the loss for this batch, and why mask the prompt?

A. 64 tokens, so the loss rewards answering rather than reproducing the prompt.

B. 61 tokens, since the internal shift drops one response label in every sequence.

C. 107 tokens, the prompt tokens, which supervise the model's understanding of instructions.

D. 171 tokens, all positions, so that the model learns to reproduce the prompts as well.

---

**Q82.**  A RAG assistant runs on a model with an 8,192-token context window. Before any retrieved text is added, the request already holds a 600-token system prompt, a 150-token user question, and 1,400 tokens of cached conversation history. A further 1,024 tokens are reserved for the model's answer. Retrieved chunks are appended strictly in reranker order. The packer stops at the first chunk that would exceed the remaining budget, and it never truncates a chunk or skips ahead to smaller ones.

```python
WINDOW, SYSTEM, QUESTION, HISTORY, ANSWER = 8192, 600, 150, 1400, 1024
chunks = [900, 1250, 700, 1100, 800, 1150, 250]     # token sizes, in reranked order

budget = WINDOW - SYSTEM - QUESTION - HISTORY - ANSWER
used, packed = 0, 0
for size in chunks:
    if used + size > budget:
        break
    used += size; packed += 1
print(budget, packed, used)
```

Which option matches the printed output?

A. 6,042, 6 and 5,900, since the answer reserve is only claimed after generation starts.

B. 6,418, 7 and 6,150, since cached history tokens do not count against the context window.

C. 5,018, 6 and 5,000, since the 250-token chunk fits once the 1,150 is skipped.

D. 5,018, 5 and 4,750, stopping when the sixth chunk would overflow the budget.

---

**Q83.**  A team doubles the number of heads in a transformer layer from 8 to 16, keeping d_model = 1,024 and the standard four d_model × d_model projections.

| Configuration | Heads | d_model |
|---|---|---|
| Before | 8 | 1,024 |
| After | 16 | 1,024 |

Which statement is correct?

A. Projection parameters stay at about 4.2 million, while each head's width falls from 128 to 64.

B. Projection parameters stay at about 4.2 million, while head width rises from 64 to 128.

C. Projection parameters double to about 8.4 million, since every new head needs its own matrices.

D. Projection parameters halve to about 2.1 million, since each head becomes half as wide.

---

**Q84.**  A hybrid retriever fuses a BM25 ranking and a dense-vector ranking with reciprocal rank fusion, score(d) = Σ 1/(k + rankᵢ(d)) with k = 60. A document missing from a list contributes nothing for that list.

```python
from collections import defaultdict

bm25  = ["D3", "D1", "D7", "D2", "D5"]
dense = ["D2", "D1", "D5", "D3", "D9"]

score = defaultdict(float)
for ranking in (bm25, dense):
    for rank, doc in enumerate(ranking, start=1):
        score[doc] += 1 / (60 + rank)
best = max(score, key=score.get)
print(best, round(score[best], 4))
```

Which option matches the printed output?

A. D3 with about 0.0320, since first place in BM25 dominates the fused score.

B. D2 with about 0.0320, since dense retrieval is weighted more heavily in fusion.

C. D3 and D2 tie at about 0.0320, since each tops one of the two lists.

D. D1 with about 0.0323, ranked second in both lists, so neither ranker put it first.

---

**Q85.**  A legal-research RAG system embeds whole 1,500-token contract sections. Retrieval often returns the right section, but similarity scores are diluted because each section mixes several unrelated clauses, and precise clause-level questions are frequently missed. When the team switches to 150-token chunks, retrieval gets sharper but the generator sees clauses without their defined terms and surrounding conditions.

Which design addresses both problems?

A. Embed 1,500-token sections with 50% overlap, so every clause lands in two neighbouring sections.

B. Embed 150-token chunks and raise top-k to 100, so enough surrounding context is retrieved.

C. Embed whole contracts as single vectors, so the generator always receives every defined term.

D. Embed small child chunks for matching, then pass each match's parent section along.

---

**Q86.**  A team adopts an asymmetric retrieval embedding model whose model card says queries must be prefixed with "query: " and documents with "passage: ". Their indexing and search code is:

```python
from sentence_transformers import SentenceTransformer

enc = SentenceTransformer("intfloat/e5-base-v2")

doc_vecs = enc.encode([d.text for d in docs], normalize_embeddings=True)
index.add(doc_vecs)

def search(q, k=10):
    qv = enc.encode([q], normalize_embeddings=True)
    return index.search(qv, k)
```

Offline evaluation shows recall@10 well below the model card's reported numbers. Which change is most likely to close the gap?

A. Disable normalization, since asymmetric models are trained for raw inner-product search instead.

B. Raise k from 10 to 50, since asymmetric models spread relevant passages widely.

C. Add the "query: " and "passage: " prefixes the model expects, then re-index.

D. Switch to Euclidean distance, since cosine similarity is unsuited to models trained with prefixes.

---

**Q87.**  A team must choose how many candidates the first-stage retriever passes to a cross-encoder reranker, which then keeps the top 5 for the generator. Reranker latency grows linearly with the number of candidates. On 500 labelled queries:

| Candidates retrieved (k) | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| Recall@k | 0.61 | 0.74 | 0.89 | 0.93 | 0.94 |
| Precision@k | 0.48 | 0.36 | 0.22 | 0.10 | 0.05 |

A reviewer says: "Precision collapses as k grows, so we should send the reranker as few candidates as possible." Which assessment is correct?

A. Use k = 5, since the reranker keeps 5 anyway and extra candidates only add latency.

B. Recall matters for the first stage, so k = 20 captures most gains before returns flatten.

C. The reviewer is right, since the reranker needs high precision at k to produce good final results.

D. Use k = 100, since the highest recall always gives the reranker the best possible final set.

---

**Q88.**  A team tests answer accuracy as the single relevant chunk is moved among 20 retrieved chunks in the prompt, with the other 19 chunks irrelevant.

| Position of relevant chunk | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Answer accuracy | 0.82 | 0.66 | 0.55 | 0.63 | 0.79 |

Which change to context assembly is best supported by these results?

A. Place the highest-ranked chunk in the exact middle, since accuracy at position 10 is most stable.

B. Increase the retrieved chunks to 40, since more context always offsets the weaker middle positions.

C. Place the highest-ranked chunks at the start and end of the context, with weaker ones in the middle.

D. Place the chunks in random order, since the position of the relevant chunk has no consistent effect.

---

**Q89.**  A team measures the rank of the first relevant document for four test queries, before and after adding a cross-encoder reranker.

```python
first_relevant_before = [3, 1, 5, 2]
first_relevant_after  = [1, 1, 2, 1]

mrr = lambda ranks: sum(1 / r for r in ranks) / len(ranks)
print(round(mrr(first_relevant_before), 3), round(mrr(first_relevant_after), 3))
```

Which option matches the printed output?

A. 0.508 and 0.875, but the change is noise since the median rank barely moves.

B. 0.508 and 0.875, an improvement in mean reciprocal rank after reranking.

C. 0.875 and 0.508, since reranking pushes relevant documents lower in the list.

D. 2.75 and 1.25, the mean ranks, so lower values indicate better reranking.

---

**Q90.**  An industrial-support RAG system answers technician questions about fault codes such as "E-4471" and part numbers such as "VX-220-B". Dense retrieval alone often returns passages about similar but different codes, because embeddings treat "E-4471" and "E-4417" as nearly identical strings.

Which retrieval change best fits this failure?

A. Increase the embedding dimension, since larger vectors always separate near-identical identifiers better.

B. Add BM25 keyword search and fuse its results with dense retrieval, since exact tokens match lexically.

C. Lowercase and strip digits from the queries, so identifiers match their surrounding descriptions.

D. Raise top-k for dense retrieval, since the exact code will appear further down the ranking.

---

# ANSWER KEY

| Q | Ans | Domain | Q | Ans | Domain | Q | Ans | Domain |
|---|-----|--------|---|-----|--------|---|-----|--------|
| 1 | A | Stats | 31 | C | PyTorch/TF | 61 | C | NLP |
| 2 | B | Stats | 32 | A | PyTorch/TF | 62 | B | NLP |
| 3 | B | Stats | 33 | B | ML | 63 | D | NLP |
| 4 | A | Stats | 34 | D | PyTorch/TF | 64 | C | NLP |
| 5 | D | Stats | 35 | D | ML | 65 | A | NLP |
| 6 | C | Stats | 36 | D | ML | 66 | B | CV |
| 7 | B | Stats | 37 | B | ML | 67 | C | NLP |
| 8 | D | Stats | 38 | C | ML | 68 | B | CV |
| 9 | D | Prompts-Agents-RAG | 39 | C | ML | 69 | B | CV |
| 10 | C | Stats | 40 | A | ML | 70 | C | CV |
| 11 | A | Prompts-Agents-RAG | 41 | A | DL | 71 | D | CV |
| 12 | B | Prompts-Agents-RAG | 42 | D | ML | 72 | A | CV |
| 13 | A | Prompts-Agents-RAG | 43 | D | DL | 73 | B | CV |
| 14 | A | Prompts-Agents-RAG | 44 | C | DL | 74 | D | LLM |
| 15 | B | Prompts-Agents-RAG | 45 | D | DL | 75 | C | CV |
| 16 | A | Prompts-Agents-RAG | 46 | A | DL | 76 | A | LLM |
| 17 | B | RAI-Gov | 47 | B | DL | 77 | C | LLM |
| 18 | A | Prompts-Agents-RAG | 48 | A | DL | 78 | C | LLM |
| 19 | A | RAI-Gov | 49 | C | DL | 79 | B | LLM |
| 20 | B | RAI-Gov | 50 | B | RAI-T | 80 | B | LLM |
| 21 | D | RAI-Gov | 51 | C | DL | 81 | A | LLM |
| 22 | A | RAI-Gov | 52 | A | RAI-T | 82 | D | RAG |
| 23 | D | RAI-Gov | 53 | A | RAI-T | 83 | A | LLM |
| 24 | C | RAI-Gov | 54 | D | RAI-T | 84 | D | RAG |
| 25 | C | PyTorch/TF | 55 | B | RAI-T | 85 | D | RAG |
| 26 | D | RAI-Gov | 56 | B | RAI-T | 86 | C | RAG |
| 27 | A | PyTorch/TF | 57 | D | RAI-T | 87 | B | RAG |
| 28 | D | PyTorch/TF | 58 | C | NLP | 88 | C | RAG |
| 29 | C | PyTorch/TF | 59 | C | RAI-T | 89 | B | RAG |
| 30 | A | PyTorch/TF | 60 | D | NLP | 90 | B | RAG |

# EXPLANATIONS

**Q1 — A.** The interaction contrast is (46 − 40) − (38 − 44) = 12. For a balanced 2 × 2 design with n = 5, SS_AB = n·c²/4 = 5 × 144/4 = 180 on 1 df, so F = 180/12 = 15.0 > 4.49 (p ≈ 0.001). The layout main effect is exactly zero because the column means are both 42, but layout B helps in the morning and hurts in the evening. D misses the interaction. B miscomputes F. C assigns the interaction to the main effect.

**Q2 — B.** Requiring both positives gives sensitivity 0.90 × 0.85 = 0.765 and specificity 1 − (0.20 × 0.05) = 0.99. At 2% prevalence, PPV rises from 0.084 (Test 1 alone) to 0.61. C gives the either-positive rule. D takes one test's figures. A combines the tests wrongly.

**Q3 — B.** OR = (60 × 210)/(40 × 90) = 3.5. A case-control design fixes the case:control ratio, so disease risk cannot be read from the table. Bayes uses P(E|D) = 0.60, P(E|¬D) = 0.30 and the 1% prevalence: P(D|E) = 0.006/(0.006 + 0.297) = 0.0198 ≈ 2.0%. C is the journalist's error (60/150). D misnames the ratio of exposure rates as the odds ratio. A reports P(E|D).

**Q4 — A.** VIF = 1/(1 − 0.95) = 20 for each tenure feature, so their coefficient standard errors are inflated by √20 ≈ 4.5. Collinearity destabilizes the coefficients, not the predictions, which can remain accurate. Dropping one of the near-duplicates usually resolves it. C overstates the harm to predictions. D misreads a VIF of 1.67, which is benign. B misreads the scale.

**Q5 — D.** The pooled test assumes equal variances, which is badly violated (SD 4 against 10, with the small group having the large variance). That shrinks its standard error and gives t = 2.51, p = 0.016. Welch gives t = 1.55 with Welch–Satterthwaite df ≈ 9.7 and p = 0.15. Welch is the appropriate test, so the difference is not significant. C ignores the unequal variances. A swaps the results. B uses the pooled df.

**Q6 — C.** The overall churn rate is 100/700. The expected basic-plan churners are 300 × 100/700 = 42.86. χ² = 14.0 with (3 − 1)(2 − 1) = 2 df, p ≈ 0.0009. D miscounts the df. A is wrong: the Yates correction applies only to 2 × 2 tables. B confuses expected and observed counts.

**Q7 — B.** Without centring, the first singular vector points toward the data's mean, so it absorbs the large mean offset and shows 0.999 of the "variance". Centred PCA puts PC1 along (0.71, 0.71) and explains 0.95. C is wrong: np.linalg.svd does not centre. D accepts the artefact. A reverses the two figures.

**Q8 — D.** After one month, v₁ = (0.650, 0.285, 0.065). After two, v₂ = v₁P = (0.606, 0.269, 0.126), so 60.6% are active and 12.6% churned. B stops after one step. C multiplies P @ P @ v, which treats rows as destinations. A reads the at-risk share as churned.

**Q9 — D.** Principle: an explicit rule beats an illustrative example for semantics and edge cases. P3 states how to read ambiguous slash dates (03/04/2025 → 3 April), how to expand two-digit years, and that impossible dates such as 31/02/2025 are INVALID. P5 gives the day/month rule but not the other edge cases. P4's "British conventions" implies day/month order, which settles the main ambiguity. P2's examples are both unambiguous (day > 12), so they never demonstrate how 03/04 should be read, though they do show year expansion. That puts P4 above P2 (positions 3–4). P1 teaches the least. B swaps P2 and P4. C puts P5 first. A misplaces P4.

**Q10 — C.** Design effect = 1 + (m − 1)ρ = 1 + 19 × 0.05 = 1.95, so n_eff = 2,000/1.95 ≈ 1,026. Clustering nearly halves the information. B omits the cluster size. D confuses the cluster size with the design effect. A divides by √1.95, which is the standard-error inflation, not the sample-size factor.

**Q11 — A.** The model issued two parallel tool_use blocks. Line 12 takes only the first, so the next request contains an assistant turn with two tool_use IDs but only one tool_result. The API rejects that: every tool_use must be answered in the following user turn. The fix is to loop over all tool_use blocks and return every result. B is wrong: the SDK accepts its own content objects. C is wrong: tool results belong in a user turn. D is wrong: max_tokens limits output length and does not cause this error.

**Q12 — B.** Principle: an example beats every verbal description for exact surface format. P5's two examples show spacing, brackets and the escaped inner quotes. P2 shows one real line but no escaping. P4's template fixes the structure with less concrete punctuation context than a real line. P1's prose is thorough but cannot pin every spacing detail, so it falls below the template (positions 3–4). P3 is vague. C swaps P1 and P4. D puts the prose first. A puts the template above a real example.

**Q13 — A.** Principle: chain-of-thought wins and terse output ranks last for multi-step arithmetic. P1 forces every step, including the floor and the round-up, and verifies them. P4's worked example demonstrates the round-up (90 → 96) and the zero floor, the two historical failure points, which P5's step list omits. P5 still forces the intermediate values, which beats P3's bare list of inputs (positions 3–4). P2 forbids reasoning. For K-114: 38 × 6 + 50 = 278; 278 − 200 = 78; rounded up to 96. B puts P5 above P4. C swaps P3 and P5. D ranks the example first.

**Q14 — A.** The returned grades are [1, 3, 0]. DCG@3 = 1/1 + 7/1.585 + 0 = 5.417. The ideal ordering of all judged documents is [3, 3, 2], giving IDCG@3 = 7 + 4.417 + 1.5 = 12.917, so nDCG@3 = 0.42. D normalizes by retrieved documents only, which ignores the missed perfect document d5 and overstates quality. B uses linear gain, but the code uses 2^r − 1. C is the unnormalized DCG.

**Q15 — B.** Principle: the detailed prompt carries a subtly harmful instruction. P3 is the most thorough, but its "only values you are certain" filter suppresses borderline items such as the device ID, the masked account number and the phone number inside a note, which is exactly the recall the task needs. It ranks last. P2 has the same coverage and explicitly keeps uncertain cases. P5 adds indirect identifiers. P4 lists direct identifiers only, but still beats P1's unspecified scope (positions 3–4). C swaps P1 and P4. A and D reward P3's detail.

**Q16 — A.** Principle: negative constraints backfire, and positive reframing generalizes better. P4 restricts the model to the passages and supplies a fallback behaviour for unanswerable questions. P5 restricts without a fallback. P3 frames the passages positively as the source. P2 is a stack of prohibitions that names what to avoid but not what to do, so models tend to fill gaps anyway. That puts P2 below P3 (positions 3–4). P1 is vague and negative. C swaps P2 and P3. D puts P5 first. B ranks the negative list second.

**Q17 — B.** Post-deployment governance for a broad rollout needs an ongoing, structured loop. Feedback is collected from the people using each model, triaged, owned, answered and fed into updates and re-evaluation. That addresses the scattered complaints, silent abandonment and unknown quality drift. D is a one-off snapshot that cannot track drift. A measures system health, not answer quality or user harm. C measures generic capability, not performance in these teams' tasks.

**Q18 — A.** Principle: persona wins outright when the objective is register. P3's character carries the warmth, pacing and plain vocabulary together. P4 combines audience and rules. P2 names the audience, which shapes register better than P1's bare style rules (positions 3–4). P5 is vague. B swaps P1 and P2. C puts P4 first. D underrates P4.

**Q19 — A.** Both groups are approved at 500/1000 = 0.50, so the demographic-parity difference is 0 and the disparate-impact ratio is 1.0. TPR is 320/400 = 0.80 for group A and 120/200 = 0.60 for group B, an equal-opportunity gap of 0.20. FPRs are 0.30 and 0.475. B and C misread the approval rates. D reports the FPR gap as the equal-opportunity gap.

**Q20 — B.** Known impossibility results (Kleinberg et al.; Chouldechova) show that when base rates differ and prediction is imperfect, a score cannot be calibrated within groups and also equalize false-positive and false-negative rates. One criterion must be traded off. C ignores proxies and the underlying arithmetic. D is false. A is wrong: group-specific thresholds can equalize error rates but break calibration.

**Q21 — D.** Raw prompts containing customer and HR data are stored without a retention limit and used for training. Users were told logging was for troubleshooting, so this breaches purpose limitation, data minimization and retention. It needs a lawful basis, PII handling, a retention limit, and consent or notice. C and A are minor technical points. B is wrong: attributing actions to users is usually required for accountability.

**Q22 — A.** High-risk obligations centre on risk management, data governance and bias testing, technical documentation, record-keeping, transparency, and human oversight, especially for automated rejections. B treats transparency as sufficient. C confuses accuracy with fairness; historical hires can encode bias. D substitutes intent for process.

**Q23 — D.** The card lacks intended uses, out-of-scope uses and limitations. The model was trained on product-surface photos from two plants, so injury-photo triage is far outside its training distribution and is safety-critical. Documenting scope would stop the reuse or force a new evaluation. B and A help engineering, not responsible use. C is wrong because more in-distribution accuracy says nothing about the new domain.

**Q24 — C.** An AUC of 0.91 shows gender is highly recoverable from the remaining features, so the model can still act on it through proxies. "Fairness through unawareness" fails, and the outcomes must be tested for disparities. B is the fallacy. D overclaims: recoverability is a risk, not proof of disparate outcomes. A overclaims: other proxies can remain, and removal can cost legitimate signal.

**Q25 — C.** The model was moved to CUDA on line 4, but each batch stays on the CPU. The first forward pass on line 10 raises a device-mismatch RuntimeError ("Expected all tensors to be on the same device"). The fix is xb, yb = xb.to(device), yb.to(device). The script has other real bugs that do not block it. Line 11 feeds probabilities to CrossEntropyLoss, which already applies log-softmax; that runs but weakens gradients (option B claims it raises). opt.zero_grad() is never called, so gradients accumulate; that runs but corrupts updates (option D claims it raises). A further bug is absent from the options: model.eval() is never called before evaluate, so dropout and batch-norm layers stay in training mode during validation. A is false: an optimizer built after .to() is the recommended order. This relies on documented PyTorch behaviour.

**Q26 — D.** The harm is reproducible and public, and it carries safety risk. Contain first with a guardrail, topic block or rollback, while the full fix is tested. Notify the relevant internal and external stakeholders, then fix the root cause and document the incident. A leaves the harm live for two days. B destroys evidence needed for accountability. C misstates the facts: reproduction shows the harm is systematic, not a one-off.

**Q27 — A.** Line 6 broadcasts (4, 5) over the batch, giving (32, 4, 5). Line 7's transpose returns a non-contiguous view of shape (32, 5, 4). reshape copies when needed, so line 8 succeeds. view never copies and fails on these strides, raising "view size is not compatible with input tensor's size and stride… Use .reshape(...)". Line 10 would be valid: (32, 20) @ (20, 8). B, C and D misjudge broadcasting, reshape and matmul. This relies on documented PyTorch behaviour.

**Q28 — D.** ceil(1003/32) = 32 batches, with the last holding 1003 − 31 × 32 = 11 photos. The default collate stacks samples along a new first dimension, giving (32, 3, 64, 64) images and a (32,) label tensor. A applies drop_last=True. B invents padding and a channels-last layout. C stacks along the wrong axis.

**Q29 — C.** 64 → conv k5 → 60 → pool → 30 → conv k3 p1 → 30 → pool → 15. The flatten has 32 × 15 × 15 = 7,200 features. A assumes the first convolution is padded (64 → 32 → 16, giving 8,192). B mis-trims. D ignores the pooling and the second convolution.

**Q30 — A.** nn.DataParallel wraps the network as .module, so every saved key reads like "module.conv1.weight". A bare Net expects "conv1.weight", so load_state_dict(strict=True) reports both missing and unexpected keys. The fix is to save model.module.state_dict(), or strip the prefix before loading. B is wrong: map_location="cpu" handles CUDA tensors. C is wrong: the call saved model weights. D inverts what map_location does. This relies on documented PyTorch behaviour.

**Q31 — C.** BCEWithLogitsLoss applies the sigmoid internally. loss_a = −log σ(2) = 0.127. loss_b feeds σ(2) = 0.881 in as if it were a logit, so the loss computes −log σ(0.881) = 0.347, squashing twice. This inflates the loss and flattens gradients. D assumes the loss detects probabilities, which it does not. A swaps the two values. B invents thresholding.

**Q32 — A.** Step 1: v = 0.5, so w = 0.95. Step 2: v = 0.95, so w = 0.855. Step 3: v = 1.355, so w = 0.7195 ≈ 0.72. C is plain SGD (1 − 3 × 0.05). D is the weight after two steps. B drops the decay factor (μ = 1).

**Q33 — B.** max_features="sqrt" draws √64 = 8 candidates at each split. P(a given feature is among them) = 8/64 = 0.125, so in expectation it is available at the root of 200 × 0.125 = 25 trees. Every other tree must root on something else, which is how random forests decorrelate their trees. C ignores feature subsampling. D uses the log2 rule (6/64 × 200 = 18.75). A counts the trees where the feature is unavailable.

**Q34 — D.** Dense(64) has 784 × 64 + 64 = 50,240 parameters and Dense(10) has 64 × 10 + 10 = 650, for 50,890 in total. The output layer already applies softmax, yet the loss is told to expect logits (from_logits=True), so softmax is effectively applied twice. Training still runs but converges slowly to a worse fit. Use from_logits=False, or drop the output activation. C gives only the first layer's count and invents a one-hot requirement. A is wrong: Flatten has no parameters. B is wrong: Dense biases are trainable by default.

**Q35 — D.** High-dimensional sparse text with few samples suits a linear SVM, which is fast and strong on TF-IDF. Predicting a continuous rupee amount with one signed effect per feature is linear regression. A uses clustering for a labelled task. B's k-NN degrades in 30,000 sparse dimensions, and logistic regression predicts classes, not amounts. C also misuses logistic regression for a continuous target.

**Q36 — D.** Contact when p·C_FN > (1 − p)·C_FP. That gives p > C_FP/(C_FP + C_FN) = 10/100 = 0.10. Because missing a churner costs nine times as much as a wasted offer, the threshold should be low. B inverts the formula. C ignores the costs. A uses the cost ratio 10/90 instead of the share 10/100.

**Q37 — B.** The city mean is computed over all rows, including each customer's own label and every validation-fold label, before cross-validation. For small cities, the encoded value nearly reveals the customer's own outcome, which inflates AUC. The fix is to compute target encodings inside each training fold, with out-of-fold or smoothed encoding. A misjudges the direction. C is wrong: using the target is exactly the leak. D invents a rule about feature scale.

**Q38 — C.** At α = 0.01: T1 = 0.21, T2 = 0.12 + 0.04 = 0.16, T3 = 0.08 + 0.09 = 0.17, T4 = 0.05 + 0.20 = 0.25. T2 is lowest. D gives T3's score at α = 0.005. A ignores the penalty. B overstates the penalty's effect.

**Q39 — C.** Inertia never increases as k grows, so its falling at k = 6 is guaranteed, not evidence. The drops are 480, 220, 40, 25 and 20. Gains collapse after k = 3, which is the elbow. D treats inertia as a quality score. A picks the largest single drop, ignoring the second large one. B misreads the table, since inertia does not rise.

**Q40 — A.** Per 10,000 transactions, Model A gives 80 true positives and 495 false positives, so precision is 0.14. Model B gives 70 true positives and 99 false positives, so precision is 0.41. ROC-AUC ignores the base rate, so a small difference in FPR dominates precision when positives are rare. D confuses precision with TPR. B swaps the values. C confuses precision with specificity.

**Q41 — A.** The LSTM cell state updates additively: keep a forget-gated share of the old state, then add the input-gated candidate, c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t. This additive path is what lets gradients survive many time steps. C swaps which gate scales which term. D adds a squashing step that does not exist. B is the coupled-gate variant, which has no separate forget gate, so it contradicts the f_t line above it.

**Q42 — D.** Validation loss bottoms out at 400 rounds (0.335) and then rises, which is overfitting. A smaller learning rate takes smaller steps, so reaching a similar loss typically needs roughly proportionally more rounds, about twice as many at 0.05. B ignores the rise. C reverses the effect. A stops before the minimum.

**Q43 — D.** PyTorch's GRU computes the candidate as n_t = tanh(W_in x + b_in + r_t ⊙ (W_hn h + b_hn)), with the reset gate applied to the transformed hidden term. A uses a sigmoid, but the candidate must be tanh. B applies the reset gate to the input term. C uses the update gate where the reset gate belongs.

**Q44 — C.** The output gate filters a tanh-squashed copy of the new cell state, h_t = o_t ⊙ tanh(c_t). D omits the tanh, letting the unbounded cell state leak out. A reverses the order of squashing and gating. B uses the previous cell state c_{t−1}.

**Q45 — D.** Each gate has 256 × 128 input weights, 256 × 256 hidden weights and 2 × 256 biases, totalling 98,816. Four gates give 395,264. C counts one bias per gate. A uses three gates, which is the GRU count. B counts a single gate.

**Q46 — A.** 100 → conv k3 → 98 → conv k3 s2 p1 → floor(99/2) + 1 = 49 → maxpool 2 → 24 → conv k5 p2 → 24 → avgpool 3 s3 → 8. Flatten = 40 × 8 × 8 = 2,560. C rounds incorrectly. D ignores the padding. B ignores the pooling.

**Q47 — B.** ReLU zeroes about half its inputs, so He initialization doubles the variance: std = √(2/512) = 0.0625. Xavier assumes linear or tanh-like units and under-scales deep ReLU stacks. C and D pick Xavier. A attaches Xavier's uniform value to He's name.

**Q48 — A.** Running mean = 0.9 × 2.0 + 0.1 × 3.0 = 2.1. The unbiased batch variance is 1.0 × 8/7 = 1.143, so the running variance is 0.9 × 4.0 + 0.1 × 1.143 = 3.714 ≈ 3.71. D uses the biased variance (3.70). B inverts the momentum. C wrongly claims the batch statistics replace the running ones. This relies on documented PyTorch behaviour.

**Q49 — C.** PyTorch uses inverted dropout. In training, each element is zeroed with probability 0.2, and survivors are scaled by 1/(1 − 0.2) = 1.25 so the expected value is unchanged. In evaluation, dropout is the identity. D scales by the keep probability during training. A and B describe classic (non-inverted) dropout, which scales at inference.

**Q50 — B.** With n = 19, k = ⌈20 × 0.9⌉ = 18, so q̂ is the 18th smallest score, 0.71. Classes enter the set when 1 − p ≤ 0.71: class 0 (0.45) and class 1 (0.70) qualify, but class 2 (0.90) does not, giving {0, 1}. The finite-sample correction raises the quantile above the plain 90th percentile (0.62), which is what guarantees coverage. A uses that uncorrected quantile. C ignores the computed threshold. D takes the maximum score as q̂.

**Q51 — C.** z = 1.0 and ŷ = σ(1.0) = 0.731. Under BCE, ∂L/∂w = (ŷ − y)·x = −0.269 × 1.5 = −0.403. Under squared error, ∂L/∂w = (ŷ − y)·ŷ(1 − ŷ)·x = −0.269 × 0.197 × 1.5 = −0.079. The extra σ′ factor shrinks the gradient, which is why BCE trains sigmoid outputs faster. B swaps the two. D drops the input x. A omits σ′ from the squared-error gradient.

**Q52 — A.** φ_a = ½[(14 − 10) + (20 − 12)] = 6 and φ_b = ½[(12 − 10) + (20 − 14)] = 4. They sum to 10 = v(a, b) − v(∅), which is the efficiency property. The interaction surplus of 4 is split equally, but the solo effects differ. B counts only the first-added gains. D counts only the last-added gains. C splits the whole effect equally.

**Q53 — A.** Model A drops 0.22 and retains 0.70/0.92 = 76%. Model B drops 0.27 and retains 0.68/0.95 = 72%. Model B wins on clean accuracy but is the less robust model under the deployment corruptions. B swaps the two. C misreads the retention figures. D invents a relative-accuracy definition.

**Q54 — D.** A faithful explanation should change substantially when the learned weights are destroyed. Rank correlation stays at 0.93 even with every layer randomized, so the maps reflect input structure (edges, anatomy) rather than what the model learned. This is the Adebayo et al. sanity check. B misreads insensitivity as robustness. C misattributes the result to the model. A inverts the expected behaviour.

**Q55 — B.** Deletion tests faithfulness. If an explainer's top-ranked pixels really drive the prediction, removing them first should collapse accuracy fastest. E1 falls to 0.10 by 40%, while E2 falls only to 0.30, much closer to random. Human preference measures plausibility, not faithfulness. C conflates the two. A inverts the test. D misreads the shared starting point, which is identical by design.

**Q56 — B.** At t = 0.8, seven cases are kept, one of them wrong (conf 0.90), so risk is 1/7 = 0.143. At t = 0.9, five are kept, still including the wrong 0.90 case, so risk is 1/5 = 0.20. Raising the threshold increased risk here, because the remaining error sits above it. C wrongly assumes the error is dropped. D reports the abstention rate as coverage. A uses full-coverage risk at 0.8.

**Q57 — D.** Attention weights show where information was mixed in one layer. They are not a faithful measure of which tokens caused the output, since values, later layers and residual paths all matter. Studies find attention can be changed substantially without changing predictions. Use validated attribution methods, or check attention against deletion or perturbation tests. B and A overclaim. C overclaims in the opposite direction.

**Q58 — C.** T1 has 12 distinct trigrams and T2 has 11. They share 9, with 14 in the union, giving 0.643. T1 and T3 share 9 of 15 (0.600). T2 and T3 share 6 of 17 (0.353). A dropped letter disrupts fewer trigrams than a substituted letter at the start of a word, so the typo pair edges out the near-homophone pair. D reports the second pair. A misattributes that pair's score. B gives the lowest pair.

**Q59 — C.** The inputs are stable (PSI ≤ 0.04) and the model's predicted default rate has not moved, but observed defaults and AUC have deteriorated sharply. The relationship between features and outcome changed after the reporting rule, which is concept drift. Detecting and fixing it requires fresh labelled outcomes, followed by retraining or recalibration. B describes covariate shift, which the PSI rules out. D discards exactly the evidence of drift. A relies on input monitoring, which cannot see concept drift.

**Q60 — D.** idf: refund = log₁₀(20) = 1.301, delayed = log₁₀(5) = 0.699, courier = log₁₀(100) = 2.000. D1 = 3 × 1.301 + 0.699 = 4.60. D2 = 1.301 + 2 × 2.0 = 5.30. D3 = 4 × 0.699 + 2.0 = 4.80. The ranking is D2, D3, D1. B ranks by raw counts. C overweights the frequent term. A misorders D1 and D3.

**Q61 — C.** The likelihood ratio is (0.05 × 0.02)/(0.01 × 0.01) = 10. With prior odds 1:1 the posterior is 10/11 = 0.91. With a 10% prior, the odds are (1/9) × 10 = 1.11, so the posterior is 0.53. The base rate matters even with strong evidence. D ignores the prior. A swaps the two. B scales the posterior by the prior instead of updating the odds.

**Q62 — B.** Two of 20 tokens are mis-tagged (token 7 and token 15), so token accuracy is 0.90. At entity level, only PER matches exactly: P = 1/2, R = 1/3, F1 = 0.40. Abundant O tokens inflate token accuracy. A counts the truncated span as a match. C swaps the two. D miscounts the token errors.

**Q63 — D.** The average of "not" and "good" is (0.40, 1.05), nearly parallel to "good" (cosine 0.995) and opposite to "bad" (−0.99). The small "not" vector barely moves the average, so negation is lost and the review reads as positive. Order-insensitive averaging cannot represent negation. B and A assume negation is captured. C invents a halving rule.

**Q64 — C.** 1,000 words cost about 2,900 tokens in Hindi against 1,300 in English, a ratio of 2.23. That raises cost and latency, and it shrinks the effective context: 8,192/2.9 ≈ 2,824 Hindi words fit, against 6,301 English words. D subtracts instead of dividing. A is false. B gives the English word capacity.

**Q65 — A.** The same words carry different sentiment in the new domain, so the input–label relationship has shifted. The fix is to adapt the model with in-domain labelled data, through fine-tuning or mixed training, and to monitor by domain. B misdiagnoses the problem and would deepen the bias toward phone reviews. C is unsupported, since these words are not stop words. D dismisses a well-known effect.

**Q66 — B.** (224/14)² = 256 patches plus the class token gives 257 tokens. Each head computes a 257 × 257 score matrix, so 16 heads give 16 × 66,049 = 1,056,784 scores per layer. A drops the class token, which attends like any other token. C counts one head only. D uses patch size 16, not the stated 14.

**Q67 — C.** Perplexity is exp(−mean log p) = (0.2 × 0.5 × 0.1 × 0.4)^(−1/4) = 0.004^(−0.25) = 3.98. B uses the arithmetic mean. D omits the root. A reports the geometric mean probability (0.25) instead of its inverse.

**Q68 — B.** In Canny's hysteresis, pixels above the high threshold seed edges, and pixels between low and high survive only if connected to a seed. Lowering low from 100 to 40 lets fainter pixels along strong traces join their chains, closing gaps. Isolated faint segments still have no seed and are dropped. C ignores the connectivity requirement. D is wrong: the high threshold is unchanged. A is wrong: the low threshold governs which weak pixels survive.

**Q69 — B.** FC6 = 9,216 × 4,096 + 4,096 = 37.75M. FC7 = 16.78M and FC8 = 4.10M, so the FC layers total 58.6M, about 96% of AlexNet's roughly 61M parameters. The convolutions do most of the computation, but the FC layers hold most of the weights. C forgets to multiply by the 4,096 outputs. D stops at FC6. A misranks the layers.

**Q70 — C.** The spur touches blob 1, so the raw mask has 4 components: two blobs and two noise pixels. Opening is erosion then dilation with a 3 × 3 kernel, which removes the noise pixels and the spur and restores the blobs to 16 + 20 = 36 pixels. That leaves 2 components. A counts the attached spur separately. B gives the count after erosion only. D describes closing.

**Q71 — D.** Precision and recall after each detection run (1.00, 0.2), (1.00, 0.4), (0.67, 0.4), (0.75, 0.6), (0.60, 0.6), (0.50, 0.6), (0.57, 0.8). The interpolated curve gives AP = 0.4 × 1.00 + 0.2 × 0.75 + 0.2 × 0.571 = 0.664. Recall never reaches 1, because one parcel is never detected. C averages precision at the true positives only, ignoring the missed object. A and B report the final precision and recall instead of the area under the curve.

**Q72 — A.** Digits tolerate small rotations and shifts, but 180° rotation or flipping can turn a 6 into a 9 or mirror the strokes. Laterality-dependent X-ray findings change meaning under horizontal flips, so photometric changes are safe there but flips are not. Nadir aerial tiles have no canonical orientation, so flips and 90° rotations preserve the labels. B, C and D each include a label-changing transform or needlessly forgo safe augmentation.

**Q73 — B.** With little data and a similar domain, frozen pretrained features plus a new head work well and avoid overfitting. With abundant data and a very different domain (radar, single channel), deeper or full fine-tuning lets the network adapt low-level filters, since ImageNet edges and colours transfer weakly. C reverses the logic. D wastes the pretraining. A under-adapts to the radar domain.

**Q74 — D.** MHA uses 4d² = 67.1M parameters. MQA keeps Wq and Wo at d² each and shrinks Wk and Wv to d × 128, giving 2d² + 2 × 4096 × 128 = 34.6M. The main motivation is that the KV cache, and the memory traffic per decoding step, shrinks 32 times, which speeds up memory-bound decoding. B has the right count but the wrong reason, since prefill FLOPs fall only modestly. C shrinks every projection. A ignores the K and V projection change.

**Q75 — C.** NumPy uint8 arithmetic wraps modulo 256, so 200 + 100 = 300 − 256 = 44. cv2.add saturates and clips to 255. Brightening with plain NumPy can therefore turn bright pixels dark. A and D misattribute the behaviours. B is wrong: NumPy keeps the uint8 type for uint8 operands.

**Q76 — A.** Each matrix grows by 96,000 × 4,096 = 393M parameters, so both together add 786M. A larger vocabulary encodes text in fewer tokens, especially multilingual text, which cuts sequence length and attention cost. B counts one matrix and invents an attention change. C contradicts the fewer-tokens effect. D double-counts and overclaims.

**Q77 — C.** Masking key 3 with −∞ removes it. Softmax over (2.0, 1.0, 0.5) gives e² : e¹ : e^0.5 = 7.39 : 2.72 : 1.65, normalized to (0.629, 0.231, 0.140). D ignores the mask. A ignores the scores. B skips the exponential.

**Q78 — C.** 512 × 0.15 = 76.8 tokens are selected, and 76.8 × 0.8 = 61.4 become [MASK]. Downstream inputs never contain [MASK], so leaving some selected tokens unchanged or random reduces the pretrain–finetune mismatch, and forces the model to build good representations of every token. B gives the right numbers with the wrong rationale. D applies 15% after 80%. A miscounts.

**Q79 — B.** Columns 0–1 are open to every row. Row 0 also gets a 1 at column 1 from the prefix (bidirectional within the prefix): [1 1 0 0 0]. Row 3 is causal up to itself: [1 1 1 1 0]. C ignores the prefix override. D opens every row. A blocks generated tokens from earlier generated tokens.

**Q80 — B.** self.pos holds rows 0–2,047. pos_ids for T = 3,000 includes 2,048–2,999, so the lookup at line 11 raises "IndexError: index out of range in self". Learned absolute positions cannot extrapolate. Fixes are truncation, sliding windows, or relative or rotary schemes. C, D and A misplace the failure. This relies on documented PyTorch behaviour.

**Q81 — A.** Only response labels are non-negative: 24 + 31 + 9 = 64. The internal shift discards the label at position 0, which is a masked prompt token, so the response count is unchanged. Masking the prompt focuses the loss on producing answers instead of spending capacity on copying instructions. D trains on everything. B misapplies the shift. C trains on the prompts only.

**Q82 — D.** The budget is 8,192 − 600 − 150 − 1,400 − 1,024 = 5,018. Running totals are 900, 2,150, 2,850, 3,950 and 4,750. The sixth chunk (1,150) would bring the total to 5,900 > 5,018, so packing stops at 5 chunks and 4,750 tokens. C describes skip-ahead packing, which the stated rule forbids. A omits the answer reserve. B omits the cached history, which still occupies the window.

**Q83 — A.** The four projections remain d × d = 1,024² each, about 4.19M in total, whatever the head count, because heads split the same width. d_head = 1,024/16 = 64, down from 128. C and D wrongly tie parameters to the head count. B has the width change backwards.

**Q84 — D.** D1 = 1/62 + 1/62 = 0.03226. D3 = 1/61 + 1/64 = 0.03202, and D2 has the same score. D1 never ranks first in either list, but consistent rank 2 beats one first and one fourth place. This is RRF's consensus effect. A and B overweight a single list. C misses that D1 beats the tie.

**Q85 — D.** Small-to-big (parent–child) retrieval matches on precise child chunks but returns their larger parent sections, so the generator gets defined terms and conditions. A keeps the dilution. B floods the context with noise and still splits clauses from their definitions. C dilutes the vectors even further.

**Q86 — C.** E5-style models are trained with role prefixes. Omitting them misaligns query and passage embeddings, which degrades recall well below the reported numbers. The fix is to encode "query: …" and "passage: …" and re-index the documents. A is wrong: these models are used with normalized cosine similarity. B treats the symptom. D is wrong: for normalized vectors, Euclidean distance and cosine give the same ranking.

**Q87 — B.** The first stage should maximize recall cheaply, and the reranker restores precision. Recall rises from 0.74 to 0.89 between k = 10 and k = 20, then gains flatten (0.93 at 50 and 0.94 at 100) while latency keeps growing. C and A misread the role of first-stage precision. D overpays for marginal recall.

**Q88 — C.** Accuracy is U-shaped by position: highest at the start (0.82) and end (0.79), lowest in the middle (0.55). This is the "lost in the middle" effect, so the best chunks belong at the edges. D ignores the pattern. A inverts it. B overclaims; extra context typically hurts.

**Q89 — B.** Before reranking, MRR = (1/3 + 1 + 1/5 + 1/2)/4 = 0.508. After, MRR = (1 + 1 + 1/2 + 1)/4 = 0.875, a clear improvement. D reports mean ranks, which the code does not compute. A dismisses a large gain. C swaps the two values.

**Q90 — B.** Exact identifiers are a lexical-matching problem. BM25 scores the exact tokens "E-4471" and "VX-220-B", and fusing it with dense retrieval (hybrid search) keeps semantic recall for descriptive questions. A cannot reliably separate near-identical strings. C destroys the identifiers. D still ranks the near-duplicates highly.

---
