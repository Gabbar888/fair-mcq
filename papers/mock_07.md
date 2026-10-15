# AI Assessment — Mock 07
## 90 questions | 150 minutes | MCQ, single correct

Each question has exactly one correct answer. There is no negative marking. The paper is long by design; budget about 100 seconds per question and move on when stuck.

---

**Q1.** A payments company screens 200,000 card transactions per day. Historically, 0.5% of transactions are fraudulent. The fraud model outputs a score, and the team is choosing between two operating thresholds. Every flagged transaction goes into a manual review queue, and the review team can handle at most 2,000 alerts per day. The risk lead argues for T1: "Its specificity is excellent, so the queue will be almost entirely real fraud."

| Threshold | Sensitivity | Specificity |
|---|---|---|
| T1 | 0.92 | 0.980 |
| T2 | 0.80 | 0.995 |

Which threshold can the review team actually operate within capacity, and what does its alert queue look like?

A. T1, which fits capacity because a 0.98 specificity caps false alarms near 2% of alerts.

B. T2, which fits capacity, with roughly 45% of alerts genuine fraud.

C. T2, which fits capacity, and roughly 80% of its alerts are genuine fraud.

D. Neither, since both thresholds exceed 2,000 alerts once false positives are added.

---

**Q2.** A subscription business tags customers as "at risk" if they will cancel (churn) within the next 60 days. In a particular segment, 10% of customers are at risk. A retention analyst observes two behavioural signals for a given customer, one after the other: first the customer files a billing complaint, and a week later the customer downgrades to a cheaper plan. The signals are conditionally independent given risk status.

| Signal | P(signal \| at risk) | P(signal \| not at risk) |
|---|---|---|
| Billing complaint | 0.60 | 0.15 |
| Plan downgrade | 0.50 | 0.10 |

The analyst updates after the complaint, then performs a second update after the downgrade. What is the probability that this customer is at risk after both updates?

A. About 0.36

B. About 0.31

C. About 0.67

D. About 0.69

---

**Q3.** To study churn, an analyst draws a case-control sample: 500 customers who churned last quarter and 500 who were retained. Each customer is flagged by whether they were still on the legacy app version.

| | Legacy app | Current app | Total |
|---|---|---|---|
| Churned | 200 | 300 | 500 |
| Retained | 100 | 400 | 500 |

A logistic regression of churn on the legacy flag is fitted to this sample. In the full customer base, the quarterly churn rate is 8%. What are the fitted coefficient on the legacy flag, and the probability that a legacy-app customer in the full base churns?

A. Coefficient ≈ 0.98; P(churn | legacy) ≈ 0.15

B. Coefficient ≈ 0.69; P(churn | legacy) ≈ 0.19

C. Coefficient ≈ 0.98; P(churn | legacy) ≈ 0.67

D. Coefficient ≈ 0.69; P(churn | legacy) ≈ 0.67

---

**Q4.** An agronomy team runs a balanced experiment with three fertilizer types (Factor A) and two irrigation schedules (Factor B), with four plots per combination. The analysis software crashed partway through printing the two-way ANOVA table, leaving several cells blank.

| Source | SS | df | MS | F |
|---|---|---|---|---|
| Fertilizer (A) | 84 | 2 | | |
| Irrigation (B) | 30 | | | |
| A × B | 24 | | | |
| Error | 54 | | | |
| Total | 192 | 23 | | |

Reference critical values at α = 0.05: F(1, 18) = 4.41, F(2, 18) = 3.55, F(2, 23) = 3.42.

Which statement about the interaction term is correct?

A. Interaction F is 8.0, exceeding both critical values, so the interaction is significant.

B. Interaction F is 4.0, but below F(1, 18) = 4.41, so it is not significant.

C. Interaction F is 4.0, above its own critical value, so the main effects need conditional reading.

D. Interaction F is 1.44 when tested against the total mean square, so it is negligible.

---

**Q5.** A field trial records yield (tonnes per hectare) for two crops, wheat (W) and rice (R), each grown under two fertilizers, F1 and F2, with three plots per combination.

| Crop | Fertilizer | Plot yields |
|---|---|---|
| W | F1 | 10, 12, 14 |
| W | F2 | 18, 20, 22 |
| R | F1 | 14, 16, 18 |
| R | F2 | 14, 16, 18 |

A two-way ANOVA with interaction is fitted. Which statement is supported by the data?

A. Crop has no effect on yield, since both crops average exactly 16 overall.

B. Interaction F is 6.0, since the crop × fertilizer term carries two degrees of freedom.

C. Interaction F is 12.0, and the crop main effect F is likewise 12.0.

D. Interaction F is 12.0, so the benefit of fertilizer F2 depends heavily on which crop is grown.

---

**Q6.** A telecom analyst regresses monthly support cost per customer on three predictors. The model R² is 0.81, yet two coefficients have large p-values. To diagnose this, she regresses each predictor on the other two and records the auxiliary R².

| Predictor | Coefficient | Std. error | p-value | Auxiliary R² |
|---|---|---|---|---|
| tenure_months | 0.42 | 0.31 | 0.18 | 0.90 |
| total_charges | 0.08 | 0.07 | 0.26 | 0.95 |
| monthly_charges | 1.10 | 0.24 | < 0.001 | 0.50 |

Which statement about total_charges is correct?

A. VIF about 1.05, since VIF is one over the auxiliary R².

B. VIF 20, with its standard error inflated roughly 4.5-fold.

C. VIF 20, with its standard error inflated roughly 20-fold, as VIF scales it directly.

D. It is fine; monthly_charges, with the smallest p-value, drives the collinearity instead.

---

**Q7.** A hospital network compares two drugs for the same infection. Clinicians tended to give Drug Y to mild cases and Drug X to severe cases. Recovery counts are below.

| Severity | Drug X recovered / treated | Drug Y recovered / treated |
|---|---|---|
| Mild | 90 / 100 | 255 / 300 |
| Severe | 180 / 300 | 50 / 100 |

The board asks which drug had the greater overall effect. Taking the combined patient mix of both drugs (400 mild, 400 severe) as the standard population, which conclusion is correct?

A. Drug X, with a standardized recovery of about 75% against 68% for Y.

B. Drug Y, since its pooled recovery of about 76% exceeds X's 68%.

C. Drug Y, with a standardized recovery of about 75% against 68% for X.

D. Neither; once groups are weighted equally, the two drugs recover at equal rates.

---

**Q8.** A device-telemetry team tracks two mean-centred features per user session: x₁ (screen-on minutes) and x₂ (app launches). Their sample covariance matrix is below. The team wants to compress the pair into a single principal component.

| | x₁ | x₂ |
|---|---|---|
| x₁ | 5 | 2 |
| x₂ | 2 | 2 |

What share of total variance does the first principal component capture, and along which direction does it lie?

A. About 71%, as x₁'s variance of 5 is 71% of the trace.

B. About 86%, along (1, 2)/√5, pointing mostly towards app launches.

C. About 86%, along (2, 1)/√5, mostly towards screen-on minutes.

D. About 86%, along (1, 1)/√2, as the off-diagonal entries are equal.

---

**Q9.** A billing assistant built on an LLM must compute invoice totals from order details, and the output is checked by finance against the correct figure. One representative order: a customer buys 37 licences, priced in bands as below, then receives a 10% loyalty discount on the subtotal, after which 18% GST is applied.

| Licences in band | Price per licence |
|---|---|
| 1–20 | ₹1,200 |
| 21–50 | ₹1,000 |

**Objective:** the final total is arithmetically correct, across many such orders. Five system prompts were tried; each is followed by the order details.

**P1:** "Compute the invoice total. Think step by step, then give the final total."

**P2:** "Compute the invoice total. Return only the final number, with no working, so our billing system can parse it."

**P3:** "Compute the invoice total. Note that band pricing applies per licence within each band, and the discount applies before tax. Give the final total."

**P4:** "Compute the invoice total. Write each intermediate amount (each band's subtotal, the discounted subtotal, the tax) on its own line, then state the final total."

**P5:** "You are a meticulous chartered accountant with twenty years of experience. Compute the invoice total and state the final figure."

Rank the prompts from most to least effective against the objective.

A. P4 > P1 > P5 > P3 > P2

B. P1 > P4 > P3 > P5 > P2

C. P4 > P1 > P3 > P5 > P2

D. P3 > P4 > P1 > P5 > P2

---

**Q10.** A product team pre-registered a two-sided, two-sample t-test at α = 0.05 for an A/B test on average weekly spend. Equal variances are assumed. After the test, the product manager writes: "The variant clearly won — we expected an improvement and spend went up."

| Group | n | Mean spend | Std. deviation |
|---|---|---|---|
| Control | 64 | 50 | 16 |
| Variant | 64 | 55 | 16 |

Which statement is correct?

A. t ≈ 1.77; the 95% confidence interval excludes zero, so p falls below 0.05.

B. t ≈ 1.77; the pre-registered test is not significant, and the 95% interval includes zero.

C. t ≈ 0.31, since the 5-unit lift is divided by the pooled standard deviation of 16.

D. t ≈ 1.77; p ≈ 0.04 is significant at 0.05, because an improvement was expected.

---

**Q11.** An engineering team uses an LLM as the first stage of security review on pull requests to a Flask service that builds SQL strings, calls shell utilities and renders Jinja templates. Every finding the model raises is triaged by a human security engineer, who can dismiss false positives in seconds; a missed vulnerability, by contrast, ships to production.

**Objective:** maximize recall of injection-class vulnerabilities in the reviewed code.

**P1:** "Review this code for injection vulnerabilities. For each finding, give the line number and a one-sentence reason."

**P2:** "Review this code for injection vulnerabilities. List every place where user-controlled input reaches a SQL query, a shell command or a template, even where exploitation looks unlikely."

**P3:** "Review this code for security issues of any kind."

**P4:** "You are a senior application-security engineer. Review this code for injection vulnerabilities, giving line number, CWE ID, severity and a proposed fix for each. Report only issues you are confident are exploitable in production."

**P5:** "Review this code for injection vulnerabilities, covering SQL injection, OS command injection and server-side template injection, and explain each one."

Rank the prompts from most to least effective against the objective.

A. P2 > P5 > P1 > P3 > P4

B. P2 > P5 > P3 > P1 > P4

C. P4 > P2 > P5 > P1 > P3

D. P2 > P4 > P5 > P1 > P3

---

**Q12.** An internal assistant writes incident summaries, and every ticket it mentions must be cited in the house reference format so a downstream script can hyperlink it. The script matches the format character for character. A correctly formatted reference looks like `[OPS-1182 · 2025-11-02 · Sev3]`: square brackets, the ticket key, a middle dot (·) with a single space on each side, an ISO date, another spaced middle dot, and the severity written as "Sev" followed directly by a digit.

**Objective:** every ticket reference in the output matches the house format exactly.

**P1:** "You are our incident documentation bot. Cite each ticket in square brackets with its key, date and severity."

**P2:** "Cite each ticket using our standard ticket reference format."

**P3:** "Cite each ticket in square brackets: the ticket key, then the date as YYYY-MM-DD, then the severity as 'Sev' plus its number, separated by dots with a space either side."

**P4:** "Cite each ticket with its key, date and severity."

**P5:** "Cite each ticket exactly like this example: [OPS-1182 · 2025-11-02 · Sev3]"

Rank the prompts from most to least effective against the objective.

A. P3 > P5 > P1 > P4 > P2

B. P5 > P3 > P4 > P1 > P2

C. P3 > P1 > P5 > P4 > P2

D. P5 > P3 > P1 > P4 > P2

---

**Q13.** A retrieval pipeline pulls the relevant clause from a 40-page refund policy, and an LLM condenses it for a support agent mid-call. A typical retrieved clause reads: "Refunds are available within 30 days of delivery for unopened items; opened items qualify within 14 days only if defective, subject to inspection; items bought during clearance sales are excluded unless the defect is reported within 48 hours; refunds to cards issued outside India may take up to 21 business days."

**Objective:** the condensed version preserves every condition, exception and deadline in the clause.

**P1:** "Summarize this policy clause for a support agent. Keep every condition, exception and deadline; shorten the wording, but do not drop any qualifier."

**P2:** "Summarize this policy clause for a support agent."

**P3:** "Summarize this policy clause for a support agent in exactly 25 words, as one sentence, beginning with the refund window."

**P4:** "Summarize this policy clause for a support agent as a bulleted list, with one bullet per rule."

**P5:** "Summarize this policy clause for a support agent in plain, friendly language that a new hire can follow easily."

Rank the prompts from most to least effective against the objective.

A. P3 > P1 > P4 > P2 > P5

B. P1 > P4 > P2 > P5 > P3

C. P1 > P4 > P5 > P2 > P3

D. P4 > P1 > P2 > P5 > P3

---

**Q14.** A RAG assistant for an electronics retailer answers customer questions from the top four retrieved product-manual passages. Evaluation runs on a set of questions in which about a third have no answer anywhere in the retrieved passages; on those, any specific claim the model makes counts as unfaithful.

**Objective:** faithfulness — every claim in the answer is supported by the retrieved passages.

**P1:** "Answer the question. Do not make up facts that are not in the passages, such as prices, delivery dates or warranty terms."

**P2:** "Answer the question using only the passages below."

**P3:** "Answer the question using only the passages below. If they do not contain the answer, reply 'Not covered in the provided documents' and state what information is missing."

**P4:** "Answer the question. Do not hallucinate. Do not use outside knowledge. Do not guess."

**P5:** "Answer the question from the passages below; where they are silent, say so."

Rank the prompts from most to least effective against the objective.

A. P3 > P5 > P4 > P2 > P1

B. P4 > P3 > P5 > P2 > P1

C. P3 > P5 > P2 > P4 > P1

D. P3 > P2 > P5 > P1 > P4

---

**Q15.** A game studio generates hundreds of lines of ambient dialogue for a non-player character: Captain Rustom Mirza, a retired naval officer living in 1940s Bombay. Writers review the lines in batches, and the main complaint so far is that the character "sounds like a different person from one line to the next".

**Objective:** a consistent, recognisable voice for this character across all generated lines.

**P1:** "Write lines for this character. Use short, clipped sentences, draw on nautical metaphors, sound gruff, and avoid modern slang."

**P2:** "You are Captain Rustom Mirza, 64, retired from the Royal Indian Navy, living in 1940s Bombay. You are gruff but fond of the harbour children, speak in clipped sentences, reach for sea metaphors, and distrust anything newfangled. Speak as him."

**P3:** "Write dialogue for this character that stays consistent in tone across all lines and fits a 1940s setting."

**P4:** "Write lines for Captain Mirza. Keep each line under 12 words and include exactly one nautical term per line."

**P5:** "You are Captain Rustom Mirza, a gruff retired naval officer. Speak as him."

Rank the prompts from most to least effective against the objective.

A. P1 > P2 > P5 > P3 > P4

B. P2 > P5 > P3 > P1 > P4

C. P2 > P1 > P5 > P3 > P4

D. P2 > P5 > P1 > P3 > P4

---

**Q16.** A hybrid retriever combines dense and sparse scores for four candidate documents. Their raw BM25 scores against the query are:

| Doc | BM25 |
|---|---|
| d1 | 7.2 |
| d2 | 1.0 |
| d3 | 9.0 |
| d4 | 3.0 |

```python
import numpy as np

q = np.array([1.0, 0.0, 1.0])
emb = {"d1": np.array([1.0, 1.0, 1.0]),
       "d2": np.array([0.5, 0.0, 0.5]),
       "d3": np.array([1.0, 0.0, 0.0]),
       "d4": np.array([0.0, 1.0, 1.0])}
bm25 = {"d1": 7.2, "d2": 1.0, "d3": 9.0, "d4": 3.0}

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

lo, hi = min(bm25.values()), max(bm25.values())
alpha = 0.6
scores = {}
for d, v in emb.items():
    dense = cosine(q, v)
    sparse = (bm25[d] - lo) / (hi - lo)
    scores[d] = alpha * dense + (1 - alpha) * sparse

ranked = sorted(scores, key=scores.get, reverse=True)
print(ranked)
```

What does the script print?

A. `['d3', 'd1', 'd2', 'd4']`

B. `['d1', 'd3', 'd4', 'd2']`

C. `['d2', 'd1', 'd3', 'd4']`

D. `['d3', 'd1', 'd4', 'd2']`

---

**Q17.** A manufacturing group has rolled out a set of locally hosted language models across 40 business units: a contract summarizer for procurement, a maintenance-log assistant for plant engineers, a policy Q&A bot for HR and several others. The models run on the group's own servers, so the vendor receives no usage data. Six months in, unit heads report scattered problems — a summarizer that occasionally omits penalty clauses, an assistant that misreads a plant's shorthand — but nobody can say how often these happen, which units are affected, or whether anything was done about them. The governance board asks how feedback and post-deployment management should work from here.

Which approach should the board adopt?

A. Run a structured survey of all users at the three-month mark, then retrain on the aggregated findings.

B. Set up a standing channel where users flag issues, with triage, named owners and tracked responses.

C. Rely on automated drift and latency dashboards, since user reports are anecdotal and inconsistent.

D. Retrain monthly on all logged prompts and outputs, so feedback is captured implicitly from usage.

---

**Q18.** A lender audits its loan-approval model across two applicant groups. "Qualified" means the applicant went on to repay a comparable loan in a held-out historical cohort.

| Group | Applicants | Qualified | Approved | Approved and qualified |
|---|---|---|---|---|
| A | 1,000 | 600 | 500 | 480 |
| B | 1,000 | 400 | 500 | 380 |

The compliance lead's summary reads: "The model is fair: both groups are approved at exactly the same rate, so qualified applicants in each group have the same chance of approval."

Which statement correctly evaluates this summary?

A. Both parts hold, because equal approval rates imply equal approval chances for qualified applicants.

B. Demographic parity fails, since group A has 200 more qualified applicants than group B.

C. Demographic parity holds, but qualified A applicants are approved more often than qualified B.

D. Demographic parity holds, but qualified B applicants are approved more often than qualified A.

---

**Q19.** A customer-support agent can look up orders and issue refunds. In testing, a user asks "Where is order 88213?" The logs show the model calling `get_order_status` with `{"order_id": "88213"}` on step 0, then again with identical arguments on steps 1 through 5, after which the user receives the fallback apology. The tool itself returns correct data every time.

```python
import json

SYSTEM_PROMPT = ("You are a support agent. Use tools to look up facts. "
                 "Never guess order details.")
TOOLS = {"get_order_status": get_order_status,
         "issue_refund": issue_refund}

def run_agent(user_msg, max_steps=6):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}]
    for step in range(max_steps):
        reply = llm.chat(messages=messages, tools=TOOL_SCHEMAS)   # line 7
        log.info("step=%d tool_calls=%s", step, reply.tool_calls)
        if not reply.tool_calls:                                   # line 9
            return reply.content
        messages.append(reply.as_message())                        # line 10
        for call in reply.tool_calls:
            fn = TOOLS[call.name]
            args = json.loads(call.arguments)
            result = fn(**args)
            tool_msg = {"role": "tool", "tool_call_id": call.id,
                        "content": json.dumps(result)}
            history = messages + [tool_msg]                        # line 17
    return "Sorry, I couldn't complete that request."
```

Which marked line is responsible for the repeated calls?

A. Line 10, since appending the assistant's tool-call turn makes the model re-issue it.

B. Line 17, where the tool message is combined with the conversation.

C. Line 7, since passing tool schemas on every step re-invites the same call.

D. Line 9, since the loop can only exit when the model stops calling tools.

---

**Q20.** A national retailer uses a third-party LLM API to rank incoming CVs for store-manager roles; recruiters then interview the top 20%. The vendor's terms of service state that outputs are provided "as is". An internal audit finds that candidates from colleges in two particular states are shortlisted at less than half the rate of otherwise similar candidates. The HR director's position is that "the bias sits in the vendor's model, so it is the vendor's problem to fix."

Which statement best reflects sound AI governance?

A. The retailer remains accountable for its hiring outcomes and must audit, fix and own the process.

B. Accountability transfers to the vendor, since the bias originates in the vendor's training data.

C. Accountability is shared equally, as the API terms set up a joint-responsibility arrangement.

D. The HR director is personally liable, since she approved the tool without a bias review.

---

**Q21.** A financial-services firm's AI governance committee is assigning review tiers to five proposed use cases. The highest tier requires a documented impact assessment and meaningful human oversight before any output takes effect.

| ID | Use case | Who is affected | Does a person review before it takes effect? |
|---|---|---|---|
| U1 | Summarize internal meeting transcripts | Employees reading notes | Yes, attendees read the summary |
| U2 | Auto-reject insurance claims under ₹5,000 scored as likely fraud | Individual claimants | No |
| U3 | Suggest code completions in the IDE | Internal codebase | Yes, engineers accept or reject |
| U4 | Rank loan applicants and auto-set their credit limits | Individual applicants | No |
| U5 | Draft personalized marketing emails | Existing customers | Yes, marketing approves templates |

Which two use cases belong in the highest tier?

A. U1 and U5, since both produce text that staff or customers read at high volume.

B. U2 and U3, as both act directly on financial or production systems.

C. U4 and U5, since both reach external customers and shape their decisions.

D. U2 and U4, which set individual outcomes with no human review.

---

**Q22.** A telecom company's support chat collected transcripts under a notice saying the data would be "used to resolve your query and improve the quality of our support". The data science team now wants to train a churn-propensity model on those transcripts, including sentiment scores, to decide which customers receive retention discounts. They argue that keeping customers happier is part of good support.

Which position is most defensible?

A. Proceed, since "improve support quality" covers any model that improves the customer experience.

B. Proceed after stripping names and numbers, as de-identified transcripts need no new basis.

C. Treat it as a new purpose needing its own assessment, notice and lawful basis.

D. Proceed internally, because purpose limits apply to sharing data rather than internal reuse.

---

**Q23.** A speech-to-text vendor supplies a model card with its call-centre transcription model. A state government plans to use it for a rural agricultural helpline where most callers speak Bhojpuri, Maithili or strongly accented Hindi.

| Model card field | Content |
|---|---|
| Intended use | Customer-support call transcription |
| Training data | 4,000 hours of call-centre audio from six metro cities |
| Evaluation | Overall word error rate 8.2% on a held-out metro test set |
| Licence | Commercial, per-minute pricing |
| Contact | Vendor support email |

Before deployment, which gap in this model card matters most?

A. No error rates broken down by dialect or accent group.

B. No licence terms on transcripts, which govern how the state may reuse its call records.

C. No training energy or carbon figures, which responsible-AI reporting increasingly expects.

D. No architecture details, without which reviewers cannot judge the model's suitability.

---

**Q24.** An insurer routes every AI claim recommendation through a human reviewer, and its governance policy describes this as "human-in-the-loop oversight". Operations data for the first half of the year:

| Month | Recommendations | Reviewer overrides | Median review time |
|---|---|---|---|
| Jan | 12,400 | 310 | 41 s |
| Mar | 13,800 | 118 | 19 s |
| Jun | 15,900 | 32 | 7 s |

In June, an independent audit of 200 randomly sampled approved recommendations found that 9% should have been decided differently.

What does this evidence most directly indicate?

A. Model accuracy rose steadily, so reviewers rightly needed less time to confirm each output.

B. Reviewers are overloaded, so adding staff would by itself restore January's override rate.

C. The oversight has become nominal, with reviewers largely deferring to the model.

D. The audit figure is likely sampling noise, as 200 claims is too small to rely on.

---

**Q25.** A feature-pooling step from a sensor-classification model is shown below. Trace the shapes.

```python
import torch

x = torch.randn(8, 3, 32)        # (batch, channels, length)
w = torch.randn(32, 16)
b = torch.randn(16)

h = x @ w + b
h = h.transpose(1, 2)
s = h.mean(dim=-1, keepdim=True)
g = torch.softmax(h - s, dim=1)
pooled = (g * h).sum(dim=1)
out = pooled.unsqueeze(0).repeat(2, 1, 1).flatten(1)
```

What are the shapes of `pooled` and `out`?

A. `pooled` is (8, 3); `out` is (2, 24)

B. `pooled` is (8, 16); `out` is (2, 128)

C. `pooled` is (8, 3); `out` is (16, 3)

D. `pooled` is (8, 16); `out` is (2, 24)

---

**Q26.** An e-learning company recorded a trainer's voice to build a synthetic voice model used to narrate its compliance courses. At hiring, she signed a form consenting to "use of my likeness and voice for company purposes". She has since left the company and written to ask that her voice no longer be used. The company has continued generating new courses with the synthetic voice, arguing that the clips are newly synthesized rather than recordings of her.

What is the central ethical problem?

A. None, since the company owns work product created during her employment, including her voice.

B. The consent was too broad and not revocable after she left.

C. Mainly disclosure; labelling the narration as synthetic would fully resolve the concern.

D. None, as the clips are newly synthesized and contain no recording of her.

---

**Q27.** A five-class tabular classifier is trained on a machine with one CUDA GPU. `train_loader` yields `xb` as float32 of shape (64, 40) and `yb` as int64 of shape (64,). When the script is run, it fails with a runtime error during the first training batch.

```python
import torch
import torch.nn as nn

device = torch.device("cuda")

class Classifier(nn.Module):
    def __init__(self, n_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_cls), nn.Softmax(dim=1))          # line 10
    def forward(self, x):
        return self.net(x)

model = Classifier(40, 5).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()                                               # line 19
    total = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)                              # line 24
        loss.backward()
        opt.step()
        total += loss                                           # line 27
    model.eval()
    with torch.no_grad():                                       # line 29
        val_acc = evaluate(model, val_loader)
    print(epoch, total / len(train_loader), val_acc)
```

The script contains several bugs. Which one stops it from running?

A. Line 10, since CrossEntropyLoss rejects inputs that are already normalized probabilities.

B. Line 27, since accumulating loss tensors exhausts GPU memory within the first batch.

C. Line 24, computing the loss with `yb` on the CPU but logits on the GPU.

D. Line 29, since `no_grad` disables autograd for the next epoch's backward pass.

---

**Q28.** A forecasting team wraps a multivariate series of 1,000 time steps with 3 features each in the dataset below and iterates over one epoch.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class WindowDataset(Dataset):
    def __init__(self, series, window=24, horizon=6):
        self.x = torch.tensor(series, dtype=torch.float32)   # (1000, 3)
        self.window, self.horizon = window, horizon

    def __len__(self):
        return len(self.x) - self.window - self.horizon + 1

    def __getitem__(self, i):
        x = self.x[i : i + self.window]
        y = self.x[i + self.window : i + self.window + self.horizon, 0]
        return x, y

ds = WindowDataset(series)
loader = DataLoader(ds, batch_size=64, shuffle=True)

for step, (xb, yb) in enumerate(loader):
    pass
```

How many batches does one epoch produce, and what is the shape of `xb` in the final batch?

A. 15 batches; the final `xb` is (64, 24, 3)

B. 16 batches; the final `xb` is (11, 24, 3)

C. 16 batches; the final `xb` is (16, 24, 3)

D. 16 batches; the final `xb` is (11, 3, 24)

---

**Q29.** An engineer adapts a pretrained ResNet-18 to a four-class defect-detection task, intending to train only the new classification head. When the script runs, it raises a runtime error.

```python
import torch
import torch.nn as nn
import torchvision

model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 4)                    # line 5
for p in model.parameters():                                     # line 6
    p.requires_grad = False
opt = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)  # line 8-9
loss_fn = nn.CrossEntropyLoss()

for images, labels in train_loader:
    logits = model(images)
    loss = loss_fn(logits, labels)
    opt.zero_grad()
    loss.backward()                                              # line 16
    opt.step()
```

Which error is raised, and where?

A. A shape error at line 5, as the head inherits ImageNet's 1,000-class input.

B. A RuntimeError at line 16, as no tensor in the graph requires grad.

C. A RuntimeError at line 6, since requires_grad cannot be changed on pretrained weights.

D. A ValueError at lines 8–9, as the optimizer receives an empty parameter list.

---

**Q30.** A sentiment model is trained on a GPU and checkpointed. In the evaluation notebook it gives stable predictions, but the CPU serving code below returns slightly different probabilities each time the same text is submitted.

```python
import torch
from model import SentimentNet          # embeddings -> LSTM -> dropout(0.3) -> linear
from tokenize_utils import tokenizer

model = SentimentNet(vocab_size=30000, emb=128, hidden=256, n_cls=3)
state = torch.load("ckpt/best.pt", map_location="cpu")
model.load_state_dict(state["model"])

@torch.no_grad()
def predict(text):
    ids = tokenizer(text)                       # LongTensor (seq_len,)
    logits = model(ids.unsqueeze(0))
    return logits.softmax(-1).squeeze(0).tolist()
```

Which single change makes the served predictions match the notebook's?

A. Call `torch.manual_seed(0)` inside `predict`, as dropout uses the global RNG.

B. Call `model.eval()` once, directly after `load_state_dict`.

C. Remove `map_location`, so weights load onto their saved device.

D. Replace `@torch.no_grad()` with `torch.inference_mode()`, which also switches dropout off.

---

**Q31.** A platform team audits how five production models pair their final layer with a loss function.

| Model | Task | Final layer output | Loss and targets |
|---|---|---|---|
| T1 | 5-class product category, one label each | raw logits (N, 5) | `CrossEntropyLoss`, int64 targets (N,) |
| T2 | 12 independent content tags per item | raw logits (N, 12) | `BCEWithLogitsLoss`, float multi-hot targets (N, 12) |
| T3 | Binary churn flag | `sigmoid` probabilities (N, 1) | `BCEWithLogitsLoss`, float targets (N, 1) |
| T4 | 12 independent content tags per item | `softmax` over 12 (N, 12) | `CrossEntropyLoss`, float multi-hot targets (N, 12) |
| T5 | Delivery time in minutes | raw output (N, 1) | `MSELoss`, float targets (N,) |

Which models are correctly specified?

A. T1, T2 and T5, as T5's targets broadcast to the output shape.

B. T1, T3 and T5, as sigmoid outputs suit any binary loss.

C. T2 and T4, as both use multi-hot float targets for tagging.

D. T1 and T2, with T3, T4 and T5 each mis-specified.

---

**Q32.** A small image classifier is defined layer by layer as follows.

| Order | Layer |
|---|---|
| 1 | `nn.Conv2d(3, 16, kernel_size=3, padding=1)` |
| 2 | `nn.BatchNorm2d(16)` |
| 3 | `nn.ReLU()` |
| 4 | `nn.MaxPool2d(2)` |
| 5 | `nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)` |
| 6 | `nn.BatchNorm2d(32)` |
| 7 | `nn.ReLU()` |
| 8 | `nn.AdaptiveAvgPool2d(1)` |
| 9 | `nn.Flatten()` |
| 10 | `nn.Linear(32, 10)` |

What does `sum(p.numel() for p in model.parameters() if p.requires_grad)` return?

A. 5,514

B. 5,386

C. 5,482

D. 5,578

---

**Q33.** A broadband provider defines churn as follows: a customer active on the snapshot date of 31 March 2026 is labelled `churned = 1` if they cancel at any point in the 90 days after that date. The features below are built for every customer active on 31 March. The `customers` table was exported from the live CRM on 1 October 2026; the `events` table is an append-only log with one row per event. Five-fold cross-validated ROC-AUC comes out at 0.97, but when the model scores the next quarter's customers in production, its AUC is 0.68.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

snap = pd.Timestamp("2026-03-31")
df = customers[customers.status_on(snap) == "active"].copy()

df["tenure_m"] = (snap - df.signup_date).dt.days / 30                     # line 9
calls = events[(events.type == "support") &
               (events.date <= snap) &
               (events.date > snap - pd.Timedelta(days=90))]
df["calls_90d"] = df.id.map(calls.groupby("cust_id").size()).fillna(0)    # line 13
df["login_gap"] = (snap - df.last_login_date).dt.days                     # line 14
df["plan_price"] = df.plan.map(PRICE_TABLE)

X = df[["tenure_m", "calls_90d", "login_gap", "plan_price"]]
y = df.churned
Xs = StandardScaler().fit_transform(X)                                    # line 19
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)            # line 20
print(cross_val_score(LogisticRegression(max_iter=1000), Xs, y,
                      cv=cv, scoring="roc_auc").mean())
```

Which line best explains the gap between cross-validated and production performance?

A. Line 19, since fitting the scaler on all rows leaks every fold's statistics into training.

B. Line 20, since shuffling mixes customers across folds so each fold sees the others.

C. Line 13, since counting support calls inside the label window reveals who later cancels.

D. Line 14, as `last_login_date` comes from a CRM table exported after the label window had closed.

---

**Q34.** A churn model is trained on 1,000 customers, each described by 64 features.

```python
from sklearn.ensemble import RandomForestClassifier

print(X.shape)            # (1000, 64)
rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=7,
)
rf.fit(X, y)
print(rf.oob_score_)
```

For any single tree in the forest, about how many of the 1,000 customers are out-of-bag, and how many features are considered as split candidates at each node?

A. About 368 customers out-of-bag; 8 candidate features per split.

B. About 632 customers out-of-bag, as bootstrap retains 63.2%; 8 per split.

C. About 368 customers out-of-bag; all 64 features, as sqrt applies per tree.

D. About 632 customers out-of-bag; all 64 features at every single split.

---

**Q35.** A logistics team clusters eight delivery depots by their (x, y) map coordinates using a hand-written k-means loop.

```python
import numpy as np

X = np.array([[1, 1], [2, 1], [1, 2],
              [8, 8], [9, 8], [8, 9],
              [4, 5], [5, 4]], dtype=float)
C = np.array([[2, 2], [6, 6]], dtype=float)     # initial centroids

for _ in range(2):
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)
    labels = d.argmin(axis=1)
    C = np.array([X[labels == k].mean(axis=0) for k in range(2)])

print(C.round(2))
```

What does the script print?

A. `[[2.6 2.6] [8.33 8.33]]`

B. `[[1.33 1.33] [8.33 8.33]]`

C. `[[1.33 1.33] [6.8 6.8]]`

D. `[[2. 2.] [6. 6.]]`

---

**Q36.** Three runs of the same image classifier are compared. Each cell shows training loss / validation loss at the end of the epoch.

| Epoch | Run 1 (lr 1e-3, no regularization) | Run 2 (lr 1e-5) | Run 3 (lr 0.5) |
|---|---|---|---|
| 1 | 1.82 / 1.85 | 2.29 / 2.29 | 2.10 / 2.15 |
| 2 | 1.10 / 1.21 | 2.27 / 2.28 | 3.84 / 4.02 |
| 4 | 0.48 / 0.82 | 2.24 / 2.24 | 1.93 / 2.05 |
| 6 | 0.21 / 0.97 | 2.21 / 2.22 | 7.40 / 7.71 |
| 8 | 0.06 / 1.24 | 2.19 / 2.19 | nan / nan |

Which diagnosis is correct?

A. Run 1 overfits after epoch 4; Run 3 diverges, so its learning rate is clearly too high.

B. Run 1 overfits after epoch 4; Run 2 overfits too, as its validation loss matches training.

C. Run 3 diverges from exploding activations, so adding dropout would stabilize its training.

D. Run 2 is best, since matching training and validation losses show strong generalization.

---

**Q37.** A telecom retention team calls customers whom the churn model flags. "Churn" means cancelling within the next 30 days. Below are the scores for a validation sample of twelve customers, sorted by score, together with whether each actually churned. The team currently flags customers at a threshold of 0.5.

```python
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
y_prob = np.array([0.92, 0.81, 0.66, 0.41, 0.38, 0.36,
                   0.33, 0.29, 0.22, 0.18, 0.12, 0.05])

flag = (y_prob >= 0.35).astype(int)
```

The retention lead proposes moving the threshold to 0.35, claiming: "We'll catch more churners and the call list will be more accurate." Which statement correctly evaluates the claim?

A. Recall rises to 0.60 and precision rises to 0.67, so the whole claim holds.

B. Recall rises to 0.60 while precision falls to 0.50, so only the claim's first half holds.

C. Recall rises to 0.50 while precision falls to 0.60, so only the first half holds.

D. Recall rises to 0.60 while precision stays at 0.67, so the whole claim holds.

---

**Q38.** A streaming service predicts whether a customer will choose its premium plan from annual income (₹) and age. A k-nearest-neighbours model is fitted twice: once on the raw features and once with standardization.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = np.array([[52000, 24], [54000, 58], [61000, 61], [49000, 22],
              [90000, 27], [88000, 26], [47000, 63], [95000, 60]], dtype=float)
y = np.array(["basic", "premium", "premium", "basic",
              "basic", "basic", "premium", "premium"])
q = np.array([[53000, 60]], dtype=float)

raw = KNeighborsClassifier(n_neighbors=3).fit(X, y)
scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)).fit(X, y)
print(raw.predict(q)[0], scaled.predict(q)[0])
```

What does the script print?

A. `basic premium`

B. `premium premium`

C. `basic basic`

D. `premium basic`

---

**Q39.** A retail analytics team has four requests on its backlog.

| ID | Request | Data available |
|---|---|---|
| S1 | Group 600 stores into a handful of segments by weekly sales profile for regional planning | Sales histories; no segment labels exist |
| S2 | When an adjuster opens a new insurance claim, show the five most similar past claims | 40,000 past claims with numeric features |
| S3 | Predict which loyalty members will lapse, as rules the store staff can read and act on | 18 months of labelled member history |
| S4 | Classify 5,000 short supplier emails as invoice or not invoice | Labelled emails as 20,000 sparse TF-IDF features |

Which assignment of methods is most appropriate?

A. S1 k-NN, S2 k-means, S3 decision tree, S4 linear SVM

B. S1 k-means, S2 k-NN, S3 linear SVM, S4 decision tree

C. S1 k-means, S2 k-NN, S3 decision tree, S4 linear SVM

D. S1 k-means, S2 decision tree, S3 k-NN, S4 linear SVM

---

**Q40.** A team tunes an RBF-kernel SVM with 5-fold grid search on its training data. The mean cross-validated accuracies are below; the team reports 0.94 as the model's expected accuracy. On a held-out test set never used during tuning, the selected model scores 0.89. For reference, the configuration C = 100, γ = 1 reached 1.00 training accuracy.

| | γ = 0.01 | γ = 0.1 | γ = 1 |
|---|---|---|---|
| C = 0.1 | 0.78 | 0.83 | 0.80 |
| C = 1 | 0.86 | 0.91 | 0.85 |
| C = 10 | 0.89 | 0.94 | 0.83 |
| C = 100 | 0.90 | 0.92 | 0.81 |

Which statement is correct?

A. The test set is likely unrepresentative, since a tuned cross-validation score is unbiased.

B. Picking the best of twelve configurations on the same folds biased 0.94 upward.

C. C = 100 with γ = 1 underfits, as a wide margin lowers its cross-validated accuracy.

D. Refitting with C = 100, γ = 1 would close the gap, as it fits training data perfectly.

---

**Q41.** A team writes an LSTM cell from scratch in NumPy for an on-device keyword spotter. One line was lost in a merge.

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lstm_cell(x, h_prev, c_prev, W, U, b, H):
    # W: (4H, D), U: (4H, H), b: (4H,) ; gate blocks ordered i, f, g, o
    z = W @ x + U @ h_prev + b
    i = sigmoid(z[0:H])
    f = sigmoid(z[H:2*H])
    g = np.tanh(z[2*H:3*H])
    o = sigmoid(z[3*H:4*H])
    # ---- missing line ----
    h = o * np.tanh(c)
    return h, c
```

Which line completes the cell correctly?

A.
```python
c = i * c_prev + f * g
```

B.
```python
c = f * c_prev + o * g
```

C.
```python
c = f * c_prev + i * g
```

D.
```python
c = f * c_prev + i * np.tanh(g)
```

---

**Q42.** A GRU cell is implemented using the same conventions as PyTorch's `nn.GRU`. One line is missing.

```python
import numpy as np

def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def gru_cell(x, h_prev, P):
    z = sigmoid(P.Wz @ x + P.Uz @ h_prev + P.bz)
    # ---- missing line ----
    n = np.tanh(P.Wn @ x + P.bn + r * (P.Un @ h_prev + P.bhn))
    h = (1 - z) * n + z * h_prev
    return h
```

Which line completes the cell?

A.
```python
r = sigmoid(P.Wr @ x + P.Ur @ h_prev + P.br)
```

B.
```python
r = np.tanh(P.Wr @ x + P.Ur @ h_prev + P.br)
```

C.
```python
r = sigmoid(P.Wr @ x + P.Ur @ (z * h_prev) + P.br)
```

D.
```python
r = sigmoid(P.Wr @ x + P.Ur @ n + P.br)
```

---

**Q43.** Four tree-based models are trained on the same fraud dataset and evaluated on the same validation split.

| Model | Train error | Validation error |
|---|---|---|
| Single tree, max depth 2 | 0.24 | 0.25 |
| Single tree, max depth 20 | 0.01 | 0.22 |
| Bagging: 300 trees, max depth 20 | 0.02 | 0.14 |
| Boosting: 300 trees, max depth 2 | 0.09 | 0.12 |

Which statement is supported by these results?

A. Bagging depth-2 trees would match boosting, since averaging many models also cuts bias.

B. Boosting depth-20 trees would beat both, as stronger base learners lower bias further.

C. The depth-20 tree's gap reflects high bias, which bagging resolves through resampling.

D. Bagging cut the deep tree's variance; boosting cut the shallow tree's bias.

---

**Q44.** While refactoring an LSTM step, a developer pasted its five computation lines in the wrong order. They are numbered below; `x`, `h_prev`, `c_prev` and all weights are already defined.

```python
# (1)
h = o * np.tanh(c)
# (2)
f = sigmoid(Wf @ x + Uf @ h_prev + bf)
# (3)
c = f * c_prev + i * g
# (4)
o = sigmoid(Wo @ x + Uo @ h_prev + bo)
# (5)
i, g = sigmoid(Wi @ x + Ui @ h_prev + bi), np.tanh(Wg @ x + Ug @ h_prev + bg)
```

Which execution order computes a correct LSTM step?

A. 2 → 5 → 4 → 1 → 3

B. 5 → 2 → 3 → 4 → 1

C. 5 → 3 → 2 → 4 → 1

D. 4 → 2 → 1 → 5 → 3

---

**Q45.** A crop-disease classifier takes 64 × 64 RGB leaf images. The model below fails at the first `forward` call because `IN` is set incorrectly.

```python
import torch.nn as nn

IN = ...   # to be determined

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=5),                         # no padding
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(IN, 10),
)
```

What value of `IN` lets a batch of shape (N, 3, 64, 64) pass through?

A. 4096

B. 14400

C. 1024

D. 3136

---

**Q46.** A two-weight network is checked by hand against autograd.

```python
import torch

x = torch.tensor(2.0)
y = torch.tensor(1.0)

w1 = torch.tensor(0.5, requires_grad=True)
b1 = torch.tensor(-0.5, requires_grad=True)
w2 = torch.tensor(-1.5, requires_grad=True)
b2 = torch.tensor(1.0, requires_grad=True)

h = torch.relu(w1 * x + b1)
y_hat = w2 * h + b2
loss = (y_hat - y) ** 2
loss.backward()

print(w1.grad.item(), w2.grad.item())
```

What does the script print?

A. dL/dw1 = −0.75, dL/dw2 = 4.5

B. dL/dw1 = 2.25, dL/dw2 = −0.75

C. dL/dw1 = 4.5, dL/dw2 = −0.75

D. dL/dw1 = −4.5, dL/dw2 = −0.75

---

**Q47.** A developer checks how dropout behaves in each mode.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.ones(1, 8) * 3.0
drop = nn.Dropout(p=0.25)

drop.train()
a = drop(x)
drop.eval()
b = drop(x)

print(sorted(set(a.flatten().tolist())), b.flatten().tolist()[:2])
```

With this seed, at least one element is dropped and at least one is kept. What is printed?

A. Training values {0.0, 4.0}; eval output [3.0, 3.0]

B. Training values {0.0, 3.0}; eval output [2.25, 2.25]

C. Training values {0.0, 4.0}; eval output [2.25, 2.25]

D. Training values {0.0, 3.0}; eval output [3.0, 3.0]

---

**Q48.** Four candidate downsampling layers are evaluated on a 32 × 32 feature map. Output size is floor((n + 2p − k) / s) + 1.

| Config | Kernel k | Stride s | Padding p |
|---|---|---|---|
| C1 | 3 | 2 | 0 |
| C2 | 3 | 2 | 1 |
| C3 | 5 | 2 | 1 |
| C4 | 4 | 2 | 0 |

Which configuration or configurations produce a 16 × 16 output?

A. C1, since stride 2 halves any even-sized 32-pixel input to exactly 16.

B. C2 only, with every other configuration giving 15 × 15.

C. C3, as its wider kernel is offset by a padding of 1 on each side.

D. C2 and C4, as both apply stride 2 to an even-sized input map.

---

**Q49.** A team trains a vanilla tanh RNN and an LSTM on 60-step sequences, with the loss computed only at the final step. They log the gradient norm reaching the hidden state at earlier time steps.

| Time step t | Vanilla RNN ‖∂L/∂hₜ‖ | LSTM ‖∂L/∂hₜ‖ |
|---|---|---|
| 60 | 1.00 | 1.00 |
| 50 | 0.21 | 0.82 |
| 40 | 0.03 | 0.71 |
| 20 | 4 × 10⁻⁴ | 0.55 |
| 1 | 2 × 10⁻⁸ | 0.41 |

Which statement best explains the difference?

A. The RNN's gradients shrink because tanh saturates at the output layer used at the final step.

B. Gradient clipping would restore the RNN's early-step gradients, since clipping rescales small norms upward.

C. The LSTM's additive cell-state path, with forget gates near 1, lets gradients reach early steps largely intact.

D. The LSTM avoids decay because its sigmoid gates have derivatives that never fall below 0.25.

---

**Q50.** A churn model's analyst computes permutation importance (drop in validation AUC when a feature is shuffled) and also a drop-column check (retrain without the features, measure the AUC drop). `tenure_months` and `tenure_days` have a correlation of 0.98.

| Feature(s) | Permutation importance | Drop-column AUC loss |
|---|---|---|
| tenure_months | 0.004 | — |
| tenure_days | 0.005 | — |
| monthly_spend | 0.061 | 0.048 |
| region (14 levels) | 0.012 | 0.010 |
| tenure_months + tenure_days together | — | 0.090 |

Which conclusion is supported?

A. The two tenure features mask each other; together they carry the most signal.

B. Both tenure features can be dropped, since each contributes under 0.01 AUC.

C. Region is overstated, as permutation importance favours high-cardinality features.

D. Monthly spend causes churn, since its permutation importance is the largest.

---

**Q51.** A document-classification model is reviewed before launch. Its predictions on 2,000 validation documents are grouped by the confidence of the top class.

| Confidence bin | Mean confidence | Accuracy in bin | Documents |
|---|---|---|---|
| 0.5–0.6 | 0.55 | 0.52 | 100 |
| 0.6–0.7 | 0.65 | 0.58 | 150 |
| 0.7–0.8 | 0.75 | 0.66 | 250 |
| 0.8–0.9 | 0.85 | 0.72 | 500 |
| 0.9–1.0 | 0.97 | 0.85 | 1,000 |

The product owner wants to auto-file any document above 0.9 confidence, arguing that "a 0.97 confidence means it is right 97% of the time." Which statement about the model's expected calibration error (ECE, count-weighted) and behaviour is correct?

A. ECE ≈ 0.088, overconfident, from the mean gap across the five bins.

B. ECE ≈ 0.11, with the model underconfident in all five bins.

C. ECE ≈ 0.11, yet calibrated overall, as the top bin reaches 85%.

D. ECE ≈ 0.11, with the model overconfident in all five bins.

---

**Q52.** A fully connected network for MNIST has the layers below. The team is choosing weight initializations.

| Layer | Fan-in → fan-out | Activation after layer |
|---|---|---|
| L1 | 784 → 512 | ReLU |
| L2 | 512 → 256 | ReLU |
| L3 | 256 → 10 | none (logits) |

Which statement is correct?

A. L2 under He normal has std ≈ 0.051, as He averages fan-in and fan-out for ReLU.

B. L3 under Xavier uniform has limit ≈ 0.153, computed from its fan-in of 256 alone.

C. All weights can start at zero here, since ReLU itself breaks symmetry between units.

D. L2 under He normal has std 0.0625, set by its fan-in of 512 and the ReLU gain of 2.

---

**Q53.** A credit model outputs a log-odds score, and an applicant is approved if the predicted probability of repayment is at least 0.5. For one applicant, the SHAP explanation in log-odds units is:

| Component | Value |
|---|---|
| Base value (expected model output) | −1.2 |
| Monthly income | +0.9 |
| Credit utilization | −1.1 |
| Late payments, last 12 months | −0.8 |
| Years at current employer | +0.4 |

Which statement about this applicant is correct?

A. p ≈ 0.86, approved, as negative SHAP values lower the rejection risk.

B. p ≈ 0.14, rejected; credit utilization pushes hardest toward rejection of all four features.

C. p ≈ 0.35, rejected; credit utilization pushes hardest toward rejection overall.

D. p ≈ 0.14, rejected; late payments push hardest, being a recurring risk.

---

**Q54.** A lender monitors the income distribution of incoming applicants against the training data using the population stability index, PSI = Σ (actual% − expected%) × ln(actual% / expected%), with percentages as proportions. Its policy reads PSI below 0.10 as stable, 0.10 to 0.25 as a moderate shift needing investigation, and above 0.25 as a significant shift requiring retraining review.

| Monthly income | Training share | Last month's share |
|---|---|---|
| Below ₹20k | 10% | 25% |
| ₹20k–40k | 30% | 35% |
| ₹40k–60k | 35% | 25% |
| Above ₹60k | 25% | 15% |

Which statement is correct?

A. PSI ≈ 0.40, a significant shift, from summed absolute bin differences.

B. PSI ≈ 0.23, a significant shift, as two bins moved 10+ points.

C. PSI ≈ 0.23, a moderate shift needing investigation.

D. PSI ≈ 0.15, a moderate shift, from the two lowest-income bins.

---

**Q55.** A traffic-sign classifier reaches 97% accuracy on clean test images and 11% under an FGSM attack with ε = 4/255. The team adds a defence: each input passes through JPEG compression and a non-differentiable quantization step before the model. Re-running the same FGSM attack against the defended model gives 85% accuracy. An external red team then runs an adaptive attack that approximates the preprocessing's gradient, and accuracy falls to 9%.

What is the most accurate conclusion?

A. The defence is effective, since accuracy under FGSM rose from 11% to 85%.

B. The defence mainly obscures the attack's gradients rather than making the model itself more robust.

C. The adaptive attack is unrealistic, as attackers cannot see the preprocessing pipeline.

D. An ε of 4/255 is too large to matter, as such changes are visible to drivers.

---

**Q56.** A pneumonia classifier trained on chest X-rays from Hospital A reaches AUC 0.94 on Hospital A's held-out data but 0.71 at a partner hospital. Grad-CAM heatmaps for positive predictions at Hospital A concentrate on the top-right corner, where Hospital A's portable scanners stamp a "PORTABLE" marker. Portable scans are taken mostly of bedridden, sicker patients; the partner hospital's scanners add no such marker.

What is the most appropriate conclusion and next step?

A. Grad-CAM is unreliable on X-rays, so the heatmaps should be set aside for this decision.

B. The partner's images are likely lower quality, so the model needs more training epochs.

C. The marker is a valid clinical signal, since portable scans really do flag sicker patients.

D. The model has likely learned a shortcut; mask the marker region and re-evaluate both sites.

---

**Q57.** An insurer auto-decides claims when the model's top-class confidence exceeds a threshold and routes the rest to human adjusters. Validation results (in-distribution) are below. The insurer receives 10,000 claims a day and requires at least 98% accuracy on automated decisions, with as much automation as that allows.

| Threshold | Coverage (share auto-decided) | Accuracy on auto-decided |
|---|---|---|
| 0.50 | 100% | 93.1% |
| 0.70 | 91% | 95.8% |
| 0.80 | 84% | 97.4% |
| 0.85 | 78% | 98.1% |
| 0.90 | 69% | 98.9% |
| 0.95 | 52% | 99.4% |

Which threshold should be chosen, and how many erroneous automated decisions per day should be expected?

A. Threshold 0.85, with about 148 erroneous automated decisions per day.

B. Threshold 0.80, about 218 errors per day, as 97.4% rounds up to 98%.

C. Threshold 0.90, about 76 errors per day, keeping a margin above 98%.

D. Threshold 0.85, about 190 errors per day, from 1.9% of all 10,000 claims.

---

**Q58.** A loan applicant was rejected (monthly income ₹42k, credit utilization 78%, age 29, one year at current address). The bank's explainability tool proposes four counterfactuals, and the bank wants to show the applicant one that is valid (it actually flips the decision), actionable and as small a change as possible.

| Counterfactual | Change from the actual application | Model decision |
|---|---|---|
| CF1 | Age 29 → 41 | Approved |
| CF2 | Utilization 78% → 45% and income ₹42k → ₹60k | Approved |
| CF3 | Utilization 78% → 55% | Approved |
| CF4 | Utilization 78% → 70% | Rejected |

Which counterfactual should be shown?

A. CF1, as a single-feature change is the sparsest possible explanation.

B. CF2, as changing two levers gives the most robust path to approval.

C. CF3, which flips the decision through one actionable change.

D. CF4, as it requires the smallest change of all four candidates.

---

**Q59.** A logistics company flags near-duplicate customer complaints using word-trigram Jaccard similarity.

```python
docs = {
    "A": "the delivery was late and the package was damaged on arrival",
    "B": "the package was damaged on arrival and the delivery was late",
    "C": "the delivery was late but the package arrived in perfect condition",
}

def trigrams(text):
    w = text.split()
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}

def jaccard(a, b):
    return len(a & b) / len(a | b)

for p, q in [("A", "B"), ("A", "C"), ("B", "C")]:
    print(p, q, round(jaccard(trigrams(docs[p]), trigrams(docs[q])), 3))
```

Which statement about the output is correct?

A. A and B score 1.0, since the two complaints contain exactly the same words.

B. A–B scores 0.5, and C scores 0.125 against both A and B.

C. A–B scores 0.5, and A–C 0.25, as both open identically.

D. A–C scores highest, as the two share their entire opening clause.

---

**Q60.** A support team weights words in tickets with a hand-written TF-IDF, where tf is the term's count divided by the document's length, and idf = log₁₀(N / df).

```python
import math

docs = ["refund not processed refund delayed",
        "battery drains fast battery overheats",
        "refund processed quickly",
        "battery fine screen cracked",
        "screen cracked on delivery"]
toks = [d.split() for d in docs]
N = len(docs)

def tfidf(word, i):
    tf = toks[i].count(word) / len(toks[i])
    df = sum(word in t for t in toks)
    return tf * math.log10(N / df)

d1 = {w: round(tfidf(w, 0), 3) for w in set(toks[0])}
print(max(d1, key=d1.get), max(d1.values()))
```

What does the script print?

A. "not", ≈ 0.14, having the highest IDF of any word in d1.

B. "refund", ≈ 0.40, as it appears twice among five tokens.

C. "processed", ≈ 0.08, as it occurs in two of the documents.

D. "refund", ≈ 0.16, as its doubled term frequency outweighs the two rarer words.

---

**Q61.** A multinomial Naive Bayes spam filter was trained on 40 spam and 60 ham messages over a six-word vocabulary. It uses Laplace smoothing (α = 1).

```python
import math

V = ["free", "win", "meeting", "offer", "report", "today"]
counts = {
    "spam": {"free": 20, "win": 15, "meeting": 2,  "offer": 12, "report": 1,  "today": 10},
    "ham":  {"free": 3,  "win": 1,  "meeting": 25, "offer": 4,  "report": 20, "today": 27},
}
prior = {"spam": 0.4, "ham": 0.6}
msg = ["free", "meeting", "today"]

def log_post(c):
    total = sum(counts[c].values())
    return math.log(prior[c]) + sum(
        math.log((counts[c][w] + 1) / (total + len(V))) for w in msg)

ls, lh = log_post("spam"), log_post("ham")
print(1 / (1 + math.exp(lh - ls)))
```

How is the message classified, and what does the script print?

A. Ham, with P(spam | message) ≈ 0.26 under the 0.4 / 0.6 priors.

B. Spam, with P(spam | message) ≈ 0.74, as spam words outnumber ham words.

C. Ham, with P(spam) ≈ 0.34, as the 0.4 / 0.6 priors cancel out.

D. Ham, with P(spam) ≈ 0.24, as smoothing changes nothing for seen words.

---

**Q62.** A product team builds Bag-of-Words features for a review-sentiment model.

```python
import re

STOP = {"the", "is", "a", "an", "and", "not", "no", "was", "it", "this", "very"}

def preprocess(text):
    tokens = re.findall(r"[a-z']+", text.lower())
    return [w for w in tokens if w not in STOP]

reviews = ["The battery is not good",
           "The battery is very good",
           "No complaints, battery good"]

vocab = sorted({w for r in reviews for w in preprocess(r)})

def bow(text):
    v = [0] * len(vocab)
    for w in preprocess(text):
        v[vocab.index(w)] += 1
    return v

print(vocab)
print([bow(r) for r in reviews])
```

Which statement about the output is correct?

A. Review 1 becomes [1, 1, 0, 1], keeping "not", so it differs from review 2.

B. Reviews 1 and 3 both become [1, 0, 1], as preprocessing strips both negations.

C. Reviews 1 and 2 both become [1, 0, 1], while review 3 instead becomes [1, 1, 1].

D. Reviews 1 and 2 differ, as "very" survives as an intensifier in review 2.

---

**Q63.** An NER model emits BIO tags, and the helper below converts them to entity spans. The model's tagging of one sentence contains some inconsistent tags.

```python
def spans(tokens, tags):
    out, cur, typ = [], [], None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if cur:
                out.append((" ".join(cur), typ))
            cur, typ = [tok], tag[2:]
        elif tag.startswith("I-") and typ == tag[2:]:
            cur.append(tok)
        else:
            if cur:
                out.append((" ".join(cur), typ))
            cur, typ = [], None
    if cur:
        out.append((" ".join(cur), typ))
    return out

tokens = ["Ravi", "Sinha", "visited", "Patna", "and", "Bodh", "Gaya",
          "with", "Google", "Cloud", "staff"]
tags   = ["B-PER", "I-PER", "O", "B-LOC", "O", "I-LOC", "I-LOC",
          "O", "B-ORG", "I-LOC", "O"]
print(spans(tokens, tags))
```

What does the script print?

A. `[('Ravi Sinha', 'PER'), ('Patna', 'LOC'), ('Bodh Gaya', 'LOC'), ('Google Cloud', 'ORG')]`

B. `[('Ravi Sinha', 'PER'), ('Patna', 'LOC'), ('Bodh Gaya', 'LOC'), ('Google', 'ORG')]`

C. `[('Ravi Sinha', 'PER'), ('Patna', 'LOC'), ('Google Cloud', 'ORG')]`

D. `[('Ravi Sinha', 'PER'), ('Patna', 'LOC'), ('Google', 'ORG')]`

---

**Q64.** A toy Word2Vec model has learned the 3-dimensional embeddings below. An analogy query computes v = king − man + woman and returns the vocabulary word with the highest cosine similarity to v, excluding the three query words.

| Word | Embedding |
|---|---|
| king | (0.80, 0.65, 0.10) |
| man | (0.70, 0.10, 0.05) |
| woman | (0.10, 0.15, 0.75) |
| queen | (0.35, 0.85, 0.55) |
| prince | (0.75, 0.55, 0.20) |
| princess | (0.20, 0.60, 0.75) |
| monarch | (0.55, 0.75, 0.40) |

Which word does the query return?

A. queen, with cosine ≈ 0.95, the highest among the candidates.

B. princess, with cosine ≈ 1.00, narrowly ahead of queen at 0.95.

C. prince, the candidate nearest to king itself at ≈ 0.99.

D. monarch, with cosine ≈ 0.87, closest once inputs are excluded.

---

**Q65.** A byte-pair-encoding tokenizer is trained on the word-frequency table below, starting from single characters and with no end-of-word marker. Each step merges the most frequent adjacent symbol pair across the corpus, weighted by word frequency.

| Word | Frequency |
|---|---|
| low | 5 |
| lower | 2 |
| newest | 6 |
| widest | 3 |

After exactly four merges, how is the unseen word "lowest" tokenized?

A. `['lo', 'w', 'est']`

B. `['low', 'e', 's', 't']`

C. `['low', 'est']`

D. `['l', 'o', 'w', 'est']`

---

**Q66.** A preprocessing script for a shelf-monitoring camera builds a synthetic test frame and brightens it two different ways.

```python
import numpy as np
import cv2

img = np.zeros((480, 640, 3), dtype=np.uint8)
img[:, :320] = (255, 0, 0)          # left half
img[:, 320:] = (0, 0, 200)          # right half

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
a = img + 100
b = cv2.add(img, np.full_like(img, 100))

print(rgb[0, 0], a[0, 0], b[0, 0])
```

What does the script print?

A. `[  0   0 255] [ 99 100 100] [255 100 100]`

B. `[255   0   0] [ 99 100 100] [255 100 100]`

C. `[  0   0 255] [255 100 100] [ 99 100 100]`

D. `[  0   0 255] [255 100 100] [255 100 100]`

---

**Q67.** An inspection system detects scratches on metal sheets. The engineer compares three Canny settings on the same blurred image.

```python
import cv2

gray = cv2.imread("sheet_0412.png", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

edges1 = cv2.Canny(blur, 50, 150)    # (low, high) thresholds
edges2 = cv2.Canny(blur, 100, 150)   # low threshold raised
edges3 = cv2.Canny(blur, 50, 250)    # high threshold raised
```

Which statement about `edges2` and `edges3` relative to `edges1` is correct?

A. edges2 gains new edge pixels, since a higher low threshold admits more weak gradients.

B. edges3 keeps every strong edge of edges1, losing only isolated noise pixels.

C. Both are subsets of edges1, as raising either threshold only ever removes edge pixels from it.

D. edges2 equals edges1, as the low threshold only matters above the high one.

---

**Q68.** A three-class review-sentiment classifier is evaluated on 1,000 reviews. Rows are true labels; columns are predictions.

| True \ Predicted | Positive | Neutral | Negative |
|---|---|---|---|
| Positive (700) | 680 | 15 | 5 |
| Neutral (200) | 40 | 150 | 10 |
| Negative (100) | 30 | 30 | 40 |

The team lead writes: "With 87% accuracy and a weighted F1 of 0.86, the model handles all three sentiments well." Which statement correctly evaluates the claim?

A. Macro F1 is about 0.74, dragged down mainly by the negative class's recall of 0.40.

B. Macro F1 is about 0.86, matching weighted F1, as both average per-class scores.

C. Negative precision is 0.40, so most reviews predicted as negative are wrong.

D. Neutral is the weakest class, as its 200 reviews are most often confused.

---

**Q69.** A detector outputs five candidate boxes (x1, y1, x2, y2) for "person" in one frame. Post-processing applies a score threshold and then greedy non-maximum suppression.

```python
boxes  = [[10, 10, 110, 110], [20, 20, 120, 120], [60, 60, 160, 160],
          [65, 65, 165, 165], [300, 300, 350, 350]]
scores = [0.95, 0.90, 0.85, 0.80, 0.40]

def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)

order = sorted([i for i in range(5) if scores[i] >= 0.5],
               key=lambda i: -scores[i])
keep = []
for i in order:
    if all(iou(boxes[i], boxes[k]) < 0.5 for k in keep):
        keep.append(i)
print(keep)
```

What does the script print?

A. `[0, 2, 4]`

B. `[0, 2]`

C. `[0, 2, 3]`

D. `[0, 1, 2, 3]`

---

**Q70.** A traffic-sign classifier for 43 sign classes, including "keep left", "keep right", "turn left ahead" and "turn right ahead", is trained with the pipeline below. Overall validation accuracy is 96%, but the confusion matrix shows the left/right pairs confused with each other roughly half the time, while every other class is above 97%.

```python
import torchvision.transforms as T

train_tf = T.Compose([
    T.RandomResizedCrop(64, scale=(0.8, 1.0)),          # line 2
    T.RandomHorizontalFlip(p=0.5),                      # line 3
    T.ColorJitter(brightness=0.3, contrast=0.3),        # line 4
    T.RandomRotation(10),                               # line 5
    T.ToTensor(),
    T.Normalize(mean=[0.34, 0.31, 0.32],
                std=[0.27, 0.26, 0.27]),                # line 7
])
```

Which line is most responsible for the left/right confusion?

A. Line 2, since cropping 80% of the image can cut away the arrow that marks direction.

B. Line 5, since a 10° rotation turns a left-pointing arrow into a right-pointing one.

C. Line 3, since it mirrors images without changing their direction labels.

D. Line 4, since jitter shifts the blue arrow's colour towards the sign's background.

---

**Q71.** A study group summarizes AlexNet (2012) from its paper. The first convolutional layer receives a 227 × 227 × 3 input.

| Layer | Configuration |
|---|---|
| Conv1 | 96 filters, 11 × 11, stride 4, no padding, followed by ReLU |
| Pool1 | 3 × 3 max pooling, stride 2 |
| Conv2–Conv5 | 5 × 5 then 3 × 3 kernels, ReLU throughout |
| FC6, FC7 | 4,096 units each |
| FC8 | 1,000-way softmax |

Which statement about AlexNet is accurate?

A. Conv1 gives 55 × 55 maps, and the network used ReLU, with dropout in FC6 and FC7 layers.

B. Conv1 gives 56 × 56 maps, and batch normalization followed each convolutional layer.

C. Conv1 gives 55 × 55 maps, and 3 × 3 kernels were used throughout to save parameters.

D. Conv1 gives 54 × 54 maps, and residual connections let its eight layers train stably.

---

**Q72.** A grayscale camera feed has salt noise. The 3 × 3 neighbourhood around one pixel is shown; the centre value is the noisy pixel.

| | | |
|---|---|---|
| 12 | 14 | 13 |
| 11 | **255** | 15 |
| 13 | 12 | 14 |

The engineer compares a 3 × 3 median filter with a Laplacian kernel [[0, 1, 0], [1, −4, 1], [0, 1, 0]] applied at the centre pixel (no normalization).

What values do the two operations produce at the centre?

A. Median 14; Laplacian +968

B. Median 13; Laplacian +968

C. Median 40; Laplacian −968

D. Median 13; Laplacian −968

---

**Q73.** An ML team has three image projects and a ResNet-50 pretrained on ImageNet.

| Project | Labelled images | Similarity to ImageNet photos |
|---|---|---|
| P1 | 1,200 retail product photos | High |
| P2 | 900,000 multispectral satellite tiles | Low |
| P3 | 3,000 grayscale chest X-rays | Low |

Which strategy assignment is most appropriate?

A. P1 fine-tune all layers; P2 frozen feature extraction; P3 train from scratch.

B. P1 frozen feature extraction; P2 fine-tune all layers; P3 fine-tune later layers.

C. P1 frozen feature extraction; P2 frozen feature extraction; P3 fine-tune all layers.

D. P1 train from scratch; P2 fine-tune later layers; P3 frozen feature extraction.

---

**Q74.** A tutorial implements causal scaled dot-product attention on a toy three-token sequence, with queries equal to keys.

```python
import math
import torch

Q = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
K = Q.clone()
d = Q.size(-1)

scores = Q @ K.T / math.sqrt(d)
mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
weights = scores.softmax(dim=-1)

print(weights[1].numpy().round(2))
```

What does the script print?

A. `[0.33 0.67 0.  ]`

B. `[0.27 0.73 0.  ]`

C. `[0.2  0.4  0.  ]`

D. `[0.2  0.4  0.4 ]`

---

**Q75.** A Vision Transformer front end is implemented as follows and run on a batch of eight 384 × 384 RGB images.

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img=384, patch=16, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos

m = PatchEmbed()
out = m(torch.randn(8, 3, 384, 384))
print(out.shape, sum(p.numel() for p in m.parameters()))
```

What does the script print?

A. (8, 576, 768) and 1,033,728 parameters

B. (8, 576, 768) and 590,592 parameters

C. (8, 197, 768) and 742,656 parameters

D. (8, 577, 768) and 1,034,496 parameters

---

**Q76.** An inference team serves a decoder-only model and is considering converting its multi-head attention (MHA) to multi-query attention (MQA). It estimates the KV-cache size as follows.

```python
cfg = dict(
    n_layers=32,
    n_heads=32,          # query heads
    head_dim=128,
    seq_len=8192,
    batch=8,
    bytes_per_val=2,     # fp16
)

def kv_cache_gib(cfg, n_kv_heads):
    vals = (2 * cfg["n_layers"] * n_kv_heads * cfg["head_dim"]
            * cfg["seq_len"] * cfg["batch"])
    return vals * cfg["bytes_per_val"] / 2**30

mha = kv_cache_gib(cfg, n_kv_heads=cfg["n_heads"])
mqa = kv_cache_gib(cfg, n_kv_heads=1)
print(mha, mqa)
```

With MHA the cache is 32 GiB. Which statement about the MQA version is correct?

A. 16 GiB, as MQA shares only keys across heads while values stay per-head.

B. 1 GiB, with no quality change, since all 32 query heads are kept.

C. 1 GiB, at some quality cost from all 32 query heads sharing a single K/V head.

D. 32 GiB, as cache size depends on the model width rather than on head count.

---

**Q77.** A from-scratch sinusoidal positional encoding is implemented for a 4-dimensional toy model.

```python
import numpy as np

def positional_encoding(pos, d_model):
    out = np.zeros(d_model)
    for i in range(0, d_model, 2):
        angle = pos / (10000 ** (i / d_model))
        out[i] = np.sin(angle)
        out[i + 1] = np.cos(angle)
    return out

print(positional_encoding(1, 4).round(3))
```

What does the script print?

A. `[0.54  0.841 1.    0.01 ]`

B. `[0.841 0.54  0.    1.   ]`

C. `[0.841 0.54  0.841 0.54 ]`

D. `[0.841 0.54  0.01  1.   ]`

---

**Q78.** A team trains a small decoder-only language model. The training script raises no exception, but the loss is `nan` from the very first step, even with a tiny learning rate and fp32 precision. The data pipeline has been verified separately.

```python
import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, d_head, p_drop=0.1):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_head)
        self.d_head = d_head
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):                                   # x: (B, T, d_model)
        T = x.size(1)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device),
                          diagonal=0).bool()                # line 14
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)   # line 15
        scores = scores.masked_fill(mask, float("-inf"))    # line 16
        attn = scores.softmax(dim=-1)                       # line 17
        attn = self.drop(attn)
        return attn @ v
```

Which line causes the `nan` loss?

A. Line 15, since dividing by √d_head drives the scores into underflow.

B. Line 14, where the mask is built with the main diagonal included.

C. Line 16, since `-inf` must be replaced by a large finite negative constant.

D. Line 17, as softmax over the last axis mixes queries rather than keys.

---

**Q79.** A greedy decoding helper is written for a decoder-only model whose forward pass returns logits of shape (batch, seq_len, vocab). Whatever the prompt, the 20 generated tokens are all the same token, and that token changes only when the *first* word of the prompt changes.

```python
import torch

@torch.no_grad()
def generate(model, ids, max_new=20):
    model.eval()
    for _ in range(max_new):
        logits = model(ids)                                  # line 6
        next_id = logits[:, 0, :].argmax(-1, keepdim=True)   # line 7
        ids = torch.cat([ids, next_id], dim=1)               # line 8
    return ids
```

Which line explains the behaviour?

A. Line 8, since concatenating along dim 1 appends the new token to the batch.

B. Line 6, since re-encoding the full sequence each step resets the model's state.

C. Line 7, which reads the logits at position 0.

D. None of them, since greedy argmax decoding inevitably repeats a single high-probability token.

---

**Q80.** A platform team chooses a Transformer variant for three products.

| Product | Task |
|---|---|
| T1 | Classify support tickets into 25 categories using the full ticket text |
| T2 | Open-ended chat assistant that continues a conversation |
| T3 | Translate product manuals from English to Hindi |

Which assignment matches each task to the most suitable architecture?

A. T1 decoder-only with causal attention; T2 encoder-only; T3 encoder–decoder.

B. T1 encoder–decoder; T2 decoder-only with causal attention; T3 encoder-only.

C. T1 encoder-only; T2 encoder–decoder with cross-attention; T3 decoder-only.

D. T1 encoder-only; T2 decoder-only with causal attention; T3 encoder–decoder.

---

**Q81.** A 32-layer model with d_model = 4096 is adapted with LoRA, rank r = 8, on the query and value projections only (each a 4096 × 4096 matrix). All original weights are frozen.

| Setting | Value |
|---|---|
| Layers | 32 |
| Adapted matrices per layer | W_q, W_v |
| Shape of each adapted matrix | 4096 × 4096 |
| LoRA rank | 8 |

How many trainable parameters does LoRA add, and what fraction is that of the adapted W_q and W_v weights?

A. About 4.19M trainable, about 0.39% of the adapted W_q and W_v weights.

B. About 2.10M, counting the single A factor that each adapted matrix receives.

C. About 4.19M trainable, about 3.9% of the adapted W_q and W_v weights.

D. About 1.07B, since each adapter must match its 4096 × 4096 matrix.

---

**Q82.** A hybrid search service fuses a BM25 ranking and a dense-retrieval ranking with Reciprocal Rank Fusion.

```python
bm25_ranked  = ["d3", "d1", "d7", "d2", "d5"]
dense_ranked = ["d1", "d4", "d3", "d6", "d2"]

def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

print(rrf([bm25_ranked, dense_ranked])[:3])
```

What does the script print?

A. `['d3', 'd1', 'd7']`

B. `['d1', 'd4', 'd3']`

C. `['d1', 'd3', 'd4']`

D. `['d1', 'd3', 'd2']`

---

**Q83.** A model has d_model = 1024 and 16 attention heads per layer. The team wants to double the context window from 4,096 to 8,192 tokens without changing anything else.

| Property | Value |
|---|---|
| d_model | 1,024 |
| Heads per layer | 16 |
| Current context | 4,096 tokens |
| Proposed context | 8,192 tokens |

Which statement is correct?

A. Head dimension is 64; doubling context doubles score memory, as scores scale with tokens.

B. Head dimension is 64; doubling context quadruples score memory, while the projection weights stay unchanged.

C. Head dimension is 1,024, since each head attends across the full model width.

D. Head dimension is 64; doubling context quadruples memory and doubles the projection weights.

---

**Q84.** A RAG assistant runs on a model with an 8,192-token context window. Every request is assembled in a fixed order: system prompt, conversation history, retrieved chunks, then the user's question. A fixed number of tokens is reserved for the model's answer. Retrieved chunks must be added in rank order; a chunk may not be skipped to fit a later, smaller one, and the first chunk that would exceed the budget ends the packing. Each chunk is wrapped in a citation header that costs 40 tokens.

| Component | Tokens |
|---|---|
| System prompt | 600 |
| Conversation history | 1,450 |
| User question | 120 |
| Reserved for the answer | 1,024 |

| Chunk rank | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Chunk tokens (without header) | 820 | 1,150 | 640 | 980 | 1,210 | 450 |

How many chunks are included in the request?

A. Five chunks, as the 40-token headers fit within the remaining margin.

B. Four chunks, with the fifth pushing the prompt just past its budget.

C. Six chunks, as the answer reserve is not part of the prompt itself.

D. Five chunks, skipping the 1,210-token chunk in favour of the 450-token one.

---

**Q85.** For one evaluation query, the collection contains 6 relevant documents. The retriever's top 10 results are judged as follows (1 = relevant).

| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Relevant? | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |

An engineer proposes passing the top 10 instead of the top 5 to the generator, arguing that "retrieval quality improves on both precision and recall." Which statement correctly evaluates the claim?

A. Recall rises from 0.50 to 0.83 while precision falls from 0.60 to 0.50.

B. Precision rises from 0.50 to 0.60 and recall from 0.50 to 0.83, so both improve.

C. Recall rises from 0.50 to 1.00, as every relevant document appears in the top 10.

D. Precision@10 is 0.83 and recall@10 is 0.50, so precision is what improves.

---

**Q86.** Two retrievers are compared on four queries. Each query has exactly one relevant document; the table gives the rank at which each retriever returned it.

| Query | Retriever R1 | Retriever R2 |
|---|---|---|
| q1 | 1 | 2 |
| q2 | 4 | 2 |
| q3 | 1 | 2 |
| q4 | 10 | 2 |

The generator only ever sees the top 3 retrieved documents. Which statement is correct?

A. R1 leads on MRR (0.59 vs 0.50), so it also serves the top-3 generator better.

B. R2 is better on both, with MRR 0.50 against 0.41 and hit rate@3 of 1.0.

C. R1 leads on MRR (0.59 vs 0.50), but R2 leads on hit rate@3 (1.0 vs 0.5).

D. R1 leads on MRR (0.59 vs 0.50), but R2 leads on hit rate@3 (0.75 vs 0.5).

---

**Q87.** A team chunks policy documents with a fixed-size sliding window before embedding them. For testing, a 1,000-token document is represented by its token positions.

```python
def chunk(tokens, size=300, overlap=50):
    step = size - overlap
    return [tokens[i : i + size] for i in range(0, len(tokens), step)]

doc = list(range(1000))
chunks = chunk(doc)
print(len(chunks), len(chunks[-1]))
```

What does the script print?

A. `4 300`

B. `4 250`

C. `5 200`

D. `4 100`

---

**Q88.** A legal-research assistant stuffs 20 retrieved passages into a long-context model. An evaluation places the one passage containing the answer at different positions among the 20 and measures answer accuracy.

| Position of the answer passage | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Accuracy | 78% | 55% | 49% | 53% | 71% |

In production, the answer passage is usually somewhere among the retriever's top five. Which change is most likely to improve accuracy?

A. Add more passages, since a long-context model reads every position equally well.

B. Sort passages by publication date, so the model reads recent rulings first.

C. Move the best passages to the middle, where accuracy is lowest, to force attention.

D. Rerank, keep fewer passages, and put the strongest at both ends.

---

**Q89.** A RAG pipeline retrieves the top 100 passages from 12 million documents using a bi-encoder, then reranks them with a cross-encoder and passes the top 5 to the generator. A new engineer asks why the pipeline does not simply use the more accurate cross-encoder for the first-stage retrieval as well.

What is the main reason?

A. It must score each query–document pair jointly, so document vectors cannot be precomputed.

B. It produces lower-quality relevance scores than a bi-encoder does on short passages.

C. It cannot process queries longer than its 512-token maximum input length.

D. Its scores are not comparable across documents, so they cannot be used to rank.

---

**Q90.** A company's RAG system embeds documents and queries with the same 768-dimensional model, v1. The team upgrades the query encoder to v2, which also outputs 768-dimensional vectors and scores higher on public benchmarks, but leaves the existing document index unchanged. No errors appear, yet recall@10 on their evaluation set falls from 0.82 to 0.31.

What is the most likely cause, and the fix?

A. v2 is lower quality for this domain, so roll back until a fine-tuned v2 is available.

B. The ANN index parameters need retuning, as recall depends on the search breadth used.

C. Queries and documents now live in different embedding spaces; re-embed the corpus with v2.

D. Cosine similarity must be replaced with dot product, as v2 outputs unnormalized vectors.

---

# ANSWER KEY

| Q | Ans | Domain | Q | Ans | Domain | Q | Ans | Domain |
|---|-----|--------|---|-----|--------|---|-----|--------|
| 1 | B | Stats | 31 | D | PyTorch | 61 | A | NLP |
| 2 | D | Stats | 32 | C | PyTorch | 62 | C | NLP |
| 3 | A | Stats | 33 | D | ML | 63 | D | NLP |
| 4 | C | Stats | 34 | A | ML | 64 | B | NLP |
| 5 | D | Stats | 35 | C | ML | 65 | C | NLP |
| 6 | B | Stats | 36 | A | PyTorch | 66 | A | CV |
| 7 | A | Stats | 37 | B | ML | 67 | C | CV |
| 8 | C | Stats | 38 | A | ML | 68 | A | NLP |
| 9 | C | Prompts/RAG | 39 | C | ML | 69 | B | CV |
| 10 | B | Stats | 40 | B | ML | 70 | C | CV |
| 11 | A | Prompts/RAG | 41 | C | DL | 71 | A | CV |
| 12 | D | Prompts/RAG | 42 | A | DL | 72 | D | CV |
| 13 | B | Prompts/RAG | 43 | D | ML | 73 | B | CV |
| 14 | C | Prompts/RAG | 44 | B | DL | 74 | A | LLM |
| 15 | D | Prompts/RAG | 45 | D | DL | 75 | D | CV |
| 16 | A | Prompts/RAG | 46 | C | DL | 76 | C | LLM |
| 17 | B | RAI Gov | 47 | A | DL | 77 | D | LLM |
| 18 | D | RAI Gov | 48 | B | DL | 78 | B | LLM |
| 19 | B | Prompts/RAG | 49 | C | DL | 79 | C | LLM |
| 20 | A | RAI Gov | 50 | A | RAI Tech | 80 | D | LLM |
| 21 | D | RAI Gov | 51 | D | RAI Tech | 81 | A | LLM |
| 22 | C | RAI Gov | 52 | D | DL | 82 | D | RAG |
| 23 | A | RAI Gov | 53 | B | RAI Tech | 83 | B | LLM |
| 24 | C | RAI Gov | 54 | C | RAI Tech | 84 | B | RAG |
| 25 | A | PyTorch | 55 | B | RAI Tech | 85 | A | RAG |
| 26 | B | RAI Gov | 56 | D | RAI Tech | 86 | C | RAG |
| 27 | C | PyTorch | 57 | A | RAI Tech | 87 | B | RAG |
| 28 | B | PyTorch | 58 | C | RAI Tech | 88 | D | RAG |
| 29 | D | PyTorch | 59 | B | NLP | 89 | A | RAG |
| 30 | B | PyTorch | 60 | D | NLP | 90 | C | RAG |

# EXPLANATIONS

**Q1 — B.** There are 1,000 fraudulent and 199,000 legitimate transactions. T1 catches 920 frauds but also flags 2% of 199,000 = 3,980 legitimate ones, for 4,900 alerts — well over capacity, and only 920/4,900 ≈ 19% are fraud, so the risk lead's claim fails. T2 catches 800 frauds and flags 0.5% of 199,000 = 995 legitimate ones, for 1,795 alerts — within capacity, with precision 800/1,795 ≈ 45%. A confuses specificity (a rate over legitimate transactions) with the false-alarm share of alerts; C confuses sensitivity with precision; D forgets that T2's false positives are only 995.

**Q2 — D.** After the complaint: 0.10×0.60 / (0.10×0.60 + 0.90×0.15) = 0.06/0.195 ≈ 0.308. That posterior becomes the new prior: 0.308×0.50 / (0.308×0.50 + 0.692×0.10) ≈ 0.690. Equivalently, prior odds 1/9 × LR 4 × LR 5 = 20/9, giving 20/29 ≈ 0.69. A (0.36) applies the downgrade to the original 10% prior; B (0.31) stops after the first update; C (0.67) multiplies the prior probability rather than the prior odds by the likelihood ratios.

**Q3 — A.** The odds ratio is (200×400)/(300×100) = 2.67, and a logistic coefficient is the log odds ratio, ln 2.67 ≈ 0.98. Case-control sampling preserves the odds ratio, so the coefficient is valid, but the intercept and sample proportions are not, so 200/300 = 0.67 (C) is not a population probability. Using the population base rate: P(legacy | churn) = 0.40, P(legacy | retain) = 0.20, so P(churn | legacy) = 0.08×0.40 / (0.08×0.40 + 0.92×0.20) = 0.032/0.216 ≈ 0.15. B uses ln 2, the log of the ratio of proportions rather than odds, and multiplies the prior odds by the odds ratio rather than the likelihood ratio (0.40/0.20 = 2), which gives 0.19. D combines the ln 2 coefficient with the invalid sample proportion.

**Q4 — C.** With 3 × 2 cells and 24 plots, df are A = 2, B = 1, A×B = (3−1)(2−1) = 2, error = 24 − 6 = 18. MS_error = 54/18 = 3, MS_AB = 24/2 = 12, so F_AB = 4.0, compared against F(2, 18) = 3.55 — significant (p ≈ 0.037). A significant interaction means the fertilizer effect differs by irrigation schedule, so main effects should be read conditionally. A forgets to divide SS_AB by its df (24/3 = 8). B uses the wrong numerator df. D uses the total mean square (192/23 ≈ 8.35) instead of the error mean square as the denominator.

**Q5 — D.** Cell means: W·F1 = 12, W·F2 = 20, R·F1 = 16, R·F2 = 16. Both crop means are 16, fertilizer means are 14 and 18, grand mean 16. Each interaction residual (cell − row − column + grand) is ±2, so SS_AB = 3 × 4 × 4 = 48 on 1 df. Within-cell SS = 4 × 8 = 32 on 8 df, MS_error = 4, so F_AB = 12.0 (p ≈ 0.009): F2 adds 8 tonnes for wheat and nothing for rice. A reads the zero main effect as "no effect", which the interaction contradicts. B uses 2 df for a 2 × 2 interaction, which has (2−1)(2−1) = 1. C is wrong because the crop main effect F is 0; it is the fertilizer F that also equals 12.

**Q6 — B.** VIF = 1/(1 − R²_aux) = 1/(1 − 0.95) = 20. VIF inflates the coefficient's variance, so the standard error is inflated by √20 ≈ 4.5. A inverts the formula (1/R²). C confuses variance inflation with standard-error inflation. D is wrong on two counts: monthly_charges has the lowest auxiliary R² (VIF 2), and a small p-value indicates a precisely estimated coefficient, not a collinearity source. High model R² alongside insignificant individual coefficients is the classic multicollinearity signature.

**Q7 — A.** X beats Y within each severity group (90% vs 85% mild, 60% vs 50% severe), but pooled rates reverse (X 270/400 = 67.5%, Y 305/400 = 76.25%) because X was given mostly to severe cases — Simpson's paradox. Standardizing to the combined mix (50% mild, 50% severe): X = 0.5×0.90 + 0.5×0.60 = 0.75; Y = 0.5×0.85 + 0.5×0.50 = 0.675. B reads the confounded pooled figure. C transposes the standardized rates. D is false because X is ahead in both strata, so no weighting can make them equal.

**Q8 — C.** Eigenvalues of [[5, 2], [2, 2]] solve λ² − 7λ + 6 = 0, giving 6 and 1. PC1 explains 6/7 ≈ 86%. For λ = 6, (5 − 6)v₁ + 2v₂ = 0 gives v₁ = 2v₂, direction (2, 1)/√5 — leaning towards x₁, the higher-variance feature. A uses the largest single variance rather than the largest eigenvalue. B transposes the eigenvector. D assumes the direction is the 45° diagonal, which holds only when the two variances are equal; symmetry of the covariance matrix is always true and says nothing about direction.

**Q9 — C.** Principle: chain-of-thought wins for multi-step arithmetic, and terse output ranks last. P4 forces every intermediate quantity to be written before the total, which is the strongest scaffold; P1 asks for step-by-step reasoning without specifying the steps, so it is second. The decisive comparison is P3 versus P5: P3 restates the order of operations (band pricing per licence, discount before tax), which removes the two most common error sources, while P5's persona adds tone but no working and no rule clarification. P2 forbids working entirely, so the model must do all four steps in a single pass. A swaps P3 and P5; B puts generic reasoning above structured reasoning; D ranks rule restatement above actual reasoning.

**Q10 — B.** SE of the difference = 16 × √(2/64) ≈ 2.83, so t = 5/2.83 ≈ 1.77 on 126 df. The two-sided p ≈ 0.080, so the pre-registered test is not significant, and the 95% CI is 5 ± 1.98 × 2.83 ≈ (−0.6, 10.6), which includes zero. A gets t right but misreads the interval, which includes zero. C computes Cohen's d (effect size), not t. D reports the one-sided p (≈ 0.040); switching to a one-sided test after seeing the data because an improvement was "expected" is not legitimate when a two-sided test was pre-registered.

**Q11 — A.** Principle: the most detailed prompt carries a harmful filter. Since human triage makes false positives cheap and misses expensive, recall is what matters. P2 explicitly asks for every input-to-sink path, including unlikely ones. P5 enumerates the three injection families, widening coverage. The decisive comparison is P1 versus P3: P1 is focused on injection, while P3's "issues of any kind" spreads attention across all vulnerability classes. P4 is the most precise prompt but ranks last, because "only issues you are confident are exploitable" suppresses exactly the uncertain findings the triage stage exists to catch. B swaps P1 and P3; C and D reward P4's detail.

**Q12 — D.** Principle: an example beats every verbal description for exact surface format. P5 shows the format, including the middle dot and the "Sev3" spacing, which prose struggles to pin down. P3 is the best description, but "dots" invites a full stop (.) rather than the middle dot (·), so the downstream match fails. The decisive comparison is P1 versus P4: P1 at least fixes the square brackets, while P4 specifies neither brackets nor separators; P1's persona contributes nothing either way. P2 relies on a "standard format" the model has never seen. A and C put the description above the example; B swaps P1 and P4.

**Q13 — B.** Principle: the most precisely specified prompt ranks last because its precision is aimed at the wrong target. A 25-word, single-sentence cap cannot hold four conditions, three deadlines and the clearance exception, so P3 guarantees omissions. P1 names the objective directly. P4's one-bullet-per-rule structure tends to keep each rule intact. The decisive comparison is P2 versus P5: P2 is neutral, while P5's "plain, friendly, easy to follow" actively encourages dropping qualifiers such as "subject to inspection" and "unless reported within 48 hours". A rewards P3's precision; C swaps P2 and P5; D puts the bullet structure above the explicit preservation instruction.

**Q14 — C.** Principle: negative constraints backfire, and positive reframing generalizes. P3 defines the allowed source and supplies a concrete fallback behaviour for unanswerable questions, which make up a third of the evaluation. P5 also gives a fallback, less precisely. The decisive comparison is P2 versus P4: P2 states what to use, while P4 lists what to avoid without saying what to do instead, so on unanswerable questions the model still feels obliged to produce something. P1 is last: it covers only three named categories and writes "prices, delivery dates, warranty terms" into context, priming exactly those claims. A swaps P2 and P4; B rewards the stacked prohibitions; D ranks the vaguer fallback below the prompt with no fallback.

**Q15 — D.** Principle: persona wins when the objective is voice. P2 gives a full character (age, history, temperament, speech habits) that the model can stay inside across hundreds of lines; P5 is a thinner version of the same approach. The decisive comparison is P1 versus P3: P1 gives concrete stylistic rules, while P3 merely restates the objective ("consistent in tone") without saying what the tone is. P4 is precise but mechanical: a word cap and a one-term-per-line quota produce formulaic lines rather than a voice. A and C rank rules above persona; B swaps P1 and P3.

**Q16 — A.** Cosine similarities: d1 = 2/(√2·√3) ≈ 0.816, d2 = 1.0 (same direction as q), d3 ≈ 0.707, d4 = 0.5. Min-max BM25 over 1.0–9.0: d1 = 0.775, d2 = 0, d3 = 1, d4 = 0.25. Hybrid scores with α = 0.6: d3 ≈ 0.824, d1 ≈ 0.800, d2 = 0.600, d4 = 0.400. B is what you get by using the raw dot product instead of cosine (d1 = 2 inflates d1). C ranks by cosine alone, ignoring the sparse term. D uses raw rather than normalized BM25, letting the sparse scores dominate and lifting d4 above d2.

**Q17 — B.** The failures are context-specific (a plant's shorthand, a particular clause type) and only users see them, so post-deployment management needs a continuous, structured feedback process: users can flag issues at any time, reports are triaged, each model has a named owner, and responses are tracked to closure. A is one-off — issues arising after month three go uncaptured, and retraining is not the only response. C covers operational health, but drift and latency dashboards cannot detect a summary that silently omits a penalty clause. D retrains on unlabelled usage, which captures nothing about whether outputs were wrong and can reinforce errors.

**Q18 — D.** Approval rates are 500/1,000 = 50% in both groups, so demographic parity holds. True positive rates differ: group A 480/600 = 0.80, group B 380/400 = 0.95. Qualified B applicants are approved more often than qualified A applicants, so equal opportunity fails and the second half of the summary is false. A wrongly infers equal opportunity from equal overall approval rates. B confuses differing base rates with differing approval rates. C transposes the true positive rates.

**Q19 — B.** Line 17 builds a new list, `history`, containing the tool result, but `history` is never used; `messages` never receives the tool message. On the next step the model sees its own tool call with no result, so it calls the tool again, until `max_steps` runs out. The fix is `messages.append(tool_msg)`. A is wrong because tool-calling APIs require the assistant's tool-call turn in the history before the matching tool result. C is wrong because tool schemas are normally passed on every call. D describes the intended exit condition, which would trigger once the model had the result.

**Q20 — A.** The organization that deploys an AI system in a consequential decision remains accountable for that decision. Using a vendor model, or vendor terms that disclaim liability, does not move accountability for hiring outcomes off the retailer. The retailer must audit, mitigate (for example, removing college as an input or recalibrating), and govern the process. B and C misread the effect of vendor terms; "as is" terms disclaim vendor responsibility rather than accept it. D personalizes what is an organizational accountability failure; governance assigns named owners, but liability for the outcome sits with the organization.

**Q21 — D.** The highest tier is for uses that determine outcomes for individuals with no human review before the decision takes effect. U2 (automatic claim rejection) and U4 (automatic credit limits) both meet that bar. A and C rely on volume or external reach, but U1 and U5 are reviewed text that sets no individual outcome. B includes U3, where the engineer reviews every suggestion before it takes effect.

**Q22 — C.** Purpose limitation means data collected for one stated purpose cannot simply be reused for a materially different one. Targeting retention discounts is commercial profiling that affects what customers are offered, which is not "resolving queries or improving support". It needs its own assessment, a transparent notice and a valid lawful basis. A stretches the notice beyond what a reasonable customer would expect. B fails because free-text transcripts are hard to de-identify reliably, and de-identification does not legitimize a new purpose. D is wrong because purpose limitation governs any use, including internal use.

**Q23 — A.** The deployment population (rural, Bhojpuri and Maithili speakers) differs sharply from the training and evaluation population (metro call-centre audio). A single 8.2% error rate on metro data reveals nothing about performance on these speakers, and transcription errors on a helpline could misroute farmers' requests. Disaggregated evaluation across the relevant groups is the decisive missing information. B, C and D describe real fields a model card can include, but none bears on whether the model will work for these callers.

**Q24 — C.** Overrides fell from 2.5% to 0.2% and median review time fell from 41 s to 7 s, while the audit found 9% of approved recommendations wrong (about 18 of 200, far from noise). If the model had improved, as A claims, the audit error rate would not be this high. Reviewers are rubber-stamping, so the "human-in-the-loop" control exists on paper only — a classic automation-bias pattern. B overstates: volume rose only 28% while review time fell 83%, and the fix involves process design (evidence displays, sampled audits, override-rate monitoring), not staffing alone. D is wrong because 18 errors in 200 is a substantial signal.

**Q25 — A.** `x @ w` maps (8, 3, 32) × (32, 16) → (8, 3, 16), and `b` broadcasts over the last dimension. The transpose gives (8, 16, 3); `s` is (8, 16, 1); the softmax keeps (8, 16, 3) and normalizes over the 16-dimension. Summing over dim 1 gives `pooled` = (8, 3). `unsqueeze(0)` → (1, 8, 3), `repeat(2, 1, 1)` → (2, 8, 3), and `flatten(1)` merges everything after dim 0 → (2, 24). B and D forget the transpose when summing over dim 1. C confuses `flatten(1)` with flattening the first two dims.

**Q26 — B.** Valid consent for using a person's voice or likeness should be informed, specific and revocable. A blanket "company purposes" clause signed at hiring did not specifically cover building a synthetic voice model or using it indefinitely after her departure, and she has now explicitly withdrawn consent. A misapplies work-product ownership to a person's identity. C treats disclosure as sufficient, but labelling does not supply consent. D relies on a technicality: a synthetic voice derived from and recognizable as her still uses her identity.

**Q27 — C.** `xb` is moved to the GPU but `yb` is not, so line 24 passes CUDA logits and a CPU target to the loss and raises a device-mismatch RuntimeError on the first batch. The other bugs are real but do not stop execution. Line 10's Softmax before `CrossEntropyLoss` double-normalizes and weakens gradients, but the loss accepts any float input (A). Line 27 accumulates loss tensors rather than `loss.item()`, which holds references but does not crash (B). A further real bug, deliberately absent from the options, is the missing `opt.zero_grad()`, so gradients accumulate across batches. D is wrong: `no_grad` affects only its own block.

**Q28 — B.** `__len__` = 1000 − 24 − 6 + 1 = 971 samples. With `batch_size=64` and the default `drop_last=False`, there are ⌈971/64⌉ = 16 batches: 15 full batches of 64 (960 samples) and a final batch of 11. Each `x` is a window of shape (24, 3), so the final `xb` is (11, 24, 3). A assumes the partial batch is dropped. C forgets the horizon term (976 samples → last batch 16). D swaps the time and feature axes.

**Q29 — D.** The freezing loop at lines 6–7 runs after the new head is attached at line 5, so it freezes the new head too. The `filter` therefore yields nothing, and `torch.optim.Adam` raises `ValueError: optimizer got an empty parameter list` at construction, before any training step. B describes the error that would follow at `loss.backward()` if the optimizer were built over all parameters, but execution never gets there. A is wrong because `model.fc.in_features` is 512, read from the existing layer. C is wrong because `requires_grad` can be set on any leaf parameter. The fix is to freeze first, then replace `fc`.

**Q30 — B.** The model contains dropout, and a freshly constructed `nn.Module` is in training mode, so dropout stays active at serving time. `model.eval()` switches dropout (and any batch norm) to inference behaviour, matching the notebook, which evaluated in eval mode. A makes outputs repeatable but still applies a dropout mask, so they would not match the notebook. C is irrelevant to randomness, and loading CUDA tensors on a CPU server would fail. D is wrong because `inference_mode`, like `no_grad`, only disables gradient tracking; module training mode is separate.

**Q31 — D.** T1 (logits + `CrossEntropyLoss` with class indices) and T2 (logits + `BCEWithLogitsLoss` with multi-hot targets) are correct. T3 applies a sigmoid and then a loss that applies sigmoid internally, so it squashes twice. T4 uses a softmax, which forces the 12 independent tags to compete for probability mass, and `CrossEntropyLoss` then applies log-softmax again. T5 compares an (N, 1) output with an (N,) target; broadcasting produces an (N, N) difference matrix, so the loss is silently wrong (PyTorch only warns). A therefore misreads broadcasting as harmless, B misses the double sigmoid, and C misses T4's softmax.

**Q32 — C.** Conv1: 3×16×3×3 + 16 = 448. BN1: 2×16 = 32 (γ and β). Conv2: 16×32×3×3 = 4,608, with no bias. BN2: 2×32 = 64. Linear: 32×10 + 10 = 330. Total = 5,482. A adds a bias to conv2 (+32 = 5,514). B omits the BatchNorm affine parameters (5,386). D counts BatchNorm's running mean and variance, which are buffers rather than parameters (5,578).

**Q33 — D.** `last_login_date` comes from a CRM export taken on 1 October, after the 90-day label window closed. For customers who cancelled in April–June, the last login predates the cancellation, so `snap − last_login_date` is small or even negative in a pattern that differs systematically from retained customers, whose last login is in September. The feature encodes the label — target leakage — and no such signal exists when scoring live customers. A is a real but minor leak; scaler statistics barely move AUC. B is wrong because each customer appears once, so shuffling cannot share a customer across folds. C is wrong because lines 10–12 restrict calls to the 90 days *before* the snapshot, which is a legitimate feature.

**Q34 — A.** Each bootstrap sample draws 1,000 rows with replacement. The chance a given row is never drawn is (1 − 1/1000)¹⁰⁰⁰ ≈ e⁻¹ ≈ 0.368, so about 368 rows are out-of-bag and about 632 are in the bag. `max_features="sqrt"` gives √64 = 8 candidate features, re-sampled at every split, which is what decorrelates the trees. B and D confuse the in-bag share with the out-of-bag share. C and D misread the feature subsampling as per-tree or absent.

**Q35 — C.** First pass: (4, 5) and (5, 4) are each √5 ≈ 2.24 from (6, 6) and √13 ≈ 3.61 from (2, 2), so they join the second cluster. Updated centroids: mean of (1,1), (2,1), (1,2) = (1.33, 1.33); mean of the other five points = (34/5, 34/5) = (6.8, 6.8). The second pass reassigns nothing, so the output is unchanged. A assigns the two middle points to the first cluster. B drops the middle points from the second cluster's mean. D prints the initial centroids, as if the update never ran.

**Q36 — A.** Run 1's training loss keeps falling while validation loss bottoms out at epoch 4 and then rises, which is classic overfitting; early stopping or regularization is the remedy. Run 3's loss oscillates wildly and reaches NaN, which is divergence from an excessive learning rate (0.5). Run 2 is underfitting from a tiny learning rate: both losses barely move from ln 10 ≈ 2.30 (chance for 10 classes). B and D misread Run 2's matching curves, which reflect a model that has learned almost nothing. C prescribes dropout, which regularizes but does not fix a step size that overshoots.

**Q37 — B.** There are 5 churners. At 0.5, the flagged set is the top three scores: TP = 2, FP = 1, so precision 0.67 and recall 0.40. At 0.35, six customers are flagged (scores down to 0.36): TP = 3 (0.92, 0.66, 0.41), FP = 3, so precision 3/6 = 0.50 and recall 3/5 = 0.60. More churners are caught, but the call list becomes less accurate. A reports the old precision as if unchanged. C transposes the new values. D catches the extra churner but misses the two non-churners (0.38, 0.36) that are also flagged.

**Q38 — A.** On raw features, income differences (thousands of rupees) swamp age differences (tens of years). The three nearest incomes to ₹53,000 are ₹54,000 (premium), ₹52,000 (basic) and ₹49,000 (basic), so the vote is basic. After standardization, age carries equal weight, and the nearest neighbours become (54,000, 58), (47,000, 63) and (61,000, 61) — all premium. D transposes the outputs; B and C assume scaling cannot change the neighbour set.

**Q39 — C.** S1 has no labels and asks for groups, which is clustering (k-means). S2 asks for the most similar past items to a query, which is nearest-neighbour retrieval (k-NN). S3 requires readable rules from labelled data, which a decision tree provides directly. S4 is high-dimensional sparse text with a few thousand labelled examples, the classic setting for a linear SVM. A swaps k-NN and k-means; B swaps the decision tree and the SVM; D swaps k-NN and the decision tree.

**Q40 — B.** The 0.94 is the maximum of twelve noisy cross-validation estimates computed on the same folds that were used to choose the configuration, so it is optimistically biased (selection bias). An unbiased estimate needs nested cross-validation or an untouched test set, which here gives 0.89. A blames the test set for a well-known bias. C inverts the behaviour: large C and large γ fit training data tightly (1.00 training accuracy) and overfit, and the lower cross-validation score reflects variance, not a wide margin. D picks the most overfit configuration.

**Q41 — C.** The cell state update is cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ: the forget gate scales the old memory, and the input gate scales the new candidate. A swaps the roles of the input and forget gates. B uses the output gate, which only controls what is exposed as h. D applies tanh twice, since g is already a tanh output.

**Q42 — A.** The reset gate is a sigmoid of the current input and previous hidden state, computed before the candidate n uses it. In PyTorch's formulation, r multiplies (Uₙh + bₕₙ) inside the candidate, exactly as the given n line does. B uses tanh, which produces values in (−1, 1) rather than a gate in (0, 1). C feeds the update gate into the reset computation, which neither formulation does. D refers to n before it has been computed.

**Q43 — D.** The depth-20 tree fits training data almost perfectly but validates poorly — high variance. Bagging averages many such trees, cutting validation error from 0.22 to 0.14 without changing the base learner's bias. The depth-2 tree has similar, high train and validation error — high bias. Boosting fits successive shallow trees to the residuals, lowering bias, and gets the best validation error (0.12). A is wrong because averaging high-bias models leaves bias unchanged. B is wrong because boosting already-low-bias deep trees mainly adds variance. C mislabels the deep tree's problem as bias.

**Q44 — B.** The dependencies are: c needs f, i and g; h needs o and c. In 5 → 2 → 3 → 4 → 1, i and g, then f, are computed before c; o is computed before h; and h comes last. A computes h before c exists. C computes c before f. D computes h before i, g and c.

**Q45 — D.** 64 → conv k=5, no padding: 60 → pool: 30 → conv k=3, s=2, p=1: ⌊(30 + 2 − 3)/2⌋ + 1 = 15 → pool: ⌊15/2⌋ = 7 → conv k=3, p=1: 7. Flatten: 64 × 7 × 7 = 3,136. A (4,096 = 64 × 8 × 8) assumes the first convolution preserves size. B (14,400 = 64 × 15 × 15) ignores either the stride-2 convolution or the second pool. C (1,024 = 64 × 4 × 4) applies one downsampling too many.

**Q46 — C.** Forward: w1·x + b1 = 0.5, so ReLU gives h = 0.5 (active, derivative 1); ŷ = −1.5 × 0.5 + 1 = 0.25; loss = (0.25 − 1)² = 0.5625. dL/dŷ = 2(ŷ − y) = −1.5. dL/dw2 = −1.5 × h = −0.75. dL/dw1 = −1.5 × w2 × 1 × x = −1.5 × −1.5 × 2 = 4.5. A swaps the two. B omits the factor x = 2. D drops the negative sign of w2.

**Q47 — A.** PyTorch uses inverted dropout. In training, each element is zeroed with probability 0.25, and survivors are scaled by 1/(1 − 0.25), so 3.0 becomes 4.0. In eval mode, dropout is the identity, so the output is 3.0. B describes non-inverted dropout, which leaves training values unscaled and scales by 0.75 at inference. C and D mix the two conventions.

**Q48 — B.** C1: ⌊(32 − 3)/2⌋ + 1 = 15. C2: ⌊(32 + 2 − 3)/2⌋ + 1 = 16. C3: ⌊(32 + 2 − 5)/2⌋ + 1 = 15. C4: ⌊(32 − 4)/2⌋ + 1 = 15. Only C2 gives 16. A, C and D all assume stride 2 alone halves the size, ignoring the kernel and padding terms.

**Q49 — C.** In a vanilla RNN, the gradient passes through a product of Jacobians, each containing W_hh and tanh′ ≤ 1, so it decays geometrically across steps. The LSTM's cell state is updated additively (c = f ⊙ c_prev + i ⊙ g), so ∂cₜ/∂cₜ₋₁ = fₜ; with forget gates near 1, gradients flow back largely unattenuated. A misplaces the problem in the output layer rather than the recurrence. B is wrong because clipping caps large norms and never scales small ones up. D is wrong because the sigmoid derivative has a *maximum* of 0.25, which would itself cause decay.

**Q50 — A.** Permutation importance shuffles one feature at a time. When two features are near-duplicates, the model can rely on whichever remains intact, so each looks unimportant on its own. Removing both costs 0.090 AUC, more than any other feature. B draws exactly the wrong conclusion from the individual scores. C describes a known bias of impurity-based (MDI) importance, not permutation importance. D reads a predictive importance as causal.

**Q51 — D.** ECE = Σ (nᵦ/N) × |accᵦ − confᵦ| = 0.05×0.03 + 0.075×0.07 + 0.125×0.09 + 0.25×0.13 + 0.5×0.12 ≈ 0.11. Accuracy is below confidence in every bin, so the model is overconfident throughout; overall accuracy is 75.7% against a mean confidence of 86.8%. In the top bin, a 0.97 confidence corresponds to 85% accuracy, so the product owner's claim fails. A uses the unweighted mean gap (0.088), ignoring that half the documents sit in the top bin. B mislabels the direction. C treats 85% accuracy as proof of calibration, but calibration compares accuracy with confidence.

**Q52 — D.** He (Kaiming) normal initialization uses std = √(2 / fan_in); for L2 that is √(2/512) = 0.0625. The factor 2 compensates for ReLU zeroing half its inputs. A gives the Xavier normal value √(2/(512 + 256)) ≈ 0.051, which is not He. B is wrong because Xavier uniform uses both fans: √(6/(256 + 10)) ≈ 0.150, not √(6/256) ≈ 0.153. C is wrong because zero weights make every unit in a layer compute the same output and receive the same gradient, so they never differentiate; ReLU does nothing to break that symmetry.

**Q53 — B.** SHAP values are additive: output = base value + Σ contributions = −1.2 + 0.9 − 1.1 − 0.8 + 0.4 = −1.8 in log-odds. σ(−1.8) ≈ 0.14 < 0.5, so the applicant is rejected. The largest negative contribution is credit utilization (−1.1). A flips the sign. C names the right feature but forgets the base value (σ(−0.6) ≈ 0.35). D picks late payments (−0.8), which push less than utilization.

**Q54 — C.** Terms: (0.25 − 0.10) ln 2.5 ≈ 0.137; (0.35 − 0.30) ln(35/30) ≈ 0.008; (0.25 − 0.35) ln(25/35) ≈ 0.034; (0.15 − 0.25) ln(0.6) ≈ 0.051. Total ≈ 0.23, inside the 0.10–0.25 moderate band, so investigation is required, but it does not yet cross the retraining threshold. A sums absolute differences, which is not PSI. B reads the value against the wrong band. D omits half the bins (0.137 + 0.008 ≈ 0.15).

**Q55 — B.** The 11% → 85% jump appears only against an attack that relies on gradients the defence has made unusable. Once the attacker approximates the preprocessing gradient (an adaptive attack), accuracy drops to 9%, no better than undefended. This is gradient masking (obfuscation): the model is no more robust, the attack just got harder to compute. Robustness claims must be tested against adaptive attacks; adversarial training is the standard way to build genuine robustness. A trusts the non-adaptive evaluation. C assumes security through obscurity. D is wrong because a 4/255 perturbation is imperceptible.

**Q56 — D.** The heatmaps and the external performance drop both point to shortcut learning: the model associates the portable-scanner marker (a proxy for sicker patients) with pneumonia instead of reading the lung fields. The right next step is a test of that hypothesis — mask or crop the marker region, re-evaluate internally and externally, and retrain on marker-free images. A discards the evidence without testing it. B prescribes more training, which would reinforce the shortcut. C confuses a proxy that correlates with the label at one site with a valid feature; the marker's meaning depends on the hospital's workflow.

**Q57 — A.** The lowest threshold meeting 98% is 0.85 (98.1%). Coverage is 78% of 10,000 = 7,800 claims, and errors are 7,800 × 1.9% ≈ 148 per day, with the remaining 2,200 going to adjusters. B fails the requirement (97.4% < 98%). C meets it but sacrifices automation the requirement asks to maximize. D applies the error rate to all claims, including those routed to humans. Note that these rates are in-distribution; out-of-distribution claims may be confidently wrong, so production monitoring is still needed.

**Q58 — C.** CF3 flips the decision (valid), changes something the applicant can act on (actionable), and is the smallest change that works (proximal and sparse). CF1 changes an immutable attribute, age, which makes it useless as guidance and raises fairness concerns. CF2 also works, but asks for more than necessary. CF4 is the smallest change, but it does not flip the decision, so it is not a counterfactual at all.

**Q59 — B.** Each 11-word document has 9 trigrams. A and B share 6 ("the delivery was", "delivery was late", "the package was", "package was damaged", "was damaged on", "damaged on arrival"), so Jaccard = 6 / (9 + 9 − 6) = 6/12 = 0.5. C shares only "the delivery was" and "delivery was late" with each of A and B, giving 2 / (9 + 9 − 2) = 2/16 = 0.125 in both cases. A is the unigram (bag-of-words) view, where A and B are identical; trigrams capture word order. C and D overstate the shared opening.

**Q60 — D.** "refund": tf = 2/5 = 0.4, df = 2, idf = log₁₀(5/2) ≈ 0.398, weight ≈ 0.159. "not" and "delayed": tf = 0.2, df = 1, idf = log₁₀ 5 ≈ 0.699, weight ≈ 0.140. "processed": 0.2 × 0.398 ≈ 0.080. The higher term frequency of "refund" outweighs the rarer words' higher IDF. A ranks by IDF alone. B reports tf without IDF. C picks a lower-weighted word.

**Q61 — A.** Spam total = 60, ham total = 80, V = 6. P(w | spam) uses denominator 66: free 21/66, meeting 3/66, today 11/66. P(w | ham) uses denominator 86: free 4/86, meeting 26/86, today 28/86. Log-posteriors: spam ≈ −6.94, ham ≈ −5.90, so P(spam | msg) = 1/(1 + e^(−5.90 + 6.94)) ≈ 0.26 — ham. B transposes the posterior. C would follow from equal priors (≈ 0.34), but the code uses 0.4 / 0.6, and priors do not cancel when they differ. D drops smoothing (≈ 0.24); smoothing changes every probability's denominator even when all words were seen.

**Q62 — C.** The stop list contains "not", "no" and "very", so review 1 → [battery, good], review 2 → [battery, good], review 3 → [complaints, battery, good]. With vocab [battery, complaints, good], reviews 1 and 2 both map to [1, 0, 1] — a negative and a positive review become indistinguishable — and review 3 is [1, 1, 1]. Removing negations with a generic stop list is a classic sentiment-analysis mistake. A and D each assume some negation or intensifier survives; B forgets that "complaints" is not a stop word, so review 3 keeps it.

**Q63 — D.** "Ravi Sinha" (B-PER, I-PER) closes at the O and is emitted. "Patna" (B-LOC) closes at "and". "Bodh" and "Gaya" are I-LOC with no open span (typ is None after the O), so they fall into the else branch and are discarded. "Google" opens an ORG span; "Cloud" is I-LOC, a type mismatch, so the else branch closes ("Google", ORG) and drops "Cloud". A and B assume orphan I- tags start new spans. A and C assume the I-LOC continues an ORG span.

**Q64 — B.** v = (0.80 − 0.70 + 0.10, 0.65 − 0.10 + 0.15, 0.10 − 0.05 + 0.75) = (0.20, 0.70, 0.80). Cosine with princess ≈ 0.999, queen ≈ 0.954, monarch ≈ 0.872, prince ≈ 0.675. The intuitive answer is queen, but in these learned vectors princess is closer. Analogy results depend on the geometry, not on the expected word. A picks the expected answer without computing. C uses similarity to king alone. D misranks monarch.

**Q65 — C.** Pair counts: (e, s) = 9 and (s, t) = 9 (newest × 6 + widest × 3) lead. Whichever of these merges first, the second step completes "est" (count 9). Then (l, o) and (o, w) tie at 7 (low × 5 + lower × 2), and either way the fourth merge completes "low". Applying the four merges to "lowest" gives ['low', 'est']. A, B and D each stop one merge short in some part of the word.

**Q66 — A.** OpenCV stores channels as BGR, so (255, 0, 0) is pure blue, and converting to RGB gives [0, 0, 255]. `img + 100` is NumPy uint8 arithmetic, which wraps modulo 256: 255 + 100 = 355 → 99. `cv2.add` saturates, clipping at 255. B ignores the channel swap. C transposes the NumPy and OpenCV behaviours. D assumes NumPy also saturates.

**Q67 — C.** Canny's hysteresis marks pixels above the high threshold as strong edges, then keeps weaker pixels (above the low threshold) only if they connect to a strong edge. Raising the low threshold drops weaker connecting pixels, so contours can fragment. Raising the high threshold removes seeds, so whole chains with no remaining strong pixel disappear — including some that were strong under the old setting, which is why B is wrong. Either change can only shrink the edge set. A reverses the effect of the low threshold. D is wrong because the low threshold governs exactly the pixels *between* the two thresholds.

**Q68 — A.** Per-class F1: positive ≈ 0.94 (P 0.91, R 0.97), neutral ≈ 0.76 (P 0.77, R 0.75), negative ≈ 0.52 (P 40/55 ≈ 0.73, R 40/100 = 0.40). Macro F1 ≈ 0.74. Weighted F1 (0.86) is dominated by the 700 positive reviews and hides the fact that the model misses 60% of negative reviews. B equates macro and weighted averaging. C transposes negative precision and recall. D is wrong because neutral's F1 (0.76) is well above negative's.

**Q69 — B.** Box 4 (score 0.40) is removed by the score threshold. Box 0 is kept first. Box 1 has IoU ≈ 0.68 with box 0 and is suppressed. Box 2 has IoU ≈ 0.14 with box 0 and is kept. Box 3 has IoU ≈ 0.82 with box 2 and is suppressed. Result: [0, 2]. A skips the score filter. C compares box 3 only against box 0 (IoU ≈ 0.11). D keeps every box above the score threshold without suppression.

**Q70 — C.** A horizontal flip turns a "keep left" sign into a "keep right" sign but leaves the label unchanged, so half the training examples for each directional class carry the wrong label. The model cannot separate the pairs and settles near 50% between them. Flip augmentation must be removed for direction-dependent classes (or the label swapped with the flip). A is unlikely at an 80% minimum scale. B is wrong because a 10° rotation cannot reverse an arrow's direction. D is wrong because a ±30% brightness or contrast change does not swap direction.

**Q71 — A.** Conv1 output: (227 − 11)/4 + 1 = 55. AlexNet popularized ReLU in deep CNNs, used dropout (p = 0.5) in FC6 and FC7, overlapping max pooling, and local response normalization, trained across two GPUs. B is wrong because batch normalization arrived in 2015, and AlexNet used LRN. C describes VGG's uniform 3 × 3 design; AlexNet's first layer is 11 × 11. D describes ResNet (2015).

**Q72 — D.** Sorted neighbourhood: 11, 12, 12, 13, 13, 14, 14, 15, 255, so the median is 13; the outlier is discarded entirely. Laplacian: (14 + 11 + 15 + 12) − 4 × 255 = 52 − 1,020 = −968. A second-derivative filter amplifies isolated spikes, which is why it is usually applied after smoothing. A misorders the values and flips the sign. B flips the sign. C reports the mean (≈ 39.9), not the median.

**Q73 — B.** P1 is small and similar to ImageNet, so pretrained features already fit; freeze the backbone and train a new head, since fine-tuning on 1,200 images risks overfitting. P2 is large and dissimilar (multispectral), so there is enough data to fine-tune the whole network, adapting even the early filters. P3 is small and dissimilar, so keep the generic early layers (edges, textures) frozen and fine-tune the later, more task-specific layers. A, C and D each mismatch data size and domain gap: training from scratch on 3,000 or 1,200 images overfits, and freezing on very different data underfits.

**Q74 — A.** Row 1 (the second token) attends to keys 0 and 1; key 2 is masked. Raw scores q₁·k₀ = 0 and q₁·k₁ = 1, scaled by 1/√2 → 0 and 0.707. softmax([0, 0.707]) = [0.33, 0.67]. B omits the √d scaling (softmax([0, 1]) = [0.27, 0.73]). C applies the mask after softmax without renormalizing. D ignores the mask, so the second token attends to the future.

**Q75 — D.** 384/16 = 24 patches per side, so 576 patches plus 1 CLS token = 577 tokens of dimension 768. Parameters: patch projection 3 × 16 × 16 × 768 + 768 = 590,592; CLS token 768; positional embeddings 577 × 768 = 443,136. Total 1,034,496. A drops the CLS token. B drops the CLS token and counts only the convolution. C assumes 224 × 224 input (196 + 1 = 197 tokens).

**Q76 — C.** The cache scales with the number of K/V heads: 32 GiB × (1/32) = 1 GiB. MQA keeps all query heads but shares a single key head and value head across them, which cuts the cache and memory bandwidth by 32× but reduces representational capacity, usually costing some quality. Grouped-query attention (GQA) is the middle ground. A invents a key-only sharing scheme. B denies the trade-off. D ignores the head-count term in the formula.

**Q77 — D.** For i = 0, the angle is 1/10000⁰ = 1, giving sin 1 ≈ 0.841 and cos 1 ≈ 0.540. For i = 2, the angle is 1/10000^(2/4) = 1/100 = 0.01, giving sin 0.01 ≈ 0.010 and cos 0.01 ≈ 1.000. A swaps sin and cos. B uses the exponent 2i/d with i already stepping by 2, a common bug (angle 0.0001). C uses the same frequency for every dimension pair.

**Q78 — B.** `torch.triu(..., diagonal=0)` marks the main diagonal as masked as well as the future. The first query (position 0) then has every key masked, so its scores are all −∞, softmax returns `nan`, and the `nan` propagates into the loss. The causal mask needs `diagonal=1`, so each token can attend to itself. A is wrong because scaling stabilizes rather than underflows. C is wrong because `-inf` is fine whenever at least one key in the row is visible. D is wrong because `dim=-1` is the key axis, which is correct.

**Q79 — C.** In a causal decoder, the logits at position 0 depend only on the first token, so reading `logits[:, 0, :]` returns the same prediction every step, determined by the first prompt word. The next-token prediction lives at the last position, `logits[:, -1, :]`. A is wrong because dim 1 is the sequence axis, which is correct. B is wrong because re-encoding the sequence is inefficient but correct. D blames greedy decoding, which can be repetitive but would still respond to the whole prompt.

**Q80 — D.** Classification over a full input benefits from bidirectional context, so an encoder-only model (BERT-style) suits T1. Open-ended continuation is autoregressive generation with a causal mask, so decoder-only suits T2. Translation maps one full sequence to another, with the decoder cross-attending to an encoder, so encoder–decoder suits T3. A, B and C each misplace at least one of these.

**Q81 — A.** For each adapted matrix, LoRA adds A (r × d) and B (d × r): 2 × 4096 × 8 = 65,536 parameters. Across 2 matrices × 32 layers: 4,194,304 ≈ 4.19M. The adapted weights total 32 × 2 × 4096² ≈ 1.07B, so the ratio is 0.39%. B counts only one factor. C misplaces the decimal point. D confuses the low-rank update's effective shape with its parameter count.

**Q82 — D.** With k = 60: d1 = 1/62 + 1/61 ≈ 0.03252; d3 = 1/61 + 1/63 ≈ 0.03227; d2 = 1/64 + 1/65 ≈ 0.03101; d4 = 1/62 ≈ 0.01613; d7 = 1/63 ≈ 0.01587. Documents that appear in both lists dominate, even at modest ranks, which is the point of RRF. A follows BM25 alone. B and C overweight d4's high dense rank, but a single list contribution cannot beat two.

**Q83 — B.** Head dimension = 1024 / 16 = 64. Each head's attention score matrix is T × T, so doubling T quadruples score memory and compute. The Q, K, V and output projection weights (4 × 1024² ≈ 4.19M per layer) do not depend on sequence length, so they are unchanged. A treats attention as linear in T. C ignores that heads split d_model. D wrongly ties parameter count to context length.

**Q84 — B.** Available for chunks = 8,192 − 600 − 1,450 − 120 − 1,024 = 4,998 tokens. With 40-token headers, the cumulative chunk cost is 860, 2,050, 2,730, 3,750, then 5,000 at chunk 5, which exceeds 4,998 by 2. So four chunks fit. A ignores the headers (cumulative 4,800 for five chunks). C forgets the answer reserve, giving a 6,022 budget and six chunks. D skips a chunk, which the packing rule forbids.

**Q85 — A.** Top 5 has 3 relevant: P@5 = 0.60, R@5 = 3/6 = 0.50. Top 10 has 5 relevant: P@10 = 0.50, R@10 = 5/6 ≈ 0.83. Recall improves and precision falls, so the generator gets more relevant passages but also more noise. B gets P@5 wrong. C assumes all six relevant documents are retrieved, but one is missing. D transposes precision and recall at k = 10.

**Q86 — C.** MRR(R1) = (1 + 1/4 + 1 + 1/10)/4 = 0.5875; MRR(R2) = (1/2 × 4)/4 = 0.50. Hit rate@3 (relevant document within the top 3): R1 finds it for q1 and q3 only, 0.5; R2 finds it for all four, 1.0. Because the generator sees only the top 3, R2 serves the application better despite its lower MRR — the right metric depends on how results are consumed. A judges by MRR alone and ignores hit rate@3. B miscomputes R1's MRR. D miscounts R2's hit rate@3, which is 4/4.

**Q87 — B.** step = 300 − 50 = 250, so chunks start at 0, 250, 500 and 750 (range stops before 1,000). The last chunk is tokens 750–999, which is 250 tokens; Python slicing truncates at the end of the list. A assumes padding to full size. C uses a step of size − 2 × overlap. D treats the overlap as extra rather than shared.

**Q88 — D.** Accuracy is highest when the answer is at the beginning or end of the context and lowest in the middle — the "lost in the middle" effect. Reranking concentrates the likely answers at the top, keeping fewer passages shrinks the weak middle, and placing the strongest passages at the edges puts them where the model attends best. A contradicts the data. B optimizes for a criterion unrelated to relevance. C does the opposite of what the data shows.

**Q89 — A.** A cross-encoder reads the query and document together in one forward pass, so nothing can be computed ahead of time. Scoring 12 million documents per query would take 12 million transformer passes. A bi-encoder embeds documents offline and retrieves by fast vector search, and the cross-encoder then refines a small candidate set. B is wrong because cross-encoders are generally more accurate. C is a length limit, not the reason. D is wrong because cross-encoder scores are used precisely to rank documents for a given query.

**Q90 — C.** Two different models, even with the same output dimension, produce vectors in unrelated embedding spaces. Comparing v2 query vectors with v1 document vectors makes similarity scores nearly meaningless, and because the shapes match, nothing errors. The fix is to re-embed the whole corpus with v2, or keep both on v1. A blames model quality without evidence. B affects recall only marginally, not a drop from 0.82 to 0.31. D addresses normalization, which cannot explain a near-total loss of relevance.

---
