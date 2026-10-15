# Mock Practice — Topic 1: Statistical Thinking and Applications
## 20 questions | 33 minutes | MCQ, single correct

Choose the single best option for each question. Assume two-sided tests at the 5% level unless stated otherwise.

---

**Q1.** An agronomy team grows two crops under three fertilizers, with three plots per combination, and records yield in quintals per plot. They fit a two-way ANOVA with crop and fertilizer as factors, including their interaction, and compute every F-ratio against the within-cell error mean square.

| Crop | Fertilizer F1 | Fertilizer F2 | Fertilizer F3 |
|---|---|---|---|
| Wheat | 20, 22, 21 | 26, 28, 27 | 31, 30, 32 |
| Maize | 24, 23, 25 | 25, 27, 26 | 26, 25, 27 |

Which option gives the F-ratio for the crop × fertilizer interaction and the F-ratio for the fertilizer main effect?

A. Interaction F is 56 and fertilizer F is 24, as interaction pools main effects.

B. Interaction F is 48 and fertilizer F is 112, as SS needs no division.

C. Interaction F is 24, and fertilizer F is 56 here.

D. Interaction F is 24 and fertilizer F is 4.5, as crop is second.

---

**Q2.** A bank screens 10,000 card transactions with a fraud model. Fraud is rare, and the investigation team reviews every flagged transaction by hand.

| | Actually fraud | Actually legitimate |
|---|---|---|
| Flagged | 45 | 180 |
| Not flagged | 15 | 9,760 |

A team lead says: "The analyst queue is dominated by noise despite excellent specificity, yet the model still catches most of the fraud that exists." Which reading of the table supports that statement?

A. Sensitivity is 0.75 and roughly four in five flagged cases are false alarms, even though specificity is high.

B. Specificity is 0.75, so roughly one in four legitimate transactions is flagged wrongly, which swamps the queue.

C. Precision is 0.98, so the queue is mostly genuine fraud and the volume reflects true cases.

D. Sensitivity is 0.20, because only 45 of the 225 flagged cases are fraud, so most fraud is missed.

---

**Q3.** A screening test for a rare condition has sensitivity 0.90 and specificity 0.95. The condition affects 2% of the screened population. A clinician converts the prevalence into prior odds, multiplies by the positive likelihood ratio, and then converts the resulting odds back into a probability for a patient who tests positive.

Which pair of values is correct for this patient?

A. Posterior odds are 18.0 and the posterior probability is 0.95.

B. Posterior odds are 0.27 and the posterior probability is 0.37.

C. Posterior odds are 0.37 and the posterior probability is 0.37.

D. Posterior odds are 0.37 and the posterior probability is 0.27.

---

**Q4.** A plant has three machines. They produce 50%, 30% and 20% of output, with defect rates of 2%, 5% and 10% respectively. One machine is picked using those output shares, and its items are then inspected one at a time. The first two items inspected are both defective. After the first item, the posterior over machines is used as the prior for the second item.

Which option gives the posterior probability that the picked machine is the third one, and the probability that the next item is defective?

A. Posterior is 0.68 and the next-item defect probability is 0.066.

B. Posterior is 0.68 and the next-item defect probability is 0.082.

C. Posterior is 0.44 and the next-item defect probability is 0.082.

D. Posterior is 0.20 and the next-item defect probability is 0.045.

---

**Q5.** A hospital trial gives 400 patients each to Drug P and Drug Q. Patients are split into a younger and an older group, and allocation was not balanced across the groups.

| Drug | Group | Patients | Recovered |
|---|---|---|---|
| P | Younger | 40 | 36 |
| P | Older | 360 | 192 |
| Q | Younger | 360 | 306 |
| Q | Older | 40 | 16 |

Using the pooled recovery rate across all 400 patients for each drug, which statement is correct?

A. Drug Q recovers more patients overall, at 80.5% against 57.0% for Drug P, despite trailing within each group.

B. Drug P recovers more patients overall, since it leads in both age groups separately.

C. Drug P recovers more patients overall, at 71.7% against 62.5% for Drug Q.

D. Both drugs recover the same share overall, since the group sizes are swapped.

---

**Q6.** A churn model, which predicts whether a subscriber cancels within the next 30 days, uses three predictors. A data scientist computes variance inflation factors directly from the predictor correlation matrix.

```python
import numpy as np
corr = np.array([[1.0, 0.9, 0.5],
                 [0.9, 1.0, 0.4],
                 [0.5, 0.4, 1.0]])
vif = np.diag(np.linalg.inv(corr))
```

Which predictor has the largest VIF, and by what factor is its coefficient standard error inflated relative to uncorrelated predictors?

A. Predictor x2 has the largest VIF, roughly 5.4, inflating its standard error about 5.4 times.

B. Predictor x1 has the largest VIF, roughly 6.0, inflating its standard error about 6.0 times.

C. Predictor x2 has the largest VIF, roughly 5.4, inflating its standard error about 2.3 times.

D. Predictor x1 has the largest VIF, roughly 6.0, inflating its standard error 2.4 times.

---

**Q7.** An engineer measures request latency (ms) on the same eight servers before and after a configuration change, then runs the following analysis.

```python
import numpy as np
from scipy import stats

before = np.array([212, 198, 240, 225, 205, 231, 219, 246])
after  = np.array([205, 190, 236, 215, 200, 222, 214, 237])

t, p = stats.ttest_ind(before, after)
print(round(t, 2), round(p, 3))
```

The team is deciding whether latency changed. Which statement about this analysis is correct?

A. Printed t is about 0.86, and a paired test gives about 9.03 with 14 degrees of freedom.

B. Printed t is about 0.86, and a paired test gives about 9.03 with 7 degrees of freedom.

C. Printed t is about 9.03, and a paired test gives about 0.86 with 7 degrees of freedom.

D. Printed t is about 0.86, and a paired test gives the same value, since the mean difference matches.

---

**Q8.** Two sensor features have sample covariance matrix [[6, 2], [2, 3]] and use different units. An analyst fits one-component PCA twice, first on the raw features and then after standardizing.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

raw = PCA(n_components=1).fit(X)
std = make_pipeline(StandardScaler(), PCA(n_components=1)).fit(X)
print(raw.explained_variance_ratio_, std[-1].explained_variance_ratio_)
```

Which option gives the two printed explained-variance ratios, in the order printed?

A. Raw is roughly 0.74 and standardized is roughly 0.78.

B. Raw is roughly 0.78 and standardized is roughly 0.78.

C. Raw is roughly 0.78 and standardized is roughly 0.74.

D. Raw is roughly 0.67 and standardized is roughly 0.74.

---

**Q9.** The following script raises a runtime error when executed.

```python
import numpy as np

A = np.array([[1, 2],
              [0, 1],
              [3, 1]])
B = np.array([[2, 0, 1],
              [1, 3, 0]])

G = A.T @ A
H = A @ B
K = B @ A
M = H.T @ K
s = np.trace(K) + np.linalg.det(G)
print(M.shape, s)
```

Which line raises the error, and why?

A. `H = A @ B` fails because A is 3x2 and B is 2x3, so the shapes misalign.

B. `K = B @ A` fails because swapping the order of the factors makes the product undefined.

C. `s = np.trace(K) + np.linalg.det(G)` fails because G is singular, so its determinant raises.

D. `M = H.T @ K` fails because H.T is 3x3 and K is 2x2, so inner dimensions differ.

---

**Q10.** A telecom firm models churn, defined as a subscriber cancelling within the next 30 days. The fitted logistic regression has the coefficients below, on the log-odds scale.

| Term | Coefficient |
|---|---|
| Intercept | -2.00 |
| Tenure (months) | -0.05 |
| Support tickets (last 30 days) | 0.40 |
| Premium plan (1 = yes) | -0.80 |

For a premium subscriber with 12 months of tenure and 3 recent tickets, which pair gives the predicted churn probability and the effect of one extra ticket on the odds of churn?

A. Churn probability is roughly 0.10, and each extra ticket raises the odds by about 49%.

B. Churn probability is roughly 0.10, and each extra ticket raises the odds by about 40%.

C. Churn probability is roughly 0.20, and each extra ticket raises the odds by about 49%.

D. Churn probability is roughly 0.90, and each extra ticket raises the odds by about 49%.

---

**Q11.** A two-way ANOVA with Factor A (3 levels), Factor B (2 levels) and five replicates per cell produced the partly completed table below. The critical values at the 5% level are F(1, 24) = 4.26 and F(2, 24) = 3.40.

| Source | SS | df | MS | F |
|---|---|---|---|---|
| Factor A | 84 | 2 | | |
| Factor B | 30 | 1 | | |
| A × B | 60 | 2 | | |
| Error | 216 | 24 | | |
| Total | 390 | 29 | | |

Which effects are significant at the 5% level?

A. Factor A and the interaction are significant, as each F-ratio exceeds 3.40.

B. Factor A alone is significant, since the interaction F of 3.33 falls just below 3.40.

C. Factors A and B are significant, as the F-ratio for B exceeds 4.26.

D. None of the effects are significant, since every F-ratio is below its critical value.

---

**Q12.** A churn analyst samples 36 customers at random and records monthly spend, getting a mean of 52.0 and a sample standard deviation of 12.0. She runs:

```python
import numpy as np
from scipy import stats

n, m, s = 36, 52.0, 12.0
lo, hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=s)
print(round(lo, 1), round(hi, 1))
```

She reports the printed range as the 95% confidence interval for mean monthly spend. Which statement is correct?

A. It prints roughly (47.9, 56.1), the 95% interval for the mean.

B. It prints roughly (27.6, 76.4), the correct 95% interval for the mean.

C. It prints roughly (27.6, 76.4), six times wider than the interval for the mean.

D. It prints roughly (48.1, 55.9), because the z critical value replaces t.

---

**Q13.** A help desk receives calls as a Poisson process averaging six per hour. Each call is independently escalated with probability one half, regardless of arrivals. A manager wants the probability that at least five calls are escalated in a given hour.

Which value is correct?

A. About 0.11 for six binomial trials at one half.

B. About 0.71 for a Poisson count with mean six.

C. About 0.58 for a Poisson count at least three.

D. About 0.18 for a Poisson count, mean three.

---

**Q14.** A regression with three predictors was fitted to 25 observations, giving the ANOVA decomposition below.

| Source | SS | df |
|---|---|---|
| Regression | 350 | 3 |
| Residual | 150 | 21 |
| Total | 500 | 24 |

Which option gives the adjusted R-squared and the overall F-statistic?

A. Adjusted R-squared is about 0.66 and the overall F is about 16.3.

B. Adjusted R-squared is about 0.70 and F is about 16.3.

C. Adjusted R-squared is about 0.66 and F is about 2.3.

D. Adjusted R-squared is about 0.67 and F is about 17.1.

---

**Q15.** A subscription firm will survey n = 1,000 customers to estimate the overall churn rate. Basic-plan customers are 80% of the base with a true churn rate of 4%; Pro-plan customers are 20% with a rate of 20%. The standard error of the estimated churn rate is computed under three designs: simple random sampling, stratified sampling with proportional allocation, and stratified sampling with Neyman allocation.

Which option lists the three standard errors, in percentage points, in that order?

A. 0.82, 0.82 and 0.75 respectively.

B. 0.79, 0.82 and 0.75 respectively.

C. 0.82, 0.79 and 0.75 respectively.

D. 0.82, 0.75 and 0.79 respectively.

---

**Q16.** A growth team compares ten page variants against a control and collects one p-value per variant. They count discoveries three ways: raw p < 0.05, Bonferroni at an overall level of 0.05, and Benjamini–Hochberg at q = 0.05.

```python
import numpy as np

pvals = np.array([0.001, 0.004, 0.009, 0.012, 0.021,
                  0.034, 0.048, 0.20, 0.47, 0.81])
m = len(pvals)
raw = (pvals < 0.05).sum()
```

Which option gives the number of declared discoveries under raw, Bonferroni and Benjamini–Hochberg, in that order?

A. 7, 2 and 4 discoveries respectively.

B. 7, 2 and 5 discoveries respectively.

C. 7, 3 and 5 discoveries respectively.

D. 6, 2 and 5 discoveries respectively.

---

**Q17.** An analyst reviews ten order values and decides whether to drop an extreme one.

```python
import numpy as np

x = np.array([42, 45, 47, 48, 50, 51, 53, 55, 58, 250])
y = x[x < 100]
print(x.mean(), np.median(x), x.std(ddof=1))
print(y.mean(), np.median(y), y.std(ddof=1))
```

Comparing the second printed line with the first, which statement is correct?

A. The mean falls by 20.0, the median stays at 50.5, and the standard deviation falls from 63.5 to 5.0.

B. The mean falls by 0.5, the median falls by 20.0, and the standard deviation falls from 63.5 to 5.0.

C. The mean falls by 20.0, the median falls by 0.5, and the standard deviation falls from 63.5 to 4.7.

D. The mean falls by 20.0, the median falls by 0.5, and the standard deviation falls from 63.5 to 5.0.

---

**Q18.** A colleague claims that Pearson and Spearman coefficients differ only by rounding when two variables are strongly associated. To test this claim, you run the following.

```python
import numpy as np
from scipy import stats

x = np.arange(1, 11)
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0])
print(np.corrcoef(x, y)[0, 1], stats.spearmanr(x, y)[0])
```

Which option describes the two printed values?

A. Pearson is roughly 0.59 and Spearman is exactly 1.00.

B. Pearson is roughly 0.99 and Spearman is roughly 0.59, since ranks absorb outliers.

C. Pearson is roughly 1.00 and Spearman is roughly 0.59, since ranks inflate small gaps.

D. Both are roughly 0.59, because the outlier also distorts the ranks.

---

**Q19.** A data scientist compresses a matrix with a truncated SVD and then checks how much of it survives.

```python
import numpy as np

B = np.array([[4., 0., 0.],
              [3., -5., 0.],
              [0., 0., 2.]])
U, s, Vt = np.linalg.svd(B)
B2 = (U[:, :2] * s[:2]) @ Vt[:2]
print(np.linalg.norm(B - B2), (s[:2] ** 2).sum() / (s ** 2).sum())
```

Which option gives the two printed values, in order?

A. Roughly 4.00 and 0.93, the squared residual norm and the energy share.

B. Roughly 3.74 and 0.93, the residual norm of a rank-one approximation.

C. Roughly 2.00 and 0.93, the residual norm and the retained energy share.

D. Roughly 2.00 and 0.83, the residual norm and the singular-value share.

---

**Q20.** A pre-registered two-sided A/B test with 6,000 users per arm records 480 conversions for control and 540 for the variant. The team uses a pooled two-proportion z-test at the 5% level.

Which option correctly states the test statistic, the p-value and the decision?

A. z is about 2.78 with p near 0.005, so the variant wins clearly.

B. z is about 1.96 with p near 0.050, so the null is narrowly rejected.

C. z is about 1.96 with p near 0.025, so the null is rejected comfortably.

D. z is about 1.39 with p near 0.165, so the null is retained.

---

# ANSWER KEY

| Q | Ans | Domain |
|---|-----|--------|
| 1 | C | Stats |
| 2 | A | Stats |
| 3 | D | Stats |
| 4 | B | Stats |
| 5 | A | Stats |
| 6 | D | Stats |
| 7 | B | Stats |
| 8 | C | Stats |
| 9 | D | Stats |
| 10 | A | Stats |
| 11 | B | Stats |
| 12 | C | Stats |
| 13 | D | Stats |
| 14 | A | Stats |
| 15 | C | Stats |
| 16 | B | Stats |
| 17 | D | Stats |
| 18 | A | Stats |
| 19 | C | Stats |
| 20 | B | Stats |

# EXPLANATIONS

**Q1 — C.** Cell means: Wheat 21/27/31, Maize 24/26/26; grand mean 25.83. SS crop 4.5, SS fertilizer 112, SS interaction 48, SSE 12 with 12 df, so MSE = 1. F interaction = (48/2)/1 = 24; F fertilizer = (112/2)/1 = 56. A swaps the two. B reports SS/MSE without dividing by df. D gives the crop F (4.5/1) as the fertilizer value. Both effects exceed their critical values (F(2,12) = 3.89).

**Q2 — A.** Sensitivity = 45/60 = 0.75; specificity = 9760/9940 = 0.982; precision = 45/225 = 0.20, so four in five alerts are false. B swaps sensitivity and specificity. C reads specificity as precision. D reads precision (0.20) as sensitivity; missed fraud is only 15 of 60.

**Q3 — D.** Prior odds = 0.02/0.98 = 0.0204. LR+ = 0.90/0.05 = 18. Posterior odds = 0.367; probability = 0.367/1.367 = 0.269. A treats the likelihood ratio as the posterior. B transposes the pair. C reports the odds as if they were the probability.

**Q4 — B.** After the first defect: posterior = (0.222, 0.333, 0.444). After the second: (0.068, 0.254, 0.678). Next-item defect probability = 0.068(0.02) + 0.254(0.05) + 0.678(0.10) = 0.082. A uses the stale one-step predictive (0.066). C stops after one update. D reports the prior and the unconditional defect rate (0.045).

**Q5 — A.** Pooled: P = 228/400 = 57.0%, Q = 322/400 = 80.5%. Drug P is better within each group (90% vs 85%, 53.3% vs 40%), but Q was mostly given to the healthier younger group, so the pooled figure reverses (Simpson's paradox). B uses the within-group lead for the pooled claim. C averages the group rates without weighting by group size. D is not supported.

**Q6 — D.** The diagonal of the inverse correlation matrix gives VIF = (6.00, 5.36, 1.36). x1 is highest because it also correlates with x3. Standard error inflation is sqrt(VIF) = 2.45 for x1. A and C pick x2, which has the second-largest VIF. B treats VIF as the standard-error multiplier; it is the variance multiplier.

**Q7 — B.** The same servers are measured twice, so the data are paired. The independent test prints t = 0.86 (p = 0.41) because between-server variance swamps the effect. The paired test uses the differences (mean 7.125, sd 2.23): t = 9.03 with n − 1 = 7 df. A uses the unpaired df. C swaps the two. D ignores that the denominators differ.

**Q8 — C.** Raw: eigenvalues 7 and 2, so PC1 share = 7/9 = 0.78. Standardized: correlation = 2/sqrt(18) = 0.471, eigenvalues 1.471 and 0.529, so PC1 share = 0.736. A swaps the order. B assumes standardizing changes nothing. D uses the first feature's share of total variance (6/9) for the raw case.

**Q9 — D.** H = A @ B is 3x3 and K = B @ A is 2x2, so H.T @ K has inner dimensions 3 and 2 and raises a ValueError. A and B compute fine. C is wrong because det(G) = 35 and a singular matrix returns 0 anyway rather than raising.

**Q10 — A.** Log-odds = −2.00 − 0.60 + 1.20 − 0.80 = −2.20, so p = 0.0998 ≈ 0.10. exp(0.40) = 1.49, so each extra ticket multiplies the odds by 1.49, or +49%. B reads the log-odds coefficient as a percentage. C omits the premium term (p ≈ 0.20). D reports the probability of not churning.

**Q11 — B.** MSE = 216/24 = 9. F(A) = 42/9 = 4.67 > 3.40; F(B) = 30/9 = 3.33 < 4.26; F(A×B) = 30/9 = 3.33 < 3.40 (p ≈ 0.053). Only A is significant. A claims the interaction exceeds 3.40, which it does not. C compares B's F with the wrong cutoff. D ignores factor A.

**Q12 — C.** The scale argument should be the standard error s/sqrt(n) = 2, not s = 12. The printed range is 52 ± 2.030 × 12 = (27.6, 76.4), exactly sqrt(36) = 6 times wider than the correct interval (47.9, 56.1). A is the correct interval, not the printed one. B misdescribes the printed range. D is wrong: t.interval uses the t critical value.

**Q13 — D.** Thinning a Poisson(6) stream with probability 0.5 gives Poisson(3) escalations. P(X ≥ 5) = 0.185. A treats the call count as fixed at six (0.109). B uses the unthinned mean (0.715). C uses a threshold of three (0.577).

**Q14 — A.** R² = 350/500 = 0.70; adjusted = 1 − 0.30 × 24/21 = 0.657. F = (350/3)/(150/21) = 16.33. B reports unadjusted R². C divides SS regression by SS residual. D uses n − p instead of n − p − 1 (adjusted 0.673, F = 17.1).

**Q15 — C.** The overall rate is 7.2%. SRS: sqrt(0.072 × 0.928/1000) = 0.817 points. Proportional: sqrt((0.8 × 0.04 × 0.96 + 0.2 × 0.20 × 0.80)/1000) = 0.792. Neyman allocates 662 and 338 by W × sigma, giving 0.749. Proportional allocation always does at least as well as SRS, and Neyman at least as well as proportional. A treats proportional as no gain. B and D break that ordering.

**Q16 — B.** Raw: 7 below 0.05. Bonferroni threshold = 0.005, so 2 survive. BH compares the sorted p-values with k × 0.05/10; p(5) = 0.021 ≤ 0.025 but p(6) = 0.034 > 0.030, so the largest passing k = 5. A stops BH early. C uses a looser Bonferroni cutoff. D miscounts raw (7 values lie below 0.05).

**Q17 — D.** Before: mean 69.9, median 50.5, sd 63.5. After: mean 49.9, median 50.0, sd 5.0. Mean drops by 20.0, median by only 0.5. A says the median is unchanged. B swaps mean and median changes. C reports the population sd (ddof=0) for the second line.

**Q18 — A.** y is strictly increasing in x, so the ranks match exactly and Spearman = 1.00. The single outlier drags Pearson down to 0.593. B and C reverse the pattern. D claims ranks are distorted, but a monotone relation keeps identical ranks.

**Q19 — C.** Singular values are 6.32, 3.16 and 2.00. The Frobenius residual of the rank-2 fit is the discarded singular value, 2.00. Retained energy = (40 + 10)/54 = 0.926. A gives the squared norm. B gives the rank-one residual (sqrt(14) = 3.74). D uses unsquared singular values (0.83).

**Q20 — B.** Pooled p = 0.085; SE = sqrt(0.085 × 0.915 × 2/6000) = 0.00509; z = 0.01/0.00509 = 1.964; two-sided p = 0.0495 < 0.05. A omits the factor 2 in the variance of the difference. C quotes the one-sided p. D divides z by sqrt(2) again.

---
