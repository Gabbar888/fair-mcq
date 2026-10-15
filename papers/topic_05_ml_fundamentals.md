# Mock Practice — Topic 5: Machine Learning Fundamentals
## 20 questions | 33 minutes | MCQ, single correct

Choose the single best option for each question. Churn always means a customer cancelling or going inactive within the window stated in the question.

---

**Q1.** A subscription business builds a churn tree on 20 customers, 10 of whom cancelled in the following quarter. Three candidate root splits are evaluated with Gini impurity, and the split giving the lowest weighted child impurity is chosen.

| Candidate split | Left child (churn, stay) | Right child (churn, stay) |
|---|---|---|
| Contract = monthly | (9, 3) | (1, 7) |
| Support tickets > 2 | (5, 1) | (5, 9) |
| Age < 30 | (6, 4) | (4, 6) |

Which split is chosen, and what is its weighted Gini impurity?

A. Support tickets, at about 0.40, as its left child is purest.

B. Contract type, at a weighted impurity of about 0.31 across both children.

C. Contract type, at about 0.19, the reduction from the root impurity.

D. Age, at about 0.48, as it splits customers into equal halves.

---

**Q2.** A churn classifier uses k-NN with k = 3 and Euclidean distance on two raw features: annual income in thousands and age in years. A colleague proposes adding `StandardScaler` (population standard deviation) before the distance computation. The query customer has income 60 and age 26.

| Customer | Income (k) | Age | Label |
|---|---|---|---|
| c1 | 58 | 61 | stay |
| c2 | 63 | 55 | stay |
| c3 | 66 | 48 | stay |
| c4 | 30 | 27 | churn |
| c5 | 120 | 24 | churn |
| c6 | 95 | 29 | churn |
| c7 | 20 | 70 | stay |

Which option gives the prediction without scaling and with scaling, in that order?

A. Churn without scaling and stay with scaling, as scaling amplifies income.

B. Stay in both cases, as scaling preserves every pairwise distance ordering.

C. Churn in both cases, as the three churners are the youngest.

D. Stay without scaling, and churn once both features are standardized.

---

**Q3.** An analyst runs a single k-means iteration by hand to check a library result.

```python
import numpy as np

X = np.array([[1, 1], [2, 1], [4, 3],
              [5, 4], [1, 2], [6, 5]], dtype=float)
C = np.array([[1, 1], [5, 4]], dtype=float)      # initial centroids

d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
labels = d.argmin(axis=1)
C_new = np.array([X[labels == k].mean(axis=0) for k in range(2)])
inertia = ((X - C_new[labels]) ** 2).sum()
print(C_new.round(2), round(inertia, 2))
```

Which option matches the printed output?

A. Centroids (1.33, 1.33) and (5.00, 4.00), with inertia about 5.33.

B. Centroids (1.33, 1.33) and (5.00, 4.00), with inertia 6.00 from the old centroids.

C. Centroids (1.00, 1.00) and (5.00, 4.00), since centroids move only after convergence.

D. Centroids (2.50, 2.00) and (5.50, 4.50), with inertia about 5.33 overall.

---

**Q4.** A team estimates out-of-sample performance for a churn model on 12,000 customers, of whom 6% churned within 60 days.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = load_churn_features()          # 400 candidate features
X = SelectKBest(f_classif, k=20).fit_transform(X, y)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(scores.mean())
```

Which statement about the printed AUC is correct?

A. It is optimistic, because the scaler is refit inside each fold and leaks validation statistics.

B. It is pessimistic, because stratification forces scarce churners into every fold and starves training.

C. It is optimistic, because feature selection used the labels of every validation fold.

D. It is unbiased, because shuffled stratified folds keep every fold representative of the population.

---

**Q5.** A retention team scores 15 customers for 60-day churn and contacts everyone at or above a threshold. Scores in descending order, with true outcomes (1 = churned):

| Score | 0.92 | 0.85 | 0.81 | 0.74 | 0.66 | 0.58 | 0.52 | 0.47 | 0.41 | 0.33 | 0.28 | 0.22 | 0.15 | 0.09 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Churned | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

The manager claims: "Moving the threshold from 0.5 to 0.7 wastes fewer calls on loyal customers, but lets more real churners slip away." Which pair gives precision and recall at a threshold of 0.7 and supports that claim?

A. Precision 0.75 and recall 0.60, three of four calls reaching real churners.

B. Precision 0.60 and recall 0.75, as fewer calls raise the recall.

C. Precision 0.57 and recall 0.80, since the flagged set barely changes.

D. Precision 0.75 and recall 0.90, as nine of ten loyal customers go uncontacted.

---

**Q6.** A random forest is trained on 500 rows and 64 features, with 200 trees, bootstrap sampling, and `max_features="sqrt"`.

Which option gives the expected number of trees for which a particular row is out-of-bag, and the number of features considered at each split?

A. About 126.5 trees, the in-bag share, and 8 features per split.

B. About 73.5 trees, and 6 features from the base-2 logarithm.

C. About 73.5 trees, and 8 features considered at each split.

D. About 100 trees, as bootstraps keep half the rows, and 8 features.

---

**Q7.** A bank has two separate problems. First, it has two million unlabelled card customers with numeric spending profiles and wants to discover natural segments for marketing. Second, it has 50,000 past loan applications, each labelled as defaulted or repaid, and the regulator requires a model whose individual decisions can be traced as explicit if-then rules.

Which pair of algorithms fits the two problems, in order?

A. k-NN for the segments, grouping customers with neighbours, and a decision tree for loans.

B. k-means for the segments, and a decision tree for the loan defaults.

C. k-means for the segments, and k-NN for loans, as neighbours explain decisions.

D. k-NN for the segments, and k-means to cluster applicants by risk.

---

**Q8.** A team fits polynomial regressions of increasing degree to predict monthly revenue per customer and records the root-mean-squared error on training and validation folds.

| Degree | Train RMSE | Validation RMSE |
|---|---|---|
| 1 | 4.8 | 5.0 |
| 3 | 2.1 | 2.4 |
| 6 | 1.2 | 2.9 |
| 10 | 0.3 | 6.5 |

Which statement is supported by the table?

A. Degree 1 suffers from high variance, since both its errors are large.

B. Degree 10 has the lowest bias and should be chosen for deployment.

C. Degree 6 is underfitting, since validation error exceeds training error there.

D. Degree 3 balances bias and variance best, and degree 10 overfits most.

---

**Q9.** A trained linear SVM has weight vector w = (3, 4) and bias b = −2, with the decision function f(x) = w·x + b and margin boundaries at f(x) = ±1.

Which option gives the full margin width and the position of the training point (0.6, 0.3) relative to the margin?

A. Margin width is 0.4, and the point lies exactly on the margin boundary.

B. Margin width is 0.2, and the point lies inside the margin at 0.2.

C. Margin width is 0.4, and the point lies inside, as 1.0 is below 2.

D. Margin width is 2.5, and the point lies outside, as f(x) is positive.

---

**Q10.** AdaBoost starts with eight training samples, each with weight 0.125. The first decision stump misclassifies two of them. The learner weight is α = ½·ln((1 − ε)/ε), and sample weights are multiplied by exp(±α) and renormalized.

| Quantity | Value |
|---|---|
| Samples | 8 |
| Initial weight per sample | 0.125 |
| Misclassified by stump 1 | 2 |

Which option gives α for the first stump and the new weight of each misclassified sample?

A. α is about 1.10, and each misclassified sample weighs 0.25 after renormalizing.

B. α is about 0.55, and each misclassified sample weighs about 0.22 unnormalized.

C. α is about 0.55, and each misclassified sample weighs 0.50 after renormalizing.

D. α is about 0.55, and each misclassified sample weighs 0.25 once weights are renormalized.

---

**Q11.** A data scientist tunes a gradient-boosting churn model on 500 customers, 50 of whom churned within 90 days.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

grid = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(GradientBoostingClassifier(), grid, cv=cv,
                      scoring="average_precision", refit=True)
search.fit(X, y)
```

Which option gives the total number of model fits performed and the number of churners in each validation fold?

A. 60 fits in total, and 10 churners in each validation fold.

B. 61 fits in total, and 10 churners in each validation fold.

C. 61 fits, and all 50 churners, since stratification keeps every churner.

D. 13 fits, one per configuration plus refit, and 10 churners.

---

**Q12.** A small ordinary least squares fit is computed by hand.

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 4, 7, 8], dtype=float)

b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b0 = y.mean() - b1 * x.mean()
print(round(b1, 2), round(b0, 2), round(b0 + b1 * 6, 2))
```

Which option matches the three printed values?

A. Slope 1.8, intercept 1.2, and a prediction of 12.0 at x = 6.

B. Slope 1.43, intercept 1.1, from regressing x on y, and 9.7.

C. Slope 1.2, intercept 1.8, and a prediction of 9.0 at x = 6.

D. Slope 1.2, intercept 1.8, and 7.2, since the intercept cancels.

---

**Q13.** An engineer fits an L1-regularized logistic regression for 30-day churn on 80 standardized features, sweeping the regularization setting.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

for C in [0.01, 0.1, 1, 10, 100]:
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C)
    m.fit(X_train, y_train)
    nz = np.count_nonzero(m.coef_)
    print(C, nz, round(m.score(X_train, y_train), 3))
```

As C increases from 0.01 to 100, which pattern is the output most likely to show?

A. The non-zero coefficient count tends to grow, and training accuracy tends to rise overall.

B. The non-zero count tends to shrink, since C multiplies the L1 penalty term.

C. The non-zero count stays at 80, since L1 shrinks but never zeroes weights.

D. Training accuracy tends to fall, since a larger C means a stronger penalty.

---

**Q14.** A streaming service defines churn as a subscriber making no payment in the 90 days after a snapshot date of 31 March 2026. Features must use only what was known on the snapshot date. The feature pipeline is:

```python
import pandas as pd

SNAP = pd.Timestamp("2026-03-31")
END = SNAP + pd.Timedelta(days=90)

paid_after = payments[(payments.date > SNAP) & (payments.date <= END)]
label = ~customers.cid.isin(paid_after.cid)            # 1 = churned

f_recency = (SNAP - payments[payments.date <= SNAP]
             .groupby("cid").date.max()).dt.days
f_tickets = tickets[tickets.date <= SNAP].groupby("cid").size()
f_tenure  = (SNAP - customers.set_index("cid").signup_date).dt.days
f_refunds = refunds[refunds.date <= END].groupby("cid").size()
```

Which feature leaks information about the label?

A. `f_recency`, since it comes from the same payments table as the label.

B. `f_tickets`, since ticket volume is strongly correlated with cancelling.

C. `f_tenure`, since signup dates can postdate the snapshot for new users.

D. `f_refunds`, whose counting window runs to the end of the label period.

---

**Q15.** Each tree in a bagged ensemble has prediction variance 4.0, and any two trees have prediction correlation 0.3. The ensemble averages 50 trees.

Which option gives the variance of the ensemble average, and the floor that variance approaches as the number of trees grows without limit?

A. Variance about 0.08, with a floor of zero, as averaging cancels noise.

B. Variance about 1.26, approaching a floor of 1.20 as more trees are added.

C. Variance about 1.26, with a floor of 0.08 set by tree count.

D. Variance about 1.20, rising to 1.26, as correlation grows with trees.

---

**Q16.** A churn model scores eight customers. The true labels are shown (1 = churned within 30 days).

| Customer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Score | 0.90 | 0.80 | 0.70 | 0.60 | 0.55 | 0.50 | 0.40 | 0.30 |
| Churned | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |

What is the ROC AUC?

A. 0.75, the recall from flagging every score of 0.5 or more.

B. 0.31, the share of positive–negative pairs that are correctly ordered.

C. 0.69, computed over all sixteen positive–negative customer pairs.

D. 0.50, because accuracy at a 0.5 threshold is one half.

---

**Q17.** A gradient-boosting regressor with squared loss and learning rate 0.1 is trained on five customers, with targets of monthly spend 10, 12, 20, 22 and 30. The initial prediction is the target mean. Each round fits a depth-1 tree on the current residuals, and in both rounds it splits customers {1, 2} from {3, 4, 5}. Each leaf predicts the mean residual of its customers.

Which option gives the prediction for customer 1 after the first round and after the second round?

A. 17.32 after the first round and 18.02 after the second, as residuals shrink.

B. 11.00 after the first round, with the full leaf value applied, then 10.30.

C. 18.02 after the first round, then 17.24, reusing the first-round residuals.

D. 18.02 after the first round, and about 17.32 after the second round.

---

**Q18.** A churn tree is being built on 16 customers, half of whom churned. Three features are considered:

| Feature | Children (churn, stay) |
|---|---|
| Plan | basic (6, 2), premium (2, 6) |
| Region | north (3, 3), south (3, 3), east (2, 2) |
| Customer ID | 16 children, each a single customer |

Information gain is computed with base-2 entropy, and gain ratio divides gain by the split's intrinsic information. Which feature does information gain select, and which does gain ratio select?

A. Customer ID under information gain, and Customer ID under gain ratio.

B. Customer ID under information gain, and Plan under the penalizing gain ratio.

C. Plan under both criteria, as each ID child holds one customer.

D. Plan under information gain, and Customer ID under gain ratio.

---

**Q19.** An analyst compares k-NN settings on a tiny one-dimensional dataset using leave-one-out cross-validation.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut

x = np.array([1.0, 1.5, 2.2, 3.0, 3.4, 4.1, 5.0, 5.6]).reshape(-1, 1)
y = np.array([0, 0, 1, 0, 1, 1, 1, 0])

for k in (1, 3):
    m = KNeighborsClassifier(n_neighbors=k)
    loo = cross_val_score(m, x, y, cv=LeaveOneOut()).mean()
    train = m.fit(x, y).score(x, y)
    print(k, round(train, 3), round(loo, 3))
```

Which option matches the output for k = 1, and which k does LOOCV prefer?

A. Training accuracy 1.000 and LOOCV 1.000, so k = 1 is preferred.

B. Training accuracy 0.625 and LOOCV 0.375, so k = 3 is preferred.

C. Training accuracy 1.000 and LOOCV 0.375, so k = 3 is preferred here.

D. Training accuracy 1.000 and LOOCV 0.625, so k = 1 is preferred.

---

**Q20.** k-means has produced two clusters. Cluster A holds (1, 1), (1, 2) and (2, 1); cluster B holds (6, 5), (7, 5) and (6, 6). Distances are Euclidean.

What is the silhouette coefficient of the point (1, 1)?

A. About 0.90, averaging cluster A distances with the point itself included.

B. About 0.85, using its mean distances to cluster A and cluster B.

C. About 0.15, the ratio of within-cluster to nearest-cluster distance.

D. About −0.85, since the point sits at the edge of its cluster.

---

# ANSWER KEY

| Q | Ans | Domain |
|---|-----|--------|
| 1 | B | ML |
| 2 | D | ML |
| 3 | A | ML |
| 4 | C | ML |
| 5 | A | ML |
| 6 | C | ML |
| 7 | B | ML |
| 8 | D | ML |
| 9 | A | ML |
| 10 | D | ML |
| 11 | B | ML |
| 12 | C | ML |
| 13 | A | ML |
| 14 | D | ML |
| 15 | B | ML |
| 16 | C | ML |
| 17 | D | ML |
| 18 | A | ML |
| 19 | C | ML |
| 20 | B | ML |

# EXPLANATIONS

**Q1 — B.** Weighted Gini: contract = 12/20 × 0.375 + 8/20 × 0.219 = 0.3125; tickets = 0.405; age = 0.480. The lowest impurity wins, so contract is chosen. A has a pure-looking left child but a large, mixed right child. C reports the gain (0.5 − 0.3125 = 0.1875) as the impurity. D confuses balanced child sizes with purity.

**Q2 — D.** Raw distances are dominated by income: the nearest three are c3, c2 and c4 (stay, stay, churn), so the prediction is stay. After standardization, age matters comparably: the nearest are c4, c6 and c3 (churn, churn, stay), so the prediction is churn. A reverses the effect. B is false because scaling changes relative distances. C ignores the unscaled case.

**Q3 — A.** Assignments are {(1,1), (2,1), (1,2)} to centroid 1 and {(4,3), (5,4), (6,5)} to centroid 2. The new centroids are (1.33, 1.33) and (5, 4). Inertia against the new centroids is 5.33. B gives the inertia against the old centroids (6.0), but the code uses C_new. C skips the update step. D uses the wrong memberships.

**Q4 — C.** SelectKBest is fit on all labelled rows before cross-validation, so every validation fold's labels influence which features survive, inflating AUC. With 400 candidates and 6% positives, the leak can be large. The fix is to put selection inside the pipeline. A is wrong because refitting the scaler per fold is the correct practice. B and D misjudge stratification, which is neutral here.

**Q5 — A.** At 0.7, the flagged scores are 0.92, 0.85, 0.81 and 0.74, so TP = 3, FP = 1 and FN = 2. Precision is 0.75 and recall 0.60. At 0.5, precision is 0.57 and recall 0.80, so precision rises and recall falls, matching the claim. B transposes the pair. C gives the 0.5 values. D reports specificity (0.90) as recall.

**Q6 — C.** P(a row is out-of-bag) = (1 − 1/500)^500 ≈ 0.368, so about 200 × 0.368 ≈ 73.5 trees. sqrt(64) = 8 features per split. A counts in-bag trees. B uses the log2 rule. D wrongly assumes a bootstrap keeps half the rows; it keeps about 63% of unique rows.

**Q7 — B.** Unlabelled segment discovery is clustering (k-means). Labelled default prediction with traceable if-then rules fits a decision tree. A and D use k-NN, a supervised method that needs labels, for clustering. C's k-NN gives no explicit rule set for a regulator.

**Q8 — D.** Degree 3 has the lowest validation error with a small train–validation gap. Degree 10 has the largest gap (0.3 against 6.5), the signature of overfitting. A mislabels degree 1, where both errors are high and close, as variance; it is bias. B picks by training error. C mislabels degree 6, which is starting to overfit, not underfit.

**Q9 — A.** ‖w‖ = 5, so the margin width is 2/5 = 0.4. f(0.6, 0.3) = 1.8 + 1.2 − 2 = 1.0, which lies exactly on the f = +1 boundary. B halves the margin; distance 0.2 from the hyperplane equals the half-margin, which is the boundary itself. C compares f with the wrong quantity. D inverts the width.

**Q10 — D.** ε = 2/8 = 0.25 and α = ½ ln 3 ≈ 0.549. The unnormalized weights are 0.125 × √3 = 0.2165 (misclassified) and 0.125/√3 = 0.0722 (correct). Total = 0.866, so after normalizing each misclassified sample weighs 0.25, and the misclassified pair together holds half the mass. A drops the ½. B is the pre-normalization value. C gives the combined mass of both samples.

**Q11 — B.** 4 × 3 = 12 configurations × 5 folds = 60 fits, plus one refit on all data with refit=True, for 61 total. Stratified 5-fold puts 50/5 = 10 churners in each validation fold. A omits the refit. C confuses the fold count with the total. D ignores the folds.

**Q12 — C.** x̄ = 3 and ȳ = 5.4. Sxy = 12 and Sxx = 10, so b1 = 1.2 and b0 = 5.4 − 3.6 = 1.8. The prediction at x = 6 is 9.0. A swaps slope and intercept. B inverts the slope from regressing x on y. D drops the intercept.

**Q13 — A.** In scikit-learn, C is the inverse of regularization strength. Larger C weakens the L1 penalty, so more coefficients become non-zero and training fit improves. The L1 path is not strictly monotone, hence "tends to". B and D treat C as the penalty multiplier. C is wrong because L1 produces exact zeros.

**Q14 — D.** `f_refunds` counts refunds up to END, which is inside the label window. A refund implies a payment in that window, so the feature encodes the outcome. A is wrong: recency is cut at SNAP, and sharing a table is not leakage. B describes a legitimate predictive signal. C is a data-quality concern, not label leakage.

**Q15 — B.** Var = ρσ² + (1 − ρ)σ²/B = 0.3 × 4 + 0.7 × 4/50 = 1.256. As B → ∞, the variance approaches ρσ² = 1.20. This is why random forests decorrelate trees with feature subsampling. A assumes independent trees. C misattributes the floor. D swaps the values.

**Q16 — C.** There are 4 positives and 4 negatives, giving 16 pairs. Correctly ordered pairs: the 0.90 positive beats all 4 negatives; 0.70 and 0.60 each beat 3; 0.40 beats 1. That gives 11/16 = 0.6875. A is recall at 0.5 (3/4). B is 1 − AUC, the share of misordered pairs. D is accuracy at 0.5, which is unrelated to AUC.

**Q17 — D.** F0 = 18.8. The residuals for {1, 2} average −7.8, so F1 = 18.8 − 0.78 = 18.02. The new residuals are −8.02 and −6.02, with mean −7.02, so F2 = 18.02 − 0.702 = 17.32. A reverses the order. B ignores the learning rate. C reuses the first-round leaf value.

**Q18 — A.** Root entropy is 1.0. IG(plan) = 1 − 0.811 = 0.189 and IG(region) = 0. IG(ID) = 1.0, since every child is pure. Gain ratio: plan 0.189/1.0 = 0.189; ID 1.0/log2(16) = 0.25. So ID still wins under gain ratio here. B assumes the penalty always suffices. C and D misjudge the plain gain.

**Q19 — C.** 1-NN always predicts each training point's own label, so training accuracy is 1.000. LOOCV with k = 1 gets 3 of 8 correct (0.375). k = 3 scores 0.625 on LOOCV, so it is preferred. A confuses training accuracy with LOOCV accuracy. B gives the k = 3 training accuracy. D gives the k = 3 LOOCV score.

**Q20 — B.** a = mean distance to (1,2) and (2,1) = 1.0. b = mean distance to cluster B = (6.40 + 7.21 + 7.07)/3 = 6.90. s = (b − a)/max(a, b) = 0.855. A includes the zero self-distance in a. C is a/b. D flips the sign, but the point is well matched.

---
