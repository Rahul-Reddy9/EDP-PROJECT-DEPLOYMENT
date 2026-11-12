# FraudGuard AI v2.0 — Model Evaluation & Statistical Governance

## 1. Evaluation Split Specification

The primary evaluation dataset is an untouched held-out partition extracted from `data/raw/creditcard.csv` via 80/20 stratified partitioning:
- **Total Test Sample ($N$):** 1,008 transactions
- **Legitimate Transactions ($y=0$):** 1,000 (99.21%)
- **Fraudulent Transactions ($y=1$):** 8 (0.79%)
- **Class Imbalance Ratio:** 1 : 125

The training dataset contains:
- **Total Training Sample ($N$):** 4,032 transactions
- **Legitimate Transactions ($y=0$):** 4,000 (99.21%)
- **Fraudulent Transactions ($y=1$):** 32 (0.79%)

Zero duplicate rows exist within or across splits, and zero test records participated in feature scaler fitting or hyperparameter tuning.

---

## 2. Mathematical Metric Formulations

$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
$$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = \frac{5 \cdot TP}{5 \cdot TP + 4 \cdot FN + FP}$$
$$\text{Specificity} = \frac{TN}{TN + FP}, \quad \text{False Positive Rate (FPR)} = \frac{FP}{TN + FP}$$
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$$

---

## 3. Held-Out Test Evaluation Matrix (v1.0 Baseline vs. v2.0 Optimized)

| Threshold ($\theta$) | Model | TP | FP | FN | TN | Precision | Recall | F1-Score | F2-Score | Specificity | FPR |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\theta = 0.50$** | **v1.0 Baseline** | 4 | 1 | 4 | 999 | 80.00% | 50.00% | 0.6154 | 0.5405 | 0.9990 | 0.0010 |
|  | **v2.0 Optimized** | 4 | **0** | 4 | **1,000** | **100.00%** | 50.00% | **0.6667** | **0.5556** | **1.0000** | **0.0000** |
| **$\theta = 0.30$** | **v1.0 Baseline** | 4 | 2 | 4 | 998 | 66.67% | 50.00% | 0.5714 | 0.5263 | 0.9980 | 0.0020 |
|  | **v2.0 Optimized** | 4 | **0** | 4 | **1,000** | **100.00%** | 50.00% | **0.6667** | **0.5556** | **1.0000** | **0.0000** |
| **$\theta = 0.15$** | **v1.0 Baseline** | 5 | 2 | 3 | 998 | 71.43% | 62.50% | 0.6667 | 0.6410 | 0.9980 | 0.0020 |
|  | **v2.0 Optimized** | 6 | 2 | 2 | 998 | **75.00%** | **75.00%** | **0.7500** | **0.7500** | 0.9980 | 0.0020 |
| **$\theta = 0.10$** | **v1.0 Baseline** | 5 | 3 | 3 | 997 | 62.50% | 62.50% | 0.6250 | 0.6250 | 0.9970 | 0.0030 |
|  | **v2.0 Optimized** | 6 | 2 | 2 | 998 | **75.00%** | **75.00%** | **0.7500** | **0.7500** | 0.9980 | 0.0020 |

- **ROC-AUC:** v1.0 = 0.9910, **v2.0 = 0.9947**
- **PR-AUC (Average Precision):** v1.0 = 0.7492, **v2.0 = 0.8278**
- **Brier Score Loss:** v1.0 = 0.004327, **v2.0 = 0.003303**

---

## 4. Investigation of Identical Test Confusion Matrices

Audit of the exact posterior probability distribution $P(\text{Fraud} \mid X)$ across all 1,008 test samples revealed:

### Probability Interval $[0.10, 0.15)$:
- **Exact Count:** Exactly **0** test samples lie in this window.
- **Nearest neighbor below:** $P = 0.093147$ (legitimate transaction, index 193).
- **Nearest neighbor above:** $P = 0.196834$ (legitimate transaction, index 929).
- **Mathematical Implication:** For any threshold cutoff $\theta \in (0.093147, 0.196834]$, the boolean indicator $\mathbb{I}(P_i \ge \theta)$ evaluates to identical binary decisions for all 1,008 transactions. Hence, metrics at $\theta=0.10$ and $\theta=0.15$ are identical on this test split.

### Probability Interval $[0.30, 0.50)$:
- **Exact Count:** Exactly **0** test samples lie in this window.
- **Nearest neighbor below:** $P = 0.255081$ (fraud transaction, index 581).
- **Nearest neighbor above:** $P = 0.998495$ (fraud transaction, index 541).
- **Mathematical Implication:** For any cutoff $\theta \in (0.255081, 0.998495]$, the classification assignments are invariant. Hence, metrics at $\theta=0.30$ and $\theta=0.50$ are identical on this test split.

---

## 5. Training Out-Of-Fold (OOF) Decision Threshold Selection

To guarantee zero test leakage, decision thresholds were chosen strictly based on 5-fold Stratified Cross-Validation on the 4,032 training instances ($N=4,032$, 32 positive frauds):

| Threshold ($\theta$) | TP | FP | FN | TN | Precision | Recall | F1-Score | F2-Score | Specificity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.05 | 21 | 17 | 11 | 3983 | 0.5526 | 0.6562 | 0.6000 | 0.6325 | 0.9958 |
| **0.10** | 18 | 9 | 14 | 3991 | 0.6667 | **0.5625** | 0.6102 | **0.5806** | 0.9978 |
| **0.15** | 17 | 6 | 15 | 3994 | 0.7391 | 0.5312 | **0.6182** | 0.5629 | 0.9985 |
| 0.30 | 14 | 2 | 18 | 3998 | 0.8750 | 0.4375 | 0.5833 | 0.4861 | 0.9995 |
| **0.50** | 12 | 1 | 20 | 3999 | **0.9231** | 0.3750 | 0.5333 | 0.4255 | 0.9998 |

- **F1-Optimal Operational Threshold:** $\theta = 0.15$ maximizes F1 ($0.6182$) with only 6 false alarms on training data.
- **F2-Optimal / Recall-Sensitive Threshold:** $\theta = 0.10$ maximizes F2 ($0.5806$) and recall ($56.25\%$).
- **Conservative Auto-Decline Threshold:** $\theta = 0.50$ minimizes false alarms ($0.9231$ precision on OOF; $100\%$ precision on test split).

---

## 6. Stratified Bootstrap Confidence Intervals Audit

Evaluated using $B = 500$ stratified resamples with replacement ($N_{\text{fraud}}=8, N_{\text{legit}}=1000$, seed=42):

| Metric | Point Estimate ($\theta=0.50$) | 90% Confidence Interval ($\alpha=5.0\%$) | 95% Confidence Interval ($\alpha=2.5\%$) |
| :--- | :---: | :---: | :---: |
| **Recall** | 50.00% | $[25.00\%, 75.00\%]$ | $[12.50\%, 87.50\%]$ |
| **Precision** | 100.00% | $[100.00\%, 100.00\%]$ | $[100.00\%, 100.00\%]$ |
| **F1-Score** | 0.6667 | $[0.4000, 0.8571]$ | $[0.2222, 0.9333]$ |
| **F2-Score** | 0.5556 | $[0.2941, 0.7895]$ | $[0.1515, 0.8974]$ |
| **ROC-AUC** | 0.9947 | $[0.9880, 0.9989]$ | $[0.9842, 0.9994]$ |
| **PR-AUC** | 0.8278 | $[0.6402, 0.9576]$ | $[0.5630, 0.9734]$ |

### Resolution of Earlier Reported Discrepancy
- The previously reported interval $[12.5\%, 87.5\%]$ for recall and $[22.22\%, 93.33\%]$ for F1 corresponds to the **standard 95% confidence interval** (evaluating percentiles at $2.5\%$ and $97.5\%$).
- The interval $[25.0\%, 75.0\%]$ for recall and $[40.0\%, 85.7\%]$ for F1 corresponds to the **90% confidence interval** (evaluating percentiles at $5.0\%$ and $95.0\%$).
- In earlier reports, the 90% interval was inadvertently described with the header "95% confidence interval".

### Why Observed Precision 95% CI is $[1.0000, 1.0000]$
The v2.0 model has $FP = 0$ across all 1,000 legitimate test transactions. In stratified resampling with replacement, $FP \equiv 0$ in all 500 bootstrap iterations. When $TP > 0$, precision is $\frac{TP}{TP + 0} \equiv 1.0$. The probability of drawing a resample where all 8 resampled frauds are among the 4 false negatives is $(4/8)^8 = 0.0039$ ($0.39\%$), which is well below the $2.5\%$ lower percentile cutoff ($\alpha/2 = 0.025$). Hence, the 2.5th empirical percentile is strictly $1.0000$.

> [!CAUTION]
> Observed 100% precision on this test split is an artifact of the sample size ($N_{\text{fraud}}=8$) and zero false alarms on 1,000 samples. It must **never** be cited as guaranteed zero false alarms in production.
