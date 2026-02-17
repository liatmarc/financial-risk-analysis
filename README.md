<<<<<<< HEAD
# Insurance Loss Modeling Using a Frequency--Severity Framework
=======
#  Financial Risk Analysis — Credit Default Modeling
>>>>>>> 8e383244418fd83671aac8a61c1a5db2cda98a30

## Executive Summary

<<<<<<< HEAD
This project implements an actuarially consistent insurance loss
modeling pipeline inspired by the French Motor Third-Party Liability
dataset.
=======
## 🚀 Objective
>>>>>>> 8e383244418fd83671aac8a61c1a5db2cda98a30

Rather than directly predicting loss ratio (which is highly volatile at
the individual policy level), the modeling framework decomposes risk
into:

<<<<<<< HEAD
-   **Frequency modeling** (Poisson GLM and XGBoost)
-   **Severity modeling** (Gamma-style GLM and XGBoost)
-   **Expected Loss = Frequency × Severity**
-   Portfolio-level evaluation using **cumulative lift analysis**
=======
## 🧪 Approach
>>>>>>> 8e383244418fd83671aac8a61c1a5db2cda98a30

This mirrors real-world actuarial pricing and underwriting workflows.

------------------------------------------------------------------------

## Modeling Framework

<<<<<<< HEAD
### 1. Frequency Model

-   Target: Claim Count
-   Model types: Poisson GLM (baseline) and XGBoost (Poisson objective)
-   Exposure handled appropriately in modeling
-   Result: Modest but realistic predictive signal consistent with motor
    insurance data
=======
## 🔍 Business-Oriented Threshold Selection

At the default 0.5 cutoff, recall for defaulters was low due to imbalance.
A threshold sweep identified 0.25 as a balanced decision point:
Increased recall for defaulters from ~13% to >30%.  
Nearly tripled high-risk borrower detection.   
Maintained controlled false positives.   
>>>>>>> 8e383244418fd83671aac8a61c1a5db2cda98a30

### 2. Severity Model

-   Target: Claim Amount per Claim (conditional on claim occurrence)
-   Model types: Gamma-style Tweedie GLM and log-scale XGBoost
-   Accounts for heavy-tailed loss distributions

<<<<<<< HEAD
### 3. Combined Expected Loss

Expected Loss is computed as:

    Expected Loss = E[Frequency] × E[Severity]

This allows stable risk ranking without denominator volatility
introduced by loss ratios.

------------------------------------------------------------------------
=======
## Run

```
pip install -r requirements.txt
python run_pipeline.py

```
Dataset expected at:

```
data/raw/cs-training.csv

```
>>>>>>> 8e383244418fd83671aac8a61c1a5db2cda98a30

## Portfolio Evaluation

Rather than focusing on policy-level R² (which is unstable for loss
ratio modeling), performance is evaluated using:

-   Cumulative Lift Curves
-   Portfolio Selection Analysis
-   Underwriting Segmentation Impact

Key Insight: Ranking policies by expected loss produces meaningful
portfolio stratification. Lower predicted risk segments contain
materially less than proportional realized loss.

------------------------------------------------------------------------

## Lessons Learned

-   Loss ratio is highly volatile at the policy level.
-   Leakage can easily produce artificially high R² if claims or
    premium-derived fields are included as features.
-   Proper actuarial decomposition (frequency + severity) produces more
    realistic and defensible results.
-   Portfolio lift is more informative than individual-level regression
    metrics in underwriting applications.

------------------------------------------------------------------------

## Business Framing

This project demonstrates how machine learning can be applied
responsibly within insurance modeling by:

-   Respecting actuarial structure
-   Avoiding data leakage
-   Handling exposure correctly
-   Benchmarking GLM vs gradient boosting
-   Evaluating performance through underwriting impact rather than raw
    regression accuracy

------------------------------------------------------------------------

## 30-Second Interview Summary

"I built a frequency--severity insurance modeling pipeline instead of
directly predicting loss ratio, because policy-level loss ratios are
inherently volatile. By modeling frequency and severity separately and
evaluating performance using cumulative lift, the framework demonstrates
measurable underwriting segmentation aligned with real actuarial
practice."
