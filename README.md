# Financial Risk Analysis — Credit Default Modeling

Production-style machine learning pipeline to predict borrower default risk using the Give Me Some Credit dataset (250,000 borrowers).

## Objective

Estimate the probability of financial distress within two years and translate predictions into actionable lending risk tiers.

## Approach

Logistic Regression (interpretable baseline)
AUC: 0.791

Random Forest (nonlinear modeling)
AUC: 0.827

Class imbalance addressed via class weighting and systematic threshold tuning.

## Business-Oriented Threshold Selection

At the default 0.5 cutoff, recall for defaulters was low due to imbalance.
A threshold sweep identified 0.25 as a balanced decision point:
Increased recall for defaulters from ~13% to >30%
Nearly tripled high-risk borrower detection
Maintained controlled false positives

This demonstrates how classification thresholds directly impact financial tradeoffs between credit losses and rejected revenue opportunities.

## Tech Stack
Python · pandas · scikit-learn · statsmodels · modular package structure

Run

pip install -r requirements.txt
python run_pipeline.py

Dataset expected at:

data/raw/cs-training.csv


