import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.data import load_data, clean_data
from src.model import train_logistic_regression, train_random_forest
from src.evaluation import evaluate_binary_model
from src.scoring import assign_risk_scores

RAW_PATH = "data/raw/cs-training.csv"
CLEAN_PATH = "data/processed/cleaned_data.csv"

df = clean_data(load_data(RAW_PATH))
df.to_csv(CLEAN_PATH, index=False)

X = df.drop(columns="SeriousDlqin2yrs")
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# --- Logistic Regression ---
logit = train_logistic_regression(X_train, y_train)
logit_probs = logit.predict(sm.add_constant(X_test))

logit_metrics = evaluate_binary_model(y_test, logit_probs, threshold=0.5)
print("\n=== Logistic Regression ===")
print(f"AUC: {logit_metrics['auc']:.3f}")
print(logit_metrics["confusion_matrix"])
print(logit_metrics["report"])

# Save logistic risk scores
risk_scores_logit = assign_risk_scores(logit_probs)
risk_scores_logit.to_csv("output/risk_scores_logit.csv", index=False)

# --- Random Forest ---
rf = train_random_forest(
    X_train, y_train,
    n_estimators=200,
    max_depth=None,
    random_state=RANDOM_STATE
)
rf_probs = rf.predict_proba(X_test)[:, 1]

rf_metrics = evaluate_binary_model(y_test, rf_probs, threshold=0.5)
print("\n=== Random Forest ===")
print(f"AUC: {rf_metrics['auc']:.3f}")
print(rf_metrics["confusion_matrix"])
print(rf_metrics["report"])

# Save RF risk scores
risk_scores_rf = assign_risk_scores(rf_probs)
risk_scores_rf.to_csv("output/risk_scores_rf.csv", index=False)

print("\nSaved outputs to /output:")
print(" - output/risk_scores_logit.csv")
print(" - output/risk_scores_rf.csv")

from src.thresholds import threshold_sweep

# --- Threshold tuning tables ---
logit_thresh = threshold_sweep(y_test.values, logit_probs)
rf_thresh = threshold_sweep(y_test.values, rf_probs)

print("\nTop RF thresholds by F1 (default class=1):")
print(rf_thresh.sort_values("f1_1", ascending=False).head(10)[
    ["threshold", "precision_1", "recall_1", "f1_1", "tp", "fp", "fn"]
])

# Save for later analysis
logit_thresh.to_csv("output/threshold_sweep_logit.csv", index=False)
rf_thresh.to_csv("output/threshold_sweep_rf.csv", index=False)

from src.thresholds import pick_threshold_for_recall

t_star = pick_threshold_for_recall(rf_thresh, target_recall=0.30)
print("\nRF threshold achieving recall>=0.30 (highest such threshold):", t_star)
