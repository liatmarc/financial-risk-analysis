import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def threshold_sweep(y_true, probs, thresholds=None):
    """
    Sweep thresholds and return a table of precision/recall/f1 and confusion matrix counts.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, ..., 0.95

    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)

        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        rows.append({
            "threshold": float(t),
            "precision_1": prec,
            "recall_1": rec,
            "f1_1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

def pick_threshold_for_recall(df, target_recall=0.30):
    """
    Pick the highest threshold that achieves at least target recall (fewer false positives).
    """
    eligible = df[df["recall_1"] >= target_recall]
    if eligible.empty:
        return None
    return float(eligible["threshold"].max())
