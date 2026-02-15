from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate_binary_model(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, preds),
        "report": classification_report(y_true, preds),
    }


