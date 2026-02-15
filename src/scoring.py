import pandas as pd
from .config import RISK_BINS, RISK_LABELS

def assign_risk_scores(probs):
    df = pd.DataFrame({"Prob_Default": probs})
    df["Risk_Level"] = pd.cut(df["Prob_Default"], bins=RISK_BINS, labels=RISK_LABELS)
    return df

