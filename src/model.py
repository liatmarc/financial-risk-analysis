import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X, y):
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const)
    return model.fit(disp=False)

def train_random_forest(X, y, n_estimators=200, max_depth=None, random_state=123):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"  # helpful for imbalanced default data
    )
    rf.fit(X, y)
    return rf

