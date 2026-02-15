import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=df.columns[0])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.query("RevolvingUtilizationOfUnsecuredLines <= 1").copy()

    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )

    return df



