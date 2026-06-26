import pandas as pd
from scipy.stats import chi2_contingency
from itertools import combinations


def detect_mnar_pairs(df: pd.DataFrame, alpha: float = 0.05) -> list[tuple]:
    """Check every variable pair for correlated missingness using chi-square."""
    cols = df.columns.tolist()
    indicators = {col: df[col].isna().astype(int) for col in cols}

    flagged = []
    for col_a, col_b in combinations(cols, 2):
        table = pd.crosstab(indicators[col_a], indicators[col_b])
        if table.shape == (2, 2):
            _, p, _, _ = chi2_contingency(table)
            if p < alpha:
                flagged.append((col_a, col_b, p))

    return flagged
