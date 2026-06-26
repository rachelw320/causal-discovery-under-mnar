import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
from sklearn.linear_model import LogisticRegression
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


def _encode(series: pd.Series) -> pd.Series:
    """Encode a categorical series as integers, keeping NaN as NaN."""
    codes = series.astype("category").cat.codes.astype(float)
    codes[codes == -1] = np.nan
    return codes


def _logistic_pvalue(x: np.ndarray, y: np.ndarray) -> float | None:
    """Wald test p-value for the predictor coefficient in logistic regression."""
    if y.sum() == 0 or y.sum() == len(y):
        return None
    try:
        model = LogisticRegression(fit_intercept=True, max_iter=1000, solver="lbfgs")
        model.fit(x.reshape(-1, 1), y)
        p_hat = model.predict_proba(x.reshape(-1, 1))[:, 1]
        w = p_hat * (1 - p_hat)
        x_aug = np.column_stack([np.ones(len(x)), x])
        v = (x_aug * w[:, None]).T @ x_aug
        se = np.sqrt(np.diag(np.linalg.inv(v)))
        z = model.coef_[0, 0] / se[1]
        return float(2 * (1 - norm.cdf(abs(z))))
    except Exception:
        return None


def detect_mnar_pairs_logistic(df: pd.DataFrame, alpha: float = 0.05) -> list[tuple]:
    """Flag pairs where observed values of one variable predict missingness of the other."""
    cols = df.columns.tolist()
    encoded = {col: _encode(df[col]) for col in cols}

    flagged = []
    for col_a, col_b in combinations(cols, 2):
        a, b = encoded[col_a], encoded[col_b]

        # direction A predicts B missing
        mask_ab = a.notna()
        p_ab = _logistic_pvalue(a[mask_ab].values, b[mask_ab].isna().astype(int).values)

        # direction B predicts A missing
        mask_ba = b.notna()
        p_ba = _logistic_pvalue(b[mask_ba].values, a[mask_ba].isna().astype(int).values)

        valid = [p for p in [p_ab, p_ba] if p is not None]
        if valid:
            p_min = min(valid)
            if p_min < alpha:
                flagged.append((col_a, col_b, p_min))

    return flagged
