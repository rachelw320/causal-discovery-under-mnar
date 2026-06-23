import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_test(baseline_shd: list, condition_shd: list) -> dict:
    """
    Wilcoxon signed-rank test comparing SHD distributions from two conditions.

    Used to determine whether the difference in structural accuracy between
    baseline (complete data) and a missingness condition is statistically significant.

    Returns stat, p-value, and a plain-English significance label.
    """
    stat, p = wilcoxon(baseline_shd, condition_shd)
    return {
        "statistic": stat,
        "p_value": p,
        "significant": p < 0.05,
        "label": "p < 0.05 (significant)" if p < 0.05 else "p >= 0.05 (not significant)",
    }


def compare_conditions(results: dict) -> pd.DataFrame:
    """
    Compare all missingness conditions against the baseline using Wilcoxon tests.

    results: dict keyed by condition label (e.g. 'baseline', 'MNAR_0.3'),
             each value is a list of SHD scores across bootstrap iterations.

    Returns a DataFrame with one row per comparison.
    """
    baseline = results["baseline"]
    rows = []

    for label, shd_scores in results.items():
        if label == "baseline":
            continue
        test = wilcoxon_test(baseline, shd_scores)
        test["condition"] = label
        rows.append(test)

    return pd.DataFrame(rows)[["condition", "statistic", "p_value", "significant", "label"]]
