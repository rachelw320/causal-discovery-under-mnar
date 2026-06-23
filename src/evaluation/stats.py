import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_test(baseline_shd: list, condition_shd: list) -> dict:
    """Wilcoxon signed-rank test between two SHD distributions."""
    stat, p = wilcoxon(baseline_shd, condition_shd)
    return {
        "statistic": stat,
        "p_value": p,
        "significant": p < 0.05,
        "label": "p < 0.05 (significant)" if p < 0.05 else "p >= 0.05 (not significant)",
    }


def compare_conditions(results: dict) -> pd.DataFrame:
    """Compare each condition against 'baseline' using Wilcoxon tests."""
    baseline = results["baseline"]
    rows = []

    for label, shd_scores in results.items():
        if label == "baseline":
            continue
        test = wilcoxon_test(baseline, shd_scores)
        test["condition"] = label
        rows.append(test)

    return pd.DataFrame(rows)[["condition", "statistic", "p_value", "significant", "label"]]
