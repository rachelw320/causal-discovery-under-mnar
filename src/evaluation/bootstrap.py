import numpy as np
import pandas as pd
from typing import Callable
import config
from src.evaluation.metrics import evaluate


def bootstrap_run(
    df: pd.DataFrame,
    run_algorithm: Callable,
    true_edges: set,
    all_nodes: list,
    n_iterations: int = config.BOOTSTRAP_ITERATIONS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Run algorithm on n_iterations bootstrap samples, return metrics DataFrame."""
    rng = np.random.default_rng(seed)
    records = []

    for i in range(n_iterations):
        sample = df.sample(n=len(df), replace=True, random_state=int(rng.integers(1e6)))
        result = evaluate(true_edges, run_algorithm(sample), all_nodes)
        result["iteration"] = i
        records.append(result)

    return pd.DataFrame(records)


def summarise_bootstrap(results_df: pd.DataFrame) -> dict:
    """Return mean and std for each metric across bootstrap iterations."""
    return {
        "shd_mean": results_df["shd"].mean(),
        "shd_std": results_df["shd"].std(),
        "fp_rate_mean": results_df["fp_rate"].mean(),
        "fp_rate_std": results_df["fp_rate"].std(),
        "fn_rate_mean": results_df["fn_rate"].mean(),
        "fn_rate_std": results_df["fn_rate"].std(),
    }
