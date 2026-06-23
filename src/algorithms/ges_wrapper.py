import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges


def run_ges(df: pd.DataFrame) -> set:
    """
    Run the GES algorithm on df and return learned edges as a set of (i_name, j_name) tuples.

    Uses BIC score with chi-square penalty, suitable for discrete data.

    Rows with any NaN are dropped before running — causal-learn does not handle NaN natively.
    Missing data experiments rely on the degradation this causes.
    """
    df_clean = df.dropna()
    col_names = df_clean.columns.tolist()
    data_array = df_clean.apply(lambda col: col.astype("category").cat.codes).to_numpy()

    record = ges(data_array, score_func="local_score_BIC")
    g = record["G"]

    edges = set()
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            if g.graph[i, j] == -1 and g.graph[j, i] == 1:
                # directed edge i -> j
                edges.add((col_names[i], col_names[j]))

    return edges
