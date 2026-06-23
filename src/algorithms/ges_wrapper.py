import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges


def run_ges(df: pd.DataFrame) -> set:
    """Run GES on df (BDeu score) and return directed edges. Drops NaN rows first."""
    df_clean = df.dropna()
    col_names = df_clean.columns.tolist()
    data_array = df_clean.apply(lambda col: col.astype("category").cat.codes).to_numpy()

    record = ges(data_array, score_func="local_score_BDeu", parameters={"sample_prior": 1})
    g = record["G"]

    edges = set()
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            if g.graph[i, j] == -1 and g.graph[j, i] == 1:  # directed edge i -> j
                edges.add((col_names[i], col_names[j]))

    return edges
