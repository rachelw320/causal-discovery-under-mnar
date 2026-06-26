import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges


def run_ges(df: pd.DataFrame) -> set:
    """Run GES on df (BDeu score) and return directed edges. Drops NaN rows first."""
    df_clean = df.dropna()
    col_names = df_clean.columns.tolist()
    data_array = df_clean.apply(lambda col: col.astype("category").cat.codes).to_numpy()

    record = ges(data_array, score_func="local_score_BDeu",
                 parameters={"sample_prior": 1, "structure_prior": 0})
    g = record["G"]

    edges = set()
    n = len(col_names)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = g.graph[i, j], g.graph[j, i]
            if a == -1 and b == 1:
                edges.add((col_names[i], col_names[j]))
            elif a == 1 and b == -1:
                edges.add((col_names[j], col_names[i]))
            elif a == -1 and b == -1:
                # undirected edge in CPDAG: orient alphabetically
                u, v = sorted([col_names[i], col_names[j]])
                edges.add((u, v))

    return edges
