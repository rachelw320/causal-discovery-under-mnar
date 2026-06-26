import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq


def run_pc(df: pd.DataFrame) -> set:
    """Run PC on df (chi-square test) and return directed edges. Drops NaN rows first."""
    df_clean = df.dropna()
    col_names = df_clean.columns.tolist()
    data_array = df_clean.apply(lambda col: col.astype("category").cat.codes).to_numpy()

    cg = pc(data_array, alpha=0.05, indep_test=chisq)

    edges = set()
    n = len(col_names)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cg.G.graph[i, j], cg.G.graph[j, i]
            if a == -1 and b == 1:
                edges.add((col_names[i], col_names[j]))
            elif a == 1 and b == -1:
                edges.add((col_names[j], col_names[i]))
            elif a == -1 and b == -1:
                # undirected edge in CPDAG: orient alphabetically
                u, v = sorted([col_names[i], col_names[j]])
                edges.add((u, v))

    return edges
