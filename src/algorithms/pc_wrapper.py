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
        for j in range(n):
            if cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == 1:  # directed edge i -> j
                edges.add((col_names[i], col_names[j]))

    return edges
