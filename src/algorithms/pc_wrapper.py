import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq


def run_pc(df: pd.DataFrame) -> set:
    """
    Run the PC algorithm on df and return learned edges as a set of (i_name, j_name) tuples.

    Uses chi-square conditional independence test, suitable for categorical/discrete data
    as produced by pgmpy benchmark networks.

    Rows with any NaN are dropped before running — causal-learn does not handle NaN natively.
    Missing data experiments rely on the degradation this causes.
    """
    df_clean = df.dropna()
    col_names = df_clean.columns.tolist()
    data_array = df_clean.apply(lambda col: col.astype("category").cat.codes).to_numpy()

    cg = pc(data_array, alpha=0.05, indep_test=chisq)

    edges = set()
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            if cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == 1:
                # directed edge i -> j
                edges.add((col_names[i], col_names[j]))

    return edges
