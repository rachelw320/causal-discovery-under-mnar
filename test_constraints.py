import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs, detect_mnar_pairs_logistic
from src.detection.constrain import apply_constraints
from src.evaluation.metrics import compute_shd

network = load_network("sachs")
df = sample_data(network)
true_edges = get_true_edges(network)
all_nodes = list(df.columns)

df_mnar = inject_missingness(df, "MNAR", 0.30)
flagged_chi2 = detect_mnar_pairs(df_mnar)
flagged_logistic = detect_mnar_pairs_logistic(df_mnar)
print(f"Chi-square flagged: {len(flagged_chi2)},  logistic flagged: {len(flagged_logistic)}")

# shared helper to convert df to int array after dropping NaN rows
def to_array(dataframe):
    clean = dataframe.dropna()
    return clean.apply(lambda c: c.astype("category").cat.codes).to_numpy(), clean.columns.tolist()

# shared helper to extract directed edges from CPDAG
def extract_edges(graph_matrix, col_names):
    edges = set()
    n = len(col_names)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = graph_matrix[i, j], graph_matrix[j, i]
            if a == -1 and b == 1:
                edges.add((col_names[i], col_names[j]))
            elif a == 1 and b == -1:
                edges.add((col_names[j], col_names[i]))
            elif a == -1 and b == -1:
                u, v = sorted([col_names[i], col_names[j]])
                edges.add((u, v))
    return edges

data_arr, col_names = to_array(df_mnar)

# run 1: no constraints
cg_none = pc(data_arr, alpha=0.05, indep_test=chisq, show_progress=False, node_names=col_names)
edges_none = extract_edges(cg_none.G.graph, col_names)
shd_none = compute_shd(true_edges, edges_none, all_nodes)

# run 2: selective constraints from chi-square flagged pairs
bk_chi2 = apply_constraints(df_mnar, flagged_chi2, true_edges)
cg_chi2 = pc(data_arr, alpha=0.05, indep_test=chisq, show_progress=False,
             node_names=col_names, background_knowledge=bk_chi2)
edges_chi2 = extract_edges(cg_chi2.G.graph, col_names)
shd_chi2 = compute_shd(true_edges, edges_chi2, all_nodes)

# run 3: selective constraints from logistic flagged pairs
bk_logistic = apply_constraints(df_mnar, flagged_logistic, true_edges)
cg_logistic = pc(data_arr, alpha=0.05, indep_test=chisq, show_progress=False,
                 node_names=col_names, background_knowledge=bk_logistic)
edges_logistic = extract_edges(cg_logistic.G.graph, col_names)
shd_logistic = compute_shd(true_edges, edges_logistic, all_nodes)

# run 4: global constraints using all true edges as upper bound
all_flagged = [(a, b, 0.0) for a, b in true_edges]
bk_global = apply_constraints(df_mnar, all_flagged, true_edges)
cg_global = pc(data_arr, alpha=0.05, indep_test=chisq, show_progress=False,
               node_names=col_names, background_knowledge=bk_global)
edges_global = extract_edges(cg_global.G.graph, col_names)
shd_global = compute_shd(true_edges, edges_global, all_nodes)

print(f"\nSHD with no constraints:                    {shd_none}")
print(f"SHD with chi-square selective constraints:  {shd_chi2}")
print(f"SHD with logistic selective constraints:    {shd_logistic}")
print(f"SHD with global constraints:                {shd_global}")
