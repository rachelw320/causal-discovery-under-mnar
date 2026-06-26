import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs
from src.detection.constrain import apply_constraints
from src.evaluation.metrics import compute_shd

network = load_network("sachs")
df = sample_data(network)
true_edges = get_true_edges(network)
all_nodes = list(df.columns)

df_mnar = inject_missingness(df, "MNAR", 0.30)
flagged_pairs = detect_mnar_pairs(df_mnar)
print(f"Flagged pairs: {len(flagged_pairs)}")

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

# run 2: selective constraints from flagged pairs only
bk_selective = apply_constraints(df_mnar, flagged_pairs, true_edges)
cg_selective = pc(data_arr, alpha=0.05, indep_test=chisq, show_progress=False,
                  node_names=col_names, background_knowledge=bk_selective)
edges_selective = extract_edges(cg_selective.G.graph, col_names)
shd_selective = compute_shd(true_edges, edges_selective, all_nodes)

# run 3: global constraints using all true edges as upper bound
all_flagged = [(a, b, 0.0) for a, b in true_edges]
bk_global = apply_constraints(df_mnar, all_flagged, true_edges)
cg_global = pc(data_arr, alpha=0.05, indep_test=chisq, show_progress=False,
               node_names=col_names, background_knowledge=bk_global)
edges_global = extract_edges(cg_global.G.graph, col_names)
shd_global = compute_shd(true_edges, edges_global, all_nodes)

print(f"\nSHD with no constraints:          {shd_none}")
print(f"SHD with selective constraints:   {shd_selective}")
print(f"SHD with global constraints:      {shd_global}")
