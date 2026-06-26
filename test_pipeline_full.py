import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

import config
from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs, detect_mnar_pairs_logistic
from src.detection.constrain import apply_constraints
from src.evaluation.metrics import compute_shd

SEVERITIES = [0.10, 0.20, 0.30, 0.50]


def to_array(dataframe):
    clean = dataframe.dropna()
    return clean.apply(lambda c: c.astype("category").cat.codes).to_numpy(), clean.columns.tolist()


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


def run_pc(arr, col_names, bk=None):
    cg = pc(arr, alpha=0.05, indep_test=chisq, show_progress=False,
            node_names=col_names, background_knowledge=bk)
    return extract_edges(cg.G.graph, col_names)


def bootstrap_condition(df_mnar, bk, true_edges, all_nodes, rng):
    shds = []
    for _ in range(config.BOOTSTRAP_ITERATIONS):
        sample = df_mnar.sample(n=len(df_mnar), replace=True,
                                random_state=int(rng.integers(1e6)))
        arr, names = to_array(sample)
        edges = run_pc(arr, names, bk)
        shds.append(compute_shd(true_edges, edges, all_nodes))
    return np.mean(shds), np.std(shds)


os.makedirs(config.RESULTS_DIR, exist_ok=True)
rows = []

for dataset in config.DATASETS:
    print(f"\n=== {dataset.upper()} ===")
    network = load_network(dataset)
    df = sample_data(network)
    true_edges = get_true_edges(network)
    all_nodes = list(df.columns)
    global_flagged = [(a, b, 0.0) for a, b in true_edges]

    for severity in SEVERITIES:
        print(f"  MNAR {int(severity * 100)}%")
        rng = np.random.default_rng(config.RANDOM_SEED)
        df_mnar = inject_missingness(df, "MNAR", severity)

        flagged_chi2 = detect_mnar_pairs(df_mnar)
        flagged_logistic = detect_mnar_pairs_logistic(df_mnar)

        bk_chi2 = apply_constraints(df_mnar, flagged_chi2, true_edges)
        bk_logistic = apply_constraints(df_mnar, flagged_logistic, true_edges)
        bk_global = apply_constraints(df_mnar, global_flagged, true_edges)

        conditions = [
            ("no_constraints", None),
            ("chi2_selective", bk_chi2),
            ("logistic_selective", bk_logistic),
            ("global_oracle", bk_global),
        ]

        for label, bk in conditions:
            print(f"    {label}...")
            mean_shd, std_shd = bootstrap_condition(df_mnar, bk, true_edges, all_nodes, rng)
            rows.append({
                "dataset": dataset,
                "severity": severity,
                "condition": label,
                "shd_mean": round(mean_shd, 3),
                "shd_std": round(std_shd, 3),
            })

results_df = pd.DataFrame(rows)
out_path = os.path.join(config.RESULTS_DIR, "pipeline_results.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

print("\nSummary:")
print(f"{'dataset':<8} {'severity':<10} {'condition':<22} {'mean SHD':<10} {'std'}")
print("-" * 60)
for _, row in results_df.iterrows():
    print(f"{row['dataset']:<8} {int(row['severity']*100):<10} {row['condition']:<22} "
          f"{row['shd_mean']:<10.2f} {row['shd_std']:.2f}")
