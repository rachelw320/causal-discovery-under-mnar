import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

import config
from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs_logistic
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


def run_pc_for_edges(arr, names, bk=None):
    cg = pc(arr, alpha=0.05, indep_test=chisq, show_progress=False,
            node_names=names, background_knowledge=bk)
    return extract_edges(cg.G.graph, names)


def collect_paired_shds(df_mnar, bk_logistic, true_edges, all_nodes, bootstrap_seeds):
    """Return paired SHD lists using the same bootstrap resample for both conditions."""
    if len(bootstrap_seeds) == 0:
        raise ValueError("bootstrap_seeds must not be empty")

    shds_none = []
    shds_logistic = []

    for seed in bootstrap_seeds:
        sample = df_mnar.sample(n=len(df_mnar), replace=True, random_state=int(seed))
        clean = sample.dropna()
        if len(clean) < 3:
            raise ValueError(f"Insufficient complete cases for bootstrap seed {seed}: {len(clean)}")

        arr = clean.apply(lambda c: c.astype("category").cat.codes).to_numpy()
        names = clean.columns.tolist()

        try:
            edges_none = run_pc_for_edges(arr, names, None)
        except Exception as exc:
            raise RuntimeError(f"PC failed for no-constraints bootstrap seed {seed}: {exc}") from exc

        try:
            edges_logistic = run_pc_for_edges(arr, names, bk_logistic)
        except Exception as exc:
            raise RuntimeError(f"PC failed for logistic constraints bootstrap seed {seed}: {exc}") from exc

        shds_none.append(compute_shd(true_edges, edges_none, all_nodes))
        shds_logistic.append(compute_shd(true_edges, edges_logistic, all_nodes))

    if len(shds_none) != len(shds_logistic):
        raise ValueError("Paired SHD lists do not have the same length")

    return shds_none, shds_logistic


os.makedirs(config.RESULTS_DIR, exist_ok=True)
rows = []

for dataset in config.DATASETS:
    print(f"\n=== {dataset.upper()} ===")
    network = load_network(dataset)
    df = sample_data(network)
    true_edges = get_true_edges(network)
    all_nodes = list(df.columns)

    for severity in SEVERITIES:
        print(f"  MNAR {int(severity * 100)}%")
        df_mnar = inject_missingness(df, "MNAR", severity)

        flagged_logistic = detect_mnar_pairs_logistic(df_mnar)
        bk_logistic = apply_constraints(df_mnar, flagged_logistic, true_edges)

        rng = np.random.default_rng(config.RANDOM_SEED)
        bootstrap_seeds = rng.integers(1_000_000, size=config.BOOTSTRAP_ITERATIONS)

        print("    collecting paired no-constraints and logistic scores...")
        shds_none, shds_logistic = collect_paired_shds(df_mnar, bk_logistic, true_edges, all_nodes, bootstrap_seeds)

        diffs = np.array(shds_none) - np.array(shds_logistic)
        if np.allclose(diffs, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(shds_none, shds_logistic, zero_method="wilcox")

        rows.append({
            "dataset": dataset,
            "severity": int(severity * 100),
            "mean_shd_none": round(np.mean(shds_none), 3),
            "mean_shd_logistic": round(np.mean(shds_logistic), 3),
            "p_value": float(p),
            "significant": p < 0.05,
        })

results_df = pd.DataFrame(rows)
out_path = os.path.join(config.RESULTS_DIR, "significance_results.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

print(f"\n{'dataset':<8} {'severity':<10} {'SHD none':<12} {'SHD logistic':<15} {'p value':<12} {'significant'}")
print("-" * 65)
for _, row in results_df.iterrows():
    print(f"{row['dataset']:<8} {row['severity']:<10} {row['mean_shd_none']:<12.2f} "
          f"{row['mean_shd_logistic']:<15.2f} {float(row['p_value']):.6e} {row['significant']}")
