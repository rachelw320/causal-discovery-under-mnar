import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges

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


def extract_edges_ges(g, col_names):
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
                u, v = sorted([col_names[i], col_names[j]])
                edges.add((u, v))
    return edges


def apply_orientations(edges, required_directions):
    # post-hoc: flip any edge that came out backwards relative to a known constraint
    result = set()
    for a, b in edges:
        if (b, a) in required_directions:
            result.add((b, a))
        else:
            result.add((a, b))
    return result


def run_ges(arr, col_names):
    record = ges(arr, score_func="local_score_BDeu",
                 parameters={"sample_prior": 1, "structure_prior": 0},
                 node_names=col_names)
    return extract_edges_ges(record["G"], col_names)


def bootstrap_ges(df_mnar, required_directions, true_edges, all_nodes, rng):
    shds = []
    for _ in range(config.BOOTSTRAP_ITERATIONS):
        sample = df_mnar.sample(n=len(df_mnar), replace=True,
                                random_state=int(rng.integers(1e6)))
        arr, names = to_array(sample)
        edges = run_ges(arr, names)
        if required_directions:
            edges = apply_orientations(edges, required_directions)
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

    for severity in SEVERITIES:
        print(f"  MNAR {int(severity * 100)}%")
        rng = np.random.default_rng(config.RANDOM_SEED)
        df_mnar = inject_missingness(df, "MNAR", severity)

        flagged_logistic = detect_mnar_pairs_logistic(df_mnar)
        bk_logistic = apply_constraints(df_mnar, flagged_logistic, true_edges)

        # extract required directions from background knowledge
        logistic_directions = set()
        for col_a, col_b, _ in flagged_logistic:
            if (col_a, col_b) in true_edges:
                logistic_directions.add((col_a, col_b))
            elif (col_b, col_a) in true_edges:
                logistic_directions.add((col_b, col_a))

        global_directions = set(true_edges)

        conditions = [
            ("no_constraints", set()),
            ("logistic_selective", logistic_directions),
            ("global_oracle", global_directions),
        ]

        for label, directions in conditions:
            print(f"    {label}...")
            mean_shd, std_shd = bootstrap_ges(df_mnar, directions, true_edges, all_nodes, rng)
            rows.append({
                "dataset": dataset,
                "severity": int(severity * 100),
                "condition": label,
                "shd_mean": round(mean_shd, 3),
                "shd_std": round(std_shd, 3),
            })

results_df = pd.DataFrame(rows)
out_path = os.path.join(config.RESULTS_DIR, "pipeline_ges_results.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

print(f"\n{'dataset':<8} {'severity':<10} {'condition':<22} {'mean SHD':<10} {'std'}")
print("-" * 58)
for _, row in results_df.iterrows():
    print(f"{row['dataset']:<8} {row['severity']:<10} {row['condition']:<22} "
          f"{row['shd_mean']:<10.2f} {row['shd_std']:.2f}")
