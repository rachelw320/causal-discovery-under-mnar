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
from src.detection.detector import detect_mnar_pairs_logistic
from src.evaluation.metrics import compute_shd, compute_fp_rate, compute_fn_rate

MECHANISMS = ["MNAR", "MNAR_THRESHOLD", "MNAR_GRADIENT", "MNAR_CORRELATED", "MNAR_MIXED"]
SEVERITY = 0.30
BOOTSTRAP_N = 30
SEED = 42


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


def run_pc(arr, col_names):
    cg = pc(arr, alpha=0.05, indep_test=chisq, show_progress=False, node_names=col_names)
    return extract_edges(cg.G.graph, col_names)


def bootstrap_mechanism(df_missing, true_edges, all_nodes, seed):
    rng = np.random.default_rng(seed)
    shds, fps, fns = [], [], []
    for _ in range(BOOTSTRAP_N):
        sample = df_missing.sample(n=len(df_missing), replace=True,
                                   random_state=int(rng.integers(1e6)))
        arr, names = to_array(sample)
        edges = run_pc(arr, names)
        shds.append(compute_shd(true_edges, edges, all_nodes))
        fps.append(compute_fp_rate(true_edges, edges, all_nodes))
        fns.append(compute_fn_rate(true_edges, edges, all_nodes))
    return np.mean(shds), np.std(shds), np.mean(fps), np.mean(fns)


def print_verification(df_original, df_missing, mechanism_name):
    """Print overall rate, per-variable rate, and per-category breakdown for first 3 columns."""
    total_cells = df_original.shape[0] * df_original.shape[1]
    overall_rate = df_missing.isna().sum().sum() / total_cells
    print(f"\n  Mechanism: {mechanism_name}")
    print(f"  Overall missingness rate: {overall_rate:.3f} (target {SEVERITY})")

    print("  Per-variable missingness rate:")
    for col in df_original.columns:
        col_rate = df_missing[col].isna().mean()
        print(f"    {col}: {col_rate:.3f}")

    print("  Per-category missingness for first 3 variables:")
    for col in df_original.columns[:3]:
        print(f"    {col}:")
        for cat_val in df_original[col].unique():
            idx = df_original[col] == cat_val
            n_total = idx.sum()
            n_missing = df_missing.loc[idx, col].isna().sum()
            print(f"      category={cat_val}: {n_missing}/{n_total} = {n_missing/n_total:.3f}")


os.makedirs(config.RESULTS_DIR, exist_ok=True)

# load Sachs
print("Loading Sachs network...")
sachs_net = load_network("sachs")
sachs_df = sample_data(sachs_net, n=5000, seed=SEED)
sachs_true = get_true_edges(sachs_net)
sachs_nodes = list(sachs_df.columns)

# load Asia
print("Loading Asia network...")
asia_net = load_network("asia")
asia_df = sample_data(asia_net, n=5000, seed=SEED)
asia_true = get_true_edges(asia_net)
asia_nodes = list(asia_df.columns)


# verification tables
print("\n=== VERIFICATION: Sachs 30% missingness ===")
for mech in MECHANISMS:
    df_m = inject_missingness(sachs_df, mech, SEVERITY, seed=SEED)
    print_verification(sachs_df, df_m, mech)


# PC bootstrap on Sachs
print("\n=== PC BOOTSTRAP: Sachs 30% ===")
print(f"{'Mechanism':<20} {'SHD mean':>10} {'SHD std':>10} {'FP mean':>10} {'FN mean':>10}")
print("-" * 62)
sachs_rows = []
for mech in MECHANISMS:
    df_m = inject_missingness(sachs_df, mech, SEVERITY, seed=SEED)
    shd_m, shd_s, fp_m, fn_m = bootstrap_mechanism(df_m, sachs_true, sachs_nodes, SEED)
    print(f"{mech:<20} {shd_m:>10.3f} {shd_s:>10.3f} {fp_m:>10.3f} {fn_m:>10.3f}")
    sachs_rows.append({
        "dataset": "sachs",
        "mechanism": mech,
        "shd_mean": round(shd_m, 3),
        "shd_std": round(shd_s, 3),
        "fp_mean": round(fp_m, 3),
        "fn_mean": round(fn_m, 3),
    })

sachs_results = pd.DataFrame(sachs_rows)
out_sachs = os.path.join(config.RESULTS_DIR, "mnar_mechanisms_sachs.csv")
sachs_results.to_csv(out_sachs, index=False)
print(f"\nSaved to {out_sachs}")


# PC bootstrap on Asia
print("\n=== PC BOOTSTRAP: Asia 30% ===")
print(f"{'Mechanism':<20} {'SHD mean':>10} {'SHD std':>10} {'FP mean':>10} {'FN mean':>10}")
print("-" * 62)
asia_rows = []
for mech in MECHANISMS:
    df_m = inject_missingness(asia_df, mech, SEVERITY, seed=SEED)
    shd_m, shd_s, fp_m, fn_m = bootstrap_mechanism(df_m, asia_true, asia_nodes, SEED)
    print(f"{mech:<20} {shd_m:>10.3f} {shd_s:>10.3f} {fp_m:>10.3f} {fn_m:>10.3f}")
    asia_rows.append({
        "dataset": "asia",
        "mechanism": mech,
        "shd_mean": round(shd_m, 3),
        "shd_std": round(shd_s, 3),
        "fp_mean": round(fp_m, 3),
        "fn_mean": round(fn_m, 3),
    })

asia_results = pd.DataFrame(asia_rows)
out_asia = os.path.join(config.RESULTS_DIR, "mnar_mechanisms_asia.csv")
asia_results.to_csv(out_asia, index=False)
print(f"\nSaved to {out_asia}")


# detection module on Sachs
print("\n=== LOGISTIC DETECTION: Sachs 30% (single run each mechanism) ===")
print(f"{'Mechanism':<20} {'Flagged pairs':>14} {'Matched true edges':>20}")
print("-" * 56)

# build set of undirected true edge pairs for matching
sachs_true_undirected = {frozenset([a, b]) for a, b in sachs_true}

detection_rows = []
for mech in MECHANISMS:
    df_m = inject_missingness(sachs_df, mech, SEVERITY, seed=SEED)
    flagged = detect_mnar_pairs_logistic(df_m)
    n_flagged = len(flagged)
    n_matched = sum(1 for (a, b, _) in flagged if frozenset([a, b]) in sachs_true_undirected)
    print(f"{mech:<20} {n_flagged:>14} {n_matched:>20}")
    detection_rows.append({"mechanism": mech, "flagged": n_flagged, "matched_true": n_matched})


# final summary table
print("\n=== FINAL SUMMARY: SHD mean and detection recall across both datasets ===")
sachs_dict = {r["mechanism"]: r for r in sachs_rows}
asia_dict = {r["mechanism"]: r for r in asia_rows}
det_dict = {r["mechanism"]: r for r in detection_rows}

print(f"{'Mechanism':<20} {'Sachs SHD':>10} {'Asia SHD':>10} {'Det recall':>12}")
print("-" * 56)
for mech in MECHANISMS:
    sachs_shd = sachs_dict[mech]["shd_mean"]
    asia_shd = asia_dict[mech]["shd_mean"]
    det = det_dict[mech]
    n_true = len(sachs_true)
    recall = det["matched_true"] / n_true if n_true > 0 else 0.0
    print(f"{mech:<20} {sachs_shd:>10.3f} {asia_shd:>10.3f} {recall:>12.3f}")
