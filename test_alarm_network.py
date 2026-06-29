import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer

import config
from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs, detect_mnar_pairs_logistic
from src.detection.constrain import apply_constraints
from src.evaluation.metrics import compute_shd, compute_fp_rate, compute_fn_rate

SEVERITIES = [0.10, 0.20, 0.30, 0.50]
SEED = 42
BOOTSTRAP_N = 30


# helpers

def to_array(df):
    # Alarm has 37 nodes so listwise deletion leaves almost no complete cases.
    # Use mode imputation so PC/GES always receive a full dataset.
    imp = SimpleImputer(strategy="most_frequent")
    filled = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    return filled.apply(lambda c: c.astype("category").cat.codes).to_numpy(), filled.columns.tolist()


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


def run_pc_plain(arr, col_names):
    cg = pc(arr, alpha=0.05, indep_test=chisq, show_progress=False, node_names=col_names)
    return extract_edges(cg.G.graph, col_names)


def run_pc_bk(arr, col_names, bk):
    cg = pc(arr, alpha=0.05, indep_test=chisq, show_progress=False,
            node_names=col_names, background_knowledge=bk)
    return extract_edges(cg.G.graph, col_names)


def run_ges_plain(arr, col_names):
    record = ges(arr, score_func="local_score_BDeu",
                 parameters={"sample_prior": 1, "structure_prior": 0},
                 node_names=col_names)
    return extract_edges(record["G"].graph, col_names)


def bootstrap_pc(df, true_edges, all_nodes, bk=None):
    """Bootstrap PC, return list of SHDs."""
    rng = np.random.default_rng(SEED)
    shds = []
    for _ in range(BOOTSTRAP_N):
        sample = df.sample(n=len(df), replace=True, random_state=int(rng.integers(1e6)))
        arr, names = to_array(sample)
        if bk is not None:
            edges = run_pc_bk(arr, names, bk)
        else:
            edges = run_pc_plain(arr, names)
        shds.append(compute_shd(true_edges, edges, all_nodes))
    return shds


def bootstrap_ges(df, true_edges, all_nodes):
    """Bootstrap GES, return list of SHDs."""
    rng = np.random.default_rng(SEED)
    shds = []
    for _ in range(BOOTSTRAP_N):
        sample = df.sample(n=len(df), replace=True, random_state=int(rng.integers(1e6)))
        arr, names = to_array(sample)
        edges = run_ges_plain(arr, names)
        shds.append(compute_shd(true_edges, edges, all_nodes))
    return shds


os.makedirs(config.RESULTS_DIR, exist_ok=True)


# load Alarm
print("Loading Alarm network...")
net = load_network("alarm")
df = sample_data(net, n=5000, seed=SEED)
true_edges = get_true_edges(net)
all_nodes = list(df.columns)
print(f"Nodes: {len(all_nodes)}, True edges: {len(true_edges)}")
print(f"Node names: {all_nodes}")


# section 1: baseline on clean data

print("\n=== BASELINE: PC and GES on clean Alarm (5000 samples, 30 bootstrap) ===")

pc_shds = bootstrap_pc(df, true_edges, all_nodes)
ges_shds = bootstrap_ges(df, true_edges, all_nodes)

print(f"PC  -- mean SHD: {np.mean(pc_shds):.3f}, std: {np.std(pc_shds):.3f}")
print(f"GES -- mean SHD: {np.mean(ges_shds):.3f}, std: {np.std(ges_shds):.3f}")


# section 2: MNAR degradation at all severities

print("\n=== MNAR DEGRADATION: PC and GES at 10/20/30/50% severity ===")
print(f"{'Severity':<10} {'Algorithm':<8} {'SHD mean':>10} {'SHD std':>10}")
print("-" * 42)

missingness_rows = []
# store no-constraint PC SHD lists for Wilcoxon later
pc_no_constraint_shds = {}

for sev in SEVERITIES:
    df_m = inject_missingness(df, "MNAR", sev, seed=SEED)
    pc_shds_sev = bootstrap_pc(df_m, true_edges, all_nodes)
    ges_shds_sev = bootstrap_ges(df_m, true_edges, all_nodes)

    pc_no_constraint_shds[sev] = pc_shds_sev

    for algo, shds_sev in [("PC", pc_shds_sev), ("GES", ges_shds_sev)]:
        m, s = np.mean(shds_sev), np.std(shds_sev)
        print(f"{int(sev*100):<10} {algo:<8} {m:>10.3f} {s:>10.3f}")
        missingness_rows.append({
            "dataset": "alarm",
            "severity": int(sev * 100),
            "algorithm": algo,
            "shd_mean": round(m, 3),
            "shd_std": round(s, 3),
        })

pd.DataFrame(missingness_rows).to_csv(
    os.path.join(config.RESULTS_DIR, "alarm_missingness_results.csv"), index=False
)
print("Saved to results/alarm_missingness_results.csv")


# section 3: detection at 30%

print("\n=== DETECTION: logistic detector on Alarm at 30% MNAR ===")
df_m30 = inject_missingness(df, "MNAR", 0.30, seed=SEED)

flagged_logistic = detect_mnar_pairs_logistic(df_m30)
true_undirected = {frozenset([a, b]) for a, b in true_edges}

n_flagged = len(flagged_logistic)
n_matched = sum(1 for (a, b, _) in flagged_logistic if frozenset([a, b]) in true_undirected)
precision = n_matched / n_flagged if n_flagged > 0 else 0.0
recall = n_matched / len(true_edges) if len(true_edges) > 0 else 0.0

print(f"Flagged pairs: {n_flagged}")
print(f"Matched true edges: {n_matched}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")


# section 4: pipeline evaluation at 30% MNAR

print("\n=== PIPELINE: 4 conditions at 30% MNAR, PC, 30 bootstrap ===")

flagged_chi2 = detect_mnar_pairs(df_m30)
bk_chi2 = apply_constraints(df_m30, flagged_chi2, true_edges)
bk_logistic = apply_constraints(df_m30, flagged_logistic, true_edges)
global_flagged = [(a, b, 0.0) for a, b in true_edges]
bk_global = apply_constraints(df_m30, global_flagged, true_edges)

pipeline_conditions = [
    ("no_constraints", None),
    ("chi2_selective", bk_chi2),
    ("logistic_selective", bk_logistic),
    ("global_oracle", bk_global),
]

print(f"{'Condition':<22} {'SHD mean':>10} {'SHD std':>10}")
print("-" * 44)

pipeline_rows = []
for label, bk in pipeline_conditions:
    shds_cond = bootstrap_pc(df_m30, true_edges, all_nodes, bk=bk)
    m, s = np.mean(shds_cond), np.std(shds_cond)
    print(f"{label:<22} {m:>10.3f} {s:>10.3f}")
    pipeline_rows.append({
        "dataset": "alarm",
        "severity": 30,
        "condition": label,
        "shd_mean": round(m, 3),
        "shd_std": round(s, 3),
    })

pd.DataFrame(pipeline_rows).to_csv(
    os.path.join(config.RESULTS_DIR, "alarm_pipeline_results.csv"), index=False
)
print("Saved to results/alarm_pipeline_results.csv")


# section 5: Wilcoxon test at each severity

print("\n=== WILCOXON: no constraints vs logistic selective at each severity ===")
print(f"{'Severity':<10} {'p value':>10} {'Significant':>14}")
print("-" * 36)

for sev in SEVERITIES:
    df_m = inject_missingness(df, "MNAR", sev, seed=SEED)
    flagged_log_sev = detect_mnar_pairs_logistic(df_m)
    bk_log_sev = apply_constraints(df_m, flagged_log_sev, true_edges)

    shds_no_con = pc_no_constraint_shds[sev]
    shds_logistic = bootstrap_pc(df_m, true_edges, all_nodes, bk=bk_log_sev)

    # Wilcoxon requires at least one pair to differ
    if shds_no_con == shds_logistic:
        p = 1.0
        sig = False
    else:
        _, p = wilcoxon(shds_no_con, shds_logistic)
        sig = p < 0.05

    sig_label = "yes" if sig else "no"
    print(f"{int(sev*100):<10} {p:>10.4f} {sig_label:>14}")
