import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

import config
from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs_logistic
from src.evaluation.metrics import compute_shd

NOISE_LEVELS = [0.10, 0.20, 0.30, 0.50]
NOISE_REPEATS = 10
BOOTSTRAP_N = 30
SEED = 42
SEVERITY = 0.30

BAR_DARK_PURPLE = "#5B2C8D"
BAR_DARK_PINK   = "#C0547A"
BAR_SAGE_GREEN  = "#52A875"


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


def build_clean_bk(flagged_pairs, true_edges):
    bk = BackgroundKnowledge()
    for col_a, col_b, _ in flagged_pairs:
        if (col_a, col_b) in true_edges:
            bk.add_required_by_pattern(col_a, col_b)
        elif (col_b, col_a) in true_edges:
            bk.add_required_by_pattern(col_b, col_a)
    return bk


def build_noisy_bk(flagged_pairs, true_edges, noise_rate, rng):
    bk = BackgroundKnowledge()
    for col_a, col_b, _ in flagged_pairs:
        if (col_a, col_b) in true_edges:
            src, dst = col_a, col_b
        elif (col_b, col_a) in true_edges:
            src, dst = col_b, col_a
        else:
            continue
        if rng.random() < noise_rate:
            src, dst = dst, src
        bk.add_required_by_pattern(src, dst)
    return bk


def bootstrap_pc(df, true_edges, all_nodes, bk=None):
    rng = np.random.default_rng(SEED)
    shds = []
    for _ in range(BOOTSTRAP_N):
        sample = df.sample(n=len(df), replace=True, random_state=int(rng.integers(1e6)))
        arr, names = to_array(sample)
        edges = run_pc(arr, names, bk)
        shds.append(compute_shd(true_edges, edges, all_nodes))
    return np.mean(shds)


os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("Loading Sachs network...")
net = load_network("sachs")
df = sample_data(net, n=5000, seed=SEED)
true_edges = get_true_edges(net)
all_nodes = list(df.columns)

print(f"Injecting {int(SEVERITY*100)}% MNAR missingness...")
df_mnar = inject_missingness(df, "MNAR", SEVERITY, seed=SEED)

print("Running logistic detector...")
flagged = detect_mnar_pairs_logistic(df_mnar)
print(f"  Flagged pairs: {len(flagged)}")

global_flagged = [(a, b, 0.0) for a, b in true_edges]

# baselines
print("\n--- Baselines ---")
baselines = {}
for label, bk in [
    ("no_constraints", None),
    ("logistic_clean", build_clean_bk(flagged, true_edges)),
    ("global_oracle",  build_clean_bk(global_flagged, true_edges)),
]:
    m = bootstrap_pc(df_mnar, true_edges, all_nodes, bk)
    baselines[label] = m
    print(f"  {label:<22} SHD mean={m:.3f}")

# multi-seed noisy constraints
print(f"\n--- Noisy constraints ({NOISE_REPEATS} seeds x {BOOTSTRAP_N} bootstrap each) ---")
full_rows = []
for noise in NOISE_LEVELS:
    repeat_means = []
    for rep in range(NOISE_REPEATS):
        noise_rng = np.random.default_rng(SEED * 100 + int(noise * 1000) + rep)
        bk_noisy = build_noisy_bk(flagged, true_edges, noise, noise_rng)
        m = bootstrap_pc(df_mnar, true_edges, all_nodes, bk_noisy)
        repeat_means.append(m)
        full_rows.append({
            "noise_rate": noise,
            "repeat": rep,
            "shd_mean": round(m, 3),
        })
    print(f"  {int(noise*100):>3}% noise | "
          f"mean={np.mean(repeat_means):.3f}  "
          f"std={np.std(repeat_means):.3f}  "
          f"min={np.min(repeat_means):.3f}  "
          f"max={np.max(repeat_means):.3f}")

# save full results
full_df = pd.DataFrame(full_rows)
full_path = os.path.join(config.RESULTS_DIR, "noisy_constraints_multiseed_results.csv")
full_df.to_csv(full_path, index=False)
print(f"\nFull results saved to {full_path}")

# summary table
summary_rows = []
for noise in NOISE_LEVELS:
    sub = full_df[full_df["noise_rate"] == noise]["shd_mean"]
    summary_rows.append({
        "noise_rate_pct": int(noise * 100),
        "mean_shd": round(sub.mean(), 3),
        "std_shd":  round(sub.std(), 3),
        "min_shd":  round(sub.min(), 3),
        "max_shd":  round(sub.max(), 3),
    })
summary = pd.DataFrame(summary_rows)
sum_path = os.path.join(config.RESULTS_DIR, "noisy_constraints_summary.csv")
summary.to_csv(sum_path, index=False)
print(f"Summary saved to {sum_path}")

print("\n=== SUMMARY TABLE ===")
print(f"{'Noise %':>8} {'Mean SHD':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 46)
for _, r in summary.iterrows():
    print(f"{r['noise_rate_pct']:>8} {r['mean_shd']:>10.3f} {r['std_shd']:>8.3f} "
          f"{r['min_shd']:>8.3f} {r['max_shd']:>8.3f}")

print(f"\nBaselines for reference:")
for k, v in baselines.items():
    print(f"  {k:<22} {v:.3f}")

# figure: line plot with shaded error bands
sns.set_theme(style="whitegrid", font_scale=1.2)
fig, ax = plt.subplots(figsize=(10, 6))

noise_pcts = [int(n * 100) for n in NOISE_LEVELS]
means = summary["mean_shd"].values
stds  = summary["std_shd"].values

ax.plot(noise_pcts, means, color=BAR_DARK_PURPLE, marker="o",
        linewidth=2, markersize=7, label="Noisy logistic constraints")
ax.fill_between(noise_pcts, means - stds, means + stds,
                color=BAR_DARK_PURPLE, alpha=0.2)

ax.axhline(baselines["no_constraints"], color=BAR_DARK_PINK,
           linestyle="--", linewidth=1.5, label="No constraints")
ax.axhline(baselines["logistic_clean"], color="#E8883A",
           linestyle="--", linewidth=1.5, label="Logistic clean (0% noise)")
ax.axhline(baselines["global_oracle"], color=BAR_SAGE_GREEN,
           linestyle="--", linewidth=1.5, label="Global oracle")

ax.set_xlabel("Noise level (% of constraint directions flipped)")
ax.set_ylabel("Mean SHD")
ax.set_title("PC on Sachs at 30% MNAR: robustness to noisy domain knowledge\n"
             "(shaded band = ±1 SD across 10 noise-seed repeats)")
ax.set_xticks(noise_pcts)
ax.set_xticklabels([f"{n}%" for n in noise_pcts])
ax.legend(fontsize=9)

plt.tight_layout()
fig_path = "figures/noisy_constraints_sachs.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {fig_path}")
