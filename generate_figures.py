import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

os.makedirs("figures", exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({"figure.dpi": 150, "axes.labelpad": 10, "xtick.major.pad": 6, "ytick.major.pad": 6})
COLORS = {"MCAR": "#4C72B0", "MAR": "#DD8452", "MNAR": "#55A868"}

# pink/purple palette for all bar charts
BAR_DARK_PURPLE  = "#5B2C8D"
BAR_MED_PURPLE   = "#9B59B6"
BAR_LIGHT_PURPLE = "#C39BD3"
BAR_DARK_PINK    = "#C0547A"
BAR_MED_PINK     = "#E8769A"
BAR_LIGHT_PINK   = "#F4B8CC"
BAR_LIGHT_BLUE   = "#E8883A"
BAR_SAGE_GREEN   = "#52A875"

# 4-bar pipeline figures: pink / blue / purple / dark-purple for maximum separation
COND_COLORS = {
    "no_constraints":     BAR_DARK_PINK,
    "chi2_selective":     BAR_LIGHT_BLUE,
    "logistic_selective": BAR_MED_PURPLE,
    "global_oracle":      BAR_DARK_PURPLE,
}
COND_LABELS = {
    "no_constraints": "No constraints",
    "chi2_selective": "Chi-square selective",
    "logistic_selective": "Logistic selective",
    "global_oracle": "Global oracle",
}

miss = pd.read_csv("results/missingness_results.csv")
miss["rate_pct"] = (miss["rate"] * 100).astype(int)
base = pd.read_csv("results/baseline_results.csv")
pipe = pd.read_csv("results/pipeline_results.csv")
pipe["severity_pct"] = (pipe["severity"] * 100).astype(int)
pipe_ges = pd.read_csv("results/pipeline_ges_results.csv")
det = pd.read_csv("results/detection_eval.csv")
sig = pd.read_csv("results/significance_results.csv")
alarm_miss = pd.read_csv("results/alarm_missingness_results.csv")
alarm_pipe = pd.read_csv("results/alarm_pipeline_results.csv")
nhanes_stab = pd.read_csv("results/nhanes_stability.csv")
nhanes_res = pd.read_csv("results/nhanes_results.csv")


def save(name, caption):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {name}.png")
    print(f"Caption: {caption}\n")


# fig 01: SHD vs severity by mechanism -- Asia PC
fig, ax = plt.subplots(figsize=(10, 6))
sub = miss[(miss["dataset"] == "asia") & (miss["algorithm"] == "PC")]
for mech, grp in sub.groupby("mechanism"):
    ax.errorbar(grp["rate_pct"], grp["shd_mean"], yerr=grp["shd_std"],
                label=mech, marker="o", color=COLORS[mech], capsize=4)
ax.set_xlabel("Missingness severity (%)")
ax.set_ylabel("Mean SHD")
ax.set_title("PC on Asia: SHD vs missingness severity by mechanism")
ax.legend(title="Mechanism")
save("01_asia_pc_shd_by_mechanism",
     "Figure 1. Mean SHD (plus/minus 1 SD) for PC on the Asia network across MCAR, MAR, and MNAR missingness at 10, 20, 30, and 50 percent severity. MNAR consistently produces higher SHD than MCAR, reflecting the non-random nature of the corruption.")


# fig 02: SHD vs severity by mechanism -- Asia GES
fig, ax = plt.subplots(figsize=(10, 6))
sub = miss[(miss["dataset"] == "asia") & (miss["algorithm"] == "GES")]
for mech, grp in sub.groupby("mechanism"):
    ax.errorbar(grp["rate_pct"], grp["shd_mean"], yerr=grp["shd_std"],
                label=mech, marker="o", color=COLORS[mech], capsize=4)
ax.set_xlabel("Missingness severity (%)")
ax.set_ylabel("Mean SHD")
ax.set_title("GES on Asia: SHD vs missingness severity by mechanism")
ax.legend(title="Mechanism")
save("02_asia_ges_shd_by_mechanism",
     "Figure 2. Mean SHD for GES on the Asia network under MCAR, MAR, and MNAR. GES shows a different degradation profile to PC under the same conditions.")


# fig 03: SHD vs severity by mechanism -- Sachs PC
fig, ax = plt.subplots(figsize=(10, 6))
sub = miss[(miss["dataset"] == "sachs") & (miss["algorithm"] == "PC")]
for mech, grp in sub.groupby("mechanism"):
    ax.errorbar(grp["rate_pct"], grp["shd_mean"], yerr=grp["shd_std"],
                label=mech, marker="o", color=COLORS[mech], capsize=4)
ax.set_xlabel("Missingness severity (%)")
ax.set_ylabel("Mean SHD")
ax.set_title("PC on Sachs: SHD vs missingness severity by mechanism")
ax.legend(title="Mechanism")
save("03_sachs_pc_shd_by_mechanism",
     "Figure 3. Mean SHD for PC on the Sachs network (11 nodes, 17 edges) across three missingness mechanisms. Sachs shows steeper MNAR degradation than Asia, reflecting higher graph density.")


# fig 04: SHD vs severity by mechanism -- Sachs GES
fig, ax = plt.subplots(figsize=(10, 6))
sub = miss[(miss["dataset"] == "sachs") & (miss["algorithm"] == "GES")]
for mech, grp in sub.groupby("mechanism"):
    ax.errorbar(grp["rate_pct"], grp["shd_mean"], yerr=grp["shd_std"],
                label=mech, marker="o", color=COLORS[mech], capsize=4)
ax.set_xlabel("Missingness severity (%)")
ax.set_ylabel("Mean SHD")
ax.set_title("GES on Sachs: SHD vs missingness severity by mechanism")
ax.legend(title="Mechanism")
save("04_sachs_ges_shd_by_mechanism",
     "Figure 4. Mean SHD for GES on the Sachs network across three missingness mechanisms and four severity levels.")


# fig 05: PC vs GES baseline comparison grouped bar
fig, ax = plt.subplots(figsize=(10, 6))
datasets = ["asia", "sachs"]
pc_means = [base[(base["dataset"] == d) & (base["algorithm"] == "PC")]["shd_mean"].values[0] for d in datasets]
ges_means = [base[(base["dataset"] == d) & (base["algorithm"] == "GES")]["shd_mean"].values[0] for d in datasets]
pc_stds = [base[(base["dataset"] == d) & (base["algorithm"] == "PC")]["shd_std"].values[0] for d in datasets]
ges_stds = [base[(base["dataset"] == d) & (base["algorithm"] == "GES")]["shd_std"].values[0] for d in datasets]
x = np.arange(len(datasets))
w = 0.35
ax.bar(x - w/2, pc_means, w, yerr=pc_stds, label="PC", color=BAR_DARK_PURPLE, capsize=5)
ax.bar(x + w/2, ges_means, w, yerr=ges_stds, label="GES", color=BAR_MED_PINK, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(["Asia", "Sachs"])
ax.set_ylabel("Mean SHD")
ax.set_title("Baseline SHD: PC vs GES on clean data")
ax.legend()
save("05_baseline_pc_vs_ges",
     "Figure 5. Baseline mean SHD for PC and GES on clean Asia and Sachs data (30 bootstrap iterations, 5000 samples, no missingness). GES achieves lower SHD on Asia; both algorithms struggle with the denser Sachs network.")


# fig 06: FP rate vs severity -- Asia and Sachs PC
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
for ax, ds in zip(axes, ["asia", "sachs"]):
    sub = miss[(miss["dataset"] == ds) & (miss["algorithm"] == "PC")]
    for mech, grp in sub.groupby("mechanism"):
        yerr_low = np.minimum(grp["fp_rate_std"], grp["fp_rate_mean"])
        ax.errorbar(grp["rate_pct"], grp["fp_rate_mean"],
                    yerr=[yerr_low, grp["fp_rate_std"]],
                    label=mech, marker="o", color=COLORS[mech], capsize=4)
    ax.set_title(f"PC FP rate -- {ds.capitalize()}")
    ax.set_xlabel("Missingness severity (%)")
    ax.set_ylabel("Mean FP rate")
    ax.legend(title="Mechanism")
save("06_fp_rate_by_mechanism",
     "Figure 6. Mean false positive edge rate for PC on Asia (left) and Sachs (right) under MCAR, MAR, and MNAR. A higher FP rate means the algorithm is adding spurious edges.")


# fig 07: FN rate vs severity -- Asia and Sachs PC
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
for ax, ds in zip(axes, ["asia", "sachs"]):
    sub = miss[(miss["dataset"] == ds) & (miss["algorithm"] == "PC")]
    for mech, grp in sub.groupby("mechanism"):
        ax.errorbar(grp["rate_pct"], grp["fn_rate_mean"], yerr=grp["fn_rate_std"],
                    label=mech, marker="o", color=COLORS[mech], capsize=4)
    ax.set_title(f"PC FN rate -- {ds.capitalize()}")
    ax.set_xlabel("Missingness severity (%)")
    ax.set_ylabel("Mean FN rate")
    ax.legend(title="Mechanism")
save("07_fn_rate_by_mechanism",
     "Figure 7. Mean false negative edge rate for PC on Asia (left) and Sachs (right). A higher FN rate means the algorithm is missing true edges, a distinct failure mode from false positives.")


# fig 08: pipeline 4 conditions -- Asia PC across severities
fig, ax = plt.subplots(figsize=(13, 7))
sub = pipe[pipe["dataset"] == "asia"]
severities = sorted(sub["severity_pct"].unique())
conditions = ["no_constraints", "chi2_selective", "logistic_selective", "global_oracle"]
x = np.arange(len(severities))
w = 0.2
for i, cond in enumerate(conditions):
    grp = sub[sub["condition"] == cond].sort_values("severity_pct")
    ax.bar(x + (i - 1.5) * w, grp["shd_mean"], w,
           yerr=grp["shd_std"], label=COND_LABELS[cond],
           color=COND_COLORS[cond], capsize=4)
ax.set_xticks(x)
ax.set_xticklabels([f"{s}%" for s in severities])
ax.set_xlabel("Missingness severity")
ax.set_ylabel("Mean SHD")
ax.set_title("Pipeline comparison: PC on Asia (all severities)")
ax.legend(loc="upper left", fontsize=9)
save("08_pipeline_asia_pc",
     "Figure 8. Mean SHD under four constraint conditions for PC on Asia across all MNAR severity levels. Logistic selective consistently approaches global oracle performance, demonstrating that targeted constraint injection substantially recovers structure learning quality.")


# fig 09: pipeline 4 conditions -- Sachs PC across severities
fig, ax = plt.subplots(figsize=(13, 7))
sub = pipe[pipe["dataset"] == "sachs"]
severities = sorted(sub["severity_pct"].unique())
x = np.arange(len(severities))
for i, cond in enumerate(conditions):
    grp = sub[sub["condition"] == cond].sort_values("severity_pct")
    ax.bar(x + (i - 1.5) * w, grp["shd_mean"], w,
           yerr=grp["shd_std"], label=COND_LABELS[cond],
           color=COND_COLORS[cond], capsize=4)
ax.set_xticks(x)
ax.set_xticklabels([f"{s}%" for s in severities])
ax.set_xlabel("Missingness severity")
ax.set_ylabel("Mean SHD")
ax.set_title("Pipeline comparison: PC on Sachs (all severities)")
ax.legend(loc="upper left", fontsize=9)
save("09_pipeline_sachs_pc",
     "Figure 9. Mean SHD under four constraint conditions for PC on Sachs. The gap between no_constraints and logistic_selective is larger on Sachs than Asia, suggesting the detection-constrained pipeline is more beneficial on denser graphs.")


# fig 10: pipeline GES -- Asia and Sachs
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
ges_conditions = ["no_constraints", "logistic_selective", "global_oracle"]
ges_colors = {
    "no_constraints":     COND_COLORS["no_constraints"],
    "logistic_selective": BAR_SAGE_GREEN,
    "global_oracle":      COND_COLORS["global_oracle"],
}
for ax, ds in zip(axes, ["asia", "sachs"]):
    sub = pipe_ges[pipe_ges["dataset"] == ds]
    sevs = sorted(sub["severity"].unique())
    x = np.arange(len(sevs))
    for i, cond in enumerate(ges_conditions):
        grp = sub[sub["condition"] == cond].sort_values("severity")
        ax.bar(x + (i - 1) * 0.25, grp["shd_mean"], 0.25,
               yerr=grp["shd_std"], label=COND_LABELS[cond],
               color=ges_colors[cond], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}%" for s in sevs])
    ax.set_xlabel("Missingness severity")
    ax.set_ylabel("Mean SHD")
    ax.set_title(f"GES pipeline -- {ds.capitalize()}")
    ax.legend(fontsize=8)
save("10_pipeline_ges",
     "Figure 10. Mean SHD under three constraint conditions for GES on Asia (left) and Sachs (right). GES with logistic-selective constraints shows strong improvement over no-constraint GES.")


# fig 11: detection precision and recall -- Asia and Sachs by detector
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, ds in zip(axes, ["asia", "sachs"]):
    sub = det[det["dataset"] == ds]
    sevs = sorted(sub["severity"].unique())
    x = np.arange(len(sevs))
    for j, (detector, ls) in enumerate([("chi2", "--"), ("logistic", "-")]):
        grp = sub[sub["detector"] == detector].sort_values("severity")
        ax.plot(sevs, grp["precision"].values, marker="s", linestyle=ls,
                color="#4C72B0", label=f"{detector} precision")
        ax.plot(sevs, grp["recall"].values, marker="o", linestyle=ls,
                color="#DD8452", label=f"{detector} recall")
    ax.set_xticks(sevs)
    ax.set_xticklabels([f"{s}%" for s in sevs])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Missingness severity (%)")
    ax.set_ylabel("Score")
    ax.set_title(f"Detection precision and recall -- {ds.capitalize()}")
    ax.legend(fontsize=8)
save("11_detection_precision_recall",
     "Figure 11. Precision and recall of chi-square and logistic detectors on Asia (left) and Sachs (right) across severity levels. The logistic detector achieves higher recall than chi-square, flagging more true MNAR-affected variable pairs.")


# fig 12: Wilcoxon p-values heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sig_pivot = sig.pivot(index="dataset", columns="severity", values="p_value")
sns.heatmap(sig_pivot, ax=ax, annot=True, fmt=".2e", cmap="YlOrRd_r",
            linewidths=0.5, cbar_kws={"label": "p-value"})
ax.set_title("Wilcoxon p-values: no constraints vs logistic selective")
ax.set_xlabel("Missingness severity (%)")
ax.set_ylabel("Dataset")
save("12_wilcoxon_pvalues_heatmap",
     "Figure 12. Wilcoxon signed-rank test p-values comparing no-constraint PC against logistic-selective PC across Asia and Sachs at each severity level. All p-values are below 0.05, confirming the improvement is statistically significant.")


# fig 13: SHD improvement from no_constraints to logistic_selective
fig, ax = plt.subplots(figsize=(10, 6))
for ds, color, marker in [("asia", "#4C72B0", "o"), ("sachs", "#DD8452", "s")]:
    sub = sig[sig["dataset"] == ds].sort_values("severity")
    improvement = sub["mean_shd_none"] - sub["mean_shd_logistic"]
    ax.plot(sub["severity"], improvement, marker=marker, color=color, label=ds.capitalize())
ax.set_xlabel("Missingness severity (%)")
ax.set_ylabel("SHD reduction\n(no constraints minus logistic selective)")
ax.set_title("SHD improvement from logistic-selective constraints")
ax.legend()
save("13_shd_improvement",
     "Figure 13. Absolute SHD reduction achieved by logistic-selective constraints versus no-constraint PC on Asia and Sachs. Larger values indicate greater benefit from the detection-constrained pipeline.")


# fig 14: Alarm SHD vs severity PC vs GES
fig, ax = plt.subplots(figsize=(10, 6))
for algo, color, marker in [("PC", "#4C72B0", "o"), ("GES", "#DD8452", "s")]:
    grp = alarm_miss[alarm_miss["algorithm"] == algo].sort_values("severity")
    ax.errorbar(grp["severity"], grp["shd_mean"], yerr=grp["shd_std"],
                label=algo, marker=marker, color=color, capsize=4)
ax.set_xlabel("Missingness severity (%)")
ax.set_ylabel("Mean SHD")
ax.set_title("Alarm network: PC vs GES under MNAR (mode-imputed)")
ax.legend()
save("14_alarm_pc_vs_ges",
     "Figure 14. Mean SHD for PC and GES on the Alarm network (37 nodes, 46 edges) under MNAR at four severity levels. Mode imputation was applied prior to structure learning because listwise deletion is infeasible at this network scale. GES achieves substantially lower SHD than PC on this larger graph.")


# fig 15: Alarm pipeline 4 conditions
fig, ax = plt.subplots(figsize=(10, 6))
conds = alarm_pipe["condition"].tolist()
means = alarm_pipe["shd_mean"].tolist()
stds = alarm_pipe["shd_std"].tolist()
colors = [COND_COLORS[c] for c in conds]
labels = [COND_LABELS[c] for c in conds]
bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5, width=0.5)
ax.set_ylabel("Mean SHD")
ax.set_title("Alarm pipeline: 4 conditions at 30% MNAR (PC)")
ax.set_xticklabels(labels, rotation=15, ha="right")
save("15_alarm_pipeline",
     "Figure 15. Mean SHD under four constraint conditions for PC on the Alarm network at 30 percent MNAR severity. The logistic-selective condition nearly matches the global oracle, replicating the pattern observed on Asia and Sachs at a larger scale.")


# fig 16: NHANES missingness per variable
nhanes_miss = nhanes_res[nhanes_res["type"] == "missingness"].copy()
nhanes_miss = nhanes_miss[nhanes_miss["value"] > 0]
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(nhanes_miss["variable"], nhanes_miss["value"], color=BAR_MED_PURPLE)
ax.set_xlabel("Variable")
ax.set_ylabel("Missingness (%)")
ax.set_title("NHANES: observed missingness per variable")
ax.set_xticklabels(nhanes_miss["variable"], rotation=30, ha="right")
save("16_nhanes_missingness",
     "Figure 16. Observed missingness rates per variable in the NHANES dataset. Variables such as BPXSY1 and BPXDI1 show over 30 percent missingness, consistent with informative missingness expected in real clinical data.")


# fig 17: NHANES edge stability -- PC vs PC constrained vs GES
top_n = 15
stab_top = nhanes_stab.head(top_n).copy()
stab_top["pair"] = stab_top["node_a"] + "\n" + stab_top["node_b"]
fig, ax = plt.subplots(figsize=(17, 7))
x = np.arange(len(stab_top))
w = 0.25
ax.bar(x - w, stab_top["pc_stability"], w, label="PC", color=BAR_DARK_PURPLE)
ax.bar(x, stab_top["pc_constrained_stability"], w, label="PC constrained", color=BAR_MED_PINK)
ax.bar(x + w, stab_top["ges_stability"], w, label="GES", color=BAR_SAGE_GREEN)
ax.set_xticks(x)
ax.set_xticklabels(stab_top["pair"], fontsize=7.5)
ax.set_ylabel("Bootstrap stability")
ax.set_ylim(0, 1.05)
ax.set_title("NHANES: edge stability -- PC vs PC constrained vs GES (top 15 edges)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
fig.subplots_adjust(bottom=0.22)
save("17_nhanes_stability",
     "Figure 17. Bootstrap edge stability for the top 15 most stable edges in the NHANES dataset under PC, constrained PC, and GES. High stability indicates the edge is recovered consistently across bootstrap samples regardless of the algorithm.")


# fig 18: SHD heatmap -- MNAR only, PC, Asia and Sachs
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, ds in zip(axes, ["asia", "sachs"]):
    sub = miss[(miss["algorithm"] == "PC") & (miss["dataset"] == ds)]
    pivot = sub.pivot(index="mechanism", columns="rate_pct", values="shd_mean")
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Mean SHD"})
    ax.set_title(f"SHD heatmap: PC on {ds.capitalize()}")
    ax.set_xlabel("Missingness severity (%)")
    ax.set_ylabel("Mechanism")
save("18_shd_heatmap_pc",
     "Figure 18. Heatmap of mean SHD for PC on Asia (left) and Sachs (right) across all mechanism and severity combinations. Darker cells indicate worse structure recovery.")


# fig 19: FP vs FN scatter -- all conditions, Asia PC
fig, ax = plt.subplots(figsize=(10, 7))
sub = miss[(miss["dataset"] == "asia") & (miss["algorithm"] == "PC")]
for mech, grp in sub.groupby("mechanism"):
    ax.scatter(grp["fp_rate_mean"], grp["fn_rate_mean"],
               label=mech, color=COLORS[mech], s=80, zorder=3)
    for _, row in grp.iterrows():
        ax.annotate(f"{int(row['rate_pct'])}%",
                    (row["fp_rate_mean"], row["fn_rate_mean"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
ax.set_xlabel("Mean FP rate")
ax.set_ylabel("Mean FN rate")
ax.set_title("FP vs FN trade-off: PC on Asia")
ax.legend(title="Mechanism")
save("19_fp_vs_fn_scatter",
     "Figure 19. Scatter plot of mean FP rate vs mean FN rate for PC on Asia across mechanisms and severity levels. Points are labelled by severity. MNAR tends to shift the algorithm toward higher FN rates, meaning true edges are missed rather than spurious ones added.")


# fig 20: pipeline SHD reduction -- Asia vs Sachs at 30%
fig, ax = plt.subplots(figsize=(12, 7))
for ds, color, offset in [("asia", BAR_DARK_PURPLE, -0.2), ("sachs", BAR_MED_PINK, 0.2)]:
    sub = pipe[(pipe["dataset"] == ds) & (pipe["severity_pct"] == 30)]
    sub = sub.set_index("condition").reindex(conditions)
    x = np.arange(len(conditions))
    ax.bar(x + offset, sub["shd_mean"], 0.35, yerr=sub["shd_std"],
           label=ds.capitalize(), color=color, capsize=4)
ax.set_xticks(np.arange(len(conditions)))
ax.set_xticklabels([COND_LABELS[c] for c in conditions], rotation=15, ha="right")
ax.set_ylabel("Mean SHD")
ax.set_title("Pipeline comparison at 30% MNAR: Asia vs Sachs (PC)")
ax.legend()
save("20_pipeline_30pct_comparison",
     "Figure 20. Direct comparison of the four-condition pipeline at 30 percent MNAR severity for Asia and Sachs under PC. This shows how the relative benefit of constraint injection scales with graph complexity.")


print("\nAll figures saved to figures/")
