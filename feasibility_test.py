import subprocess
import sys

def _install(package):
    try:
        __import__(package.replace("-", "_").split(">=")[0])
    except ImportError:
        print(f"Installing this package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

_install("causal-learn")
_install("pgmpy")

import numpy as np
import pandas as pd
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq


print("\n" + "="*60)
print("STEP 1: Load Sachs network and sample data")
print("="*60)

network = get_example_model("sachs")
df = BayesianModelSampling(network).forward_sample(size=1000, seed=42)

node_names = list(df.columns)
true_edges = set(network.edges())

print(f"Nodes ({len(node_names)}): {node_names}")
print(f"True edges ({len(true_edges)}): {sorted(true_edges)}")
print(f"Sample shape: {df.shape}")


def to_array(dataframe):
    return dataframe.apply(lambda c: c.astype("category").cat.codes).to_numpy()


def extract_edges(graph_matrix, names):
    edges = set()
    n = len(names)
    for i in range(n):
        for j in range(n):
            if graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:  # directed edge i -> j
                edges.add((names[i], names[j]))
    return edges


def shd(true, learned):
    return len(true.symmetric_difference(learned))


print("\n" + "="*60)
print("STEP 2: PC algorithm — clean data")
print("="*60)

data_clean = to_array(df)
cg_clean = pc(data_clean, alpha=0.05, indep_test=chisq, show_progress=False)
edges_pc_clean = extract_edges(cg_clean.G.graph, node_names)
shd_pc_clean = shd(true_edges, edges_pc_clean)

print(f"Learned edges: {sorted(edges_pc_clean)}")
print(f"SHD (PC, clean): {shd_pc_clean}")


print("\n" + "="*60)
print("STEP 3: GES algorithm — clean data")
print("="*60)

record = ges(data_clean, score_func="local_score_BDeu", parameters={"sample_prior": 1})
edges_ges_clean = extract_edges(record["G"].graph, node_names)
shd_ges_clean = shd(true_edges, edges_ges_clean)

print(f"Learned edges: {sorted(edges_ges_clean)}")
print(f"SHD (GES, clean): {shd_ges_clean}")


print("\n" + "="*60)
print("STEP 4: PC — MCAR 20%")
print("="*60)

rng = np.random.default_rng(42)
df_mcar = df.copy().astype(object)
df_mcar[rng.random(df_mcar.shape) < 0.20] = np.nan

df_mcar_complete = df_mcar.dropna()
print(f"Rows retained after listwise deletion: {len(df_mcar_complete)}/{len(df)} "
      f"({len(df_mcar_complete)/len(df):.0%})")

cg_mcar = pc(to_array(df_mcar_complete), alpha=0.05, indep_test=chisq, show_progress=False)
edges_pc_mcar = extract_edges(cg_mcar.G.graph, node_names)
shd_pc_mcar = shd(true_edges, edges_pc_mcar)

print(f"SHD (PC, MCAR 20%): {shd_pc_mcar}")


print("\n" + "="*60)
print("STEP 5: PC — MNAR 20% (high values preferentially missing)")
print("="*60)

df_mnar = df.copy().astype(object)

for col in df.columns:
    codes = df[col].astype("category").cat.codes
    above_median = codes > codes.median()
    prob = np.where(above_median, 0.40, 0.05)  # above median: 40% chance missing
    df_mnar.loc[rng.random(len(df)) < prob, col] = np.nan

print(f"Actual overall missingness rate: {df_mnar.isna().mean().mean():.1%}")

df_mnar_complete = df_mnar.dropna()
print(f"Rows retained after listwise deletion: {len(df_mnar_complete)}/{len(df)} "
      f"({len(df_mnar_complete)/len(df):.0%})")

cg_mnar = pc(to_array(df_mnar_complete), alpha=0.05, indep_test=chisq, show_progress=False)
edges_pc_mnar = extract_edges(cg_mnar.G.graph, node_names)
shd_pc_mnar = shd(true_edges, edges_pc_mnar)

print(f"SHD (PC, MNAR 20%): {shd_pc_mnar}")


print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"  PC  — Clean data : SHD = {shd_pc_clean}")
print(f"  GES — Clean data : SHD = {shd_ges_clean}")
print(f"  PC  — MCAR  20%  : SHD = {shd_pc_mcar}   (random missingness)")
print(f"  PC  — MNAR  20%  : SHD = {shd_pc_mnar}   (informative missingness)")
print()

if shd_pc_mnar > shd_pc_mcar:
    print(f"MNAR is worse than MCAR by {shd_pc_mnar - shd_pc_mcar} SHD point(s). Feasibility confirmed.")
elif shd_pc_mnar == shd_pc_mcar:
    print("MNAR and MCAR tied — likely small-sample variance. Use n=5000 + 30 bootstrap iterations.")
else:
    print("MNAR came out lower than MCAR — small-sample fluke. Use n=5000 + 30 bootstrap iterations.")
