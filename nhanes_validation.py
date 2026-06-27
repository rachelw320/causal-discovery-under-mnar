import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import io
import numpy as np
import pandas as pd
import requests
import urllib3
from collections import Counter
from itertools import combinations
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

import config
from src.detection.detector import detect_mnar_pairs, detect_mnar_pairs_logistic

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
FILES = [
    "DEMO_J.XPT", "BMX_J.XPT", "BPX_J.XPT", "TCHOL_J.XPT",
    "HDL_J.XPT", "SMQ_J.XPT", "DIQ_J.XPT", "PAQ_J.XPT", "ALQ_J.XPT",
]
KEEP_VARS = [
    "RIDAGEYR", "RIAGENDR", "INDFMPIR", "BMXBMI", "BPXSY1",
    "BPXDI1", "LBXTC", "LBDHDD", "SMQ020", "DIQ010", "PAQ605", "ALQ130",
]
NHANES_DIR = os.path.join("data", "nhanes")
os.makedirs(NHANES_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


# download files
for fname in FILES:
    fpath = os.path.join(NHANES_DIR, fname)
    if not os.path.exists(fpath):
        print(f"Downloading {fname}...")
        r = requests.get(BASE_URL + fname, verify=False,
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        with open(fpath, "wb") as f:
            f.write(r.content)
    else:
        print(f"Already have {fname}")


# load and merge on SEQN
frames = []
for fname in FILES:
    fpath = os.path.join(NHANES_DIR, fname)
    frames.append(pd.read_sas(fpath, format="xport", encoding="utf-8"))

df = frames[0]
for other in frames[1:]:
    merge_cols = ["SEQN"] + [c for c in other.columns if c != "SEQN"]
    df = df.merge(other[merge_cols], on="SEQN", how="left")

available = [v for v in KEEP_VARS if v in df.columns]
df = df[available].copy()

# replace coded missing values
for col in ["SMQ020", "DIQ010", "PAQ605"]:
    if col in df.columns:
        df[col] = df[col].replace({7.0: np.nan, 9.0: np.nan})
if "ALQ130" in df.columns:
    df["ALQ130"] = df["ALQ130"].replace({777.0: np.nan, 999.0: np.nan})

print(f"\nMerged shape: {df.shape}")
print(f"Variables: {df.columns.tolist()}")


# missingness report
print("\nMissingness report:")
miss_rows = []
for col in df.columns:
    pct = round(df[col].isna().mean() * 100, 2)
    print(f"  {col:<12}  {pct:.1f}%")
    miss_rows.append({"type": "missingness", "variable": col, "value": pct})


# discretise into 3 categories using tertile binning
def discretise(series):
    if series.dropna().nunique() <= 3:
        codes = series.astype("category").cat.codes.astype(float)
        codes[series.isna()] = np.nan
        return codes
    try:
        return pd.qcut(series, q=3, labels=[0, 1, 2]).astype(float)
    except ValueError:
        # rank first to get even bins on skewed data
        ranked = series.rank(method="first", na_option="keep")
        return pd.qcut(ranked, q=3, labels=[0, 1, 2]).astype(float)

df_disc = pd.DataFrame({col: discretise(df[col]) for col in df.columns})

df_complete = df_disc.dropna()
print(f"\nComplete cases after discretisation: {len(df_complete)} / {len(df_disc)}")


# helpers shared across PC and GES runs
def to_array(dataframe):
    return dataframe.values.astype(int), dataframe.columns.tolist()


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


def run_pc(dataframe, bk=None):
    arr, names = to_array(dataframe)
    cg = pc(arr, alpha=0.05, indep_test=chisq, show_progress=False,
            node_names=names, background_knowledge=bk)
    return extract_edges(cg.G.graph, names)


def run_ges(dataframe):
    arr, names = to_array(dataframe)
    record = ges(arr, score_func="local_score_BDeu",
                 parameters={"sample_prior": 1, "structure_prior": 0},
                 node_names=names)
    return extract_edges(record["G"].graph, names)


def bootstrap_stability(df_comp, run_fn, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    counts = Counter()
    for i in range(config.BOOTSTRAP_ITERATIONS):
        sample = df_comp.sample(n=len(df_comp), replace=True,
                                random_state=int(rng.integers(1e6)))
        for e in run_fn(sample):
            counts[e] += 1
        if (i + 1) % 10 == 0:
            print(f"    iteration {i + 1}/{config.BOOTSTRAP_ITERATIONS}")
    return {e: round(c / config.BOOTSTRAP_ITERATIONS, 3) for e, c in counts.items()}


# bootstrap edge stability on complete cases
print("\nBootstrapping PC (unconstrained)...")
pc_stability = bootstrap_stability(df_complete, run_pc)

print("Bootstrapping GES...")
ges_stability = bootstrap_stability(df_complete, run_ges)

top15 = sorted(pc_stability.items(), key=lambda x: -x[1])[:15]
print("\nTop 15 most stable PC edges:")
for edge, freq in top15:
    print(f"  {edge[0]} -> {edge[1]:<14}  {freq:.0%}")


# detection on real NHANES missing data
print("\nRunning detection on real NHANES missingness...")
flagged_chi2 = detect_mnar_pairs(df)
flagged_logistic = detect_mnar_pairs_logistic(df)
print(f"Chi-square flagged pairs: {len(flagged_chi2)}")
print(f"Logistic flagged pairs:   {len(flagged_logistic)}")


# no ground truth for NHANES so orient flagged pairs alphabetically
def build_nhanes_bk(flagged_pairs):
    bk = BackgroundKnowledge()
    for col_a, col_b, _ in flagged_pairs:
        a, b = sorted([col_a, col_b])
        bk.add_required_by_pattern(a, b)
    return bk


bk_logistic = build_nhanes_bk(flagged_logistic)

print("Bootstrapping PC (logistic constrained)...")
pc_constrained_stability = bootstrap_stability(
    df_complete, lambda d: run_pc(d, bk=bk_logistic)
)

# edge stability comparison
print("\nEdge stability: top 15 unconstrained PC edges")
print(f"  {'edge':<30} {'unconstrained':<16} {'constrained':<14} {'GES'}")
print("  " + "-" * 68)
for edge, freq in top15:
    freq_const = pc_constrained_stability.get(edge, 0.0)
    freq_ges = ges_stability.get(edge, ges_stability.get((edge[1], edge[0]), 0.0))
    print(f"  {edge[0]+' -> '+edge[1]:<30} {freq:<16.0%} {freq_const:<14.0%} {freq_ges:.0%}")


# save results
summary_rows = miss_rows.copy()
summary_rows += [
    {"type": "detection", "variable": "chi2_flagged_pairs", "value": len(flagged_chi2)},
    {"type": "detection", "variable": "logistic_flagged_pairs", "value": len(flagged_logistic)},
    {"type": "complete_cases", "variable": "n_complete", "value": len(df_complete)},
    {"type": "complete_cases", "variable": "n_total", "value": len(df_disc)},
]
pd.DataFrame(summary_rows).to_csv(
    os.path.join(config.RESULTS_DIR, "nhanes_results.csv"), index=False
)

col_names = df_complete.columns.tolist()
stability_rows = []
for a, b in combinations(col_names, 2):
    for edge in [(a, b), (b, a)]:
        pc_f = pc_stability.get(edge, 0.0)
        pc_c = pc_constrained_stability.get(edge, 0.0)
        ges_f = ges_stability.get(edge, 0.0)
        if pc_f > 0 or pc_c > 0 or ges_f > 0:
            stability_rows.append({
                "node_a": edge[0], "node_b": edge[1],
                "pc_stability": pc_f,
                "pc_constrained_stability": pc_c,
                "ges_stability": ges_f,
            })

pd.DataFrame(stability_rows).sort_values("pc_stability", ascending=False).to_csv(
    os.path.join(config.RESULTS_DIR, "nhanes_stability.csv"), index=False
)

print(f"\nSaved: results/nhanes_results.csv")
print(f"Saved: results/nhanes_stability.csv")
