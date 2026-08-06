import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import subprocess

def _install(pkg):
    try:
        __import__(pkg.replace("-", "_").split(">=")[0])
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_install("yfinance")
_install("pandas-datareader")

import numpy as np
import pandas as pd
import requests
import urllib3
from collections import Counter
from itertools import combinations

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import yfinance as yf
import pandas_datareader.data as pdr
from curl_cffi import requests as crequests

# curl_cffi session for yfinance (bypasses SSL on Windows Python 3.14)
_yf_session = crequests.Session(verify=False)

# requests session for pandas-datareader / FRED
_session = requests.Session()
_session.verify = False

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq

import config
from src.detection.detector import detect_mnar_pairs, detect_mnar_pairs_logistic

START = "2000-01-01"
END = "2024-12-31"

TICKERS = {
    "^GSPC": "SP500",
    "^VIX": "VIX",
    "^TNX": "TNX",
    "CL=F": "OIL",
    "GC=F": "GOLD",
    "DX-Y.NYB": "USD",
}

# FRED fallback for each market ticker (in case yfinance is rate limited)
FRED_MARKET_FALLBACK = {
    "SP500": "NASDAQCOM",     # NASDAQ composite, daily from 1971
    "VIX": "VIXCLS",          # CBOE VIX, daily from 1990
    "TNX": "DGS10",           # 10Y Treasury, daily from 1962
    "OIL": "DCOILWTICO",      # WTI crude oil, daily from 1986
    "GOLD": "PPIACO",         # PPI all commodities proxy, monthly from 1913
    "USD": "TWEXBGSMTH",      # broad USD index, monthly from 2006
}

FRED_SERIES = {
    "UNRATE": "UNRATE",
    "CPIAUCSL": "CPI",
    "FEDFUNDS": "FEDFUNDS",
    "HOUST": "HOUST",
    "UMCSENT": "UMCSENT",
    "BAA": "CREDSPREAD",  # Moody's BAA yield, proxy for credit conditions
}

KNOWN_RELATIONS = {
    frozenset(["SP500", "VIX"]): "equity / fear index",
    frozenset(["SP500", "TNX"]): "equity / interest rates",
    frozenset(["SP500", "OIL"]): "equity / oil",
    frozenset(["SP500", "CREDSPREAD"]): "equity / credit risk",
    frozenset(["SP500", "UMCSENT"]): "equity / consumer confidence",
    frozenset(["TNX", "FEDFUNDS"]): "treasury / fed policy",
    frozenset(["TNX", "CPI"]): "treasury / inflation",
    frozenset(["FEDFUNDS", "CPI"]): "fed policy / inflation",
    frozenset(["FEDFUNDS", "UNRATE"]): "fed policy / unemployment",
    frozenset(["VIX", "CREDSPREAD"]): "volatility / credit spreads",
    frozenset(["OIL", "CPI"]): "oil / inflation",
    frozenset(["GOLD", "USD"]): "gold / dollar",
    frozenset(["GOLD", "CPI"]): "gold / inflation hedge",
    frozenset(["HOUST", "FEDFUNDS"]): "housing / interest rates",
    frozenset(["UMCSENT", "UNRATE"]): "consumer confidence / unemployment",
}

os.makedirs(config.RESULTS_DIR, exist_ok=True)


def fred_monthly(series_id, name):
    raw = pdr.DataReader(series_id, "fred", START, END, session=_session).squeeze()
    s = raw.resample("ME").last().dropna()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s


# try yfinance first; fall back to FRED equivalent for any failed ticker
import time
print("Downloading market data (yfinance, with FRED fallback)...")
market_frames = {}
ticker_list = list(TICKERS.keys())
yf_results = {}

time.sleep(5)
try:
    raw_all = yf.download(ticker_list, start=START, end=END, auto_adjust=True,
                          progress=False, group_by="ticker", session=_yf_session)
    for ticker, name in TICKERS.items():
        try:
            lvl0 = raw_all.columns.get_level_values(0)
            if ticker in lvl0:
                s = raw_all[ticker]["Close"].resample("ME").last().dropna()
            else:
                s = raw_all["Close"][ticker].resample("ME").last().dropna()
            if len(s) > 0:
                s.index = s.index.to_period("M").to_timestamp("M")
                market_frames[name] = s
                yf_results[name] = True
                print(f"  {name}: {len(s)} months (yfinance)")
            else:
                yf_results[name] = False
        except Exception:
            yf_results[name] = False
except Exception as e:
    print(f"  yfinance batch failed: {e}")
    for name in TICKERS.values():
        yf_results[name] = False

# FRED fallback for any ticker that failed yfinance
for name, fred_id in FRED_MARKET_FALLBACK.items():
    if not yf_results.get(name, False):
        try:
            s = fred_monthly(fred_id, name)
            market_frames[name] = s
            print(f"  {name}: {len(s)} months (FRED fallback: {fred_id})")
        except Exception as e:
            print(f"  {name}: all sources failed ({e})")


# download FRED macro series
print("Downloading FRED macro data...")
fred_frames = {}
for series_id, name in FRED_SERIES.items():
    try:
        s = fred_monthly(series_id, name)
        fred_frames[name] = s
        print(f"  {name}: {len(s)} months")
    except Exception as e:
        print(f"  {name}: failed ({e})")


# merge on date
all_series = {**market_frames, **fred_frames}
df = pd.DataFrame(all_series)
df.index.name = "date"
df = df.sort_index()

print(f"\nMerged shape: {df.shape}")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Variables: {df.columns.tolist()}")


# missingness report
print("\nMissingness report:")
miss_rows = []
for col in df.columns:
    pct = round(df[col].isna().mean() * 100, 2)
    print(f"  {col:<14}  {pct:.1f}%")
    miss_rows.append({"type": "missingness", "variable": col, "value": pct})


# discretise into 3 tertile bins
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


# helpers shared across PC and GES
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


def run_pc(dataframe):
    arr, names = to_array(dataframe)
    cg = pc(arr, alpha=0.05, indep_test=chisq, show_progress=False, node_names=names)
    return extract_edges(cg.G.graph, names)


def run_ges(dataframe):
    arr, names = to_array(dataframe)
    record = ges(arr, score_func="local_score_BDeu",
                 parameters={"sample_prior": 1, "structure_prior": 0},
                 node_names=names)
    return extract_edges(record["G"].graph, names)


def bootstrap_stability(df_comp, run_fn, label, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    counts = Counter()
    for i in range(config.BOOTSTRAP_ITERATIONS):
        sample = df_comp.sample(n=len(df_comp), replace=True,
                                random_state=int(rng.integers(1e6)))
        for e in run_fn(sample):
            counts[e] += 1
        if (i + 1) % 10 == 0:
            print(f"  {label} iteration {i + 1}/{config.BOOTSTRAP_ITERATIONS}")
    return {e: round(c / config.BOOTSTRAP_ITERATIONS, 3) for e, c in counts.items()}


print("\nBootstrapping PC (30 iterations)...")
pc_stability = bootstrap_stability(df_complete, run_pc, "PC")

print("Bootstrapping GES (30 iterations)...")
ges_stability = bootstrap_stability(df_complete, run_ges, "GES")


# top 15 most stable PC edges with GES comparison and financial context
top15 = sorted(pc_stability.items(), key=lambda x: -x[1])[:15]
print("\nTop 15 most stable PC edges:")
print(f"  {'edge':<28} {'PC':<10} {'GES':<10} {'financial context'}")
print("  " + "-" * 72)
for edge, freq in top15:
    freq_ges = ges_stability.get(edge, ges_stability.get((edge[1], edge[0]), 0.0))
    context = KNOWN_RELATIONS.get(frozenset(edge), "")
    print(f"  {edge[0] + ' -> ' + edge[1]:<28} {freq:<10.0%} {freq_ges:<10.0%} {context}")


# detection on real missing data
print("\nRunning detection on real missing data...")
flagged_chi2 = detect_mnar_pairs(df)
flagged_logistic = detect_mnar_pairs_logistic(df)
print(f"Chi-square flagged pairs:  {len(flagged_chi2)}")
print(f"Logistic flagged pairs:    {len(flagged_logistic)}")


# save summary results
summary_rows = miss_rows.copy()
summary_rows += [
    {"type": "detection", "variable": "chi2_flagged_pairs", "value": len(flagged_chi2)},
    {"type": "detection", "variable": "logistic_flagged_pairs", "value": len(flagged_logistic)},
    {"type": "complete_cases", "variable": "n_complete", "value": len(df_complete)},
    {"type": "complete_cases", "variable": "n_total", "value": len(df_disc)},
]
pd.DataFrame(summary_rows).to_csv(
    os.path.join(config.RESULTS_DIR, "finance_results.csv"), index=False
)


# save edge stability
col_names = df_complete.columns.tolist()
stability_rows = []
for a, b in combinations(col_names, 2):
    for edge in [(a, b), (b, a)]:
        pc_f = pc_stability.get(edge, 0.0)
        ges_f = ges_stability.get(edge, 0.0)
        if pc_f > 0 or ges_f > 0:
            stability_rows.append({
                "node_a": edge[0], "node_b": edge[1],
                "pc_stability": pc_f,
                "ges_stability": ges_f,
            })

pd.DataFrame(stability_rows).sort_values("pc_stability", ascending=False).to_csv(
    os.path.join(config.RESULTS_DIR, "finance_stability.csv"), index=False
)

print(f"\nSaved: results/finance_results.csv")
print(f"Saved: results/finance_stability.csv")
