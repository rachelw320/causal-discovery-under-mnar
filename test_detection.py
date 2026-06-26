import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs, detect_mnar_pairs_logistic

network = load_network("sachs")
df = sample_data(network)
true_edges = get_true_edges(network)

df_mnar = inject_missingness(df, "MNAR", 0.30)
flagged_clean_chi2 = detect_mnar_pairs(df)

flagged_chi2 = detect_mnar_pairs(df_mnar)
flagged_logistic = detect_mnar_pairs_logistic(df_mnar)

# undirected true pairs for overlap check
true_pairs = {(a, b) for a, b in true_edges} | {(b, a) for a, b in true_edges}
n_true = len(true_edges)

def overlap_stats(flagged):
    matched = [(a, b, p) for a, b, p in flagged if (a, b) in true_pairs or (b, a) in true_pairs]
    tp = len(matched)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / n_true
    return matched, tp, precision, recall

matched_chi2, tp_chi2, prec_chi2, rec_chi2 = overlap_stats(flagged_chi2)
matched_log, tp_log, prec_log, rec_log = overlap_stats(flagged_logistic)

print(f"True edges in Sachs: {n_true}")
print(f"Clean data control (chi-square): {len(flagged_clean_chi2)} pairs flagged\n")

print(f"Chi-square:  {len(flagged_chi2)} flagged, {tp_chi2} match true edges  "
      f"precision={prec_chi2:.2f}  recall={rec_chi2:.2f}")
print(f"Logistic:    {len(flagged_logistic)} flagged, {tp_log} match true edges  "
      f"precision={prec_log:.2f}  recall={rec_log:.2f}")

print("\nChi-square flagged pairs:")
for col_a, col_b, p in sorted(flagged_chi2, key=lambda x: x[2]):
    mark = "*" if (col_a, col_b) in true_pairs or (col_b, col_a) in true_pairs else " "
    print(f"  {mark} {col_a} -- {col_b}  p={p:.4e}")

print("\nLogistic flagged pairs:")
for col_a, col_b, p in sorted(flagged_logistic, key=lambda x: x[2]):
    mark = "*" if (col_a, col_b) in true_pairs or (col_b, col_a) in true_pairs else " "
    print(f"  {mark} {col_a} -- {col_b}  p={p:.4e}")

print("\n* = matches a true Sachs edge")
