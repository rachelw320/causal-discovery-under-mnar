import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

import config
from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs, detect_mnar_pairs_logistic

SEVERITIES = [0.10, 0.20, 0.30, 0.50]


def precision_recall(flagged, true_edges):
    true_pairs = {(a, b) for a, b in true_edges} | {(b, a) for a, b in true_edges}
    tp = sum(1 for a, b, _ in flagged if (a, b) in true_pairs or (b, a) in true_pairs)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(true_edges) if true_edges else 0.0
    return tp, round(precision, 3), round(recall, 3)


os.makedirs(config.RESULTS_DIR, exist_ok=True)
rows = []

for dataset in config.DATASETS:
    print(f"\n=== {dataset.upper()} ===")
    network = load_network(dataset)
    df = sample_data(network)
    true_edges = get_true_edges(network)

    for severity in SEVERITIES:
        df_mnar = inject_missingness(df, "MNAR", severity)

        flagged_chi2 = detect_mnar_pairs(df_mnar)
        flagged_logistic = detect_mnar_pairs_logistic(df_mnar)

        tp_chi2, prec_chi2, rec_chi2 = precision_recall(flagged_chi2, true_edges)
        tp_log, prec_log, rec_log = precision_recall(flagged_logistic, true_edges)

        rows.append({
            "dataset": dataset, "severity": int(severity * 100), "detector": "chi2",
            "flagged": len(flagged_chi2), "true_positives": tp_chi2,
            "precision": prec_chi2, "recall": rec_chi2,
        })
        rows.append({
            "dataset": dataset, "severity": int(severity * 100), "detector": "logistic",
            "flagged": len(flagged_logistic), "true_positives": tp_log,
            "precision": prec_log, "recall": rec_log,
        })

        print(f"  MNAR {int(severity*100)}%  "
              f"chi2: flagged={len(flagged_chi2)} tp={tp_chi2} prec={prec_chi2:.2f} rec={rec_chi2:.2f}  |  "
              f"logistic: flagged={len(flagged_logistic)} tp={tp_log} prec={prec_log:.2f} rec={rec_log:.2f}")

results_df = pd.DataFrame(rows)
out_path = os.path.join(config.RESULTS_DIR, "detection_eval.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

print(f"\n{'dataset':<8} {'sev':<6} {'detector':<10} {'flagged':<9} {'tp':<5} {'precision':<11} {'recall'}")
print("-" * 58)
for _, row in results_df.iterrows():
    print(f"{row['dataset']:<8} {row['severity']:<6} {row['detector']:<10} "
          f"{row['flagged']:<9} {row['true_positives']:<5} "
          f"{row['precision']:<11.2f} {row['recall']:.2f}")
