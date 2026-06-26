import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.algorithms.pc_wrapper import run_pc
from src.algorithms.ges_wrapper import run_ges
from src.evaluation.bootstrap import bootstrap_run, summarise_bootstrap


def run_all():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    baseline_rows = []
    missingness_rows = []

    for dataset in config.DATASETS:
        print(f"\n=== {dataset.upper()} ===")

        # load network and sample clean data
        network = load_network(dataset)
        df_full = sample_data(network, n=config.SAMPLE_SIZE)
        true_edges = get_true_edges(network)
        all_nodes = list(df_full.columns)

        for algorithm_name, algorithm_fn in [("PC", run_pc), ("GES", run_ges)]:

            # baseline on clean data
            print(f"  {algorithm_name} baseline...")
            results = bootstrap_run(df_full, algorithm_fn, true_edges, all_nodes)
            summary = summarise_bootstrap(results)
            baseline_rows.append({
                "dataset": dataset,
                "algorithm": algorithm_name,
                "mechanism": "none",
                "rate": 0.0,
                **summary,
            })

            # run each missingness condition
            for mechanism in config.MISSINGNESS_MECHANISMS:
                for rate in config.MISSINGNESS_LEVELS:
                    print(f"  {algorithm_name} {mechanism} {int(rate * 100)}%...")
                    df_missing = inject_missingness(df_full, mechanism, rate)
                    results = bootstrap_run(df_missing, algorithm_fn, true_edges, all_nodes)
                    summary = summarise_bootstrap(results)
                    missingness_rows.append({
                        "dataset": dataset,
                        "algorithm": algorithm_name,
                        "mechanism": mechanism,
                        "rate": rate,
                        **summary,
                    })

    # save baseline results
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_path = os.path.join(config.RESULTS_DIR, "baseline_results.csv")
    baseline_df.to_csv(baseline_path, index=False)
    print(f"\nSaved baseline to {baseline_path}")

    # save missingness results
    missingness_df = pd.DataFrame(missingness_rows)
    missingness_path = os.path.join(config.RESULTS_DIR, "missingness_results.csv")
    missingness_df.to_csv(missingness_path, index=False)
    print(f"Saved missingness results to {missingness_path}")

    # print summary
    print("\n=== SUMMARY: mean SHD (std) per condition ===\n")
    all_rows = pd.concat([baseline_df, missingness_df], ignore_index=True)
    for _, row in all_rows.iterrows():
        label = f"{row['dataset']} | {row['algorithm']} | {row['mechanism']} {int(row['rate'] * 100)}%"
        print(f"  {label:<45}  SHD {row['shd_mean']:.2f} (±{row['shd_std']:.2f})")


if __name__ == "__main__":
    run_all()
