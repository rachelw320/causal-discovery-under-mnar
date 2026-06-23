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
    rows = []

    for dataset in config.DATASETS:
        print(f"\n=== {dataset.upper()} ===")
        network = load_network(dataset)
        df_full = sample_data(network, n=config.SAMPLE_SIZE)
        true_edges = get_true_edges(network)
        all_nodes = list(df_full.columns)

        for algorithm_name, algorithm_fn in [("PC", run_pc), ("GES", run_ges)]:
            # baseline — no missingness
            print(f"  {algorithm_name} baseline...")
            results = bootstrap_run(df_full, algorithm_fn, true_edges, all_nodes)
            summary = summarise_bootstrap(results)
            rows.append({
                "dataset": dataset,
                "algorithm": algorithm_name,
                "mechanism": "none",
                "rate": 0.0,
                **summary,
            })

            for mechanism in config.MISSINGNESS_MECHANISMS:
                for rate in config.MISSINGNESS_LEVELS:
                    print(f"  {algorithm_name} {mechanism} {int(rate*100)}%...")
                    df_missing = inject_missingness(df_full, mechanism, rate)
                    results = bootstrap_run(df_missing, algorithm_fn, true_edges, all_nodes)
                    summary = summarise_bootstrap(results)
                    rows.append({
                        "dataset": dataset,
                        "algorithm": algorithm_name,
                        "mechanism": mechanism,
                        "rate": rate,
                        **summary,
                    })

    out = pd.DataFrame(rows)
    out_path = os.path.join(config.RESULTS_DIR, "experiment_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    run_all()
