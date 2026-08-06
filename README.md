# Causal Discovery Under MNAR

This repository contains supporting material for an MSc Computer Science project on causal discovery under MNAR missingness. The retained workflow is a research pipeline rather than a packaged application: it uses Python scripts, CSV result files, and figures to reproduce the main experiments and to document the evaluation logic.

## Project overview

The project studies how missingness that is related to the missing value itself (MNAR) affects causal discovery on benchmark networks and on real-world NHANES data. The workflow compares PC and GES on synthetic data, evaluates a detection-and-constraint pipeline, and then validates the approach on a real dataset where no ground-truth causal graph is available.

The retained experiments cover:
- benchmark datasets: Asia, Sachs, and Alarm;
- missingness mechanisms: MCAR, MAR, and MNAR;
- detection strategies: chi-square and logistic regression;
- constraint-based recovery in which detector-flagged benchmark pairs are assigned directions from the known ground-truth graph, alongside a global oracle condition;
- evaluation with SHD, false-positive and false-negative rates, bootstrap stability, and paired significance testing.

## Repository structure

The core submission includes:
- [README.md](README.md): project overview, setup, and run instructions.
- [run_experiments.py](run_experiments.py): baseline and missingness experiments for Asia and Sachs.
- [test_pipeline_full.py](test_pipeline_full.py): PC pipeline comparison across four constraint conditions.
- [test_pipeline_ges.py](test_pipeline_ges.py): GES pipeline comparison.
- [test_significance.py](test_significance.py): paired Wilcoxon analysis for no-constraint vs logistic-selective PC.
- [test_detection_eval.py](test_detection_eval.py): detector precision and recall evaluation.
- [test_alarm_network.py](test_alarm_network.py): larger-scale Alarm experiment.
- [nhanes_validation.py](nhanes_validation.py): NHANES validation and bootstrap edge stability.
- [test_mnar_mechanisms.py](test_mnar_mechanisms.py): alternative MNAR injection mechanisms.
- [test_noisy_constraints.py](test_noisy_constraints.py): robustness to incorrect constraint directions.
- [generate_figures.py](generate_figures.py): figure generation from result CSV files.
- [src/](src/): reusable implementations for data loading, missingness injection, algorithms, detection, constraints, and evaluation.
- [notebooks/](notebooks/): retained development notebooks for baseline and missingness-injection stages.
- [results/](results/): CSV outputs used by the dissertation and the figure-generation pipeline.
- [figures/](figures/): generated figure images.
- [optional_experiments/](optional_experiments/): optional or exploratory scripts and outputs that were separated from the core submission.
- [bug_fix_log.txt](bug_fix_log.txt) and [rerun_log.txt](rerun_log.txt): notes about earlier fixes and reruns.

## Installation

The project was prepared for a standard virtual environment.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

A Python 3.11 environment is recommended for the final reproducibility check. The dependency set is intended to be reproducible from the requirements file and does not rely on ad-hoc runtime installation.

The final validation used Python 3.11.9, causal-learn 0.1.4.7 and pgmpy 1.1.2. causal-learn and pgmpy are pinned in requirements.txt.

## Running the project

A practical execution order is:

1. Baseline and missingness experiments
   - `python run_experiments.py`
   - Writes [results/baseline_results.csv](results/baseline_results.csv) and [results/missingness_results.csv](results/missingness_results.csv).
   - Uses synthetic data only; no internet access required.

2. Detection evaluation
   - `python test_detection_eval.py`
   - Writes [results/detection_eval.csv](results/detection_eval.csv).
   - No internet access required.

3. PC constraint pipeline
   - `python test_pipeline_full.py`
   - Writes [results/pipeline_results.csv](results/pipeline_results.csv).
   - No internet access required.

4. GES constraint pipeline
   - `python test_pipeline_ges.py`
   - Writes [results/pipeline_ges_results.csv](results/pipeline_ges_results.csv).
   - No internet access required.

5. Statistical significance
   - `python test_significance.py`
   - Writes [results/significance_results.csv](results/significance_results.csv).
   - No internet access required.

6. Alarm scalability experiment
   - `python test_alarm_network.py`
   - Writes [results/alarm_missingness_results.csv](results/alarm_missingness_results.csv) and [results/alarm_pipeline_results.csv](results/alarm_pipeline_results.csv).
   - No internet access required.

7. NHANES validation
   - `python nhanes_validation.py`
   - Writes [results/nhanes_results.csv](results/nhanes_results.csv) and [results/nhanes_stability.csv](results/nhanes_stability.csv).
   - Downloads NHANES XPT files on first run into [data/nhanes/](data/nhanes/).
   - Internet access is required for the first download.

8. Robustness to noisy constraints
   - `python test_noisy_constraints.py`
   - Writes [results/noisy_constraints_results.csv](results/noisy_constraints_results.csv) and [results/noisy_constraints_summary.csv](results/noisy_constraints_summary.csv).
   - No internet access required.

9. Figure generation
   - `python generate_figures.py`
   - Writes the PNG files in [figures/](figures/).
   - Requires the result CSV files listed above.

## NHANES data

The NHANES workflow downloads public XPT files from the CDC website on first execution. These files are stored under [data/nhanes/](data/nhanes/) and are intentionally excluded from git by the repository ignore rules. The validation uses the public NHANES 2017 to 2018 files and the variables listed in [nhanes_validation.py](nhanes_validation.py).

HTTPS certificate verification is enabled by default. If a local certificate or network configuration prevents the CDC download, macOS/Linux users can run `NHANES_SSL_VERIFY=false python nhanes_validation.py`. This disables certificate verification, prints a warning and should only be used when necessary.

## Results and figures

The repository already contains a set of result CSV files in [results/](results/) and a corresponding figure set in [figures/](figures/). The figure-generation script reads these files and creates the PNG outputs in [figures/](figures/).

## Notebooks

The first two development stages are retained as Jupyter notebooks in [notebooks/](notebooks/). The later experiments were consolidated into Python scripts for more reproducible batch execution.

## Executable-file requirement

This repository is a research experiment pipeline rather than a compiled application. The intended execution path is through Python scripts and reproducible setup instructions.

## Limitations

The retained methodology has several important limitations:
- listwise deletion and row collapse are used in the benchmark experiments;
- the SHD implementation in [src/evaluation/metrics.py](src/evaluation/metrics.py) is a directed edge-set symmetric difference rather than a standard CPDAG-aware library implementation;
- undirected edges in the PC and GES wrappers are oriented alphabetically for deterministic evaluation; this is a deterministic convention rather than a causal orientation inferred by the algorithm;
- the constraint mechanism in [src/detection/constrain.py](src/detection/constrain.py) uses known ground-truth directions for benchmark pairs, which makes the selective-constraint experiment oracle-assisted;
- NHANES has no ground-truth causal graph, so the validation uses bootstrap stability and alphabetical orientation conventions for flagged pairs;
- the Alarm experiment uses an imputation strategy because listwise deletion becomes infeasible at that network scale;
- detection quality and incorrect domain knowledge can both affect the constraint pipeline.
