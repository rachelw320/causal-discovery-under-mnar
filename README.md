# Causal Discovery Under MNAR

MSc Computer Science dissertation, Queen Mary University of London.
Supervised by Dr Anthony Constantinou, Bayesian AI Lab, 2025 to 2026.

## What this project is about

Causal discovery algorithms learn cause and effect relationships from data. They assume data is reasonably complete. In practice, especially in healthcare, data goes missing for reasons tied to the missing values themselves. This is called MNAR (Missing Not At Random). This project tests how badly MNAR degrades causal discovery and whether a detection and constraint pipeline can recover accuracy.

## What I did

I ran controlled experiments on two benchmark networks (Asia and Sachs) using PC and GES. I injected three types of missingness at four severity levels (10%, 20%, 30%, 50%) and measured structural error using SHD. I then built a two stage pipeline: a detection stage that flags variable pairs likely corrupted by MNAR, and a constraint stage that applies targeted edge corrections to those pairs before structure learning.

Two detectors were compared: chi-square (tests for correlated missingness patterns) and logistic regression (tests whether observed values of one variable predict missingness in another).

I then extended the experiments in three directions.

**Alarm network scalability.** I repeated the core pipeline on the Alarm network (37 nodes, 46 edges), which is much larger than Asia (8 nodes) and Sachs (11 nodes). At this scale, listwise deletion becomes infeasible: at 30% MNAR, roughly one row in a thousand survives as a complete case across all 37 variables. I used mode imputation as a practical alternative, which means the Alarm results measure PC and GES on imputed data. The logistic selective condition reduced mean SHD from 31.9 to 19.1, close to the global oracle at 18.4.

**NHANES real-world validation.** I applied the pipeline to NHANES 2017 to 2018 data (12 variables, 9,254 participants, 28.7% complete cases). Since there is no ground truth causal graph for real data, I used bootstrap edge stability as the evaluation criterion. Edges that appear consistently across bootstrap samples are treated as reliable. The logistic detector flagged 53 variable pairs. Constrained PC recovered several edges with stability above 0.9 that unconstrained PC missed or weakened.

**Robustness to incorrect domain knowledge.** I tested what happens when the constraint directions are partially wrong. For the Sachs network at 30% MNAR, I took the logistic selective constraints and reversed a proportion of the edge directions (10%, 20%, 30%, 50%). Each noise level was repeated across 10 random seeds to measure sensitivity to which specific edges were flipped. At 10% noise the pipeline still reduces mean SHD from 27.3 to 8.2 compared to no constraints. At 50% noise the benefit disappears and performance matches or slightly exceeds the no-constraint baseline. This puts a practical bound on how much domain knowledge error the pipeline can tolerate.

## What I found

The logistic detection pipeline reduces mean SHD by over 80% compared to unconstrained structure learning under MNAR conditions. Improvements are statistically significant at all severity levels and across both benchmark networks (Wilcoxon signed-rank, p < 0.05 in all 8 comparisons).

The chi-square detector has very low recall at low MNAR severity, dropping to 6% on Sachs at 10% missingness. The logistic detector maintains recall between 75% and 94% across all tested conditions. The pipeline generalises to both PC and GES.

At 50% MNAR on Sachs, PC under listwise deletion misses every true edge (FN rate = 1.0, std = 0 across 30 bootstrap iterations). This is the most severe failure mode in the experiments.

## How to run

```
pip install causal-learn pgmpy scipy scikit-learn pandas numpy matplotlib seaborn
python run_experiments.py
python test_pipeline_full.py
python test_significance.py
python test_alarm_network.py
python nhanes_validation.py
python test_noisy_constraints.py
python generate_figures.py
```

## Scripts

| Script | What it does |
|---|---|
| `run_experiments.py` | Main degradation experiment: PC and GES on Asia and Sachs across all missingness types and severity levels |
| `test_pipeline_full.py` | Four-condition constraint pipeline (no constraints, chi-square, logistic, oracle) on Asia and Sachs |
| `test_pipeline_ges.py` | Same pipeline for GES |
| `test_significance.py` | Wilcoxon signed-rank tests comparing no-constraint vs logistic-selective PC |
| `test_detection_eval.py` | Precision and recall of chi-square and logistic detectors across severity levels |
| `test_alarm_network.py` | Alarm network experiment: baseline, MNAR degradation, detection, pipeline, and Wilcoxon at scale (mode imputation) |
| `nhanes_validation.py` | Real-world validation on NHANES 2017 to 2018: bootstrap edge stability under PC, constrained PC, and GES |
| `test_mnar_mechanisms.py` | Comparison of four MNAR injection variants: threshold, gradient, correlated, mixed |
| `test_noisy_constraints.py` | Robustness analysis: logistic constraints with a proportion of directions reversed, 10 seeds per noise level |
| `generate_figures.py` | Generates all figures from results CSVs |

## Tools used

Python 3, causal-learn 0.1.4.7, pgmpy 1.1.2, scikit-learn, scipy, pandas, numpy, matplotlib, seaborn.
