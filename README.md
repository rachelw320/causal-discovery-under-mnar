# Causal Discovery Under MNAR

MSc Computer Science dissertation, Queen Mary University of London.
Supervised by Dr Anthony Constantinou, Bayesian AI Lab, 2025 to 2026.

## What this project is about

Causal discovery algorithms learn cause and effect relationships from data. They assume data is reasonably complete. In practice, especially in healthcare, data goes missing for reasons tied to the missing values themselves. This is called MNAR (Missing Not At Random). This project tests how badly MNAR degrades causal discovery and whether a detection and constraint pipeline can recover accuracy.

## What I did

I ran controlled experiments on two benchmark networks (Asia and Sachs) using PC and GES. I injected three types of missingness at four severity levels (10%, 20%, 30%, 50%) and measured structural error using SHD. I then built a two stage pipeline: a detection stage that flags variable pairs likely corrupted by MNAR, and a constraint stage that applies targeted edge corrections to those pairs before structure learning.

Two detectors were compared: chi-square (tests for correlated missingness patterns) and logistic regression (tests whether observed values of one variable predict missingness in another).

## What I found

The logistic detection pipeline reduces mean SHD by over 80% compared to unconstrained structure learning under MNAR conditions. Improvements are statistically significant at all severity levels and across both benchmark networks (Wilcoxon signed-rank, p < 0.05 in all 8 comparisons).

The chi-square detector has very low recall at low MNAR severity, dropping to 6% on Sachs at 10% missingness. The logistic detector maintains recall between 75% and 94% across all tested conditions. The pipeline generalises to both PC and GES.

## How to run

```
pip install causal-learn pgmpy scipy scikit-learn pandas numpy matplotlib seaborn
python run_experiments.py
python test_pipeline_full.py
python test_significance.py
```

## Tools used

Python 3, causal-learn, pgmpy, scikit-learn, scipy, pandas, numpy, matplotlib, seaborn.
