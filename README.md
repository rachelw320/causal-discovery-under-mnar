# causal-discovery-under-mnar
Investigating how MNAR missing data degrades causal structure learning, with a missingness-aware detection and constraint pipeline.


# Causal Discovery Under MNAR

## What this project is about

When doctors and researchers try to understand *why* things happen 
(not just what correlates with what), they use algorithms that learn 
cause-and-effect relationships from data. These are called causal 
discovery algorithms.

But there's a problem. These algorithms assume your data is reasonably 
complete. In reality — especially in healthcare — data goes missing 
for reasons that aren't random. A blood test might not be recorded 
because the patient was too sick to have it done. A measurement might 
be missing because the clinician already knew the result would be 
abnormal. This is called **MNAR** (Missing Not At Random), and it's 
everywhere in real clinical data.

This project asks: **how badly does MNAR break causal discovery 
algorithms, and can we fix it?**

## What I did

I ran controlled experiments where I took a dataset with a known 
correct causal structure, deliberately introduced different types of 
missing data, and measured how wrong the algorithms' outputs became.

- **Three types of missingness tested:** random (MCAR), dependent on 
  other variables (MAR), and dependent on the missing value itself (MNAR)
- **Two algorithms tested:** PC (constraint-based) and GES (score-based)
- **Two benchmark networks:** Asia (8 variables) and Sachs 
  (11 variables — a real protein signalling network)
- **Real-world validation:** NHANES healthcare survey data, which has 
  genuine clinical missingness

I then built a two-stage pipeline to try to recover accuracy:
1. **Detection stage** — automatically identifies which variable 
   relationships are likely corrupted by MNAR
2. **Constraint stage** — applies targeted corrections only to those 
   relationships, using known domain knowledge

## What I found

[Results to be added as experiments complete]

## How to run this project

[Setup instructions to be added]

## Tools and packages used

- Python 3
- [causal-learn](https://causal-learn.readthedocs.io/) — PC and GES 
  algorithms
- pgmpy — Bayesian network simulation
- pandas, numpy — data handling
- matplotlib, seaborn — visualisation

## Project background

MSc Computer Science dissertation  
Queen Mary University of London  
Supervised by Dr Anthony Constantinou, Bayesian AI Lab  
2025–2026
