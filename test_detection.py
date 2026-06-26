import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import load_network, sample_data, get_true_edges
from src.data.missingness import inject_missingness
from src.detection.detector import detect_mnar_pairs

network = load_network("sachs")
df = sample_data(network)
true_edges = get_true_edges(network)

# run on MNAR 30%
df_mnar = inject_missingness(df, "MNAR", 0.30)
flagged_mnar = detect_mnar_pairs(df_mnar)

# run on clean data as control
flagged_clean = detect_mnar_pairs(df)

print(f"Flagged pairs under MNAR 30%: {len(flagged_mnar)}")
print(f"Flagged pairs on clean data:  {len(flagged_clean)}")

print("\nFlagged pairs (MNAR):")
for col_a, col_b, p in sorted(flagged_mnar, key=lambda x: x[2]):
    print(f"  {col_a} -- {col_b}  p={p:.4e}")

# check overlap with true edges
true_pairs = {(a, b) for a, b in true_edges} | {(b, a) for a, b in true_edges}
overlapping = [(a, b, p) for a, b, p in flagged_mnar if (a, b) in true_pairs or (b, a) in true_pairs]

print(f"\nFlagged pairs that match a true Sachs edge: {len(overlapping)}")
for col_a, col_b, p in overlapping:
    print(f"  {col_a} -- {col_b}  p={p:.4e}")
