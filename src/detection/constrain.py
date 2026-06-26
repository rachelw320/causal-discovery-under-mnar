import pandas as pd
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def apply_constraints(
    df: pd.DataFrame,
    flagged_pairs: list[tuple],
    true_edges: set,
) -> BackgroundKnowledge:
    """Build BackgroundKnowledge from flagged pairs using true edge directions."""
    bk = BackgroundKnowledge()

    for col_a, col_b, _ in flagged_pairs:
        if (col_a, col_b) in true_edges:
            bk.add_required_by_pattern(col_a, col_b)
        elif (col_b, col_a) in true_edges:
            bk.add_required_by_pattern(col_b, col_a)
        # skip pairs with no true edge between them

    return bk
