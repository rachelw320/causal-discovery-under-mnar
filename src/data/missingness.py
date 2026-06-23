import numpy as np
import pandas as pd
import config


def inject_mcar(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Missing Completely At Random: each cell is independently masked with probability=rate.
    Missingness is unrelated to any observed or unobserved value.
    """
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)
    mask = rng.random(df_out.shape) < rate
    df_out[mask] = np.nan
    return df_out


def inject_mar(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Missing At Random: missingness in each column depends on the values of OTHER observed columns.
    We use the first column as the observed predictor of missingness in all other columns.
    """
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)
    cols = df_out.columns.tolist()

    # Use column 0 as the auxiliary variable driving missingness in all other columns
    anchor = df_out[cols[0]].astype("category").cat.codes
    anchor_norm = (anchor - anchor.min()) / (anchor.max() - anchor.min() + 1e-9)

    for col in cols[1:]:
        # Higher anchor value → higher probability of missingness
        prob = anchor_norm * rate * 2
        prob = np.clip(prob, 0, 1)
        mask = rng.random(len(df_out)) < prob
        df_out.loc[mask, col] = np.nan

    return df_out


def inject_mnar(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Missing Not At Random: missingness depends on the unobserved value itself.
    We mask the most common category in each column — values that are missing
    are systematically the ones that WOULD have taken a particular value.
    This is the primary experimental condition.
    """
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)

    for col in df_out.columns:
        col_data = df_out[col]
        mode_val = col_data.mode()[0]
        is_mode = col_data == mode_val
        # Mask a proportion of cells where the value equals the mode
        mode_indices = df_out.index[is_mode].tolist()
        n_to_mask = int(len(mode_indices) * rate)
        chosen = rng.choice(mode_indices, size=n_to_mask, replace=False)
        df_out.loc[chosen, col] = np.nan

    return df_out


def inject_missingness(
    df: pd.DataFrame,
    mechanism: str,
    rate: float,
    seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Dispatcher: mechanism must be one of 'MCAR', 'MAR', 'MNAR'."""
    mechanism = mechanism.upper()
    if mechanism == "MCAR":
        return inject_mcar(df, rate, seed)
    elif mechanism == "MAR":
        return inject_mar(df, rate, seed)
    elif mechanism == "MNAR":
        return inject_mnar(df, rate, seed)
    else:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Choose from MCAR, MAR, MNAR.")
