import numpy as np
import pandas as pd
import config


def inject_mcar(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Mask each cell independently with probability=rate."""
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)
    df_out[rng.random(df_out.shape) < rate] = np.nan
    return df_out


def inject_mar(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Missingness in each column driven by the value of column 0."""
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)
    cols = df_out.columns.tolist()

    anchor = df_out[cols[0]].astype("category").cat.codes
    anchor_norm = (anchor - anchor.min()) / (anchor.max() - anchor.min() + 1e-9)

    for col in cols[1:]:
        prob = np.clip(anchor_norm * rate * 2, 0, 1)
        df_out.loc[rng.random(len(df_out)) < prob, col] = np.nan

    return df_out


def inject_mnar(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Mask the most common category in each column at the given rate."""
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)

    for col in df_out.columns:
        mode_val = df_out[col].mode()[0]
        mode_indices = df_out.index[df_out[col] == mode_val].tolist()
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
    """Dispatch to inject_mcar, inject_mar, or inject_mnar."""
    mechanism = mechanism.upper()
    if mechanism == "MCAR":
        return inject_mcar(df, rate, seed)
    elif mechanism == "MAR":
        return inject_mar(df, rate, seed)
    elif mechanism == "MNAR":
        return inject_mnar(df, rate, seed)
    else:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Choose from MCAR, MAR, MNAR.")
