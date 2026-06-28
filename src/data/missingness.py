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
    anchor_norm = anchor.rank(pct=True)

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


def inject_mnar_threshold(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Values above median category code go missing at rate*1.5, below at rate*0.5."""
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)
    for col in df_out.columns:
        codes = df_out[col].astype("category").cat.codes
        median_code = codes.median()
        prob = np.where(codes > median_code, rate * 1.5, rate * 0.5)
        prob = np.clip(prob, 0, 1)
        mask = rng.random(len(df_out)) < prob
        df_out.loc[mask, col] = np.nan
    return df_out


def inject_mnar_gradient(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Missingness probability scales with percentile rank of each value."""
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)
    for col in df_out.columns:
        codes = df_out[col].astype("category").cat.codes.astype(float)
        pct_rank = codes.rank(pct=True)
        scale = rate / pct_rank.mean()
        prob = np.clip(pct_rank * scale, 0, 1)
        mask = rng.random(len(df_out)) < prob.values
        df_out.loc[mask, col] = np.nan
    return df_out


def inject_mnar_correlated(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Missingness clusters: each additional missing variable in a row raises probability."""
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)
    cols = df_out.columns.tolist()
    col_order = rng.permutation(cols)
    for col in col_order:
        other_cols = [c for c in cols if c != col]
        if other_cols:
            already_missing = df_out[other_cols].isna().sum(axis=1).values
        else:
            already_missing = np.zeros(len(df_out))
        prob = rate * 0.5 + rate * 0.1 * already_missing
        prob = np.clip(prob, 0, 0.9)
        mask = rng.random(len(df_out)) < prob
        df_out.loc[mask, col] = np.nan
    return df_out


def inject_mnar_mixed(df: pd.DataFrame, rate: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Half columns use threshold, half use gradient -- realistic mixed mechanisms."""
    rng = np.random.default_rng(seed)
    df_out = df.copy().astype(object)
    cols = df_out.columns.tolist()
    half = len(cols) // 2
    threshold_cols = cols[:half]
    gradient_cols = cols[half:]

    for col in threshold_cols:
        codes = df_out[col].astype("category").cat.codes
        median_code = codes.median()
        prob = np.where(codes > median_code, rate * 1.5, rate * 0.5)
        prob = np.clip(prob, 0, 1)
        mask = rng.random(len(df_out)) < prob
        df_out.loc[mask, col] = np.nan

    for col in gradient_cols:
        codes = df_out[col].astype("category").cat.codes.astype(float)
        pct_rank = codes.rank(pct=True)
        scale = rate / pct_rank.mean()
        prob = np.clip(pct_rank * scale, 0, 1)
        mask = rng.random(len(df_out)) < prob.values
        df_out.loc[mask, col] = np.nan

    return df_out


def inject_missingness(
    df: pd.DataFrame,
    mechanism: str,
    rate: float,
    seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Dispatch to the appropriate injection function."""
    mechanism = mechanism.upper()
    if mechanism == "MCAR":
        return inject_mcar(df, rate, seed)
    elif mechanism == "MAR":
        return inject_mar(df, rate, seed)
    elif mechanism == "MNAR":
        return inject_mnar(df, rate, seed)
    elif mechanism == "MNAR_THRESHOLD":
        return inject_mnar_threshold(df, rate, seed)
    elif mechanism == "MNAR_GRADIENT":
        return inject_mnar_gradient(df, rate, seed)
    elif mechanism == "MNAR_CORRELATED":
        return inject_mnar_correlated(df, rate, seed)
    elif mechanism == "MNAR_MIXED":
        return inject_mnar_mixed(df, rate, seed)
    else:
        raise ValueError(
            f"Unknown mechanism '{mechanism}'. "
            "Choose from MCAR, MAR, MNAR, MNAR_THRESHOLD, MNAR_GRADIENT, MNAR_CORRELATED, MNAR_MIXED."
        )
