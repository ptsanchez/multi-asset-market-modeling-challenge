import pandas as pd
from config import BASELINE_LOOKBACK

def add_lagged_returns(df: pd.DataFrame, lookback: int = BASELINE_LOOKBACK) -> pd.DataFrame:
    """
    Adds lagged returs as baseline features.
    """
    df = df.copy()
    for i in range(1, lookback + 1):
        df[f'return_lag_{i}'] = df["ret"].shift(i)

    return df

def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build baseline features.
    Currently only lagged returns.
    """

    df = add_lagged_returns(df)
    return df.dropna()