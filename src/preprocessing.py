import pandas as pd
import numpy as np

def resample_to_uniform(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Resamples OHLCV data to uniform frequency.
    """
    ohlcv_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_resampled = df.resample(freq).apply(ohlcv_dict)
    return df_resampled.ffill()

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds log returns column to DataFrame.
    """
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    df["is_active"] = (np.abs(df["ret"]) > 1e-12).astype(int)

    df["log_ret"] = df["log_ret"].replace([np.inf, -np.inf], 0).fillna(0)
    df["ret"] = df["ret"].fillna(0)

    return df

def prepare_single_asset(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Pipeline for preparing a single asset.
    """
    df_prepared = resample_to_uniform(df, freq=freq)
    df_prepared = compute_returns(df_prepared)
    return df_prepared