import pandas as pd
import numpy as np
from typing import List, Dict

def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).std()

def rolling_mean_abs_return(series: pd.Series, window: int) -> pd.Series:
    return series['log_ret'].abs().rolling(window=window).mean()

def zscore(x: pd.Series, window: int):
    return (x - x.rolling(window=window).mean()) / (x.rolling(window=window).std()+ 1e-12) # epsilon to avoid division by zero

def build_single_asset_features(df: pd.DataFrame, vol_windows=[5, 15], ret_col="log_ret") -> pd.DataFrame:
    """
    Compute baseline feasures given preprocessed single-asset dataframe with 'log_ret' and 'volume'
    Returns dataframe indexed by timestamp
    """

    feats = pd.DataFrame(index=df.index)
    for w in vol_windows:
        feats[f"vol_{w}"] = df[ret_col].rolling(w).std()
        feats[f"ma_absret_{w}"] = df[ret_col].abs().rolling(w).mean()

    if "volume" in df.columns:
        feats["vol_z_60"] = zscore(df["volume"], 60)

    feats = feats.dropna()
    return feats

def build_cross_asset_features(asset_dfs: Dict[str, pd.DataFrame],
                               selected_symbols: List[str],
                               corr_window: int = 20,
                               vol_windows=[5,15]) -> pd.DataFrame:
    """
    Build a combined feature frame across selected symbols.
    Returns a dataframe with index of timestamps and columns of cross-asset features
    """
    # compute per-asset features and then aggregate afterwards
    per_asset_feats = {}
    for sym in selected_symbols:
        df = asset_dfs.get(sym)
        if df is None:
            continue
        f = build_single_asset_features(df, vol_windows=vol_windows)
        f = f.add_prefix(f"sym_")
        per_asset_feats[sym] = f
    
    if not per_asset_feats:
        raise ValueError("No symbols found in asset_dfs")
    merged = pd.concat(per_asset_feats.values(), axis=1).dropna()

    # build rolling correlations matrix summary (avg pairwise corr among returns)
    rets = {}
    for sym in selected_symbols:
        df = asset_dfs.get(sym)
        if df is None:
            continue
        rets[sym] = df["log_ret"]

    rets_df = pd.DataFrame(rets).reindex(merged.index).dropna(axis=0)
    
    # rolling pairwise correlation -> for each time, compute average off-diagonal corr
    def rolling_avg_corr(df_rets, window):
        corrmats = df_rets.rolling(window).corr()
        # corrmats is a MultiIndex: (timestamp, asset1, asset2)
        avg_corr = corrmats.groupby(level=0).apply(
            lambda mat: mat.values[np.triu_indices_from(mat.values, k=1)].mean()
        )
        return avg_corr
    avg_corr = rolling_avg_corr(rets_df, corr_window)
    merged["avg_pair_corr"] = avg_corr.reindex(merged.index)

    merged = merged.dropna()
    return merged