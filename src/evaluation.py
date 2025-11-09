import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

def build_baseline_vol_features(df, vol_thresh):
    """
    Builds features and target for volatility baseline.
    X = lagged returns
    y = future volatility binary target

    """
    df2 = df.copy()
    
    # Define target
    df2["fwd_vol_10"] = df2["ret"].rolling(10).std().shift(-10)

    y_all = (df2["fwd_vol_10"] > vol_thresh).astype(int)

    df2 = df2.dropna()
    if df2.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Build features (lagged returns)
    X = df2[[c for c in df2.columns if c.startswith("return_lag_")]].copy()

    # Align
    y = y_all.reindex(X.index)
    
    return X, y

def walk_forward_evaluation(df: pd.DataFrame, n_splits: int = 5):
    """
    Perform walk-forward evaluation for volatility baseline.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(df)) 
    
    first_train_idx, _ = splits[0]
    first_train_df = df.iloc[first_train_idx]
    fwd_vol = first_train_df['ret'].rolling(10).std().shift(-10)
    
    # Use 75th percentile as "high vol" threshold
    VOL_THRESHOLD = fwd_vol.quantile(0.75)
    print(f"BASELINE: Using a high-volatility threshold of: {VOL_THRESHOLD}")

    results = []

    for fold, (train_index, test_index) in enumerate(splits):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        X_train, y_train = build_baseline_vol_features(train_df, VOL_THRESHOLD)
        X_test, y_test = build_baseline_vol_features(test_df, VOL_THRESHOLD)
        
        if X_test.empty or y_test.empty or X_train.empty or y_train.empty:
            print(f"Fold {fold}: No active test/train samples after building features, skipping")
            results.append(np.nan)
            continue
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append(acc)

    return {
        "split_accuracies": results,
        "average_accuracy": np.mean(results)
    }