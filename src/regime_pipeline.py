import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from regime_features import build_cross_asset_features
from regime_model import RegimeModel

def add_regime_to_target_df(target_df: pd.DataFrame, regime_series: pd.Series, prefix="regime"):
    """
    Add regime label to target_df by aligning on index.
    """
    df = target_df.copy()
    df[prefix] = regime_series.reindex(df.index)
    return df.dropna()

def walk_forward_regime_pipeline(asset_dfs: dict,
                                 selected_symbols: list,
                                 target_symbol: str,
                                 target_df: pd.DataFrame,
                                 n_splits: int = 5,
                                 regime_model_type="gmm",
                                 n_states = 3,
                                 corr_window = 20):
    

    feats_all = build_cross_asset_features(asset_dfs, selected_symbols=selected_symbols, corr_window=corr_window)
    feats_all = feats_all.reindex(target_df.index).dropna()

    target_df = target_df.loc[feats_all.index]

    if len(target_df) < n_splits * 50:
        raise ValueError(f"Insufficient data after feature alignment: {len(target_df)} observations")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(target_df))

    split_results = []

    first_train_idx, _ = splits[0]
    first_train_df = target_df.iloc[first_train_idx]
    
    fwd_vol = first_train_df['ret'].rolling(10).std().shift(-10)
    
    # "High vol" defined as the top 25% 
    VOL_THRESHOLD = fwd_vol.quantile(0.75)
    print(f"Using a high-volatility threshold of: {VOL_THRESHOLD}")


    for fold, (train_idx, test_idx) in enumerate(splits):
        train_df = target_df.iloc[train_idx]
        test_df = target_df.iloc[test_idx]
        
        train_index = train_df.index
        test_index = test_df.index

        train_features = feats_all.loc[train_index]
        test_features = feats_all.loc[test_index]

        if len(train_features) == 0:
            print(f"Fold {fold}: No training features available, skipping..")
            continue

        rm = RegimeModel(model_type=regime_model_type, n_states=n_states)
        rm.fit(train_features)

        train_states = rm.infer_states(train_features)
        test_states = rm.infer_states(test_features)
        
        train_regime_series = pd.Series(train_states, index=train_index)
        test_regime_series = pd.Series(test_states, index=test_index)

        train_with_regime = add_regime_to_target_df(train_df, train_regime_series, prefix="regime")
        test_with_regime = add_regime_to_target_df(test_df, test_regime_series, prefix="regime")
        
        def build_supervised_features(df, vol_thresh):
            df2 = df.copy()
            # lagged returns
            for i in range(1, 6):
                df2[f"lag_ret_{i}"] = df2["ret"].shift(i)
            
            # Calculate 10-minute rolling volatility, then shift back 10 steps
            df2["fwd_vol_10"] = df2["ret"].rolling(10).std().shift(-10)

            y_all = (df2["fwd_vol_10"] > vol_thresh).astype(int)

            df2 = df2.dropna()
            
            if df2.empty:
                return pd.DataFrame(), pd.Series(dtype=int)

            # Build features
            X_base = df2[[c for c in df2.columns if c.startswith("lag_ret_")]].copy()
            X_reg = pd.get_dummies(df2["regime"].astype(int).astype(str), prefix="reg")
            X = pd.concat([X_base, X_reg], axis=1)

            y = y_all.reindex(X.index)
            
            # Note that 0-return is a valid (low-vol), so don't need y_active filter this target.
            return X, y

        X_train, y_train = build_supervised_features(train_with_regime, VOL_THRESHOLD)
        X_test, y_test = build_supervised_features(test_with_regime, VOL_THRESHOLD)

        if X_test.empty or y_test.empty or X_train.empty or y_train.empty:
            print(f"Fold {fold}: No active test/train samples after building features, skipping")
            split_results.append(np.nan)
            continue
            
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        split_results.append(acc)
    
    return {
        "split_accuracies": split_results,
        "average_accuracy": np.mean(split_results)
    }