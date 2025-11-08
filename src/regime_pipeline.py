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
    
    # Build global cross-asset features. Recomputes per window

    feats_all = build_cross_asset_features(asset_dfs, selected_symbols=selected_symbols, corr_window=corr_window)
    feats_all = feats_all.reindex(target_df.index).dropna()

    target_df = target_df.loc[feats_all.index]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(target_df))

    split_results = []
    model = LogisticRegression(max_iter=1000)

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        
        
        train_df = target_df.iloc[train_idx]
        test_df = target_df.iloc[test_idx]

        # Build features using only data up to end of trin window. 
        # For regime features, build features from asset_dfs, then slice to train/test
        #feats_all = build_cross_asset_features(asset_dfs, selected_symbols=selected_symbols, corr_window=corr_window)
        #feats_all = feats_all.reindex(target_df.index).dropna()


        # Ensure train indices exist in feats_all after reindexing and dropping
        #train_index = train_df.index.intersection(feats_all.index)
        #test_index = test_df.index.intersection(feats_all.index)
        train_index = target_df.iloc[train_idx].index
        test_index = target_df.iloc[test_idx].index


        # fit regime model on training set
        rm = RegimeModel(model_type=regime_model_type, n_states=n_states)
        rm.fit(feats_all.loc[train_index])

        # infer regime labels for both train and test
        train_states = rm.infer_states(feats_all.loc[train_index])
        test_states = rm.infer_states(feats_all.loc[test_index])

        train_regime_series = pd.Series(train_states, index=train_index)
        test_regime_series = pd.Series(test_states, index=test_index)

        train_with_regime = add_regime_to_target_df(train_df, train_regime_series, prefix="regime")
        test_with_regime = add_regime_to_target_df(test_df, test_regime_series, prefix="regime")
        
        def build_supervised_features(df):
            df2 = df.copy()
            # lagged returns
            for i in range(1, 6):
                df2[f"lag_ret_{i}"] = df2["ret"].shift(i)
            df2 = df2.dropna()
            X_base = df2[[c for c in df2.columns if c.startswith("lag_ret_")]].copy()
            # one-hot encode regime
            X_reg = pd.get_dummies(df2["regime"].astype(int).astype(str), prefix="reg")
            X = pd.concat([X_base, X_reg], axis=1)
            y = (df2["ret"].shift(-1) > 0).astype(int).iloc[len(df2)-len(X):]
            return X, y

        X_train, y_train = build_supervised_features(train_with_regime)
        X_test, y_test = build_supervised_features(test_with_regime)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        split_results.append(acc)
    
    return {
        "split_accuracies": split_results,
        "average_accuracy": np.mean(split_results)
    }