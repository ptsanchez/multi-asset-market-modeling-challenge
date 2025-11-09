import pandas as pd
from regime_features import build_cross_asset_features
from regime_model import RegimeModel

def analyze_regimes(asset_dfs: dict, 
                    selected_symbols: list, 
                    target_symbol: str, 
                    target_df: pd.DataFrame,
                    regime_model_type="hmm",
                    n_states=3,
                    corr_window=20):
    """
    Fits single regime model on all available data and prints characteristics of each state.
    """
    
    print("Building features for full dataset...")
    
    feats_all = build_cross_asset_features(
        asset_dfs, 
        selected_symbols=selected_symbols, 
        corr_window=corr_window
    )
    
    feats_all_aligned = feats_all.reindex(target_df.index).dropna()
    
    print(f"Data has {feats_all_aligned.shape[0]} observations and {feats_all_aligned.shape[1]} features.")

    print(f"Fitting {regime_model_type.upper()} model with {n_states} states...")
    rm = RegimeModel(model_type=regime_model_type, n_states=n_states)
    rm.fit(feats_all_aligned)

    states = rm.infer_states(feats_all_aligned)
    feats_all_aligned['regime'] = states
    
    print("\n--- Regime Analysis ---")
    
    state_counts = feats_all_aligned['regime'].value_counts(normalize=True).sort_index()
    print("\nRegime Frequency (Percentage of time spent in each state):")
    for state, perc in state_counts.items():
        print(f"  State {state}: {perc*100:.2f}%")
        
    regime_means = feats_all_aligned.groupby('regime').mean()
    
    print("\nMean Feature Values per Regime:")
    print(regime_means.to_string(float_format="%.4f"))
    
    # Also attach regimes to the target_df to see target asset behavior
    target_with_regimes = target_df.copy()
    target_with_regimes['regime'] = pd.Series(states, index=feats_all_aligned.index)
    target_with_regimes = target_with_regimes.dropna()
    
    print(f"\nTarget Asset ({target_symbol}) Behavior per Regime:")
    
    target_stats = target_with_regimes.groupby('regime')['ret'].agg(['mean', 'std'])
    print(target_stats.to_string(float_format="%.6f"))
    
    
    return regime_means, target_stats