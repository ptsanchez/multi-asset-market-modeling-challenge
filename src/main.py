import argparse
import numpy as np
import pandas as pd
from data_loader import load_all_data
from preprocessing import prepare_single_asset
from features import build_baseline_features
from evaluation import walk_forward_evaluation
from regime_pipeline import walk_forward_regime_pipeline
from regime_analysis import analyze_regimes

def print_results(results_dict):
    """Helper to print evaluation results."""
    print("\n--- Evaluation Results ---")
    for i, acc in enumerate(results_dict.get('split_accuracies', [])):
        print(f'  Split {i+1} Accuracy: {acc*100:.2f}%')
    avg_acc = results_dict.get('average_accuracy', np.nan)
    print(f'Average Accuracy: {avg_acc*100:.2f}%')
    print("--------------------------\n")

def main(args):
    """
    Main execution pipeline operated by command-line arguments.
    """
    
    print("Loading Data...")
    data = load_all_data()

    print("Preprocessing selected symbols...")
    # Preprocess all symbols needed for any mode
    all_symbols = list(set([args.target_symbol] + args.symbols))
    
    asset_dfs = {}
    for cls in ["crypto", "equity", "fx", "index", "etf", "futures"]:
        for sym, df in data.get(cls, {}).items():
            if sym in all_symbols:
                try:
                    asset_dfs[sym] = prepare_single_asset(df)
                except Exception as e:
                    print(f"Warning: Could not process {sym}: {e}")
                    continue

    if args.target_symbol not in asset_dfs:
        print(f"Error: Target symbol {args.target_symbol} could not be loaded or processed.")
        return

    target_df = asset_dfs[args.target_symbol]
    print(f"Targeting asset: {args.target_symbol}")


    if args.mode == 'baseline-vol':
        print("Running: Volatility Baseline Model")
        
        # Build features for only the target asset
        asset_features = build_baseline_features(target_df)
        
        results = walk_forward_evaluation(
            df=asset_features, 
            n_splits=args.n_splits
        )
        print_results(results)

    elif args.mode == 'regime-vol':
        print("Running: Regime-Aware Volatility Model")
        
        # Filter to only symbols that were successfully loaded
        valid_symbols = [s for s in args.symbols if s in asset_dfs]
        print(f"Using regime features from: {valid_symbols}")
        
        results = walk_forward_regime_pipeline(
            asset_dfs=asset_dfs,
            selected_symbols=valid_symbols,
            target_symbol=args.target_symbol,
            target_df=target_df,
            n_splits=args.n_splits,
            regime_model_type=args.model_type,
            n_states=args.n_states,
            corr_window=args.corr_window
        )
        print_results(results)

    elif args.mode == 'regime-analysis':
        print("Running: Final Regime Analysis")
        
        # Filter to only symbols that were successfully loaded
        valid_symbols = [s for s in args.symbols if s in asset_dfs]
        print(f"Using regime features from: {valid_symbols}")

        # Prints own detailed analysis
        analyze_regimes(
            asset_dfs=asset_dfs,
            selected_symbols=valid_symbols,
            target_symbol=args.target_symbol,
            target_df=target_df,
            regime_model_type=args.model_type,
            n_states=args.n_states,
            corr_window=args.corr_window
        )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="RWS Multi-Asset Market Modeling Challenge")
    
    parser.add_argument(
        'mode', 
        choices=['baseline-vol', 'regime-vol', 'regime-analysis'], 
        help="The operation to run: \
              'baseline-vol': Run the baseline volatility model. \
              'regime-vol': Run the regime-aware volatility model. \
              'regime-analysis': Run the regime characteristic analysis."
    )
    
    # Model & Data Arguments
    parser.add_argument(
        '--target_symbol', 
        type=str, 
        default='BTC', 
        help="The target asset symbol to model (e.g., 'BTC', 'AMZN')."
    )
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        default=['ADA', 'BTC', 'ETH', 'SOL', 'XRP', 'VIX'],
        help="List of symbols to use for regime features (e.g., --symbols BTC GOOG VIX)"
    )
    
    # Evaluation Arguments
    parser.add_argument(
        '--n_splits', 
        type=int, 
        default=5, 
        help="Number of splits for walk-forward validation."
    )

    # Regime Model Arguments
    parser.add_argument(
        '--model_type', 
        type=str, 
        choices=['gmm', 'hmm'], 
        default='gmm', 
        help="The type of model to use for regime clustering."
    )
    parser.add_argument(
        '--n_states', 
        type=int, 
        default=3, 
        help="Number of latent states for the regime model."
    )
    parser.add_argument(
        '--corr_window', 
        type=int, 
        default=20, 
        help="Rolling window size for calculating correlations."
    )

    args = parser.parse_args()
    main(args)