from data_loader import load_all_data
from preprocessing import prepare_single_asset
from regime_pipeline import walk_forward_regime_pipeline

print("Loading Data...")

data = load_all_data()

selected_symbols = ["ADA", "BTC", "ETH", "SOL", "XRP", "VIX"]

print("Preprocessing selected symbols...")

asset_dfs = {}
for cls in ["crypto", "equity", "fx", "index"]:
    for sym, df in data.get(cls, {}).items():
        try:
            asset_dfs[sym] = prepare_single_asset(df)
        except Exception:
            # skip if preprocess fails or symbol missing
            continue

target_symbol = "BTC"
target_df = asset_dfs[target_symbol]

print(f"Targeting asset: {target_symbol}") 
print("Peforming walk forward evaluation...")

results = walk_forward_regime_pipeline(
    asset_dfs=asset_dfs,
    selected_symbols=[s for s in selected_symbols if s in asset_dfs],
    target_symbol=target_symbol,
    target_df=target_df,
    n_splits=5,
    regime_model_type="gmm",
    n_states=3,
    corr_window=20
)

for i in range(len(results['split_accuracies'])):
    print(f'Split {i+1} Accuracy: ' + str(results['split_accuracies'][i]*100) + '%')
print('Average Accuracy: ' + str(results['average_accuracy']*100) + '%')