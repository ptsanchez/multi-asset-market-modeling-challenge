from data_loader import load_all_data
from preprocessing import prepare_single_asset
from regime_analysis import analyze_regimes

print("Loading Data...")

data = load_all_data()

selected_symbols = ["ADA", "BTC", "ETH", "SOL", "XRP", "VIX"]
target_symbol = "BTC"

print("Preprocessing selected symbols...")
asset_dfs = {}
for cls in ["crypto", "equity", "fx", "index"]:
    for sym, df in data.get(cls, {}).items():
        if sym in selected_symbols or sym == target_symbol:
            try:
                asset_dfs[sym] = prepare_single_asset(df)
            except Exception as e:
                print(f"Could not process {sym}: {e}")
                continue

target_df = asset_dfs[target_symbol]

print(f"Targeting asset: {target_symbol}") 
print("Running final regime analysis...")

analyze_regimes(
    asset_dfs=asset_dfs,
    selected_symbols=[s for s in selected_symbols if s in asset_dfs],
    target_symbol=target_symbol,
    target_df=target_df,
    regime_model_type="gmm",
    n_states=3,
    corr_window=20
)