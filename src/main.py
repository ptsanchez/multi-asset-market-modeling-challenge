from data_loader import load_add_data
from preprocessing import prepare_single_asset
from features import build_baseline_features
from evaluation import walk_forward_evaluation

print("Loading data...")

data = load_add_data()
btc = data['crypto']['BTC']

btc = prepare_single_asset(btc)
btc_features = build_baseline_features(btc)

print("Building baseline features...")

results = walk_forward_evaluation(btc_features)

print("Performing walk-forward evaluation...")


print(results)