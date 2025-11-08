from data_loader import load_add_data
from preprocessing import prepare_single_asset
from features import build_baseline_features
from evaluation import walk_forward_evaluation

print("Loading data...")

data = load_add_data()
asset = data['crypto']['BTC']

print("Building baseline features...")

asset = prepare_single_asset(asset)
asset_features = build_baseline_features(asset)

print("Performing walk-forward evaluation...")

results = walk_forward_evaluation(asset_features)

for i in range(len(results['split_accuracies'])):
    print(f'Split {i+1} Accuracy: ' + str(results['split_accuracies'][i]*100) + '%')
print('Average Accuracy: ' + str(results['average_accuracy']*100) + '%')