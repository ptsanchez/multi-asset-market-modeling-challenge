import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from baseline_model import BaselineReturnPredictor

def walk_forward_evaluation(df: pd.DataFrame, n_splits: int = 5):
    """
    Perform walk-forward evaluation using time series split.
    Returns metric for each split and overall average
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_index, test_index in tscv.split(df):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        model = BaselineReturnPredictor()
        model.fit(train_df)

        preds, y_test = model.predict(test_df)
        acc = accuracy_score(y_test, preds)
        results.append(acc)

    return {
        "split_accuracies": results,
        "average_accuracy": np.mean(results)
    }