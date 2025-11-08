import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class BaselineReturnPredictor:
    """
    Simple logistic regression classifier using lagged returns as features.
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def _extract_xy(self, df: pd.DataFrame):
        """
        Given feature dataframe, extract X and y.
        y is the next-period direction (1 or 0)
        """

        df = df.copy()
        df["target"] = (df["ret"].shift(-1) > 0).astype(int)
        

        df["target_active"] = (df["is_active"].shift(-1)).fillna(0).astype(int)

        df = df.dropna() 

        # only training/evaluating on periods where the target had a move. structural artifact present in non 24/7 assets. explained in report
        df_active = df[df["target_active"] == 1].copy()

        y = df_active["target"].values
        X = df_active[[c for c in df.columns if c.startswith("return_lag_")]].values
        return X, y
    
    def fit(self, df: pd.DataFrame):
        X, y = self._extract_xy(df)
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame):
        X, y = self._extract_xy(df)
        preds = self.model.predict(X)
        return preds, y
    
    def predict_proba(self, df: pd.DataFrame):
        X, y = self._extract_xy(df)
        probs = self.model.predict_proba(X)[:, 1]
        return probs, y