import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MultiAssetFactorModel:
    def __init__(self, assets, factors):
        self.assets = assets  # list of asset names
        self.factors = factors  # list of market factors
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def extract_factors(self, data):
        # Assuming 'data' is a DataFrame with asset prices and factors
        return data[self.factors]

    def build_features(self, data):
        factors = self.extract_factors(data)
        # Create returns and feature set
        returns = data[self.assets].pct_change().dropna()
        X = self.scaler.fit_transform(factors.dropna())
        y = returns.loc[X.index]
        return X, y

    def fit(self, data):
        X, y = self.build_features(data)
        self.model.fit(X, y)

    def predict(self, new_data):
        X_new, _ = self.build_features(new_data)
        return self.model.predict(X_new)

    def get_coefficients(self):
        return self.model.coef_ 

# Usage example:
# assets = ['Asset1', 'Asset2', 'Asset3']
# factors = ['MarketFactor1', 'MarketFactor2']
# model = MultiAssetFactorModel(assets, factors)