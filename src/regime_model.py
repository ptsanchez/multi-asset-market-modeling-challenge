import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

class RegimeModel:
    def __init__(self, model_type="gmm", n_states=3, random_state=0):
        """
        model_type: 'gmm' or 'hmm'
        n_states: number of latent regimes
        """
        self.model_type = model_type
        self.n_states = n_states
        self.random_state = random_state
        self.model = None

    def fit(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            Xv = X.values
        else:
            Xv = X

        if self.model_type == "gmm":
            self.model = GaussianMixture(n_components=self.n_states, covariance_type="full", random_state=self.random_state)
            self.model.fit(Xv)
            self._fitted = True
        elif self.model_type == "hmm":
            self.model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=1000, random_state=self.random_state)
            self.model.fit(Xv)
            self._fitted = True
        else:
            raise ValueError("Unknown model type. Specify either 'gmm' or 'hmm'.")
        
    def infer_states(self, X: pd.DataFrame) -> np.ndarray:
        """
        Infer regime labels given X
        """
        if isinstance(X, pd.DataFrame):
            Xv = X.values
        else:
            Xv = X
        
        if self.model_type == "gmm":
            states = self.model.predict(Xv)
        elif self.model_type == "hmm":
            states = self.model.predict(Xv)
        else:
            raise ValueError("Unknown model type. Specify either 'gmm' or 'hmm'.")
        
        return states
    
    def score(self, X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            Xv = X.values
        else:
            Xv = X

        return self.model.score(Xv)