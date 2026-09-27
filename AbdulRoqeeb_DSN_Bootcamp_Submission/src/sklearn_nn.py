"""
Scikit-Learn Multi-Layer Perceptron (MLPRegressor) Wrapper for Tabular Neural Network Modeling.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class SklearnMLPWrapper:
    def __init__(self, hidden_layer_sizes=(128, 64, 32), max_iter=300, alpha=0.01, random_state=42):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate='adaptive',
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state
        )
        
    def fit(self, X_train, y_train, X_val, y_val, cat_cols=None):
        X_tr = X_train.astype(float).values
        X_tr_clean = np.nan_to_num(X_tr, nan=0.0)
        X_tr_scaled = self.scaler.fit_transform(X_tr_clean)
        
        self.model.fit(X_tr_scaled, y_train)
        return self
        
    def predict(self, X):
        X_mat = X.astype(float).values
        X_mat_clean = np.nan_to_num(X_mat, nan=0.0)
        X_scaled = self.scaler.transform(X_mat_clean)
        return np.clip(self.model.predict(X_scaled), 0, None)
