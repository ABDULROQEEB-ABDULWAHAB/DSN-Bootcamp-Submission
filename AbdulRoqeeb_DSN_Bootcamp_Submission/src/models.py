"""
Model definitions and wrappers for LightGBM, XGBoost, CatBoost, and Ridge.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class BaseEnsembleModel:
    def fit(self, X_train, y_train, X_val, y_val, cat_cols=None):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError

class LightGBMWrapper(BaseEnsembleModel):
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'max_depth': 5,
            'num_leaves': 25,
            'min_child_samples': 25,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, cat_cols=None):
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols, free_raw_data=False)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X, num_iteration=self.model.best_iteration), 0, None)

class XGBoostWrapper(BaseEnsembleModel):
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.02,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_weight': 4,
            'alpha': 1.0,
            'lambda': 5.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, cat_cols=None):
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X, enable_categorical=True)
        return np.clip(self.model.predict(dtest), 0, None)

class CatBoostWrapper(BaseEnsembleModel):
    def __init__(self, params=None):
        self.params = params or {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'learning_rate': 0.03,
            'depth': 5,
            'l2_leaf_reg': 5.0,
            'iterations': 2000,
            'random_seed': 42,
            'verbose': False
        }
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, cat_cols=None):
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_cols,
            early_stopping_rounds=50,
            verbose=False
        )
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)

class RidgeWrapper(BaseEnsembleModel):
    def __init__(self, alpha=20.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, random_state=42)

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
