"""
Neural Network + GBDT Hybrid Ensemble Benchmark Script.
Trains PyTorch Tabular ResNet & Sklearn MLP Neural Networks alongside CatBoost, LightGBM, and XGBoost.
"""

import sys
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize

sys.path.append('src')

from features import prepare_datasets
from models import LightGBMWrapper, XGBoostWrapper, CatBoostWrapper
from sklearn_nn import SklearnMLPWrapper

try:
    from neural_net import PyTorchTabularWrapper
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def run_nn_gbdt_hybrid_benchmark(n_splits=5, seed=42):
    print("=" * 80)
    print("  DSN MART: NEURAL NETWORK + GBDT HYBRID ENSEMBLE BENCHMARK")
    print("=" * 80)
    
    train_df, test_df, cat_cols = prepare_datasets()
    
    raw_train = pd.read_csv('data/train.csv')
    raw_test = pd.read_csv('data/test.csv')
    
    store_vol_map = {
        'Corner Shop': 0.154,
        'Superstore': 0.907,
        'Standard Supermarket': 1.066,
        'Flagship Hypermarket': 1.683
    }
    
    for df, raw in [(train_df, raw_train), (test_df, raw_test)]:
        df['price_x_store_volume'] = df['product_price'] * raw['store_format'].map(store_vol_map)
        df['price_x_store_size'] = df['product_price'] * df['store_size_ord']
        df['is_hypermarket_top_mrp'] = ((raw['store_format'] == 'Flagship Hypermarket') & (df['product_price'] > 185)).astype(int)
        
    test_ids = test_df['id'].values
    y_sales = train_df['total_sales'].values
    train_prices = train_df['product_price'].values
    test_prices = test_df['product_price'].values
    
    train_stores = raw_train['store_code'].values
    test_stores = raw_test['store_code'].values
    unique_stores = np.unique(train_stores)
    
    y_units = y_sales / train_prices
    
    drop_cols = ['id', 'total_sales']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    categorical_cols = [c for c in cat_cols if c in feature_cols]
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"PyTorch Available: {HAS_TORCH}")
    
    model_definitions = [
        ('CatBoost_Direct', CatBoostWrapper, 'sales'),
        ('CatBoost_Units',  CatBoostWrapper, 'units'),
        ('Sklearn_NN_Units', SklearnMLPWrapper, 'units'),
        ('Sklearn_NN_Direct', SklearnMLPWrapper, 'sales'),
        ('LGBM_Units',      LightGBMWrapper, 'units'),
        ('XGBoost_Units',   XGBoostWrapper,  'units')
    ]
    
    if HAS_TORCH:
        model_definitions.append(('PyTorch_NN_Units', PyTorchTabularWrapper, 'units'))
        model_definitions.append(('PyTorch_NN_Direct', PyTorchTabularWrapper, 'sales'))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_predictions = {name: np.zeros(len(X)) for name, _, _ in model_definitions}
    test_predictions = {name: np.zeros(len(X_test)) for name, _, _ in model_definitions}
    model_scores = {}
    
    for model_name, model_class, target_type in model_definitions:
        print(f"\n--- Training {model_name} (Target: {target_type.upper()}) ---")
        target_train = y_sales if target_type == 'sales' else y_units
        fold_rmses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_sales)):
            X_tr, y_tr = X.iloc[train_idx].copy(), target_train[train_idx]
            X_va, y_va = X.iloc[val_idx].copy(), target_train[val_idx]
            p_va = train_prices[val_idx]
            s_tr, s_va = train_stores[train_idx], train_stores[val_idx]
            
            if HAS_TORCH and model_class == PyTorchTabularWrapper:
                model = PyTorchTabularWrapper(batch_size=256, lr=3e-3, epochs=80, patience=12)
            elif model_class == SklearnMLPWrapper:
                model = SklearnMLPWrapper(hidden_layer_sizes=(256, 128, 64), max_iter=250, alpha=0.01, random_state=seed+fold)
            else:
                model = model_class()
                
            model.fit(X_tr, y_tr, X_va, y_va, cat_cols=categorical_cols)
            
            # Val raw prediction
            val_raw = model.predict(X_va)
            val_sales = val_raw if target_type == 'sales' else val_raw * p_va
            val_sales = np.clip(val_sales, 0, None)
            
            # Train raw prediction for out-of-fold store calibration
            tr_raw = model.predict(X_tr)
            tr_sales = tr_raw if target_type == 'sales' else tr_raw * train_prices[train_idx]
            tr_sales = np.clip(tr_sales, 0, None)
            
            # Store-level scale multipliers
            fold_ratios = {}
            for s in unique_stores:
                act_m = y_sales[train_idx][s_tr == s].mean()
                prd_m = tr_sales[s_tr == s].mean()
                fold_ratios[s] = act_m / prd_m if prd_m > 0 else 1.0
                
            val_cal = val_sales * pd.Series(s_va).map(fold_ratios).values
            oof_predictions[model_name][val_idx] = np.clip(val_cal, 0, None)
            
            fold_rmse = root_mean_squared_error(y_sales[val_idx], oof_predictions[model_name][val_idx])
            fold_rmses.append(fold_rmse)
            print(f"  Fold {fold+1}/{n_splits} RMSE: {fold_rmse:.2f}")
            
            # Test prediction
            test_raw = model.predict(X_test)
            test_sales = test_raw if target_type == 'sales' else test_raw * test_prices
            test_cal = test_sales * pd.Series(test_stores).map(fold_ratios).values
            test_predictions[model_name] += np.clip(test_cal, 0, None) / n_splits
            
        overall_oof_rmse = root_mean_squared_error(y_sales, oof_predictions[model_name])
        model_scores[model_name] = overall_oof_rmse
        print(f"  >> {model_name} Overall OOF RMSE: {overall_oof_rmse:.2f}")

    # Optimize SciPy SLSQP Blending Weights
    names = [name for name, _, _ in model_definitions]
    oof_matrix = np.column_stack([oof_predictions[name] for name in names])
    test_matrix = np.column_stack([test_predictions[name] for name in names])
    
    def ensemble_objective(weights):
        blend = np.dot(oof_matrix, weights)
        return root_mean_squared_error(y_sales, blend)
        
    init_weights = np.ones(len(names)) / len(names)
    bounds = [(0.02, 0.70) for _ in names]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    res = minimize(ensemble_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = res.x / np.sum(res.x)
    
    print("\n" + "=" * 60)
    print("  HYBRID NEURAL NET + GBDT OPTIMAL BLENDING WEIGHTS")
    print("=" * 60)
    for name, w in zip(names, opt_weights):
        print(f"  {name:22s} | Solo OOF RMSE: {model_scores[name]:8.2f} | Weight: {w*100:6.2f}%")
        
    blended_oof = np.dot(oof_matrix, opt_weights)
    blended_oof_rmse = root_mean_squared_error(y_sales, blended_oof)
    
    print(f"  >> FINAL HYBRID NEURAL NET + GBDT ENSEMBLE OOF RMSE: {blended_oof_rmse:.2f} RMSE")

if __name__ == '__main__':
    run_nn_gbdt_hybrid_benchmark()
