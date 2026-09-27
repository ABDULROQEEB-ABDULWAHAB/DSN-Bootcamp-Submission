"""
Multi-Seed Bagged Calibrated Ensemble Engine.
Trains across multiple random seeds to eliminate fold split variance,
combines Direct and Unit-Demand models, and performs leak-free store calibration.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize

sys.path.append('src')

from features import prepare_datasets
from models import LightGBMWrapper, XGBoostWrapper, CatBoostWrapper

def run_bagged_ensemble(seeds=[42, 2026, 777], n_splits=5):
    print("=" * 80)
    print(f"  DSN MART: MULTI-SEED BAGGED ENSEMBLE ({len(seeds)} SEEDS x {n_splits} FOLDS)")
    print("=" * 80)
    
    # 1. Prepare Datasets
    print("\n[1/5] Loading datasets and engineering features...")
    train_df, test_df, cat_cols = prepare_datasets()
    
    # Add domain elasticity features
    store_vol_map = {
        'Corner Shop': 0.154,
        'Superstore': 0.907,
        'Standard Supermarket': 1.066,
        'Flagship Hypermarket': 1.683
    }
    raw_train = pd.read_csv('data/train.csv')
    raw_test = pd.read_csv('data/test.csv')
    
    for df, raw in [(train_df, raw_train), (test_df, raw_test)]:
        df['price_x_store_volume'] = df['product_price'] * raw['store_format'].map(store_vol_map)
        df['price_per_kg'] = df['product_price'] / df['product_weight_kg']
        df['price_x_store_size'] = df['product_price'] * df['store_size_ord']
        df['price_x_store_tier'] = df['product_price'] * df['store_tier_ord']
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
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Seeds to bag: {seeds}")
    
    model_definitions = [
        ('CatBoost_Direct', CatBoostWrapper, 'sales'),
        ('CatBoost_Units',  CatBoostWrapper, 'units'),
        ('LGBM_Direct',     LightGBMWrapper, 'sales'),
        ('LGBM_Units',      LightGBMWrapper, 'units'),
        ('XGBoost_Direct',  XGBoostWrapper,  'sales'),
        ('XGBoost_Units',   XGBoostWrapper,  'units')
    ]
    
    # Accumulate OOF and test predictions across seeds
    all_seed_oof = {name: np.zeros(len(X)) for name, _, _ in model_definitions}
    all_seed_test = {name: np.zeros(len(X_test)) for name, _, _ in model_definitions}
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n==================================================")
        print(f"  >>> RUNNING SEED {seed_idx + 1}/{len(seeds)} (seed={seed}) <<<")
        print(f"==================================================")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for model_name, model_class, target_type in model_definitions:
            print(f"Training {model_name} (seed {seed})...", flush=True)
            target_train = y_sales if target_type == 'sales' else y_units
            seed_oof = np.zeros(len(X))
            seed_test = np.zeros(len(X_test))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_sales)):
                X_tr, y_tr = X.iloc[train_idx].copy(), target_train[train_idx]
                X_va, y_va = X.iloc[val_idx].copy(), target_train[val_idx]
                p_va = train_prices[val_idx]
                s_tr, s_va = train_stores[train_idx], train_stores[val_idx]
                
                model = model_class()
                # Pass seed to models if supported
                if hasattr(model, 'random_seed'):
                    model.random_seed = seed
                model.fit(X_tr, y_tr, X_va, y_va, cat_cols=cat_cols)
                
                # Val prediction
                val_raw = model.predict(X_va)
                val_sales = val_raw if target_type == 'sales' else val_raw * p_va
                val_sales = np.clip(val_sales, 0, None)
                
                # Train prediction for calibration
                tr_raw = model.predict(X_tr)
                tr_sales = tr_raw if target_type == 'sales' else tr_raw * train_prices[train_idx]
                tr_sales = np.clip(tr_sales, 0, None)
                
                # Store ratios
                fold_ratios = {}
                for s in unique_stores:
                    act_m = y_sales[train_idx][s_tr == s].mean()
                    prd_m = tr_sales[s_tr == s].mean()
                    fold_ratios[s] = act_m / prd_m if prd_m > 0 else 1.0
                    
                val_cal = val_sales * pd.Series(s_va).map(fold_ratios).values
                seed_oof[val_idx] = np.clip(val_cal, 0, None)
                
                # Test prediction
                test_raw = model.predict(X_test)
                test_sales = test_raw if target_type == 'sales' else test_raw * test_prices
                test_cal = test_sales * pd.Series(test_stores).map(fold_ratios).values
                seed_test += np.clip(test_cal, 0, None) / n_splits
                
            model_oof_rmse = root_mean_squared_error(y_sales, seed_oof)
            print(f"  -> {model_name} (Seed {seed}) OOF RMSE: {model_oof_rmse:.2f}", flush=True)
            
            all_seed_oof[model_name] += seed_oof / len(seeds)
            all_seed_test[model_name] += seed_test / len(seeds)

    # 3. Assess Bagged Single Models
    print("\n" + "=" * 60)
    print("  BAGGED SINGLE MODEL PERFORMANCE ACROSS ALL SEEDS")
    print("=" * 60)
    names = [name for name, _, _ in model_definitions]
    model_scores = {}
    for name in names:
        score = root_mean_squared_error(y_sales, all_seed_oof[name])
        model_scores[name] = score
        print(f"  {name:22s} | Bagged OOF RMSE: {score:.2f}")

    # 4. SciPy Optimization for Bagged Ensemble
    oof_matrix = np.column_stack([all_seed_oof[name] for name in names])
    test_matrix = np.column_stack([all_seed_test[name] for name in names])
    
    def ensemble_objective(weights):
        blend = np.dot(oof_matrix, weights)
        return root_mean_squared_error(y_sales, blend)
        
    init_weights = np.ones(len(names)) / len(names)
    bounds = [(0.02, 0.70) for _ in names]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    res = minimize(ensemble_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = res.x / np.sum(res.x)
    
    print("\nOptimal Bagged Ensemble Weights:")
    for name, w in zip(names, opt_weights):
        print(f"  {name:22s} : {w*100:6.2f}%")
        
    blended_oof = np.dot(oof_matrix, opt_weights)
    blended_oof_rmse = root_mean_squared_error(y_sales, blended_oof)
    
    print(f"FINAL BAGGED ENSEMBLE OOF RMSE: {blended_oof_rmse:.2f} RMSE")
    
    # 5. Generate and Export Test Submission
    blended_test = np.dot(test_matrix, opt_weights)
    blended_test = np.clip(blended_test, 0, None)
    
    # Tail Uncapping for STORE-7WS High MRP Products
    # High MRP products (>185 NGN) in STORE-7WS suffer from tree leaf shrinkage
    tail_mask = (raw_test['store_code'] == 'STORE-7WS') & (raw_test['product_price'] > 185)
    print(f"\nTail uncapping check: {tail_mask.sum()} items qualify in STORE-7WS")
    
    # Create baseline bagged submission
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_bagged = pd.DataFrame({'id': test_ids, 'total_sales': np.round(blended_test, 2)})
    sub_filename = f"submissions/sub_bagged_ensemble_cv{blended_oof_rmse:.2f}_{timestamp}.csv"
    sub_bagged.to_csv(sub_filename, index=False)
    sub_bagged.to_csv("submissions/latest_submission.csv", index=False)
    
    print(f"Saved: {sub_filename}")
    print(f"Updated: submissions/latest_submission.csv")
    
    return sub_filename, blended_oof_rmse

if __name__ == '__main__':
    run_bagged_ensemble(seeds=[42, 2026, 777])
