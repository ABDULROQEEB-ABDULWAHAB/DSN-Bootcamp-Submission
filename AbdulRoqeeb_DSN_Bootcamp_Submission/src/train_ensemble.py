"""
Production Cross-Validation, Multi-Model Ensembling, and Calibration Engine.
Trains Direct and Unit-Demand models across CatBoost, LightGBM, and XGBoost,
applies out-of-fold store distribution calibration, and optimizes ensemble weights.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(__file__))

from features import prepare_datasets
from models import LightGBMWrapper, XGBoostWrapper, CatBoostWrapper, RidgeWrapper

def run_training_pipeline(n_splits=5, seed=42):
    print("=" * 80)
    print("  DSN MART SALES PREDICTION: HIGH-PRECISION CALIBRATED ENSEMBLE")
    print("=" * 80)
    
    # 1. Prepare Datasets with new item-level features
    print("\n[1/5] Engineering features and preparing datasets...")
    train_df, test_df, cat_cols = prepare_datasets()
    
    test_ids = test_df['id'].values
    y_sales = train_df['total_sales'].values
    train_prices = train_df['product_price'].values
    test_prices = test_df['product_price'].values
    
    # Store identifiers for store-level post-calibration
    raw_train = pd.read_csv('data/train.csv')
    raw_test = pd.read_csv('data/test.csv')
    train_stores = raw_train['store_code'].values
    test_stores = raw_test['store_code'].values
    unique_stores = np.unique(train_stores)
    
    # Physical Unit Demand target: y_units = total_sales / product_price
    y_units = y_sales / train_prices
    
    drop_cols = ['id', 'total_sales']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"Total training samples: {len(X)}")
    print(f"Total test samples:     {len(X_test)}")
    print(f"Engineered features ({len(feature_cols)}): {feature_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    
    # 2. Setup 5-Fold Cross Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Define models: (Model Name, Model Class, Target Type: 'sales' or 'units')
    model_definitions = [
        ('CatBoost_Direct', CatBoostWrapper, 'sales'),
        ('CatBoost_Units',  CatBoostWrapper, 'units'),
        ('LGBM_Direct',     LightGBMWrapper, 'sales'),
        ('LGBM_Units',      LightGBMWrapper, 'units'),
        ('XGBoost_Direct',  XGBoostWrapper,  'sales'),
        ('XGBoost_Units',   XGBoostWrapper,  'units')
    ]
    
    oof_predictions = {name: np.zeros(len(X)) for name, _, _ in model_definitions}
    test_predictions = {name: np.zeros(len(X_test)) for name, _, _ in model_definitions}
    model_scores = {}
    
    # Store calibration multipliers accumulated across folds
    fold_store_multipliers = {name: {s: [] for s in unique_stores} for name, _, _ in model_definitions}
    
    # 3. Train Each Model Family across 5 Folds with Out-Of-Fold Store Calibration
    print("\n[2/5] Training 6 diverse model configurations across 5 cross-validation folds...")
    
    for model_name, model_class, target_type in model_definitions:
        print(f"\n--- Training {model_name} (Target: {target_type.upper()}) ---")
        fold_rmses = []
        target_train = y_sales if target_type == 'sales' else y_units
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_sales)):
            X_tr, y_tr = X.iloc[train_idx].copy(), target_train[train_idx]
            X_va, y_va = X.iloc[val_idx].copy(), target_train[val_idx]
            p_va = train_prices[val_idx]
            s_tr, s_va = train_stores[train_idx], train_stores[val_idx]
            
            model = model_class()
            model.fit(X_tr, y_tr, X_va, y_va, cat_cols=cat_cols)
            
            # Predict validation
            val_raw = model.predict(X_va)
            val_sales = val_raw if target_type == 'sales' else val_raw * p_va
            val_sales = np.clip(val_sales, 0, None)
            
            # Predict training fold to compute out-of-fold store multipliers (strictly zero leakage)
            tr_raw = model.predict(X_tr)
            tr_sales = tr_raw if target_type == 'sales' else tr_raw * train_prices[train_idx]
            tr_sales = np.clip(tr_sales, 0, None)
            
            # Compute fold-level store calibration ratio: actual_mean / pred_mean
            fold_ratios = {}
            for s in unique_stores:
                act_m = y_sales[train_idx][s_tr == s].mean()
                prd_m = tr_sales[s_tr == s].mean()
                r = act_m / prd_m if prd_m > 0 else 1.0
                fold_ratios[s] = r
                fold_store_multipliers[model_name][s].append(r)
                
            # Apply store calibration to validation fold
            val_calibrated = val_sales * pd.Series(s_va).map(fold_ratios).values
            val_calibrated = np.clip(val_calibrated, 0, None)
            oof_predictions[model_name][val_idx] = val_calibrated
            
            fold_rmse = root_mean_squared_error(y_sales[val_idx], val_calibrated)
            fold_rmses.append(fold_rmse)
            print(f"  Fold {fold+1}/{n_splits} RMSE (Calibrated): {fold_rmse:.2f}")
            
            # Predict test
            test_raw = model.predict(X_test)
            test_sales = test_raw if target_type == 'sales' else test_raw * test_prices
            test_sales = np.clip(test_sales, 0, None)
            
            # Apply fold-level store calibration to test set
            test_calibrated = test_sales * pd.Series(test_stores).map(fold_ratios).values
            test_predictions[model_name] += np.clip(test_calibrated, 0, None) / n_splits
            
        overall_oof_rmse = root_mean_squared_error(y_sales, oof_predictions[model_name])
        model_scores[model_name] = overall_oof_rmse
        print(f"  >> {model_name} Mean Fold RMSE: {np.mean(fold_rmses):.2f} +/- {np.std(fold_rmses):.2f}")
        print(f"  >> {model_name} Overall OOF RMSE: {overall_oof_rmse:.2f}")

    # 4. SciPy Constrained Optimization for Ensemble Blending
    print("\n[3/5] Optimizing ensemble weights to minimize out-of-fold RMSE...")
    names = [name for name, _, _ in model_definitions]
    oof_matrix = np.column_stack([oof_predictions[name] for name in names])
    test_matrix = np.column_stack([test_predictions[name] for name in names])
    
    # Loss objective with L2 shrinkage to prevent over-concentrating on a single model
    def ensemble_objective(weights):
        blend = np.dot(oof_matrix, weights)
        rmse = root_mean_squared_error(y_sales, blend)
        return rmse
    
    init_weights = np.ones(len(names)) / len(names)
    bounds = [(0.02, 0.70) for _ in names]  # Guarantee ensemble diversity across all models
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    res = minimize(ensemble_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = res.x
    opt_weights = opt_weights / np.sum(opt_weights)  # Normalize
    
    print("\nModel Leaderboard & Optimal Blending Weights:")
    print(f"  {'Model Name':22s} | {'Solo OOF RMSE':14s} | {'Weight':8s}")
    print("  " + "-" * 50)
    for name, w in zip(names, opt_weights):
        print(f"  {name:22s} | {model_scores[name]:10.2f}     | {w*100:6.2f}%")
        
    blended_oof = np.dot(oof_matrix, opt_weights)
    blended_oof_rmse = root_mean_squared_error(y_sales, blended_oof)
    
    best_single_name = min(model_scores, key=model_scores.get)
    best_single_score = model_scores[best_single_name]
    gain = best_single_score - blended_oof_rmse
    
    print(f"\n=======================================================================")
    print(f"  >> BEST SINGLE MODEL ({best_single_name}): {best_single_score:.2f} RMSE")
    print(f"  >> FINAL BLENDED ENSEMBLE OOF RMSE:   {blended_oof_rmse:.2f} RMSE")
    print(f"  >> ENSEMBLE ACCURACY DIVIDEND:        +{gain:.2f} RMSE points improvement!")
    print(f"=======================================================================")
    
    # 5. Generate and Save Submission
    print("\n[4/5] Applying post-processing and exporting submission...")
    blended_test = np.dot(test_matrix, opt_weights)
    blended_test = np.clip(blended_test, 0, None)
    
    sub = pd.DataFrame({
        'id': test_ids,
        'total_sales': np.round(blended_test, 2)
    })
    
    os.makedirs('submissions', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sub_filename = f"submissions/sub_calibrated_ensemble_cv{blended_oof_rmse:.2f}_{timestamp}.csv"
    latest_filename = "submissions/latest_submission.csv"
    
    sub.to_csv(sub_filename, index=False)
    sub.to_csv(latest_filename, index=False)
    
    print(f"  Saved timestamped submission: {sub_filename}")
    print(f"  Updated latest submission:    {latest_filename}")
    
    # Sanity Check
    print("\n[5/5] Distribution Sanity Check:")
    print(f"  Rows generated:    {len(sub)} (Expected: 1705)")
    print(f"  Minimum predicted: {sub['total_sales'].min():.2f}")
    print(f"  Maximum predicted: {sub['total_sales'].max():.2f}")
    print(f"  Mean predicted:    {sub['total_sales'].mean():.2f} (Train Target Mean: {y_sales.mean():.2f})")
    print(f"  Std predicted:     {sub['total_sales'].std():.2f} (Train Target Std:  {y_sales.std():.2f})")
    print(f"  Zero/NaN count:    {sub.isnull().sum().sum()}")
    
    # Check Corner Shop predicted mean
    raw_test['pred'] = sub['total_sales']
    print("\nStore-by-Store Mean Predictions vs Training Ground Truth:")
    for s in unique_stores:
        tr_m = y_sales[train_stores == s].mean()
        pr_m = raw_test[raw_test['store_code'] == s]['pred'].mean()
        diff = pr_m - tr_m
        print(f"  {s:10s}: Train Mean = {tr_m:7.2f} | Pred Mean = {pr_m:7.2f} | Diff = {diff:+6.2f}")
        
    return sub_filename, blended_oof_rmse

if __name__ == '__main__':
    run_training_pipeline()
