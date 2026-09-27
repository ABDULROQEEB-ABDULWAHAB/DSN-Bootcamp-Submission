"""
ULTIMATE Best Submission Engine for DSN Mart Sales Prediction.

KEY INSIGHTS FROM ERROR DECOMPOSITION:
1. Top 20% of sales (mean ~4895) contribute 53.3% of total RMSE -> we massively under-predict high-value items
2. Bottom 20% (mean ~349) we over-predict (pred mean ~818) -> 8.9% of error 
3. Predicted std = 1229 vs actual std = 1698 -> predictions are too "regressed to mean" (variance ratio 0.72)
4. STORE-7WS (Flagship Hypermarket) alone contributes 19.4% of error with RMSE 1453
5. Corner shops are well-predicted (0.4% each) -> Focus should be on large stores
6. Price x Store interaction is the #1 feature -> this relationship is KEY

STRATEGY:
A. Use log-transform on target to handle the right-skewed distribution -> reduces influence of large values
B. Multi-objective: train on both log(sales) and raw sales  
C. Use Huber loss which is more robust to outliers
D. Target encoding with proper fold-based leakage prevention
E. More aggressive hyperparameter diversity across models
F. Stacking (2-level) instead of simple blending
G. Proper post-processing with variance calibration
"""

import os
import sys
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import minimize

sys.path.append('src')

def load_and_engineer():
    """Complete feature engineering pipeline."""
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    target = train['total_sales'].copy()
    n_train = len(train)
    
    combined = pd.concat([train.drop(columns=['total_sales']), test], ignore_index=True)
    
    # 1. Category standardization
    combined['product_category'] = combined['product_category'].str.strip().str.title()
    
    # 2. Product weight imputation
    weight_lookup = combined.dropna(subset=['product_weight_kg']).groupby('product_code')['product_weight_kg'].mean()
    combined['product_weight_kg'] = combined['product_weight_kg'].fillna(combined['product_code'].map(weight_lookup))
    cat_weight = combined.groupby('product_category')['product_weight_kg'].transform('median')
    combined['product_weight_kg'] = combined['product_weight_kg'].fillna(cat_weight)
    combined['product_weight_kg'] = combined['product_weight_kg'].fillna(combined['product_weight_kg'].median())
    
    # 3. Store size imputation
    store_size_impute = {'STORE-JOR': 'Small', 'STORE-DKU': 'Small', 'STORE-OYG': 'Small'}
    combined['store_size'] = combined['store_size'].fillna(combined['store_code'].map(store_size_impute))
    
    # 4. Zero visibility correction
    combined['visibility_was_zero'] = (combined['shelf_visibility'] == 0.0).astype(int)
    pos_vis = combined[combined['shelf_visibility'] > 0]
    prod_vis_lookup = pos_vis.groupby('product_code')['shelf_visibility'].mean()
    cat_vis_lookup = pos_vis.groupby('product_category')['shelf_visibility'].median()
    
    zero_mask = combined['shelf_visibility'] == 0.0
    combined.loc[zero_mask, 'shelf_visibility'] = combined.loc[zero_mask, 'product_code'].map(prod_vis_lookup)
    combined['shelf_visibility'] = combined['shelf_visibility'].fillna(combined['product_category'].map(cat_vis_lookup))
    combined['shelf_visibility'] = combined['shelf_visibility'].fillna(combined['shelf_visibility'].median())
    
    # 5. Fat content
    non_edible = ['Health And Hygiene', 'Household', 'Others']
    combined['is_edible'] = (~combined['product_category'].isin(non_edible)).astype(int)
    combined.loc[combined['is_edible'] == 0, 'fat_content'] = 'Non-Edible'
    
    # === FEATURE ENGINEERING ===
    df = combined.copy()
    
    # Price features
    df['log_price'] = np.log1p(df['product_price'])
    df['sqrt_price'] = np.sqrt(df['product_price'])
    df['price_squared'] = df['product_price'] ** 2
    
    # Price relative to groups
    for group in ['product_category', 'store_code', 'store_format', 'store_location_tier']:
        grp_mean = df.groupby(group)['product_price'].transform('mean')
        grp_std = df.groupby(group)['product_price'].transform('std').replace(0, 1)
        grp_median = df.groupby(group)['product_price'].transform('median')
        df[f'price_to_{group}_mean'] = df['product_price'] / grp_mean
        df[f'price_diff_{group}_mean'] = df['product_price'] - grp_mean
        df[f'price_zscore_{group}'] = (df['product_price'] - grp_mean) / grp_std
        df[f'price_to_{group}_median'] = df['product_price'] / grp_median
    
    # Price tiers
    df['price_tier'] = pd.qcut(df['product_price'], q=5, labels=[1,2,3,4,5]).astype(int)
    df['mrp_band'] = pd.cut(df['product_price'], bins=[0, 69, 136, 203, 500], labels=[1,2,3,4]).astype(int)
    
    # Visibility features
    df['log_visibility'] = np.log1p(df['shelf_visibility'])
    cat_vis_mean = df.groupby('product_category')['shelf_visibility'].transform('mean')
    store_vis_mean = df.groupby('store_code')['shelf_visibility'].transform('mean')
    df['vis_to_cat_mean'] = df['shelf_visibility'] / (cat_vis_mean + 1e-6)
    df['vis_to_store_mean'] = df['shelf_visibility'] / (store_vis_mean + 1e-6)
    df['price_times_visibility'] = df['product_price'] * df['shelf_visibility']
    df['log_price_vis'] = np.log1p(df['product_price'] * df['shelf_visibility'])
    
    # Product features
    df['product_family'] = df['product_code'].str[:6]
    df['product_code_freq'] = df.groupby('product_code')['id'].transform('count')
    
    item_mean_vis = df.groupby('product_code')['shelf_visibility'].transform('mean')
    df['item_vis_ratio'] = df['shelf_visibility'] / (item_mean_vis + 1e-6)
    
    item_mean_price = df.groupby('product_code')['product_price'].transform('mean')
    df['item_price_diff'] = df['product_price'] - item_mean_price
    df['item_price_ratio'] = df['product_price'] / (item_mean_price + 1e-6)
    
    # Store features
    df['store_age_sq'] = df['store_age_years'] ** 2
    df['log_store_age'] = np.log1p(df['store_age_years'])
    df['store_age_bucket'] = pd.cut(df['store_age_years'], bins=[0, 25, 35, 45, 100], labels=[1,2,3,4]).astype(int)
    
    size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
    tier_map = {'Tier_1': 1, 'Tier_2': 2, 'Tier_3': 3}
    df['store_size_ord'] = df['store_size'].map(size_map).astype(int)
    df['store_tier_ord'] = df['store_location_tier'].map(tier_map).astype(int)
    
    # Store volume multiplier (empirical)
    store_vol_map = {
        'Corner Shop': 0.154, 'Superstore': 0.907,
        'Standard Supermarket': 1.066, 'Flagship Hypermarket': 1.683
    }
    df['store_volume_mult'] = df['store_format'].map(store_vol_map)
    
    # Interaction features
    df['store_profile'] = df['store_format'] + '_' + df['store_size'] + '_' + df['store_location_tier']
    df['cat_store_combo'] = df['product_category'] + '_' + df['store_code']
    df['cat_format_combo'] = df['product_category'] + '_' + df['store_format']
    df['cat_tier_combo'] = df['product_category'] + '_' + df['store_location_tier']
    
    df['cat_store_freq'] = df.groupby('cat_store_combo')['id'].transform('count')
    df['cat_format_freq'] = df.groupby('cat_format_combo')['id'].transform('count')
    df['cat_tier_freq'] = df.groupby('cat_tier_combo')['id'].transform('count')
    
    # Price x Store interactions
    df['price_x_volume'] = df['product_price'] * df['store_volume_mult']
    df['price_x_size'] = df['product_price'] * df['store_size_ord']
    df['price_x_tier'] = df['product_price'] * df['store_tier_ord']
    df['price_x_age'] = df['product_price'] * df['store_age_years']
    df['log_price_x_volume'] = np.log1p(df['product_price']) * df['store_volume_mult']
    
    # Weight features
    df['price_per_kg'] = df['product_price'] / df['product_weight_kg'].replace(0, np.nan)
    df['price_per_kg'] = df['price_per_kg'].fillna(df['price_per_kg'].median())
    df['log_price_per_kg'] = np.log1p(df['price_per_kg'])
    df['weight_bracket'] = pd.cut(df['product_weight_kg'], bins=[0, 5, 10, 15, 25], labels=[1,2,3,4]).astype(int)
    
    # POS Impulse flag
    df['is_pos_impulse'] = ((df['visibility_was_zero'] == 1) & (df['product_price'] < 250)).astype(int)
    
    # Visibility share
    format_vis_mean = df.groupby('store_format')['shelf_visibility'].transform('mean')
    df['vis_to_format_mean'] = df['shelf_visibility'] / (format_vis_mean + 1e-6)
    df['visibility_share'] = df['shelf_visibility'] / (item_mean_vis + 1e-6)
    
    # Store item depth
    df['store_presence_count'] = df.groupby('product_code')['store_code'].transform('count')
    
    # Hypermarket premium
    df['is_hypermarket_top_mrp'] = ((df['store_format'] == 'Flagship Hypermarket') & (df['product_price'] > 185)).astype(int)
    df['is_corner_shop'] = (df['store_format'] == 'Corner Shop').astype(int)
    
    # Encode categoricals
    categorical_cols = [
        'product_code', 'product_family', 'fat_content', 'product_category',
        'store_code', 'store_size', 'store_location_tier', 'store_format',
        'store_profile', 'cat_store_combo', 'cat_format_combo', 'cat_tier_combo',
        'weight_bracket', 'mrp_band', 'price_tier'
    ]
    
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        num_classes = len(le.classes_)
        cat_type = pd.CategoricalDtype(categories=list(range(num_classes)))
        df[col] = pd.Series(le.transform(df[col].astype(str)), dtype=cat_type)
    
    # Split back
    train_feats = df.iloc[:n_train].copy().reset_index(drop=True)
    test_feats = df.iloc[n_train:].copy().reset_index(drop=True)
    train_feats['total_sales'] = target.values
    
    return train_feats, test_feats, categorical_cols


def add_target_encoding(X_train, y_train, X_test, cat_cols, n_splits=5, seed=42):
    """Proper out-of-fold target encoding to prevent leakage."""
    X_tr = X_train.copy().reset_index(drop=True)
    X_te = X_test.copy().reset_index(drop=True)
    
    # Create a target series with matching index
    y_series = pd.Series(y_train, index=X_tr.index)
    
    te_cols_to_encode = ['store_code', 'product_category', 'store_format', 'store_profile', 
                         'cat_store_combo', 'cat_format_combo', 'product_family']
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = y_train.mean()
    smooth_factor = 10
    
    for col in te_cols_to_encode:
        te_col_name = f'{col}_te'
        X_tr[te_col_name] = global_mean
        X_te[te_col_name] = global_mean
        
        # Full training set encoding for test
        col_vals = X_tr[col].copy()
        means_dict = {}
        counts_dict = {}
        for val in col_vals.unique():
            mask = col_vals == val
            means_dict[val] = y_series[mask].mean()
            counts_dict[val] = mask.sum()
        
        smoothed_full = {}
        for val in means_dict:
            smoothed_full[val] = (means_dict[val] * counts_dict[val] + global_mean * smooth_factor) / (counts_dict[val] + smooth_factor)
        
        X_te[te_col_name] = X_te[col].map(smoothed_full).fillna(global_mean)
        
        # OOF encoding for train
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
            fold_col_vals = X_tr.iloc[tr_idx][col]
            fold_y = y_series.iloc[tr_idx]
            
            fold_means = {}
            fold_counts = {}
            for val in fold_col_vals.unique():
                mask = fold_col_vals == val
                fold_means[val] = fold_y[mask.values].mean() if hasattr(mask, 'values') else fold_y[mask].mean()
                fold_counts[val] = mask.sum()
            
            smoothed_fold = {}
            for val in fold_means:
                smoothed_fold[val] = (fold_means[val] * fold_counts[val] + global_mean * smooth_factor) / (fold_counts[val] + smooth_factor)
            
            X_tr.loc[va_idx, te_col_name] = X_tr.iloc[va_idx][col].map(smoothed_fold).fillna(global_mean).values
    
    te_feature_names = [f'{col}_te' for col in te_cols_to_encode]
    # Ensure TE columns are float64, not category
    for col_name in te_feature_names:
        X_tr[col_name] = X_tr[col_name].astype(np.float64)
        X_te[col_name] = X_te[col_name].astype(np.float64)
    return X_tr, X_te, te_feature_names


def run_ultimate_pipeline():
    print("=" * 80)
    print("  DSN MART: ULTIMATE BEST SUBMISSION ENGINE")
    print("=" * 80)
    
    # 1. Prepare data
    print("\n[1/6] Engineering features...")
    train_df, test_df, cat_cols = load_and_engineer()
    
    raw_train = pd.read_csv('data/train.csv')
    raw_test = pd.read_csv('data/test.csv')
    
    test_ids = test_df['id'].values
    y = train_df['total_sales'].values
    y_log = np.log1p(y)
    prices_train = train_df['product_price'].values
    prices_test = test_df['product_price'].values
    stores_train = raw_train['store_code'].values
    stores_test = raw_test['store_code'].values
    formats_train = raw_train['store_format'].values
    unique_stores = np.unique(stores_train)
    
    # Unit demand
    y_units = y / prices_train
    y_log_units = np.log1p(y_units)
    
    drop_cols = ['id', 'total_sales']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X_all = train_df[feature_cols].copy()
    X_test_all = test_df[feature_cols].copy()
    
    # Add target encodings
    print("  Adding target encodings (OOF)...")
    X_all, X_test_all, te_cols = add_target_encoding(X_all, y, X_test_all, cat_cols)
    feature_cols = [c for c in X_all.columns if c not in drop_cols]
    
    print(f"  Features: {len(feature_cols)}, Categoricals: {len(cat_cols)}")
    
    # 2. Define diverse model configs
    seeds = [42, 2026, 777, 101, 888]
    n_splits = 5
    
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    
    model_configs = [
        # CatBoost variants
        ('CB_Direct_d6', 'catboost', 'sales', {'depth': 6, 'l2_leaf_reg': 5.0, 'learning_rate': 0.025}),
        ('CB_Direct_d7', 'catboost', 'sales', {'depth': 7, 'l2_leaf_reg': 3.0, 'learning_rate': 0.02}),
        ('CB_Direct_d5', 'catboost', 'sales', {'depth': 5, 'l2_leaf_reg': 7.0, 'learning_rate': 0.03}),
        ('CB_Units_d6',  'catboost', 'units', {'depth': 6, 'l2_leaf_reg': 5.0, 'learning_rate': 0.025}),
        ('CB_Units_d7',  'catboost', 'units', {'depth': 7, 'l2_leaf_reg': 3.0, 'learning_rate': 0.02}),
        ('CB_LogSales',  'catboost', 'log_sales', {'depth': 6, 'l2_leaf_reg': 5.0, 'learning_rate': 0.025}),
        ('CB_LogUnits',  'catboost', 'log_units', {'depth': 6, 'l2_leaf_reg': 5.0, 'learning_rate': 0.025}),
        # LightGBM variants
        ('LGB_Direct_d6', 'lgbm', 'sales', {'max_depth': 6, 'num_leaves': 31, 'learning_rate': 0.018, 'min_child_samples': 20}),
        ('LGB_Direct_d7', 'lgbm', 'sales', {'max_depth': 7, 'num_leaves': 50, 'learning_rate': 0.015, 'min_child_samples': 15}),
        ('LGB_Direct_d5', 'lgbm', 'sales', {'max_depth': 5, 'num_leaves': 25, 'learning_rate': 0.02, 'min_child_samples': 25}),
        ('LGB_Units_d6',  'lgbm', 'units', {'max_depth': 6, 'num_leaves': 31, 'learning_rate': 0.018, 'min_child_samples': 20}),
        ('LGB_Units_d7',  'lgbm', 'units', {'max_depth': 7, 'num_leaves': 50, 'learning_rate': 0.015, 'min_child_samples': 15}),
        ('LGB_LogSales',  'lgbm', 'log_sales', {'max_depth': 6, 'num_leaves': 31, 'learning_rate': 0.018, 'min_child_samples': 20}),
        ('LGB_LogUnits',  'lgbm', 'log_units', {'max_depth': 6, 'num_leaves': 31, 'learning_rate': 0.018, 'min_child_samples': 20}),
        ('LGB_Huber',     'lgbm_huber', 'sales', {'max_depth': 6, 'num_leaves': 31, 'learning_rate': 0.018, 'min_child_samples': 20}),
        # XGBoost variants
        ('XGB_Direct_d5', 'xgb', 'sales', {'max_depth': 5, 'learning_rate': 0.018, 'subsample': 0.8, 'colsample_bytree': 0.7}),
        ('XGB_Direct_d6', 'xgb', 'sales', {'max_depth': 6, 'learning_rate': 0.015, 'subsample': 0.85, 'colsample_bytree': 0.65}),
        ('XGB_Units_d5',  'xgb', 'units', {'max_depth': 5, 'learning_rate': 0.018, 'subsample': 0.8, 'colsample_bytree': 0.7}),
        ('XGB_LogSales',  'xgb', 'log_sales', {'max_depth': 5, 'learning_rate': 0.018, 'subsample': 0.8, 'colsample_bytree': 0.7}),
    ]
    
    n_models = len(model_configs)
    print(f"\n[2/6] Training {n_models} model configs x {len(seeds)} seeds x {n_splits} folds...")
    
    # Storage
    all_oof = {name: np.zeros(len(X_all)) for name, _, _, _ in model_configs}
    all_test = {name: np.zeros(len(X_test_all)) for name, _, _, _ in model_configs}
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  SEED {seed_idx+1}/{len(seeds)} (seed={seed})")
        print(f"{'='*60}")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for model_name, model_type, target_type, hparams in model_configs:
            # Choose target
            if target_type == 'sales':
                y_target = y.copy()
            elif target_type == 'units':
                y_target = y_units.copy()
            elif target_type == 'log_sales':
                y_target = y_log.copy()
            elif target_type == 'log_units':
                y_target = y_log_units.copy()
            
            seed_oof = np.zeros(len(X_all))
            seed_test = np.zeros(len(X_test_all))
            
            for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all)):
                X_tr = X_all.iloc[tr_idx]
                X_va = X_all.iloc[va_idx]
                y_tr = y_target[tr_idx]
                y_va = y_target[va_idx]
                
                if model_type == 'catboost':
                    cb_params = {
                        'loss_function': 'RMSE',
                        'eval_metric': 'RMSE',
                        'learning_rate': hparams['learning_rate'],
                        'depth': hparams['depth'],
                        'l2_leaf_reg': hparams['l2_leaf_reg'],
                        'iterations': 2500,
                        'random_seed': seed,
                        'verbose': False
                    }
                    model = CatBoostRegressor(**cb_params)
                    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_cols,
                             early_stopping_rounds=50, verbose=False)
                    pred_va = model.predict(X_va)
                    pred_test = model.predict(X_test_all)
                    
                elif model_type in ('lgbm', 'lgbm_huber'):
                    lgb_params = {
                        'objective': 'huber' if model_type == 'lgbm_huber' else 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'learning_rate': hparams['learning_rate'],
                        'max_depth': hparams['max_depth'],
                        'num_leaves': hparams['num_leaves'],
                        'min_child_samples': hparams['min_child_samples'],
                        'feature_fraction': 0.75,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 1,
                        'reg_alpha': 1.0,
                        'reg_lambda': 5.0,
                        'random_state': seed,
                        'n_jobs': -1,
                        'verbose': -1
                    }
                    if model_type == 'lgbm_huber':
                        lgb_params['huber_delta'] = 1.0
                    
                    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
                    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)
                    model = lgb.train(lgb_params, dtrain, num_boost_round=2500, valid_sets=[dval],
                                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                    pred_va = model.predict(X_va)
                    pred_test = model.predict(X_test_all)
                    
                elif model_type == 'xgb':
                    xgb_params = {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'learning_rate': hparams['learning_rate'],
                        'max_depth': hparams['max_depth'],
                        'subsample': hparams['subsample'],
                        'colsample_bytree': hparams['colsample_bytree'],
                        'min_child_weight': 4,
                        'alpha': 1.0,
                        'lambda': 5.0,
                        'random_state': seed,
                        'n_jobs': -1,
                        'tree_method': 'hist'
                    }
                    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
                    dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
                    model = xgb.train(xgb_params, dtrain, num_boost_round=2500,
                                     evals=[(dtrain, 'train'), (dval, 'val')],
                                     early_stopping_rounds=50, verbose_eval=False)
                    pred_va = model.predict(dval)
                    pred_test = model.predict(xgb.DMatrix(X_test_all, enable_categorical=True))
                
                # Convert predictions back to sales space
                if target_type == 'units':
                    pred_va_sales = pred_va * prices_train[va_idx]
                    pred_test_sales = pred_test * prices_test
                elif target_type == 'log_sales':
                    pred_va_sales = np.expm1(pred_va)
                    pred_test_sales = np.expm1(pred_test)
                elif target_type == 'log_units':
                    pred_va_sales = np.expm1(pred_va) * prices_train[va_idx]
                    pred_test_sales = np.expm1(pred_test) * prices_test
                else:
                    pred_va_sales = pred_va
                    pred_test_sales = pred_test
                
                pred_va_sales = np.clip(pred_va_sales, 0, None)
                pred_test_sales = np.clip(pred_test_sales, 0, None)
                
                # Store-level calibration
                s_tr = stores_train[tr_idx]
                s_va = stores_train[va_idx]
                
                # Get training fold predictions for calibration
                if model_type == 'catboost':
                    pred_tr = model.predict(X_tr)
                elif model_type in ('lgbm', 'lgbm_huber'):
                    pred_tr = model.predict(X_tr)
                elif model_type == 'xgb':
                    pred_tr = model.predict(dtrain)
                
                if target_type == 'units':
                    pred_tr_sales = pred_tr * prices_train[tr_idx]
                elif target_type == 'log_sales':
                    pred_tr_sales = np.expm1(pred_tr)
                elif target_type == 'log_units':
                    pred_tr_sales = np.expm1(pred_tr) * prices_train[tr_idx]
                else:
                    pred_tr_sales = pred_tr
                pred_tr_sales = np.clip(pred_tr_sales, 0, None)
                
                fold_ratios = {}
                for s in unique_stores:
                    act_m = y[tr_idx][s_tr == s].mean()
                    prd_m = pred_tr_sales[s_tr == s].mean()
                    fold_ratios[s] = act_m / prd_m if prd_m > 0 else 1.0
                
                val_cal = pred_va_sales * pd.Series(s_va).map(fold_ratios).values
                test_cal = pred_test_sales * pd.Series(stores_test).map(fold_ratios).values
                
                seed_oof[va_idx] = np.clip(val_cal, 0, None)
                seed_test += np.clip(test_cal, 0, None) / n_splits
            
            oof_rmse = root_mean_squared_error(y, seed_oof)
            print(f"  [{model_name:18s} | seed={seed:4d}] OOF RMSE: {oof_rmse:.2f}")
            
            all_oof[model_name] += seed_oof / len(seeds)
            all_test[model_name] += seed_test / len(seeds)
    
    # 3. Model Performance Summary
    print(f"\n{'='*70}")
    print(f"  BAGGED MODEL PERFORMANCE")
    print(f"{'='*70}")
    names = [n for n, _, _, _ in model_configs]
    model_scores = {}
    for name in names:
        score = root_mean_squared_error(y, all_oof[name])
        model_scores[name] = score
        print(f"  {name:22s} | OOF RMSE: {score:.2f}")
    
    # 4. Level 1: SLSQP Ensemble Optimization
    print(f"\n[3/6] Optimizing Level-1 ensemble weights...")
    oof_matrix = np.column_stack([all_oof[n] for n in names])
    test_matrix = np.column_stack([all_test[n] for n in names])
    
    def ensemble_loss(w):
        blend = np.dot(oof_matrix, w)
        return root_mean_squared_error(y, blend)
    
    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 0.5) for _ in range(n_models)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Try multiple initializations
    best_result = None
    best_loss = float('inf')
    
    for attempt in range(5):
        if attempt == 0:
            w0 = init_w
        else:
            np.random.seed(attempt * 123)
            w0 = np.random.dirichlet(np.ones(n_models))
        
        res = minimize(ensemble_loss, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})
        if res.fun < best_loss:
            best_loss = res.fun
            best_result = res
    
    opt_w = best_result.x / best_result.x.sum()
    
    print("\nOptimal Weights:")
    for name, w in zip(names, opt_w):
        if w > 0.001:
            print(f"  {name:22s}: {w*100:6.2f}%")
    
    blended_oof = np.dot(oof_matrix, opt_w)
    blended_oof_rmse = root_mean_squared_error(y, blended_oof)
    print(f"\n  Level-1 Blended OOF RMSE: {blended_oof_rmse:.2f}")
    
    # 5. Level 2: Stacking with Ridge on OOF predictions
    print(f"\n[4/6] Level-2 stacking with Ridge regression...")
    
    scaler = StandardScaler()
    oof_scaled = scaler.fit_transform(oof_matrix)
    test_scaled = scaler.transform(test_matrix)
    
    best_alpha = None
    best_stack_rmse = float('inf')
    
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        kf_stack = KFold(n_splits=5, shuffle=True, random_state=42)
        stack_oof = np.zeros(len(y))
        
        for tr_idx, va_idx in kf_stack.split(oof_scaled):
            ridge = Ridge(alpha=alpha, random_state=42)
            ridge.fit(oof_scaled[tr_idx], y[tr_idx])
            stack_oof[va_idx] = ridge.predict(oof_scaled[va_idx])
        
        stack_oof = np.clip(stack_oof, 0, None)
        stack_rmse = root_mean_squared_error(y, stack_oof)
        
        if stack_rmse < best_stack_rmse:
            best_stack_rmse = stack_rmse
            best_alpha = alpha
    
    print(f"  Best stacking alpha: {best_alpha}, RMSE: {best_stack_rmse:.2f}")
    
    # Fit final stacker on all data
    ridge_final = Ridge(alpha=best_alpha, random_state=42)
    ridge_final.fit(oof_scaled, y)
    stacked_test = np.clip(ridge_final.predict(test_scaled), 0, None)
    
    # 6. Final blend: combine SLSQP blend + stacked predictions
    print(f"\n[5/6] Final meta-blend...")
    
    blended_test = np.dot(test_matrix, opt_w)
    blended_test = np.clip(blended_test, 0, None)
    
    # Use OOF to determine best ratio
    best_meta_blend = None
    best_meta_rmse = float('inf')
    
    kf_meta = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for slsqp_w in np.arange(0.3, 1.01, 0.05):
        stack_w = 1.0 - slsqp_w
        
        meta_oof = np.zeros(len(y))
        for tr_idx, va_idx in kf_meta.split(oof_matrix):
            # SLSQP part
            slsqp_part = np.dot(oof_matrix[va_idx], opt_w)
            
            # Stack part
            ridge_temp = Ridge(alpha=best_alpha, random_state=42)
            ridge_temp.fit(oof_scaled[tr_idx], y[tr_idx])
            stack_part = ridge_temp.predict(oof_scaled[va_idx])
            
            meta_oof[va_idx] = slsqp_w * slsqp_part + stack_w * stack_part
        
        meta_oof = np.clip(meta_oof, 0, None)
        meta_rmse = root_mean_squared_error(y, meta_oof)
        
        if meta_rmse < best_meta_rmse:
            best_meta_rmse = meta_rmse
            best_meta_blend = (slsqp_w, stack_w)
    
    print(f"  Best meta blend: SLSQP={best_meta_blend[0]:.2f}, Stack={best_meta_blend[1]:.2f}")
    print(f"  Meta-blend OOF RMSE: {best_meta_rmse:.2f}")
    
    # Choose the best approach
    if best_meta_rmse < blended_oof_rmse:
        final_test = best_meta_blend[0] * blended_test + best_meta_blend[1] * stacked_test
        final_cv = best_meta_rmse
        method = "Meta-Blend (SLSQP + Stacking)"
    else:
        final_test = blended_test
        final_cv = blended_oof_rmse
        method = "SLSQP Blend"
    
    final_test = np.clip(final_test, 0, None)
    
    print(f"\n  Using method: {method}")
    print(f"  Final CV RMSE: {final_cv:.2f}")
    
    # 7. Save
    print(f"\n[6/6] Saving submission...")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sub = pd.DataFrame({'id': test_ids, 'total_sales': np.round(final_test, 2)})
    
    os.makedirs('submissions', exist_ok=True)
    sub_file = f"submissions/sub_ultimate_cv{final_cv:.2f}_{timestamp}.csv"
    latest_file = "submissions/latest_submission.csv"
    
    sub.to_csv(sub_file, index=False)
    sub.to_csv(latest_file, index=False)
    
    print(f"  Saved: {sub_file}")
    print(f"  Updated: {latest_file}")
    
    # Sanity check
    print(f"\n{'='*70}")
    print(f"  SUBMISSION DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"  Rows:     {len(sub)}")
    print(f"  Min:      {sub['total_sales'].min():.2f}")
    print(f"  Max:      {sub['total_sales'].max():.2f}")
    print(f"  Mean:     {sub['total_sales'].mean():.2f} (Train: {y.mean():.2f})")
    print(f"  Std:      {sub['total_sales'].std():.2f} (Train: {y.std():.2f})")
    print(f"  Nulls:    {sub.isnull().sum().sum()}")
    
    # Store-level check
    raw_test['pred'] = sub['total_sales'].values
    print(f"\n  Store-Level Check:")
    for s in unique_stores:
        tr_m = y[stores_train == s].mean()
        pr_m = raw_test[raw_test['store_code'] == s]['pred'].mean()
        diff_pct = (pr_m - tr_m) / tr_m * 100
        print(f"    {s:10s}: Train={tr_m:7.2f} | Pred={pr_m:7.2f} | Diff={diff_pct:+6.1f}%")
    
    return sub_file, final_cv


if __name__ == '__main__':
    run_ultimate_pipeline()
