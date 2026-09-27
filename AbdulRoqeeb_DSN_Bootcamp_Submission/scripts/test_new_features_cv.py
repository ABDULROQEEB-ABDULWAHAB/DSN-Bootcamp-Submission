import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor
import lightgbm as lgb
from features import prepare_datasets

train_df, test_df, cat_cols = prepare_datasets()

# Compute the new features on train_df
store_vol_map = {
    'Corner Shop': 0.154,
    'Superstore': 0.907,
    'Standard Supermarket': 1.066,
    'Flagship Hypermarket': 1.683
}
raw_train = pd.read_csv('data/train.csv')
train_df['price_x_store_volume'] = train_df['product_price'] * raw_train['store_format'].map(store_vol_map)
train_df['price_per_kg'] = train_df['product_price'] / train_df['product_weight_kg']
train_df['price_x_store_size'] = train_df['product_price'] * train_df['store_size_ord']
train_df['price_x_store_tier'] = train_df['product_price'] * train_df['store_tier_ord']
train_df['is_hypermarket_top_mrp'] = ((raw_train['store_format'] == 'Flagship Hypermarket') & (train_df['product_price'] > 185)).astype(int)

y = train_df['total_sales'].values
p = train_df['product_price'].values
u = y / p
train_stores = raw_train['store_code'].values
unique_stores = np.unique(train_stores)

drop_cols = ['id', 'total_sales']
feature_cols = [c for c in train_df.columns if c not in drop_cols]
X = train_df[feature_cols].copy()

print(f"Testing CatBoost Unit Demand with new features ({len(feature_cols)} total)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_cal = np.zeros(len(y))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    X_tr, u_tr, p_tr = X.iloc[tr_idx], u[tr_idx], p[tr_idx]
    X_va, u_va, p_va = X.iloc[va_idx], u[va_idx], p[va_idx]
    s_tr, s_va = train_stores[tr_idx], train_stores[va_idx]
    
    cb = CatBoostRegressor(depth=5, learning_rate=0.03, iterations=1200, cat_features=cat_cols, verbose=False, random_seed=42)
    cb.fit(X_tr, u_tr, eval_set=(X_va, u_va), early_stopping_rounds=40)
    
    val_sales = cb.predict(X_va) * p_va
    tr_sales = cb.predict(X_tr) * p_tr
    
    ratios = {s: y[tr_idx][s_tr == s].mean() / tr_sales[s_tr == s].mean() for s in unique_stores}
    oof_cal[va_idx] = val_sales * pd.Series(s_va).map(ratios).values

oof_rmse = root_mean_squared_error(y, oof_cal)
print(f"CatBoost Units + New Features OOF RMSE: {oof_rmse:.2f} (Previous solo CatBoost Units was 1071.77)")
