import sys
import os
sys.path.append('src')
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from features import prepare_datasets

train_df, test_df, cat_cols = prepare_datasets()
y = train_df['total_sales'].values
p = train_df['product_price'].values
u = y / p  # unit demand

drop_cols = ['id', 'total_sales']
X = train_df[[c for c in train_df.columns if c not in drop_cols]].copy()

raw_train = pd.read_csv('data/train.csv')
train_stores = raw_train['store_code'].values
unique_stores = np.unique(train_stores)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_approach(name, fit_predict_fn):
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr, p_tr, u_tr = X.iloc[tr_idx], y[tr_idx], p[tr_idx], u[tr_idx]
        X_va, y_va, p_va, u_va = X.iloc[va_idx], y[va_idx], p[va_idx], u[va_idx]
        s_tr, s_va = train_stores[tr_idx], train_stores[val_idx] if 'val_idx' in locals() else train_stores[va_idx]
        
        preds_va, preds_tr = fit_predict_fn(X_tr, y_tr, p_tr, u_tr, X_va, y_va, p_va, u_va, s_tr, s_va)
        
        # Calibration
        ratios = {}
        for s in unique_stores:
            act_m = y_tr[s_tr == s].mean()
            prd_m = preds_tr[s_tr == s].mean()
            ratios[s] = act_m / prd_m if prd_m > 0 else 1.0
            
        cal = preds_va * pd.Series(s_va).map(ratios).values
        oof[va_idx] = np.clip(cal, 0, None)
        
    rmse = root_mean_squared_error(y, oof)
    print(f"{name:35s} | OOF RMSE: {rmse:.2f} | Max OOF: {oof.max():.2f} | Std OOF: {oof.std():.2f}")
    return oof

print("Comparing Model Architectures on 5-Fold CV:")

# 1. CatBoost Direct
def cb_direct(X_tr, y_tr, p_tr, u_tr, X_va, y_va, p_va, u_va, s_tr, s_va):
    cb = CatBoostRegressor(depth=5, learning_rate=0.03, iterations=1200, cat_features=cat_cols, verbose=False, random_seed=42)
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40)
    return cb.predict(X_va), cb.predict(X_tr)
evaluate_approach("CatBoost Direct", cb_direct)

# 2. CatBoost Unit Demand
def cb_units(X_tr, y_tr, p_tr, u_tr, X_va, y_va, p_va, u_va, s_tr, s_va):
    cb = CatBoostRegressor(depth=5, learning_rate=0.03, iterations=1200, cat_features=cat_cols, verbose=False, random_seed=42)
    cb.fit(X_tr, u_tr, eval_set=(X_va, u_va), early_stopping_rounds=40)
    return cb.predict(X_va) * p_va, cb.predict(X_tr) * p_tr
evaluate_approach("CatBoost Unit Demand", cb_units)

# 3. LightGBM Unit Demand
def lgb_units(X_tr, y_tr, p_tr, u_tr, X_va, y_va, p_va, u_va, s_tr, s_va):
    lgbm = lgb.LGBMRegressor(num_leaves=18, max_depth=5, learning_rate=0.03, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    lgbm.fit(X_tr, u_tr, eval_set=[(X_va, u_va)], callbacks=[lgb.early_stopping(40, verbose=False)])
    return lgbm.predict(X_va) * p_va, lgbm.predict(X_tr) * p_tr
evaluate_approach("LightGBM Unit Demand", lgb_units)
